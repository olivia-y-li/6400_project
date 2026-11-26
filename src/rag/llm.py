"""Thin wrapper around the Anthropic Messages API: forced tool output, disk cache, cost.

The API rejects a temperature parameter for this model ("Extra inputs are not
permitted"), so sampling uses the API default. Reproducibility comes from the disk
cache instead: each request is stored by a hash of the full request, so a rerun replays
the first answer exactly and costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL = "claude-haiku-4-5"
# (input, output) USD per million tokens. Check against the current pricing page.
PRICES_PER_MTOK = {"claude-haiku-4-5": (1.00, 5.00)}


# Hard ceiling on total live spend across ALL runs (USD): the prepaid credit minus a margin
# for any difference between this arithmetic and the real bill.
LIFETIME_CAP_USD = 33.0
# Worst-case cost reserved for a call before it is made, so parallel workers cannot overshoot.
CALL_RESERVE_USD = 0.01


class BudgetExceeded(RuntimeError):
    pass


def cache_spend(cache_dir: str | Path = "data/llm_cache") -> float:
    """Total cost of every live call ever made, recomputed from the tokens stored in the cache."""
    total = 0.0
    for f in Path(cache_dir).glob("*.json"):
        d = json.loads(f.read_text())
        total += cost_usd(d.get("model", MODEL), d["input_tokens"], d["output_tokens"])
    return total


def load_dotenv(path: str | Path = ".env") -> None:
    """Set variables from a local .env file without overriding the real environment."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class LLMResult:
    output: dict[str, Any]  # the tool call's input, i.e. the structured answer
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float  # wall-clock of the live call (replayed from the cache on a hit)
    cached: bool


class LLM:
    def __init__(self, client: Any = None, cache_dir: str | Path = "data/llm_cache",
                 model: str = MODEL, max_tokens: int = 300,
                 spend_cap_usd: float = LIFETIME_CAP_USD) -> None:
        self._client = client
        self.cache_dir = Path(cache_dir)
        self.model, self.max_tokens = model, max_tokens
        self.spend_cap_usd = spend_cap_usd
        self._lock = threading.Lock()
        self._spent = cache_spend(self.cache_dir)  # everything spent so far, plus this session
        if model not in PRICES_PER_MTOK:
            raise ValueError(f"no price recorded for {model!r}; add it to PRICES_PER_MTOK")

    @property
    def client(self):
        if self._client is None:
            import anthropic

            load_dotenv()
            self._client = anthropic.Anthropic(max_retries=5)
        return self._client

    def _key(self, system: str, user: str, tool: dict) -> str:
        blob = json.dumps({"model": self.model, "max_tokens": self.max_tokens,
                           "system": system, "user": user, "tool": tool}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def call_tool(self, system: str, user: str, tool: dict) -> LLMResult:
        """One request that must answer through `tool`; returns the tool's input."""
        path = self.cache_dir / f"{self._key(system, user, tool)}.json"
        if path.exists():
            d = json.loads(path.read_text())
            return LLMResult(d["output"], d["input_tokens"], d["output_tokens"],
                             cost_usd(self.model, d["input_tokens"], d["output_tokens"]),
                             d["latency_s"], cached=True)

        with self._lock:  # reserve the worst case first; refuse if it could pass the cap
            if self._spent + CALL_RESERVE_USD > self.spend_cap_usd:
                raise BudgetExceeded(
                    f"spend ${self._spent:.2f} is within ${CALL_RESERVE_USD:.2f} of the "
                    f"${self.spend_cap_usd:.2f} lifetime cap")
            self._spent += CALL_RESERVE_USD
        start = time.perf_counter()
        try:
            resp = self._create(system, user, tool)
        except BaseException:
            with self._lock:
                self._spent -= CALL_RESERVE_USD  # nothing was billed for a failed request
            raise
        latency = time.perf_counter() - start
        blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        n_in, n_out = resp.usage.input_tokens, resp.usage.output_tokens
        with self._lock:  # a response was billed even if its content turns out to be unusable
            self._spent += cost_usd(self.model, n_in, n_out) - CALL_RESERVE_USD
        if not blocks:
            raise RuntimeError(f"no tool_use block in response (stop_reason={resp.stop_reason})")
        out = dict(blocks[0].input)
        return self._store(path, out, n_in, n_out, latency)

    def _create(self, system: str, user: str, tool: dict):
        return self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
        )

    def _store(self, path: Path, out: dict, n_in: int, n_out: int, latency: float) -> LLMResult:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps({"output": out, "input_tokens": n_in, "output_tokens": n_out,
                                   "latency_s": latency, "model": self.model}))
        tmp.replace(path)  # atomic, so concurrent workers never read a half-written entry
        return LLMResult(out, n_in, n_out, cost_usd(self.model, n_in, n_out), latency, cached=False)


if __name__ == "__main__":
    print(f"total live spend so far: ${cache_spend():.4f} (lifetime cap ${LIFETIME_CAP_USD:.2f})")
