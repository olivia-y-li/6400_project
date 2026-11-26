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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL = "claude-haiku-4-5"
# (input, output) USD per million tokens. Check against the current pricing page.
PRICES_PER_MTOK = {"claude-haiku-4-5": (1.00, 5.00)}


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
                 model: str = MODEL, max_tokens: int = 300) -> None:
        self._client = client
        self.cache_dir = Path(cache_dir)
        self.model, self.max_tokens = model, max_tokens
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

        start = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
        )
        latency = time.perf_counter() - start
        blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not blocks:
            raise RuntimeError(f"no tool_use block in response (stop_reason={resp.stop_reason})")
        out = dict(blocks[0].input)
        n_in, n_out = resp.usage.input_tokens, resp.usage.output_tokens

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps({"output": out, "input_tokens": n_in, "output_tokens": n_out,
                                   "latency_s": latency, "model": self.model}))
        tmp.replace(path)  # atomic, so concurrent workers never read a half-written entry
        return LLMResult(out, n_in, n_out, cost_usd(self.model, n_in, n_out), latency, cached=False)
