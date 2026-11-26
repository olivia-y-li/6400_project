"""Run Hybrid + Rerank + Decomposition on a split and save its ranked chunks and traces.

python -m rag.decomp_run --data data/processed/hotpotqa --out data/results/hotpotqa/decomp.json --splits dev
Writes the same per_question layout as rag.report, so rag.qa can answer from it:
python -m rag.qa --report data/results/hotpotqa/decomp.json --system hybrid_rerank_decomp ...
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .chunker import MIN_OVERLAP
from .decompose import DecompositionRetriever
from .evaluate import summarize
from .llm import LLM
from .metrics import score_question
from .qa import BudgetExceeded
from .retrieval import build_systems
from .schema import Chunk, Question, read_jsonl

SYSTEM = "hybrid_rerank_decomp"


def run_decomposition(data_dir: str | Path, out_json: str | Path, llm: LLM,
                      splits: tuple[str, ...] = ("dev",), k: int = 5,
                      min_overlap: float = MIN_OVERLAP, workers: int = 4,
                      limit: int | None = None, max_cost: float = 5.0,
                      include_original: bool = False) -> dict[str, Any]:
    data_dir = Path(data_dir)
    chunks = list(read_jsonl(data_dir / "chunks.jsonl", Chunk))
    by_id = {c.chunk_id: c for c in chunks}
    reranker = build_systems(chunks, ["hybrid_rerank"])["hybrid_rerank"]
    system = DecompositionRetriever(reranker.hybrid, reranker, chunks, llm, include_original)

    lock = threading.Lock()
    live_cost = [0.0]
    results: dict[str, Any] = {}
    per_question: dict[str, list[dict]] = {}
    errors: list[str] = []

    def work(q: Question) -> dict:
        with lock:
            if live_cost[0] >= max_cost:
                raise BudgetExceeded(f"live spend ${live_cost[0]:.2f} reached the ${max_cost:.2f} cap")
        ranked, trace = system.run(q.question, k)
        with lock:
            live_cost[0] += trace.live_cost_usd
        return {"qid": q.qid, "retrieved": ranked,
                **score_question(q, ranked, by_id, k, min_overlap), "trace": trace.to_dict()}

    for split in splits:
        questions = list(read_jsonl(data_dir / f"questions_{split}.jsonl", Question))
        if limit:
            questions = questions[:limit]
        done: list[tuple[Question, dict]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(q, pool.submit(work, q)) for q in questions]
            for q, f in futures:
                try:
                    done.append((q, f.result()))
                except BudgetExceeded as e:
                    errors.append(str(e))
        key = f"{SYSTEM}/{split}"
        per_question[key] = [d for _, d in done]
        if done:
            qs = [q for q, _ in done]
            results[key] = {**summarize([d for _, d in done], qs), "n_scored": len(done),
                            "n_expected": len(questions),
                            "trace_totals": totals([d["trace"] for _, d in done])}

    out = {"system": SYSTEM, "k": k, "min_overlap": min_overlap,
           "include_original": include_original, "error": errors[0] if errors else None,
           "results": results, "per_question": per_question}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1))
    return out


def totals(traces: list[dict]) -> dict[str, float]:
    n = len(traces)
    return {
        "mean_llm_calls": sum(t["llm_calls"] for t in traces) / n,
        "mean_sub_questions": sum(len(t["sub_questions"]) for t in traces) / n,
        "pct_decomposed": 100 * sum(len(t["sub_questions"]) > 1 for t in traces) / n,
        "mean_candidates": sum(t["n_candidates"] for t in traces) / n,
        "total_cost_usd": sum(t["cost_usd"] for t in traces),
        "mean_cost_usd": sum(t["cost_usd"] for t in traces) / n,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag.decomp_run")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--splits", nargs="+", choices=["dev", "eval"], default=["dev"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-cost", type=float, default=5.0, help="stop after this much live spend (USD)")
    p.add_argument("--include-original", action="store_true",
                   help="ablation: also pool the undecomposed question's candidates")
    p.add_argument("--cache-dir", default="data/llm_cache")
    a = p.parse_args(argv)
    out = run_decomposition(a.data, a.out, LLM(cache_dir=a.cache_dir), tuple(a.splits), a.k,
                            workers=a.workers, limit=a.limit, max_cost=a.max_cost,
                            include_original=a.include_original)
    print(json.dumps({"error": out["error"], "results": {
        k: {m: v[m] for m in ("n_scored", "recall", "complete_recall", "mrr", "trace_totals")}
        for k, v in out["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
