"""Generate cited answers for one system and split, and score them.

python -m rag.qa --data data/processed/hotpotqa --report data/results/hotpotqa/report.json \
    --system hybrid_rerank --split dev --out data/results/hotpotqa/qa
Reads the ranked chunks that rag.report saved, so retrieval is not recomputed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .generate import generate_answer
from .llm import LLM
from .metrics import bootstrap_ci
from .qa_metrics import score_answer
from .schema import Chunk, Question, read_jsonl, write_jsonl


class BudgetExceeded(RuntimeError):
    pass


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))] if s else 0.0


def run_qa(data_dir: str | Path, report_json: str | Path, system: str, split: str,
           out_dir: str | Path, llm: LLM, workers: int = 8, limit: int | None = None,
           max_cost: float = 5.0) -> dict[str, Any]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    chunks = {c.chunk_id: c for c in read_jsonl(data_dir / "chunks.jsonl", Chunk)}
    questions = {q.qid: q for q in read_jsonl(data_dir / f"questions_{split}.jsonl", Question)}
    ranked = json.loads(Path(report_json).read_text())["per_question"][f"{system}/{split}"]
    if limit:
        ranked = ranked[:limit]

    lock = threading.Lock()
    live_cost = [0.0]  # spend on calls that were not served from the cache

    def work(item: dict) -> dict:
        q = questions[item["qid"]]
        with lock:
            if live_cost[0] >= max_cost:
                raise BudgetExceeded(f"live spend ${live_cost[0]:.2f} reached the ${max_cost:.2f} cap")
        got = [chunks[cid] for cid in item["retrieved"]]
        g = generate_answer(llm, q.question, got)
        if not g.llm.cached:
            with lock:
                live_cost[0] += g.llm.cost_usd
        gold = set(q.gold_chunk_ids)
        return {
            "qid": q.qid, "system": system, "split": split, "question": q.question,
            "qtype": q.qtype, "gold_answers": q.gold_answers, "answer": g.answer,
            "retrieved": item["retrieved"], "citations": g.citations,
            "invalid_citations": g.invalid_citations,
            "cited_docs": sorted({chunks[c].doc_title for c in g.citations}),
            "cites_gold_chunk": any(c in gold for c in g.citations),
            **score_answer(g.answer, q.gold_answers, q.source),
            "input_tokens": g.llm.input_tokens, "output_tokens": g.llm.output_tokens,
            "cost_usd": g.llm.cost_usd, "gen_latency_s": g.llm.latency_s, "cached": g.llm.cached,
        }

    rows: list[dict] = []
    error: str | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, item) for item in ranked]
        for f in futures:
            try:
                rows.append(f.result())
            except BudgetExceeded as e:
                error = str(e)
    rows.sort(key=lambda r: r["qid"])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{system}_{split}.jsonl", rows)
    summary = summarize(rows, system, split, len(ranked), error)
    (out_dir / f"{system}_{split}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize(rows: list[dict], system: str, split: str, n_expected: int,
              error: str | None = None) -> dict[str, Any]:
    n = len(rows)
    lat = [r["gen_latency_s"] for r in rows]
    mean = lambda key: sum(r[key] for r in rows) / n if n else 0.0
    ci = bootstrap_ci([{"recall": r["em"], "complete_recall": r["f1"], "mrr": 0.0} for r in rows]) if n else {}
    return {
        "system": system, "split": split, "n_scored": n, "n_expected": n_expected, "error": error,
        "em": mean("em"), "f1": mean("f1"),
        "em_ci95": list(ci["recall"]) if ci else None, "f1_ci95": list(ci["complete_recall"]) if ci else None,
        "pct_with_valid_citation": 100 * sum(bool(r["citations"]) for r in rows) / n if n else 0.0,
        "pct_citing_gold_chunk": 100 * sum(r["cites_gold_chunk"] for r in rows) / n if n else 0.0,
        "n_invalid_citations": sum(len(r["invalid_citations"]) for r in rows),
        "mean_input_tokens": mean("input_tokens"), "mean_output_tokens": mean("output_tokens"),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "mean_cost_usd": mean("cost_usd"),
        "n_cached": sum(r["cached"] for r in rows),
        "gen_latency_s": {"mean": statistics.fmean(lat) if lat else 0.0,
                          "p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95)},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag.qa")
    p.add_argument("--data", required=True)
    p.add_argument("--report", required=True, help="report.json written by rag.report")
    p.add_argument("--system", required=True)
    p.add_argument("--split", choices=["dev", "eval"], default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-cost", type=float, default=5.0, help="stop after this much live spend (USD)")
    p.add_argument("--cache-dir", default="data/llm_cache")
    a = p.parse_args(argv)
    summary = run_qa(a.data, a.report, a.system, a.split, a.out, LLM(cache_dir=a.cache_dir),
                     a.workers, a.limit, a.max_cost)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
