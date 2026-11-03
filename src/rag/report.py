"""Run every retriever variant on dev and eval and write one comparison report.

python -m rag.report --data data/processed/hotpotqa --out data/results/hotpotqa
Variants are compared on dev; eval is the number to report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import MIN_OVERLAP
from .evaluate import score_split, summarize
from .retrieval import make_retriever
from .schema import Chunk, Question, read_jsonl

VARIANTS = [("bm25", False), ("bm25", True), ("dense", False), ("dense", True)]


def name(retriever: str, title: bool) -> str:
    return retriever + ("+title" if title else "")


def paired(a: list[dict], b: list[dict], metric: str = "complete_recall") -> dict[str, int]:
    """Questions where a beats b / b beats a / tie, on one metric."""
    return {
        "a_better": sum(x[metric] > y[metric] for x, y in zip(a, b)),
        "b_better": sum(y[metric] > x[metric] for x, y in zip(a, b)),
        "tie": sum(x[metric] == y[metric] for x, y in zip(a, b)),
    }


def run(data_dir: str | Path, out_dir: str | Path, k: int = 5,
        min_overlap: float = MIN_OVERLAP) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    chunks = list(read_jsonl(data_dir / "chunks.jsonl", Chunk))
    by_id = {c.chunk_id: c for c in chunks}
    splits = {s: list(read_jsonl(data_dir / f"questions_{s}.jsonl", Question))
              for s in ("dev", "eval")}

    results, per_q_all = {}, {}
    for retriever, title in VARIANTS:
        r = make_retriever(retriever, chunks, title)
        for split, qs in splits.items():
            per_q, details = score_split(r, qs, by_id, k, min_overlap)
            key = f"{name(retriever, title)}/{split}"
            results[key] = summarize(per_q, qs)
            per_q_all[key] = (per_q, details)

    comparison = {
        "bm25_vs_dense_eval": paired(per_q_all["bm25/eval"][0], per_q_all["dense/eval"][0]),
        "title_effect_bm25_eval": paired(per_q_all["bm25/eval"][0], per_q_all["bm25+title/eval"][0]),
        "title_effect_dense_eval": paired(per_q_all["dense/eval"][0], per_q_all["dense+title/eval"][0]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(
        {"k": k, "min_overlap": min_overlap, "results": results, "comparison": comparison,
         "per_question": {key: d for key, (_, d) in per_q_all.items()}}, indent=1))
    (out_dir / "report.md").write_text(to_markdown(results, comparison, k, min_overlap))
    return {"results": results, "comparison": comparison}


def to_markdown(results: dict, comparison: dict, k: int, min_overlap: float) -> str:
    def row(key: str) -> str:
        r = results[key]
        cell = lambda m: f"{r[m]:.3f} [{r['ci95'][m][0]:.3f}, {r['ci95'][m][1]:.3f}]"
        return f"| {key} | {cell('recall')} | {cell('complete_recall')} | {cell('mrr')} |"

    head = f"| variant/split | Recall@{k} | Complete Recall@{k} | MRR@{k} |\n|---|---|---|---|\n"
    lines = [f"# HotpotQA retrieval results (k={k}, min_overlap={min_overlap})\n",
             "Mean [95% bootstrap CI over questions]. Report the eval rows; dev is for choosing variants.\n",
             head + "\n".join(row(k_) for k_ in results), "\n## By question type (eval, Complete Recall)\n",
             "| variant | bridge | comparison |\n|---|---|---|"]
    for key, r in results.items():
        if key.endswith("/eval"):
            bt = r["by_qtype"]
            lines.append(f"| {key[:-5]} | {bt['bridge']['complete_recall']:.3f} (n={bt['bridge']['n']}) "
                         f"| {bt['comparison']['complete_recall']:.3f} (n={bt['comparison']['n']}) |")
    lines += ["\n## Failure breakdown (eval)\n", "| variant | complete | partial | miss |\n|---|---|---|---|"]
    for key, r in results.items():
        if key.endswith("/eval"):
            f = r["failures"]
            lines.append(f"| {key[:-5]} | {f['complete']} | {f['partial']} | {f['miss']} |")
    lines += ["\n## Paired comparisons on eval (questions where one has the higher Complete Recall)\n"]
    for k_, v in comparison.items():
        lines.append(f"- {k_}: first better {v['a_better']}, second better {v['b_better']}, tie {v['tie']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag.report")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min-overlap", type=float, default=MIN_OVERLAP)
    a = p.parse_args(argv)
    run(a.data, a.out, a.k, a.min_overlap)
    print((Path(a.out) / "report.md").read_text())


if __name__ == "__main__":
    main()
