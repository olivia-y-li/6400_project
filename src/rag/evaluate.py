"""CLI: python -m rag.evaluate --data data/processed/hotpotqa --retriever bm25"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import MIN_OVERLAP
from .metrics import aggregate, bootstrap_ci, failure_breakdown, score_question
from .retrieval import make_retriever
from .schema import Chunk, Question, read_jsonl


def score_split(r, questions: list[Question], by_id: dict[str, Chunk], k: int,
                min_overlap: float) -> tuple[list[dict], list[dict]]:
    per_q, details = [], []
    for q in questions:
        ranked = r.search(q.question, k)  # the question text only; no gold information
        s = score_question(q, ranked, by_id, k, min_overlap)
        per_q.append(s)
        details.append({"qid": q.qid, "retrieved": ranked, **s})
    return per_q, details


def summarize(per_q: list[dict], questions: list[Question]) -> dict:
    return {
        **aggregate(per_q),
        "ci95": {m: [round(x, 4) for x in ci] for m, ci in bootstrap_ci(per_q).items()},
        "failures": failure_breakdown(per_q),
        "by_qtype": {
            t: {**aggregate([s for s, q in zip(per_q, questions) if q.qtype == t]),
                "n": sum(q.qtype == t for q in questions)}
            for t in sorted({q.qtype for q in questions if q.qtype})
        },
    }


def evaluate(data_dir: str | Path, retriever: str, k: int = 5, split: str = "eval",
             min_overlap: float = MIN_OVERLAP, prepend_title: bool = False,
             out_path: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    chunks = list(read_jsonl(data_dir / "chunks.jsonl", Chunk))
    by_id = {c.chunk_id: c for c in chunks}
    questions = list(read_jsonl(data_dir / f"questions_{split}.jsonl", Question))

    r = make_retriever(retriever, chunks, prepend_title)
    per_q, details = score_split(r, questions, by_id, k, min_overlap)
    result = {
        "retriever": retriever, "split": split, "k": k, "min_overlap": min_overlap,
        "prepend_title": prepend_title, "n_questions": len(questions),
        "n_chunks": len(chunks), **summarize(per_q, questions),
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**result, "per_question": details}, indent=1))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag.evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--retriever", choices=["bm25", "dense"], required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--split", choices=["eval", "dev"], default="eval")
    p.add_argument("--min-overlap", type=float, default=MIN_OVERLAP)
    p.add_argument("--prepend-title", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)
    print(json.dumps(evaluate(a.data, a.retriever, a.k, a.split, a.min_overlap,
                              a.prepend_title, a.out), indent=2))


if __name__ == "__main__":
    main()
