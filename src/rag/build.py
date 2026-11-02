"""CLI: python -m rag.build --benchmark hotpotqa --out data/processed/hotpotqa"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from .chunker import CHUNK_SIZE, MIN_OVERLAP, OVERLAP, chunk_documents, index_units, label_questions
from .parse_hotpot import build_corpus, parse_hotpot
from .sampling import sample_questions
from .schema import Chunk, Question, write_jsonl

log = logging.getLogger("rag.build")


def get_raw_rows(benchmark: str, level: str | None = "hard") -> list[dict[str, Any]]:
    """Single swap point for the teammate's loader."""
    if benchmark == "hotpotqa":
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        if level:
            ds = ds.filter(lambda x: x["level"] == level)
        return [dict(r) for r in ds]
    raise NotImplementedError(f"benchmark {benchmark!r} is not implemented yet")


def build_hotpotqa(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    n_eval: int = 500,
    n_dev: int = 50,
    seed: int = 6400,
    size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    min_overlap: float = MIN_OVERLAP,
) -> dict[str, Any]:
    out = Path(out_dir)

    # Pass 1: parse everything so drops happen before sampling.
    _, all_questions, full_stats = parse_hotpot(rows)
    eval_q, dev_q = sample_questions(all_questions, n_eval, n_dev, seed)

    # Pass 2: corpus holds only the sampled questions' paragraphs (eval + dev together).
    documents, questions, corpus_stats = build_corpus(eval_q + dev_q, rows)
    chunks = chunk_documents(documents, size, overlap)
    questions = label_questions(questions, chunks, min_overlap)
    eval_out, dev_out = questions[: len(eval_q)], questions[len(eval_q):]

    write_jsonl(out / "documents.jsonl", documents)
    write_jsonl(out / "chunks.jsonl", chunks)
    write_jsonl(out / "questions_eval.jsonl", eval_out)
    write_jsonl(out / "questions_dev.jsonl", dev_out)

    stats = compute_stats(documents, chunks, eval_out, dev_out, full_stats, corpus_stats, min_overlap)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def compute_stats(documents, chunks: list[Chunk], eval_q: list[Question], dev_q: list[Question],
                  full_stats, corpus_stats, min_overlap: float) -> dict[str, Any]:
    lengths = [len(c.text) for c in chunks]
    idx = index_units(chunks)

    def covered(q: Question) -> bool:
        units = {u for s in q.gold_evidence_sets for u in s}
        return all(any(f >= min_overlap for _, f in idx.get(u, ())) for u in units)

    uncovered = [q.qid for q in eval_q if not covered(q)]
    n_units = [len({u for s in q.gold_evidence_sets for u in s}) for q in eval_q]
    stats: dict[str, Any] = {
        "n_documents": len(documents),
        "n_sections": sum(len(d.sections) for d in documents),
        "n_chunks": len(chunks),
        "n_questions_eval": len(eval_q),
        "n_questions_dev": len(dev_q),
        "chunk_len": {
            "min": min(lengths, default=0),
            "mean": round(mean(lengths), 1) if lengths else 0,
            "max": max(lengths, default=0),
        },
        "questions_dropped": {
            "no_valid_supporting_facts": full_stats["dropped_no_valid_supporting_facts"],
        },
        "hotpot_invalid_supporting_facts_all_rows": {
            "title_missing_from_context": full_stats["invalid_fact_title_missing"],
            "sent_id_out_of_range": full_stats["invalid_fact_sent_id_out_of_range"],
        },
        "hotpot_invalid_supporting_facts_sampled": {
            "title_missing_from_context": corpus_stats["invalid_fact_title_missing"],
            "sent_id_out_of_range": corpus_stats["invalid_fact_sent_id_out_of_range"],
        },
        "hotpot_conflicting_duplicate_paragraphs_all_rows": full_stats["conflicting_duplicate_paragraphs"],
        "hotpot_conflicting_duplicate_paragraphs_sampled": corpus_stats["conflicting_duplicate_paragraphs"],
        "mean_gold_units_per_eval_question": round(mean(n_units), 2) if n_units else 0,
        "mean_gold_chunks_per_eval_question":
            round(mean(len(q.gold_chunk_ids) for q in eval_q), 2) if eval_q else 0,
        "pct_eval_questions_fully_covered":
            round(100 * (1 - len(uncovered) / len(eval_q)), 2) if eval_q else None,
        "uncovered_eval_qids": uncovered[:20],
    }
    share = _share_over_256_tokens(chunks)
    if share is not None:
        stats["pct_chunks_over_256_minilm_tokens"] = share
    return stats


def _share_over_256_tokens(chunks: list[Chunk]) -> float | None:
    """Only when sentence-transformers (and the model's tokenizer) is available."""
    try:
        from sentence_transformers import SentenceTransformer

        tok = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").tokenizer
        over = sum(len(tok(c.text, add_special_tokens=True)["input_ids"]) > 256 for c in chunks)
        return round(100 * over / len(chunks), 2) if chunks else 0.0
    except Exception as e:  # optional metric; never fail the build
        log.info("skipping token-length stat: %s", e)
        return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag.build")
    p.add_argument("--benchmark", choices=["hotpotqa", "qasper"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--seed", type=int, default=6400)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=OVERLAP)
    p.add_argument("--min-overlap", type=float, default=MIN_OVERLAP)
    p.add_argument("--level", default="hard", help="HotpotQA difficulty filter ('' for all)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rows = get_raw_rows(args.benchmark, args.level or None)
    if args.benchmark != "hotpotqa":
        raise NotImplementedError("QASPER parsing is not implemented yet")
    stats = build_hotpotqa(rows, args.out, args.n_eval, args.n_dev, args.seed,
                           args.chunk_size, args.overlap, args.min_overlap)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
