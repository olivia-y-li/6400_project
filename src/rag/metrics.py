"""Recall@k, Complete Recall@k and MRR@k.

A gold unit counts as retrieved if some top-k chunk covers it at >= min_overlap.
With several gold evidence sets (QASPER annotators) a question takes its best set.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .chunker import MIN_OVERLAP
from .schema import Chunk, Question


def covered_units(ranked_ids: Sequence[str], chunks: Mapping[str, Chunk],
                  min_overlap: float = MIN_OVERLAP) -> set[str]:
    return {
        u for cid in ranked_ids for u, frac in chunks[cid].unit_coverage.items()
        if frac >= min_overlap
    }


def score_question(q: Question, ranked_ids: Sequence[str], chunks: Mapping[str, Chunk],
                   k: int = 5, min_overlap: float = MIN_OVERLAP) -> dict[str, float]:
    top = list(ranked_ids[:k])
    got = covered_units(top, chunks, min_overlap)
    recall = max(len(got & set(s)) / len(set(s)) for s in q.gold_evidence_sets)
    complete = float(any(set(s) <= got for s in q.gold_evidence_sets))
    gold = set(q.gold_chunk_ids)
    rr = next((1 / r for r, cid in enumerate(top, 1) if cid in gold), 0.0)
    return {"recall": recall, "complete_recall": complete, "mrr": rr}


def aggregate(per_question: Sequence[Mapping[str, float]]) -> dict[str, float]:
    n = len(per_question)
    return {m: sum(p[m] for p in per_question) / n for m in ("recall", "complete_recall", "mrr")}
