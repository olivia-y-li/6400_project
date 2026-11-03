"""Recall@k, Complete Recall@k and MRR@k.

A gold unit counts as retrieved if some top-k chunk covers it at >= min_overlap.
With several gold evidence sets (QASPER annotators) a question takes its best set.
"""
from __future__ import annotations

import random
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
    return {m: sum(p[m] for p in per_question) / n for m in METRICS}


METRICS = ("recall", "complete_recall", "mrr")


def bootstrap_ci(per_question: Sequence[Mapping[str, float]], n_boot: int = 1000,
                 seed: int = 6400, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI of each metric's mean, resampling questions."""
    rng = random.Random(seed)
    n = len(per_question)
    means = {m: [] for m in METRICS}
    for _ in range(n_boot):
        sample = [per_question[rng.randrange(n)] for _ in range(n)]
        for m in METRICS:
            means[m].append(sum(p[m] for p in sample) / n)
    lo, hi = int(n_boot * alpha / 2), int(n_boot * (1 - alpha / 2)) - 1
    return {m: (sorted(v)[lo], sorted(v)[hi]) for m, v in means.items()}


def failure_breakdown(per_question: Sequence[Mapping[str, float]]) -> dict[str, int]:
    """complete: all gold found; partial: some but not all; miss: no gold unit in top-k."""
    out = {"complete": 0, "partial": 0, "miss": 0}
    for p in per_question:
        out["complete" if p["complete_recall"] else "partial" if p["recall"] > 0 else "miss"] += 1
    return out
