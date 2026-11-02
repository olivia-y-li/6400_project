from __future__ import annotations

import random

from .schema import Question


def sample_questions(
    questions: list[Question], n_eval: int = 500, n_dev: int = 50, seed: int = 6400
) -> tuple[list[Question], list[Question]]:
    """Disjoint (eval, dev) samples, deterministic for a seed regardless of input order."""
    if n_eval + n_dev > len(questions):
        raise ValueError(
            f"requested {n_eval}+{n_dev} questions but only {len(questions)} available"
        )
    pool = sorted(questions, key=lambda q: q.qid)
    picked = random.Random(seed).sample(pool, n_eval + n_dev)
    return picked[:n_eval], picked[n_eval:]
