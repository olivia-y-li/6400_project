"""Answer EM and token F1, SQuAD-style as used by the HotpotQA and QASPER scorers.

A prediction is scored against every gold answer (one per usable annotator for QASPER)
and takes the best score.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

_SPECIAL = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str, special_tokens: bool = False) -> float:
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if special_tokens and (p in _SPECIAL or g in _SPECIAL) and p != g:
        return 0.0  # HotpotQA: yes/no/noanswer must match exactly
    p_tok, g_tok = p.split(), g.split()
    common = Counter(p_tok) & Counter(g_tok)
    n_same = sum(common.values())
    if n_same == 0:
        return float(p_tok == g_tok)  # both empty counts as a match
    precision, recall = n_same / len(p_tok), n_same / len(g_tok)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def score_answer(prediction: str, golds: Sequence[str], source: str) -> dict[str, float]:
    special = source == "hotpotqa"
    return {
        "em": max(exact_match(prediction, g) for g in golds),
        "f1": max(f1_score(prediction, g, special) for g in golds),
    }
