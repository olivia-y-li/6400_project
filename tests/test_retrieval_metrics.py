import pytest

from rag.metrics import score_question
from rag.retrieval import BM25Retriever, tokenize
from rag.schema import Chunk, Question


def chunk(cid, text, cov=None):
    return Chunk(cid, "d", "s", "Title", "", text, 0, len(text), cov or {})


def question(sets, gold_chunks):
    return Question("q", "hotpotqa", "?", ["a"], sets, gold_chunks, None, [])


CHUNKS = {c.chunk_id: c for c in [
    chunk("c1", "x", {"u1": 1.0}), chunk("c2", "x", {"u2": 1.0}),
    chunk("c3", "x", {"u3": 0.3}), chunk("c4", "x", {"u3": 1.0}),
]}


def test_recall_complete_and_mrr():
    q = question([["u1", "u2"]], ["c1", "c2"])
    s = score_question(q, ["c3", "c2", "c1"], CHUNKS)
    assert s == {"recall": 1.0, "complete_recall": 1.0, "mrr": 0.5}
    s = score_question(q, ["c3", "c2", "c4"], CHUNKS)
    assert s == {"recall": 0.5, "complete_recall": 0.0, "mrr": 0.5}
    assert score_question(q, ["c3", "c4"], CHUNKS)["mrr"] == 0.0


def test_k_cutoff_and_min_overlap():
    q = question([["u1", "u3"]], ["c1", "c4"])
    assert score_question(q, ["c3", "c1"], CHUNKS, k=1)["recall"] == 0.0
    # c3 only covers 30% of u3, below the 0.5 threshold
    assert score_question(q, ["c3", "c1"], CHUNKS)["complete_recall"] == 0.0
    assert score_question(q, ["c3", "c1"], CHUNKS, min_overlap=0.2)["complete_recall"] == 1.0


def test_multiple_evidence_sets_use_best_set():
    q = question([["u1", "u2"], ["u3"]], ["c1", "c2", "c4"])
    s = score_question(q, ["c4"], CHUNKS)
    assert s["complete_recall"] == 1.0 and s["recall"] == 1.0


def test_bm25_ranks_matching_chunk_first():
    pytest.importorskip("rank_bm25")
    cs = [chunk("a", "The river flows north."), chunk("b", "Oslo is the capital of Norway."),
          chunk("c", "Pizza and pasta are Italian foods.")]
    assert BM25Retriever(cs).search("what is the capital of Norway", 2)[0] == "b"
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_bootstrap_ci_and_failures():
    from rag.metrics import bootstrap_ci, failure_breakdown

    per_q = [{"recall": 1.0, "complete_recall": 1.0, "mrr": 1.0},
             {"recall": 0.5, "complete_recall": 0.0, "mrr": 0.5},
             {"recall": 0.0, "complete_recall": 0.0, "mrr": 0.0}] * 20
    ci = bootstrap_ci(per_q, n_boot=200)
    for m, (lo, hi) in ci.items():
        assert lo <= sum(p[m] for p in per_q) / len(per_q) <= hi
    assert ci == bootstrap_ci(per_q, n_boot=200)  # seeded
    assert failure_breakdown(per_q) == {"complete": 20, "partial": 20, "miss": 20}


def test_paired_counts():
    from rag.report import paired

    a = [{"complete_recall": 1.0}, {"complete_recall": 0.0}, {"complete_recall": 1.0}]
    b = [{"complete_recall": 0.0}, {"complete_recall": 0.0}, {"complete_recall": 1.0}]
    assert paired(a, b) == {"a_better": 1, "b_better": 0, "tie": 2}
