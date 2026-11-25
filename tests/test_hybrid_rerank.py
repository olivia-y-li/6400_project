import pytest

from rag.metrics import paired_bootstrap
from rag.retrieval import HybridRetriever, RerankRetriever, build_systems, rrf_fuse
from rag.schema import Chunk


class Fixed:
    """A retriever that always returns the same ranking."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def search(self, query, k=5):
        self.calls.append(k)
        return self.ranking[:k]


def chunk(cid, text=None):
    return Chunk(cid, "d", "s", "T", "", text or cid, 0, 1, {})


def test_rrf_scores_and_order():
    # a: 1/61 + 1/62 (rank 1 and 2); b: 1/62 + 1/61 (rank 2 and 1) -> tie, broken by id
    assert rrf_fuse([["a", "b"], ["b", "a"]]) == ["a", "b"]
    # c appears in both lists, so it beats x and y that appear in one list at the same ranks
    assert rrf_fuse([["x", "c"], ["y", "c"]])[0] == "c"
    # exact scores with the default constant of 60
    # c is rank 3 and rank 1: 1/63 + 1/61 beats a, rank 1 in one list only: 1/61
    assert rrf_fuse([["a", "b", "c"], ["c"]]) == ["c", "a", "b"]


def test_rrf_constant_matters():
    # rrf_k=60: b (rank 2 in both lists) beats a and c (rank 1 in one list)
    # rrf_k=0:  a and c score 1 and b scores 1/2 + 1/2 = 1, a three-way tie broken by id
    lists = [["a", "b"], ["c", "b"]]
    assert rrf_fuse(lists, rrf_k=60) == ["b", "a", "c"]
    assert rrf_fuse(lists, rrf_k=0) == ["a", "b", "c"]


def test_hybrid_pool_and_truncation():
    bm25 = Fixed([f"b{i}" for i in range(100)])
    dense = Fixed([f"d{i}" for i in range(100)])
    h = HybridRetriever(bm25, dense, pool=50)
    cands = h.candidates("q")
    assert len(cands) == 50 and bm25.calls == [50] and dense.calls == [50]
    assert h.search("q", 5) == cands[:5]
    assert set(cands[:2]) == {"b0", "d0"}


def test_rerank_orders_by_score_and_keeps_fused_order_on_ties():
    chunks = [chunk(c) for c in "abcde"]
    hybrid = HybridRetriever(Fixed(list("abcde")), Fixed(list("abcde")), pool=5)
    score = {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.5, "e": 0.0}
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [score[t] for _, t in pairs]

    r = RerankRetriever(hybrid, chunks, scorer=scorer)
    assert r.search("my query", 3) == ["b", "c", "d"]  # b before c: tie keeps fused order
    assert all(q == "my query" for q, _ in seen) and len(seen) == 5


def test_rerank_scores_chunk_text_without_title():
    chunks = [Chunk("a", "d", "s", "SOME TITLE", "", "body text", 0, 9, {})]
    hybrid = HybridRetriever(Fixed(["a"]), Fixed(["a"]), pool=5)
    seen = []
    RerankRetriever(hybrid, chunks, scorer=lambda p: seen.extend(p) or [1.0]).search("q", 1)
    assert seen == [("q", "body text")]


def test_build_systems_rejects_unknown():
    with pytest.raises(ValueError):
        build_systems([], ["nope"])


def test_paired_bootstrap():
    a = [{"m": 0.0}] * 40 + [{"m": 1.0}] * 10
    b = [{"m": 1.0}] * 50
    r = paired_bootstrap(a, b, "m", n_boot=500)
    assert r["diff"] == pytest.approx(0.8) and r["p"] < 0.01 and r["ci95_low"] > 0.5
    same = paired_bootstrap(a, a, "m", n_boot=500)
    assert same["diff"] == 0 and same["p"] == 1.0
    assert r == paired_bootstrap(a, b, "m", n_boot=500)  # seeded
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:3], "m")
