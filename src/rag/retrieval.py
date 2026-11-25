"""Retrievers over the chunk corpus. Only Question.question is ever used as a query."""
from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

from .schema import Chunk

_TOKEN = re.compile(r"\w+")


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[str]:
        """Return the top-k chunk_ids, best first."""


def index_text(chunk: Chunk, prepend_title: bool = False) -> str:
    return f"{chunk.doc_title}. {chunk.text}" if prepend_title else chunk.text


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], prepend_title: bool = False) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(index_text(c, prepend_title)) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in top]


class DenseRetriever:
    """all-MiniLM-L6-v2 embeddings, ranked by cosine similarity.

    The embeddings are L2-normalized, so their dot product is cosine similarity.
    Keeping the small milestone corpus in a NumPy matrix is equivalent to a flat
    inner-product index and avoids a FAISS/PyTorch native-library conflict on
    macOS that makes the interpreter exit unexpectedly.
    """

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: Sequence[Chunk], prepend_title: bool = False,
                 batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.ids = [c.chunk_id for c in chunks]
        # Avoid automatically selecting Apple MPS: some PyTorch/MPS combinations
        # crash during MiniLM inference.  CPU is portable and sufficient for the
        # small evaluation corpus used by this project.
        self.model = SentenceTransformer(self.MODEL, device="cpu")
        self.embeddings = self.model.encode(
            [index_text(c, prepend_title) for c in chunks],
            batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")

    def search(self, query: str, k: int = 5) -> list[str]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        top = np.argsort(-(self.embeddings @ q[0]), kind="stable")[:k]
        return [self.ids[i] for i in top]


POOL = 50  # candidates taken from each retriever, and kept after fusion
RRF_K = 60


def rrf_fuse(rankings: Sequence[Sequence[str]], rrf_k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank), rank from 1.

    Ties break on chunk_id so the order is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


class HybridRetriever:
    """Top `pool` from BM25 and from dense, fused with RRF."""

    def __init__(self, bm25: Retriever, dense: Retriever, pool: int = POOL,
                 rrf_k: int = RRF_K) -> None:
        self.bm25, self.dense, self.pool, self.rrf_k = bm25, dense, pool, rrf_k

    def candidates(self, query: str) -> list[str]:
        """The fused top `pool` chunk_ids, best first."""
        fused = rrf_fuse([self.bm25.search(query, self.pool), self.dense.search(query, self.pool)],
                         self.rrf_k)
        return fused[: self.pool]

    def search(self, query: str, k: int = 5) -> list[str]:
        return self.candidates(query)[:k]


class RerankRetriever:
    """Rescore the hybrid's fused top `pool` with a cross-encoder and keep the top k.

    Chunks are scored against the query exactly as stored (no title prefix), so every
    system sees the same chunk text. `scorer` maps [(query, text), ...] to scores and
    is injectable for tests; by default the cross-encoder is loaded on first use, on CPU.
    """

    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, hybrid: HybridRetriever, chunks: Sequence[Chunk], scorer=None,
                 batch_size: int = 16) -> None:
        self.hybrid = hybrid
        self.text = {c.chunk_id: c.text for c in chunks}
        self.batch_size = batch_size
        self._scorer = scorer

    @property
    def scorer(self):
        if self._scorer is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.MODEL, device="cpu")
            self._scorer = lambda pairs: model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False)
        return self._scorer

    def rerank(self, query: str, candidates: Sequence[str], k: int = 5) -> list[str]:
        scores = self.scorer([(query, self.text[cid]) for cid in candidates])
        order = np.argsort(-np.asarray(scores, dtype="float64"), kind="stable")  # ties keep fused order
        return [candidates[i] for i in order[:k]]

    def search(self, query: str, k: int = 5) -> list[str]:
        return self.rerank(query, self.hybrid.candidates(query), k)


SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank")


def build_systems(chunks: Sequence[Chunk], names: Sequence[str] = SYSTEMS,
                  prepend_title: bool = False) -> dict[str, Retriever]:
    """Build the named systems, sharing one BM25 index and one dense index between them."""
    unknown = set(names) - set(SYSTEMS)
    if unknown:
        raise ValueError(f"unknown retriever(s) {sorted(unknown)}")
    need_bm25 = bool({"bm25", "hybrid", "hybrid_rerank"} & set(names))
    need_dense = bool({"dense", "hybrid", "hybrid_rerank"} & set(names))
    bm25 = BM25Retriever(chunks, prepend_title) if need_bm25 else None
    dense = DenseRetriever(chunks, prepend_title) if need_dense else None
    hybrid = HybridRetriever(bm25, dense) if {"hybrid", "hybrid_rerank"} & set(names) else None
    built = {"bm25": bm25, "dense": dense, "hybrid": hybrid,
             "hybrid_rerank": RerankRetriever(hybrid, chunks) if hybrid else None}
    return {n: built[n] for n in names}


def make_retriever(name: str, chunks: Sequence[Chunk], prepend_title: bool = False) -> Retriever:
    return build_systems(chunks, [name], prepend_title)[name]
