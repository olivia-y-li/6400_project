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
    """all-MiniLM-L6-v2 embeddings, cosine similarity via inner product on a FAISS flat index."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: Sequence[Chunk], prepend_title: bool = False,
                 batch_size: int = 64) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.ids = [c.chunk_id for c in chunks]
        self.model = SentenceTransformer(self.MODEL)
        emb = self.model.encode(
            [index_text(c, prepend_title) for c in chunks],
            batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int = 5) -> list[str]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(q, k)
        return [self.ids[i] for i in idx[0] if i >= 0]


def make_retriever(name: str, chunks: Sequence[Chunk], prepend_title: bool = False) -> Retriever:
    if name == "bm25":
        return BM25Retriever(chunks, prepend_title)
    if name == "dense":
        return DenseRetriever(chunks, prepend_title)
    raise ValueError(f"unknown retriever {name!r}")
