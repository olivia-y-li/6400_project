"""Fixed character-window chunking within sections, plus gold-chunk labeling."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .schema import Chunk, Document, Question, Section

CHUNK_SIZE = 1000
OVERLAP = 100
MIN_OVERLAP = 0.5


def chunk_section(
    doc: Document,
    section: Section,
    size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    stride = size - overlap
    text = section.text
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                chunk_id=f"{section.section_id}::c{len(chunks)}",
                doc_id=doc.doc_id,
                section_id=section.section_id,
                doc_title=doc.title,
                section_name=section.name,
                text=text[start:end],
                start=start,
                end=end,
                unit_coverage=_coverage(section, start, end),
            )
        )
        if end == len(text):  # last chunk ends exactly at len(text); no redundant tail
            break
        start += stride
    return chunks


def _coverage(section: Section, start: int, end: int) -> dict[str, float]:
    cov: dict[str, float] = {}
    for u in section.units:  # units are in text order
        if u.start >= end:
            break
        length = u.end - u.start
        if length <= 0:
            continue
        inter = min(end, u.end) - max(start, u.start)
        if inter > 0:
            cov[u.unit_id] = inter / length
    return cov


def chunk_documents(
    docs: Iterable[Document], size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[Chunk]:
    return [c for d in docs for s in d.sections for c in chunk_section(d, s, size, overlap)]


def index_units(chunks: Iterable[Chunk]) -> dict[str, list[tuple[str, float]]]:
    """unit_id -> [(chunk_id, coverage)]."""
    idx: dict[str, list[tuple[str, float]]]
    idx = defaultdict(list)
    for c in chunks:
        for unit_id, frac in c.unit_coverage.items():
            idx[unit_id].append((c.chunk_id, frac))
    return idx


def gold_chunk_ids(
    q: Question, unit_index: dict[str, list[tuple[str, float]]], min_overlap: float = MIN_OVERLAP
) -> list[str]:
    """Chunks covering any gold unit (union over evidence sets) at >= min_overlap."""
    out: list[str] = []
    for ev_set in q.gold_evidence_sets:
        for unit_id in ev_set:
            for chunk_id, frac in unit_index.get(unit_id, ()):
                if frac >= min_overlap and chunk_id not in out:
                    out.append(chunk_id)
    return out


def label_questions(
    questions: Iterable[Question], chunks: list[Chunk], min_overlap: float = MIN_OVERLAP
) -> list[Question]:
    idx = index_units(chunks)
    return [replace(q, gold_chunk_ids=gold_chunk_ids(q, idx, min_overlap)) for q in questions]
