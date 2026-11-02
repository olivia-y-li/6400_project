"""Dataclasses shared by every stage of the pipeline, plus JSONL io."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class EvidenceUnit:
    unit_id: str  # HotpotQA: f"{doc_id}::s{sent_id}"; QASPER: f"{section_id}::p{j}"
    start: int  # char offset into Section.text (inclusive)
    end: int  # char offset (exclusive); Section.text[start:end] == unit text (stripped)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvidenceUnit":
        return cls(unit_id=d["unit_id"], start=d["start"], end=d["end"])


@dataclass
class Section:
    section_id: str  # f"{doc_id}::sec{i}"
    name: str  # "" if missing
    text: str
    units: list[EvidenceUnit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "name": self.name,
            "text": self.text,
            "units": [u.to_dict() for u in self.units],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Section":
        return cls(
            section_id=d["section_id"],
            name=d["name"],
            text=d["text"],
            units=[EvidenceUnit.from_dict(u) for u in d["units"]],
        )


@dataclass
class Document:
    doc_id: str
    title: str
    source: str  # "hotpotqa" | "qasper"
    sections: list[Section] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "sections": [s.to_dict() for s in self.sections],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            source=d["source"],
            sections=[Section.from_dict(s) for s in d["sections"]],
        )


@dataclass
class Chunk:
    chunk_id: str  # f"{section_id}::c{n}"
    doc_id: str
    section_id: str
    doc_title: str
    section_name: str
    text: str  # == section.text[start:end]
    start: int
    end: int
    unit_coverage: dict[str, float] = field(default_factory=dict)  # only units with > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            section_id=d["section_id"],
            doc_title=d["doc_title"],
            section_name=d["section_name"],
            text=d["text"],
            start=d["start"],
            end=d["end"],
            unit_coverage=dict(d["unit_coverage"]),
        )


@dataclass
class Question:
    qid: str
    source: str
    question: str  # ONLY this field is passed to retrieval (no evidence leakage)
    gold_answers: list[str]
    gold_evidence_sets: list[list[str]]  # list of unit_id sets
    gold_chunk_ids: list[str]  # chunks covering ANY gold unit at >= min_overlap
    qtype: str | None
    doc_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Question":
        return cls(
            qid=d["qid"],
            source=d["source"],
            question=d["question"],
            gold_answers=list(d["gold_answers"]),
            gold_evidence_sets=[list(s) for s in d["gold_evidence_sets"]],
            gold_chunk_ids=list(d["gold_chunk_ids"]),
            qtype=d["qtype"],
            doc_ids=list(d["doc_ids"]),
        )


def write_jsonl(path: str | Path, items: Iterable[Any]) -> int:
    """Write objects with a to_dict() (or plain dicts) one per line; returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            d = item.to_dict() if hasattr(item, "to_dict") else item
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: Any = None) -> Iterator[Any]:
    """Yield dicts, or cls.from_dict(d) when cls is given."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                yield cls.from_dict(d) if cls is not None else d
