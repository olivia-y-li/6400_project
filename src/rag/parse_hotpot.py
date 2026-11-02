"""HotpotQA (distractor) raw rows -> Documents + Questions."""
from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from typing import Any, Iterable

from .schema import Document, EvidenceUnit, Question, Section

log = logging.getLogger(__name__)

SOURCE = "hotpotqa"


def normalize_title(title: str) -> str:
    return unicodedata.normalize("NFC", " ".join(title.split()))


def build_section_text(sentences: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate sentences; return (text, spans) with spans on the stripped sentences.

    A single space is inserted between two pieces only when neither side has
    whitespace at the join. Invariant: text[start:end] == sentence.strip().
    """
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    length = 0
    last = ""
    for sent in sentences:
        if length and sent and not last.isspace() and not sent[0].isspace():
            pieces.append(" ")
            length += 1
        lead = len(sent) - len(sent.lstrip())
        stripped = sent.strip()
        start = length + lead
        spans.append((start, start + len(stripped)))
        pieces.append(sent)
        length += len(sent)
        if sent:
            last = sent[-1]
    text = "".join(pieces)
    for (s, e), sent in zip(spans, sentences):
        assert text[s:e] == sent.strip()
    return text, spans


class HotpotParser:
    """Stateful so that identical paragraphs across questions share one Document."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self._versions: dict[str, list[str]] = {}  # normalized title -> doc_ids
        self.stats: Counter[str] = Counter()

    def add_paragraph(self, title: str, sentences: list[str]) -> str:
        norm = normalize_title(title)
        text, spans = build_section_text(sentences)
        versions = self._versions.setdefault(norm, [])
        for doc_id in versions:
            if self.documents[doc_id].sections[0].text == text:
                return doc_id
        n = len(versions) + 1
        base = f"hp::{norm}"
        doc_id = base if n == 1 else f"{base}#{n}"
        if n > 1:
            self.stats["conflicting_duplicate_paragraphs"] += 1
            log.info("title %r has differing text; created %s", title, doc_id)
        section_id = f"{doc_id}::sec0"
        units = [
            EvidenceUnit(unit_id=f"{doc_id}::s{i}", start=s, end=e)
            for i, (s, e) in enumerate(spans)
        ]
        self.documents[doc_id] = Document(
            doc_id=doc_id,
            title=norm,
            source=SOURCE,
            sections=[Section(section_id=section_id, name="", text=text, units=units)],
        )
        versions.append(doc_id)
        return doc_id

    def parse_row(self, row: dict[str, Any]) -> Question | None:
        """Register the row's paragraphs and build its Question (None if dropped)."""
        self.stats["rows_seen"] += 1
        qid = row["id"]
        ctx = row["context"]

        title_to_doc: dict[str, str] = {}
        doc_ids: list[str] = []
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            doc_id = self.add_paragraph(title, list(sentences))
            title_to_doc.setdefault(normalize_title(title), doc_id)
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

        unit_ids: list[str] = []
        sf = row["supporting_facts"]
        for title, sent_id in zip(sf["title"], sf["sent_id"]):
            doc_id = title_to_doc.get(normalize_title(title))
            if doc_id is None:
                self.stats["invalid_fact_title_missing"] += 1
                log.warning("%s: supporting fact title %r not in context; skipped", qid, title)
                continue
            n_units = len(self.documents[doc_id].sections[0].units)
            if not 0 <= sent_id < n_units:
                self.stats["invalid_fact_sent_id_out_of_range"] += 1
                log.warning(
                    "%s: sent_id %s out of range for %r (%d sentences); skipped",
                    qid, sent_id, title, n_units,
                )
                continue
            unit_id = f"{doc_id}::s{sent_id}"
            if unit_id not in unit_ids:
                unit_ids.append(unit_id)

        if not unit_ids:
            self.stats["dropped_no_valid_supporting_facts"] += 1
            log.warning("%s: no valid supporting facts; question dropped", qid)
            return None

        self.stats["questions_kept"] += 1
        return Question(
            qid=qid,
            source=SOURCE,
            question=row["question"],
            gold_answers=[row["answer"]],
            gold_evidence_sets=[unit_ids],
            gold_chunk_ids=[],  # filled in by the chunker once chunks exist
            qtype=row.get("type"),
            doc_ids=doc_ids,
        )


def parse_hotpot(rows: Iterable[dict[str, Any]]) -> tuple[list[Document], list[Question], Counter]:
    parser = HotpotParser()
    questions = [q for q in map(parser.parse_row, rows) if q is not None]
    return list(parser.documents.values()), questions, parser.stats


def build_corpus(
    questions: list[Question], raw_rows: Iterable[dict[str, Any]]
) -> tuple[list[Document], list[Question], Counter]:
    """Re-parse only the rows behind `questions` so the corpus holds just their documents.

    Returns the documents plus the questions re-resolved against this corpus (doc
    ids and unit ids can differ from the full-dataset parse when titles conflict),
    in the order given.
    """
    by_id = {r["id"]: r for r in raw_rows}
    rows = [by_id[q.qid] for q in questions]
    return parse_hotpot(rows)
