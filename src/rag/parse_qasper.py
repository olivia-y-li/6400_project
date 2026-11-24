"""QASPER raw rows -> Documents + Questions."""
from __future__ import annotations

import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable

from .schema import Document, EvidenceUnit, Question, Section

log = logging.getLogger(__name__)

SOURCE = "qasper"
FLOAT_PREFIX = "FLOAT SELECTED:"
FUZZY_THRESHOLD = 0.9


def norm_ws(text: str) -> str:
    return " ".join(text.split())


def build_document(row: dict[str, Any]) -> Document:
    """Section 0 is the abstract; then one Section per non-empty full_text section."""
    doc_id = f"qa::{row['id']}"
    ft = row["full_text"]
    raw = [("Abstract", [row.get("abstract") or ""])]
    raw += [(name or "", paras) for name, paras in zip(ft["section_name"], ft["paragraphs"])]

    sections: list[Section] = []
    for name, paras in raw:
        paras = [p.strip() for p in paras if p and p.strip()]
        if not paras:
            continue
        section_id = f"{doc_id}::sec{len(sections)}"
        units, parts, pos = [], [], 0
        for j, p in enumerate(paras):
            units.append(EvidenceUnit(f"{section_id}::p{j}", pos, pos + len(p)))
            parts.append(p)
            pos += len(p) + 2  # "\n\n" separator
        text = "\n\n".join(parts)
        for u, p in zip(units, paras):
            assert text[u.start:u.end] == p
        sections.append(Section(section_id, name.strip(), text, units))
    return Document(doc_id, row["title"].strip(), SOURCE, sections)


class EvidenceMapper:
    """Maps an evidence string to one paragraph unit of a paper."""

    def __init__(self, doc: Document) -> None:
        self.paras: list[tuple[str, str]] = []  # (unit_id, whitespace-normalized text)
        self.headings = {norm_ws(sec.name) for sec in doc.sections if sec.name}
        for sec in doc.sections:
            for u in sec.units:
                self.paras.append((u.unit_id, norm_ws(sec.text[u.start:u.end])))

    def map(self, evidence: str) -> tuple[str | None, str]:
        """Return (unit_id or None, kind).

        kind is exact/substring/fuzzy, or for unmapped strings section_heading (the
        annotator pointed at a section title, not a paragraph) or unmatched.
        """
        ev = norm_ws(evidence)
        if not ev:
            return None, "unmatched"
        for unit_id, para in self.paras:
            if ev == para:
                return unit_id, "exact"
        for unit_id, para in self.paras:
            if ev in para:
                return unit_id, "substring"
        best, best_ratio = None, 0.0
        for unit_id, para in self.paras:
            sm = SequenceMatcher(None, ev, para, autojunk=False)
            if sm.real_quick_ratio() < FUZZY_THRESHOLD or sm.quick_ratio() < FUZZY_THRESHOLD:
                continue
            ratio = sm.ratio()
            if ratio > best_ratio:
                best, best_ratio = unit_id, ratio
        if best is not None and best_ratio >= FUZZY_THRESHOLD:
            return best, "fuzzy"
        return None, "section_heading" if ev in self.headings else "unmatched"


def answer_text(a: dict[str, Any]) -> str:
    """Extractive spans joined with ', ', else free-form answer, else Yes/No."""
    spans = [s.strip() for s in a.get("extractive_spans") or [] if s and s.strip()]
    if spans:
        return ", ".join(spans)
    if (a.get("free_form_answer") or "").strip():
        return a["free_form_answer"].strip()
    if a.get("yes_no") is not None:
        return "Yes" if a["yes_no"] else "No"
    return ""


def parse_paper(row: dict[str, Any], stats: Counter) -> tuple[Document, list[Question]]:
    doc = build_document(row)
    mapper = EvidenceMapper(doc)
    qas = row["qas"]
    questions: list[Question] = []
    for question, qid, answers in zip(qas["question"], qas["question_id"], qas["answers"]):
        stats["questions_seen"] += 1
        sets: list[list[str]] = []
        gold_answers: list[str] = []
        for ann in answers["answer"]:
            stats["annotators_seen"] += 1
            if ann["unanswerable"]:
                stats["annotators_unanswerable"] += 1
                continue
            unit_ids: list[str] = []
            for ev in ann["evidence"]:
                stats["evidence_strings"] += 1
                if ev.startswith(FLOAT_PREFIX):
                    stats["evidence_float_dropped"] += 1
                    continue
                unit_id, kind = mapper.map(ev)
                stats[f"evidence_{kind}"] += 1
                if kind == "unmatched":
                    log.warning("%s: unmatched evidence %r", qid, ev[:80])
                elif unit_id is not None and unit_id not in unit_ids:
                    unit_ids.append(unit_id)
            if not unit_ids:
                stats["annotators_no_mapped_evidence"] += 1
                continue
            stats["annotators_usable"] += 1
            sets.append(unit_ids)
            gold_answers.append(answer_text(ann))
        if not sets:
            stats["dropped_no_usable_annotator"] += 1
            continue
        stats["questions_kept"] += 1
        questions.append(Question(
            qid=qid, source=SOURCE, question=question.strip(), gold_answers=gold_answers,
            gold_evidence_sets=sets, gold_chunk_ids=[], qtype=None, doc_ids=[doc.doc_id],
        ))
    return doc, questions


def parse_qasper(rows: Iterable[dict[str, Any]]) -> tuple[list[Document], list[Question], Counter]:
    stats: Counter = Counter()
    docs: list[Document] = []
    questions: list[Question] = []
    for row in rows:
        doc, qs = parse_paper(row, stats)
        docs.append(doc)
        questions.extend(qs)
    return docs, questions, stats


def build_corpus(questions: list[Question], documents: list[Document]) -> list[Document]:
    """Only the papers behind the given questions (all of each paper)."""
    needed = {d for q in questions for d in q.doc_ids}
    return [d for d in documents if d.doc_id in needed]
