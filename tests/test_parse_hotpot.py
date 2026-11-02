import logging

from rag.parse_hotpot import build_section_text, parse_hotpot
from tests.helpers import ALPHA, BETA, hp_row


def test_spans_reconstruct_stripped_sentences():
    sents = [" One.", "Two.", "  Three. ", "", " Four."]
    text, spans = build_section_text(sents)
    for (s, e), sent in zip(spans, sents):
        assert text[s:e] == sent.strip()
    assert text == " One. Two.  Three.  Four."  # one space inserted between "One." and "Two."


def test_documents_have_one_section_and_all_sentences_as_units():
    docs, _, _ = parse_hotpot([hp_row("q1", "Q?", "A", {"Alpha": ALPHA, "Beta": BETA}, [("Alpha", 1)])])
    assert [d.doc_id for d in docs] == ["hp::Alpha", "hp::Beta"]
    sec = docs[0].sections[0]
    assert sec.section_id == "hp::Alpha::sec0" and sec.name == ""
    assert [u.unit_id for u in sec.units] == ["hp::Alpha::s0", "hp::Alpha::s1"]
    assert sec.text[sec.units[1].start:sec.units[1].end] == "It is based in Oslo."


def test_question_fields():
    _, qs, _ = parse_hotpot([hp_row("q1", "Q?", "A", {"Alpha": ALPHA, "Beta": BETA},
                                    [("Alpha", 0), ("Beta", 1)], qtype="comparison")])
    q = qs[0]
    assert (q.qid, q.question, q.gold_answers, q.qtype) == ("q1", "Q?", ["A"], "comparison")
    assert q.gold_evidence_sets == [["hp::Alpha::s0", "hp::Beta::s1"]]
    assert q.doc_ids == ["hp::Alpha", "hp::Beta"]


def test_duplicate_title_across_questions_yields_one_document():
    rows = [
        hp_row("q1", "Q1?", "A", {"Alpha": ALPHA, "Beta": BETA}, [("Alpha", 0)]),
        hp_row("q2", "Q2?", "B", {"Alpha": ALPHA}, [("Alpha", 1)]),
    ]
    docs, qs, _ = parse_hotpot(rows)
    assert len(docs) == 2
    assert qs[1].gold_evidence_sets == [["hp::Alpha::s1"]]


def test_conflicting_duplicate_gets_suffix_and_resolves_per_question():
    alt = [" Alpha was founded in 2001."]
    rows = [
        hp_row("q1", "Q1?", "A", {"Alpha": ALPHA}, [("Alpha", 0)]),
        hp_row("q2", "Q2?", "B", {"Alpha": alt}, [("Alpha", 0)]),
        hp_row("q3", "Q3?", "C", {"Alpha": alt}, [("Alpha", 0)]),
    ]
    docs, qs, stats = parse_hotpot(rows)
    assert [d.doc_id for d in docs] == ["hp::Alpha", "hp::Alpha#2"]
    assert qs[0].gold_evidence_sets == [["hp::Alpha::s0"]]
    assert qs[1].gold_evidence_sets == [["hp::Alpha#2::s0"]] == qs[2].gold_evidence_sets
    assert stats["conflicting_duplicate_paragraphs"] == 1


def test_invalid_facts_are_skipped_and_logged(caplog):
    row = hp_row("q1", "Q?", "A", {"Alpha": ALPHA},
                 [("Alpha", 0), ("Alpha", 9), ("Alpha", -1), ("Missing", 0)])
    with caplog.at_level(logging.WARNING):
        _, qs, stats = parse_hotpot([row])
    assert qs[0].gold_evidence_sets == [["hp::Alpha::s0"]]
    assert stats["invalid_fact_sent_id_out_of_range"] == 2
    assert stats["invalid_fact_title_missing"] == 1
    assert "out of range" in caplog.text and "not in context" in caplog.text


def test_question_with_no_valid_facts_is_dropped_and_counted():
    row = hp_row("q1", "Q?", "A", {"Alpha": ALPHA}, [("Alpha", 7)])
    docs, qs, stats = parse_hotpot([row])
    assert qs == [] and stats["dropped_no_valid_supporting_facts"] == 1
