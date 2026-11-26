import logging

from rag.build import build_qasper
from rag.chunker import chunk_section, gold_chunk_ids, index_units
from rag.parse_qasper import EvidenceMapper, answer_text, build_document, parse_qasper
from rag.schema import Question, read_jsonl
from tests.helpers import qa_ann, qasper_row

INTRO = ["Intro paragraph one is here.", "", "Intro paragraph two follows right after."]
METHOD = ["We train the encoder with a contrastive objective on a large corpus of text."]


def paper(qas, pid="p1"):
    return qasper_row(pid, "The abstract text.",
                      [("Introduction", INTRO), (None, ["Unnamed section text."]),
                       ("Method ::: Encoder", METHOD), ("Empty", [""])], qas)


def test_document_structure_and_offsets():
    doc = build_document(paper([]))
    assert doc.doc_id == "qa::p1"
    assert [s.name for s in doc.sections] == ["Abstract", "Introduction", "", "Method ::: Encoder"]
    assert [s.section_id for s in doc.sections] == [f"qa::p1::sec{i}" for i in range(4)]
    intro = doc.sections[1]
    assert intro.text == "Intro paragraph one is here.\n\nIntro paragraph two follows right after."
    assert [u.unit_id for u in intro.units] == ["qa::p1::sec1::p0", "qa::p1::sec1::p1"]
    for sec in doc.sections:
        for u in sec.units:
            assert sec.text[u.start:u.end].strip() == sec.text[u.start:u.end]


def test_evidence_mapping_kinds():
    m = EvidenceMapper(build_document(paper([])))
    assert m.map("Intro paragraph two   follows right after.") == ("qa::p1::sec1::p1", "exact")
    assert m.map("encoder with a contrastive objective") == ("qa::p1::sec3::p0", "substring")
    fuzzy = "We train the encoder with a contrastive objective on a large corpus of texts."
    assert m.map(fuzzy) == ("qa::p1::sec3::p0", "fuzzy")
    assert m.map("Something that appears nowhere in the paper at all.") == (None, "unmatched")
    assert m.map("Method ::: Encoder") == (None, "section_heading")


def test_answer_text_priority():
    assert answer_text(qa_ann([], spans=["a", "b"], free="x", yes_no=True)) == "a, b"
    assert answer_text(qa_ann([], free="x", yes_no=True)) == "x"
    assert answer_text(qa_ann([], yes_no=True)) == "Yes"
    assert answer_text(qa_ann([], yes_no=False)) == "No"


def test_questions_annotators_and_drops(caplog):
    qas = [
        ("Q1?", "q1", [qa_ann(["FLOAT SELECTED: Figure 1: x", "Intro paragraph one is here."], spans=["s"]),
                       qa_ann(["encoder with a contrastive objective"], yes_no=False),
                       qa_ann([], unanswerable=True)]),
        ("Q2?", "q2", [qa_ann([], unanswerable=True)]),
        ("Q3?", "q3", [qa_ann(["FLOAT SELECTED: Table 2"], free="only a table")]),
        ("Q4?", "q4", [qa_ann(["Not in the paper whatsoever, really."], free="x")]),
    ]
    with caplog.at_level(logging.WARNING):
        docs, questions, stats = parse_qasper([paper(qas)])
    assert [q.qid for q in questions] == ["q1"]
    q = questions[0]
    assert q.gold_evidence_sets == [["qa::p1::sec1::p0"], ["qa::p1::sec3::p0"]]
    assert q.gold_answers == ["s", "No"]
    assert (q.source, q.qtype, q.doc_ids) == ("qasper", None, ["qa::p1"])
    assert stats["evidence_float_dropped"] == 2
    assert stats["annotators_unanswerable"] == 2
    assert stats["dropped_no_usable_annotator"] == 3
    assert stats["evidence_unmatched"] == 1 and "unmatched evidence" in caplog.text


def test_long_paragraph_coverage_is_relative_to_one_window():
    long_para = " ".join(f"word{i}" for i in range(700))  # ~4.9k chars
    row = qasper_row("p2", "abs", [("S", [long_para])],
                     [("Q?", "q", [qa_ann([long_para], free="a")])])
    docs, questions, _ = parse_qasper([row])
    doc, sec = docs[0], docs[0].sections[1]
    chunks = chunk_section(doc, sec)
    gold_unit = questions[0].gold_evidence_sets[0][0]
    # chunks fully inside the paragraph count as holding it; each is 100% of what one chunk can hold
    assert chunks[1].unit_coverage[gold_unit] == 1.0
    assert gold_chunk_ids(questions[0], index_units(chunks), 0.5)  # still retrievable
    assert len(gold_chunk_ids(questions[0], index_units(chunks), 0.5)) >= 4


def test_end_to_end_build(tmp_path):
    rows = [paper([("Question number %d?" % pid, f"q{pid}", [qa_ann(["Intro paragraph one is here."], free="a")])],
                  pid=f"p{pid}") for pid in range(8)]
    stats = build_qasper(rows, tmp_path, n_eval=5, n_dev=3, seed=1)
    ev = list(read_jsonl(tmp_path / "questions_eval.jsonl", Question))
    dev = list(read_jsonl(tmp_path / "questions_dev.jsonl", Question))
    assert (len(ev), len(dev)) == (5, 3)
    assert not {q.qid for q in ev} & {q.qid for q in dev}
    assert stats["pct_eval_questions_fully_covered"] == 100.0
    assert stats["n_documents"] == 8  # papers behind the sampled questions
    assert all(q.gold_chunk_ids for q in ev + dev)
    assert (tmp_path / "stats.json").exists()
