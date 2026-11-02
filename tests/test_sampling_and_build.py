import json

import pytest

from rag.build import build_hotpotqa
from rag.sampling import sample_questions
from rag.schema import Chunk, Document, Question, read_jsonl
from tests.helpers import ALPHA, BETA, hp_row


def make_questions(n):
    return [Question(f"q{i:03d}", "hotpotqa", "?", ["a"], [["u"]], [], None, []) for i in range(n)]


def test_sampling_deterministic_and_disjoint():
    qs = make_questions(100)
    e1, d1 = sample_questions(qs, 30, 10, seed=1)
    e2, d2 = sample_questions(list(reversed(qs)), 30, 10, seed=1)
    assert [q.qid for q in e1] == [q.qid for q in e2] and [q.qid for q in d1] == [q.qid for q in d2]
    assert (len(e1), len(d1)) == (30, 10)
    assert not {q.qid for q in e1} & {q.qid for q in d1}
    assert [q.qid for q in sample_questions(qs, 30, 10, seed=2)[0]] != [q.qid for q in e1]


def test_sampling_too_few_questions():
    with pytest.raises(ValueError):
        sample_questions(make_questions(5), 4, 4)


def rows(n):
    out = []
    for i in range(n):
        paras = {"Alpha": ALPHA, "Beta": BETA, f"Extra {i}": [f" Filler {i} sentence."]}
        out.append(hp_row(f"q{i}", f"Which question number {i}?", f"ans{i}", paras,
                          [("Alpha", 0), (f"Extra {i}", 0)]))
    return out


def test_end_to_end_build(tmp_path):
    stats = build_hotpotqa(rows(12), tmp_path, n_eval=8, n_dev=4, seed=3)
    for f in ["documents.jsonl", "chunks.jsonl", "questions_eval.jsonl", "questions_dev.jsonl", "stats.json"]:
        assert (tmp_path / f).exists()
    docs = list(read_jsonl(tmp_path / "documents.jsonl", Document))
    chunks = list(read_jsonl(tmp_path / "chunks.jsonl", Chunk))
    ev = list(read_jsonl(tmp_path / "questions_eval.jsonl", Question))
    dev = list(read_jsonl(tmp_path / "questions_dev.jsonl", Question))
    assert (len(ev), len(dev)) == (8, 4)
    assert not {q.qid for q in ev} & {q.qid for q in dev}
    assert len(docs) == 2 + 12  # shared Alpha/Beta + one extra per question
    assert stats["n_documents"] == len(docs) and stats["n_chunks"] == len(chunks)
    assert stats["pct_eval_questions_fully_covered"] == 100.0
    assert json.loads((tmp_path / "stats.json").read_text())["n_questions_eval"] == 8
    chunk_ids = {c.chunk_id for c in chunks}
    for q in ev + dev:
        assert q.gold_chunk_ids and set(q.gold_chunk_ids) <= chunk_ids
        assert "hp::Alpha::sec0::c0" in q.gold_chunk_ids  # shared paragraph is gold for all


def test_question_field_has_no_evidence_leakage(tmp_path):
    raw = rows(6)
    build_hotpotqa(raw, tmp_path, n_eval=4, n_dev=2, seed=3)
    raw_by_id = {r["id"]: r for r in raw}
    for name in ["questions_eval.jsonl", "questions_dev.jsonl"]:
        for d in read_jsonl(tmp_path / name):
            r = raw_by_id[d["qid"]]
            assert d["question"] == r["question"]  # exactly the raw question, nothing appended
            for sent in ALPHA + [s for ss in r["context"]["sentences"] for s in ss]:
                assert sent.strip() not in d["question"]


def test_schema_roundtrip():
    q = Question("q", "hotpotqa", "?", ["a"], [["u1"], ["u2"]], ["c"], "bridge", ["d"])
    assert Question.from_dict(q.to_dict()) == q
    docs = list(__import__("rag.parse_hotpot", fromlist=["x"]).parse_hotpot(
        [hp_row("q", "Q?", "A", {"Alpha": ALPHA}, [("Alpha", 0)])])[0])
    assert Document.from_dict(docs[0].to_dict()) == docs[0]
