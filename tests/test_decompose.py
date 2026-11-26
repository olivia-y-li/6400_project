import pytest

from rag.decompose import (DecompositionRetriever, fill_placeholders, interleave,
                           needed_answers, parse_sub_questions)
from rag.llm import LLMResult
from rag.retrieval import RerankRetriever
from rag.schema import Chunk

ORIGINAL = "Who directed the film that won Best Picture in 1998?"


def chunk(cid):
    return Chunk(cid, "d", "s", "T", "", f"text of {cid}", 0, 1, {})


class FakeLLM:
    def __init__(self, subs, answer="Titanic"):
        self.subs, self.answer, self.calls = subs, answer, []

    def call_tool(self, system, user, tool):
        self.calls.append((tool["name"], user))
        if tool["name"] == "decompose_question":
            out = {"sub_questions": self.subs}
        else:
            out = {"answer": self.answer, "citations": []}
        return LLMResult(out, 100, 10, 0.001, 0.5, cached=False)


class FakeHybrid:
    def __init__(self, table):
        self.table, self.queries = table, []

    def candidates(self, query):
        self.queries.append(query)
        return self.table[query]


def make(subs, table, answer="Titanic", scores=None):
    seen = []
    chunks = [chunk(c) for c in "abcdefg"]
    scores = scores or {}

    def scorer(pairs):
        seen.append(pairs)
        return [scores.get(t.split()[-1], 0.0) for _, t in pairs]

    hybrid = FakeHybrid(table)
    llm = FakeLLM(subs, answer)
    reranker = RerankRetriever(hybrid, chunks, scorer=scorer)
    return DecompositionRetriever(hybrid, reranker, chunks, llm), hybrid, llm, seen


def test_dependent_subquestion_is_filled_from_earlier_passages():
    subs = ["Which film won Best Picture in 1998?", "Who directed {1}?"]
    table = {subs[0]: ["a", "b"], "Who directed Titanic?": ["c", "d"]}
    d, hybrid, llm, seen = make(subs, table, scores={"c": 3, "a": 2, "d": 1, "b": 0})
    final, trace = d.run(ORIGINAL, k=3)
    assert trace.resolved == [subs[0], "Who directed Titanic?"]
    assert trace.intermediate_answers == {"1": "Titanic"}
    assert [n for n, _ in llm.calls] == ["decompose_question", "answer_question"]
    assert (trace.llm_calls, trace.input_tokens, trace.output_tokens) == (2, 200, 20)
    assert trace.cost_usd == pytest.approx(0.002)
    # merged pool is interleaved by rank, then scored against the ORIGINAL question
    assert [t.split()[-1] for _, t in seen[-1]] == ["a", "c", "b", "d"]
    assert all(q == ORIGINAL for q, _ in seen[-1])
    assert final == ["c", "a", "d"] and trace.n_candidates == 4


def test_independent_subquestions_need_no_intermediate_calls():
    subs = ["When was X built?", "When was Y built?"]
    d, _, llm, _ = make(subs, {subs[0]: ["a"], subs[1]: ["b"]})
    final, trace = d.run("Which is older, X or Y?")
    assert [n for n, _ in llm.calls] == ["decompose_question"]
    assert trace.intermediate_answers == {} and sorted(final) == ["a", "b"]


def test_single_subquestion_matches_plain_hybrid_rerank():
    d, hybrid, _, _ = make([ORIGINAL], {ORIGINAL: ["a", "b", "c"]}, scores={"b": 1})
    assert d.search(ORIGINAL, 2) == d.reranker.search(ORIGINAL, 2) == ["b", "a"]


def test_unusable_intermediate_answer_falls_back_to_the_earlier_question():
    subs = ["Which film won Best Picture in 1998?", "Who directed {1}?"]
    assert fill_placeholders(subs, 1, {1: "<UNKNOWN>"}) == "Who directed Which film won Best Picture in 1998??"
    assert fill_placeholders(subs, 1, {}) == fill_placeholders(subs, 1, {1: ""})
    assert fill_placeholders(subs, 1, {1: "Titanic"}) == "Who directed Titanic?"


def test_bad_references_and_unusable_output():
    subs = ["Who is {2}?", "What is {1}?", "Plain question?"]
    assert needed_answers(subs) == {1}  # {2} in question 1 points forward and is ignored
    assert fill_placeholders(subs, 0, {}) == "Who is ?"
    assert parse_sub_questions(None, "orig") == ["orig"]
    assert parse_sub_questions(["", "  "], "orig") == ["orig"]
    assert parse_sub_questions([f"q{i}" for i in range(9)], "o") == [f"q{i}" for i in range(4)]


def test_interleave():
    assert interleave([["a", "b", "c"], ["b", "d"]]) == ["a", "b", "d", "c"]
    assert interleave([]) == []


def test_include_original_adds_a_pool():
    subs = ["Part one?", "Part two?"]
    table = {subs[0]: ["a"], subs[1]: ["b"], ORIGINAL: ["c"]}
    d, hybrid, _, _ = make(subs, table)
    d.include_original = True
    final, trace = d.run(ORIGINAL)
    assert sorted(final) == ["a", "b", "c"] and ORIGINAL in hybrid.queries


def test_decomp_run_end_to_end(tmp_path, monkeypatch):
    import json

    from rag import decomp_run
    from rag.schema import Question, write_jsonl

    chunks = [Chunk(f"c{i}", "d", "s", "T", "", f"text {i}", 0, 6, {f"u{i}": 1.0}) for i in range(4)]
    qs = [Question(f"q{i}", "hotpotqa", f"Question {i}?", ["a"], [[f"u{i}"]], [f"c{i}"], "bridge", [])
          for i in range(4)]
    write_jsonl(tmp_path / "chunks.jsonl", chunks)
    write_jsonl(tmp_path / "questions_dev.jsonl", qs)

    class OneChunkHybrid:
        def candidates(self, query):
            return [f"c{query.split()[-1][:-1]}", "c3"] if query.startswith("Question") else ["c3"]

    class FakeRerank:
        hybrid = OneChunkHybrid()

        def rerank(self, query, cands, k=5):
            return list(cands)[:k]

    monkeypatch.setattr(decomp_run, "build_systems", lambda chunks, names: {"hybrid_rerank": FakeRerank()})

    class PassThroughLLM:
        def call_tool(self, system, user, tool):
            return LLMResult({"sub_questions": [user.removeprefix("Question: ")]}, 50, 5, 0.001, 0.1, False)

    out = decomp_run.run_decomposition(tmp_path, tmp_path / "decomp.json", PassThroughLLM(), workers=2)
    r = out["results"]["hybrid_rerank_decomp/dev"]
    assert r["n_scored"] == 4 and r["complete_recall"] == 1.0
    assert r["trace_totals"]["mean_llm_calls"] == 1 and r["trace_totals"]["pct_decomposed"] == 0
    saved = json.loads((tmp_path / "decomp.json").read_text())
    assert [e["qid"] for e in saved["per_question"]["hybrid_rerank_decomp/dev"]] == ["q0", "q1", "q2", "q3"]


def test_decomp_run_stops_at_budget(tmp_path, monkeypatch):
    from rag import decomp_run
    from rag.schema import Question, write_jsonl

    chunks = [Chunk("c0", "d", "s", "T", "", "text", 0, 4, {"u0": 1.0})]
    qs = [Question(f"q{i}", "hotpotqa", f"Q{i}?", ["a"], [["u0"]], ["c0"], None, []) for i in range(6)]
    write_jsonl(tmp_path / "chunks.jsonl", chunks)
    write_jsonl(tmp_path / "questions_dev.jsonl", qs)

    class H:
        def candidates(self, q):
            return ["c0"]

    class R:
        hybrid = H()

        def rerank(self, q, c, k=5):
            return list(c)[:k]

    monkeypatch.setattr(decomp_run, "build_systems", lambda chunks, names: {"hybrid_rerank": R()})

    class L:
        def call_tool(self, s, u, t):
            return LLMResult({"sub_questions": ["x"]}, 1, 1, 0.01, 0.1, False)

    out = decomp_run.run_decomposition(tmp_path, tmp_path / "o.json", L(), workers=1, max_cost=0.025)
    assert out["error"] and "cap" in out["error"]
    assert 0 < out["results"]["hybrid_rerank_decomp/dev"]["n_scored"] < 6
