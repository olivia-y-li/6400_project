import json
from types import SimpleNamespace

import pytest

from rag.generate import build_user_prompt, generate_answer
from rag.llm import LLM, cost_usd
from rag.qa import run_qa
from rag.qa_metrics import exact_match, f1_score, normalize_answer, score_answer
from rag.schema import Chunk, Question, write_jsonl


class FakeClient:
    """Stands in for anthropic.Anthropic; replies with a canned tool call."""

    def __init__(self, answer="Paris", citations=None, fail=False):
        self.answer, self.citations, self.fail = answer, citations, fail
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")],
                                   stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1))
        ids = kw["tools"][0]["input_schema"]["properties"]["citations"]["items"]["enum"]
        cites = ids[:1] if self.citations is None else self.citations
        block = SimpleNamespace(type="tool_use", input={"answer": self.answer, "citations": cites})
        return SimpleNamespace(content=[block], stop_reason="tool_use",
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


def chunk(cid, text="Paris is the capital of France.", title="France"):
    return Chunk(cid, "d", "s", title, "", text, 0, len(text), {})


def test_normalize_and_f1():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("the Eiffel tower", "Eiffel Tower") == 1.0
    assert f1_score("Eiffel Tower in Paris", "Eiffel Tower") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert f1_score("cat", "dog") == 0.0
    assert f1_score("", "") == 1.0


def test_hotpot_yes_no_must_match_exactly_but_qasper_does_not_special_case():
    assert score_answer("Yes, it is", ["Yes"], "hotpotqa")["f1"] == 0.0
    assert score_answer("Yes", ["Yes"], "hotpotqa") == {"em": 1.0, "f1": 1.0}
    assert score_answer("Yes it is", ["Yes"], "qasper")["f1"] == pytest.approx(0.5)


def test_best_gold_answer_wins():
    s = score_answer("blue", ["red", "blue", "green"], "qasper")
    assert s == {"em": 1.0, "f1": 1.0}


def test_cost():
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        LLM(client=FakeClient(), model="not-a-model")


def test_prompt_contains_ids_titles_and_question_only():
    p = build_user_prompt("What?", [chunk("c1", "alpha"), chunk("c2", "beta", "Other")])
    assert "[chunk_id: c1]" in p and "[document: Other]" in p and p.endswith("Question: What?")


def test_generation_validates_citations_and_caches(tmp_path):
    client = FakeClient(answer=" Paris ", citations=["c2", "bogus", "c2"])
    llm = LLM(client=client, cache_dir=tmp_path)
    chunks = [chunk("c1"), chunk("c2")]
    g = generate_answer(llm, "Capital?", chunks)
    assert g.answer == "Paris" and g.citations == ["c2"] and g.invalid_citations == ["bogus"]
    assert not g.llm.cached and g.llm.cost_usd == pytest.approx(0.00125)
    kw = client.calls[0]
    assert "temperature" not in kw and "extra_body" not in kw
    assert kw["tool_choice"] == {"type": "tool", "name": "answer_question"}
    g2 = generate_answer(llm, "Capital?", chunks)  # served from disk, no second API call
    assert g2.llm.cached and len(client.calls) == 1 and g2.answer == "Paris"
    generate_answer(llm, "Capital?", [chunk("c1"), chunk("c3")])  # different input -> new call
    assert len(client.calls) == 2


def test_missing_tool_call_raises_and_is_not_cached(tmp_path):
    llm = LLM(client=FakeClient(fail=True), cache_dir=tmp_path)
    with pytest.raises(RuntimeError):
        generate_answer(llm, "Q?", [chunk("c1")])
    assert not list(tmp_path.glob("*.json"))


def make_run(tmp_path, n=4):
    data = tmp_path / "data"
    chunks = [chunk(f"c{i}", f"text {i}", f"Doc{i}") for i in range(n)]
    qs = [Question(f"q{i}", "hotpotqa", f"Question {i}?", ["Paris"], [[f"u{i}"]], [f"c{i}"], "bridge", [])
          for i in range(n)]
    write_jsonl(data / "chunks.jsonl", chunks)
    write_jsonl(data / "questions_dev.jsonl", qs)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"per_question": {"hybrid/dev": [
        {"qid": f"q{i}", "retrieved": [f"c{i}", f"c{(i + 1) % n}"]} for i in range(n)]}}))
    return data, report


def test_run_qa_scores_and_logs(tmp_path):
    data, report = make_run(tmp_path)
    llm = LLM(client=FakeClient(answer="Paris"), cache_dir=tmp_path / "cache")
    s = run_qa(data, report, "hybrid", "dev", tmp_path / "out", llm, workers=2)
    assert (s["n_scored"], s["em"], s["f1"]) == (4, 1.0, 1.0)
    assert s["pct_with_valid_citation"] == 100 and s["pct_citing_gold_chunk"] == 100
    assert s["total_cost_usd"] == pytest.approx(4 * 0.00125)
    rows = [json.loads(l) for l in (tmp_path / "out" / "hybrid_dev.jsonl").read_text().splitlines()]
    assert rows[0]["cited_docs"] == ["Doc0"] and rows[0]["citations"] == ["c0"]
    s2 = run_qa(data, report, "hybrid", "dev", tmp_path / "out2", llm)  # all cached
    assert s2["n_cached"] == 4


def test_run_qa_stops_at_budget(tmp_path):
    data, report = make_run(tmp_path, n=6)
    llm = LLM(client=FakeClient(), cache_dir=tmp_path / "cache")
    s = run_qa(data, report, "hybrid", "dev", tmp_path / "out", llm, workers=1, max_cost=0.002)
    assert s["error"] and "cap" in s["error"] and 0 < s["n_scored"] < 6
