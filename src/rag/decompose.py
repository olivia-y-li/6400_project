"""Hybrid + Rerank + Decomposition: the full system.

An LLM splits a multi-hop question into sub-questions. A later sub-question may refer to
the answer of an earlier one as {1}, {2}, ...; that answer is filled in by answering the
earlier sub-question from its own retrieved passages. Every sub-question goes through
hybrid retrieval, the candidate pools are merged, and the merged pool is reranked against
the ORIGINAL question down to k chunks.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .generate import generate_answer
from .retrieval import HybridRetriever, RerankRetriever
from .schema import Chunk

MAX_SUBQUESTIONS = 4
_PLACEHOLDER = re.compile(r"\{(\d+)\}")

DECOMPOSE_SYSTEM = """You break a question into the search questions needed to answer it.

Rules:
- Return the shortest list of simple sub-questions (at most 4) whose answers together answer \\
the question. If the question is already simple, return it unchanged as a single sub-question.
- A sub-question may use the answer to an earlier one by writing {1} for the first answer, \\
{2} for the second, and so on. Use a placeholder only when the later question truly cannot be \\
asked without that answer.
- For a comparison of several things, ask one sub-question per thing; they need no placeholders.
- Each sub-question must be self-contained: include the names and terms needed to search for it.
- Do not answer the questions.
- Reply only by calling the decompose_question tool.

Examples:
Question: Who directed the film that won Best Picture in 1998?
Sub-questions: ["Which film won Best Picture in 1998?", "Who directed {1}?"]
Question: Which is older, the Eiffel Tower or the Statue of Liberty?
Sub-questions: ["When was the Eiffel Tower built?", "When was the Statue of Liberty built?"]
Question: What does BERT stand for?
Sub-questions: ["What does BERT stand for?"]"""


def decompose_tool() -> dict:
    return {
        "name": "decompose_question",
        "description": "Submit the ordered list of sub-questions.",
        "input_schema": {
            "type": "object",
            "properties": {"sub_questions": {
                "type": "array", "items": {"type": "string"},
                "description": "Ordered sub-questions; later ones may use {1}, {2}, ... for earlier answers.",
            }},
            "required": ["sub_questions"],
        },
    }


def parse_sub_questions(raw: Any, original: str) -> list[str]:
    """Clean the model's list; fall back to the original question if it is unusable."""
    items = raw if isinstance(raw, list) else []
    subs = [str(s).strip() for s in items if str(s).strip()][:MAX_SUBQUESTIONS]
    return subs or [original]


def valid_refs(sub_questions: Sequence[str], i: int) -> set[int]:
    """1-based numbers of earlier sub-questions that sub-question i refers to."""
    return {int(m) for m in _PLACEHOLDER.findall(sub_questions[i]) if 1 <= int(m) <= i}


def needed_answers(sub_questions: Sequence[str]) -> set[int]:
    return {j for i in range(len(sub_questions)) for j in valid_refs(sub_questions, i)}


def usable_answer(answer: str) -> bool:
    return bool(answer.strip()) and "unknown" not in answer.lower()


def fill_placeholders(sub_questions: Sequence[str], i: int, answers: dict[int, str]) -> str:
    """Sub-question i with {j} replaced by earlier answers.

    If an earlier answer is missing or unusable, the earlier sub-question's own text stands
    in for it, so the search query still carries the right terms. A placeholder that points
    forward or out of range is dropped.
    """
    def sub(m: re.Match) -> str:
        j = int(m.group(1))
        if j not in valid_refs(sub_questions, i):
            return ""
        return answers[j] if usable_answer(answers.get(j, "")) else sub_questions[j - 1]

    return " ".join(_PLACEHOLDER.sub(sub, sub_questions[i]).split())


def interleave(pools: Sequence[Sequence[str]]) -> list[str]:
    """Merge ranked pools round-robin by rank, dropping duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for rank in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if rank < len(pool) and pool[rank] not in seen:
                seen.add(pool[rank])
                out.append(pool[rank])
    return out


@dataclass
class DecompTrace:
    sub_questions: list[str] = field(default_factory=list)  # as proposed by the model
    resolved: list[str] = field(default_factory=list)  # after filling in earlier answers
    intermediate_answers: dict[str, str] = field(default_factory=dict)  # "1" -> answer
    n_candidates: int = 0  # size of the merged pool that was reranked
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    live_cost_usd: float = 0.0  # cost of calls not served from the cache
    stage_latency_s: dict[str, float] = field(default_factory=lambda: {
        "decompose": 0.0, "intermediate_answer": 0.0, "retrieval": 0.0,
        "intermediate_rerank": 0.0, "final_rerank": 0.0})

    def add_llm(self, res, stage: str) -> None:
        self.llm_calls += 1
        self.input_tokens += res.input_tokens
        self.output_tokens += res.output_tokens
        self.cost_usd += res.cost_usd
        if not res.cached:
            self.live_cost_usd += res.cost_usd
        self.stage_latency_s[stage] += res.latency_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecompositionRetriever:
    def __init__(self, hybrid: HybridRetriever, reranker: RerankRetriever,
                 chunks: Sequence[Chunk], llm, include_original: bool = False) -> None:
        self.hybrid, self.reranker, self.llm = hybrid, reranker, llm
        self.chunks = {c.chunk_id: c for c in chunks}
        self.include_original = include_original  # ablation: also pool the undecomposed question

    def run(self, question: str, k: int = 5) -> tuple[list[str], DecompTrace]:
        trace = DecompTrace()
        res = self.llm.call_tool(DECOMPOSE_SYSTEM, f"Question: {question}", decompose_tool())
        trace.add_llm(res, "decompose")
        subs = parse_sub_questions(res.output.get("sub_questions"), question)
        trace.sub_questions = subs
        needed = needed_answers(subs)

        pools: list[list[str]] = []
        answers: dict[int, str] = {}
        for i in range(len(subs)):
            resolved = fill_placeholders(subs, i, answers)
            trace.resolved.append(resolved)

            t = time.perf_counter()
            candidates = self.hybrid.candidates(resolved)
            trace.stage_latency_s["retrieval"] += time.perf_counter() - t
            pools.append(candidates)

            if i + 1 in needed:  # a later sub-question needs this answer
                t = time.perf_counter()
                top = self.reranker.rerank(resolved, candidates, 5)
                trace.stage_latency_s["intermediate_rerank"] += time.perf_counter() - t
                gen = generate_answer(self.llm, resolved, [self.chunks[c] for c in top])
                trace.add_llm(gen.llm, "intermediate_answer")
                answers[i + 1] = gen.answer
                trace.intermediate_answers[str(i + 1)] = gen.answer

        if self.include_original:
            t = time.perf_counter()
            pools.append(self.hybrid.candidates(question))
            trace.stage_latency_s["retrieval"] += time.perf_counter() - t

        merged = interleave(pools)
        trace.n_candidates = len(merged)
        t = time.perf_counter()
        final = self.reranker.rerank(question, merged, k)  # scored against the original question
        trace.stage_latency_s["final_rerank"] += time.perf_counter() - t
        return final, trace

    def search(self, question: str, k: int = 5) -> list[str]:
        return self.run(question, k)[0]
