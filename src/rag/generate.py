"""Answer a question from retrieved chunks, naming the chunks the answer relies on.

The same model, prompt and settings are used for every system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .llm import LLM, LLMResult
from .schema import Chunk

SYSTEM_PROMPT = """You answer questions using only the numbered passages provided.

Rules:
- Reply with only the answer itself: the name, number, date, place or short phrase the question \
asks for. No full sentence, no explanation, no qualifiers, and no parenthetical remarks.
- Copy the wording used in the passages for that answer where you can.
- For a yes/no question, answer exactly "Yes" or "No".
- Always commit to your single best answer. Never say the passages do not contain the answer and \
never write "unknown"; if they are unclear, give the most likely answer from the closest passage.
- Always cite at least one passage: put the chunk IDs of every passage your answer relies on, or \
the closest one if you are unsure, in the citations field, copied exactly as written. Do not put \
citations in the answer text.
- Reply only by calling the answer_question tool."""


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    passages = "\n\n".join(
        f"[chunk_id: {c.chunk_id}]\n[document: {c.doc_title}]\n{c.text}" for c in chunks)
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


def answer_tool(chunk_ids: Sequence[str]) -> dict:
    return {
        "name": "answer_question",
        "description": "Submit the final short answer and the chunk IDs that support it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The short, direct answer."},
                "citations": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(chunk_ids)},
                    "minItems": 1,
                    "description": "chunk_ids of the passages the answer relies on; at least one.",
                },
            },
            "required": ["answer", "citations"],
        },
    }


@dataclass
class Generation:
    answer: str
    citations: list[str]  # valid, deduplicated chunk_ids from the retrieved set
    invalid_citations: list[str]  # anything else the model returned
    llm: LLMResult


def generate_answer(llm: LLM, question: str, chunks: Sequence[Chunk]) -> Generation:
    ids = [c.chunk_id for c in chunks]
    res = llm.call_tool(SYSTEM_PROMPT, build_user_prompt(question, chunks), answer_tool(ids))
    cites = [str(c) for c in res.output.get("citations") or []]
    valid = list(dict.fromkeys(c for c in cites if c in ids))
    return Generation(
        answer=str(res.output.get("answer", "")).strip(),
        citations=valid,
        invalid_citations=[c for c in cites if c not in ids],
        llm=res,
    )
