"""Tiny hand-built rows in HuggingFace's HotpotQA schema."""


def hp_row(qid, question, answer, paragraphs, facts, qtype="bridge", level="hard"):
    """paragraphs: {title: [sentences]}; facts: [(title, sent_id)]."""
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": qtype,
        "level": level,
        "supporting_facts": {"title": [t for t, _ in facts], "sent_id": [i for _, i in facts]},
        "context": {"title": list(paragraphs), "sentences": [list(s) for s in paragraphs.values()]},
    }


ALPHA = [" Alpha was founded in 1990.", " It is based in Oslo."]
BETA = [" Beta is a river.", " It flows north."]
