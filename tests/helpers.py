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


def qa_ann(evidence, unanswerable=False, spans=(), free="", yes_no=None):
    return {"unanswerable": unanswerable, "extractive_spans": list(spans), "yes_no": yes_no,
            "free_form_answer": free, "evidence": list(evidence), "highlighted_evidence": []}


def qasper_row(pid, abstract, sections, qas, title="A Paper"):
    """sections: [(name, [paragraphs])]; qas: [(question, question_id, [annotation dicts])]."""
    return {
        "id": pid, "title": title, "abstract": abstract,
        "full_text": {"section_name": [n for n, _ in sections], "paragraphs": [p for _, p in sections]},
        "qas": {
            "question": [q for q, _, _ in qas],
            "question_id": [i for _, i, _ in qas],
            "answers": [{"answer": a, "annotation_id": [], "worker_id": []} for _, _, a in qas],
        },
    }
