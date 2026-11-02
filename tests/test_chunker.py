import pytest

from rag.chunker import chunk_section, gold_chunk_ids, index_units
from rag.schema import Document, EvidenceUnit, Question, Section


def make_doc(text, units=()):
    sec = Section("d::sec0", "", text, [EvidenceUnit(*u) for u in units])
    return Document("d", "D", "hotpotqa", [sec]), sec


@pytest.mark.parametrize("n", [0, 1, 999, 1000, 1001, 1850, 1900, 1901, 1950, 2800, 5000])
def test_chunks_reconstruct_and_overlap(n):
    text = "".join(chr(97 + i % 26) for i in range(n))
    doc, sec = make_doc(text)
    chunks = chunk_section(doc, sec)
    if n == 0:
        assert chunks == []
        return
    for c in chunks:
        assert c.text == text[c.start:c.end]
    assert chunks[0].start == 0 and chunks[-1].end == n
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end - nxt.start == 100
        assert nxt.end > prev.end  # no chunk lies wholly inside its predecessor
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    assert covered == set(range(n))
    if n <= 1000:
        assert len(chunks) == 1


def test_start_offsets_follow_stride():
    doc, sec = make_doc("x" * 3000)
    assert [c.start for c in chunk_section(doc, sec)] == [0, 900, 1800, 2700]


def test_custom_params_and_validation():
    doc, sec = make_doc("x" * 50)
    assert [(c.start, c.end) for c in chunk_section(doc, sec, size=20, overlap=5)] == [
        (0, 20), (15, 35), (30, 50)]
    with pytest.raises(ValueError):
        chunk_section(doc, sec, size=10, overlap=10)


def test_unit_coverage_for_sentence_split_across_chunks():
    # unit spans [950, 1050): 50 chars in chunk0 [0,1000), 100 chars... chunk1 is [900,1900)
    doc, sec = make_doc("x" * 2000, [("d::s0", 0, 100), ("d::s1", 950, 1050), ("d::s2", 1500, 1600)])
    c0, c1, c2 = chunk_section(doc, sec)
    assert c0.unit_coverage == {"d::s0": 1.0, "d::s1": 0.5}
    assert c1.unit_coverage == {"d::s1": 1.0, "d::s2": 1.0}
    assert c2.unit_coverage == {}  # chunk2 = [1800, 2000)
    # a unit clipped at the boundary
    doc, sec = make_doc("x" * 2000, [("d::s0", 980, 1180)])
    c0, c1, _ = chunk_section(doc, sec)
    assert c0.unit_coverage == {"d::s0": pytest.approx(0.1)}
    assert c1.unit_coverage == {"d::s0": 1.0}


def test_gold_chunk_ids_respect_min_overlap():
    doc, sec = make_doc("x" * 2000, [("d::s0", 980, 1180)])
    chunks = chunk_section(doc, sec)
    idx = index_units(chunks)
    q = Question("q", "hotpotqa", "?", ["a"], [["d::s0"]], [], None, ["d"])
    assert gold_chunk_ids(q, idx, 0.5) == ["d::sec0::c1"]
    assert gold_chunk_ids(q, idx, 0.05) == ["d::sec0::c0", "d::sec0::c1"]
