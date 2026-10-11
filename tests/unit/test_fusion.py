import uuid

import pytest

from app.db.repositories.chunks import ChunkHit
from app.rag.retrieval import reciprocal_rank_fusion

IDS = {name: uuid.UUID(int=n) for n, name in enumerate("ABCDEFGHIJKLXY", start=1)}


def ranking(*names: str, score: float = 0.5) -> list[ChunkHit]:
    """A ranked list; the first name is rank 1."""
    return [
        ChunkHit(IDS[name], uuid.uuid4(), name, 0, name, None, None, None, score, rank)
        for rank, name in enumerate(names, start=1)
    ]


def names(hits: list[ChunkHit]) -> list[str]:
    return [h.text for h in hits]


def test_scores_match_the_formula_computed_by_hand():
    # A: dense #1 and keyword #3 -> 1/61 + 1/63
    # B: dense #2 only           -> 1/62
    # C: keyword #1 only         -> 1/61
    fused = reciprocal_rank_fusion(
        {"dense": ranking("A", "B"), "keyword": ranking("C", "D", "A")}, limit=10
    )

    scores = {h.text: h.score for h in fused}
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["B"] == pytest.approx(1 / 62)
    assert scores["C"] == pytest.approx(1 / 61)
    assert names(fused) == ["A", "C", "B", "D"]


def test_agreement_beats_a_single_list_at_equal_rank():
    fused = reciprocal_rank_fusion(
        {"dense": ranking("A", "B"), "keyword": ranking("C", "B")}, limit=10
    )

    # B is #2 in both lists (2/62 = 0.0323); A and C are #1 in one list each (1/61 = 0.0164).
    assert names(fused)[0] == "B"


def test_rrf_k_controls_whether_agreement_or_a_top_spot_wins():
    # X is #1 in dense only; Y is #10 in both lists.
    dense = ranking("X", "B", "C", "D", "E", "F", "G", "H", "I", "Y")
    keyword = ranking("J", "K", "L", "D", "E", "F", "G", "H", "I", "Y")

    flat = reciprocal_rank_fusion({"dense": dense, "keyword": keyword}, limit=20, rrf_k=60)
    steep = reciprocal_rank_fusion({"dense": dense, "keyword": keyword}, limit=20, rrf_k=1)

    position = lambda hits, name: names(hits).index(name)  # noqa: E731
    # k=60: 2/70 = 0.029 for Y beats 1/61 = 0.016 for X.  k=1: 1/2 = 0.5 for X beats 2/11 for Y.
    assert position(flat, "Y") < position(flat, "X")
    assert position(steep, "X") < position(steep, "Y")


def test_ties_are_broken_by_chunk_id_so_the_order_is_stable():
    first = reciprocal_rank_fusion({"dense": ranking("B"), "keyword": ranking("A")}, limit=10)
    second = reciprocal_rank_fusion({"keyword": ranking("A"), "dense": ranking("B")}, limit=10)

    assert names(first) == names(second) == ["A", "B"]  # same score; A has the smaller id


def test_limit_and_ranks_of_the_fused_list():
    fused = reciprocal_rank_fusion(
        {"dense": ranking("A", "B", "C"), "keyword": ranking("D", "E")}, limit=3
    )

    assert len(fused) == 3
    assert [h.rank for h in fused] == [1, 2, 3]


def test_debug_records_how_each_list_ranked_the_chunk():
    fused = reciprocal_rank_fusion(
        {"dense": ranking("A", score=0.83), "keyword": ranking("B", "A", score=0.1)}, limit=10
    )

    a = next(h for h in fused if h.text == "A")
    assert a.debug["dense_rank"] == 1 and a.debug["dense_score"] == 0.83
    assert a.debug["keyword_rank"] == 2 and a.debug["keyword_score"] == 0.1
    assert a.debug["rrf_score"] == pytest.approx(a.score)
    b = next(h for h in fused if h.text == "B")
    assert "dense_rank" not in b.debug


def test_no_results_from_any_list_gives_an_empty_result():
    assert reciprocal_rank_fusion({"dense": [], "keyword": []}, limit=5) == []
