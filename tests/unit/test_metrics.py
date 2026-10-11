import math

import pytest

from evals.metrics import SourceSpan, covers, mean_scores, retrieval_scores

DOC = "fastapi/tutorial/cors.md"
E1 = [SourceSpan(DOC, 100, 200)]  # one evidence item = all occurrences of its quote
E2 = [SourceSpan(DOC, 500, 600)]


def chunk(start: int, end: int, source: str = DOC) -> SourceSpan:
    return SourceSpan(source, start, end)


MISS = chunk(1000, 1400)


# --- 4B: when does a chunk contain a quote? ----------------------------------------------


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        (chunk(0, 400), True),  # whole quote inside
        (chunk(140, 400), True),  # 60 of 100 characters
        (chunk(150, 400), True),  # exactly half
        (chunk(160, 400), False),  # 40 of 100
        (chunk(300, 400), False),  # no overlap
        (chunk(0, 400, source="other.md"), False),  # same offsets, other document
    ],
)
def test_chunk_contains_a_quote_when_it_holds_at_least_half_of_it(span, expected):
    assert covers(span, E1[0]) is expected


# --- 4C: the hand-computed examples ------------------------------------------------------


def test_example_a_one_quote_found_at_rank_two():
    scores = retrieval_scores([MISS, chunk(50, 250), MISS, MISS, MISS], [E1], k=5)

    assert scores.recall == 1.0
    assert scores.hit == 1.0
    assert scores.mrr == 0.5
    assert scores.ndcg == pytest.approx(1 / math.log2(3))  # 0.631


def test_example_b_multi_hop_both_quotes_found_at_ranks_two_and_five():
    ranked = [MISS, chunk(50, 250), MISS, MISS, chunk(480, 620)]

    scores = retrieval_scores(ranked, [E1, E2], k=5)

    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert (scores.recall, scores.all_evidence, scores.mrr) == (1.0, 1.0, 0.5)
    assert scores.ndcg == pytest.approx(dcg / ideal)  # 0.624


def test_example_c_a_second_chunk_with_the_same_quote_adds_nothing():
    ranked = [chunk(50, 250), chunk(90, 300), MISS, MISS, MISS]

    scores = retrieval_scores(ranked, [E1, E2], k=5)

    assert (scores.recall, scores.all_evidence, scores.mrr) == (0.5, 0.0, 1.0)
    assert scores.ndcg == pytest.approx(1 / (1 + 1 / math.log2(3)))  # 0.613
    assert scores.ndcg <= 1.0


def test_nothing_relevant_scores_zero_everywhere():
    scores = retrieval_scores([MISS] * 5, [E1], k=5)

    assert (scores.recall, scores.hit, scores.mrr, scores.ndcg) == (0.0, 0.0, 0.0, 0.0)


def test_only_the_top_k_count():
    ranked = [MISS] * 5 + [chunk(100, 200)]

    assert retrieval_scores(ranked, [E1], k=5).hit == 0.0
    assert retrieval_scores(ranked, [E1], k=6).hit == 1.0


def test_perfect_ranking_scores_one():
    scores = retrieval_scores([chunk(100, 200), chunk(500, 600)], [E1, E2], k=5)

    assert (scores.recall, scores.mrr, scores.ndcg) == (1.0, 1.0, pytest.approx(1.0))


def test_a_quote_occurring_twice_is_found_at_either_place():
    item = [SourceSpan(DOC, 100, 200), SourceSpan(DOC, 900, 1000)]

    assert retrieval_scores([chunk(880, 1100)], [item], k=5).recall == 1.0


def test_a_question_without_evidence_cannot_be_scored():
    with pytest.raises(ValueError):
        retrieval_scores([MISS], [], k=5)


def test_means_are_taken_metric_by_metric():
    a = retrieval_scores([MISS, chunk(100, 200)], [E1], k=5)  # mrr 0.5
    b = retrieval_scores([MISS] * 5, [E1], k=5)  # mrr 0

    assert mean_scores([a, b]).mrr == 0.25
    assert mean_scores([a, b]).recall == 0.5
