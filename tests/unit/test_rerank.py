import uuid

import pytest

from app.db.repositories.chunks import ChunkHit
from app.rag import rerank as rerank_module
from app.rag.rerank import CrossEncoderReranker, passage_for, rerank
from tests.fakes import FakeReranker

QUESTION = "how do I run cleanup code after the response"


def hit(text: str, rank: int, *, section: str | None = None, debug=None) -> ChunkHit:
    return ChunkHit(
        uuid.uuid4(), uuid.uuid4(), "Doc", 0, text, section, None, None, 0.5, rank, debug or {}
    )


def test_hits_are_reordered_by_reranker_score_and_cut_to_top_k():
    hits = [
        hit("Response status codes are numbers.", 1),
        hit("Code after yield runs after the response is sent: cleanup code.", 2),
        hit("Cookies are parameters.", 3),
    ]

    reranked = rerank(FakeReranker(), QUESTION, hits, top_k=2)

    assert [h.text for h in reranked] == [hits[1].text, hits[0].text]
    assert [h.rank for h in reranked] == [1, 2]


def test_debug_keeps_earlier_signals_and_the_position_before_reranking():
    hits = [
        hit("Unrelated text.", 1, debug={"dense_rank": 1, "rrf_score": 0.03}),
        hit("cleanup code after the response", 2, debug={"keyword_rank": 1}),
    ]

    top = rerank(FakeReranker(), QUESTION, hits, top_k=2)[0]

    assert top.debug["keyword_rank"] == 1
    assert top.debug["pre_rerank_rank"] == 2
    assert top.debug["pre_rerank_score"] == 0.5
    assert top.debug["rerank_score"] == top.score


def test_equal_scores_keep_their_retrieval_order():
    hits = [hit("alpha", 1), hit("beta", 2), hit("gamma", 3)]  # no question words: all score 0

    assert [h.text for h in rerank(FakeReranker(), QUESTION, hits, top_k=3)] == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_nothing_to_rerank_gives_nothing():
    assert rerank(FakeReranker(), QUESTION, [], top_k=5) == []


def test_reranker_reads_the_section_path_before_the_text():
    assert passage_for(hit("Use Form.", 1, section="Form Data > Recap")) == (
        "Form Data > Recap\n\nUse Form."
    )
    assert passage_for(hit("Use Form.", 1)) == "Doc\n\nUse Form."


def test_get_reranker_is_built_once(monkeypatch):
    built = []
    monkeypatch.setattr(
        rerank_module, "CrossEncoderReranker", lambda *a, **kw: built.append(1) or object()
    )
    rerank_module.get_reranker.cache_clear()

    assert rerank_module.get_reranker() is rerank_module.get_reranker()
    assert len(built) == 1
    rerank_module.get_reranker.cache_clear()


# --- The real model ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cross_encoder():
    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


@pytest.mark.model
def test_cross_encoder_scores_the_answering_passage_highest(cross_encoder):
    passages = [
        "Background tasks run after returning a response, for example sending an email.",
        "Use a dependency with yield; the code after yield runs after the response is sent, "
        "which is where you close the database session.",
        "Declare cookie parameters with Cookie.",
    ]

    scores = cross_encoder.score("Where do I close the database session after a request?", passages)

    assert scores.index(max(scores)) == 1
    assert len(scores) == 3


@pytest.mark.model
def test_cross_encoder_handles_no_passages(cross_encoder):
    assert cross_encoder.score("anything", []) == []
