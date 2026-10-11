import json
import uuid

import pytest

from app.db.repositories.chunks import ChunkHit
from app.rag.generation import (
    ABSTAIN_MESSAGE,
    JSON_REMINDER,
    BadModelOutputError,
    generate_answer,
)
from tests.fakes import FakeLLM

QUESTION = "How do I declare a cookie parameter?"


def hits(count: int = 3, *, top_score: float = 0.8) -> list[ChunkHit]:
    return [
        ChunkHit(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title=f"Doc {n}",
            chunk_index=0,
            text=f"Passage number {n}.",
            section_path=None,
            page_start=None,
            page_end=None,
            score=top_score - 0.01 * (n - 1),
            rank=n,
        )
        for n in range(1, count + 1)
    ]


def reply(answer: str, citations: list[int], abstain: bool = False) -> str:
    return json.dumps({"answer": answer, "citations": citations, "abstain": abstain})


def answer_with(llm_reply, sources=None, **kwargs):
    llm = llm_reply if isinstance(llm_reply, FakeLLM) else FakeLLM(llm_reply)
    sources = hits() if sources is None else sources
    return generate_answer(llm, QUESTION, sources, max_tokens=300, **kwargs), llm


# --- Valid answers -----------------------------------------------------------------------


def test_cited_answer_maps_markers_to_the_retrieved_chunks():
    sources = hits()

    result, llm = answer_with(reply("Use Cookie [1], like Query [3].", [1, 3]), sources)

    assert result.abstained is False
    assert result.answer == "Use Cookie [1], like Query [3]."
    assert [(c.marker, c.hit) for c in result.citations] == [(1, sources[0]), (3, sources[2])]
    assert llm.calls[0]["max_tokens"] == 300
    assert llm.calls[0]["json_schema"]["required"] == ["answer", "citations", "abstain"]


def test_repeated_and_grouped_markers_are_counted_once_in_order_of_appearance():
    result, _ = answer_with(reply("A [2][1]. B [1, 2]. C [2].", [1, 2]))

    assert [c.marker for c in result.citations] == [2, 1]


def test_reply_wrapped_in_a_markdown_code_fence_is_still_parsed():
    result, _ = answer_with("```json\n" + reply("Use Cookie [1].", [1]) + "\n```")

    assert result.answer == "Use Cookie [1]."


# --- Citations that don't check out ------------------------------------------------------


def test_citation_to_a_source_that_does_not_exist_is_dropped_and_removed_from_the_text():
    result, _ = answer_with(reply("Use Cookie [1]. Cookies expire [7].", [1, 7]))

    assert result.answer == "Use Cookie [1]. Cookies expire."
    assert [c.marker for c in result.citations] == [1]
    assert result.dropped_citations == [7]


def test_invalid_number_inside_a_group_is_removed_but_the_rest_is_kept():
    result, _ = answer_with(reply("Use Cookie [1, 9].", [1, 9]))

    assert result.answer == "Use Cookie [1]."


def test_citation_listed_but_never_shown_in_the_answer_is_dropped():
    result, _ = answer_with(reply("Use Cookie [1].", [1, 2]))

    assert [c.marker for c in result.citations] == [1]
    assert result.dropped_citations == [2]


def test_marker_shown_in_the_answer_counts_even_if_missing_from_the_list():
    result, _ = answer_with(reply("Use Cookie [2].", []))

    assert [c.marker for c in result.citations] == [2]


def test_answer_without_any_valid_citation_is_turned_into_an_abstention():
    result, _ = answer_with(reply("Cookies are declared with Cookie [8].", [8]))

    assert result.abstained is True
    assert result.abstain_reason == "no_valid_citations"
    assert result.answer == ABSTAIN_MESSAGE
    assert result.citations == []


# --- Abstention --------------------------------------------------------------------------


def test_model_abstention_returns_the_standard_message():
    result, _ = answer_with(reply("abstain", [], abstain=True))

    assert (result.abstained, result.abstain_reason, result.answer) == (
        True,
        "model",
        ABSTAIN_MESSAGE,
    )


def test_no_sources_abstains_without_calling_the_model():
    result, llm = answer_with(reply("never used", [1]), sources=[])

    assert result.abstain_reason == "no_sources"
    assert llm.calls == []


def test_weak_retrieval_abstains_without_calling_the_model():
    result, llm = answer_with(reply("never used", [1]), hits(top_score=0.4), min_score=0.5)

    assert result.abstain_reason == "low_retrieval_score"
    assert llm.calls == []


# --- Unusable output ---------------------------------------------------------------------


def test_invalid_json_is_retried_once_with_a_reminder():
    llm = FakeLLM(["Sure! Here is the answer: use Cookie.", reply("Use Cookie [1].", [1])])

    result, _ = answer_with(llm)

    assert result.answer == "Use Cookie [1]."
    assert len(llm.calls) == 2
    assert llm.calls[1]["user"].endswith(JSON_REMINDER)
    assert len(result.llm_calls) == 2


def test_invalid_json_twice_is_an_error():
    with pytest.raises(BadModelOutputError):
        answer_with(FakeLLM(["not json", "still not json"]))


@pytest.mark.parametrize(
    "bad_reply",
    [
        '{"answer": "x", "citations": "1", "abstain": false}',
        '{"answer": "x", "citations": [true], "abstain": false}',
        '{"answer": "x", "citations": [1], "abstain": "no"}',
        '{"answer": null, "citations": [], "abstain": false}',
        '["answer", 1]',
    ],
)
def test_json_in_the_wrong_shape_is_unusable(bad_reply):
    with pytest.raises(BadModelOutputError):
        answer_with(FakeLLM([bad_reply, bad_reply]))


def test_reply_cut_off_by_the_output_limit_is_an_error_without_retry():
    llm = FakeLLM('{"answer": "Use Cookie [1] and then', finish_reason="length")

    with pytest.raises(BadModelOutputError):
        answer_with(llm)
    assert len(llm.calls) == 1
