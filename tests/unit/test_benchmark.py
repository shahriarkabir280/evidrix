import pytest
from pydantic import ValidationError

from evals.benchmark import CORPUS_DIR, Question, check, find_quote, load_questions

# --- Quote matching ----------------------------------------------------------------------


def test_quote_is_found_despite_different_line_breaks_and_spaces():
    text = "Intro.\nThe default level is\nWARNING,   which means only\tevents above."

    (span,) = find_quote(text, "The default level is WARNING, which means only events")

    assert text[span.char_start : span.char_end] == (
        "The default level is\nWARNING,   which means only\tevents"
    )


def test_every_occurrence_is_returned():
    spans = find_quote("use Cookie here, then use Cookie there", "use Cookie")

    assert [(s.char_start, s.char_end) for s in spans] == [(0, 10), (22, 32)]


def test_missing_quote_is_not_found():
    assert find_quote("some text", "other words entirely") == []


def test_matching_is_case_sensitive():
    assert find_quote("Use HTTPException", "use httpexception") == []


# --- Label consistency -------------------------------------------------------------------

EVIDENCE = [{"source": "fastapi/tutorial/cors.md", "quote": "An origin is the combination"}]


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "factual", "answerable": False, "evidence": [], "expected_answer": None},
        {"type": "factual", "answerable": True, "evidence": [], "expected_answer": "x"},
        {
            "type": "unanswerable",
            "answerable": False,
            "evidence": EVIDENCE,
            "expected_answer": None,
        },
        {"type": "unanswerable", "answerable": False, "evidence": [], "expected_answer": "x"},
    ],
)
def test_inconsistent_labels_are_rejected(fields):
    with pytest.raises(ValidationError):
        Question(id="q", split="dev", question="?", **fields)


# --- The benchmark file itself -----------------------------------------------------------


def test_benchmark_has_the_planned_shape():
    questions = load_questions()

    assert len(questions) == 60
    assert len(load_questions(split="dev")) == 36
    assert len(load_questions(split="test")) == 24
    counts = {t: sum(q.type == t for q in questions) for t in {q.type for q in questions}}
    assert counts == {
        "factual": 20,
        "identifier": 10,
        "multi_hop": 10,
        "distractor": 8,
        "unanswerable": 12,
    }


@pytest.mark.skipif(not CORPUS_DIR.is_dir(), reason="corpus not downloaded (make corpus)")
def test_every_evidence_quote_is_found_in_its_source():
    assert check(load_questions()) == []
