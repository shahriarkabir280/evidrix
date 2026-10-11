import json

import pytest

from evals.judge import Judge, JudgeError, strip_citations
from evals.metrics import SourceSpan, abstention_scores, citation_precision
from tests.fakes import FakeLLM

DOC = "fastapi/tutorial/cors.md"
E1 = [SourceSpan(DOC, 100, 200)]


# --- Citation precision ------------------------------------------------------------------


def test_citation_precision_is_the_share_of_cited_chunks_holding_evidence():
    cited = [SourceSpan(DOC, 50, 250), SourceSpan(DOC, 900, 1000)]

    assert citation_precision(cited, [E1]) == 0.5


def test_no_citations_has_no_precision():
    assert citation_precision([], [E1]) is None


# --- Abstention --------------------------------------------------------------------------


def test_abstention_scores_computed_by_hand():
    # 4 unanswerable: 3 abstained (TP), 1 answered.  6 answerable: 1 abstained (FP), 5 answered.
    outcomes = [(False, True)] * 3 + [(False, False)] + [(True, True)] + [(True, False)] * 5

    scores = abstention_scores(outcomes)

    assert scores.precision == pytest.approx(3 / 4)  # TP / (TP + FP)
    assert scores.recall == pytest.approx(3 / 4)  # TP / unanswerable
    assert scores.f1 == pytest.approx(0.75)
    assert scores.false_abstention_rate == pytest.approx(1 / 6)


def test_a_system_that_never_abstains_has_no_precision_and_zero_recall():
    scores = abstention_scores([(False, False), (True, False)])

    assert (scores.precision, scores.recall, scores.f1) == (None, 0.0, None)
    assert scores.false_abstention_rate == 0.0


def test_a_system_that_always_abstains_is_caught_by_the_false_abstention_rate():
    scores = abstention_scores([(False, True), (True, True), (True, True)])

    assert scores.recall == 1.0
    assert scores.false_abstention_rate == 1.0


# --- Judge (with a scripted model) -------------------------------------------------------


def test_faithfulness_is_the_share_of_supported_claims():
    reply = {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}
    llm = FakeLLM(json.dumps(reply))

    result = Judge(llm).faithfulness(
        "Use Cookie [1]. It is fast [2].", ["source one", "source two"]
    )

    assert result.score == 0.5
    assert [c.supported for c in result.claims] == [True, False]
    sent = llm.calls[0]["user"]
    assert "[1] source one" in sent and "[2] source two" in sent
    assert "Use Cookie. It is fast." in sent  # markers stripped from the answer


def test_answer_without_claims_has_no_faithfulness_score():
    assert Judge(FakeLLM('{"claims": []}')).faithfulness("Hmm.", ["s"]).score is None


@pytest.mark.parametrize(
    ("verdict", "score"), [("correct", 1.0), ("partial", 0.5), ("incorrect", 0.0)]
)
def test_correctness_verdicts_map_to_scores(verdict, score):
    llm = FakeLLM(json.dumps({"verdict": verdict, "reason": "r"}))

    result = Judge(llm).correctness("q?", "reference", "answer [1]")

    assert (result.verdict, result.score) == (verdict, score)
    assert "Reference answer: reference" in llm.calls[0]["user"]


@pytest.mark.parametrize(
    "reply", ["not json", '{"verdict": "mostly"}', '{"claims": [{"claim": "x"}]}']
)
def test_unparseable_judge_replies_raise(reply):
    judge = Judge(FakeLLM(reply))

    with pytest.raises(JudgeError):
        judge.correctness("q", "r", "a") if "claims" not in reply else judge.faithfulness(
            "a", ["s"]
        )


def test_citation_markers_are_removed_before_judging():
    assert strip_citations("Use Cookie [1], like Query [2][3] or Path [1, 4].") == (
        "Use Cookie, like Query or Path."
    )
