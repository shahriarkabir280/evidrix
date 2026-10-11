import json

from evals.benchmark import Question
from evals.judge import Judge
from evals.report import per_question
from evals.run import judge_answer, summarize
from tests.fakes import FakeLLM


def row(qid, qtype, answerable, *, retrieval=None, **extra):
    r = {"id": qid, "type": qtype, "answerable": answerable, "errors": [], "timings_ms": {}}
    if retrieval is not None:
        r["retrieval"] = {"recall": retrieval, "all_evidence": retrieval, "hit": retrieval,
                          "mrr": retrieval, "ndcg": retrieval}  # fmt: skip
    return r | extra


def question(answerable=True):
    return Question(
        id="q1",
        split="dev",
        type="factual" if answerable else "unanswerable",
        question="When do background tasks run?",
        answerable=answerable,
        evidence=[
            {
                "source": "fastapi/tutorial/background-tasks.md",
                "quote": "after returning a response",
            }
        ]
        if answerable
        else [],
        expected_answer="After the response." if answerable else None,
    )


def test_retrieval_means_skip_unanswerable_questions_and_split_by_type():
    rows = [
        row("a", "factual", True, retrieval=1.0, timings_ms={"embed": 10.0}),
        row("b", "identifier", True, retrieval=0.0, timings_ms={"embed": 30.0}),
        row("c", "unanswerable", False, timings_ms={"embed": 20.0}),
    ]

    summary = summarize(rows, "retrieval")

    assert summary["retrieval"]["mrr"] == 0.5
    assert summary["retrieval_by_type"]["factual"]["mrr"] == 1.0
    assert "unanswerable" not in summary["retrieval_by_type"]
    assert summary["latency_ms"]["embed"]["p50"] == 20.0


def test_answer_summary_averages_each_metric_over_the_questions_it_applies_to():
    rows = [
        row("a", "factual", True, retrieval=1.0, abstained=False, citation_precision=1.0,
            correctness={"score": 1.0}, faithfulness={"score": 0.5, "claims": []},
            usage={"prompt_tokens": 100, "completion_tokens": 10}),
        row("b", "factual", True, retrieval=1.0, abstained=True, citation_precision=None,
            correctness={"score": 0.0}, usage={"prompt_tokens": 100, "completion_tokens": 10}),
        row("c", "unanswerable", False, abstained=True,
            usage={"prompt_tokens": 100, "completion_tokens": 10}),
    ]  # fmt: skip

    answers = summarize(rows, "full")["answers"]

    assert answers["correctness"] == 0.5  # the refused answerable question counts as 0
    assert answers["faithfulness"] == 0.5  # only the answered question has claims to check
    assert answers["citation_precision"] == 1.0
    assert answers["abstention"]["recall"] == 1.0
    assert answers["abstention"]["false_abstention_rate"] == 0.5


def test_refusing_an_answerable_question_is_incorrect_without_asking_the_judge():
    llm = FakeLLM()
    r = row("q1", "factual", True, answer="I couldn't find this.", abstained=True)

    judge_answer(r, question(), ["passage"], Judge(llm))

    assert r["correctness"]["verdict"] == "incorrect"
    assert "faithfulness" not in r
    assert llm.calls == []


def test_an_answered_question_gets_both_verdicts():
    replies = [
        json.dumps({"verdict": "correct", "reason": "same fact"}),
        json.dumps({"claims": [{"claim": "after the response", "supported": True}]}),
    ]
    r = row("q1", "factual", True, answer="After the response [1].", abstained=False)

    judge_answer(r, question(), ["They run after returning a response."], Judge(FakeLLM(replies)))

    assert r["correctness"]["score"] == 1.0
    assert r["faithfulness"]["score"] == 1.0


def test_report_reads_each_metric_only_where_it_applies():
    result = {"questions": [
        row("a", "factual", True, retrieval=0.5, correctness={"score": 1.0}),
        row("b", "unanswerable", False),
    ]}  # fmt: skip

    assert per_question(result, "mrr") == {"a": 0.5}
    assert per_question(result, "correctness") == {"a": 1.0}
    assert per_question(result, "faithfulness") == {}
