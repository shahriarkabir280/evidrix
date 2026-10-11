"""Calibration: the judge must get clear-cut cases right before its verdicts are trusted.

Runs against the real judge model (make test-live).
"""

import pytest

from evals.judge import get_judge

pytestmark = pytest.mark.live

SOURCE = (
    "The default level is WARNING, which means that only events of this level and above "
    "will be tracked. Use logging.basicConfig(filename='example.log') to log to a file."
)


@pytest.fixture(scope="module")
def judge():
    return get_judge()


def test_supported_answer_is_judged_faithful(judge):
    result = judge.faithfulness(
        "The default level is WARNING [1]. You can log to a file with basicConfig [1].", [SOURCE]
    )

    assert result.score == 1.0


def test_invented_claim_is_judged_unsupported(judge):
    result = judge.faithfulness(
        "The default level is WARNING [1]. Log files are rotated every 24 hours by default [1].",
        [SOURCE],
    )

    assert result.score is not None and result.score < 1.0
    assert any(not c.supported and "24" in c.claim for c in result.claims)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "Only WARNING and more severe messages are tracked unless configured otherwise.",
            "correct",
        ),
        ("The default level is DEBUG, so every message is printed.", "incorrect"),
        ("I couldn't find this in the selected documents.", "incorrect"),
    ],
)
def test_correctness_verdicts_on_clear_cases(judge, answer, expected):
    result = judge.correctness(
        "What is the default level of Python's logging module?",
        "WARNING: only WARNING and more severe events are tracked.",
        answer,
    )

    assert result.verdict == expected, result.reason
