from evals.calibrate_abstention import calibrate

ANSWERABLE = {"a1": True, "a2": True, "a3": True, "u1": False, "u2": False, "u3": False}


def test_threshold_sits_in_the_middle_of_the_gap_and_blocks_no_answerable_question():
    scores = {"u1": -4.0, "u2": -3.0, "a1": -1.0, "a2": 2.0, "u3": 3.0, "a3": 5.0}

    result = calibrate(scores, ANSWERABLE)

    assert result.threshold == -2.0  # halfway between -3.0 (caught) and -1.0 (lowest answerable)
    assert result.margin == 1.0
    assert sorted(result.caught_unanswerable) == ["u1", "u2"]
    assert result.blocked_answerable == []
    assert result.separated is False  # u3 scores above answerable questions


def test_no_threshold_when_every_unanswerable_question_scores_above_an_answerable_one():
    scores = {"a1": 0.69, "u1": 0.71, "u2": 0.73, "u3": 0.79, "a2": 0.85, "a3": 0.89}

    result = calibrate(scores, ANSWERABLE)

    assert result.threshold is None
    assert result.caught_unanswerable == []


def test_fully_separated_scores_catch_every_unanswerable_question():
    scores = {"u1": 0.1, "u2": 0.2, "u3": 0.3, "a1": 0.7, "a2": 0.8, "a3": 0.9}

    result = calibrate(scores, ANSWERABLE)

    assert result.separated is True
    assert len(result.caught_unanswerable) == 3
    assert result.threshold == 0.5
