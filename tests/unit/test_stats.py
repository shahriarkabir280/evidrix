import pytest

from evals.stats import bootstrap_ci, paired_bootstrap_diff


def test_constant_values_give_a_zero_width_interval():
    interval = bootstrap_ci([0.5] * 20)

    assert (interval.mean, interval.low, interval.high) == (0.5, 0.5, 0.5)


def test_interval_contains_the_mean_and_stays_within_the_data_range():
    values = [0.0, 1.0] * 15

    interval = bootstrap_ci(values)

    assert 0.0 <= interval.low <= interval.mean <= interval.high <= 1.0
    assert interval.mean == 0.5


def test_more_questions_give_a_narrower_interval():
    small = bootstrap_ci([0.0, 1.0] * 5)
    large = bootstrap_ci([0.0, 1.0] * 100)

    assert (large.high - large.low) < (small.high - small.low)


def test_same_seed_gives_the_same_interval():
    values = [0.1, 0.9, 0.4, 0.7, 0.3]

    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_identical_systems_have_a_zero_difference():
    scores = [0.2, 0.8, 1.0, 0.0, 0.5]

    diff = paired_bootstrap_diff(scores, scores)

    assert (diff.mean, diff.low, diff.high) == (0.0, 0.0, 0.0)
    assert not diff.excludes_zero()


def test_a_consistent_improvement_excludes_zero():
    baseline = [0.5] * 30
    better = [0.6] * 30

    diff = paired_bootstrap_diff(baseline, better)

    assert diff.mean == pytest.approx(0.1)
    assert diff.excludes_zero()


def test_a_noisy_small_difference_does_not_exclude_zero():
    baseline = [1.0, 0.0] * 10
    other = [0.0, 1.0] * 10  # same mean, opposite questions

    assert not paired_bootstrap_diff(baseline, other).excludes_zero()


def test_pairing_requires_the_same_questions():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0], [1.0])
