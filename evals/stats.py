"""Bootstrap confidence intervals for small benchmarks.

Component 4E.

With a few dozen questions, a difference of a few points between two configurations can be chance.
Resampling the questions shows how much a mean would move with a different sample of questions.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

N_RESAMPLES = 1000
SEED = 0


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def bootstrap_ci(
    values: Sequence[float], *, level: float = 0.95, n: int = N_RESAMPLES, seed: int = SEED
) -> Interval:
    """Percentile bootstrap interval for the mean of `values`."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(n))
    return Interval(sum(values) / size, *_percentiles(means, level))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    level: float = 0.95,
    n: int = N_RESAMPLES,
    seed: int = SEED,
) -> Interval:
    """Interval for mean(b) - mean(a) when both were measured on the same questions.

    Each resample draws the same questions for both systems, so question difficulty cancels out
    and only the systems' difference varies.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci([y - x for x, y in zip(a, b, strict=True)], level=level, n=n, seed=seed)


def _percentiles(sorted_means: list[float], level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    last = len(sorted_means) - 1
    return sorted_means[round(tail * last)], sorted_means[round((1 - tail) * last)]
