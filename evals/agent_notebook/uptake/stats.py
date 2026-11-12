"""The decision rules' intervals (design-agent-uptake §5): Wilson for a rate, Newcombe
for a difference of two rates, a bootstrap for a ratio of medians."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

Z95 = 1.959963984540054


def wilson(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """The 95% Wilson score interval of *successes* in *trials*."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe(
    successes_1: int, trials_1: int, successes_2: int, trials_2: int, z: float = Z95
) -> tuple[float, float]:
    """The 95% interval of rate 1 minus rate 2, from each rate's Wilson interval
    (Newcombe's hybrid score method)."""
    p1, p2 = successes_1 / trials_1, successes_2 / trials_2
    l1, u1 = wilson(successes_1, trials_1, z)
    l2, u2 = wilson(successes_2, trials_2, z)
    d = p1 - p2
    return (
        d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def median_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """median(numerator) / median(denominator), with its 95% percentile bootstrap
    interval, each sample resampled on its own."""
    rng = random.Random(seed)
    ratios = sorted(
        statistics.median(rng.choices(numerator, k=len(numerator)))
        / statistics.median(rng.choices(denominator, k=len(denominator)))
        for _ in range(resamples)
    )
    point = statistics.median(numerator) / statistics.median(denominator)
    return (point, ratios[int(0.025 * resamples)], ratios[int(0.975 * resamples) - 1])
