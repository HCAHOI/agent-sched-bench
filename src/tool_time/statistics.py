"""Task-cluster resampling used by retained offline evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np

def permutation_simultaneous_labels(
    contributions: np.ndarray,
    observed: np.ndarray,
    *,
    confidence_level: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Paired sign-flip randomization certificate over the cost family.

    For each draw, flip each task's whole (across-cost) contribution vector by an
    independent +/-1 -- one sign vector shared across cost columns per draw, so
    the joint cross-cost structure is preserved -- and form the null cost totals.
    The one-sided randomization p-values (with the standard +1 correction so the
    test is valid at finite ``draws``) are compared against the Bonferroni
    one-sided family tail ``alpha/(2m)``. Exact under SIGN-SYMMETRY of the paired
    deltas (a sharper null than ``E[delta]=0``), hence calibrated where the
    percentile bound is not. The smallest resolvable p-value is
    ``max(2^-n, 1/(draws+1))`` -- the exact-test floor OR the Monte-Carlo floor,
    whichever binds; certification at tail ``alpha/(2m)`` is impossible below it.
    """
    if draws < 1:
        raise ValueError("permutation draws must be positive")
    task_count, cost_count = contributions.shape
    alpha = 1.0 - confidence_level
    family_tail = alpha / (2.0 * cost_count)
    rng = np.random.Generator(np.random.PCG64(seed))
    ge = np.zeros(cost_count, dtype=np.int64)
    le = np.zeros(cost_count, dtype=np.int64)
    tol = 1e-9
    batch_size = 4096
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        signs = rng.choice(
            np.array([-1.0, 1.0]), size=(stop - start, task_count)
        )
        null_totals = signs @ contributions
        ge += np.sum(null_totals >= observed - tol, axis=0)
        le += np.sum(null_totals <= observed + tol, axis=0)
    p_positive = (1.0 + ge) / (draws + 1.0)
    p_harmful = (1.0 + le) / (draws + 1.0)
    point_updates: list[dict[str, Any]] = []
    for column in range(cost_count):
        if p_positive[column] <= family_tail:
            label = "positive"
        elif p_harmful[column] <= family_tail:
            label = "harmful"
        else:
            label = "inconclusive"
        point_updates.append(
            {
                "permutation_label": label,
                "permutation_p_positive": float(p_positive[column]),
                "permutation_p_harmful": float(p_harmful[column]),
            }
        )
    return {
        "points": point_updates,
        "config": {
            "method": "paired_signflip_randomization",
            "draws": draws,
            "seed": seed,
            "bit_generator": "PCG64",
            "simultaneous_method": "bonferroni_signflip",
            "simultaneous_family_size": cost_count,
            "simultaneous_tail_probability": family_tail,
            "exact_test_floor_p_value": 2.0 ** (-task_count),
            "monte_carlo_floor_p_value": 1.0 / (draws + 1.0),
            "min_achievable_p_value": max(2.0 ** (-task_count), 1.0 / (draws + 1.0)),
        },
    }


def resample_task_totals(
    contributions: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Resample complete task contribution vectors in bounded memory."""

    task_count, cost_count = contributions.shape
    if task_count < 1 or cost_count < 1:
        raise ValueError("task contribution matrix must be non-empty")
    rng = np.random.Generator(np.random.PCG64(seed))
    probabilities = np.full(task_count, 1.0 / task_count, dtype=float)
    totals = np.empty((replicates, cost_count), dtype=float)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        multiplicities = rng.multinomial(
            task_count,
            probabilities,
            size=stop - start,
        )
        totals[start:stop] = multiplicities @ contributions
    return totals


__all__ = ["permutation_simultaneous_labels", "resample_task_totals"]
