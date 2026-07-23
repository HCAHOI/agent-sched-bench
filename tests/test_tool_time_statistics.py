from __future__ import annotations

import numpy as np
import pytest

from tool_time.statistics import (
    permutation_simultaneous_labels,
    resample_task_totals,
)


def test_task_resampling_is_reproducible_and_keeps_cost_vectors_together() -> None:
    contributions = np.asarray([[1.0, 10.0], [2.0, 20.0]])

    first = resample_task_totals(contributions, replicates=20, seed=7)
    second = resample_task_totals(contributions, replicates=20, seed=7)

    assert np.array_equal(first, second)
    assert first.shape == (20, 2)
    assert np.array_equal(first[:, 1], first[:, 0] * 10.0)


def test_sign_flip_labels_a_clean_positive_result() -> None:
    contributions = np.ones((12, 1), dtype=float)

    result = permutation_simultaneous_labels(
        contributions,
        contributions.sum(axis=0),
        confidence_level=0.95,
        draws=2_000,
        seed=0,
    )

    assert result["points"][0]["permutation_label"] == "positive"
    assert result["config"]["simultaneous_family_size"] == 1


def test_sign_flip_rejects_nonpositive_draw_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        permutation_simultaneous_labels(
            np.ones((2, 1)),
            np.ones(1),
            confidence_level=0.95,
            draws=0,
            seed=0,
        )
