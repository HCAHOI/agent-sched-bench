from __future__ import annotations
from unittest.mock import patch

import numpy as np
import pytest

import tool_time.policy as utility_clock
from tool_time.prior import LatencyPriorNode
from tool_time.policy import (
    _utility_matrix,
    _utility_sum,
    robust_utility_trigger_stats,
)


@pytest.mark.parametrize("restore_cost_ms", [0.0, 47.0])
def test_prefix_utility_sum_matches_matrix_at_boundaries(
    restore_cost_ms: float,
) -> None:
    values = [151.0, 0.0, 50.0, 49.0, 100.0, 101.0, 150.0, 51.0]
    candidates = np.asarray([0.0, 1.0, 49.0, 50.0, 99.0, 100.0])
    kwargs = {
        "threshold_ms": 100.0,
        "kv_cost_ms": 50.0,
        "restore_cost_ms": restore_cost_ms,
    }

    expected = np.sum(
        _utility_matrix(np.asarray(values), candidates, **kwargs), axis=0
    )
    actual = _utility_sum(values, candidates, **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)


def test_prefix_roundoff_is_treated_as_a_tie() -> None:
    values_by_task = {
        "t0": [705.0, 748.7],
        "t1": [458.3, 612.6, 533.0, 769.8, 694.2, 704.9],
    }
    node = LatencyPriorNode(
        values=sorted(value for values in values_by_task.values() for value in values),
        values_by_task={
            task_id: sorted(values) for task_id, values in values_by_task.items()
        },
        source="test",
        group_key=None,
    )

    result = robust_utility_trigger_stats(
        node,
        parent=None,
        threshold_ms=600.0,
        kv_cost_ms=100.0,
        restore_cost_ms=5.0,
    )

    assert result.trigger_ms == 600.0
    assert result.normalized_advantage == 0.0


def test_streaming_loo_stops_after_binding_parent_curve() -> None:
    child = LatencyPriorNode(
        values=[150.0, 160.0],
        values_by_task={"t0": [150.0], "t1": [160.0]},
        source="test",
        group_key=None,
    )
    parent = LatencyPriorNode(
        values=[10.0, 20.0],
        values_by_task={"t0": [10.0], "t1": [20.0]},
        source="test",
        group_key=None,
    )

    with patch.object(
        utility_clock, "_utility_sum", wraps=utility_clock._utility_sum
    ) as utility_sum:
        result = robust_utility_trigger_stats(
            child,
            parent=parent,
            threshold_ms=100.0,
            kv_cost_ms=100.0,
            restore_cost_ms=94.0,
        )

    assert result.trigger_ms == 100.0
    assert utility_sum.call_count == 1


def test_imbalanced_loo_scales_the_roundoff_tie_bound() -> None:
    dominant = [100.5] * 98 + [101.000000001]
    minor = [101.0]
    node = LatencyPriorNode(
        values=sorted(dominant + minor),
        values_by_task={"dominant": sorted(dominant), "minor": minor},
        source="test",
        group_key=None,
    )

    result = robust_utility_trigger_stats(
        node,
        parent=None,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
        restore_cost_ms=94.0,
    )

    assert result.trigger_ms == pytest.approx(1.000000001)


def test_single_task_fallback_still_validates_the_partition() -> None:
    node = LatencyPriorNode(
        values=[10.0, 20.0],
        values_by_task={"t0": [10.0]},
        source="test",
        group_key=None,
    )

    with pytest.raises(ValueError, match="task partition"):
        robust_utility_trigger_stats(
            node,
            parent=None,
            threshold_ms=100.0,
            kv_cost_ms=100.0,
        )
