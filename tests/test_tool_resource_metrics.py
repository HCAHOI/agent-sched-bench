from __future__ import annotations

import pytest

from tool_resource.metrics import (
    ecdf_quantile,
    empirical_crps,
    interval_coverage,
    pinball_loss,
)


def test_inverse_ecdf_uses_type_one_quantiles() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert ecdf_quantile(values, 0.5) == 2.0
    assert ecdf_quantile(values, 0.9) == 4.0


def test_pinball_loss_matches_hand_calculation() -> None:
    assert pinball_loss(3.0, 2.0, 0.9) == pytest.approx(0.9)
    assert pinball_loss(1.0, 2.0, 0.9) == pytest.approx(0.1)


def test_empirical_crps_matches_two_member_hand_calculation() -> None:
    assert empirical_crps([0.0, 2.0], 1.0) == pytest.approx(0.5)
    assert empirical_crps([0.0, 2.0], 0.0) == pytest.approx(0.5)


def test_interval_coverage_is_inclusive() -> None:
    assert interval_coverage(
        [1.0, 2.0, 3.0],
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0],
    ) == pytest.approx(2.0 / 3.0)


def test_metric_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="quantile"):
        ecdf_quantile([1.0], 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        empirical_crps([], 1.0)
    with pytest.raises(ValueError, match="equal length"):
        interval_coverage([1.0], [0.0, 1.0], [2.0])
