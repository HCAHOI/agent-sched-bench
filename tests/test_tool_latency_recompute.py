from __future__ import annotations

import pytest

from trace_collect.tool_latency_recompute import (
    effective_min_restore_ms,
    recompute_restore_ms,
    validate_recompute_rate,
)


def test_recompute_restore_is_linear_in_context() -> None:
    # 2000 tokens at 0.15 ms/token = 300 ms.
    assert recompute_restore_ms(2000, 0.15) == pytest.approx(300.0)
    # Doubling the context doubles the cost.
    assert recompute_restore_ms(4000, 0.15) == pytest.approx(600.0)
    assert recompute_restore_ms(0, 0.15) == 0.0


def test_effective_min_picks_cheaper_restore() -> None:
    # Short context: recompute (300) beats swap-in (940 = 0.94 * 1000).
    assert effective_min_restore_ms(940.0, 300.0) == 300.0
    # Long context: swap-in (940) beats recompute (0.15 * 70000 = 10500).
    assert effective_min_restore_ms(940.0, 10500.0) == 940.0
    # The crossover context for these costs is 940 / 0.15 ≈ 6267 tokens.
    assert effective_min_restore_ms(940.0, recompute_restore_ms(6267, 0.15)) == 940.0
    assert (
        effective_min_restore_ms(940.0, recompute_restore_ms(6266, 0.15))
        == recompute_restore_ms(6266, 0.15)
    )


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_recompute_rate_rejects_invalid(bad: float) -> None:
    with pytest.raises(ValueError, match="recompute_rate_ms_per_token"):
        validate_recompute_rate(bad)


def test_recompute_restore_rejects_negative_context() -> None:
    with pytest.raises(ValueError, match="context_length_tokens"):
        recompute_restore_ms(-1.0, 0.15)


def test_effective_min_rejects_negative_input() -> None:
    with pytest.raises(ValueError, match="recompute_restore_ms"):
        effective_min_restore_ms(940.0, -1.0)
