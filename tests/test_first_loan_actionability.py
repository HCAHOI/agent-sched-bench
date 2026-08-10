from __future__ import annotations

import pytest

from scripts.evaluation.evaluate_first_loan_actionability import (
    ToolTimeline,
    admission_opportunity_ms,
    first_emitted_loan,
)


def test_first_emitted_loan_uses_trigger_order_and_physical_completion() -> None:
    tools = (
        ToolTimeline(0, 0, "slow-copy", 0.0, 10_000.0, 10_000.0, 500.0, 80.0, 4.0, 32_768),
        ToolTimeline(0, 1, "later-trigger", 1_000.0, 10_000.0, 12_000.0, 10.0, 5.0, 1.0, 8_192),
    )

    loan = first_emitted_loan(tools, (0.0, 0.0))

    assert loan is not None
    assert loan["tool_index"] == 0
    assert loan["decision_ms"] == 0.0
    assert loan["usable_ms"] == 500.0
    assert loan["critical_path_stall_ms"] == 80.0


def test_first_emitted_loan_ignores_tools_that_finish_before_trigger() -> None:
    tools = (
        ToolTimeline(0, 0, "short", 0.0, 100.0, 1_000.0, 10.0, 5.0, 1.0, 8_192),
        ToolTimeline(1, 0, "long", 2_000.0, 8_000.0, 12_000.0, 20.0, 7.0, 2.0, 16_384),
    )

    loan = first_emitted_loan(tools, (5_000.0, 5_000.0))

    assert loan is not None
    assert loan["turn_index"] == 1
    assert loan["decision_ms"] == 7_000.0
    assert loan["usable_ms"] == 7_020.0
    assert loan["critical_path_stall_ms"] == pytest.approx(7.0)
    assert first_emitted_loan(tools[:1], (5_000.0,)) is None


def test_task_completion_beats_an_unfinished_swap() -> None:
    loan = {"usable_ms": 2_000.0}

    assert admission_opportunity_ms(loan, 1_000.0) == 1_000.0
    assert admission_opportunity_ms(loan, 3_000.0) == 2_000.0
    assert admission_opportunity_ms(None, 1_000.0) == 1_000.0
