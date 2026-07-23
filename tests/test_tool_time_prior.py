from __future__ import annotations

import pytest

from tool_time.command import make_row_command_prefix_keys
from tool_time.prior import (
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
    validate_profile_eval_disjoint,
)


def _row(
    sample_id: str,
    task_id: str,
    latency_ms: float,
    command: str,
    *,
    source_trace: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "source_trace": source_trace or f"trace-{task_id}",
        "tool_name": "exec",
        "latency_ms": latency_ms,
        "tool_args": {"command": command},
    }


def test_hazard_recheck_prices_restore_cost() -> None:
    values = [80.0, 80.0, 150.0]
    kwargs = {"threshold_ms": 100.0, "kv_cost_ms": 100.0}

    assert hazard_recheck_ms(values, **kwargs) == 0.0
    assert hazard_recheck_ms(values, restore_cost_ms=35.0, **kwargs) == 80.0
    assert hazard_recheck_ms([], **kwargs) == 100.0
    with pytest.raises(ValueError, match="restore_cost_ms"):
        hazard_recheck_ms(values, restore_cost_ms=float("nan"), **kwargs)


def test_prior_hierarchy_selects_supported_command_prefix() -> None:
    keyer = make_row_command_prefix_keys("command", max_depth=4)
    rows = [
        _row("a", "task-a", 100.0, "pytest -q"),
        _row("b", "task-b", 200.0, "pytest -q"),
        _row("c", "task-c", 10.0, "ls"),
    ]
    prior = build_latency_prior(rows, row_group_keys=keyer)
    keys = keyer(_row("eval", "task-z", 0.0, "pytest -q"))

    hierarchy = latency_prior_hierarchy(
        prior,
        "exec",
        keys,
        min_tool_history=1,
        min_profile_tasks=2,
    )

    assert [node.source for node in hierarchy] == [
        "prior_global",
        "prior_tool",
        "prior_group",
        "prior_group",
    ]
    assert hierarchy[-1].values == [100.0, 200.0]
    assert hierarchy[-1].group_key == "exec:pytest -q"


def test_prior_hierarchy_backs_off_from_single_task_prefix() -> None:
    keyer = make_row_command_prefix_keys("command", max_depth=4)
    prior = build_latency_prior(
        [
            _row("a", "task-a", 100.0, "pytest -q"),
            _row("b", "task-b", 200.0, "ls"),
        ],
        row_group_keys=keyer,
    )
    keys = keyer(_row("eval", "task-z", 0.0, "pytest -q"))

    hierarchy = latency_prior_hierarchy(
        prior,
        "exec",
        keys,
        min_tool_history=1,
        min_profile_tasks=2,
    )

    assert hierarchy[-1].source == "prior_tool"


def test_profile_eval_tasks_and_traces_must_be_disjoint() -> None:
    prior = build_latency_prior([_row("a", "task-a", 100.0, "pytest")])
    with pytest.raises(ValueError, match="disjoint traces"):
        validate_profile_eval_disjoint(
            [_row("b", "task-z", 50.0, "ls", source_trace="trace-task-a")],
            prior=prior,
        )
    with pytest.raises(ValueError, match="disjoint logical tasks"):
        validate_profile_eval_disjoint(
            [_row("b", "task-a", 50.0, "ls", source_trace="other-trace")],
            prior=prior,
        )


def test_empty_profile_and_eval_fail_fast() -> None:
    with pytest.raises(ValueError, match="no profile rows"):
        build_latency_prior([])
    prior = build_latency_prior([_row("a", "task-a", 100.0, "pytest")])
    with pytest.raises(ValueError, match="no latency rows"):
        validate_profile_eval_disjoint([], prior=prior)
