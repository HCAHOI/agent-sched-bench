from __future__ import annotations

import math

import pytest

from trace_collect.tool_latency_offline_probe import evaluate_offline_probe_clock
from trace_collect.tool_latency_prequential import evaluate_prequential_updates


def _row(
    sample_id: str,
    task_id: str,
    latency_ms: float,
    *,
    start: float,
    command: str = "apt-get update",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_trace": f"trace-{task_id}",
        "task_id": task_id,
        "tool_name": "exec",
        "tool_ts_start": start,
        "tool_ts_end": start + latency_ms / 1000.0,
        "latency_ms": latency_ms,
        "tool_args": {"command": command},
    }


def _profile() -> list[dict[str, object]]:
    return [
        _row("p0a", "p0", 50.0, start=0.0),
        _row("p0b", "p0", 300.0, start=1.0),
        _row("p1a", "p1", 500.0, start=0.0),
        _row("p1b", "p1", 900.0, start=1.0),
        _row("p2a", "p2", 100.0, start=0.0),
        _row("p2b", "p2", 700.0, start=1.0),
        _row("p3a", "p3", 200.0, start=0.0),
        _row("p3b", "p3", 800.0, start=1.0),
    ]


def _evaluate(
    rows: list[dict[str, object]],
    *,
    mode: str,
    runtimes: dict[str, float],
    task_order: list[str],
    selected_guard: float | None = None,
) -> dict[str, object]:
    return evaluate_prequential_updates(
        rows,
        profile_rows=_profile(),
        task_order=task_order,
        update_mode=mode,  # type: ignore[arg-type]
        update_runtime_ms=runtimes,
        kv_costs_ms=[400.0],
        guard_ms=0.0,
        selected_guard_normalized=selected_guard,
        min_tool_history=1,
        min_profile_tasks=1,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        restore_cost_fraction=0.94,
    )


def test_frozen_arm_reproduces_existing_offline_gated_robust_trigger() -> None:
    eval_rows = [
        _row("e0a", "e0", 150.0, start=0.0),
        _row("e0b", "e0", 650.0, start=1.0),
    ]
    static = evaluate_offline_probe_clock(
        eval_rows,
        profile_rows=_profile(),
        kv_costs_ms=[400.0],
        guard_ms=0.0,
        inner_folds=2,
        min_tool_history=1,
        min_profile_tasks=1,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        restore_cost_fraction=0.94,
        include_calibration_trace=True,
    )
    guard = static["robust_calibration"]["selected_guard_normalized"]
    frozen = _evaluate(
        eval_rows,
        mode="frozen",
        runtimes={},
        task_order=["e0"],
        selected_guard=guard,
    )
    expected = {
        row["sample_id"]: row["offline_gated_robust_trigger_ms"]
        for row in static["decisions"]
    }
    actual = {row["sample_id"]: row["trigger_ms"] for row in frozen["decisions"]}
    assert actual == expected
    assert frozen["final_model_version"] == 0
    trace = static["calibration_trace"]
    assert [fold["inner_fold"] for fold in trace["folds"]] == [1, 2]
    assert {row["calibration_inner_fold"] for row in trace["probe_decisions"]} == {1, 2}
    assert len(trace["probe_decisions"]) == len(_profile())


def test_task_updates_publish_only_after_task_boundary() -> None:
    rows = [
        _row("e0a", "e0", 1000.0, start=0.0),
        _row("e0b", "e0", 10.0, start=1.00005),
        _row("e0c", "e0", 10.0, start=2.0),
        _row("e1a", "e1", 10.0, start=0.0),
    ]
    runtimes = {str(row["sample_id"]): 0.10 for row in rows}
    task = _evaluate(
        rows,
        mode="task",
        runtimes=runtimes,
        task_order=["e0", "e1"],
        selected_guard=0.0,
    )

    assert {
        row["sample_id"]: row["model_version"] for row in task["decisions"]
    } == {"e0a": 0, "e0b": 0, "e0c": 0, "e1a": 3}
    assert {
        row["sample_id"]: row["trigger_ms"] for row in task["decisions"]
    } == {"e0a": 300.0, "e0b": 300.0, "e0c": 300.0, "e1a": 300.0}
    assert task["final_model_version"] == 4
    assert task["final_model_state_hash"] == (
        "d8ac57d20c583c0488899c7ec4acfe18923de9c534e812c91073b8dc7ed85bfd"
    )
    assert [row["sample_id"] for row in task["updates"]] == [
        "e0a",
        "e0b",
        "e0c",
        "e1a",
    ]
    assert all(row["published_ts"] is None for row in task["updates"])


def test_call_update_mode_is_not_supported() -> None:
    rows = [_row("e0a", "e0", 10.0, start=0.0)]
    with pytest.raises(ValueError, match="unknown update mode"):
        _evaluate(rows, mode="call", runtimes={"e0a": 0.1}, task_order=["e0"])


def test_same_start_calls_share_one_model_snapshot() -> None:
    rows = [
        _row("e0a", "e0", 0.0, start=0.0),
        _row("e0b", "e0", 0.0, start=0.0),
        _row("e0c", "e0", 10.0, start=1.0),
    ]
    runtimes = {str(row["sample_id"]): 0.01 for row in rows}
    result = _evaluate(rows, mode="task", runtimes=runtimes, task_order=["e0"])
    versions = {row["sample_id"]: row["model_version"] for row in result["decisions"]}
    hashes = {row["sample_id"]: row["model_state_hash"] for row in result["decisions"]}
    assert versions == {"e0a": 0, "e0b": 0, "e0c": 0}
    assert len(set(hashes.values())) == 1
    assert math.isfinite(result["updates"][0]["update_runtime_ms"])
