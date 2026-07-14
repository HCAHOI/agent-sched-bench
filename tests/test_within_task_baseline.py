from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.restore_cost_analysis import run_within_task_baseline
from trace_collect.tool_latency_within_task import within_task_trigger_rows


def test_within_task_history_is_strictly_causal() -> None:
    rows = [
        _row("c1", ts=0.0, latency_ms=150.0),
        _row("c2", ts=1.0, latency_ms=80.0),
        _row("c3", ts=2.0, latency_ms=200.0),
    ]

    scored = {
        row["sample_id"]: row
        for row in within_task_trigger_rows(rows, kv_costs_ms=[100.0], guard_ms=0.0)
    }

    # First call has no history and waits for the deadline.
    assert scored["c1"]["within_task_source"] == "none"
    assert scored["c1"]["within_task_trigger_ms"] == 100.0
    # Second call sees only [150]: the single long sample makes k=50 hide the
    # full cost (ties resolve latest), i.e. the last-value rule.
    assert scored["c2"]["within_task_history_count"] == 1
    assert scored["c2"]["within_task_trigger_ms"] == 50.0
    # Third call sees [150, 80]: k=0 earns (100-20)/2, beating k=50 and k=80.
    assert scored["c3"]["within_task_history_count"] == 2
    assert scored["c3"]["within_task_trigger_ms"] == 0.0


def test_within_task_running_call_never_leaks_into_history() -> None:
    # 4.2% of real trace calls start before the previous call's end; a still
    # running call's latency is unobservable and must stay out of history.
    rows = [
        _row("long", ts=0.0, latency_ms=5000.0),
        _row("during", ts=1.0, latency_ms=80.0),
        _row("after", ts=6.0, latency_ms=80.0),
    ]

    scored = {
        row["sample_id"]: row
        for row in within_task_trigger_rows(rows, kv_costs_ms=[100.0], guard_ms=0.0)
    }

    assert scored["during"]["within_task_source"] == "none"
    assert scored["during"]["within_task_trigger_ms"] == 100.0
    assert scored["after"]["within_task_history_count"] == 2


def test_within_task_same_timestamp_calls_cannot_see_each_other() -> None:
    rows = [
        _row("t1", ts=5.0, latency_ms=150.0),
        _row("t2", ts=5.0, latency_ms=150.0),
    ]

    scored = within_task_trigger_rows(rows, kv_costs_ms=[100.0], guard_ms=0.0)

    assert all(row["within_task_source"] == "none" for row in scored)
    assert all(row["within_task_trigger_ms"] == 100.0 for row in scored)


def test_within_task_prefers_deepest_prefix_context() -> None:
    rows = [
        _row("install", ts=0.0, latency_ms=5000.0, command="pip install x"),
        _row("test-a", ts=1.0, latency_ms=10.0, command="pytest a"),
        _row("test-b", ts=2.0, latency_ms=10.0, command="pytest b"),
        _row("list", ts=3.0, latency_ms=10.0, command="ls -la"),
    ]

    scored = {
        row["sample_id"]: row
        for row in within_task_trigger_rows(
            rows,
            kv_costs_ms=[100.0],
            guard_ms=0.0,
            command_field="command",
        )
    }

    # test-b shares the "exec:pytest" prefix with test-a only, so the slow
    # install never contaminates its history; a short sample waits.
    assert scored["test-b"]["within_task_source"] == "within_group:exec:pytest"
    assert scored["test-b"]["within_task_history_count"] == 1
    assert scored["test-b"]["within_task_trigger_ms"] == 100.0
    # ls has no shared prefix and falls back to the tool level; the 5 s
    # install (ends at t=5.0) has not completed by t=3.0 and stays out.
    assert scored["list"]["within_task_source"] == "within_tool"
    assert scored["list"]["within_task_history_count"] == 2


def test_run_within_task_baseline_merges_and_scores(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirmation"
    (confirmation / "provenance").mkdir(parents=True)
    (confirmation / "provenance" / "manifest.json").write_text(
        json.dumps(
            {
                "fold_count": 1,
                "inner_folds": 2,
                "costs_ms": [100.0],
                "guard_ms": 0.0,
                "min_tool_history": 1,
                "min_profile_tasks": 1,
                "command_field": None,
                "max_prefix_depth": 4,
                "skip_leading_cd": False,
            }
        ),
        encoding="utf-8",
    )
    (confirmation / "data").mkdir()
    eval_rows = [
        _row("a1", ts=0.0, latency_ms=150.0, task_id="task-a"),
        _row("a2", ts=1.0, latency_ms=150.0, task_id="task-a"),
        _row("b1", ts=0.0, latency_ms=80.0, task_id="task-b"),
        _row("b2", ts=1.0, latency_ms=80.0, task_id="task-b"),
    ]
    (confirmation / "data" / "f1_eval.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in eval_rows),
        encoding="utf-8",
    )
    mode_b = tmp_path / "mode-b"
    (mode_b / "rho_0.0").mkdir(parents=True)
    (mode_b / "rho_0.0" / "f1_decisions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "latency_ms": row["latency_ms"],
                    "kv_cost_ms": 100.0,
                    "threshold_ms": 100.0,
                    "deadline_trigger_ms": 100.0,
                    "offline_gated_robust_trigger_ms": 100.0,
                },
                sort_keys=True,
            )
            + "\n"
            for row in eval_rows
        ),
        encoding="utf-8",
    )

    result = run_within_task_baseline(
        confirmation,
        mode_b_root=mode_b,
        output_root=tmp_path / "b1",
        restore_cost_fractions=[0.0],
        replicates=100,
        confidence_level=0.95,
        seed=0,
    )

    coverage = result["history_coverage"]["0.0"]
    assert coverage == {
        "with_history": 2,
        "without_history": 2,
        "early_within_task_triggers": 1,
    }
    # a2 fires at 50 on a long repeat (+100 vs the deadline's 0); every other
    # call either lacks history or saw a short sample and waits.
    points = {
        name: comparison["by_restore_cost_fraction"]["0.0"]["points"]["100.0"]
        for name, comparison in result["comparisons"].items()
    }
    assert points["within_task_vs_deadline"]["paired_delta_ms"] == 100.0
    assert points["gated_vs_within_task"]["paired_delta_ms"] == -100.0
    summary = (tmp_path / "b1" / "summary.md").read_text(encoding="utf-8")
    assert "within_task_vs_deadline" in summary


def test_within_task_rejects_duplicate_sample_ids() -> None:
    rows = [
        _row("dup", ts=0.0, latency_ms=10.0),
        _row("dup", ts=1.0, latency_ms=10.0),
    ]

    with pytest.raises(ValueError, match="duplicate within-task sample_id"):
        within_task_trigger_rows(rows, kv_costs_ms=[100.0], guard_ms=0.0)


def _row(
    sample_id: str,
    *,
    ts: float,
    latency_ms: float,
    task_id: str = "task-a",
    command: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "source_trace": f"trace-{task_id}",
        "task_id": task_id,
        "tool_name": "exec",
        "tool_ts_start": ts,
        "tool_ts_end": ts + latency_ms / 1000.0,
        "latency_ms": latency_ms,
    }
    if command is not None:
        row["tool_args"] = {"command": command}
    return row


__all__: list[str] = []
