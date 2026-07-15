from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.restore_cost_analysis import (
    render_recompute_restore_markdown,
    run_recompute_restore_sweep,
)
from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap


def _write_trace(path: Path, agent_id: str, prompt_tokens: int, action_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "trace_metadata", "trace_format_version": 5, "instance_id": agent_id},
        {
            "type": "action",
            "action_type": "llm_call",
            "agent_id": agent_id,
            "iteration": 0,
            "action_id": "llm_0",
            "ts_start": 0.0,
            "ts_end": 0.5,
            "data": {"prompt_tokens": prompt_tokens},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": agent_id,
            "iteration": 0,
            "action_id": action_id,
            "ts_start": 0.5,
            "ts_end": 0.6,
            "data": {"tool_name": "exec", "tool_call_id": action_id},
        },
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _decision(
    sample_id: str,
    task_id: str,
    *,
    latency_ms: float,
    gated_trigger_ms: float,
    cost_ms: float = 1000.0,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "latency_ms": latency_ms,
        "kv_cost_ms": cost_ms,
        "threshold_ms": cost_ms,
        "deadline_trigger_ms": cost_ms,
        "offline_gated_robust_trigger_ms": gated_trigger_ms,
    }


def test_recompute_restore_sweep_matches_hand_computed_deltas(tmp_path: Path) -> None:
    # task-a: short call (200 ms), gate fires early at 0, context 1000 tokens.
    # task-b: long call (1500 ms), gate fires early at 0, context 2000 tokens.
    trace_a = tmp_path / "task-a" / "trace.jsonl"
    trace_b = tmp_path / "task-b" / "trace.jsonl"
    _write_trace(trace_a, "task-a", prompt_tokens=1000, action_id="tool_0")
    _write_trace(trace_b, "task-b", prompt_tokens=2000, action_id="tool_0")
    sid_a = f"{trace_a}:task-a:0:tool_0"
    sid_b = f"{trace_b}:task-b:0:tool_0"

    cv = tmp_path / "cv"
    cv.mkdir()
    (cv / "f1_decisions.jsonl").write_text(
        json.dumps(
            _decision(sid_a, "task-a", latency_ms=200.0, gated_trigger_ms=0.0),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (cv / "f2_decisions.jsonl").write_text(
        json.dumps(
            _decision(sid_b, "task-b", latency_ms=1500.0, gated_trigger_ms=0.0),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_recompute_restore_sweep(
        tmp_path,
        restore_cost_fractions=[1.0],
        recompute_rates_ms_per_token=[0.15],
        replicates=100,
        confidence_level=0.95,
        seed=0,
    )

    assert result["context_length_stats"] == {
        "sample_count": 2,
        "min": 1000,
        "median": 2000,
        "max": 2000,
    }
    comparisons = result["by_recompute_rate"]["0.15"]["comparisons"]
    deltas = {
        name: comparison["by_restore_cost_fraction"]["1.0"]["points"]["1000.0"][
            "paired_delta_ms"
        ]
        for name, comparison in comparisons.items()
    }
    # swap restore = 1.0 * 1000 = 1000; recompute = 0.15 * 1000 = 150 for task-a,
    # 0.15 * 2000 = 300 for task-b (only charged on short fires).
    # task-a fire at 0 on a short call: exposed = 1000 - 200 = 800.
    #   min-restore utility = -800 - 150; swap utility = -800 - 1000.
    # task-b fire at 0 on a long call: hides the full 1000, no restore either way.
    assert deltas["min_restore_vs_swap_restore"] == pytest.approx(850.0)  # 850 + 0
    assert deltas["min_restore_gated_vs_deadline"] == pytest.approx(50.0)  # -950 + 1000
    assert deltas["swap_restore_gated_vs_deadline"] == pytest.approx(-800.0)  # -1800 + 1000
    # C2 - C3 must equal the isolated mechanism increment C1.
    assert deltas["min_restore_gated_vs_deadline"] - deltas[
        "swap_restore_gated_vs_deadline"
    ] == pytest.approx(deltas["min_restore_vs_swap_restore"])

    summary = render_recompute_restore_markdown(result)
    assert "## recompute rate 0.15 ms/token" in summary
    assert "### min_restore_vs_swap_restore" in summary


def test_bootstrap_per_side_restore_fields_override_scalar() -> None:
    # One short early fire; baseline pays swap restore, treatment pays recompute.
    rows = [
        {
            "sample_id": "a",
            "task_id": "task-a",
            "latency_ms": 200.0,
            "kv_cost_ms": 1000.0,
            "threshold_ms": 1000.0,
            "deadline_trigger_ms": 1000.0,
            "offline_gated_robust_trigger_ms": 0.0,
            "swap_restore_ms": 1000.0,
            "recompute_restore_ms": 150.0,
        }
    ]

    result = paired_task_cluster_bootstrap(
        rows,
        costs_ms=[1000.0],
        replicates=50,
        confidence_level=0.95,
        seed=0,
        baseline_trigger_field="offline_gated_robust_trigger_ms",
        treatment_trigger_field="offline_gated_robust_trigger_ms",
        baseline_restore_cost_ms_field="swap_restore_ms",
        treatment_restore_cost_ms_field="recompute_restore_ms",
    )

    # Same trigger, restore differs: (-800 - 150) - (-800 - 1000) = 850.
    assert result["points"]["1000.0"]["paired_delta_ms"] == pytest.approx(850.0)
    assert result["baseline_restore_cost_ms_field"] == "swap_restore_ms"
    assert result["treatment_restore_cost_ms_field"] == "recompute_restore_ms"


def test_bootstrap_rejects_missing_restore_field() -> None:
    rows = [
        {
            "sample_id": "a",
            "task_id": "task-a",
            "latency_ms": 200.0,
            "kv_cost_ms": 1000.0,
            "threshold_ms": 1000.0,
            "deadline_trigger_ms": 1000.0,
            "offline_gated_robust_trigger_ms": 0.0,
        }
    ]

    with pytest.raises(ValueError, match="numeric restore field"):
        paired_task_cluster_bootstrap(
            rows,
            costs_ms=[1000.0],
            replicates=10,
            confidence_level=0.95,
            seed=0,
            baseline_trigger_field="deadline_trigger_ms",
            treatment_trigger_field="offline_gated_robust_trigger_ms",
            treatment_restore_cost_ms_field="missing_field",
            enforce_gated_treatment=False,
        )
