from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.tool_latency_dataset import write_tool_latency_jsonl
from trace_collect.tool_latency_transfer import run_transfer_evaluation


# Frozen config small enough to hand-compute. command_field None groups by tool
# only; guard 0 makes threshold == kv cost == 100.
_CONFIG = {
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
# Every profile and eval task shares this latency distribution. At threshold 100
# / kv 100 the two 80 ms calls sit in the short region and the 150 ms call sits
# in the swap band (100 < 150 < 200).
_LATENCIES = [80.0, 80.0, 150.0]


def _seconds(latency_ms: float) -> float:
    """Second-scale timestamp end for a latency, mirroring trace extraction."""
    return latency_ms / 1000.0


def _canonical_ms(latency_ms: float) -> float:
    """Latency as it survives the extractor's (ts_end - ts_start) * 1000 round trip.

    The profile rows are built directly while the eval rows are extracted from
    trace timestamps; routing both through this identical float round trip keeps
    a short eval call exactly equal to the profile-derived trigger, so firing is
    decided by the policy, not by floating-point drift at the boundary.
    """
    return (_seconds(latency_ms) - 0.0) * 1000.0


def _build_confirmation_root(tmp_path: Path) -> Path:
    root = tmp_path / "confirmation"
    (root / "provenance").mkdir(parents=True)
    (root / "provenance" / "manifest.json").write_text(
        json.dumps(_CONFIG), encoding="utf-8"
    )
    (root / "data").mkdir()
    profile_rows = [
        {
            "sample_id": f"task-{task}-{index}",
            "source_trace": f"profile-trace-{task}",
            "task_id": f"task-{task}",
            "tool_name": "exec",
            "tool_ts_start": 0.0,
            "tool_ts_end": _seconds(latency),
            "latency_ms": _canonical_ms(latency),
        }
        for task in range(8)
        for index, latency in enumerate(_LATENCIES)
    ]
    (root / "data" / "all.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in profile_rows),
        encoding="utf-8",
    )
    return root


def _write_eval_trace(trace_path: Path, *, instance_id: str) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "instance_id": instance_id,
        }
    ]
    for index, latency in enumerate(_LATENCIES):
        records.append(
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": f"{instance_id}-t{index}",
                "agent_id": "agent-a",
                "iteration": index,
                "ts_start": 0.0,
                "ts_end": _seconds(latency),
                "data": {"tool_name": "exec", "tool_call_id": f"{instance_id}-c{index}"},
            }
        )
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_eval_trace_root(tmp_path: Path, instance_ids: list[str]) -> Path:
    eval_root = tmp_path / "eval-traces"
    for instance_id in instance_ids:
        _write_eval_trace(eval_root / instance_id / "trace.jsonl", instance_id=instance_id)
    return eval_root


def test_transfer_matches_hand_computed_gated_deltas(tmp_path: Path) -> None:
    confirmation_root = _build_confirmation_root(tmp_path)
    eval_root = _build_eval_trace_root(tmp_path, ["eval-0", "eval-1"])

    result = run_transfer_evaluation(
        confirmation_root,
        output_root=tmp_path / "transfer",
        restore_cost_fractions=[0.0, 0.35],
        replicates=200,
        confidence_level=0.95,
        seed=0,
        eval_trace_root=eval_root,
    )

    assert result["mode"] == "cross_benchmark_transfer"
    assert result["profile_task_count"] == 8
    assert result["eval_task_count"] == 2
    assert result["eval_trace_count"] == 2
    assert result["eval_row_count"] == 6
    assert "previously used during method development" in result["exposure_note"]

    # rho=0: every gated trigger fires at 0; rho=0.35 the band call is only
    # worth firing at k=80, so the gate waits past the two short calls.
    triggers_by_fraction = {}
    for fraction in ("0.0", "0.35"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "transfer" / f"rho_{fraction}" /
                         "transfer_decisions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        triggers_by_fraction[fraction] = [
            row["offline_gated_robust_trigger_ms"] for row in rows
        ]
    assert all(trigger == 0.0 for trigger in triggers_by_fraction["0.0"])
    assert all(
        trigger == pytest.approx(80.0) for trigger in triggers_by_fraction["0.35"]
    )

    gated = result["comparisons"]["gated_vs_deadline"]["by_restore_cost_fraction"]
    zero_delta = gated["0.0"]["points"]["100.0"]["paired_delta_ms"]
    refit_delta = gated["0.35"]["points"]["100.0"]["paired_delta_ms"]
    # 2 eval tasks x [80, 80, 150] -> 2 long (150) and 4 short (80) calls.
    # rho=0 fires at 0: each long hides the full 100 cost over the deadline's 0,
    # each short is exposed for kv - remaining = 100 - 80 = 20.
    assert zero_delta == pytest.approx(2 * 100.0 - 4 * 20.0)
    # rho=0.35 fires at 80 on long calls only (+40 each), shorts never fire.
    assert refit_delta == pytest.approx(2 * 40.0)

    summary = (tmp_path / "transfer" / "summary.md").read_text(encoding="utf-8")
    assert "## gated_vs_deadline" in summary
    assert "Cross-benchmark transfer (E2)" in summary


def test_transfer_eval_latencies_input_matches_trace_input(tmp_path: Path) -> None:
    confirmation_root = _build_confirmation_root(tmp_path)
    eval_rows = [
        {
            "sample_id": f"eval-{task}-{index}",
            "source_trace": f"eval-trace-{task}",
            "task_id": f"eval-{task}",
            "tool_name": "exec",
            "tool_ts_start": 0.0,
            "tool_ts_end": _seconds(latency),
            "latency_ms": _canonical_ms(latency),
        }
        for task in range(2)
        for index, latency in enumerate(_LATENCIES)
    ]
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in eval_rows),
        encoding="utf-8",
    )

    result = run_transfer_evaluation(
        confirmation_root,
        output_root=tmp_path / "transfer",
        restore_cost_fractions=[0.0],
        replicates=200,
        confidence_level=0.95,
        seed=0,
        eval_latencies=eval_path,
    )

    gated = result["comparisons"]["gated_vs_deadline"]["by_restore_cost_fraction"]
    assert gated["0.0"]["points"]["100.0"]["paired_delta_ms"] == pytest.approx(
        2 * 100.0 - 4 * 20.0
    )


def test_transfer_rejects_both_or_neither_eval_input(tmp_path: Path) -> None:
    confirmation_root = _build_confirmation_root(tmp_path)
    eval_root = _build_eval_trace_root(tmp_path, ["eval-0"])
    eval_path = tmp_path / "eval.jsonl"
    write_tool_latency_jsonl([], eval_path)

    with pytest.raises(ValueError, match="exactly one"):
        run_transfer_evaluation(
            confirmation_root,
            output_root=tmp_path / "out-both",
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
            eval_trace_root=eval_root,
            eval_latencies=eval_path,
        )

    with pytest.raises(ValueError, match="exactly one"):
        run_transfer_evaluation(
            confirmation_root,
            output_root=tmp_path / "out-neither",
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_transfer_asserts_task_disjointness(tmp_path: Path) -> None:
    confirmation_root = _build_confirmation_root(tmp_path)
    # Reuse a profile task id as the eval instance id to force overlap.
    eval_root = _build_eval_trace_root(tmp_path, ["task-0", "eval-1"])

    with pytest.raises(AssertionError, match="disjoint"):
        run_transfer_evaluation(
            confirmation_root,
            output_root=tmp_path / "transfer",
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
            eval_trace_root=eval_root,
        )


def test_transfer_refuses_existing_output_root(tmp_path: Path) -> None:
    confirmation_root = _build_confirmation_root(tmp_path)
    eval_root = _build_eval_trace_root(tmp_path, ["eval-0", "eval-1"])
    output_root = tmp_path / "transfer"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="stale output"):
        run_transfer_evaluation(
            confirmation_root,
            output_root=output_root,
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
            eval_trace_root=eval_root,
        )
