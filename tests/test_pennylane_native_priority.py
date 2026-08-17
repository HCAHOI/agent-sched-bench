from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from trace_collect.openclaw_host_runtime import _canonical_json_sha256

from scripts.evaluation.evaluate_pennylane_native_priority import (
    CELL_DIRS,
    CELL_ORDER,
    EXPECTED_COUNTS,
    MAX_MODEL_LEN,
    MODEL,
    PAIRING,
    TASK_ORDER,
    aggregate_gates,
    compare_pair,
    evaluate,
    quantile,
    validate_cell,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/evaluation/evaluate_pennylane_native_priority.py"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_run_metadata(root: Path) -> None:
    _write_json(
        root / "run-metadata.json",
        {
            "schema_version": 1,
            "git_commit": "deadbeef",
            "model": MODEL,
            "task_ids": list(TASK_ORDER),
            "expected_counts": {
                key: value for key, value in EXPECTED_COUNTS.items() if key != "tasks"
            },
            "cell_order": list(CELL_ORDER),
            "container_cpu_cap": 2,
            "host_request_admission_cap": None,
            "vllm_scheduling_policy": "priority",
            "server_queueing_metric": "shadow_generation.ttft_ms",
            "paired_workload_contract": 2,
            "power_limit_w": 250.0,
            "max_model_len": MAX_MODEL_LEN,
            "source_max_context_tokens": 111_057,
            "run_root": str(root.resolve()),
            "gpu": "NVIDIA A100 80GB PCIe, 81920",
        },
    )


def _task_actions(
    task_index: int, cell_name: str
) -> tuple[list[dict], list[dict], int]:
    action_count = 122 if task_index < 4 else 121
    llm_count = 62 if task_index < 2 else 61
    exec_count = 47 if task_index < 2 else 46
    source: list[dict] = []
    replay: list[dict] = []
    priority = 1 if cell_name.startswith("priority-feedback") and task_index >= 4 else 0
    for index in range(action_count):
        action_id = f"task-{task_index}-action-{index}"
        if index < llm_count:
            messages = [{"role": "user", "content": f"task {task_index}"}]
            source_action = {
                "action_id": action_id,
                "action_type": "llm_call",
                "data": {"completion_tokens": 10, "messages_in": messages},
            }
            replay_action = {
                "type": "action",
                "action_id": action_id,
                "action_type": "llm_call",
                "data": {
                    "shadow_generation": {
                        "model": MODEL,
                        "messages_sha256": _canonical_json_sha256(messages),
                        "prompt_tokens": 5,
                        "source_action_id": action_id,
                        "source_action_index": index,
                        "requested_completion_tokens": 10,
                        "returned_completion_tokens": 10,
                        "request_priority": priority,
                        "ttft_ms": 100.0,
                    }
                },
            }
        else:
            tool_index = index - llm_count
            tool_name = "exec" if tool_index < exec_count else "read"
            tool_args = json.dumps({"command": f"echo {task_index}-{tool_index}"})
            data = {
                "tool_name": tool_name,
                "tool_call_id": f"call-{task_index}-{tool_index}",
                "tool_args": tool_args,
            }
            source_action = {
                "action_id": action_id,
                "action_type": "tool_exec",
                "data": data,
            }
            replay_action = {
                "type": "action",
                "action_id": action_id,
                "action_type": "tool_exec",
                "data": dict(data),
            }
        source.append(source_action)
        replay.append(replay_action)
    return source, replay, exec_count


def _make_cell(
    root: Path,
    cell_name: str,
    *,
    mean_jct: float = 100.0,
    makespan: float = 110.0,
    ttft: float = 100.0,
) -> Path:
    index = CELL_ORDER.index(cell_name)
    method, repetition = cell_name.rsplit("-r", 1)
    arm = "feedback" if method == "priority-feedback" else method
    priority = 1 if method == "priority-feedback" else None
    cell = root / CELL_DIRS[cell_name]
    output = cell / "output"
    cell.mkdir(parents=True)
    _write_json(
        cell / "cell-metadata.json",
        {
            "schema_version": 1,
            "cell_name": cell_name,
            "method": method,
            "repetition": int(repetition),
            "simulator_arm": arm,
            "borrower_priority": priority,
            "git_commit": "deadbeef",
            "model": MODEL,
        },
    )
    simulate = (
        f"trace_collect.cli simulate --container-cpus 2 --shadow-llm-model {MODEL} "
        f"--tool-gap-loan-arm {arm}"
    )
    if priority is not None:
        simulate += " --tool-gap-borrower-priority 1"
    (cell / "simulate.argv").write_text(simulate, encoding="utf-8")
    (cell / "vllm.argv").write_text(
        f"vllm serve {MODEL} --scheduling-policy priority "
        f"--max-model-len {MAX_MODEL_LEN}",
        encoding="utf-8",
    )
    _write_json(cell / "models.json", {"data": [{"id": MODEL}]})
    for name in ("cell-exit-code", "simulate-exit-code", "vllm-exit-code"):
        (cell / name).write_text("0\n", encoding="utf-8")
    (cell / "vllm.pid").write_text(f"{1000 + index}\n", encoding="utf-8")
    start = datetime(2026, 8, 18, tzinfo=timezone.utc) + timedelta(hours=index)
    timestamps = {
        "cell-start-utc.txt": start,
        "vllm-start-utc.txt": start + timedelta(seconds=1),
        "vllm-stop-utc.txt": start + timedelta(seconds=4),
        "cell-end-utc.txt": start + timedelta(seconds=5),
    }
    for name, value in timestamps.items():
        (cell / name).write_text(value.isoformat(), encoding="utf-8")
    for name in ("simulate.log", "vllm.log", "telemetryd.log", "resource-agentd.log"):
        (cell / name).write_text("healthy\n", encoding="utf-8")
    simulate_start = start + timedelta(seconds=2)
    simulate_end = start + timedelta(seconds=3)
    (cell / "simulate-start-utc.txt").write_text(
        simulate_start.isoformat(), encoding="utf-8"
    )
    (cell / "simulate-end-utc.txt").write_text(
        simulate_end.isoformat(), encoding="utf-8"
    )
    (cell / "gpu-telemetry.csv").write_text(
        "timestamp_utc_s,power_draw_w,temperature_gpu_c,memory_used_mib,sw_thermal,hw_thermal\n"
        f"{start.timestamp()}, 100, 40, 1000, Not Active, Not Active\n"
        f"{(start + timedelta(seconds=1)).timestamp()}, 100, 40, 1000, Not Active, Not Active\n"
        f"{(start + timedelta(seconds=2)).timestamp()}, 100, 40, 1000, Not Active, Not Active\n"
        f"{(start + timedelta(seconds=3)).timestamp()}, 100, 40, 1000, Not Active, Not Active\n"
        f"{(start + timedelta(seconds=4)).timestamp()}, 100, 40, 1000, Not Active, Not Active\n",
        encoding="utf-8",
    )

    task_rows = []
    for task_index, task_id in enumerate(TASK_ORDER):
        attempt = output / task_id / "attempt_1"
        attempt.mkdir(parents=True)
        source, replay, exec_count = _task_actions(task_index, cell_name)
        for action in replay:
            shadow = action["data"].get("shadow_generation")
            if shadow is not None:
                shadow["ttft_ms"] = ttft
        _write_json(
            attempt / "openclaw_host_replay_request.json", {"source_actions": source}
        )
        _write_jsonl(attempt / "openclaw_host_replay.jsonl", replay)
        _write_json(
            attempt / "openclaw_host_replay_status.json",
            {
                "success": True,
                "expected_actions": len(source),
                "emitted_actions": len(source),
                "action_sequence_matches": True,
                "provider_request_sequence_matches": True,
                "collection_validity": "valid",
                "formal_completeness": "complete",
                "telemetry_integrity_failed": False,
                "telemetry_errors": [],
            },
        )
        calls = []
        for action in source:
            data = action["data"]
            if action["action_type"] != "tool_exec" or data["tool_name"] != "exec":
                continue
            calls.append(
                {
                    "tool_call_id": data["tool_call_id"],
                    "eligible_for_kb": True,
                    "telemetry_quality": "ok",
                    "telemetry_status": "ok",
                    "invalid_reasons": [],
                    "clauses": [
                        {
                            "argv": ["echo", "ok"],
                            "mapping_evidence": "initial_invocation_exact",
                            "telemetry_quality": "ok",
                            "ts_start": 1.0,
                            "ts_end": 2.0,
                            "latency_ms": 1000.0,
                            "cpu_ns_cumulative": 1_000_000,
                            "peak_cpu_cores": 0.1,
                            "sampled_peak_rss_mb": 1.0,
                            "availability": {
                                "latency": "ok",
                                "cpu": "ok",
                                "memory": "ok",
                                "disk_io": "ok",
                            },
                            "disk_io": {
                                "read_bytes_total": 0,
                                "write_bytes_total": 0,
                                "read_write_bytes_total": 0,
                                "cancelled_write_bytes_total": 0,
                            },
                        }
                    ],
                }
            )
        assert len(calls) == exec_count
        _write_json(
            attempt / "resource_observations.json",
            {
                "collection_validity": "valid",
                "telemetry_quality": "ok",
                "formal_completeness": "complete",
                "cleanup": "ok",
                "mode": "resource",
                "status_model": "workload_telemetry_formal_v1",
                "calls": calls,
                "call_coverage": {
                    "total_call_count": exec_count,
                    "eligible_call_count": exec_count,
                    "withheld_call_count": 0,
                },
                "session_summary": {
                    "collector_health": "healthy",
                    "errors": [],
                    "loss_counters": {"total": 0},
                },
            },
        )
        _write_json(
            attempt / "container_startup.json",
            {
                "phases": [
                    {
                        "name": "start_task_container",
                        "cpu_controls": {"nano_cpus": 2_000_000_000},
                    }
                ]
            },
        )
        task_rows.append({"label": task_id, "ready_to_terminal_s": mean_jct})
    _write_json(
        output / "throughput_summary.json",
        {
            "tasks": task_rows,
            "completed_traces": 8,
            "failed_traces": 0,
            "action_count": 972,
            "llm_call_count": 490,
            "ready_to_all_terminal_s": makespan,
        },
    )
    return cell


def test_quantile_uses_linear_interpolation() -> None:
    assert quantile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_frozen_pairing_and_80_percent_tail_boundaries() -> None:
    assert PAIRING == (
        ("fixed-r1", "feedback-r1", "priority-feedback-r1"),
        ("fixed-r2", "feedback-r2", "priority-feedback-r2"),
    )
    comparison = compare_pair(
        {"mean_jct_s": 100, "makespan_s": 100, "p95_ttft_ms": 100, "p99_ttft_ms": 100},
        {"mean_jct_s": 80, "makespan_s": 90, "p95_ttft_ms": 110, "p99_ttft_ms": 110},
        {"mean_jct_s": 84, "makespan_s": 95, "p95_ttft_ms": 105, "p99_ttft_ms": 104.5},
    )
    assert comparison["metrics"]["priority_feedback_gain_retention"] == pytest.approx(
        0.8
    )
    assert comparison["checks"]["priority_p95_ttft_at_most_1_05x_fixed"]
    assert comparison["checks"]["priority_p99_ttft_at_most_1_05x_fixed"]
    assert comparison["pass"]

    no_feedback_gain = compare_pair(
        {
            "mean_jct_s": 100,
            "makespan_s": 100,
            "p95_ttft_ms": 100,
            "p99_ttft_ms": 100,
        },
        {
            "mean_jct_s": 100,
            "makespan_s": 100,
            "p95_ttft_ms": 100,
            "p99_ttft_ms": 100,
        },
        {
            "mean_jct_s": 90,
            "makespan_s": 90,
            "p95_ttft_ms": 100,
            "p99_ttft_ms": 99,
        },
    )
    assert no_feedback_gain["metrics"]["priority_feedback_gain_retention"] is None
    json.dumps(no_feedback_gain, allow_nan=False)


def test_geometric_mean_gate_boundary() -> None:
    rows = [
        {"pass": True, "metrics": {"priority_feedback_p99_ttft_ratio": 0.95}},
        {"pass": True, "metrics": {"priority_feedback_p99_ttft_ratio": 0.95}},
    ]
    value, gates = aggregate_gates(rows, cells_valid=True)
    assert value == pytest.approx(0.95)
    assert gates["geometric_mean_priority_feedback_p99_ratio_at_most_0_95"]
    rows[1]["metrics"]["priority_feedback_p99_ttft_ratio"] = 0.950001
    _, gates = aggregate_gates(rows, cells_valid=True)
    assert not gates["geometric_mean_priority_feedback_p99_ratio_at_most_0_95"]


def test_validate_cell_checks_priority_without_reporting_outcomes(
    tmp_path: Path,
) -> None:
    _make_run_metadata(tmp_path)
    cell = _make_cell(tmp_path, "priority-feedback-r1")
    result = validate_cell(tmp_path, "priority-feedback-r1")
    assert result["valid"] is True
    assert result["counts"] == EXPECTED_COUNTS
    assert "metrics" not in result
    assert result["health_evidence"]["oom_xid_scope"] == (
        "no OOM/XID marker in retained logs; kernel log not captured"
    )

    trace = cell / "output" / TASK_ORDER[4] / "attempt_1/openclaw_host_replay.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[0]["data"]["shadow_generation"]["request_priority"] = 0
    _write_jsonl(trace, rows)
    with pytest.raises(ValueError, match="shadow-generation identity"):
        validate_cell(tmp_path, "priority-feedback-r1")


def test_validate_cell_accepts_explicitly_withheld_nonempty_call(
    tmp_path: Path,
) -> None:
    _make_run_metadata(tmp_path)
    cell = _make_cell(tmp_path, "fixed-r1")
    task = cell / "output" / TASK_ORDER[0] / "attempt_1"
    path = task / "resource_observations.json"
    value = json.loads(path.read_text())
    call = value["calls"][0]
    call.update(
        {
            "eligible_for_kb": False,
            "telemetry_quality": "ok",
            "telemetry_status": "ok",
            "invalid_reasons": [{"kind": "target_unavailable"}],
        }
    )
    call["clauses"][0]["telemetry_quality"] = "invalid"
    value["formal_completeness"] = "partial"
    value["call_coverage"]["eligible_call_count"] -= 1
    value["call_coverage"]["withheld_call_count"] += 1
    _write_json(path, value)
    status_path = task / "openclaw_host_replay_status.json"
    status = json.loads(status_path.read_text())
    status["formal_completeness"] = "partial"
    _write_json(status_path, status)
    assert validate_cell(tmp_path, "fixed-r1")["valid"] is True


def test_validate_cell_cli_is_compact_and_validity_only(tmp_path: Path) -> None:
    _make_run_metadata(tmp_path)
    _make_cell(tmp_path, "fixed-r1")
    completed = subprocess.run(
        [
            str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
            str(SCRIPT),
            "--run-root",
            str(tmp_path),
            "--validate-cell",
            "fixed-r1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert "metrics" not in payload
    assert "decision" not in payload


@pytest.mark.parametrize(
    "failure",
    [
        "order",
        "tool_args",
        "loss",
        "cpu_cap",
        "oom",
        "exit",
        "ebpf",
        "gpu_truncated",
    ],
)
def test_validate_cell_rejects_validity_failures(tmp_path: Path, failure: str) -> None:
    _make_run_metadata(tmp_path)
    cell = _make_cell(tmp_path, "fixed-r1")
    if failure == "order":
        path = cell / "output/throughput_summary.json"
        value = json.loads(path.read_text())
        value["tasks"][0], value["tasks"][1] = value["tasks"][1], value["tasks"][0]
        _write_json(path, value)
    elif failure == "tool_args":
        path = cell / "output" / TASK_ORDER[0] / "attempt_1/openclaw_host_replay.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["data"]["tool_args"] = "changed"
        _write_jsonl(path, rows)
    elif failure == "loss":
        path = cell / "output" / TASK_ORDER[0] / "attempt_1/resource_observations.json"
        value = json.loads(path.read_text())
        value["session_summary"]["loss_counters"]["total"] = 1
        _write_json(path, value)
    elif failure == "cpu_cap":
        path = cell / "output" / TASK_ORDER[0] / "attempt_1/container_startup.json"
        value = json.loads(path.read_text())
        value["phases"][0]["cpu_controls"]["nano_cpus"] = 1_000_000_000
        _write_json(path, value)
    elif failure == "oom":
        (cell / "vllm.log").write_text("CUDA out of memory", encoding="utf-8")
    elif failure == "exit":
        (cell / "simulate-exit-code").write_text("1\n", encoding="utf-8")
    elif failure == "ebpf":
        path = cell / "output" / TASK_ORDER[0] / "attempt_1/resource_observations.json"
        value = json.loads(path.read_text())
        value["calls"][0]["clauses"] = []
        _write_json(path, value)
    else:
        start = datetime.fromisoformat(
            (cell / "cell-start-utc.txt").read_text().strip()
        ).timestamp()
        (cell / "gpu-telemetry.csv").write_text(
            "timestamp_utc_s,power_draw_w,temperature_gpu_c,memory_used_mib,sw_thermal,hw_thermal\n"
            f"{start}, 100, 40, 1000, Not Active, Not Active\n"
            f"{start + 4}, 100, 40, 1000, Not Active, Not Active\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError):
        validate_cell(tmp_path, "fixed-r1")


def test_final_evaluation_applies_both_repetitions_and_geomean(tmp_path: Path) -> None:
    _make_run_metadata(tmp_path)
    for name in CELL_ORDER:
        if name.startswith("fixed"):
            values = (100.0, 110.0, 100.0)
        elif name.startswith("priority"):
            values = (84.0, 95.0, 104.5)
        else:
            values = (80.0, 90.0, 110.0)
        _make_cell(
            tmp_path, name, mean_jct=values[0], makespan=values[1], ttft=values[2]
        )
    result = evaluate(tmp_path)
    assert result["decision"] == "GO"
    assert [row["cells"] for row in result["paired_results"]] == [
        {"fixed": fixed, "feedback": feedback, "priority_feedback": priority}
        for fixed, feedback, priority in PAIRING
    ]
    assert result["geometric_mean_priority_feedback_p99_ratio"] == pytest.approx(0.95)
    assert all(result["gates"].values())
