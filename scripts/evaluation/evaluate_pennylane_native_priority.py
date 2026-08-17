#!/usr/bin/env python3
"""Validate and score the frozen PennyLane native-priority experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
from typing import Any, Mapping, Sequence

from trace_collect.openclaw_host_runtime import (
    OpenClawReplayProvider,
    _canonical_json_sha256,
)


TASK_ORDER = tuple(
    f"PennyLaneAI__pennylane-{suffix}"
    for suffix in (3182, 5835, 4366, 4251, 1405, 6062, 5623, 6939)
)
CELL_ORDER = (
    "fixed-r1",
    "feedback-r1",
    "priority-feedback-r1",
    "priority-feedback-r2",
    "feedback-r2",
    "fixed-r2",
)
PAIRING = (
    ("fixed-r1", "feedback-r1", "priority-feedback-r1"),
    ("fixed-r2", "feedback-r2", "priority-feedback-r2"),
)
EXPECTED_COUNTS = {
    "tasks": 8,
    "actions": 972,
    "llm_calls": 490,
    "tool_calls": 482,
    "shell_execs": 370,
}
MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 131_072
RESULT_SCHEMA = "pennylane-native-priority-result-v1"
CELL_DIRS = {
    name: f"cell_{index:02d}_{name}" for index, name in enumerate(CELL_ORDER, 1)
}
_OOM = re.compile(r"\b(?:cuda out of memory|out of memory|oom[_ -]kill(?:ed)?)\b", re.I)
_XID = re.compile(r"\bNVRM:\s*Xid\b|\bGPU Xid\b", re.I)
_EXIT = re.compile(r"(?:^|\n)Exit code:\s*(-?\d+)\s*(?:\n|$)")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if not 0.0 <= probability <= 1.0 or any(
        not math.isfinite(value) for value in ordered
    ):
        raise ValueError("invalid quantile input")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cell_metrics(
    jcts: Sequence[float], ttfts: Sequence[float], makespan: float
) -> dict[str, float]:
    if (
        len(jcts) != EXPECTED_COUNTS["tasks"]
        or len(ttfts) != EXPECTED_COUNTS["llm_calls"]
    ):
        raise ValueError("metric population differs from the frozen protocol")
    if (
        not math.isfinite(makespan)
        or makespan <= 0.0
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in jcts)
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in ttfts
        )
    ):
        raise ValueError("non-positive or non-finite physical metric")
    return {
        "mean_jct_s": statistics.fmean(jcts),
        "p95_jct_s": quantile(jcts, 0.95),
        "makespan_s": float(makespan),
        "p95_ttft_ms": quantile(ttfts, 0.95),
        "p99_ttft_ms": quantile(ttfts, 0.99),
    }


def compare_pair(
    fixed: Mapping[str, float],
    feedback: Mapping[str, float],
    priority: Mapping[str, float],
) -> dict[str, Any]:
    """Apply every repetition-local frozen effect and tail gate."""
    fixed_jct = float(fixed["mean_jct_s"])
    feedback_gain = fixed_jct - float(feedback["mean_jct_s"])
    priority_gain = fixed_jct - float(priority["mean_jct_s"])
    feedback_reduction = feedback_gain / fixed_jct
    priority_reduction = priority_gain / fixed_jct
    retention = priority_gain / feedback_gain if feedback_gain > 0.0 else None
    priority_feedback_p99 = float(priority["p99_ttft_ms"]) / float(
        feedback["p99_ttft_ms"]
    )
    checks = {
        "feedback_mean_jct_reduction_at_least_5pct": feedback_reduction >= 0.05,
        "feedback_makespan_strictly_lower": feedback["makespan_s"]
        < fixed["makespan_s"],
        "priority_mean_jct_reduction_at_least_5pct": priority_reduction >= 0.05,
        "priority_makespan_strictly_lower": priority["makespan_s"]
        < fixed["makespan_s"],
        "priority_retains_at_least_80pct_feedback_gain": retention is not None
        and retention >= 0.80,
        "priority_p95_ttft_at_most_1_05x_fixed": priority["p95_ttft_ms"]
        / fixed["p95_ttft_ms"]
        <= 1.05,
        "priority_p99_ttft_at_most_1_05x_fixed": priority["p99_ttft_ms"]
        / fixed["p99_ttft_ms"]
        <= 1.05,
        "priority_p99_ttft_strictly_below_feedback": priority_feedback_p99 < 1.0,
    }
    return {
        "metrics": {
            "feedback_mean_jct_reduction_pct": feedback_reduction * 100.0,
            "priority_mean_jct_reduction_pct": priority_reduction * 100.0,
            "priority_feedback_gain_retention": retention,
            "feedback_fixed_makespan_ratio": feedback["makespan_s"]
            / fixed["makespan_s"],
            "priority_fixed_makespan_ratio": priority["makespan_s"]
            / fixed["makespan_s"],
            "priority_fixed_p95_ttft_ratio": priority["p95_ttft_ms"]
            / fixed["p95_ttft_ms"],
            "priority_fixed_p99_ttft_ratio": priority["p99_ttft_ms"]
            / fixed["p99_ttft_ms"],
            "priority_feedback_p99_ttft_ratio": priority_feedback_p99,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def aggregate_gates(
    paired: Sequence[Mapping[str, Any]], *, cells_valid: bool
) -> tuple[float, dict[str, bool]]:
    """Apply the one cross-repetition p99 gate and combine frozen checks."""
    if len(paired) != 2:
        raise ValueError("the frozen protocol requires exactly two repetitions")
    ratios = [
        float(row["metrics"]["priority_feedback_p99_ttft_ratio"]) for row in paired
    ]
    geometric_mean = math.sqrt(math.prod(ratios))
    return geometric_mean, {
        "all_cells_valid": cells_valid,
        "both_repetition_local_gates_pass": all(bool(row["pass"]) for row in paired),
        "geometric_mean_priority_feedback_p99_ratio_at_most_0_95": geometric_mean
        <= 0.95,
    }


def _expected_priority(cell_name: str, task_id: str) -> int:
    if cell_name.startswith("priority-feedback") and task_id in TASK_ORDER[4:]:
        return 1
    return 0


def _action_signature(action: Mapping[str, Any]) -> tuple[Any, ...]:
    data = action.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("action lacks data")
    action_type = action.get("action_type")
    action_id = action.get("action_id")
    if (
        not isinstance(action_id, str)
        or not action_id
        or action_type
        not in {
            "llm_call",
            "tool_exec",
        }
    ):
        raise ValueError("action identity differs from the replay contract")
    base = (action_id, action_type)
    if action_type == "tool_exec":
        return base + (
            data.get("tool_name"),
            data.get("tool_call_id"),
            data.get("tool_args"),
        )
    return base + (data.get("completion_tokens"),)


def _validate_task(
    attempt: Path,
    *,
    cell_name: str,
    task_id: str,
) -> dict[str, Any]:
    status = _load(attempt / "openclaw_host_replay_status.json")
    request = _load(attempt / "openclaw_host_replay_request.json")
    source = request.get("source_actions")
    if not isinstance(source, list) or any(
        not isinstance(action, Mapping) for action in source
    ):
        raise ValueError(f"source actions missing: {task_id}")
    if len({action.get("action_id") for action in source}) != len(source):
        raise ValueError(f"source action IDs are not unique: {task_id}")
    replay = [
        row
        for row in _records(attempt / "openclaw_host_replay.jsonl")
        if row.get("type") == "action"
    ]
    if (
        len(replay) != len(source)
        or status.get("expected_actions") != len(source)
        or status.get("emitted_actions") != len(source)
    ):
        raise ValueError(f"action count mismatch: {task_id}")

    requested_tokens = returned_tokens = tool_calls = shell_execs = 0
    expected_exec_call_ids: list[str] = []
    ttfts: list[float] = []
    for index, (source_action, replay_action) in enumerate(
        zip(source, replay, strict=True)
    ):
        if _action_signature(source_action)[:2] != _action_signature(replay_action)[:2]:
            raise ValueError(f"source/replay action mismatch: {task_id}:{index}")
        source_data = source_action["data"]
        replay_data = replay_action["data"]
        if source_action.get("action_type") == "tool_exec":
            tool_calls += 1
            if _action_signature(source_action) != _action_signature(replay_action):
                raise ValueError(f"source/replay tool mismatch: {task_id}:{index}")
            tool_result = str(replay_data.get("tool_result") or "")
            exit_codes = _EXIT.findall(tool_result)
            if exit_codes and int(exit_codes[-1]) == 137 and _OOM.search(tool_result):
                raise ValueError(f"OOM terminal evidence: {task_id}:{index}")
            if source_data.get("tool_name") == "exec":
                expected_exec_call_ids.append(str(source_data["tool_call_id"]))
                shell_execs += 1
            continue
        shadow = replay_data.get("shadow_generation")
        completion_tokens = source_data.get("completion_tokens")
        expected_messages = OpenClawReplayProvider._shadow_messages(
            source_data.get("messages_in")
        )
        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
            or not isinstance(shadow, Mapping)
            or shadow.get("source_action_id") != source_action.get("action_id")
            or shadow.get("source_action_index") != index
            or shadow.get("requested_completion_tokens") != completion_tokens
            or shadow.get("returned_completion_tokens") != completion_tokens
            or shadow.get("request_priority") != _expected_priority(cell_name, task_id)
            or shadow.get("model") != MODEL
            or shadow.get("messages_sha256")
            != _canonical_json_sha256(expected_messages)
            or "admission_wait_ms" in shadow
            or "end_to_end_ttft_ms" in shadow
        ):
            raise ValueError(f"shadow-generation identity mismatch: {task_id}:{index}")
        prompt_tokens = shadow.get("prompt_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or prompt_tokens + completion_tokens > MAX_MODEL_LEN
        ):
            raise ValueError(f"invalid prompt length: {task_id}:{index}")
        ttft = shadow.get("ttft_ms")
        if isinstance(ttft, bool) or not isinstance(ttft, (int, float)) or ttft < 0:
            raise ValueError(f"invalid TTFT: {task_id}:{index}")
        ttfts.append(float(ttft))
        requested_tokens += int(shadow["requested_completion_tokens"])
        returned_tokens += int(shadow["returned_completion_tokens"])

    required_status = {
        "success": True,
        "action_sequence_matches": True,
        "provider_request_sequence_matches": True,
        "collection_validity": "valid",
        "telemetry_integrity_failed": False,
    }
    if any(status.get(key) != value for key, value in required_status.items()):
        raise ValueError(f"invalid replay status: {task_id}")
    if status.get("telemetry_errors") != []:
        raise ValueError(f"telemetry errors present: {task_id}")

    observations = _load(attempt / "resource_observations.json")
    cleanup = observations.get("cleanup")
    if isinstance(cleanup, Mapping):
        cleanup = cleanup.get("cleanup_status")
    calls = observations.get("calls")
    coverage = observations.get("call_coverage")
    session_summary = observations.get("session_summary", {})
    loss = session_summary.get("loss_counters", {})
    if (
        observations.get("collection_validity") != "valid"
        or observations.get("telemetry_quality") != "ok"
        or cleanup != "ok"
        or observations.get("mode") != "resource"
        or observations.get("status_model") != "workload_telemetry_formal_v1"
        or not isinstance(calls, list)
        or not isinstance(coverage, Mapping)
        or coverage.get("total_call_count") != shell_execs
        or len(calls) != shell_execs
        or coverage.get("eligible_call_count", 0)
        + coverage.get("withheld_call_count", 0)
        != shell_execs
        or observations.get("formal_completeness") not in {"complete", "partial"}
        or status.get("formal_completeness") != observations.get("formal_completeness")
        or session_summary.get("collector_health") != "healthy"
        or session_summary.get("errors") != []
        or loss.get("total") != 0
        or any(not isinstance(call, Mapping) for call in calls)
    ):
        raise ValueError(f"invalid eBPF evidence: {task_id}")
    if [call.get("tool_call_id") for call in calls] != expected_exec_call_ids:
        raise ValueError(f"eBPF call mapping mismatch: {task_id}")
    if (
        sum(call.get("eligible_for_kb") is True for call in calls)
        != coverage["eligible_call_count"]
        or sum(call.get("eligible_for_kb") is not True for call in calls)
        != coverage["withheld_call_count"]
    ):
        raise ValueError(f"eBPF call coverage mismatch: {task_id}")
    clause_count = sum(_validate_call(call, task_id=task_id) for call in calls)
    if clause_count <= 0:
        raise ValueError(f"eBPF clauses are empty: {task_id}")

    startup = _load(attempt / "container_startup.json")
    phases = startup.get("phases")
    start_phase = (
        next(
            (
                phase
                for phase in phases
                if isinstance(phase, Mapping)
                and phase.get("name") == "start_task_container"
            ),
            None,
        )
        if isinstance(phases, list)
        else None
    )
    if (
        not isinstance(start_phase, Mapping)
        or start_phase.get("cpu_controls", {}).get("nano_cpus") != 2_000_000_000
    ):
        raise ValueError(f"container CPU cap mismatch: {task_id}")

    return {
        "action_count": len(source),
        "llm_calls": len(ttfts),
        "tool_calls": tool_calls,
        "shell_execs": shell_execs,
        "clause_count": clause_count,
        "withheld_calls": int(coverage["withheld_call_count"]),
        "requested_completion_tokens": requested_tokens,
        "returned_completion_tokens": returned_tokens,
        "ttft_ms": ttfts,
        "source_digest": _digest(source),
    }


def _nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def _validate_call(call: Mapping[str, Any], *, task_id: str) -> int:
    clauses = call.get("clauses")
    if not isinstance(clauses, list):
        raise ValueError(f"eBPF call lacks clauses: {task_id}")
    if call.get("eligible_for_kb") is not True:
        if call.get("telemetry_status") not in {
            "ok",
            "invalid",
            "unavailable",
        } or not call.get("invalid_reasons"):
            raise ValueError(f"withheld eBPF call lacks explicit evidence: {task_id}")
        return 0
    if not clauses:
        raise ValueError(f"eligible eBPF call has no clauses: {task_id}")
    for clause in clauses:
        if not isinstance(clause, Mapping):
            raise ValueError(f"malformed eBPF clause: {task_id}")
        availability = clause.get("availability")
        start = clause.get("ts_start")
        end = clause.get("ts_end")
        disk = clause.get("disk_io")
        if (
            not isinstance(availability, Mapping)
            or not isinstance(clause.get("argv"), list)
            or not clause.get("mapping_evidence")
            or clause.get("telemetry_quality") != "ok"
            or not _nonnegative_number(start)
            or not _nonnegative_number(end)
            or end < start
            or not _nonnegative_number(clause.get("cpu_ns_cumulative"))
            or not isinstance(disk, Mapping)
        ):
            raise ValueError(f"malformed eBPF clause aggregate: {task_id}")
        required = {
            "latency": ("latency_ms",),
            "cpu": ("peak_cpu_cores",),
            "memory": ("sampled_peak_rss_mb",),
            "disk_io": (
                "read_bytes_total",
                "write_bytes_total",
                "read_write_bytes_total",
                "cancelled_write_bytes_total",
            ),
        }
        for target, fields in required.items():
            if availability.get(target) != "ok":
                continue
            source = disk if target == "disk_io" else clause
            if any(not _nonnegative_number(source.get(field)) for field in fields):
                raise ValueError(f"missing numeric {target} aggregate: {task_id}")
    return len(clauses)


def _scan_health_logs(cell: Path) -> dict[str, Any]:
    names = ("simulate.log", "vllm.log", "telemetryd.log", "resource-agentd.log")
    for name in names:
        text = (cell / name).read_text(encoding="utf-8", errors="replace")
        if _OOM.search(text):
            raise ValueError(f"OOM marker in {name}")
        if _XID.search(text):
            raise ValueError(f"GPU XID marker in {name}")
    return {
        "logs_scanned": list(names),
        "oom_xid_scope": "no OOM/XID marker in retained logs; kernel log not captured",
    }


def _validate_gpu_telemetry(
    path: Path, *, required_start: datetime, required_end: datetime
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if len(rows) < 2:
        raise ValueError("GPU telemetry has fewer than two samples")
    if any(
        row.get("sw_thermal", "Not Active") != "Not Active"
        or row.get("hw_thermal", "Not Active") != "Not Active"
        for row in rows
    ):
        raise ValueError("thermal slowdown recorded")
    timestamps = [float(row["timestamp_utc_s"]) for row in rows]
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if (
        any(not math.isfinite(value) for value in timestamps)
        or timestamps[0] > required_start.timestamp()
        or timestamps[-1] < required_end.timestamp()
        or any(gap <= 0.0 or gap > 3.0 for gap in gaps)
    ):
        raise ValueError("GPU telemetry lacks continuous required-interval coverage")
    energy_wh = 0.0
    for left, right in zip(rows, rows[1:]):
        elapsed = float(right["timestamp_utc_s"]) - float(left["timestamp_utc_s"])
        energy_wh += (
            (float(left["power_draw_w"]) + float(right["power_draw_w"]))
            * 0.5
            * elapsed
            / 3600.0
        )
    return {
        "samples": len(rows),
        "first_sample_utc_s": timestamps[0],
        "last_sample_utc_s": timestamps[-1],
        "energy_wh_trapezoid": energy_wh,
        "max_gpu_temperature_c": max(float(row["temperature_gpu_c"]) for row in rows),
        "max_memory_used_mib": max(float(row["memory_used_mib"]) for row in rows),
    }


def _parse_utc(path: Path) -> datetime:
    return datetime.fromisoformat(
        path.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
    )


def _option_value(argv: Sequence[str], option: str) -> str | None:
    indices = [index for index, value in enumerate(argv) if value == option]
    if not indices:
        return None
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"invalid {option} in argv")
    return argv[indices[0] + 1]


def _cell_contract(
    run_root: Path, cell_name: str
) -> tuple[Path, dict[str, Any], dict[str, Any], list[str], list[str]]:
    metadata = _load(run_root / "run-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("model") != MODEL
        or metadata.get("task_ids") != list(TASK_ORDER)
        or metadata.get("cell_order") != list(CELL_ORDER)
        or metadata.get("expected_counts")
        != {key: value for key, value in EXPECTED_COUNTS.items() if key != "tasks"}
        or metadata.get("container_cpu_cap") != 2
        or metadata.get("host_request_admission_cap") is not None
        or metadata.get("vllm_scheduling_policy") != "priority"
        or metadata.get("server_queueing_metric") != "shadow_generation.ttft_ms"
        or metadata.get("paired_workload_contract") != 2
        or metadata.get("power_limit_w") != 250.0
        or metadata.get("max_model_len") != MAX_MODEL_LEN
        or metadata.get("source_max_context_tokens") != 111_057
        or metadata.get("run_root") != str(run_root.resolve())
        or "A100" not in str(metadata.get("gpu"))
    ):
        raise ValueError("run metadata differs from the freeze")
    cell = run_root / CELL_DIRS[cell_name]
    cell_metadata = _load(cell / "cell-metadata.json")
    method, repetition_text = cell_name.rsplit("-r", 1)
    expected_arm = "feedback" if method == "priority-feedback" else method
    expected_priority = 1 if method == "priority-feedback" else None
    if (
        cell_metadata.get("schema_version") != 1
        or cell_metadata.get("cell_name") != cell_name
        or cell_metadata.get("method") != method
        or cell_metadata.get("repetition") != int(repetition_text)
        or cell_metadata.get("simulator_arm") != expected_arm
        or cell_metadata.get("borrower_priority") != expected_priority
        or cell_metadata.get("git_commit") != metadata.get("git_commit")
        or cell_metadata.get("model") != MODEL
    ):
        raise ValueError(f"cell metadata differs from the freeze: {cell_name}")

    simulate_argv = shlex.split((cell / "simulate.argv").read_text(encoding="utf-8"))
    vllm_argv = shlex.split((cell / "vllm.argv").read_text(encoding="utf-8"))
    if (
        _option_value(simulate_argv, "--container-cpus") != "2"
        or _option_value(simulate_argv, "--shadow-llm-model") != MODEL
        or _option_value(simulate_argv, "--tool-gap-loan-arm") != expected_arm
        or _option_value(simulate_argv, "--tool-gap-borrower-priority")
        != ("1" if expected_priority == 1 else None)
        or "--shadow-llm-max-concurrency" in simulate_argv
    ):
        raise ValueError(f"simulator argv differs from the freeze: {cell_name}")
    if (
        "serve" not in vllm_argv
        or MODEL not in vllm_argv
        or _option_value(vllm_argv, "--scheduling-policy") != "priority"
        or _option_value(vllm_argv, "--max-model-len") != str(MAX_MODEL_LEN)
    ):
        raise ValueError(f"vLLM argv differs from the freeze: {cell_name}")
    models = _load(cell / "models.json")
    if not any(
        isinstance(row, Mapping) and row.get("id") == MODEL
        for row in models.get("data", [])
    ):
        raise ValueError(f"vLLM model readiness differs from the freeze: {cell_name}")
    return cell, metadata, cell_metadata, simulate_argv, vllm_argv


def _load_cell(run_root: Path, cell_name: str, *, final: bool) -> dict[str, Any]:
    if cell_name not in CELL_ORDER:
        raise ValueError(f"unknown frozen cell: {cell_name}")
    cell, run_metadata, cell_metadata, simulate_argv, vllm_argv = _cell_contract(
        run_root, cell_name
    )
    output = cell / "output"
    summary = _load(output / "throughput_summary.json")
    task_rows = summary.get("tasks")
    if (
        not isinstance(task_rows, list)
        or tuple(row.get("label") for row in task_rows) != TASK_ORDER
        or summary.get("completed_traces") != EXPECTED_COUNTS["tasks"]
        or summary.get("failed_traces") != 0
    ):
        raise ValueError(f"task order/completion mismatch: {cell_name}")

    task_results = [
        _validate_task(
            output / task_id / "attempt_1", cell_name=cell_name, task_id=task_id
        )
        for task_id in TASK_ORDER
    ]
    counts = {
        "tasks": len(task_results),
        "actions": sum(row["action_count"] for row in task_results),
        "llm_calls": sum(row["llm_calls"] for row in task_results),
        "tool_calls": sum(row["tool_calls"] for row in task_results),
        "shell_execs": sum(row["shell_execs"] for row in task_results),
    }
    if (
        counts != EXPECTED_COUNTS
        or summary.get("action_count") != counts["actions"]
        or summary.get("llm_call_count") != counts["llm_calls"]
    ):
        raise ValueError(f"frozen population count mismatch: {cell_name}")
    exit_names = (
        ("cell-exit-code", "simulate-exit-code", "vllm-exit-code")
        if final
        else ("simulate-exit-code",)
    )
    if any(
        int((cell / name).read_text(encoding="utf-8").strip()) != 0
        for name in exit_names
    ):
        raise ValueError(f"nonzero cell exit code: {cell_name}")

    cell_start = _parse_utc(cell / "cell-start-utc.txt")
    server_start = _parse_utc(cell / "vllm-start-utc.txt")
    server_stop = _parse_utc(cell / "vllm-stop-utc.txt") if final else None
    cell_end = _parse_utc(cell / "cell-end-utc.txt") if final else None
    if not cell_start <= server_start or (
        final
        and not (
            isinstance(server_stop, datetime)
            and isinstance(cell_end, datetime)
            and server_start < server_stop <= cell_end
        )
    ):
        raise ValueError(f"invalid server lifecycle timestamps: {cell_name}")
    server_pid = int((cell / "vllm.pid").read_text(encoding="utf-8").strip())
    if server_pid <= 0:
        raise ValueError(f"invalid vLLM pid: {cell_name}")

    simulate_start = _parse_utc(cell / "simulate-start-utc.txt")
    simulate_end = _parse_utc(cell / "simulate-end-utc.txt")
    if not server_start <= simulate_start < simulate_end:
        raise ValueError(f"invalid simulation timestamps: {cell_name}")
    health = _scan_health_logs(cell)
    gpu = _validate_gpu_telemetry(
        cell / "gpu-telemetry.csv",
        required_start=server_start if final else simulate_start,
        required_end=(server_stop if final else simulate_end) - timedelta(seconds=2),
    )
    result = {
        "cell": cell_name,
        "valid": True,
        "counts": counts,
        "requested_completion_tokens": sum(
            row["requested_completion_tokens"] for row in task_results
        ),
        "returned_completion_tokens": sum(
            row["returned_completion_tokens"] for row in task_results
        ),
        "clause_count": sum(row["clause_count"] for row in task_results),
        "withheld_calls": sum(row["withheld_calls"] for row in task_results),
        "source_task_digests": {
            task_id: row["source_digest"]
            for task_id, row in zip(TASK_ORDER, task_results, strict=True)
        },
        "gpu": gpu,
        "health_evidence": {
            **health,
            "thermal": "no thermal slowdown flag in retained GPU telemetry",
        },
        "lifecycle": {
            "cell_start_utc": cell_start.isoformat(),
            "vllm_start_utc": server_start.isoformat(),
            "vllm_stop_utc": server_stop.isoformat() if server_stop else None,
            "cell_end_utc": cell_end.isoformat() if cell_end else None,
            "vllm_pid": server_pid,
            "wall_time_s": (cell_end - cell_start).total_seconds()
            if cell_end
            else None,
        },
        "config": {
            "cell_metadata": cell_metadata,
            "simulate_argv": simulate_argv,
            "vllm_argv": vllm_argv,
        },
    }
    if final:
        jcts = [float(row["ready_to_terminal_s"]) for row in task_rows]
        ttfts = [value for row in task_results for value in row["ttft_ms"]]
        result["metrics"] = _cell_metrics(
            jcts, ttfts, float(summary["ready_to_all_terminal_s"])
        )
    return result


def validate_cell(run_root: Path, cell_name: str) -> dict[str, Any]:
    """Validate one frozen cell without reporting any outcome metric."""
    cell = _load_cell(run_root, cell_name, final=False)
    return {
        "cell": cell_name,
        "valid": True,
        "counts": cell["counts"],
        "requested_completion_tokens": cell["requested_completion_tokens"],
        "returned_completion_tokens": cell["returned_completion_tokens"],
        "source_task_digests": cell["source_task_digests"],
        "gpu_telemetry_samples": cell["gpu"]["samples"],
        "health_evidence": cell["health_evidence"],
        "config": cell["config"],
    }


def evaluate(run_root: Path) -> dict[str, Any]:
    """Validate all cells, calculate paired metrics, and apply the frozen gate."""
    protocol = _load(run_root / "run-metadata.json")
    cells = [_load_cell(run_root, name, final=True) for name in CELL_ORDER]
    by_name = {cell["cell"]: cell for cell in cells}
    if len({tuple(cell["source_task_digests"].items()) for cell in cells}) != 1:
        raise ValueError("source action populations differ across cells")
    lifecycles = [cell["lifecycle"] for cell in cells]
    if len({row["vllm_start_utc"] for row in lifecycles}) != len(CELL_ORDER) or any(
        datetime.fromisoformat(right["vllm_start_utc"])
        <= datetime.fromisoformat(left["vllm_stop_utc"])
        for left, right in zip(lifecycles, lifecycles[1:])
    ):
        raise ValueError("cells lack fresh distinct vLLM lifecycles")

    paired = []
    for repetition, (fixed_name, feedback_name, priority_name) in enumerate(PAIRING, 1):
        comparison = compare_pair(
            by_name[fixed_name]["metrics"],
            by_name[feedback_name]["metrics"],
            by_name[priority_name]["metrics"],
        )
        paired.append(
            {
                "repetition": repetition,
                "cells": {
                    "fixed": fixed_name,
                    "feedback": feedback_name,
                    "priority_feedback": priority_name,
                },
                **comparison,
            }
        )
    geometric_mean, gates = aggregate_gates(
        paired, cells_valid=all(cell["valid"] for cell in cells)
    )
    return {
        "schema": RESULT_SCHEMA,
        "experiment": "pennylane-native-priority-physical-v1",
        "decision": "GO" if all(gates.values()) else "NO-GO",
        "protocol": protocol,
        "provenance": {
            "run_root": str(run_root.resolve()),
            "run_metadata_sha256": hashlib.sha256(
                (run_root / "run-metadata.json").read_bytes()
            ).hexdigest(),
        },
        "cells": cells,
        "paired_results": paired,
        "geometric_mean_priority_feedback_p99_ratio": geometric_mean,
        "gates": gates,
        "cost": {
            "cell_makespan_s_total": sum(
                cell["metrics"]["makespan_s"] for cell in cells
            ),
            "measured_cell_wall_time_s_total": sum(
                cell["lifecycle"]["wall_time_s"] for cell in cells
            ),
            "gpu_energy_wh_trapezoid_total": sum(
                cell["gpu"]["energy_wh_trapezoid"] for cell in cells
            ),
            "physical_cells": len(cells),
            "gpu_telemetry_samples": sum(cell["gpu"]["samples"] for cell in cells),
        },
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--validate-cell", choices=CELL_ORDER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.validate_cell:
        if args.output is not None:
            parser.error("--validate-cell does not accept --output")
        try:
            result = validate_cell(args.run_root, args.validate_cell)
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(
                json.dumps(
                    {"cell": args.validate_cell, "valid": False, "error": str(error)},
                    separators=(",", ":"),
                )
            )
            return 1
        print(json.dumps(result, separators=(",", ":")))
        return 0
    if args.output is None:
        parser.error("final evaluation requires --output")
    result = evaluate(args.run_root)
    expected_output = result["protocol"].get("result_path")
    if (
        expected_output is not None
        and args.output.resolve() != Path(expected_output).resolve()
    ):
        parser.error("--output differs from frozen run metadata")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"decision": result["decision"], "output": str(args.output)},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
