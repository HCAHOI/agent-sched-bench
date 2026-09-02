#!/usr/bin/env python3
"""Validate and summarize one serving experiment's telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_REQUIRED_COUNTERS = {
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
}
_OPTIONAL_COUNTERS = {
    "cached_prompt_tokens": "vllm:prompt_tokens_cached_total",
    "recomputed_prompt_tokens": "vllm:prompt_tokens_recomputed_total",
}
_COUNTERS = {**_REQUIRED_COUNTERS, **_OPTIONAL_COUNTERS}
_PROM_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?P<labels>\{.*\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+[^\s]+)?$"
)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text()), name)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} from {path}: {exc}") from exc


def _prometheus_samples(path: Path) -> dict[str, dict[str, float]]:
    wanted = set(_COUNTERS.values())
    samples: dict[str, dict[str, float]] = {name: {} for name in wanted}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read Prometheus snapshot {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_SAMPLE.match(line)
        if match is None or match.group("name") not in wanted:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        if labels in samples[name]:
            raise ValueError(f"duplicate {name}{labels} in {path}:{line_number}")
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise ValueError(
                f"invalid value for {name} in {path}:{line_number}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid counter {name} in {path}:{line_number}")
        samples[name][labels] = value
    missing = sorted(name for name in _REQUIRED_COUNTERS.values() if not samples[name])
    if missing:
        raise ValueError(f"Prometheus snapshot {path} lacks: {', '.join(missing)}")
    return samples


def _counter_deltas(start_path: Path, final_path: Path) -> dict[str, int | None]:
    start = _prometheus_samples(start_path)
    final = _prometheus_samples(final_path)
    deltas: dict[str, int | None] = {}
    for output_name, metric_name in _COUNTERS.items():
        if bool(start[metric_name]) != bool(final[metric_name]):
            raise ValueError(
                f"Prometheus metric availability changed for {metric_name}"
            )
        if not start[metric_name]:
            deltas[output_name] = None
            continue
        if start[metric_name].keys() != final[metric_name].keys():
            raise ValueError(f"Prometheus series changed for {metric_name}")
        delta = 0.0
        for labels, start_value in start[metric_name].items():
            final_value = final[metric_name][labels]
            if final_value < start_value:
                raise ValueError(f"Prometheus counter reset for {metric_name}{labels}")
            delta += final_value - start_value
        rounded = round(delta)
        if not math.isclose(delta, rounded, abs_tol=1e-6):
            raise ValueError(f"Prometheus counter delta is not integral: {metric_name}")
        deltas[output_name] = int(rounded)
    prefix_hits = deltas["prefix_cache_hits"]
    prefix_queries = deltas["prefix_cache_queries"]
    assert prefix_hits is not None and prefix_queries is not None
    if prefix_hits > prefix_queries:
        raise ValueError("prefix cache hits exceed prefix cache queries")
    return deltas


def _load_tasks(
    summary_path: Path,
    summary: dict[str, Any],
    *,
    allow_background: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], float, float, str]:
    attempted = _nonnegative_int(
        summary.get("attempted_traces"), "attempted_traces", positive=True
    )
    completed = _nonnegative_int(summary.get("completed_traces"), "completed_traces")
    failed = _nonnegative_int(summary.get("failed_traces"), "failed_traces")
    if completed != attempted or failed != 0:
        raise ValueError("throughput summary does not describe a fully successful run")
    raw_tasks = summary.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != attempted:
        raise ValueError("throughput summary task count is inconsistent")
    tasks: dict[str, dict[str, Any]] = {}
    makespan_s = 0.0
    for index, raw_task in enumerate(raw_tasks):
        task = _object(raw_task, f"tasks[{index}]")
        task_id = task.get("run_instance_id")
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError(f"invalid or duplicate task ID at tasks[{index}]")
        if task.get("success") is not True:
            raise ValueError(f"task {task_id} did not succeed")
        arrival_s = _finite(
            task.get("arrival_s"), f"{task_id}.arrival_s", nonnegative=True
        )
        ready_s = _finite(
            task.get("ready_to_terminal_s"),
            f"{task_id}.ready_to_terminal_s",
            nonnegative=True,
        )
        tasks[task_id] = task
        makespan_s = max(makespan_s, arrival_s + ready_s)
    if makespan_s <= 0:
        raise ValueError("scheduled makespan must be positive")

    arrival_zero = summary.get("arrival_zero_wall_time_s")
    common_ready = summary.get("common_ready_wall_time_s")
    if arrival_zero is not None:
        window_start = _finite(
            arrival_zero, "arrival_zero_wall_time_s", nonnegative=True
        )
        window_source = "arrival_zero_wall_time_s"
    elif common_ready is not None:
        window_start = _finite(
            common_ready, "common_ready_wall_time_s", nonnegative=True
        )
        makespan_s = max(
            _finite(
                task.get("ready_to_terminal_s"),
                f"{task_id}.ready_to_terminal_s",
                nonnegative=True,
            )
            for task_id, task in tasks.items()
        )
        window_source = "common_ready_wall_time_s"
    else:
        raise ValueError("throughput summary lacks a measured run start time")
    output_dir = summary_path.parent
    startup_by_task: dict[str, str] = {}
    for startup_path in sorted(output_dir.glob("*/attempt_*/container_startup.json")):
        startup = _read_json(startup_path, "container_startup")
        task_id = startup.get("run_instance_id")
        if task_id not in tasks and not allow_background:
            raise ValueError(f"unexpected task in {startup_path}: {task_id!r}")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"invalid task in {startup_path}: {task_id!r}")
        if task_id in startup_by_task:
            raise ValueError(f"multiple container_startup files for {task_id}")
        if startup.get("status") != "success":
            if (
                allow_background
                and task_id not in tasks
                and isinstance(startup.get("error"), dict)
                and startup["error"].get("type") == "CancelledError"
            ):
                continue
            raise ValueError(f"container startup did not succeed for {task_id}")
        started_at = startup.get("started_at")
        if not isinstance(started_at, str):
            raise ValueError(f"missing started_at for {task_id}")
        try:
            parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid started_at for {task_id}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"started_at lacks timezone for {task_id}")
        startup_by_task[task_id] = started_at
    if not tasks.keys() <= startup_by_task.keys():
        missing = sorted(tasks.keys() - startup_by_task.keys())
        raise ValueError(f"missing container_startup files: {', '.join(missing)}")
    for task_id, started_at in startup_by_task.items():
        if task_id not in tasks:
            continue
        tasks[task_id] = {
            **tasks[task_id],
            "task_preparation_started_at": started_at,
        }
    return tasks, startup_by_task, window_start, makespan_s, window_source


def _load_requests(
    trace_path: Path,
    tasks: dict[str, dict[str, Any]],
    task_startups: dict[str, str],
    expected_measurement_count: int,
    *,
    allow_background: bool = False,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    seen_task_indices: set[tuple[str, int]] = set()
    try:
        handle = trace_path.open()
    except OSError as exc:
        raise ValueError(f"cannot open combined trace {trace_path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = _object(json.loads(line), f"trace line {line_number}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {trace_path}:{line_number}") from exc
            if event.get("type") != "action" or event.get("action_type") != "llm_call":
                continue
            data = _object(event.get("data"), f"trace line {line_number}.data")
            shadow = _object(
                data.get("shadow_generation"),
                f"trace line {line_number}.shadow_generation",
            )
            task_id = data.get("run_instance_id")
            if task_id not in task_startups:
                raise ValueError(f"unknown request task at trace line {line_number}")
            request_id = shadow.get("request_id")
            if (
                not isinstance(request_id, str)
                or not request_id
                or request_id in seen_request_ids
            ):
                raise ValueError(
                    f"invalid or duplicate request ID at trace line {line_number}"
                )
            source_index = _nonnegative_int(
                shadow.get("source_action_index"),
                f"{request_id}.source_action_index",
            )
            if (task_id, source_index) in seen_task_indices:
                raise ValueError(f"duplicate source request index for {task_id}")
            prompt_tokens = _nonnegative_int(
                shadow.get("prompt_tokens"),
                f"{request_id}.prompt_tokens",
                positive=True,
            )
            requested_tokens = _nonnegative_int(
                shadow.get("requested_completion_tokens"),
                f"{request_id}.requested_completion_tokens",
                positive=True,
            )
            generation_tokens = _nonnegative_int(
                shadow.get("returned_completion_tokens"),
                f"{request_id}.returned_completion_tokens",
                positive=True,
            )
            if (
                requested_tokens != generation_tokens
                or shadow.get("finish_reason") != "length"
            ):
                raise ValueError(
                    f"request {request_id} did not return its forced length"
                )
            raw_cached = shadow.get("cached_prompt_tokens")
            cached_tokens = (
                None
                if raw_cached is None
                else _nonnegative_int(raw_cached, f"{request_id}.cached_prompt_tokens")
            )
            if cached_tokens is not None and cached_tokens > prompt_tokens:
                raise ValueError(f"cached tokens exceed prompt tokens for {request_id}")
            ttft_ms = _finite(
                shadow.get("ttft_ms"), f"{request_id}.ttft_ms", nonnegative=True
            )
            latency_ms = _finite(
                shadow.get("latency_ms"), f"{request_id}.latency_ms", nonnegative=True
            )
            if latency_ms < ttft_ms:
                raise ValueError(f"latency is shorter than TTFT for {request_id}")
            started_s = _finite(event.get("ts_start"), f"{request_id}.ts_start")
            ended_s = _finite(event.get("ts_end"), f"{request_id}.ts_end")
            if ended_s < started_s:
                raise ValueError(f"request timestamps are reversed for {request_id}")
            requests.append(
                {
                    "request_id": request_id,
                    "task_id": task_id,
                    "action_id": event.get("action_id"),
                    "source_action_index": source_index,
                    "request_started_wall_time_s": started_s,
                    "request_ended_wall_time_s": ended_s,
                    "task_preparation_started_at": task_startups[task_id],
                    "prompt_tokens": prompt_tokens,
                    "cached_prompt_tokens": cached_tokens,
                    "generation_tokens": generation_tokens,
                    "ttft_s": ttft_ms / 1000.0,
                    "latency_s": latency_ms / 1000.0,
                    **(
                        {"measurement_task": task_id in tasks}
                        if allow_background
                        else {}
                    ),
                }
            )
            seen_request_ids.add(request_id)
            seen_task_indices.add((task_id, source_index))
    measured_count = sum(request["task_id"] in tasks for request in requests)
    if measured_count != expected_measurement_count:
        raise ValueError(
            "measured request count differs: "
            f"trace={measured_count}, summary={expected_measurement_count}"
        )
    if not allow_background and len(requests) != expected_measurement_count:
        raise ValueError("request count includes unexpected background tasks")
    if not tasks.keys() <= {request["task_id"] for request in requests}:
        raise ValueError("combined trace does not contain requests for every task")

    request_task_ids = {request["task_id"] for request in requests}
    first_index = {
        task_id: min(
            request["source_action_index"]
            for request in requests
            if request["task_id"] == task_id
        )
        for task_id in request_task_ids
    }
    for request in requests:
        request["first_request"] = (
            request["source_action_index"] == first_index[request["task_id"]]
        )
        decode_tokens = request["generation_tokens"] - 1
        decode_s = request["latency_s"] - request["ttft_s"]
        if decode_tokens:
            if decode_s <= 0:
                raise ValueError(
                    f"decode interval is not positive for {request['request_id']}"
                )
            request["tpot_s"] = decode_s / decode_tokens
            request["decode_tokens_per_s"] = decode_tokens / decode_s
        else:
            request["tpot_s"] = None
            request["decode_tokens_per_s"] = None
    requests.sort(
        key=lambda request: (
            request["request_started_wall_time_s"],
            request["request_id"],
        )
    )
    return requests


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_s": sum(values) / len(values) if values else None,
        "p50_s": _percentile(values, 0.50),
        "p95_s": _percentile(values, 0.95),
        "p99_s": _percentile(values, 0.99),
        "max_s": max(values) if values else None,
    }


def _gpu_summary(
    path: Path, window_start_s: float, window_end_s: float
) -> dict[str, Any]:
    expected_header = [
        "timestamp_s",
        "power_w",
        "memory_mib",
        "utilization_pct",
        "memory_activity_pct",
    ]
    rows: list[tuple[float, float, float]] = []
    previous_timestamp: float | None = None
    try:
        handle = path.open(newline="")
    except OSError as exc:
        raise ValueError(f"cannot read GPU telemetry {path}: {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("GPU telemetry is empty") from exc
        if header != expected_header:
            raise ValueError("GPU telemetry header is invalid")
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(expected_header):
                raise ValueError(f"GPU telemetry row {line_number} is incomplete")
            values = [
                _finite(value, f"GPU row {line_number}") for value in map(float, row)
            ]
            timestamp, _, _, utilization, memory_activity = values
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("GPU timestamps are not strictly increasing")
            previous_timestamp = timestamp
            if not (0 <= utilization <= 100 and 0 <= memory_activity <= 100):
                raise ValueError(f"GPU percentage is out of range at row {line_number}")
            if window_start_s <= timestamp <= window_end_s:
                rows.append((timestamp, utilization, memory_activity))
    if not rows:
        raise ValueError("GPU telemetry has no samples in the scheduled run window")
    gaps = [rows[0][0] - window_start_s, window_end_s - rows[-1][0]]
    gaps.extend(right[0] - left[0] for left, right in zip(rows, rows[1:]))
    max_gap_s = max(gaps)
    utilization = [row[1] for row in rows]
    memory_activity = [row[2] for row in rows]

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "mean_pct": sum(values) / len(values),
            "p50_pct": _percentile(values, 0.50),
            "p95_pct": _percentile(values, 0.95),
            "max_pct": max(values),
        }

    return {
        "sample_count": len(rows),
        "max_sample_gap_s": max_gap_s,
        "utilization": stats(utilization),
        "memory_activity": stats(memory_activity),
        "memory_activity_definition": (
            "nvidia-smi utilization.memory: percent of the sample period during "
            "which global device memory was being read or written"
        ),
    }


def _dram_bandwidth_summary(
    path: Path, window_start_s: float, window_end_s: float
) -> dict[str, Any]:
    expected_header = [
        "start_timestamp_ns",
        "end_timestamp_ns",
        "gpu_id",
        "read_bytes_per_s",
        "write_bytes_per_s",
    ]
    rows: list[tuple[int, int, float, float]] = []
    previous_end_ns: int | None = None
    try:
        handle = path.open(newline="")
    except OSError as exc:
        raise ValueError(f"cannot read DRAM bandwidth telemetry {path}: {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("DRAM bandwidth telemetry is empty") from exc
        if header != expected_header:
            raise ValueError("DRAM bandwidth telemetry header is invalid")
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(expected_header):
                raise ValueError(
                    f"DRAM bandwidth telemetry row {line_number} is incomplete"
                )
            try:
                start_ns = int(row[0])
                end_ns = int(row[1])
                gpu_id = int(row[2])
                read_bytes_per_s = float(row[3])
                write_bytes_per_s = float(row[4])
            except ValueError as exc:
                raise ValueError(
                    f"DRAM bandwidth telemetry row {line_number} has an invalid value"
                ) from exc
            if start_ns < 0 or end_ns <= start_ns or gpu_id != 0:
                raise ValueError(
                    f"DRAM bandwidth telemetry row {line_number} is invalid"
                )
            if previous_end_ns is not None and start_ns < previous_end_ns:
                raise ValueError(
                    "DRAM bandwidth timestamps are not increasing and non-overlapping"
                )
            if not all(
                math.isfinite(value) and value >= 0
                for value in (read_bytes_per_s, write_bytes_per_s)
            ):
                raise ValueError(f"DRAM bandwidth rate is invalid at row {line_number}")
            previous_end_ns = end_ns
            if (
                end_ns > window_start_s * 1_000_000_000
                and start_ns < window_end_s * 1_000_000_000
            ):
                rows.append((start_ns, end_ns, read_bytes_per_s, write_bytes_per_s))
    if not rows:
        raise ValueError(
            "DRAM bandwidth telemetry has no samples in the scheduled run window"
        )
    window_start_ns = window_start_s * 1_000_000_000
    window_end_ns = window_end_s * 1_000_000_000
    clipped = [
        (max(start, window_start_ns), min(end, window_end_ns), read, write)
        for start, end, read, write in rows
    ]
    gaps_ns = [
        clipped[0][0] - window_start_ns,
        window_end_ns - clipped[-1][1],
    ]
    gaps_ns.extend(right[0] - left[1] for left, right in zip(clipped, clipped[1:]))
    max_gap_s = max(0, *gaps_ns) / 1_000_000_000
    read_rates = [row[2] for row in clipped]
    write_rates = [row[3] for row in clipped]
    total_rates = [read + write for read, write in zip(read_rates, write_rates)]

    def stats(values: list[float]) -> dict[str, float]:
        gb_per_s = [value / 1_000_000_000 for value in values]
        return {
            "mean_gb_per_s": sum(gb_per_s) / len(gb_per_s),
            "p50_gb_per_s": _percentile(gb_per_s, 0.50),
            "p95_gb_per_s": _percentile(gb_per_s, 0.95),
            "max_gb_per_s": max(gb_per_s),
        }

    read_bytes = sum(
        read * (end - start) / 1_000_000_000 for start, end, read, _ in clipped
    )
    write_bytes = sum(
        write * (end - start) / 1_000_000_000 for start, end, _, write in clipped
    )
    return {
        "sample_count": len(rows),
        "max_sample_gap_s": max_gap_s,
        "read": stats(read_rates),
        "write": stats(write_rates),
        "total": stats(total_rates),
        "integrated_bytes": {
            "read": read_bytes,
            "write": write_bytes,
            "total": read_bytes + write_bytes,
        },
        "scope": "CUDA context",
        "definition": (
            "CUPTI PM Sampling dram__bytes_read.sum.per_second and "
            "dram__bytes_write.sum.per_second for the CUDA context attached to "
            "the collector; values do not include other CUDA contexts"
        ),
    }


def _kv_summary(path: Path) -> dict[str, Any]:
    summary = _read_json(path, "KV event summary")
    integer_fields = (
        "stored_blocks",
        "removed_blocks",
        "clear_count",
        "first_seq",
        "last_seq",
        "batch_count",
        "event_count",
        "replayed_tail_batches",
        "tail_replay_rounds",
    )
    result: dict[str, Any] = {}
    for field in integer_fields:
        value = summary.get(field)
        if field in {"first_seq", "last_seq"} and value is None:
            result[field] = None
        else:
            result[field] = _nonnegative_int(value, f"KV summary {field}")
    gaps = summary.get("sequence_gaps")
    if not isinstance(gaps, list) or gaps:
        raise ValueError("KV event sequence_gaps must be an empty list")
    result["sequence_gaps"] = []
    removed_tokens = summary.get("removed_tokens")
    if removed_tokens is None:
        if result["removed_blocks"]:
            raise ValueError("removed_tokens is required when KV blocks were removed")
        result["removed_tokens"] = None
    else:
        result["removed_tokens"] = _nonnegative_int(
            removed_tokens, "KV summary removed_tokens"
        )
    if summary.get("tail_replay_complete") is not True:
        raise ValueError("KV tail replay did not complete")
    result["tail_replay_complete"] = True
    if result["batch_count"]:
        if result["first_seq"] is None or result["last_seq"] is None:
            raise ValueError("KV event sequence endpoints are missing")
        if result["last_seq"] < result["first_seq"]:
            raise ValueError("KV event sequence endpoints are reversed")
    elif result["first_seq"] is not None or result["last_seq"] is not None:
        raise ValueError("empty KV event summary has sequence endpoints")
    if result["clear_count"]:
        raise ValueError("KV cache was cleared during the measured run")
    return result


def summarize(
    *,
    throughput_summary_path: Path,
    gpu_csv_path: Path,
    dram_bandwidth_csv_path: Path,
    prometheus_start_path: Path,
    prometheus_final_path: Path,
    kv_events_summary_path: Path,
    output_path: Path,
    requests_output_path: Path,
) -> dict[str, Any]:
    if output_path == requests_output_path:
        raise ValueError("summary and request output paths must differ")
    throughput = _read_json(throughput_summary_path, "throughput summary")
    if "replacement_load" in throughput and "background_load" in throughput:
        raise ValueError("throughput summary cannot contain both load configurations")
    replacement_load = throughput.get("replacement_load")
    background_load = throughput.get("background_load")
    load_config = background_load if "background_load" in throughput else replacement_load
    background_enabled = (
        isinstance(load_config, dict) and load_config.get("enabled") is True
    )
    tasks, task_startups, window_start, makespan_s, window_source = _load_tasks(
        throughput_summary_path,
        throughput,
        allow_background=background_enabled,
    )
    expected_request_count = _nonnegative_int(
        throughput.get("llm_call_count"), "llm_call_count", positive=True
    )
    trace_value = throughput.get("trace_file")
    if not isinstance(trace_value, str) or not trace_value:
        raise ValueError("throughput summary lacks trace_file")
    trace_path = Path(trace_value)
    requests = _load_requests(
        trace_path,
        tasks,
        task_startups,
        expected_request_count,
        allow_background=background_enabled,
    )
    request_count = len(requests)
    counters = _counter_deltas(prometheus_start_path, prometheus_final_path)

    for request in requests:
        if request["cached_prompt_tokens"] is None:
            request["cached_prompt_tokens"] = 0
            request["cached_prompt_tokens_omitted_zero"] = True
    for request in requests:
        request.setdefault("cached_prompt_tokens_omitted_zero", False)

    prompt_tokens = sum(request["prompt_tokens"] for request in requests)
    cached_tokens = sum(request["cached_prompt_tokens"] for request in requests)
    generation_tokens = sum(request["generation_tokens"] for request in requests)
    recomputed_tokens = sum(
        1
        for request in requests
        if request["cached_prompt_tokens"] > 0
        and request["cached_prompt_tokens"] + 1 == request["prompt_tokens"]
    )
    unattributed_counter_tokens: dict[str, int | None] = {}

    def compare_counter(name: str, observed: int, counter: int | None) -> None:
        if counter is None:
            unattributed_counter_tokens[name] = None
        elif background_enabled:
            if observed > counter:
                raise ValueError(f"request {name} total exceeds Prometheus")
            unattributed_counter_tokens[name] = counter - observed
        elif observed != counter:
            raise ValueError(f"request {name} total differs from Prometheus")

    compare_counter("prompt_tokens", prompt_tokens, counters["prompt_tokens"])
    if (
        counters["cached_prompt_tokens"] is not None
        and not background_enabled
        and cached_tokens != counters["cached_prompt_tokens"]
    ):
        raise ValueError("request cached-token total differs from Prometheus")
    if (
        counters["recomputed_prompt_tokens"] is not None
        and not background_enabled
        and recomputed_tokens != counters["recomputed_prompt_tokens"]
    ):
        raise ValueError("derived recomputed-token total differs from Prometheus")
    compare_counter(
        "cached_prompt_tokens",
        cached_tokens,
        counters["cached_prompt_tokens"],
    )
    compare_counter(
        "recomputed_prompt_tokens",
        recomputed_tokens,
        counters["recomputed_prompt_tokens"],
    )
    compare_counter(
        "generation_tokens",
        generation_tokens,
        counters["generation_tokens"],
    )

    window_end = window_start + makespan_s
    gpu = _gpu_summary(gpu_csv_path, window_start, window_end)
    gpu["dram_bandwidth"] = _dram_bandwidth_summary(
        dram_bandwidth_csv_path, window_start, window_end
    )
    kv_events = _kv_summary(kv_events_summary_path)
    all_ttft = [request["ttft_s"] for request in requests]
    first_ttft = [request["ttft_s"] for request in requests if request["first_request"]]
    subsequent_ttft = [
        request["ttft_s"] for request in requests if not request["first_request"]
    ]
    if len(first_ttft) != len({request["task_id"] for request in requests}):
        raise ValueError("each task must have exactly one first request")
    prefix_queries = counters["prefix_cache_queries"]
    prefix_hits = counters["prefix_cache_hits"]
    assert prefix_queries is not None and prefix_hits is not None
    summary = {
        "schema_version": 3,
        "window": {
            "start_wall_time_s": window_start,
            "start_source": window_source,
            "scheduled_end_wall_time_s": window_end,
            "scheduled_makespan_s": makespan_s,
        },
        "request_count": request_count,
        "task_count": len(tasks),
        **(
            {
                "observed_task_count": len(
                    {request["task_id"] for request in requests}
                ),
                (
                    "background_load"
                    if "background_load" in throughput
                    else "replacement_load"
                ): load_config,
            }
            if background_enabled
            else {}
        ),
        "request_token_totals": {
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_tokens,
            "recomputed_prompt_tokens": recomputed_tokens,
            "generation_tokens": generation_tokens,
            **(
                {"unattributed_prometheus_tokens": unattributed_counter_tokens}
                if background_enabled
                else {}
            ),
        },
        "request_token_definitions": {
            "cached_prompt_tokens": (
                "sum of OpenAI usage prompt_tokens_details.cached_tokens; "
                "vLLM omits the details object when this value is zero"
            ),
            "recomputed_prompt_tokens": (
                "one token when vLLM reports cached_prompt_tokens + 1 == "
                "prompt_tokens for a request with a nonzero cache hit"
            ),
        },
        "headline": {
            "request_usage_cached_prompt_token_ratio": cached_tokens / prompt_tokens,
            "definition": "sum(request cached prompt tokens) / sum(request prompt tokens)",
        },
        "prometheus_counter_deltas": {
            **counters,
            "prefix_lookup_token_hit_ratio": (
                prefix_hits / prefix_queries if prefix_queries else None
            ),
            "prefix_lookup_token_hit_ratio_definition": (
                "vLLM prefix-cache hit-token counter delta / query-token counter delta"
            ),
        },
        "ttft": {
            "all_requests": _distribution(all_ttft),
            "first_request_per_task": _distribution(first_ttft),
            "subsequent_requests": _distribution(subsequent_ttft),
        },
        "whole_run_generation_tokens_per_s": (
            int(counters["generation_tokens"]) / makespan_s
        ),
        "whole_run_generation_throughput_definition": (
            "generation token counter delta / "
            + (
                "measured-cohort makespan seconds"
                if background_enabled
                else "scheduled makespan seconds"
            )
        ),
        "gpu": gpu,
        "kv_events": kv_events,
        "task_preparation_started_at": {
            task_id: tasks[task_id]["task_preparation_started_at"]
            for task_id in sorted(tasks)
        },
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    requests_output_path.write_text(
        "".join(json.dumps(request, sort_keys=True) + "\n" for request in requests)
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput-summary", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--dram-bandwidth-csv", type=Path, required=True)
    parser.add_argument("--prometheus-start", type=Path, required=True)
    parser.add_argument("--prometheus-final", type=Path, required=True)
    parser.add_argument("--kv-events-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summarize(
        throughput_summary_path=args.throughput_summary,
        gpu_csv_path=args.gpu_csv,
        dram_bandwidth_csv_path=args.dram_bandwidth_csv,
        prometheus_start_path=args.prometheus_start,
        prometheus_final_path=args.prometheus_final,
        kv_events_summary_path=args.kv_events_summary,
        output_path=args.output,
        requests_output_path=args.requests_output,
    )


if __name__ == "__main__":
    main()
