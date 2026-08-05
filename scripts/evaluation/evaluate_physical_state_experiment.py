#!/usr/bin/env python3
"""Freeze the causal CPU control and score the SQLGlot physical-state experiment."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.evaluation.build_physical_state_conditions import (  # noqa: E402
    PREPARED_SCHEMA,
    PROBE_CONTAINER_PATH,
    PROTOCOL_SCHEMA,
    TEMPLATE_CONTAINER_PATH,
    _trace,
    condition_schedule,
)
from scripts.evaluation.build_physical_state_manifest import (  # noqa: E402
    FROZEN_TASK_IDS,
    VERSION as MANIFEST_SCHEMA,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    Row,
    load_rows,
    load_run_rows,
)
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    ClauseResourceKB,
)
from tool_resource_eval.labels import repo_of  # noqa: E402
from harness.container_image_prep import normalize_image_reference  # noqa: E402

BASELINE_SCHEMA = "sqlglot-physical-state-current-baseline-v1"
RESULT_SCHEMA = "sqlglot-physical-state-result-v1"
CPU = "peak_cpu_cores"
DISK = "disk_read_write_bytes_total"
CPU_SAMPLE_MIN_SPAN_S = 0.400
CPU_AVAILABILITY_PAD_S = 0.050
CPU_ACTUATION_P95_S = 0.09132007875
PROBE_P95_LIMIT_MS = 100.0
RESIDENCY_MIN_DELTA = 0.50
FROZEN_CPU_EDGES = (2.0, 4.0)
FROZEN_DISK_EDGES = (float(1024 * 1024), float(100 * 1024 * 1024))
_EXIT = re.compile(r"(?:^|\n)Exit code: (-?\d+)\s*$")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _target_call_id(task: Mapping[str, Any], repo_root: Path) -> tuple[str, str]:
    trace = (repo_root / str(task["source_trace"])).resolve()
    trace_bytes = trace.read_bytes()
    matches = []
    for line_number, line in enumerate(trace_bytes.decode().splitlines(), 1):
        action = json.loads(line)
        if action.get("action_id") == task["target_action_id"]:
            matches.append((line_number, action))
    if len(matches) != 1 or matches[0][0] != task["target_trace_line"]:
        raise ValueError(f"frozen target moved for {task['task_id']}")
    data = matches[0][1].get("data")
    call_id = data.get("tool_call_id") if isinstance(data, Mapping) else None
    if not isinstance(call_id, str):
        raise ValueError(f"frozen target lacks a call id for {task['task_id']}")
    return call_id, hashlib.sha256(trace_bytes).hexdigest()


def _target_attempt_inputs(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    results_path = run_dir / "results.jsonl"
    results_bytes = results_path.read_bytes()
    inputs = []
    for record in map(json.loads, results_bytes.splitlines()):
        task_id = record.get("instance_id")
        attempt_value = record.get("attempt_dir")
        if not isinstance(task_id, str) or not isinstance(attempt_value, str):
            raise ValueError("target results lack task or attempt identity")
        attempt = Path(attempt_value)
        if not attempt.is_absolute():
            attempt = run_dir / attempt
        attempt = attempt.resolve()
        if not attempt.is_relative_to(run_dir.resolve()):
            raise ValueError("target attempt escapes the run directory")
        row = {"task_id": task_id, "attempt_dir": str(attempt)}
        for name in ("resource_observations.json", "tool_calls.json"):
            path = attempt / name
            row[name] = {"path": str(path), "sha256": _sha256(path)}
        inputs.append(row)
    return hashlib.sha256(results_bytes).hexdigest(), inputs


def build_current_baseline(
    manifest_path: Path,
    run_dir: Path,
    public_paths: Sequence[Path],
    excluded_public_repos: Sequence[str],
    *,
    repo_root: Path = _ROOT,
) -> dict[str, Any]:
    """Freeze Current CPU predictions before each selected task begins."""
    if tuple(CANONICAL_RESOURCE_BUCKET_EDGES[CPU]) != FROZEN_CPU_EDGES:
        raise ValueError("canonical CPU bucket edges differ from the frozen protocol")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_tasks = manifest.get("tasks")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or not isinstance(manifest_tasks, list)
        or tuple(task.get("task_id") for task in manifest_tasks) != FROZEN_TASK_IDS
    ):
        raise ValueError("physical-state manifest differs from the frozen task set")

    target_results_sha256, target_attempt_inputs = _target_attempt_inputs(run_dir)
    public_inputs = [
        {"path": str(path.resolve()), "sha256": _sha256(path)} for path in public_paths
    ]
    task_ids, clause_rows, command_rows = load_run_rows(run_dir)
    if len(task_ids) != 100 or len(set(task_ids)) != 100:
        raise ValueError("Current baseline requires the frozen original 100-task run")
    target_repos = {repo_of(task_id) for task_id in task_ids}
    excluded = target_repos | set(excluded_public_repos)
    public = [row for path in public_paths for row in load_rows(path)]
    if (
        _target_attempt_inputs(run_dir)
        != (target_results_sha256, target_attempt_inputs)
        or [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in public_paths
        ]
        != public_inputs
    ):
        raise ValueError("baseline evidence changed while it was being read")
    raw_public_count = len(public)
    public = [row for row in public if row.repo not in excluded]
    if not public or {row.task_id for row in public} & set(task_ids):
        raise ValueError("public evidence is empty or overlaps the target run")
    online_public = [
        row for row in public if row.structure_known and row.pipeline_position <= 0
    ]
    kb = ClauseResourceKB.fit_public(row.observation(0.0, 1.0) for row in online_public)

    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)
    selected = {str(task["task_id"]): task for task in manifest_tasks}
    predictions: dict[str, dict[str, Any]] = {}
    for ordinal, task_id in enumerate(task_ids):
        query_ts = float(ordinal * 2 + 3)
        task = selected.get(task_id)
        if task is not None:
            call_id, source_trace_sha256 = _target_call_id(task, repo_root)
            matches = [
                row for row in commands_by_task[task_id] if row.call_id == call_id
            ]
            if len(matches) != 1 or len(matches[0].clauses) != 1:
                raise ValueError(f"selected command identity changed for {task_id}")
            command = matches[0]
            prediction = kb.predict_command_resource_buckets(
                command.repo, command.command, query_ts
            ).classifications[CPU]
            if prediction is None:
                raise ValueError(f"Current CPU prediction unavailable for {task_id}")
            predictions[task_id] = {
                "task_id": task_id,
                "source_task_ordinal": ordinal,
                "target_call_id": call_id,
                "target_call_index": command.call_index,
                "source_trace_sha256": source_trace_sha256,
                "prediction_class": prediction.label,
                "prediction_class_id": prediction.bucket_id,
                "probability_by_bucket": list(prediction.probability_by_bucket),
                "scope": prediction.scope,
                "key_kind": prediction.key_kind,
                "evidence_count": prediction.evidence_count,
                "fallback_path": list(prediction.fallback_path),
                "canonicalizer_version": prediction.canonicalizer_version,
                "arbitration": prediction.arbitration,
            }
        settle_ts = query_ts + 0.5
        for row in clauses_by_task[task_id]:
            kb.observe_completed_clause(row.observation(query_ts, settle_ts))
    if set(predictions) != set(FROZEN_TASK_IDS):
        raise ValueError("Current baseline omitted or reordered a frozen task")

    return {
        "schema": BASELINE_SCHEMA,
        "claim_bearing": False,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "target_run_dir": str(run_dir.resolve()),
        "target_results_sha256": target_results_sha256,
        "target_attempt_inputs": target_attempt_inputs,
        "public_telemetry": public_inputs,
        "excluded_public_repositories": sorted(excluded),
        "public_clause_observations_before_repo_filter": raw_public_count,
        "public_online_clause_observations": len(online_public),
        "protocol": {
            "query_time": "before any observation from the selected task",
            "update_time": "after successful whole-task finalization",
            "hierarchy": "unchanged Current raw exact/argv-prefix/bin",
            "resource": CPU,
            "bucket_edges": list(FROZEN_CPU_EDGES),
            "same_task_observations_used": False,
        },
        "tasks": [predictions[task_id] for task_id in FROZEN_TASK_IDS],
    }


def _exit_code(result: Any) -> int | None:
    match = _EXIT.search(str(result or ""))
    return None if match is None else int(match.group(1))


def _probe_result(result: Any, condition: str) -> dict[str, float | int | str]:
    lines = [line for line in str(result or "").splitlines() if line.strip()]
    if len(lines) < 2 or lines[-1] != "Exit code: 0":
        raise ValueError("physical-state probe did not exit successfully")
    value = json.loads(lines[0])
    expected = {
        "condition",
        "file_count",
        "total_bytes",
        "total_pages",
        "resident_pages",
        "resident_fraction",
        "intervention_ms",
        "probe_ms",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["condition"] != condition
    ):
        raise ValueError("physical-state probe output differs from its schema")
    integers = ("file_count", "total_bytes", "total_pages", "resident_pages")
    if any(
        isinstance(value[key], bool)
        or not isinstance(value[key], int)
        or value[key] <= 0
        for key in integers[:-1]
    ) or (
        isinstance(value["resident_pages"], bool)
        or not isinstance(value["resident_pages"], int)
        or not 0 <= value["resident_pages"] <= value["total_pages"]
    ):
        raise ValueError("physical-state probe counts are invalid")
    for key in ("resident_fraction", "intervention_ms", "probe_ms"):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], (int, float))
            or not math.isfinite(float(value[key]))
            or float(value[key]) < 0.0
        ):
            raise ValueError("physical-state probe measurement is invalid")
    expected_fraction = value["resident_pages"] / value["total_pages"]
    if not math.isclose(
        float(value["resident_fraction"]), expected_fraction, abs_tol=1e-9
    ):
        raise ValueError("physical-state probe fraction does not match page counts")
    return value


def _bucket(value: float, resource: str) -> int:
    edges = FROZEN_CPU_EDGES if resource == CPU else FROZEN_DISK_EDGES
    return bisect_left(edges, value)


def _target_metrics(observation: Any) -> dict[str, Any]:
    if (
        not isinstance(observation, Mapping)
        or observation.get("eligible_for_kb") is not True
    ):
        raise ValueError("target resource observation is unavailable")
    clauses = observation.get("clauses")
    if not isinstance(clauses, list) or len(clauses) != 1:
        raise ValueError("physical-state target is no longer a single clause")
    clause = clauses[0]
    disk = clause.get("disk_io") if isinstance(clause, Mapping) else None
    cpu = clause.get("peak_cpu_cores") if isinstance(clause, Mapping) else None
    profile = clause.get("cpu_window_profile") if isinstance(clause, Mapping) else None
    read = disk.get("read_bytes_total") if isinstance(disk, Mapping) else None
    total = disk.get("read_write_bytes_total") if isinstance(disk, Mapping) else None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (cpu, read, total)
    ):
        raise ValueError("target CPU or Disk metric is invalid")
    if not isinstance(profile, list):
        raise ValueError("target CPU window profile is unavailable")
    qualifying = [
        row
        for row in profile
        if isinstance(row, Mapping)
        and isinstance(row.get("span_s"), (int, float))
        and not isinstance(row.get("span_s"), bool)
        and float(row["span_s"]) >= CPU_SAMPLE_MIN_SPAN_S
    ]
    prefix_class = None
    decision_s = None
    if qualifying:
        decision_s = (
            float(qualifying[0]["end_offset_s"])
            + CPU_AVAILABILITY_PAD_S
            + CPU_ACTUATION_P95_S
        )
        available = [
            row
            for row in profile
            if isinstance(row, Mapping)
            and isinstance(row.get("end_offset_s"), (int, float))
            and not isinstance(row.get("end_offset_s"), bool)
            and float(row["end_offset_s"]) <= decision_s
            and isinstance(row.get("cpu_cores"), (int, float))
            and not isinstance(row.get("cpu_cores"), bool)
            and math.isfinite(float(row["cpu_cores"]))
            and float(row["cpu_cores"]) >= 0.0
        ]
        if available:
            prefix_class = max(
                _bucket(float(row["cpu_cores"]), CPU) for row in available
            )
    return {
        "disk_read_bytes": float(read),
        "disk_read_write_bytes": float(total),
        "disk_class": _bucket(float(total), DISK),
        "final_cpu_cores": float(cpu),
        "final_cpu_class": _bucket(float(cpu), CPU),
        "prefix_cpu_class": prefix_class,
        "decision_s": decision_s,
    }


def _load_telemetry(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    summaries: dict[int, dict[str, Any]] = {}
    actions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            index = row.get("manifest_index")
            if row.get("type") == "summary" and isinstance(index, int):
                if index in summaries:
                    raise ValueError(f"duplicate simulator summary for entry {index}")
                summaries[index] = row
            elif (
                row.get("type") == "action"
                and row.get("action_type") == "tool_exec"
                and isinstance(row.get("data"), Mapping)
                and isinstance(row["data"].get("manifest_index"), int)
            ):
                actions[int(row["data"]["manifest_index"])].append(row)
    return summaries, actions


def _repo_path(value: Any, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_relative_to(repo_root.resolve()):
        raise ValueError("artifact path escapes the repository")
    return path


def _validate_entry_inputs(
    entry: Mapping[str, Any],
    manifest_task: Mapping[str, Any],
    repo_root: Path,
) -> tuple[str, Path]:
    if (
        entry.get("target_action_id") != manifest_task.get("target_action_id")
        or entry.get("prepared_image_id") is None
    ):
        raise ValueError("condition target or prepared image identity changed")
    paths = {
        kind: _repo_path(entry.get(kind), repo_root)
        for kind in (
            "prepared_artifact",
            "discovery_artifact",
            "template_artifact",
            "probe_input",
        )
    }
    for kind, path in paths.items():
        if _sha256(path) != entry.get(f"{kind}_sha256"):
            raise ValueError(f"{kind} changed after condition generation")
    prepared = json.loads(paths["prepared_artifact"].read_text(encoding="utf-8"))
    if (
        prepared.get("schema") != PREPARED_SCHEMA
        or prepared.get("task_id") != entry["task_id"]
        or prepared.get("prepared_image_id") != entry["prepared_image_id"]
    ):
        raise ValueError("prepared artifact image differs from the condition")
    probe_input = paths["probe_input"].read_text(encoding="utf-8")
    trace_path = _repo_path(entry.get("trace"), repo_root)
    expected_trace = _trace(
        task_id=str(entry["task_id"]),
        condition=str(entry["condition"]),
        repeat=int(entry["repeat"]),
        probe_input=probe_input,
        target={
            "action_id": manifest_task["target_action_id"],
            "tool_args": manifest_task["target_tool_args"],
            "source_duration_ms": manifest_task["target_source_duration_ms"],
        },
    )
    if trace_path.read_text(encoding="utf-8") != expected_trace:
        raise ValueError("condition trace bytes differ from the frozen builder")
    return probe_input, trace_path


def _validate_container_startup(
    summary: Mapping[str, Any],
    entry: Mapping[str, Any],
    trace_path: Path,
    repo_root: Path,
) -> None:
    if Path(str(summary.get("source_trace") or "")).resolve() != trace_path:
        raise ValueError("simulator source trace differs from the frozen condition")
    resource_path = _repo_path(summary.get("resource_artifact_path"), repo_root)
    startup = json.loads(
        (resource_path.parent / "container_startup.json").read_text(encoding="utf-8")
    )
    expected_image = normalize_image_reference(str(entry["prepared_image_id"]))
    if (
        startup.get("status") != "success"
        or startup.get("task_instance_id") != entry["task_id"]
        or startup.get("manifest_index") != entry["order_index"]
        or startup.get("label") != entry["label"]
        or Path(str(startup.get("source_trace") or "")).resolve() != trace_path
        or startup.get("source_image") != expected_image
        or startup.get("container_id") != summary.get("tool_container_id")
    ):
        raise ValueError("container startup differs from the frozen condition")


def _validate_baseline_inputs(
    baseline: Mapping[str, Any],
    manifest_tasks: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> None:
    run_dir = _repo_path(baseline.get("target_run_dir"), repo_root)
    if _sha256(run_dir / "results.jsonl") != baseline.get("target_results_sha256"):
        raise ValueError("Current baseline target results changed")
    attempts = baseline.get("target_attempt_inputs")
    if not isinstance(attempts, list) or len(attempts) != 100:
        raise ValueError("Current baseline attempt provenance is incomplete")
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("Current baseline attempt provenance is malformed")
        attempt_dir = _repo_path(attempt.get("attempt_dir"), repo_root)
        if not attempt_dir.is_relative_to(run_dir):
            raise ValueError("Current baseline attempt escapes its run")
        for name in ("resource_observations.json", "tool_calls.json"):
            artifact = attempt.get(name)
            if not isinstance(artifact, Mapping):
                raise ValueError("Current baseline attempt artifact is missing")
            path = _repo_path(artifact.get("path"), repo_root)
            if path != attempt_dir / name or _sha256(path) != artifact.get("sha256"):
                raise ValueError("Current baseline attempt artifact changed")
    public = baseline.get("public_telemetry")
    if not isinstance(public, list) or not public:
        raise ValueError("Current baseline public provenance is incomplete")
    for artifact in public:
        path = _repo_path(artifact.get("path"), repo_root)
        if _sha256(path) != artifact.get("sha256"):
            raise ValueError("Current baseline public telemetry changed")
    baseline_tasks = {str(task["task_id"]): task for task in baseline["tasks"]}
    for task_id, manifest_task in manifest_tasks.items():
        source = _repo_path(manifest_task.get("source_trace"), repo_root)
        if _sha256(source) != baseline_tasks[task_id].get("source_trace_sha256"):
            raise ValueError("Current baseline selected source trace changed")


def _valid_summary(summary: Mapping[str, Any], entry: Mapping[str, Any]) -> str | None:
    if (
        summary.get("label") != entry["label"]
        or summary.get("task_instance_id") != entry["task_id"]
    ):
        raise ValueError("simulator summary identity differs from the frozen protocol")
    required = {
        "success": True,
        "collection_validity": "valid",
        "telemetry_quality": "ok",
        "formal_completeness": "complete",
        "replay_execution": "completed",
        "worker_returncode": 0,
        "action_sequence_matches": True,
        "telemetry_integrity_failed": False,
        "replay_failed_actions": 0,
        "unexpected_replay_failed_actions": 0,
        "missing_source_action_count": 0,
    }
    failed = [key for key, expected in required.items() if summary.get(key) != expected]
    return None if not failed else "invalid_summary:" + ",".join(failed)


def _score_entry(
    entry: Mapping[str, Any],
    manifest_task: Mapping[str, Any],
    baseline_task: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    actions: Sequence[Mapping[str, Any]],
    probe_input: str,
    trace_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    row = {
        "order_index": entry["order_index"],
        "task_id": entry["task_id"],
        "repeat": entry["repeat"],
        "condition": entry["condition"],
        "label": entry["label"],
        "valid": False,
    }
    if summary is None and not actions:
        return {**row, "invalid_reason": "missing_execution"}
    if summary is None:
        return {**row, "invalid_reason": "missing_summary"}
    summary_error = _valid_summary(summary, entry)
    if summary_error is not None:
        return {**row, "invalid_reason": summary_error}
    if not actions:
        return {**row, "invalid_reason": "missing_actions"}
    _validate_container_startup(summary, entry, trace_path, repo_root)

    probe_command = (
        f"{PROBE_CONTAINER_PATH} {entry['condition']} {TEMPLATE_CONTAINER_PATH}"
    )
    expected = (
        (
            "physical_state_template",
            "write_file",
            {"path": TEMPLATE_CONTAINER_PATH, "content": probe_input},
        ),
        (
            f"physical_state_probe_{entry['condition']}",
            "exec",
            {"command": probe_command, "timeout": 600, "working_dir": "/testbed"},
        ),
        (
            manifest_task["target_action_id"],
            "exec",
            dict(manifest_task["target_tool_args"]),
        ),
    )
    if len(actions) != len(expected):
        raise ValueError("measured action sequence differs from the frozen protocol")
    parsed_actions = []
    for action, (action_id, tool_name, expected_args) in zip(
        actions, expected, strict=True
    ):
        data = action["data"]
        raw_args = data.get("tool_args")
        args = json.loads(raw_args) if isinstance(raw_args, str) else None
        if (
            action.get("action_id") != action_id
            or data.get("task_instance_id") != entry["task_id"]
            or data.get("tool_name") != tool_name
            or args != expected_args
        ):
            raise ValueError("measured action differs from the frozen protocol")
        parsed_actions.append(data)
    probe, target = parsed_actions[1:]
    if probe.get("success") is not True or target.get("success") is not True:
        return {**row, "invalid_reason": "failed_exec_action"}
    if _exit_code(target.get("tool_result")) != 0:
        return {**row, "invalid_reason": "target_exit_mismatch"}
    try:
        probe_metrics = _probe_result(probe.get("tool_result"), str(entry["condition"]))
        target_metrics = _target_metrics(target.get("resource_observation"))
    except ValueError as exc:
        return {**row, "invalid_reason": str(exc)}

    base = int(baseline_task["prediction_class_id"])
    prefix = target_metrics["prefix_cpu_class"]
    candidate = None if prefix is None else max(base, prefix)
    truth = target_metrics["final_cpu_class"]
    change = "unavailable"
    if candidate is not None:
        change = (
            "helpful"
            if candidate == truth and base != truth
            else "harmful"
            if base == truth and candidate != truth
            else "neutral"
        )
    duration = target.get("duration_ms")
    return {
        **row,
        "valid": True,
        "probe": probe_metrics,
        **target_metrics,
        "base_cpu_class": base,
        "candidate_cpu_class": candidate,
        "cpu_change": change,
        "target_duration_ms": float(duration)
        if isinstance(duration, (int, float))
        else None,
    }


def score_experiment(
    protocol_path: Path,
    baseline_path: Path,
    telemetry_path: Path,
    *,
    repo_root: Path = _ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the frozen Disk and early-CPU mechanism gates."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    entries = protocol.get("entries")
    if protocol.get("schema") != PROTOCOL_SCHEMA or not isinstance(entries, list):
        raise ValueError("physical-state condition protocol is invalid")
    expected_schedule = condition_schedule(list(FROZEN_TASK_IDS))
    if (
        len(entries) != 48
        or [
            {key: row[key] for key in ("task_id", "repeat", "condition")}
            for row in entries
        ]
        != expected_schedule
    ):
        raise ValueError("physical-state condition schedule changed")
    if any(
        entry.get("order_index") != index
        or entry.get("label")
        != f"{entry['task_id']}/r{entry['repeat']}/{entry['condition']}"
        for index, entry in enumerate(entries)
    ):
        raise ValueError("physical-state condition order or labels changed")
    if (
        tuple(CANONICAL_RESOURCE_BUCKET_EDGES[CPU]) != FROZEN_CPU_EDGES
        or tuple(CANONICAL_RESOURCE_BUCKET_EDGES[DISK]) != FROZEN_DISK_EDGES
        or baseline.get("protocol", {}).get("bucket_edges") != list(FROZEN_CPU_EDGES)
    ):
        raise ValueError(
            "canonical resource bucket edges changed after preregistration"
        )
    manifest_path = (repo_root / str(protocol["manifest"])).resolve()
    if not manifest_path.is_relative_to(repo_root.resolve()):
        raise ValueError("physical-state manifest escapes the repository")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        protocol.get("manifest_sha256") != _sha256(manifest_path)
        or baseline.get("schema") != BASELINE_SCHEMA
        or baseline.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("physical-state manifest or Current baseline changed")
    manifest_tasks = {str(task["task_id"]): task for task in manifest["tasks"]}
    baseline_tasks = {str(task["task_id"]): task for task in baseline["tasks"]}
    if (
        tuple(manifest_tasks) != FROZEN_TASK_IDS
        or tuple(baseline_tasks) != FROZEN_TASK_IDS
    ):
        raise ValueError("physical-state task identities changed")
    for task_id, task in baseline_tasks.items():
        class_id = task.get("prediction_class_id")
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < len(RESOURCE_BUCKET_LABELS)
            or task.get("prediction_class") != RESOURCE_BUCKET_LABELS[class_id]
        ):
            raise ValueError(f"invalid frozen Current prediction for {task_id}")
    _validate_baseline_inputs(baseline, manifest_tasks, repo_root)
    entry_inputs = [
        _validate_entry_inputs(entry, manifest_tasks[str(entry["task_id"])], repo_root)
        for entry in entries
    ]

    summaries, actions = _load_telemetry(telemetry_path)
    if any(index < 0 or index >= len(entries) for index in (*summaries, *actions)):
        raise ValueError("simulator telemetry contains an unknown manifest index")
    rows = [
        _score_entry(
            entry,
            manifest_tasks[str(entry["task_id"])],
            baseline_tasks[str(entry["task_id"])],
            summaries.get(index),
            actions.get(index, ()),
            entry_inputs[index][0],
            entry_inputs[index][1],
            repo_root,
        )
        for index, entry in enumerate(entries)
    ]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    valid_tasks = {
        task_id: task_rows
        for task_id, task_rows in by_task.items()
        if len(task_rows) == 4 and all(row["valid"] for row in task_rows)
    }

    probe_ms = [
        float(row["probe"]["probe_ms"]) for task in valid_tasks.values() for row in task
    ]
    probe_p95 = _percentile(probe_ms, 0.95) if probe_ms else None
    resident_tasks: list[str] = []
    read_lower_tasks: list[str] = []
    lower_bucket_tasks: list[str] = []
    opposite_bucket_tasks: list[str] = []
    for task_id, task_rows in valid_tasks.items():
        indexed = {
            (int(row["repeat"]), str(row["condition"])): row for row in task_rows
        }
        resident_deltas = [
            float(indexed[(repeat, "warm")]["probe"]["resident_fraction"])
            - float(indexed[(repeat, "cold")]["probe"]["resident_fraction"])
            for repeat in (1, 2)
        ]
        if all(delta >= RESIDENCY_MIN_DELTA for delta in resident_deltas):
            resident_tasks.append(task_id)
        if all(
            indexed[(repeat, "warm")]["disk_read_bytes"]
            < indexed[(repeat, "cold")]["disk_read_bytes"]
            for repeat in (1, 2)
        ):
            read_lower_tasks.append(task_id)
        if all(
            indexed[(repeat, "warm")]["disk_class"]
            < indexed[(repeat, "cold")]["disk_class"]
            for repeat in (1, 2)
        ):
            lower_bucket_tasks.append(task_id)
        if all(
            indexed[(repeat, "warm")]["disk_class"]
            > indexed[(repeat, "cold")]["disk_class"]
            for repeat in (1, 2)
        ):
            opposite_bucket_tasks.append(task_id)

    disk_go = (
        len(valid_tasks) >= 10
        and probe_p95 is not None
        and probe_p95 < PROBE_P95_LIMIT_MS
        and len(resident_tasks) >= 8
        and len(read_lower_tasks) >= 8
        and len(lower_bucket_tasks) >= 4
        and len(opposite_bucket_tasks) <= 1
    )
    prefix_valid_tasks = [
        task_id
        for task_id, task_rows in valid_tasks.items()
        if all(row["prefix_cpu_class"] is not None for row in task_rows)
    ]
    prefix_rows = [
        row for task_id in prefix_valid_tasks for row in valid_tasks[task_id]
    ]
    prefix_violations = [
        row["label"]
        for row in prefix_rows
        if row["prefix_cpu_class"] > row["final_cpu_class"]
    ]
    helpful = [row for row in prefix_rows if row["cpu_change"] == "helpful"]
    harmful = [row for row in prefix_rows if row["cpu_change"] == "harmful"]
    helpful_tasks = sorted({str(row["task_id"]) for row in helpful})
    cpu_go = (
        len(prefix_valid_tasks) >= 10
        and not prefix_violations
        and len(helpful) >= 3
        and len(helpful_tasks) >= 3
        and len(harmful) <= 1
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": {
            "disk": "development_mechanism_go"
            if disk_go
            else "development_mechanism_no_go",
            "cpu": "development_mechanism_go"
            if cpu_go
            else "development_mechanism_no_go",
        },
        "claim_bearing": False,
        "inputs": {
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": _sha256(protocol_path),
            "baseline": str(baseline_path.resolve()),
            "baseline_sha256": _sha256(baseline_path),
            "telemetry": str(telemetry_path.resolve()),
            "telemetry_sha256": _sha256(telemetry_path),
        },
        "protocol": {
            "pairing": "task counts only when the frozen contrast holds in both repetitions",
            "disk_bucket_edges_bytes": list(FROZEN_DISK_EDGES),
            "cpu_bucket_edges_cores": list(FROZEN_CPU_EDGES),
            "cpu_base": "causal Current prediction before the selected task",
            "cpu_decision": {
                "minimum_observed_span_s": CPU_SAMPLE_MIN_SPAN_S,
                "availability_pad_s": CPU_AVAILABILITY_PAD_S,
                "actuation_p95_s": CPU_ACTUATION_P95_S,
                "candidate": "max(Current class, maximum completed prefix-window class)",
            },
        },
        "coverage": {
            "scheduled_tasks": len(FROZEN_TASK_IDS),
            "scheduled_executions": len(entries),
            "valid_tasks_all_four_executions": len(valid_tasks),
            "valid_executions": sum(row["valid"] for row in rows),
            "invalid_executions": [row["label"] for row in rows if not row["valid"]],
        },
        "disk": {
            "go": disk_go,
            "probe_p95_ms": probe_p95,
            "probe_p95_limit_ms_exclusive": PROBE_P95_LIMIT_MS,
            "resident_delta_minimum": RESIDENCY_MIN_DELTA,
            "resident_contrast_tasks": resident_tasks,
            "read_lower_tasks": read_lower_tasks,
            "lower_bucket_tasks": lower_bucket_tasks,
            "opposite_bucket_tasks": opposite_bucket_tasks,
            "gate": {
                "minimum_valid_tasks": 10,
                "minimum_resident_contrast_tasks": 8,
                "minimum_read_lower_tasks": 8,
                "minimum_lower_bucket_tasks": 4,
                "maximum_opposite_bucket_tasks": 1,
            },
        },
        "cpu": {
            "go": cpu_go,
            "prefix_valid_tasks": prefix_valid_tasks,
            "prefix_exceeds_final": prefix_violations,
            "helpful_changes": len(helpful),
            "helpful_tasks": helpful_tasks,
            "harmful_changes": len(harmful),
            "harmful_tasks": sorted({str(row["task_id"]) for row in harmful}),
            "gate": {
                "minimum_prefix_valid_tasks": 10,
                "minimum_helpful_changes": 3,
                "minimum_helpful_tasks": 3,
                "maximum_harmful_changes": 1,
            },
        },
        "cost": {
            "probe_ms_total": sum(probe_ms),
            "intervention_ms_total": sum(
                float(row["probe"]["intervention_ms"])
                for task in valid_tasks.values()
                for row in task
            ),
            "target_ms_total": sum(
                float(row["target_duration_ms"])
                for task in valid_tasks.values()
                for row in task
                if row["target_duration_ms"] is not None
            ),
        },
    }
    return result, rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--run-dir", type=Path, required=True)
    baseline.add_argument(
        "--public-telemetry", type=Path, action="append", required=True
    )
    baseline.add_argument("--exclude-public-repo", action="append", default=[])
    baseline.add_argument("--out", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--baseline", type=Path, required=True)
    score.add_argument("--telemetry", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--dump-rows", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "baseline":
        result = build_current_baseline(
            args.manifest,
            args.run_dir,
            args.public_telemetry,
            args.exclude_public_repo,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    result, rows = score_experiment(args.protocol, args.baseline, args.telemetry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.dump_rows.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.dump_rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
