#!/usr/bin/env python3
"""Evaluate frozen Zarr RSS predictions as CPU-idle backfill gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    PipExecEvent,
    _full_test_phases,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    _row_from_clause,
    command_resource_bucket_label,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    Row as PredictorRow,
)
from scripts.evaluation.evaluate_pennylane_pairwise_cpu_backfill import (  # noqa: E402
    _load_programs,
)
from scripts.evaluation.evaluate_semantic_work_units import (  # noqa: E402
    run as run_semantic_work_units,
)
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    ClauseResourceKB,
    _command_stages,
)
from tool_resource_eval.labels import repo_of  # noqa: E402
from tool_resource_eval.resource_admission import (  # noqa: E402
    AdmissionProgram,
    simulate_idle_backfill,
)

VERSION = "zarr-rss-backfill-v1"
SPLIT = _ROOT / "analysis/development/zarr-phase-survival-split.json"
SPLIT_SHA256 = "342e1aa17ba48b2e15af46512b144030f9e16f5f53e8d44e4763b7b63fd02a4e"
VALIDATION_COLLECTION_SCHEMA = "zarr-rss-validation-collection-v1"
CPU_CAPACITY = 8.0
RSS_CAPACITY_MB = 16_000.0
RSS_REQUESTS = (500.0, 2_000.0, RSS_CAPACITY_MB)
RSS_TARGET = "sampled_peak_rss_mb"
PHASE_TARGETS = ("latency", "peak_cpu_cores", RSS_TARGET)
MINIMUM_PHASE_FIT_TASKS = 5

_DEV_459 = _ROOT / (
    "traces/swe-rebench/gpt-5.6-sol/zarr-phase-survival-dev-replay-c1-20x-ebpf-20260811"
)
_DEV_OLD = _ROOT / (
    "traces/swe-rebench/qwen3.7-max/"
    "zarr-phase-survival-dev3-replay-c1-20x-ebpf-20260811"
)
_DEV_16 = _ROOT / (
    "traces/swe-rebench/gpt-5.6-sol/zarr-phase-survival-dev16-c1-fast-ebpf-20260811"
)


@dataclass(frozen=True)
class Dataset:
    task_ids: tuple[str, ...]
    programs: Mapping[str, AdmissionProgram]
    profiles: Mapping[str, tuple[tuple[float, float], ...]]
    clauses_by_task: Mapping[str, tuple[Any, ...]]
    commands_by_task: Mapping[str, tuple[CommandRow, ...]]
    events_by_task: Mapping[str, tuple[PipExecEvent, ...]]
    rss_source_by_command: Mapping[str, str]
    unverified_command_ids: frozenset[str]


def _development_traces(task_ids: Sequence[str]) -> dict[str, Path]:
    traces = {}
    for task_id in task_ids:
        base = (
            _DEV_459
            if task_id.endswith("-459")
            else _DEV_OLD
            if task_id.endswith(("-2244", "-2348"))
            else _DEV_16
        )
        traces[task_id] = base / task_id / "attempt_1/trace.jsonl"
    return traces


def _run_traces(run_dir: Path, expected: Sequence[str]) -> dict[str, Path]:
    contract = json.loads(
        (run_dir / "collection_contract.json").read_text(encoding="utf-8")
    )
    required_contract = {
        "schema": VALIDATION_COLLECTION_SCHEMA,
        "task_ids": list(expected),
        "model": "gpt-5.6-sol",
        "service_tier": "fast",
        "max_iterations": 100,
        "concurrency": 1,
        "required_ebpf": True,
        "cleanup_task_images": True,
    }
    if contract != required_contract:
        raise ValueError("validation collection contract differs from the freeze")
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [record.get("instance_id") for record in records] != list(expected):
        raise ValueError("validation results differ from the frozen task order")
    traces = {}
    for record in records:
        if record.get("success") is not True:
            raise ValueError("validation contains an unsuccessful final task")
        task_id = str(record["instance_id"])
        attempt = Path(str(record["attempt_dir"]))
        if not attempt.is_absolute():
            attempt = run_dir / attempt
        attempt = attempt.resolve()
        if not attempt.is_relative_to(run_dir.resolve()):
            raise ValueError("validation attempt escapes the run directory")
        trace = attempt / "trace.jsonl"
        declared_trace = Path(str(record.get("trace_file"))).resolve()
        if declared_trace != trace:
            raise ValueError("validation result trace path differs from its attempt")
        with trace.open(encoding="utf-8") as handle:
            metadata = json.loads(handle.readline())
        if (
            metadata.get("type") != "trace_metadata"
            or metadata.get("instance_id") != task_id
            or metadata.get("mode") != "collect"
            or metadata.get("model") != required_contract["model"]
            or metadata.get("api_base") != "https://chatgpt.com/backend-api/codex"
            or metadata.get("max_iterations") != required_contract["max_iterations"]
            or metadata.get("run_config", {}).get("generation", {}).get("service_tier")
            != required_contract["service_tier"]
            or metadata.get("tool_resource", {}).get("service_enabled") is not True
            or metadata.get("collection_validity") != "valid"
            or metadata.get("telemetry_quality") != "ok"
        ):
            raise ValueError(
                "validation trace metadata differs from the collection contract"
            )
        traces[task_id] = trace
    return traces


def _load_split() -> dict[str, Any]:
    raw = SPLIT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_SHA256:
        raise ValueError("Zarr split differs from the frozen manifest")
    split = json.loads(raw)
    partitions = [
        split.get(name, ()) for name in ("development", "validation", "final")
    ]
    if (
        [len(values) for values in partitions] != [20, 10, 12]
        or split.get("model_fit_exclusions") != ["zarr-developers__zarr-python-2668"]
        or len(set().union(*map(set, partitions))) != 42
    ):
        raise ValueError("Zarr split partition contract differs")
    return split


def _composed_rss_source(call: Mapping[str, Any]) -> tuple[float, str]:
    clauses = [value for value in call.get("clauses", ()) if isinstance(value, Mapping)]
    stages = _command_stages(
        [
            {
                "in_pipe": clause.get("in_pipe"),
                "in_subst": clause.get("in_subst"),
                "pipeline_position": clause.get("pipeline_position"),
            }
            for clause in clauses
        ]
    )
    values: list[float] = []
    imputed = False
    for clause in clauses:
        rss = clause.get(RSS_TARGET)
        latency = clause.get("latency_ms")
        if (
            isinstance(rss, (int, float))
            and not isinstance(rss, bool)
            and math.isfinite(rss)
            and rss >= 0.0
        ):
            values.append(float(rss))
        elif (
            RSS_TARGET in clause
            and rss is None
            and isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(latency)
            and 0.0 <= float(latency) < 500.0
        ):
            values.append(RSS_REQUESTS[0])
            imputed = True
        else:
            return RSS_CAPACITY_MB, "full_fallback"
    if not stages or len(values) != len(clauses):
        return RSS_CAPACITY_MB, "full_fallback"
    bound = max(sum(values[index] for index in stage) for stage in stages)
    return min(RSS_CAPACITY_MB, bound), (
        "clause_short_null_upper" if imputed else "observed_clause_composition"
    )


def _validated_rss_source(call: Mapping[str, Any]) -> tuple[float, str]:
    if call.get("eligible_for_kb") is not True or call.get("invalid_reasons"):
        return RSS_CAPACITY_MB, "full_fallback"
    return _composed_rss_source(call)


def _load_dataset(task_ids: Sequence[str], traces: Mapping[str, Path]) -> Dataset:
    if list(traces) != list(task_ids):
        raise ValueError("trace order differs from the requested task order")
    for path in traces.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    items = [
        {"task_id": task_id, "trace": str(traces[task_id])} for task_id in task_ids
    ]
    programs, profiles, texts, _means, _safe, excluded, _sources = _load_programs(
        {"replay": items}
    )
    if excluded or set(programs) != set(task_ids):
        raise ValueError("dataset contains an invalid or missing task")
    physical = {
        command.command_id: command
        for program in programs.values()
        for command in program.commands
    }
    clauses_by_task: dict[str, list[Any]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    events_by_task: dict[str, tuple[PipExecEvent, ...]] = {}
    source_by_command: dict[str, str] = {}
    rss_by_command: dict[str, float] = {}
    for manifest_index, task_id in enumerate(task_ids):
        artifact = json.loads(
            (traces[task_id].parent / "resource_observations.json").read_text(
                encoding="utf-8"
            )
        )
        events: list[PipExecEvent] = []
        for call_index, call in enumerate(artifact.get("calls", ())):
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("tool_call_id")
            command = call.get("command")
            if not isinstance(call_id, str) or not isinstance(command, str):
                continue
            command_id = f"{task_id}:{call_id}"
            if command_id not in physical:
                continue
            if texts.get(command_id) != command:
                raise ValueError("resource artifact command differs from the trace")
            events.append(PipExecEvent(call_id, command, ""))
            rss, source = _validated_rss_source(call)
            rss_by_command[command_id] = rss
            source_by_command[command_id] = source
            if call.get("eligible_for_kb") is not True:
                continue
            retained = tuple(
                row
                for clause in call.get("clauses", ())
                if isinstance(clause, Mapping)
                and (row := _row_from_clause(task_id, manifest_index, clause))
                is not None
            )
            clauses_by_task[task_id].extend(
                row for row in retained if row.pipeline_position <= 0
            )
            if retained:
                commands_by_task[task_id].append(
                    CommandRow(
                        task_id,
                        repo_of(task_id),
                        manifest_index,
                        call_index,
                        call_id,
                        command,
                        physical[command_id].duration_s * 1_000.0,
                        retained,
                    )
                )
        events_by_task[task_id] = tuple(events)
    if set(rss_by_command) != set(physical):
        raise ValueError("resource artifact differs from the exec trajectory")
    adjusted = {
        task_id: replace(
            program,
            commands=tuple(
                replace(command, rss_mb=rss_by_command[command.command_id])
                for command in program.commands
            ),
        )
        for task_id, program in programs.items()
    }
    return Dataset(
        tuple(task_ids),
        adjusted,
        profiles,
        {task_id: tuple(clauses_by_task[task_id]) for task_id in task_ids},
        {task_id: tuple(commands_by_task[task_id]) for task_id in task_ids},
        events_by_task,
        source_by_command,
        frozenset(
            command_id
            for command_id, source in source_by_command.items()
            if source != "observed_clause_composition"
        ),
    )


def _labels(row: CommandRow) -> dict[str, int | None]:
    return {
        "latency": CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms),
        **{
            target: command_resource_bucket_label(row, target)[0]
            for target in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }


def _fit_row(row: CommandRow) -> PredictorRow:
    unavailable = {
        "latency": None,
        **{target: None for target in CANONICAL_RESOURCE_BUCKET_EDGES},
    }
    return PredictorRow(
        f"{row.task_id}:{row.call_id}",
        row.task_id,
        row.command,
        _labels(row),
        unavailable,
        unavailable,
    )


def _phase_pmf(
    dataset: Dataset, fit_task_ids: Sequence[str]
) -> tuple[float, float, float]:
    values = []
    support = set()
    for task_id in fit_task_ids:
        phases = _full_test_phases(dataset.events_by_task[task_id])
        for row in dataset.commands_by_task[task_id]:
            label = _labels(row)[RSS_TARGET]
            if phases.get(row.call_id) == 2 and label is not None:
                values.append(label)
                support.add(task_id)
    if len(support) < MINIMUM_PHASE_FIT_TASKS:
        raise ValueError("full-test phase lacks five fit tasks")
    return tuple(values.count(bucket) / len(values) for bucket in range(3))  # type: ignore[return-value]


def _hard_index(value: Any) -> int | None:
    return None if value is None else RESOURCE_BUCKET_LABELS.index(str(value))


def _upper_index(pmf: Sequence[float] | None) -> int | None:
    return (
        None
        if pmf is None
        else max(index for index, value in enumerate(pmf) if value > 0.0)
    )


def _phase_raises(phase: int, semantic: int | None, current: int | None) -> bool:
    return (
        semantic is not None
        and current is not None
        and phase > semantic
        and phase > current
    )


def _predictions(
    fit: Dataset, target: Dataset, *, leave_one_out: bool
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    command_ids = set(target.profiles)
    reservations = {
        arm: {command_id: RSS_CAPACITY_MB for command_id in command_ids}
        for arm in ("majority", "clause_kb", "task_aware", "task_aware_upper")
    }
    scored: dict[str, list[tuple[int | None, int | None]]] = defaultdict(list)
    provenance: Counter[str] = Counter()
    phase_applied = 0
    for target_task in target.task_ids:
        fit_task_ids = tuple(
            task_id
            for task_id in fit.task_ids
            if not leave_one_out or task_id != target_task
        )
        fit_rows = [
            _fit_row(row)
            for task_id in fit_task_ids
            for row in fit.commands_by_task[task_id]
        ]
        fit_labels = [
            int(row.labels[RSS_TARGET])
            for row in fit_rows
            if row.labels[RSS_TARGET] is not None
        ]
        majority = min(range(3), key=lambda bucket: (-fit_labels.count(bucket), bucket))
        kb = ClauseResourceKB()
        for task_id in fit_task_ids:
            for row in fit.clauses_by_task[task_id]:
                kb.observe_completed_clause(row.observation(0.0, 1.0))
        validation_rows = []
        for row in target.commands_by_task[target_task]:
            predictions = kb.predict_command_resource_buckets(
                row.repo, row.command, 3.0
            ).classifications
            current = {"latency": None}
            pmfs: dict[str, tuple[float, ...] | None] = {"latency": None}
            for resource in CANONICAL_RESOURCE_BUCKET_EDGES:
                prediction = predictions.get(resource)
                current[resource] = None if prediction is None else prediction.label
                pmfs[resource] = (
                    None if prediction is None else prediction.probability_by_bucket
                )
            validation_rows.append(
                PredictorRow(
                    f"{row.task_id}:{row.call_id}",
                    row.task_id,
                    row.command,
                    _labels(row),
                    current,
                    pmfs,
                )
            )
        _semantic_result, semantic_rows = run_semantic_work_units(
            fit_rows, validation_rows
        )
        semantic_by_id = {str(row["sample_id"]): row for row in semantic_rows}
        phase_pmf = _phase_pmf(fit, fit_task_ids)
        phase_hard = max(range(3), key=phase_pmf.__getitem__)
        target_phases = _full_test_phases(target.events_by_task[target_task])
        for row in validation_rows:
            command_id = row.sample_id
            truth = row.labels[RSS_TARGET]
            reservations["majority"][command_id] = RSS_REQUESTS[majority]
            base_index = _hard_index(row.current[RSS_TARGET])
            if base_index is not None:
                reservations["clause_kb"][command_id] = RSS_REQUESTS[base_index]
            semantic = semantic_by_id[command_id]["arms"]["semantic_work_units"]
            hard = _hard_index(semantic["candidate"][RSS_TARGET])
            pmf = semantic["candidate_probability_by_bucket"][RSS_TARGET]
            source = str(semantic["provenance"][RSS_TARGET]["source"])
            call_id = command_id.split(":", 1)[1]
            if target_phases.get(call_id) == 2 and _phase_raises(
                phase_hard, hard, base_index
            ):
                hard = phase_hard
                pmf = phase_pmf
                source = "full_test_phase_monotone"
                phase_applied += 1
            upper = _upper_index(pmf)
            if hard is not None:
                reservations["task_aware"][command_id] = RSS_REQUESTS[hard]
            if upper is not None:
                reservations["task_aware_upper"][command_id] = RSS_REQUESTS[upper]
            provenance[source] += 1
            scored["majority"].append((truth, majority))
            scored["clause_kb"].append((truth, base_index))
            scored["task_aware"].append((truth, hard))
            scored["task_aware_upper"].append((truth, upper))
    return reservations, {
        "rows": sum(len(rows) for rows in target.commands_by_task.values()),
        "phase_applied": phase_applied,
        "task_aware_provenance": dict(sorted(provenance.items())),
        "metrics": {arm: _prediction_metrics(rows) for arm, rows in scored.items()},
    }


def _prediction_metrics(
    rows: Sequence[tuple[int | None, int | None]],
) -> dict[str, Any]:
    eligible = [(truth, prediction) for truth, prediction in rows if truth is not None]
    correct = sum(truth == prediction for truth, prediction in eligible)
    unavailable = sum(prediction is None for _truth, prediction in eligible)
    return {
        "eligible": len(eligible),
        "accuracy": correct / len(eligible) if eligible else None,
        "prediction_unavailable": unavailable,
        "confusion_label_by_prediction": [
            [
                sum(
                    truth == label and prediction == predicted
                    for truth, prediction in eligible
                )
                for predicted in range(3)
            ]
            for label in range(3)
        ],
    }


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    starts = list(metrics["speculative_start_ids"])
    return {
        key: metrics[key]
        for key in (
            "command_count",
            "recorded_command_service_s",
            "total_command_service_s",
            "added_service_s",
            "makespan_s",
            "mean_task_completion_s",
            "total_command_queue_s",
            "max_concurrent_commands",
            "normal_starts",
            "speculative_starts",
            "speculative_completions",
            "promotions",
            "modeled_capacity_exposure_events",
            "rss_unverified_overlap_events",
            "capacity_violation",
            "physical_capacity_violation",
            "total_cpu_work_core_s",
            "served_cpu_work_core_s",
        )
    } | {
        "service_inflation": metrics["total_command_service_s"]
        / metrics["recorded_command_service_s"]
        - 1.0,
        "speculative_task_count": len(
            {command_id.split(":", 1)[0] for command_id in starts}
        ),
        "speculative_start_ids": starts,
        "modeled_capacity_exposure_command_ids": list(
            metrics["modeled_capacity_exposure_command_ids"]
        ),
        "rss_unverified_overlap_command_ids": list(
            metrics["rss_unverified_overlap_command_ids"]
        ),
    }


def _comparison(
    serial: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "mean_task_completion_reduction": 1.0
        - candidate["mean_task_completion_s"] / serial["mean_task_completion_s"],
        "makespan_reduction": 1.0 - candidate["makespan_s"] / serial["makespan_s"],
        "service_inflation": candidate["service_inflation"],
    }


def evaluate(role: str, validation_run: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    split = _load_split()
    fit_ids = tuple(
        task_id
        for task_id in split["development"]
        if task_id not in split["model_fit_exclusions"]
    )
    fit = _load_dataset(fit_ids, _development_traces(fit_ids))
    if role == "development":
        target_traces = _development_traces(fit_ids)
        target = fit
        leave_one_out = True
    elif role == "validation" and validation_run is not None:
        validation_ids = tuple(split["validation"])
        target_traces = _run_traces(validation_run.resolve(), validation_ids)
        target = _load_dataset(validation_ids, target_traces)
        leave_one_out = False
    else:
        raise ValueError("validation requires its frozen run directory")
    reservations, prediction = _predictions(fit, target, leave_one_out=leave_one_out)
    programs = [target.programs[task_id] for task_id in sorted(target.programs)]
    command_ids = set(target.profiles)
    known_source = {
        command_id
        for command_id, source in target.rss_source_by_command.items()
        if source != "full_fallback"
    }
    common = {
        "cpu_capacity": CPU_CAPACITY,
        "rss_capacity_mb": RSS_CAPACITY_MB,
        "cpu_work_profiles": target.profiles,
        "rss_unverified_command_ids": set(target.unverified_command_ids),
        "selection": "fcfs",
    }
    raw_arms = {
        "serial8": simulate_idle_backfill(
            programs,
            cpu_capacity=CPU_CAPACITY,
            rss_capacity_mb=RSS_CAPACITY_MB,
            cpu_work_profiles=target.profiles,
            speculative_eligible_command_ids=set(),
            selection="serial",
        ),
        "source_bound_oracle": simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=known_source,
            **common,
        ),
        **{
            arm: simulate_idle_backfill(
                programs,
                speculative_eligible_command_ids=command_ids,
                rss_reservations=values,
                **common,
            )
            for arm, values in reservations.items()
        },
    }
    phase_identity = None
    if role == "development":
        phase = simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=known_source,
            require_pairwise_profile_compatibility=True,
            **common,
        )
        phase_identity = {
            "same_speculative_start_set": set(phase["speculative_start_ids"])
            == set(raw_arms["source_bound_oracle"]["speculative_start_ids"]),
            "rss_only": _comparison(
                _compact(raw_arms["serial8"]),
                _compact(raw_arms["source_bound_oracle"]),
            ),
            "full_profile": _comparison(_compact(raw_arms["serial8"]), _compact(phase)),
        }
    arms = {name: _compact(metrics) for name, metrics in raw_arms.items()}
    serial = arms["serial8"]
    comparisons = {
        name: _comparison(serial, metrics)
        for name, metrics in arms.items()
        if name != "serial8"
    }
    task = comparisons["task_aware"]
    clause = comparisons["clause_kb"]
    oracle = comparisons["source_bound_oracle"]
    task_metrics = arms["task_aware"]
    work_ok = math.isclose(
        task_metrics["total_cpu_work_core_s"],
        task_metrics["served_cpu_work_core_s"],
        rel_tol=1e-12,
        abs_tol=1e-7,
    )
    gate = {
        "mean_completion_reduction_at_least_5_percent": task[
            "mean_task_completion_reduction"
        ]
        >= 0.05,
        "captures_half_source_bound_oracle": oracle["mean_task_completion_reduction"]
        > 0.0
        and task["mean_task_completion_reduction"]
        >= 0.5 * oracle["mean_task_completion_reduction"],
        "beats_clause_kb_by_1_percentage_point": task["mean_task_completion_reduction"]
        - clause["mean_task_completion_reduction"]
        >= 0.01,
        "makespan_lower_than_serial": task["makespan_reduction"] > 0.0,
        "service_inflation_at_most_5_percent": task["service_inflation"] <= 0.05,
        "zero_source_bound_or_capacity_violations": not (
            task_metrics["modeled_capacity_exposure_events"]
            or task_metrics["capacity_violation"]
            or task_metrics["physical_capacity_violation"]
        ),
        "cpu_work_conserved": work_ok,
        "at_least_20_speculative_starts": task_metrics["speculative_starts"] >= 20,
        "at_least_5_speculative_tasks": task_metrics["speculative_task_count"] >= 5,
    }
    gate["go"] = all(gate.values())
    source_counts = Counter(target.rss_source_by_command.values())
    return {
        "schema": VERSION,
        "status": (
            "development_exposed_validation_ready"
            if role == "development"
            else "validation_go"
            if gate["go"]
            else "validation_no_go"
        ),
        "claim_bearing": False,
        "role": role,
        "protocol": {
            "fit": "19 fixed Zarr development tasks",
            "development_scoring": "leave-one-task-out",
            "validation_update": "none",
            "arrival": "one concurrent start wave; task-id FCFS tie break",
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "rss_requests_mb": list(RSS_REQUESTS),
            "short_null_policy": "clause latency <500 ms contributes 500 MB upper bucket",
            "unknown_or_unavailable": "full 16000 MB reservation",
            "primary": "Task-Aware hard RSS mean task completion",
        },
        "tasks": {
            "fit": list(fit.task_ids),
            "scored": list(target.task_ids),
            "final_read": False,
        },
        "evidence": {
            "commands": len(target.profiles),
            "eligible_prediction_rows": prediction["rows"],
            "rss_source_counts": dict(sorted(source_counts.items())),
            "unverified_source_commands": len(target.unverified_command_ids),
            "phase_diagnosis": phase_identity,
        },
        "prediction": prediction,
        "arms": arms,
        "comparisons_vs_serial8": comparisons,
        "task_aware_minus_clause_kb_mean_reduction_percentage_points": 100.0
        * (
            task["mean_task_completion_reduction"]
            - clause["mean_task_completion_reduction"]
        ),
        "gate": gate,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "inputs": {
            "split": str(SPLIT.resolve()),
            "split_sha256": SPLIT_SHA256,
            "validation_run": (
                None if validation_run is None else str(validation_run.resolve())
            ),
            "scored_traces": {
                task_id: str(target_traces[task_id]) for task_id in target.task_ids
            },
        },
        "limitations": [
            "Development outcomes selected the validation candidate and gate; only validation is untouched.",
            "Short-null RSS is a frozen policy upper bucket, not a measured physical peak.",
            "The replay models CPU sharing from isolated profiles and does not execute concurrent containers.",
            "This is one Zarr task wave and does not establish cross-repository generality.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--validation-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output already exists")
    if args.role == "development" and args.validation_run is not None:
        raise ValueError("development does not accept a validation run")
    if args.role == "validation":
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ValueError("validation requires a clean committed checkout")
    result = evaluate(args.role, args.validation_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
