#!/usr/bin/env python3
"""Evaluate clause-level CPU/RSS/Disk Heavy/Light predictions on replay traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from trace_collect.attempt_layout import build_tool_calls_from_trace  # noqa: E402
from tool_resource_eval.labels import repo_of  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_BUCKET_EDGES,
    CANONICAL_RESOURCE_HEAVY_THRESHOLDS,
    DEFAULT_HEAVY_DECISION_THRESHOLD,
    SHRINKAGE_ALPHA_GRID,
    SHORT_NULL_LIGHT_MAX_LATENCY_MS,
    STRUCTURED_ARGV_REPRESENTATION,
    RESOURCE_BUCKET_LABELS,
    ClauseObservation,
    ClauseResourceKB,
    _command_stages,
    parse_command_clauses,
)

_RESOURCE_FIELDS = {
    "peak_cpu_cores": "peak_cpu_cores",
    "sampled_peak_rss_mb": "sampled_peak_rss_mb",
    "disk_read_write_bytes_total": "disk_read_write_bytes_total",
}


@dataclass(frozen=True)
class Row:
    task_id: str
    repo: str
    manifest_index: int
    bin: str
    argv: tuple[str, ...]
    latency_ms: float
    peak_cpu_cores: float | None
    sampled_peak_rss_mb: float | None
    disk_read_write_bytes_total: float | None
    in_loop: bool = False
    in_pipe: bool = False
    in_subst: bool = False
    pipeline_position: int = -1
    structure_known: bool = True

    def observation(self, ts_start: float, ts_end: float) -> ClauseObservation:
        return ClauseObservation(
            repo=self.repo,
            bin=self.bin,
            argv=self.argv,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms=self.latency_ms,
            peak_cpu_cores=self.peak_cpu_cores,
            sampled_peak_rss_mb=self.sampled_peak_rss_mb,
            disk_read_write_bytes_total=self.disk_read_write_bytes_total,
            impute_short_null_resources_as_light=True,
            in_loop=self.in_loop,
            in_pipe=self.in_pipe,
            in_subst=self.in_subst,
            pipeline_position=self.pipeline_position,
        )


@dataclass(frozen=True)
class CommandRow:
    task_id: str
    repo: str
    manifest_index: int
    call_index: int
    call_id: str
    command: str
    duration_ms: float
    clauses: tuple[Row, ...]


@dataclass(frozen=True)
class CandidateSSelection:
    alpha: float
    latency_result_path: str
    fit_path: str
    eval_path: str
    fit_row_count: int
    eval_row_count: int

    def __post_init__(self) -> None:
        if self.alpha not in SHRINKAGE_ALPHA_GRID:
            raise ValueError(f"shrinkage alpha must be one of {SHRINKAGE_ALPHA_GRID}")


def load_candidate_s_selection(
    latency_result_path: Path,
    *,
    fit_path: Path,
    eval_path: Path,
    fit_row_count: int,
    eval_row_count: int,
) -> CandidateSSelection:
    result = json.loads(latency_result_path.read_text(encoding="utf-8"))
    selection = result.get("selection", {}).get("candidate_s_alpha")
    provenance = result.get("provenance")
    if not isinstance(selection, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("latency result has no Candidate S fit selection")
    alpha = selection.get("selected_alpha")
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or float(alpha) not in SHRINKAGE_ALPHA_GRID
    ):
        raise ValueError("latency result selected alpha is outside the fixed grid")
    if selection.get("alpha_grid") != list(SHRINKAGE_ALPHA_GRID):
        raise ValueError("latency result alpha grid differs from the fixed grid")
    if (
        selection.get("selection_target") != "exact_latency_class_accuracy"
        or selection.get("tie_break") != "larger_alpha"
        or selection.get("outer_labels_used") is not False
    ):
        raise ValueError("latency result Candidate S selection contract differs")
    if result.get("bucket_edges_ms") != list(CANONICAL_LATENCY_BUCKETS.edges_ms):
        raise ValueError("latency result bucket edges differ from the canonical objective")
    if provenance.get("candidate_s", {}).get("enabled") is not True:
        raise ValueError("latency result does not declare Candidate S enabled")
    resolved_fit = fit_path.resolve()
    resolved_eval = eval_path.resolve()
    if Path(str(provenance.get("fit_telemetry"))).resolve() != resolved_fit:
        raise ValueError("latency result fit input differs from resource fit input")
    if Path(str(provenance.get("eval_telemetry"))).resolve() != resolved_eval:
        raise ValueError("latency result eval input differs from resource eval input")
    if (
        result.get("fit_clause_observation_count") != fit_row_count
        or selection.get("fit_row_count") != fit_row_count
        or result.get("eval_clause_observation_count") != eval_row_count
    ):
        raise ValueError("latency and resource result row counts differ")
    if result.get("row_identity", {}).get("identical_row_ids_and_labels") is not True:
        raise ValueError("latency result did not reconcile outer rows and labels")
    return CandidateSSelection(
        alpha=float(alpha),
        latency_result_path=str(latency_result_path.resolve()),
        fit_path=str(resolved_fit),
        eval_path=str(resolved_eval),
        fit_row_count=fit_row_count,
        eval_row_count=eval_row_count,
    )


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _row_from_clause(
    task_id: str,
    manifest_index: int,
    clause: Mapping[str, Any],
    structure: Mapping[str, Any] | None = None,
) -> Row | None:
    if clause.get("eligible_for_kb") is not True:
        return None
    latency = clause.get("latency_ms")
    argv = clause.get("argv")
    if latency is None or not isinstance(argv, list) or not argv:
        return None
    disk = clause.get("disk_io")
    disk_total = (
        disk.get("read_write_bytes_total") if isinstance(disk, Mapping) else None
    )
    shape = clause if structure is None else structure
    return Row(
        task_id=task_id,
        repo=repo_of(task_id),
        manifest_index=manifest_index,
        bin=str(clause["bin"]),
        argv=tuple(str(value) for value in argv),
        latency_ms=float(latency),
        peak_cpu_cores=_number(clause.get("peak_cpu_cores")),
        sampled_peak_rss_mb=_number(clause.get("sampled_peak_rss_mb")),
        disk_read_write_bytes_total=_number(disk_total),
        in_loop=shape.get("in_loop") is True,
        in_pipe=shape.get("in_pipe") is True,
        in_subst=shape.get("in_subst") is True,
        pipeline_position=int(shape.get("pipeline_position", -1)),
        structure_known=structure is not None
        or all(
            key in clause
            for key in ("in_loop", "in_pipe", "in_subst", "pipeline_position")
        ),
    )


def _aligned_structures(
    telemetry: Mapping[str, Any],
) -> list[Mapping[str, Any] | None]:
    observed = telemetry.get("clauses", [])
    static = telemetry.get("static_word_intent")
    parsed = parse_command_clauses(str(telemetry.get("command", "")))
    structures = parsed["clauses"]
    identity = lambda row: (  # noqa: E731 - local matching key
        str(row.get("bin")),
        tuple(map(str, row.get("argv", []))),
    )
    if (
        not isinstance(observed, list)
        or not isinstance(static, list)
        or parsed["parse_failed"] is not False
        or len(static) != len(structures)
        or any(identity(left) != identity(right) for left, right in zip(static, structures))
    ):
        return [None] * len(observed)

    def greedy(indices: range) -> list[int] | None:
        matches: list[int] = []
        remaining = iter(indices)
        for clause in observed[:: indices.step]:
            match = next((index for index in remaining if identity(static[index]) == identity(clause)), None)
            if match is None:
                return None
            matches.append(match)
        return matches[:: indices.step]

    earliest = greedy(range(len(static)))
    latest = greedy(range(len(static) - 1, -1, -1))
    if earliest is None or latest is None:
        return [None] * len(observed)
    return [
        structures[left] if left == right else None
        for left, right in zip(earliest, latest, strict=True)
    ]


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            data = record.get("data")
            if not isinstance(data, dict):
                continue
            telemetry = data.get("clause_telemetry")
            if (
                not isinstance(telemetry, dict)
                or telemetry.get("eligible_for_kb") is not True
            ):
                continue
            task_id = data.get("task_instance_id")
            manifest_index = data.get("manifest_index")
            if not isinstance(task_id, str) or not isinstance(manifest_index, int):
                raise ValueError(f"{path}: clause telemetry row lacks task identity")
            clauses = telemetry.get("clauses", [])
            for clause, structure in zip(
                clauses, _aligned_structures(telemetry), strict=True
            ):
                if not isinstance(clause, Mapping):
                    continue
                row = _row_from_clause(task_id, manifest_index, clause, structure)
                if row is not None:
                    rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no eligible clause telemetry rows")
    return rows


def load_run_rows(
    run_dir: Path,
    *,
    results_path: Path | None = None,
) -> tuple[list[str], list[Row], list[CommandRow]]:
    """Load only successful final attempts referenced by a collector run."""

    run_dir = run_dir.resolve()
    results_path = run_dir / "results.jsonl" if results_path is None else results_path
    task_ids: list[str] = []
    rows: list[Row] = []
    commands: list[CommandRow] = []
    seen: set[str] = set()
    with results_path.open(encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            record = json.loads(line)
            task_id = record.get("instance_id")
            attempt_value = record.get("attempt_dir")
            if (
                not isinstance(task_id, str)
                or not isinstance(attempt_value, str)
                or record.get("success") is not True
            ):
                raise ValueError(
                    f"{results_path}: result {manifest_index} is not a successful "
                    "final attempt"
                )
            if task_id in seen:
                raise ValueError(f"{results_path}: duplicate task {task_id}")
            seen.add(task_id)
            attempt_dir = Path(attempt_value)
            if not attempt_dir.is_absolute():
                attempt_dir = run_dir / attempt_dir
            attempt_dir = attempt_dir.resolve()
            if not attempt_dir.is_relative_to(run_dir) or attempt_dir.parent.name != task_id:
                raise ValueError(
                    f"{results_path}: attempt path does not belong to task {task_id}"
                )
            artifact_path = attempt_dir / "resource_observations.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            required_status = {
                "collection_validity": "valid",
                "workload_execution": "completed",
                "telemetry_quality": "ok",
                "cleanup": "ok",
            }
            mismatches = {
                key: artifact.get(key)
                for key, expected in required_status.items()
                if artifact.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"{artifact_path}: final telemetry is not evidence-valid: "
                    f"{mismatches}"
                )
            tool_calls_path = attempt_dir / "tool_calls.json"
            tool_calls = (
                json.loads(tool_calls_path.read_text(encoding="utf-8"))
                if tool_calls_path.exists()
                else build_tool_calls_from_trace(attempt_dir / "trace.jsonl")
            )
            if not isinstance(tool_calls, list):
                raise ValueError(f"{tool_calls_path}: expected a JSON array")
            exec_calls: dict[str, Mapping[str, Any]] = {}
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping) or tool_call.get("tool") != "exec":
                    continue
                tool_id = tool_call.get("id")
                if not isinstance(tool_id, str) or "call_" not in tool_id:
                    raise ValueError(f"{tool_calls_path}: exec call lacks an id")
                call_id = tool_id[tool_id.index("call_") :]
                if call_id in exec_calls:
                    raise ValueError(f"{tool_calls_path}: duplicate exec call {call_id}")
                exec_calls[call_id] = tool_call
            task_ids.append(task_id)
            for call_index, call in enumerate(artifact.get("calls", [])):
                if not isinstance(call, Mapping) or call.get("eligible_for_kb") is not True:
                    continue
                call_id = call.get("tool_call_id")
                command = call.get("command")
                if not isinstance(call_id, str) or not isinstance(command, str):
                    raise ValueError(f"{artifact_path}: eligible call lacks identity")
                tool_call = exec_calls.get(call_id)
                tool_input = None if tool_call is None else tool_call.get("input")
                duration_ms = None if tool_call is None else tool_call.get("duration_ms")
                if (
                    not isinstance(tool_input, Mapping)
                    or tool_input.get("command") != command
                    or not isinstance(duration_ms, (int, float))
                    or isinstance(duration_ms, bool)
                    or not math.isfinite(duration_ms)
                    or duration_ms < 0.0
                ):
                    raise ValueError(
                        f"{artifact_path}: eligible call {call_id} has no matching "
                        "tool-call duration"
                    )
                call_rows: list[Row] = []
                for clause in call.get("clauses", []):
                    if not isinstance(clause, Mapping):
                        continue
                    row = _row_from_clause(task_id, manifest_index, clause)
                    if row is not None:
                        call_rows.append(row)
                        if row.pipeline_position <= 0:
                            rows.append(row)
                if not call_rows:
                    raise ValueError(
                        f"{artifact_path}: eligible call {call_id} has no eligible clauses"
                    )
                commands.append(
                    CommandRow(
                        task_id=task_id,
                        repo=repo_of(task_id),
                        manifest_index=manifest_index,
                        call_index=call_index,
                        call_id=call_id,
                        command=command,
                        duration_ms=float(duration_ms),
                        clauses=tuple(call_rows),
                    )
                )
    if not task_ids:
        raise ValueError(f"{results_path}: no successful final tasks")
    if not rows:
        raise ValueError(f"{run_dir}: no eligible clause telemetry rows")
    if not commands:
        raise ValueError(f"{run_dir}: no eligible command rows")
    return task_ids, rows, commands


def _label(row: Row, resource: str) -> tuple[bool | None, str]:
    value = getattr(row, _RESOURCE_FIELDS[resource])
    if value is not None:
        heavy = value > CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource]
        return heavy, "observed_heavy" if heavy else "observed_light"
    if row.latency_ms < SHORT_NULL_LIGHT_MAX_LATENCY_MS:
        return False, "short_null_imputed_light"
    return None, "null_unavailable"


def command_resource_label(
    row: CommandRow,
    resource: str,
) -> tuple[bool | None, str]:
    """Return only command labels proven by the retained clause aggregates."""

    threshold = CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource]
    field = _RESOURCE_FIELDS[resource]
    bounds: list[tuple[float, float]] = []
    saw_short_null = False
    saw_long_null = False
    for clause in row.clauses:
        value = getattr(clause, field)
        if value is not None:
            bounds.append((value, value))
        elif clause.latency_ms < SHORT_NULL_LIGHT_MAX_LATENCY_MS:
            bounds.append((0.0, threshold))
            saw_short_null = True
        else:
            bounds.append((0.0, math.inf))
            saw_long_null = True
    stages = _command_stages(
        [
            {
                "in_pipe": clause.in_pipe,
                "in_subst": clause.in_subst,
                "pipeline_position": clause.pipeline_position,
            }
            for clause in row.clauses
        ]
    )
    if stages is None:
        return None, "composition_unavailable"
    if resource == "disk_read_write_bytes_total":
        lower = sum(bound[0] for bound in bounds)
        upper = sum(bound[1] for bound in bounds)
    else:
        lower = max(max(bounds[index][0] for index in stage) for stage in stages)
        upper = max(sum(bounds[index][1] for index in stage) for stage in stages)
    if lower > threshold:
        return True, "observed_composed_heavy"
    if upper <= threshold:
        source = (
            "short_null_composed_light"
            if saw_short_null
            else "observed_composed_light"
        )
        return False, source
    if saw_long_null:
        return None, "null_unavailable"
    return None, "composition_ambiguous"


def command_resource_bucket_label(
    row: CommandRow,
    resource: str,
) -> tuple[int | None, str]:
    """Return a three-class command label only when its interval is unambiguous."""

    edges = CANONICAL_RESOURCE_BUCKET_EDGES[resource]
    field = _RESOURCE_FIELDS[resource]
    bounds: list[tuple[float, float]] = []
    saw_short_null = False
    saw_long_null = False
    for clause in row.clauses:
        value = getattr(clause, field)
        if value is not None:
            bounds.append((value, value))
        elif clause.latency_ms < SHORT_NULL_LIGHT_MAX_LATENCY_MS:
            bounds.append((0.0, edges[0]))
            saw_short_null = True
        else:
            bounds.append((0.0, math.inf))
            saw_long_null = True
    stages = _command_stages(
        [
            {
                "in_pipe": clause.in_pipe,
                "in_subst": clause.in_subst,
                "pipeline_position": clause.pipeline_position,
            }
            for clause in row.clauses
        ]
    )
    if stages is None:
        return None, "composition_unavailable"
    if resource == "disk_read_write_bytes_total":
        lower = sum(bound[0] for bound in bounds)
        upper = sum(bound[1] for bound in bounds)
    else:
        lower = max(max(bounds[index][0] for index in stage) for stage in stages)
        upper = max(sum(bounds[index][1] for index in stage) for stage in stages)
    lower_bucket = bisect_left(edges, lower)
    upper_bucket = len(edges) if math.isinf(upper) else bisect_left(edges, upper)
    if lower_bucket == upper_bucket:
        label = RESOURCE_BUCKET_LABELS[lower_bucket]
        source = (
            "short_null_composed_low"
            if saw_short_null and lower_bucket == 0
            else f"observed_composed_{label}"
        )
        return lower_bucket, source
    if saw_long_null:
        return None, "null_unavailable"
    return None, "composition_ambiguous"


def _empty_confusion() -> dict[str, Any]:
    return {
        "provenance_counts": Counter(),
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "prediction_unavailable": 0,
    }


def _finalize_resource_metric(
    raw: dict[str, Any],
    label_source_counts: Counter[str],
) -> dict[str, Any]:
    tp, tn, fp, fn = (raw[key] for key in ("tp", "tn", "fp", "fn"))
    predicted_n = tp + tn + fp + fn
    label_eligible_n = sum(
        count
        for source, count in label_source_counts.items()
        if source
        not in {"null_unavailable", "composition_ambiguous", "composition_unavailable"}
    )
    if predicted_n + raw["prediction_unavailable"] != label_eligible_n:
        raise AssertionError("resource confusion matrix does not reconcile")
    heavy = sum(
        count for source, count in label_source_counts.items() if source.endswith("heavy")
    )
    light = label_eligible_n - heavy
    majority_class = "heavy" if heavy > light else "light"
    return {
        "eligible_n": label_eligible_n,
        "prediction_available": predicted_n,
        "observed_heavy": heavy,
        "observed_light": sum(
            label_source_counts[source]
            for source in ("observed_light", "observed_composed_light")
        ),
        "short_null_imputed_light": sum(
            label_source_counts[source]
            for source in ("short_null_imputed_light", "short_null_composed_light")
        ),
        "null_unavailable": label_source_counts["null_unavailable"],
        "composition_ambiguous": label_source_counts["composition_ambiguous"],
        "composition_unavailable": label_source_counts[
            "composition_unavailable"
        ],
        "heavy_count": heavy,
        "heavy_rate": heavy / label_eligible_n if label_eligible_n else None,
        "accuracy": (
            (tp + tn) / label_eligible_n
            if label_eligible_n and raw["prediction_unavailable"] == 0
            else None
        ),
        "available_only_accuracy": (
            (tp + tn) / predicted_n if predicted_n else None
        ),
        "majority_light_accuracy": (
            1.0 - heavy / label_eligible_n if label_eligible_n else None
        ),
        "majority_class": majority_class if label_eligible_n else None,
        "majority_class_accuracy": (
            max(heavy, light) / label_eligible_n if label_eligible_n else None
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prediction_unavailable": raw["prediction_unavailable"],
        "label_source_counts": dict(label_source_counts),
        "provenance_counts": dict(raw["provenance_counts"]),
    }


def evaluate(
    fit_rows: list[Row],
    eval_rows: list[Row],
    *,
    candidate_s_selection: CandidateSSelection | None = None,
    heavy_decision_threshold: float = DEFAULT_HEAVY_DECISION_THRESHOLD,
) -> dict[str, Any]:
    shrinkage_alpha = (
        None if candidate_s_selection is None else candidate_s_selection.alpha
    )
    if candidate_s_selection is not None and (
        candidate_s_selection.fit_row_count != len(fit_rows)
        or candidate_s_selection.eval_row_count != len(eval_rows)
    ):
        raise ValueError("Candidate S selection row counts differ from evaluator rows")
    fit_tasks = {row.task_id for row in fit_rows}
    eval_tasks = {row.task_id for row in eval_rows}
    overlap = fit_tasks & eval_tasks
    if overlap:
        raise ValueError(f"fit/eval task overlap: {sorted(overlap)[:3]}")
    fit_repos = {row.repo for row in fit_rows}
    eval_repos = {row.repo for row in eval_rows}
    current_kbs = {
        repo: ClauseResourceKB.fit_public(
            (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
            heavy_decision_threshold=heavy_decision_threshold,
        )
        for repo in sorted(eval_repos)
    }
    candidate_kbs = {
        repo: ClauseResourceKB.fit_public(
            (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
            representation=STRUCTURED_ARGV_REPRESENTATION,
            heavy_decision_threshold=heavy_decision_threshold,
        )
        for repo in sorted(eval_repos)
    }
    shrinkage_kbs = (
        {
            repo: ClauseResourceKB.fit_public(
                (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
                representation=STRUCTURED_ARGV_REPRESENTATION,
                shrinkage_alpha=shrinkage_alpha,
                heavy_decision_threshold=heavy_decision_threshold,
            )
            for repo in sorted(eval_repos)
        }
        if shrinkage_alpha is not None
        else {}
    )
    by_task: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for row in eval_rows:
        by_task[(row.manifest_index, row.task_id)].append(row)

    arm_names = ["current", "candidate_r"]
    if shrinkage_alpha is not None:
        arm_names.append("candidate_s")
    metrics = {
        resource: {
            "label_source_counts": Counter(),
            "arms": {arm: _empty_confusion() for arm in arm_names},
        }
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    for task_ordinal, key in enumerate(sorted(by_task)):
        rows = by_task[key]
        query_ts = float(task_ordinal * 2 + 1)
        current_kb = current_kbs[rows[0].repo]
        candidate_kb = candidate_kbs[rows[0].repo]
        shrinkage_kb = shrinkage_kbs.get(rows[0].repo)
        for row in rows:
            predictions_by_arm = {
                "current": current_kb.predict_clause_resource_classes(
                    row.repo, row.bin, row.argv, ts_start=query_ts
                ),
                "candidate_r": candidate_kb.predict_clause_resource_classes(
                    row.repo, row.bin, row.argv, ts_start=query_ts
                ),
            }
            if shrinkage_kb is not None:
                predictions_by_arm["candidate_s"] = (
                    shrinkage_kb.predict_clause_resource_classes(
                        row.repo,
                        row.bin,
                        row.argv,
                        ts_start=query_ts,
                    )
                )
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
                label, source = _label(row, resource)
                metric = metrics[resource]
                metric["label_source_counts"][source] += 1
                if label is None:
                    continue
                for arm, predictions in predictions_by_arm.items():
                    prediction = predictions[resource]
                    arm_metric = metric["arms"][arm]
                    if prediction is None:
                        arm_metric["prediction_unavailable"] += 1
                        continue
                    arm_metric["provenance_counts"][
                        f"{prediction.scope}:{prediction.key_kind}:"
                        f"{prediction.canonicalizer_version}:"
                        f"{prediction.arbitration}"
                    ] += 1
                    predicted = prediction.label == "heavy"
                    if label and predicted:
                        arm_metric["tp"] += 1
                    elif label:
                        arm_metric["fn"] += 1
                    elif predicted:
                        arm_metric["fp"] += 1
                    else:
                        arm_metric["tn"] += 1
        close_ts = query_ts + 0.5
        for row in rows:
            observation = row.observation(query_ts, close_ts)
            current_kb.observe_completed_clause(observation)
            candidate_kb.observe_completed_clause(observation)
            if shrinkage_kb is not None:
                shrinkage_kb.observe_completed_clause(observation)

    current_metrics: dict[str, Any] = {}
    arm_metrics: dict[str, dict[str, Any]] = {
        arm: {} for arm in arm_names if arm != "current"
    }
    for resource, raw in metrics.items():
        current = _finalize_resource_metric(
            raw["arms"]["current"],
            raw["label_source_counts"],
        )
        current_metrics[resource] = {
            "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
            **current,
        }
        for arm in arm_metrics:
            candidate = _finalize_resource_metric(
                raw["arms"][arm],
                raw["label_source_counts"],
            )
            arm_metrics[arm][resource] = {
                "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
                **candidate,
                "current_accuracy": current["accuracy"],
                "accuracy_minus_current_percentage_points": (
                    None
                    if candidate["accuracy"] is None
                    or current["accuracy"] is None
                    else 100.0 * (candidate["accuracy"] - current["accuracy"])
                ),
                "accuracy_minus_majority_light_percentage_points": (
                    None
                    if candidate["accuracy"] is None
                    else 100.0
                    * (
                        candidate["accuracy"]
                        - candidate["majority_light_accuracy"]
                    )
                ),
            }
    result = {
        "artifact_type": "development_exposed_serialized_virtual_resource_classification",
        "claim_bearing": False,
        "fit": {
            "row_count": len(fit_rows),
            "task_count": len({row.task_id for row in fit_rows}),
            "repo_count": len(fit_repos),
        },
        "eval": {
            "row_count": len(eval_rows),
            "task_count": len({row.task_id for row in eval_rows}),
            "repo_count": len(eval_repos),
        },
        "row_identity": {
            "identical_label_rows_across_arms": True,
            "fit_eval_task_overlap_count": 0,
        },
        "decision_policy": {
            "heavy_decision_threshold": heavy_decision_threshold,
            "tie_decides": "light",
            "source": "declared from the action cost ratio C/(B+C), never fitted",
        },
        "label_policy": {
            "heavy_is_strictly_greater_than_threshold": True,
            "short_null_light_max_latency_ms_exclusive": SHORT_NULL_LIGHT_MAX_LATENCY_MS,
            "long_null": "unavailable",
            "cpu_unit": "cores",
            "memory_unit": "decimal_MB",
            "disk_unit": "read_plus_write_bytes_from_linux_task_io_accounting",
        },
        "candidate_r": {
            "representation": STRUCTURED_ARGV_REPRESENTATION,
            "stable_subcommand_min_distinct_fit_repositories": 3,
            "stable_subcommand_uses_labels": False,
            "arbitration": "same hard first-nonempty selection as current",
        },
        "metrics": current_metrics,
        "candidates": {
            arm: {"metrics": metrics_by_resource}
            for arm, metrics_by_resource in arm_metrics.items()
        },
    }
    if shrinkage_alpha is not None:
        result["candidate_s"] = {
            "representation": STRUCTURED_ARGV_REPRESENTATION,
            "arbitration": "deepest local plus deepest public posterior",
            "shrinkage_alpha": shrinkage_alpha,
            "alpha_source": {
                "latency_result": candidate_s_selection.latency_result_path,
                "fit_path": candidate_s_selection.fit_path,
                "eval_path": candidate_s_selection.eval_path,
                "fit_row_count": candidate_s_selection.fit_row_count,
                "eval_row_count": candidate_s_selection.eval_row_count,
                "verified": True,
            },
            "same_alpha_for_all_targets": True,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--latency-result",
        type=Path,
        help="Candidate S latency result that owns and proves the fit-selected alpha",
    )
    parser.add_argument(
        "--heavy-decision-threshold",
        type=float,
        default=DEFAULT_HEAVY_DECISION_THRESHOLD,
        help=(
            "Cut on P(Heavy), declared as C/(B+C) from the action's cost ratio. "
            "Never select this from an evaluation result."
        ),
    )
    args = parser.parse_args()
    fit_rows = load_rows(args.fit)
    eval_rows = load_rows(args.eval)
    selection = (
        None
        if args.latency_result is None
        else load_candidate_s_selection(
            args.latency_result,
            fit_path=args.fit,
            eval_path=args.eval,
            fit_row_count=len(fit_rows),
            eval_row_count=len(eval_rows),
        )
    )
    result = evaluate(
        fit_rows,
        eval_rows,
        candidate_s_selection=selection,
        heavy_decision_threshold=args.heavy_decision_threshold,
    )
    result["fit"]["path"] = str(args.fit)
    result["eval"]["path"] = str(args.eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
