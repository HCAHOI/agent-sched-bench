#!/usr/bin/env python3
"""Evaluate continuous command- and clause-survival latency updates."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _argmax_probabilities,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    Row,
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    FROZEN_VALIDATION_ROWS,
    _load_rows,
)
from scripts.evaluation.evaluate_relational_agent_state import (  # noqa: E402
    _attempt_records,
    _offset_rows,
    _telemetry_valid_records,
    _write_result_view,
)
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    COMMAND_COMPOSITION_DRAWS,
    ClauseResourceKB,
    _command_stages,
    _stratified_empirical_draws,
)
from tool_resource_eval.labels import repo_of  # noqa: E402

VERSION = "continuous-latency-v1"
ARMS = ("static", "command_survival", "clause_survival")
TICK_MS = 500.0
SNAPSHOT_MS = (500.0, 2000.0, 8000.0, 30_000.0)
MINIMUM_GAIN_PP = 5.0
MINIMUM_POSITIVE_TASKS = 10
MINIMUM_CLAUSE_CHANGED_COMMANDS = 20
MINIMUM_ACTION_LEAD_MS = 500.0
_LATENCY_SOURCE = "latency_ms"
_DEVELOPMENT_RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803"
)
_VALIDATION_RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)
_SPLIT = _ROOT / "analysis/development/sqlglot-relational-task-split.json"
_PUBLIC = (
    _ROOT
    / "traces/swe-rebench/qwen3.7-max/swe100-full-5be74da-20260726"
    / "simulate_cloud_model_c2_20260726T005356962.jsonl",
    _ROOT
    / "traces/swe-rebench/qwen3.7-max/swe277-full-5be74da-20260726"
    / "simulate_cloud_model_c2_20260726T024552768.jsonl",
)


@dataclass(frozen=True)
class ClauseTiming:
    static_index: int | None
    start_ms: float
    end_ms: float
    fallback_reason: str | None = None


@dataclass(frozen=True)
class CallTiming:
    duration_ms: float
    clauses: tuple[ClauseTiming, ...]
    clause_usable: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DynamicModel:
    command: str
    static_pmf: tuple[float, ...] | None
    command_values: tuple[float, ...] | None
    clause_values: tuple[tuple[float, ...], ...]
    clause_draws: tuple[tuple[float, ...], ...]
    stages: tuple[tuple[int, ...], ...] | None


def _pmf(values: Sequence[float]) -> tuple[float, ...]:
    counts = Counter(CANONICAL_LATENCY_BUCKETS.bucket_id(value) for value in values)
    return tuple(
        counts[index] / len(values)
        for index in range(CANONICAL_LATENCY_BUCKETS.bucket_count)
    )


def _point_mass(elapsed_ms: float) -> tuple[float, ...]:
    floor = bisect_right(CANONICAL_LATENCY_BUCKETS.edges_ms, elapsed_ms)
    return tuple(
        1.0 if index == floor else 0.0
        for index in range(CANONICAL_LATENCY_BUCKETS.bucket_count)
    )


def _condition_values(
    values: Sequence[float] | None,
    elapsed_ms: float,
) -> tuple[tuple[float, ...], bool]:
    if elapsed_ms <= 0.0:
        if not values:
            raise ValueError("time-zero conditioning requires empirical values")
        return _pmf(values), False
    survivors = (
        [] if values is None else [value for value in values if value > elapsed_ms]
    )
    if not survivors:
        return _point_mass(elapsed_ms), True
    return _pmf(survivors), False


def _identity(clause: Mapping[str, Any]) -> tuple[str, tuple[str, ...], bool, int]:
    return (
        str(clause.get("bin")),
        tuple(str(value) for value in clause.get("argv", ())),
        clause.get("in_pipe") is True,
        int(clause.get("pipeline_position", -1)),
    )


def _align_timing(
    command: str,
    action_start: float,
    duration_ms: float,
    observed: Sequence[Mapping[str, Any]],
) -> CallTiming:
    parsed = parse_command_clauses(command)
    static = parsed.get("clauses") if isinstance(parsed, dict) else None
    if parsed.get("parse_failed") or not isinstance(static, list):
        return CallTiming(duration_ms, (), False, "parse_failed")
    if _command_stages(static) is None:
        return CallTiming(duration_ms, (), False, "composition_unavailable")
    for clause in observed:
        start = clause.get("ts_start")
        end = clause.get("ts_end")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
        ):
            raise ValueError("observed clause timestamp is invalid")
    unused = set(range(len(static)))
    timings: list[ClauseTiming] = []
    for clause in sorted(observed, key=lambda item: float(item["ts_start"])):
        start = float(clause["ts_start"])
        end = float(clause["ts_end"])
        start_ms = 1000.0 * (start - action_start)
        end_ms = 1000.0 * (end - action_start)
        if start_ms < -1e-3 or end_ms < start_ms or end_ms > duration_ms + 100.0:
            raise ValueError("observed clause timestamp is outside its call")
        matches = [
            index for index in unused if _identity(static[index]) == _identity(clause)
        ]
        if len(matches) != 1:
            timings.append(
                ClauseTiming(
                    None,
                    max(0.0, start_ms),
                    end_ms,
                    "causal_alignment_ambiguous",
                )
            )
            continue
        index = matches[0]
        unused.remove(index)
        timings.append(ClauseTiming(index, max(0.0, start_ms), end_ms))
    return CallTiming(duration_ms, tuple(timings), True)


def _load_timings(
    records: Sequence[Mapping[str, Any]],
    commands: Sequence[CommandRow],
) -> dict[str, CallTiming]:
    expected = {f"{row.task_id}:{row.call_index}": row for row in commands}
    timings: dict[str, CallTiming] = {}
    for record in records:
        task_id = str(record["instance_id"])
        attempt = Path(str(record["attempt_dir"]))
        actions: dict[str, Mapping[str, Any]] = {}
        with (attempt / "trace.jsonl").open(encoding="utf-8") as handle:
            for raw in handle:
                action = json.loads(raw)
                data = action.get("data")
                if (
                    action.get("type") == "action"
                    and action.get("action_type") == "tool_exec"
                    and isinstance(data, dict)
                    and data.get("tool_name") == "exec"
                    and isinstance(data.get("tool_call_id"), str)
                ):
                    actions[str(data["tool_call_id"])] = action
        artifact = json.loads(
            (attempt / "resource_observations.json").read_text(encoding="utf-8")
        )
        for call_index, call in enumerate(artifact.get("calls", [])):
            if not isinstance(call, dict) or call.get("eligible_for_kb") is not True:
                continue
            sample_id = f"{task_id}:{call_index}"
            row = expected.get(sample_id)
            call_id = call.get("tool_call_id")
            action = actions.get(str(call_id))
            if row is None or action is None or call.get("command") != row.command:
                raise ValueError(
                    f"{sample_id}: timing identity differs from command row"
                )
            action_start = action.get("ts_start")
            if not isinstance(action_start, (int, float)) or isinstance(
                action_start, bool
            ):
                raise ValueError(f"{sample_id}: action start is invalid")
            observed = call.get("clauses")
            if not isinstance(observed, list) or not all(
                isinstance(item, dict) for item in observed
            ):
                raise ValueError(f"{sample_id}: observed clauses are invalid")
            timings[sample_id] = _align_timing(
                row.command,
                float(action_start),
                row.duration_ms,
                observed,
            )
    if set(timings) != set(expected):
        raise ValueError("timing rows differ from validation command rows")
    return timings


def _build_model(
    kb: ClauseResourceKB,
    row: CommandRow,
    query_ts: float,
) -> DynamicModel:
    result = kb.predict_command_latency_bucket(
        row.repo,
        row.command,
        query_ts,
        CANONICAL_LATENCY_BUCKETS,
    )
    static_pmf = (
        None
        if result.prediction is None
        else tuple(result.prediction.probability_by_bucket)
    )
    parsed = parse_command_clauses(row.command)
    clauses = parsed.get("clauses") if isinstance(parsed, dict) else None
    if parsed.get("parse_failed") or not isinstance(clauses, list) or not clauses:
        return DynamicModel(row.command, static_pmf, None, (), (), None)
    stages = _command_stages(clauses)
    if stages is None:
        return DynamicModel(row.command, static_pmf, None, (), (), None)
    selected = tuple(
        kb._select(  # noqa: SLF001 - development evaluator needs frozen raw evidence
            row.repo,
            _LATENCY_SOURCE,
            str(clause["bin"]),
            tuple(str(value) for value in clause["argv"]),
        )
        for clause in clauses
    )
    if any(node is None for node in selected):
        return DynamicModel(row.command, static_pmf, None, (), (), stages)
    nodes = tuple(node for node in selected if node is not None)
    clause_values = tuple(tuple(float(value) for value in node[0]) for node in nodes)
    clause_draws = tuple(
        _stratified_empirical_draws(
            values,
            f"{row.command}\0{_LATENCY_SOURCE}\0{index}\0{node[1]}\0{node[2]}",
        )
        for index, (values, node) in enumerate(zip(clause_values, nodes, strict=True))
    )
    if len(clauses) == 1:
        command_values = clause_values[0]
    else:
        composed = kb._composed_command_values(  # noqa: SLF001
            row.repo,
            row.command,
            clauses,
            _LATENCY_SOURCE,
        )
        command_values = None if composed is None else tuple(composed[0])
    if command_values is not None and static_pmf != _pmf(command_values):
        raise AssertionError("raw command values differ from Current PMF")
    return DynamicModel(
        row.command,
        static_pmf,
        command_values,
        clause_values,
        clause_draws,
        stages,
    )


def _visible_clause_state(
    timing: CallTiming,
    elapsed_ms: float,
) -> dict[int, tuple[str, float]]:
    visible: dict[int, tuple[str, float]] = {}
    for clause in timing.clauses:
        if clause.static_index is None or clause.start_ms > elapsed_ms:
            continue
        if clause.end_ms <= elapsed_ms:
            visible[clause.static_index] = (
                "completed",
                clause.end_ms - clause.start_ms,
            )
        else:
            visible[clause.static_index] = (
                "active",
                elapsed_ms - clause.start_ms,
            )
    return visible


def _compose(
    draws_by_clause: Sequence[Sequence[float]], stages: Sequence[Sequence[int]]
) -> tuple[float, ...]:
    values = []
    for draw in range(COMMAND_COMPOSITION_DRAWS):
        values.append(
            sum(
                max(draws_by_clause[index][draw] for index in stage) for stage in stages
            )
        )
    return tuple(sorted(values))


def _command_prediction(
    model: DynamicModel,
    elapsed_ms: float,
) -> tuple[int | None, tuple[float, ...] | None, bool]:
    if elapsed_ms <= 0.0:
        return (
            None
            if model.static_pmf is None
            else _argmax_probabilities(model.static_pmf),
            model.static_pmf,
            False,
        )
    pmf, fallback = _condition_values(model.command_values, elapsed_ms)
    return _argmax_probabilities(pmf), pmf, fallback


def _clause_prediction(
    model: DynamicModel,
    timing: CallTiming,
    elapsed_ms: float,
) -> tuple[int | None, tuple[float, ...] | None, dict[str, Any]]:
    command_hard, command_pmf, command_fallback = _command_prediction(model, elapsed_ms)
    visible_fallback = next(
        (
            clause.fallback_reason
            for clause in timing.clauses
            if clause.fallback_reason is not None and clause.start_ms <= elapsed_ms
        ),
        None,
    )
    if (
        elapsed_ms <= 0.0
        or not timing.clause_usable
        or visible_fallback is not None
        or model.stages is None
        or not model.clause_values
    ):
        return (
            command_hard,
            command_pmf,
            {
                "fallback_to_command": elapsed_ms > 0.0,
                "fallback_reason": timing.fallback_reason or visible_fallback,
                "command_zero_survivor": command_fallback,
                "active_zero_survivors": 0,
                "completed": 0,
                "active": 0,
                "future": len(model.clause_values),
                "skipped": 0,
            },
        )
    visible = _visible_clause_state(timing, elapsed_ms)
    stage_by_clause = {
        index: stage_index
        for stage_index, stage in enumerate(model.stages)
        for index in stage
    }
    started_stages = [stage_by_clause[index] for index in visible]
    latest_started = max(started_stages, default=-1)
    draws: list[tuple[float, ...]] = []
    counts = Counter()
    active_zero = 0
    for index, values in enumerate(model.clause_values):
        state = visible.get(index)
        if state is None and stage_by_clause[index] < latest_started:
            counts["skipped"] += 1
            draws.append((0.0,) * COMMAND_COMPOSITION_DRAWS)
        elif state is None:
            counts["future"] += 1
            draws.append(model.clause_draws[index])
        elif state[0] == "completed":
            counts["completed"] += 1
            draws.append((state[1],) * COMMAND_COMPOSITION_DRAWS)
        else:
            counts["active"] += 1
            survivors = tuple(value for value in values if value > state[1])
            if not survivors:
                active_zero += 1
                survivors = (math.nextafter(state[1], math.inf),)
            draws.append(
                _stratified_empirical_draws(
                    survivors,
                    f"{model.command}\0{_LATENCY_SOURCE}\0{index}\0survival",
                )
            )
    composed = _compose(draws, model.stages)
    pmf, zero = _condition_values(composed, elapsed_ms)
    return (
        _argmax_probabilities(pmf),
        pmf,
        {
            "fallback_to_command": False,
            "fallback_reason": None,
            "command_zero_survivor": zero,
            "active_zero_survivors": active_zero,
            **{
                name: counts[name]
                for name in ("completed", "active", "future", "skipped")
            },
        },
    )


def _update_times(timing: CallTiming) -> list[float]:
    times = {0.0}
    tick = TICK_MS
    while tick < timing.duration_ms:
        times.add(tick)
        tick += TICK_MS
    for clause in timing.clauses:
        if 0.0 < clause.start_ms < timing.duration_ms:
            times.add(clause.start_ms)
        if 0.0 < clause.end_ms < timing.duration_ms:
            times.add(clause.end_ms)
    return sorted(times)


def _trajectory(
    sample_id: str,
    task_id: str,
    command: str,
    label: int,
    model: DynamicModel,
    timing: CallTiming,
) -> dict[str, Any]:
    times = _update_times(timing)
    intervals = []
    transitions = []
    previous: tuple[int | None, int | None, int | None] | None = None
    zero_fallbacks = Counter()
    for index, start in enumerate(times):
        end = timing.duration_ms if index + 1 == len(times) else times[index + 1]
        static = (
            None
            if model.static_pmf is None
            else _argmax_probabilities(model.static_pmf)
        )
        command_hard, _command_pmf, command_zero = _command_prediction(model, start)
        clause_hard, _clause_pmf, clause_state = _clause_prediction(
            model, timing, start
        )
        if start > 0.0:
            floor = bisect_right(CANONICAL_LATENCY_BUCKETS.edges_ms, start)
            if command_hard is not None and command_hard < floor:
                raise AssertionError(
                    "command survival prediction is below elapsed floor"
                )
            if clause_hard is not None and clause_hard < floor:
                raise AssertionError(
                    "clause survival prediction is below elapsed floor"
                )
            if floor > label:
                raise AssertionError("alive command elapsed floor exceeds final label")
        predictions = (static, command_hard, clause_hard)
        interval = {
            "start_ms": start,
            "end_ms": end,
            "weight_ms": end - start,
            "predictions": dict(zip(ARMS, predictions, strict=True)),
        }
        intervals.append(interval)
        if predictions != previous:
            transitions.append({"at_ms": start, **interval["predictions"]})
            previous = predictions
        zero_fallbacks["command"] += command_zero
        zero_fallbacks["clause_command"] += clause_state["command_zero_survivor"]
        zero_fallbacks["clause_active"] += clause_state["active_zero_survivors"]

    def arm_summary(arm: str) -> dict[str, Any]:
        correct_ms = sum(
            item["weight_ms"] for item in intervals if item["predictions"][arm] == label
        )
        unavailable_ms = sum(
            item["weight_ms"] for item in intervals if item["predictions"][arm] is None
        )
        severe_ms = sum(
            item["weight_ms"]
            for item in intervals
            if item["predictions"][arm] is None or label - item["predictions"][arm] >= 2
        )
        stable_start = None
        for position, item in enumerate(intervals):
            if item["predictions"][arm] == label and all(
                later["predictions"][arm] == label for later in intervals[position:]
            ):
                stable_start = item["start_ms"]
                break
        return {
            "correct_ms": correct_ms,
            "correct_fraction": correct_ms / timing.duration_ms,
            "unavailable_ms": unavailable_ms,
            "severe_or_unavailable_ms": severe_ms,
            "stable_correct_at_ms": stable_start,
            "stable_correct_lead_ms": (
                None if stable_start is None else timing.duration_ms - stable_start
            ),
        }

    summaries = {arm: arm_summary(arm) for arm in ARMS}
    clause_actionable_change = any(
        item["predictions"]["clause_survival"]
        != item["predictions"]["command_survival"]
        and timing.duration_ms - item["start_ms"] >= MINIMUM_ACTION_LEAD_MS
        for item in intervals
    )
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "command": command,
        "duration_ms": timing.duration_ms,
        "label": label,
        "clause_usable_at_start": timing.clause_usable,
        "clause_fully_aligned": timing.clause_usable
        and all(clause.static_index is not None for clause in timing.clauses),
        "clause_fallback_reasons": sorted(
            {
                reason
                for reason in (
                    timing.fallback_reason,
                    *(clause.fallback_reason for clause in timing.clauses),
                )
                if reason is not None
            }
        ),
        "updates": len(intervals),
        "transitions": transitions,
        "arms": summaries,
        "zero_survivor_fallbacks": dict(zero_fallbacks),
        "clause_actionable_change": clause_actionable_change,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    total_ms = sum(float(row["duration_ms"]) for row in rows)
    correct_ms = sum(float(row["arms"][arm]["correct_ms"]) for row in rows)
    unavailable_ms = sum(float(row["arms"][arm]["unavailable_ms"]) for row in rows)
    severe_ms = sum(float(row["arms"][arm]["severe_or_unavailable_ms"]) for row in rows)
    leads = sorted(
        float(value)
        for row in rows
        if (value := row["arms"][arm]["stable_correct_lead_ms"]) is not None
    )
    return {
        "time_weighted_exact_accuracy": correct_ms / total_ms,
        "mean_command_correct_time_fraction": sum(
            float(row["arms"][arm]["correct_fraction"]) for row in rows
        )
        / len(rows),
        "time_weighted_severe_or_unavailable_rate": severe_ms / total_ms,
        "time_weighted_unavailable_rate": unavailable_ms / total_ms,
        "stable_correct_commands": len(leads),
        "median_stable_correct_lead_ms": (
            None if not leads else leads[len(leads) // 2]
        ),
        "total_command_ms": total_ms,
    }


def _task_delta(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    control: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    deltas = {}
    for task_id, task_rows in grouped.items():
        total = sum(float(row["duration_ms"]) for row in task_rows)
        candidate_correct = sum(
            float(row["arms"][candidate]["correct_ms"]) for row in task_rows
        )
        control_correct = sum(
            float(row["arms"][control]["correct_ms"]) for row in task_rows
        )
        deltas[task_id] = (candidate_correct - control_correct) / total
    return {
        "positive_tasks": sum(value > 1e-12 for value in deltas.values()),
        "negative_tasks": sum(value < -1e-12 for value in deltas.values()),
        "zero_tasks": sum(abs(value) <= 1e-12 for value in deltas.values()),
        "delta_by_task": deltas,
    }


def _snapshot(
    models: Mapping[str, DynamicModel],
    timings: Mapping[str, CallTiming],
    labels: Mapping[str, int],
    at_ms: float,
) -> dict[str, Any]:
    correct = Counter()
    eligible = 0
    for sample_id, model in models.items():
        timing = timings[sample_id]
        if timing.duration_ms <= at_ms:
            continue
        eligible += 1
        static = (
            None
            if model.static_pmf is None
            else _argmax_probabilities(model.static_pmf)
        )
        command, _pmf_value, _fallback = _command_prediction(model, at_ms)
        clause, _pmf_value, _state = _clause_prediction(model, timing, at_ms)
        for arm, prediction in zip(ARMS, (static, command, clause), strict=True):
            correct[arm] += prediction == labels[sample_id]
    return {
        "alive_commands": eligible,
        **{
            arm: {"correct": correct[arm], "accuracy": correct[arm] / eligible}
            for arm in ARMS
        },
    }


def _mechanism_gate(
    metrics: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    control: str,
) -> dict[str, Any]:
    delta = 100.0 * (
        metrics[candidate]["time_weighted_exact_accuracy"]
        - metrics[control]["time_weighted_exact_accuracy"]
    )
    tasks = _task_delta(rows, candidate, control)
    severe_ok = (
        metrics[candidate]["time_weighted_severe_or_unavailable_rate"]
        <= metrics[control]["time_weighted_severe_or_unavailable_rate"]
    )
    actionable = (
        sum(bool(row["clause_actionable_change"]) for row in rows)
        if candidate == "clause_survival"
        else None
    )
    go = (
        delta >= MINIMUM_GAIN_PP
        and severe_ok
        and tasks["positive_tasks"] > tasks["negative_tasks"]
        and tasks["positive_tasks"] >= MINIMUM_POSITIVE_TASKS
        and (actionable is None or actionable >= MINIMUM_CLAUSE_CHANGED_COMMANDS)
    )
    return {
        "go": go,
        "candidate": candidate,
        "control": control,
        "delta_percentage_points": delta,
        "minimum_gain_percentage_points": MINIMUM_GAIN_PP,
        "no_severe_or_unavailable_regression": severe_ok,
        "task_deltas": tasks,
        "minimum_positive_tasks": MINIMUM_POSITIVE_TASKS,
        "actionable_changed_commands": actionable,
        "minimum_actionable_changed_commands": (
            MINIMUM_CLAUSE_CHANGED_COMMANDS if actionable is not None else None
        ),
        "minimum_action_lead_ms": MINIMUM_ACTION_LEAD_MS,
    }


def _status(command_go: bool, clause_go: bool) -> str:
    if clause_go:
        return "development_clause_mechanism_go"
    if command_go:
        return "development_command_mechanism_go"
    return "development_mechanism_no_go"


def run(
    public: Sequence[Row],
    development_ids: Sequence[str],
    development_clauses: Sequence[Row],
    development_commands: Sequence[CommandRow],
    validation_ids: Sequence[str],
    validation_clauses: Sequence[Row],
    validation_commands: Sequence[CommandRow],
    timings: Mapping[str, CallTiming],
    frozen_rows: Sequence[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    combined_ids = [*development_ids, *validation_ids]
    combined_clauses = [*development_clauses, *validation_clauses]
    combined_commands = [*development_commands, *validation_commands]
    _baseline, baseline_rows = evaluate_prequential_commands(
        public,
        combined_ids,
        combined_clauses,
        combined_commands,
        {"role": "development_exposed_continuous_latency"},
        warmup_task_count=len(development_ids),
    )
    frozen_by_id = {row.sample_id: row for row in frozen_rows}
    if [row["sample_id"] for row in baseline_rows] != [
        row.sample_id for row in frozen_rows
    ]:
        raise ValueError("Current baseline rows differ from frozen validation rows")
    for row in baseline_rows:
        frozen = frozen_by_id[str(row["sample_id"])]
        current = row["current_dynamic"]
        if (
            current["latency"] != frozen.current["latency"]
            or tuple(current["probability_by_bucket"]["latency"] or ())
            != tuple(frozen.pmfs["latency"] or ())
            or row["latency_label"] != frozen.labels["latency"]
        ):
            raise ValueError(
                "Current baseline values differ from frozen validation rows"
            )

    public_evidence = [
        row for row in public if row.structure_known and row.pipeline_position <= 0
    ]
    kb = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public_evidence
    )
    dev_clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    validation_clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    validation_commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in development_clauses:
        dev_clauses_by_task[row.task_id].append(row)
    for row in validation_clauses:
        validation_clauses_by_task[row.task_id].append(row)
    for row in validation_commands:
        validation_commands_by_task[row.task_id].append(row)
    for ordinal, task_id in enumerate(development_ids):
        query_ts = float(ordinal * 2 + 3)
        for clause in dev_clauses_by_task[task_id]:
            kb.observe_completed_clause(clause.observation(query_ts, query_ts + 0.5))

    baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
    models: dict[str, DynamicModel] = {}
    sidecar: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, task_id in enumerate(validation_ids, start=len(development_ids)):
        query_ts = float(ordinal * 2 + 3)
        for command in validation_commands_by_task[task_id]:
            sample_id = f"{task_id}:{command.call_index}"
            base = baseline_by_id[sample_id]
            model = _build_model(kb, command, query_ts)
            expected_pmf = base["current_dynamic"]["probability_by_bucket"]["latency"]
            if model.static_pmf != (
                None if expected_pmf is None else tuple(expected_pmf)
            ):
                raise AssertionError("dynamic model start differs from Current")
            label = int(base["latency_label"])
            models[sample_id] = model
            sidecar.append(
                _trajectory(
                    sample_id,
                    task_id,
                    command.command,
                    label,
                    model,
                    timings[sample_id],
                )
            )
        for clause in validation_clauses_by_task[task_id]:
            kb.observe_completed_clause(clause.observation(query_ts, query_ts + 0.5))
    elapsed = time.perf_counter() - started

    labels = {str(row["sample_id"]): int(row["latency_label"]) for row in baseline_rows}
    metrics = {arm: _aggregate(sidecar, arm) for arm in ARMS}
    command_gate = _mechanism_gate(metrics, sidecar, "command_survival", "static")
    clause_gate = _mechanism_gate(
        metrics, sidecar, "clause_survival", "command_survival"
    )
    return {
        "schema": VERSION,
        "status": _status(command_gate["go"], clause_gate["go"]),
        "claim_bearing": False,
        "protocol": {
            "update_times": "command_start_then_0.5s_ticks_and_visible_clause_events",
            "static": "Current command-start PMF",
            "command_survival": "strict empirical command-duration survival",
            "clause_survival": "completed_exact_active_survival_future_static_then_compose",
            "sequential_latency": "sum",
            "pipeline_latency": "max",
            "zero_survivor": "point_mass_smallest_still_possible_bucket",
            "ambiguous_clause_alignment": "fallback_to_command_survival",
            "current_task_updates": False,
            "validation_updates": "successful whole-task settlement only",
        },
        "coverage": {
            "commands": len(sidecar),
            "tasks": len(validation_ids),
            "compound_commands": sum(
                len(row.clauses) > 1 for row in validation_commands
            ),
            "clause_usable_at_start_commands": sum(
                timing.clause_usable for timing in timings.values()
            ),
            "clause_fully_aligned_commands": sum(
                timing.clause_usable
                and all(clause.static_index is not None for clause in timing.clauses)
                for timing in timings.values()
            ),
            "clause_eventual_fallback_commands": sum(
                not timing.clause_usable
                or any(clause.static_index is None for clause in timing.clauses)
                for timing in timings.values()
            ),
            "updates": sum(int(row["updates"]) for row in sidecar),
            "zero_survivor_fallbacks": dict(
                sum(
                    (Counter(row["zero_survivor_fallbacks"]) for row in sidecar),
                    Counter(),
                )
            ),
        },
        "metrics": metrics,
        "snapshots": {
            str(int(at_ms)): _snapshot(models, timings, labels, at_ms)
            for at_ms in SNAPSHOT_MS
        },
        "gates": {
            "command_survival_vs_static": command_gate,
            "clause_survival_vs_command_survival": clause_gate,
        },
        "integrity": {
            "initial_current_rows_identical": True,
            "non_latency_targets_untouched": True,
            "final_partition_read": False,
            "development_exposed_only": True,
        },
        "cost": {
            "projection_and_scoring_seconds": elapsed,
            "new_collection": False,
            "agent_calls": 0,
        },
    }, sidecar


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    split = json.loads(_SPLIT.read_text(encoding="utf-8"))
    if len(split["development"]) != 100 or len(split["validation"]) != 50:
        raise ValueError("split differs from the frozen development protocol")
    development_ids, development_clauses, development_commands = load_run_rows(
        _DEVELOPMENT_RUN
    )
    if development_ids != split["development"]:
        raise ValueError("development task order differs from frozen split")
    records, statuses = _attempt_records(_VALIDATION_RUN, split["validation"])
    valid = _telemetry_valid_records(records, statuses)
    if len(valid) != 50 or any(row["status"] != "accepted" for row in statuses):
        raise RuntimeError("validation role is incomplete or telemetry-invalid")
    with tempfile.TemporaryDirectory(prefix="continuous-latency-") as directory:
        view = Path(directory) / "results.jsonl"
        _write_result_view(valid, view)
        validation_ids, raw_clauses, raw_commands = load_run_rows(
            _VALIDATION_RUN, results_path=view
        )
    if validation_ids != split["validation"]:
        raise ValueError("validation task order differs from frozen split")
    validation_clauses, validation_commands = _offset_rows(
        raw_clauses, raw_commands, len(development_ids)
    )
    timings = _load_timings(valid, validation_commands)
    public = [row for path in _PUBLIC for row in load_rows(path)]
    excluded = {repo_of(task_id) for task_id in (*development_ids, *validation_ids)}
    public = [row for row in public if row.repo not in excluded]
    result, rows = run(
        public,
        development_ids,
        development_clauses,
        development_commands,
        validation_ids,
        validation_clauses,
        validation_commands,
        timings,
        _load_rows(FROZEN_VALIDATION_ROWS),
    )
    result["inputs"] = {
        "development_run": str(_DEVELOPMENT_RUN.resolve()),
        "validation_run": str(_VALIDATION_RUN.resolve()),
        "validation_rows": str(FROZEN_VALIDATION_ROWS.resolve()),
        "public_telemetry": [str(path.resolve()) for path in _PUBLIC],
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
