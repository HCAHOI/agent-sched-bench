#!/usr/bin/env python3
"""Freeze and evaluate documentation-derived tool semantics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import tiktoken

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    INTERACTION_FEATURE_VERSION,
    InteractionFeatureSet,
    _InteractionPosetKB,
    _argmax_probabilities,
    _fit_stable_subcommands,
    _phase_changes,
    _predict_interaction_command,
    _predict_poset_resource_buckets,
    _row_resource_value,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    Row,
    command_resource_bucket_label,
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    BUCKETS,
    Row as HistoryRow,
    _fail_closed_metrics,
)
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    run as run_command_outcome_memory,
)
from scripts.evaluation.evaluate_multitarget_sota import _compose  # noqa: E402
from scripts.evaluation.evaluate_semantic_work_units import (  # noqa: E402
    run as run_semantic_work_units,
)
from scripts.evaluation.evaluate_full_test_phase_fresh import (  # noqa: E402
    MINIMUM_FIT_TASKS,
    apply_phase_candidate,
    fit_phase_pmfs,
    phase_coverage,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (  # noqa: E402
    _codex_call,
    _validated_usage,
)
from scripts.evaluation.classify_full_test_states import _is_tool_free_event  # noqa: E402
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.pip_semantics import parse_pip_install  # noqa: E402
from tool_resource.pytest_semantics import is_pytest_invocation  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    ClauseResourceKB,
)
from tool_resource.tool_spec import (  # noqa: E402
    ToolSpec,
    interpret_argv,
    tool_spec_schema,
    validate_tool_spec,
)
from tool_time.command import command_prefix_keys  # noqa: E402


SCHEMA = "offline-tool-semantics-splits-v1"
REPOSITORIES = {
    "sqlglot": "tobymao/sqlglot",
    "pennylane": "PennyLaneAI/pennylane",
    "dvc": "iterative/dvc",
}
DEFAULT_SIZES = {
    "sqlglot": (100, 100),
    "pennylane": (16, 16),
    "dvc": (32, 33),
}
_VALID_STATUS = {
    "collection_validity": "valid",
    "workload_execution": "completed",
    "telemetry_quality": "ok",
    "cleanup": "ok",
}
_TOOL_NAMES = ("git", "make", "pip_install", "pytest")
_TARGETS = ("latency", *CANONICAL_RESOURCE_BUCKET_EDGES)
_ARMS = (
    "majority",
    "raw_prefix",
    "clause_kb",
    "task_aware",
    "generic_poset",
    "docs_poset",
)
_RAW_PREFIX_DEPTH = 4
_BOOTSTRAP_DRAWS = 2_000
_BOOTSTRAP_SEED = 0
_GENERATION_SOURCES = _REPO_ROOT / "analysis/development/offline-tool-semantics-docs"
_GENERATION_OUTPUT = _REPO_ROOT / "analysis/results/offline-tool-semantics-sqlglot-v1"
_SPLIT_MANIFEST = _REPO_ROOT / "analysis/development/offline-tool-semantics-splits.json"
_MAX_GENERATION_INPUT_TOKENS = 64_000
_MAX_GENERATION_RESPONSE_BYTES = 65_536


def traced_task_ids(trace_root: Path, known_ids: set[str]) -> set[str]:
    """Return task IDs appearing as directory names without opening their files."""

    found: set[str] = set()
    for root, directories, _files in os.walk(trace_root):
        del root
        retained: list[str] = []
        for name in directories:
            if name in known_ids:
                found.add(name)
            else:
                retained.append(name)
        directories[:] = retained
    return found


def _ordered_tasks(
    tasks: Sequence[Mapping[str, Any]], repo: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in tasks if row.get("repo") == repo]
    if any(
        not isinstance(row.get("instance_id"), str)
        or not isinstance(row.get("created_at"), str)
        for row in selected
    ):
        raise ValueError(f"{repo}: task metadata lacks identity or created_at")
    return sorted(selected, key=lambda row: (row["created_at"], row["instance_id"]))


def _slice_exact(
    ids: Sequence[str], sizes: tuple[int, int], repo: str
) -> tuple[list[str], list[str]]:
    expected = sum(sizes)
    if len(ids) != expected:
        raise ValueError(
            f"{repo}: expected {expected} eligible tasks, found {len(ids)}"
        )
    return list(ids[: sizes[0]]), list(ids[sizes[0] :])


def build_split_manifest(
    tasks: Sequence[Mapping[str, Any]],
    traced_ids: set[str],
    *,
    sqlglot_development_ids: set[str],
    sizes: Mapping[str, tuple[int, int]] = DEFAULT_SIZES,
) -> dict[str, Any]:
    """Freeze deterministic development/validation/final task identities."""

    all_ids = [row.get("instance_id") for row in tasks]
    if any(not isinstance(task_id, str) for task_id in all_ids):
        raise ValueError("task manifest contains a non-string instance_id")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("task manifest contains duplicate instance IDs")

    ordered = {name: _ordered_tasks(tasks, repo) for name, repo in REPOSITORIES.items()}
    sql_ids = [
        row["instance_id"]
        for row in ordered["sqlglot"]
        if row["instance_id"] in sqlglot_development_ids
    ]
    if set(sql_ids) != sqlglot_development_ids:
        raise ValueError("SQLGlot development IDs differ from task metadata")
    sql_warmup, sql_scored = _slice_exact(sql_ids, sizes["sqlglot"], "sqlglot")

    fresh: dict[str, list[str]] = {}
    for name in ("pennylane", "dvc"):
        fresh[name] = [
            row["instance_id"]
            for row in ordered[name]
            if row["instance_id"] not in traced_ids
        ]
    pennylane_warmup, pennylane_validation = _slice_exact(
        fresh["pennylane"], sizes["pennylane"], "pennylane"
    )
    dvc_warmup, dvc_final = _slice_exact(fresh["dvc"], sizes["dvc"], "dvc")

    cohorts = {
        "sqlglot": {
            "development_warmup": sql_warmup,
            "development_scored": sql_scored,
        },
        "pennylane": {
            "warmup": pennylane_warmup,
            "validation": pennylane_validation,
        },
        "dvc": {"warmup": dvc_warmup, "final": dvc_final},
    }
    split_ids = [
        task_id
        for cohort in cohorts.values()
        for ids in cohort.values()
        for task_id in ids
    ]
    return {
        "schema": SCHEMA,
        "cohorts": cohorts,
        "quarantined_task_counts": {
            name: sum(row["instance_id"] in traced_ids for row in ordered[name])
            for name in ("pennylane", "dvc")
        },
        "integrity": {
            "all_split_ids_disjoint": len(split_ids) == len(set(split_ids)),
            "ordering": "created_at_then_instance_id",
            "quarantine": "any_existing_task_directory",
        },
    }


def _tool_name(argv: Sequence[str]) -> str | None:
    if parse_pip_install(argv) is not None:
        return "pip_install"
    if is_pytest_invocation(argv):
        return "pytest"
    if not argv:
        return None
    binary = PurePosixPath(str(argv[0])).name.lower()
    if binary == "git":
        return "git"
    if binary in {"make", "gmake"}:
        return "make"
    return None


def census_attempts(
    attempts: Iterable[tuple[str, str, Path]],
) -> dict[str, Any]:
    """Count documented-tool invocations without retaining commands or outcomes."""

    task_ids: set[str] = set()
    versions: set[str] = set()
    exec_commands = parsed_clauses = parse_failures = 0
    invocations = {name: 0 for name in _TOOL_NAMES}
    tool_tasks: dict[str, set[str]] = defaultdict(set)
    for task_id, version, attempt in attempts:
        status = json.loads((attempt / "resource_observations.json").read_text())
        mismatch = {
            key: status.get(key)
            for key, expected in _VALID_STATUS.items()
            if status.get(key) != expected
        }
        if mismatch:
            raise ValueError(f"{attempt}: invalid collection status {mismatch}")
        task_ids.add(task_id)
        versions.add(version)
        calls = json.loads((attempt / "tool_calls.json").read_text())
        if not isinstance(calls, list):
            raise ValueError(f"{attempt}: tool_calls.json is not an array")
        for call in calls:
            if not isinstance(call, Mapping) or call.get("tool") != "exec":
                continue
            input_value = call.get("input")
            command = (
                input_value.get("command") if isinstance(input_value, Mapping) else None
            )
            if not isinstance(command, str):
                raise ValueError(f"{attempt}: exec call lacks a command")
            exec_commands += 1
            parsed = parse_command_clauses(command)
            if parsed.get("parse_failed"):
                parse_failures += 1
                continue
            clauses = parsed.get("clauses")
            if not isinstance(clauses, list):
                raise ValueError("clause parser returned no clause list")
            parsed_clauses += len(clauses)
            for clause in clauses:
                argv_value = clause.get("argv") if isinstance(clause, Mapping) else None
                if not isinstance(argv_value, list):
                    continue
                name = _tool_name(tuple(str(value) for value in argv_value))
                if name is not None:
                    invocations[name] += 1
                    tool_tasks[name].add(task_id)
    return {
        "tasks": len(task_ids),
        "repo_versions": sorted(versions),
        "exec_commands": exec_commands,
        "parsed_clauses": parsed_clauses,
        "parse_failures": parse_failures,
        "tools": {
            name: {"invocations": invocations[name], "tasks": len(tool_tasks[name])}
            for name in _TOOL_NAMES
        },
    }


def _labels(row: CommandRow) -> dict[str, int | None]:
    return {
        "latency": CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms),
        **{
            resource: command_resource_bucket_label(row, resource)[0]
            for resource in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }


def _hard_predictions(
    pmfs: Mapping[str, Sequence[float] | None],
) -> dict[str, int | str | None]:
    return {
        target: (
            None
            if pmf is None
            else _argmax_probabilities(pmf)
            if target == "latency"
            else RESOURCE_BUCKET_LABELS[_argmax_probabilities(pmf)]
        )
        for target, pmf in pmfs.items()
    }


def _arm(
    pmfs: Mapping[str, Sequence[float] | None],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    copied = {
        target: None if pmf is None else list(pmf) for target, pmf in pmfs.items()
    }
    return {
        "prediction": _hard_predictions(copied),
        "probability_by_bucket": copied,
        "provenance": {} if provenance is None else dict(provenance),
    }


def _empirical_pmf(values: Sequence[int], bucket_count: int) -> list[float]:
    counts = Counter(values)
    return [counts[bucket] / len(values) for bucket in range(bucket_count)]


def _doc_feature_builder(
    specs: Mapping[str, ToolSpec],
    observed_versions: Mapping[str, str],
):
    def build(bin_: str, argv: Sequence[str]) -> InteractionFeatureSet:
        matches = [
            interpreted
            for name, spec in specs.items()
            if (
                interpreted := interpret_argv(
                    spec,
                    bin_,
                    argv,
                    observed_versions[name],
                )
            )
            is not None
        ]
        if len(matches) != 1:
            raise ValueError("argv is unsupported or ambiguously documented")
        return InteractionFeatureSet(matches[0].scope, matches[0].features)

    return build


def _documented_tools(
    parsed_clauses: Sequence[Mapping[str, Any]],
    specs: Mapping[str, ToolSpec],
    observed_versions: Mapping[str, str],
) -> tuple[str, ...] | None:
    tools: list[str] = []
    for clause in parsed_clauses:
        bin_ = clause.get("bin")
        argv = clause.get("argv")
        if not isinstance(bin_, str) or not isinstance(argv, list) or not argv:
            return None
        matches = [
            name
            for name, spec in specs.items()
            if interpret_argv(
                spec,
                bin_,
                tuple(str(value) for value in argv),
                observed_versions[name],
            )
            is not None
        ]
        if len(matches) != 1:
            return None
        tools.append(matches[0])
    return tuple(dict.fromkeys(tools)) if tools else None


def _public_pools(
    rows: Sequence[Row],
    feature_builder=None,
) -> tuple[
    dict[str, list[Row]],
    dict[str, dict[str, list[Row]]],
    dict[str, list[Row]],
]:
    by_scope: dict[str, list[Row]] = defaultdict(list)
    by_resource_scope = {
        resource: defaultdict(list) for resource in CANONICAL_RESOURCE_BUCKET_EDGES
    }
    by_resource = {resource: [] for resource in CANONICAL_RESOURCE_BUCKET_EDGES}
    for row in rows:
        scope = (
            row.bin
            if feature_builder is None
            else feature_builder(row.bin, row.argv).bin
        )
        by_scope[scope].append(row)
        for resource in CANONICAL_RESOURCE_BUCKET_EDGES:
            if _row_resource_value(row, resource) is None:
                continue
            by_resource_scope[resource][scope].append(row)
            by_resource[resource].append(row)
    return by_scope, by_resource_scope, by_resource


def _observe_posets(
    latency: _InteractionPosetKB,
    resources: Mapping[str, _InteractionPosetKB],
    rows: Sequence[Row],
) -> None:
    latency.observe(rows)
    for resource, kb in resources.items():
        kb.observe(
            [row for row in rows if _row_resource_value(row, resource) is not None]
        )


def _predict_posets(
    row: CommandRow,
    parsed: Mapping[str, Any],
    latency: _InteractionPosetKB,
    resources: Mapping[str, _InteractionPosetKB],
    public_by_scope: Mapping[str, Sequence[Row]],
    public_by_resource_scope: Mapping[str, Mapping[str, Sequence[Row]]],
    public: Sequence[Row],
    public_by_resource: Mapping[str, Sequence[Row]],
    *,
    feature_version: str,
) -> tuple[dict[str, Sequence[float] | None], dict[str, Any]]:
    clauses = parsed.get("clauses", [])
    parse_failed = bool(parsed.get("parse_failed"))
    latency_prediction, latency_unavailable, latency_diagnostic = (
        _predict_interaction_command(
            latency,
            row,
            clauses,
            parse_failed=parse_failed,
            public_by_bin=public_by_scope,
            public_global=public,
        )
    )
    resource_predictions, resource_unavailable, resource_diagnostic = (
        _predict_poset_resource_buckets(
            resources,
            row,
            clauses,
            parse_failed=parse_failed,
            public_by_resource_bin=public_by_resource_scope,
            public_by_resource=public_by_resource,
            feature_version=feature_version,
        )
    )
    return {
        "latency": (
            None
            if latency_prediction is None
            else latency_prediction.probability_by_bucket
        ),
        **{
            resource: (
                None
                if resource_predictions.get(resource) is None
                else resource_predictions[resource].probability_by_bucket
            )
            for resource in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }, {
        "latency": latency_diagnostic,
        "resources": resource_diagnostic,
        "unavailable": latency_unavailable or resource_unavailable,
    }


def _history_row(
    row: CommandRow,
    current: Mapping[str, Any] | None = None,
    pmfs: Mapping[str, Sequence[float] | None] | None = None,
) -> HistoryRow:
    return HistoryRow(
        sample_id=f"{row.task_id}:{row.call_index}",
        task_id=row.task_id,
        command=row.command,
        labels=_labels(row),
        current=(
            {target: None for target in _TARGETS}
            if current is None
            else {target: current.get(target) for target in _TARGETS}
        ),
        pmfs=(
            {target: None for target in _TARGETS}
            if pmfs is None
            else {
                target: (
                    None
                    if pmfs.get(target) is None
                    else tuple(pmfs[target])
                )
                for target in _TARGETS
            }
        ),
    )


def _task_aware(
    warmup_commands: Sequence[CommandRow],
    scored_commands: Sequence[CommandRow],
    baseline_by_sample: Mapping[str, Mapping[str, Any]],
    *,
    events_by_task: Mapping[str, Sequence[Any]],
) -> dict[str, dict[str, Any]]:
    fit_rows = [_history_row(row) for row in warmup_commands]
    validation_rows: list[HistoryRow] = []
    for row in scored_commands:
        base = baseline_by_sample[f"{row.task_id}:{row.call_index}"]
        current = dict(base["current_dynamic"])
        pmfs = current.pop("probability_by_bucket")
        validation_rows.append(_history_row(row, current, pmfs))
    _semantic_result, semantic = run_semantic_work_units(fit_rows, validation_rows)
    _exact_result, exact = run_command_outcome_memory(fit_rows, validation_rows)
    warmup_ids = list(dict.fromkeys(row.task_id for row in warmup_commands))
    scored_ids = list(dict.fromkeys(row.task_id for row in scored_commands))
    warmup_events = {task_id: events_by_task[task_id] for task_id in warmup_ids}
    if phase_coverage(warmup_ids, warmup_events)["phase_tasks"] < MINIMUM_FIT_TASKS:
        phase = [
            {
                **{key: value for key, value in row.items() if key != "arms"},
                "phase_applied_targets": [],
            }
            for row in semantic
        ]
    else:
        phase_pmfs, _support = fit_phase_pmfs(
            warmup_ids,
            warmup_commands,
            warmup_events,
        )
        phase = apply_phase_candidate(
            list(baseline_by_sample.values()),
            scored_commands,
            {task_id: events_by_task[task_id] for task_id in scored_ids},
            phase_pmfs,
        )
    return {
        str(semantic_row["sample_id"]): _compose(semantic_row, phase_row, exact_row)
        for semantic_row, phase_row, exact_row in zip(
            semantic, phase, exact, strict=True
        )
    }


def _metric_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    reference: str | None = None,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        value = {
            "sample_id": row["sample_id"],
            "task_id": row["task_id"],
            "command": row["command"],
            "labels": row["labels"],
            "candidate": row["arms"][arm]["prediction"],
            "candidate_probability_by_bucket": row["arms"][arm][
                "probability_by_bucket"
            ],
        }
        if reference is not None:
            value["reference"] = row["arms"][reference]["prediction"]
        output.append(value)
    return output


def _arm_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    accuracies = []
    severe = []
    arm_rows = _metric_rows(rows, arm)
    for target in _TARGETS:
        eligible = sum(row["labels"][target] is not None for row in rows)
        if not eligible:
            metrics[target] = None
            continue
        metric = _fail_closed_metrics(arm_rows, target)
        metrics[target] = metric
        accuracies.append(
            metric["exact_class_accuracy" if target == "latency" else "accuracy"]
        )
        severe.append(metric["severe_underprediction_rate"])
    return {
        "targets": metrics,
        "equal_weight_accuracy": (
            None
            if len(accuracies) != len(_TARGETS)
            else sum(accuracies) / len(accuracies)
        ),
        "equal_weight_severe_underprediction_rate": (
            None if len(severe) != len(_TARGETS) else sum(severe) / len(severe)
        ),
    }


def _macro_accuracy_difference(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> float:
    differences = []
    for target in _TARGETS:
        eligible = [row for row in rows if row["labels"][target] is not None]
        if not eligible:
            raise ValueError(f"no eligible {target} labels")
        differences.append(
            sum(
                (
                    row["arms"][left]["prediction"][target]
                    == (
                        row["labels"][target]
                        if target == "latency"
                        else RESOURCE_BUCKET_LABELS[row["labels"][target]]
                    )
                )
                - (
                    row["arms"][right]["prediction"][target]
                    == (
                        row["labels"][target]
                        if target == "latency"
                        else RESOURCE_BUCKET_LABELS[row["labels"][target]]
                    )
                )
                for row in eligible
            )
            / len(eligible)
        )
    return sum(differences) / len(differences)


def _paired_macro_bootstrap(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    if not tasks:
        raise ValueError("task bootstrap has no tasks")
    rng = random.Random(_BOOTSTRAP_SEED)
    draws = []
    for _ in range(_BOOTSTRAP_DRAWS):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        draws.append(
            _macro_accuracy_difference(
                [row for task_id in sample for row in by_task[task_id]],
                left,
                right,
            )
        )
    draws.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower = int(position)
        upper = min(lower + 1, len(draws) - 1)
        fraction = position - lower
        return draws[lower] * (1.0 - fraction) + draws[upper] * fraction

    return {
        "method": "paired_task_cluster_percentile_bootstrap",
        "cluster_unit": "task",
        "seed": _BOOTSTRAP_SEED,
        "draws": _BOOTSTRAP_DRAWS,
        "statistic": f"{left}_minus_{right}_equal_weight_accuracy",
        "point_estimate": _macro_accuracy_difference(rows, left, right),
        "interval_95": [percentile(0.025), percentile(0.975)],
    }


def _changes(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    by_target = {
        target: _phase_changes(
            _metric_rows(
                [row for row in rows if row["labels"][target] is not None],
                left,
                reference=right,
            ),
            target,
            reference="reference",
        )
        for target in _TARGETS
    }
    return {
        "by_target": by_target,
        "helpful": sum(value["helpful"] for value in by_target.values()),
        "harmful": sum(value["harmful"] for value in by_target.values()),
        "changed": sum(value["changed"] for value in by_target.values()),
        "helpful_task_ids": sorted(
            {
                task_id
                for value in by_target.values()
                for task_id in value["helpful_task_ids"]
            }
        ),
    }


def _report(
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    task_ids: Sequence[str],
    warmup_task_count: int,
) -> dict[str, Any]:
    metrics = {arm: _arm_metrics(rows, arm) for arm in _ARMS}
    docs_vs_generic = _changes(rows, "docs_poset", "generic_poset")
    docs_vs_task = _changes(rows, "docs_poset", "task_aware")
    bootstrap = _paired_macro_bootstrap(rows, "docs_poset", "generic_poset")
    generic_accuracy = metrics["generic_poset"]["equal_weight_accuracy"]
    docs_accuracy = metrics["docs_poset"]["equal_weight_accuracy"]
    task_accuracy = metrics["task_aware"]["equal_weight_accuracy"]
    generic_severe = metrics["generic_poset"][
        "equal_weight_severe_underprediction_rate"
    ]
    docs_severe = metrics["docs_poset"]["equal_weight_severe_underprediction_rate"]
    per_tool: dict[str, Any] = {}
    for tool in sorted(
        {tool for row in rows for tool in row.get("documented_tools", [])}
    ):
        selected = [row for row in rows if tool in row.get("documented_tools", [])]
        tool_docs = _arm_metrics(selected, "docs_poset")
        tool_generic = _arm_metrics(selected, "generic_poset")
        left = tool_docs["equal_weight_accuracy"]
        right = tool_generic["equal_weight_accuracy"]
        per_tool[tool] = {
            "commands": len(selected),
            "docs_poset": tool_docs,
            "generic_poset": tool_generic,
            "equal_weight_accuracy_difference": (
                None if left is None or right is None else left - right
            ),
        }
    nonnegative_tools = sum(
        value["equal_weight_accuracy_difference"] is not None
        and value["equal_weight_accuracy_difference"] >= 0.0
        for value in per_tool.values()
    )
    checkpoints = {}
    scored_ids = list(task_ids[warmup_task_count:])
    for settled in (5, 10, 20, 40):
        remaining = set(scored_ids[settled:])
        checkpoints[str(settled)] = (
            None
            if not remaining
            else {
                arm: _arm_metrics(
                    [row for row in rows if row["task_id"] in remaining], arm
                )["equal_weight_accuracy"]
                for arm in _ARMS
            }
        )
    development_go = (
        docs_accuracy is not None
        and generic_accuracy is not None
        and docs_accuracy > generic_accuracy
        and docs_vs_generic["helpful"] > docs_vs_generic["harmful"]
        and docs_severe is not None
        and generic_severe is not None
        and docs_severe <= generic_severe
        and len(docs_vs_generic["helpful_task_ids"]) > 1
    )
    validation_go = (
        bootstrap["interval_95"][0] > 0.0
        and docs_vs_generic["helpful"] > docs_vs_generic["harmful"]
        and docs_severe is not None
        and generic_severe is not None
        and docs_severe <= generic_severe
        and nonnegative_tools >= 3
        and len(docs_vs_generic["helpful_task_ids"]) > 1
        and docs_accuracy is not None
        and task_accuracy is not None
        and docs_accuracy >= task_accuracy
    )
    return {
        "schema": "offline-tool-semantics-evaluation-v1",
        "claim_bearing": False,
        "provenance": dict(provenance),
        "protocol": {
            "targets": {"latency_buckets": 5, "cpu_rss_disk_buckets": 3},
            "warmup_task_count": warmup_task_count,
            "scored_task_count": len(scored_ids),
            "causal_update": "whole-task settlement",
            "raw_prefix_max_depth": _RAW_PREFIX_DEPTH,
            "macro": "equal weight across four targets",
        },
        "arms": metrics,
        "per_tool": per_tool,
        "comparisons": {
            "docs_poset_minus_generic_poset": {
                "changes": docs_vs_generic,
                "bootstrap": bootstrap,
            },
            "docs_poset_minus_task_aware": {"changes": docs_vs_task},
        },
        "cold_start_checkpoints": checkpoints,
        "gates": {
            "development_go": development_go,
            "validation_go": validation_go,
            "nonnegative_supported_tools": nonnegative_tools,
        },
        "row_identity": {
            "identical_command_ids_and_labels": True,
            "identical_availability": True,
            "commands": len(rows),
        },
    }


def evaluate_doc_semantics(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
    specs: Mapping[str, ToolSpec],
    *,
    warmup_task_count: int,
    provenance: Mapping[str, Any],
    observed_versions: Mapping[str, str] | None = None,
    allow_unverified_versions: bool = False,
    events_by_task: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the six frozen arms on one same-repository causal stream."""

    if allow_unverified_versions and observed_versions is not None:
        raise ValueError("cannot combine verified and unverified tool versions")
    if not specs:
        raise ValueError("at least one valid ToolSpec is required")
    versions = (
        {name: spec.documented_version for name, spec in specs.items()}
        if allow_unverified_versions
        else dict(observed_versions or {})
    )
    if set(versions) != set(specs):
        raise ValueError("observed tool versions differ from generated specs")
    if not 0 < warmup_task_count < len(task_ids):
        raise ValueError("warmup must leave at least one scored task")
    if list(events_by_task) != list(task_ids):
        raise ValueError("Task-Aware requires ordered raw events for every task")

    baseline_result, baseline_rows = evaluate_prequential_commands(
        public_rows,
        task_ids,
        clause_rows,
        command_rows,
        provenance,
        warmup_task_count=warmup_task_count,
    )
    del baseline_result
    baseline_by_sample = {str(row["sample_id"]): row for row in baseline_rows}
    if len(baseline_by_sample) != len(baseline_rows):
        raise AssertionError("Clause-KB baseline contains duplicate commands")

    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    warmup_ids = list(task_ids[:warmup_task_count])
    scored_ids = list(task_ids[warmup_task_count:])
    warmup_commands = [
        row for task_id in warmup_ids for row in commands_by_task[task_id]
    ]
    scored_commands = [
        row for task_id in scored_ids for row in commands_by_task[task_id]
    ]
    if [f"{row.task_id}:{row.call_index}" for row in scored_commands] != list(
        baseline_by_sample
    ):
        raise AssertionError("Clause-KB baseline differs from command order")

    fit_labels = {target: [] for target in _TARGETS}
    for row in warmup_commands:
        for target, label in _labels(row).items():
            if label is not None:
                fit_labels[target].append(label)
    if any(not values for values in fit_labels.values()):
        raise ValueError("fit-set Majority lacks an eligible target label")
    majority = {
        target: _empirical_pmf(values, BUCKETS[target])
        for target, values in fit_labels.items()
    }

    prefix_history: dict[str, dict[str, list[int]]] = {
        target: defaultdict(list) for target in _TARGETS
    }

    def absorb_prefix(rows: Sequence[CommandRow]) -> None:
        for row in rows:
            keys = command_prefix_keys("exec", row.command, max_depth=_RAW_PREFIX_DEPTH)
            labels = _labels(row)
            for target, label in labels.items():
                if label is None:
                    continue
                for key in keys:
                    prefix_history[target][key].append(label)

    absorb_prefix(warmup_commands)
    public = tuple(
        row for row in public_rows if row.structure_known and row.pipeline_position <= 0
    )
    if not public:
        raise ValueError("no eligible public evidence")
    doc_builder = _doc_feature_builder(specs, versions)
    doc_public = tuple(
        row
        for row in public
        if _documented_tools(
            ({"bin": row.bin, "argv": list(row.argv)},), specs, versions
        )
        is not None
    )
    stable_subcommands = _fit_stable_subcommands(
        row.observation(0.0, 1.0) for row in doc_public
    )
    generic_latency = _InteractionPosetKB(stable_subcommands)
    generic_resources = {
        resource: _InteractionPosetKB(stable_subcommands)
        for resource in CANONICAL_RESOURCE_BUCKET_EDGES
    }
    generic_pools = _public_pools(doc_public) if doc_public else ({}, {}, {})
    docs_latency = _InteractionPosetKB(feature_builder=doc_builder)
    docs_resources = {
        resource: _InteractionPosetKB(feature_builder=doc_builder)
        for resource in CANONICAL_RESOURCE_BUCKET_EDGES
    }
    docs_pools = _public_pools(doc_public, doc_builder) if doc_public else ({}, {}, {})

    warmup_clauses = [row for task_id in warmup_ids for row in clauses_by_task[task_id]]
    documented_warmup = [
        row
        for row in warmup_clauses
        if _documented_tools(
            ({"bin": row.bin, "argv": list(row.argv)},), specs, versions
        )
        is not None
    ]
    _observe_posets(generic_latency, generic_resources, documented_warmup)
    _observe_posets(
        docs_latency,
        docs_resources,
        documented_warmup,
    )
    public_kb = ClauseResourceKB.fit_public(row.observation(0.0, 1.0) for row in public)

    rows: list[dict[str, Any]] = []
    for task_id in scored_ids:
        task_outputs = []
        for row in commands_by_task[task_id]:
            sample_id = f"{row.task_id}:{row.call_index}"
            base = baseline_by_sample[sample_id]
            current = dict(base["current_dynamic"])
            current_pmfs = current.pop("probability_by_bucket")
            parsed = parse_command_clauses(row.command)
            parsed_clauses = parsed.get("clauses", [])
            documented = (
                None
                if parsed.get("parse_failed")
                else _documented_tools(parsed_clauses, specs, versions)
            )
            docs_ready = (
                documented is not None
                and bool(doc_public)
                and all(
                    docs_pools[2].get(resource)
                    for resource in CANONICAL_RESOURCE_BUCKET_EDGES
                )
            )
            if docs_ready:
                generic_pmfs, generic_diagnostic = _predict_posets(
                    row,
                    parsed,
                    generic_latency,
                    generic_resources,
                    generic_pools[0],
                    generic_pools[1],
                    doc_public,
                    generic_pools[2],
                    feature_version=INTERACTION_FEATURE_VERSION,
                )
                docs_pmfs, docs_diagnostic = _predict_posets(
                    row,
                    parsed,
                    docs_latency,
                    docs_resources,
                    docs_pools[0],
                    docs_pools[1],
                    doc_public,
                    docs_pools[2],
                    feature_version="documentation-tool-spec-v1",
                )
            else:
                generic_pmfs, generic_diagnostic = (
                    current_pmfs,
                    {"fallback": "clause_kb"},
                )
                docs_pmfs, docs_diagnostic = current_pmfs, {"fallback": "clause_kb"}

            def aligned(candidate: Mapping[str, Sequence[float] | None]):
                return {
                    target: (
                        None
                        if current_pmfs[target] is None
                        else current_pmfs[target]
                        if candidate.get(target) is None
                        else candidate[target]
                    )
                    for target in _TARGETS
                }

            keys = command_prefix_keys("exec", row.command, max_depth=_RAW_PREFIX_DEPTH)
            public_latency = public_kb.predict_command_latency_bucket(
                row.repo,
                row.command,
                3.0,
                CANONICAL_LATENCY_BUCKETS,
            ).prediction
            public_resources = public_kb.predict_command_resource_buckets(
                row.repo,
                row.command,
                3.0,
            ).classifications
            public_pmfs = {
                "latency": (
                    None
                    if public_latency is None
                    else public_latency.probability_by_bucket
                ),
                **{
                    resource: (
                        None
                        if public_resources[resource] is None
                        else public_resources[resource].probability_by_bucket
                    )
                    for resource in CANONICAL_RESOURCE_BUCKET_EDGES
                },
            }
            prefix_pmfs = {}
            for target in _TARGETS:
                values = next(
                    (
                        prefix_history[target][key]
                        for key in reversed(keys)
                        if prefix_history[target].get(key)
                    ),
                    None,
                )
                if current_pmfs[target] is None:
                    prefix_pmfs[target] = None
                elif values is not None:
                    prefix_pmfs[target] = _empirical_pmf(values, BUCKETS[target])
                elif public_pmfs[target] is None:
                    raise ValueError(f"public prefix fallback lacks {target}")
                else:
                    prefix_pmfs[target] = public_pmfs[target]
            majority_pmfs = {
                target: None if current_pmfs[target] is None else majority[target]
                for target in _TARGETS
            }
            task_outputs.append(
                {
                    "sample_id": sample_id,
                    "task_id": row.task_id,
                    "command": row.command,
                    "labels": _labels(row),
                    "documented_tools": [] if documented is None else list(documented),
                    "arms": {
                        "majority": _arm(majority_pmfs),
                        "raw_prefix": _arm(prefix_pmfs),
                        "clause_kb": _arm(current_pmfs),
                        "generic_poset": _arm(
                            aligned(generic_pmfs), provenance=generic_diagnostic
                        ),
                        "docs_poset": _arm(
                            aligned(docs_pmfs), provenance=docs_diagnostic
                        ),
                    },
                }
            )
        rows.extend(task_outputs)
        absorb_prefix(commands_by_task[task_id])
        settled = clauses_by_task[task_id]
        documented_settled = [
            row
            for row in settled
            if _documented_tools(
                ({"bin": row.bin, "argv": list(row.argv)},), specs, versions
            )
            is not None
        ]
        _observe_posets(generic_latency, generic_resources, documented_settled)
        _observe_posets(
            docs_latency,
            docs_resources,
            documented_settled,
        )

    task_aware = _task_aware(
        warmup_commands,
        scored_commands,
        baseline_by_sample,
        events_by_task=events_by_task,
    )
    for row in rows:
        selected = task_aware[row["sample_id"]]
        clause_pmfs = row["arms"]["clause_kb"]["probability_by_bucket"]
        task_pmfs = {
            target: (
                None
                if clause_pmfs[target] is None
                else clause_pmfs[target]
                if selected["candidate_probability_by_bucket"][target] is None
                else selected["candidate_probability_by_bucket"][target]
            )
            for target in _TARGETS
        }
        row["arms"]["task_aware"] = _arm(
            task_pmfs,
            provenance=selected.get("provenance", {}),
        )
        if set(row["arms"]) != set(_ARMS):
            raise AssertionError("evaluation arms differ")
        availability = {
            arm: tuple(
                row["arms"][arm]["probability_by_bucket"][target] is not None
                for target in _TARGETS
            )
            for arm in _ARMS
        }
        if len(set(availability.values())) != 1:
            raise AssertionError("evaluation arms differ in prediction availability")

    return _report(rows, provenance, task_ids, warmup_task_count), rows


def _render_generation_prompt(
    template: str,
    tool: str,
    version: str,
    documentation: str,
) -> str:
    """Fill the frozen prompt without interpreting or combining tool sources."""

    placeholders = ("{{TOOL}}", "{{VERSION}}", "{{DOCUMENTATION}}")
    if any(template.count(value) != 1 for value in placeholders):
        raise ValueError("generation prompt placeholders differ from the protocol")
    prefix, suffix = template.split("{{DOCUMENTATION}}")
    return (
        prefix.replace("{{TOOL}}", tool).replace("{{VERSION}}", version)
        + documentation
        + suffix
    )


def _generation_inputs() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    protocol = json.loads(
        (_GENERATION_SOURCES / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (_GENERATION_SOURCES / "source-manifest.json").read_text(encoding="utf-8")
    )
    template = (_GENERATION_SOURCES / "prompt-template.md").read_text(encoding="utf-8")
    schema = json.loads(
        (_GENERATION_SOURCES / "tool-spec-schema.json").read_text(encoding="utf-8")
    )
    if schema != tool_spec_schema():
        raise ValueError("frozen output schema differs from host validation")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != set(_TOOL_NAMES):
        raise ValueError("source manifest differs from the fixed tool list")
    for tool, source in tools.items():
        if not isinstance(source, dict):
            raise ValueError(f"{tool}: source manifest entry is invalid")
        path = _GENERATION_SOURCES / str(source.get("file"))
        if not path.is_file() or path.stat().st_size != source.get("bytes"):
            raise ValueError(f"{tool}: source snapshot differs from its manifest")
    generation = protocol.get("generation")
    if (
        not isinstance(protocol.get("frozen_code"), dict)
        or not isinstance(protocol.get("public_telemetry"), list)
        or len(protocol["public_telemetry"]) != 2
        or not all(isinstance(path, str) for path in protocol["public_telemetry"])
        or generation
        != {
            "calls_per_tool": 1,
            "model": "gpt-5.6-sol",
            "requested_service_tier": "fast",
            "reasoning_effort": "medium",
            "temperature": "unsupported_by_codex_provider",
            "sandbox": "read-only",
            "maximum_input_tokens": _MAX_GENERATION_INPUT_TOKENS,
            "maximum_response_bytes": _MAX_GENERATION_RESPONSE_BYTES,
            "repair_calls": 0,
            "invalid_result": "tool unsupported",
        }
    ):
        raise ValueError("generation protocol differs from the frozen contract")
    return manifest, template, schema, protocol


def _generation_worktree_head() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = _GENERATION_OUTPUT.resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    for record in status.split(b"\0"):
        if not record:
            continue
        path = _REPO_ROOT / record[3:].decode()
        if record[:2] != b"??" or not path.resolve().is_relative_to(output):
            raise ValueError("generation requires a clean preregistration worktree")
    return head


def _tool_generation_paths(directory: Path, tool: str) -> dict[str, Path]:
    return {
        "prompt": directory / f"{tool}.prompt.txt",
        "schema": directory / f"{tool}.schema.json",
        "response": directory / f"{tool}.response.json",
        "events": directory / f"{tool}.events.jsonl",
        "stderr": directory / f"{tool}.stderr.txt",
    }


def _generation_events(path: Path, tool: str) -> list[dict[str, Any]]:
    try:
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{tool}: generation events differ from their artifact"
        ) from error
    if not events or any(not _is_tool_free_event(event) for event in events):
        raise ValueError(f"{tool}: generation events differ from their artifact")
    return events


def _validated_response_text(
    events: Sequence[Mapping[str, Any]], response_text: str, tool: str
) -> None:
    messages = [
        item["text"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(item := event.get("item"), dict)
        and item.get("type") == "agent_message"
        and isinstance(item.get("text"), str)
    ]
    if len(messages) != 1 or messages[0] != response_text:
        raise ValueError(f"{tool}: generation response differs from its event")


def _generation_status(
    tool: str,
    version: str,
    spec: ToolSpec | None,
    response_bytes: int,
    usage: Mapping[str, int],
) -> str:
    if (
        spec is not None
        and spec.tool == tool
        and spec.documented_version == version
        and response_bytes <= _MAX_GENERATION_RESPONSE_BYTES
        and usage["input_tokens"] <= _MAX_GENERATION_INPUT_TOKENS
    ):
        return "valid"
    return "unsupported_structural_failure"


def _generate_specs() -> None:
    if _GENERATION_OUTPUT.exists():
        raise FileExistsError(
            "generation output already exists; calls cannot be repeated"
        )
    preregistration_commit = _generation_worktree_head()
    manifest, template, schema, protocol = _generation_inputs()
    _GENERATION_OUTPUT.mkdir(parents=True)
    encoding = tiktoken.get_encoding("cl100k_base")
    tool_results: dict[str, Any] = {}
    for tool in _TOOL_NAMES:
        source = manifest["tools"][tool]
        version = str(source["documented_version"])
        source_record = {
            "file": source["file"],
            "documented_version": version,
            "bytes": source["bytes"],
        }
        documentation = (_GENERATION_SOURCES / source["file"]).read_text(
            encoding="utf-8"
        )
        prompt = _render_generation_prompt(template, tool, version, documentation)
        estimated_tokens = len(encoding.encode(prompt)) + len(
            encoding.encode(json.dumps(schema, sort_keys=True))
        )
        if estimated_tokens > _MAX_GENERATION_INPUT_TOKENS:
            raise ValueError(f"{tool}: generation input exceeds the frozen budget")
        try:
            response, cost = _codex_call(prompt, schema, _GENERATION_OUTPUT, tool)
            paths = _tool_generation_paths(_GENERATION_OUTPUT, tool)
            response_text = paths["response"].read_text(encoding="utf-8")
            _validated_response_text(
                _generation_events(paths["events"], tool), response_text, tool
            )
            response_bytes = paths["response"].stat().st_size
            status = _generation_status(
                tool,
                version,
                validate_tool_spec(response),
                response_bytes,
                cost["usage"],
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            tool_results[tool] = {
                "status": "unsupported_generation_failure",
                "source": source_record,
                "error": str(exc),
                "estimated_input_tokens": estimated_tokens,
            }
            continue
        tool_results[tool] = {
            "status": status,
            "source": source_record,
            "estimated_input_tokens": estimated_tokens,
            "response_bytes": response_bytes,
            "cost": cost,
        }
    artifact = {
        "schema": "offline-tool-semantics-generation-v1",
        "protocol": protocol["schema"],
        "frozen_code": protocol["frozen_code"],
        "preregistration_commit": preregistration_commit,
        "model": "gpt-5.6-sol",
        "requested_service_tier": "fast",
        "reasoning_effort": "medium",
        "temperature": "unsupported_by_codex_provider",
        "codex_version": subprocess.run(
            ["codex", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "tools": tool_results,
    }
    (_GENERATION_OUTPUT / "generation-artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generated_specs() -> tuple[dict[str, ToolSpec], dict[str, Any]]:
    manifest, template, schema, protocol = _generation_inputs()
    artifact = json.loads(
        (_GENERATION_OUTPUT / "generation-artifact.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema": "offline-tool-semantics-generation-v1",
        "protocol": protocol["schema"],
        "frozen_code": protocol["frozen_code"],
        "model": "gpt-5.6-sol",
        "requested_service_tier": "fast",
        "reasoning_effort": "medium",
        "temperature": "unsupported_by_codex_provider",
    }
    if (
        not isinstance(artifact, dict)
        or any(artifact.get(key) != value for key, value in expected.items())
        or set(artifact)
        != {*expected, "preregistration_commit", "codex_version", "tools"}
        or not isinstance(artifact["codex_version"], str)
        or not artifact["codex_version"]
        or not isinstance(artifact["tools"], dict)
        or set(artifact["tools"]) != set(_TOOL_NAMES)
        or artifact.get("preregistration_commit") != _generation_worktree_head()
    ):
        raise ValueError("generation artifact differs from the frozen protocol")
    specs: dict[str, ToolSpec] = {}
    encoding = tiktoken.get_encoding("cl100k_base")
    for tool in _TOOL_NAMES:
        record = artifact["tools"][tool]
        source = manifest["tools"][tool]
        version = str(source["documented_version"])
        if not isinstance(record, dict) or record.get("source") != {
            "file": source["file"],
            "documented_version": version,
            "bytes": source["bytes"],
        }:
            raise ValueError("generation artifact differs from the frozen protocol")
        prompt = _render_generation_prompt(
            template,
            tool,
            version,
            (_GENERATION_SOURCES / source["file"]).read_text(encoding="utf-8"),
        )
        paths = _tool_generation_paths(_GENERATION_OUTPUT, tool)
        if (
            not paths["prompt"].is_file()
            or not paths["schema"].is_file()
            or not paths["events"].is_file()
            or paths["prompt"].read_text(encoding="utf-8") != prompt
            or json.loads(paths["schema"].read_text(encoding="utf-8")) != schema
            or record.get("estimated_input_tokens")
            != len(encoding.encode(prompt))
            + len(encoding.encode(json.dumps(schema, sort_keys=True)))
        ):
            raise ValueError("generation artifact differs from the frozen protocol")
        if record.get("status") == "unsupported_generation_failure":
            if set(record) != {
                "status",
                "source",
                "estimated_input_tokens",
                "error",
            } or not isinstance(record["error"], str):
                raise ValueError("generation artifact differs from the frozen protocol")
            continue
        if (
            set(record)
            != {"status", "source", "estimated_input_tokens", "response_bytes", "cost"}
            or not paths["response"].is_file()
        ):
            raise ValueError("generation artifact differs from the frozen protocol")
        events = _generation_events(paths["events"], tool)
        usage = _validated_usage(events, tool)
        cost = record["cost"]
        if (
            not isinstance(cost, dict)
            or set(cost) != {"prompt_bytes", "wall_seconds", "usage"}
            or cost["prompt_bytes"] != len(prompt.encode())
            or not isinstance(cost["wall_seconds"], (int, float))
            or isinstance(cost["wall_seconds"], bool)
            or cost["wall_seconds"] < 0
            or cost["usage"] != usage
            or record["response_bytes"] != paths["response"].stat().st_size
        ):
            raise ValueError("generation artifact differs from the frozen protocol")
        response_text = paths["response"].read_text(encoding="utf-8")
        _validated_response_text(events, response_text, tool)
        value = json.loads(response_text)
        spec = validate_tool_spec(value)
        expected_status = _generation_status(
            tool, version, spec, record["response_bytes"], usage
        )
        if record["status"] != expected_status:
            raise ValueError("generation artifact differs from the frozen protocol")
        if expected_status == "valid":
            specs[tool] = spec
    return specs, artifact


def _load_development_stream() -> tuple[
    list[str],
    list[Row],
    list[CommandRow],
    dict[str, Sequence[Any]],
    dict[str, Any],
]:
    split = json.loads(_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    cohorts = split["cohorts"]["sqlglot"]
    ordered_ids = [
        *cohorts["development_warmup"],
        *cohorts["development_scored"],
    ]
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    events: dict[str, Sequence[Any]] = {}
    loaded_ids: list[str] = []
    for raw_path in split["sources"]["sqlglot_runs"]:
        run_dir = Path(raw_path)
        task_ids, clauses, commands = load_run_rows(run_dir)
        run_events = _load_exec_events(run_dir, task_ids)
        loaded_ids.extend(task_ids)
        for row in clauses:
            clauses_by_task[row.task_id].append(row)
        for row in commands:
            commands_by_task[row.task_id].append(row)
        overlap = set(events) & set(run_events)
        if overlap:
            raise ValueError(f"development runs repeat tasks: {sorted(overlap)[:3]}")
        events.update(run_events)
    if set(loaded_ids) != set(ordered_ids) or len(loaded_ids) != len(ordered_ids):
        raise ValueError("development runs differ from the frozen SQLGlot tasks")

    clauses: list[Row] = []
    commands: list[CommandRow] = []
    ordered_events: dict[str, Sequence[Any]] = {}
    for manifest_index, task_id in enumerate(ordered_ids):
        replacements: dict[int, Row] = {}

        def replaced(row: Row) -> Row:
            value = replacements.get(id(row))
            if value is None:
                value = replace(row, manifest_index=manifest_index)
                replacements[id(row)] = value
            return value

        commands.extend(
            replace(
                row,
                manifest_index=manifest_index,
                clauses=tuple(replaced(clause) for clause in row.clauses),
            )
            for row in commands_by_task[task_id]
        )
        clauses.extend(replaced(row) for row in clauses_by_task[task_id])
        ordered_events[task_id] = events[task_id]
    return ordered_ids, clauses, commands, ordered_events, split


def _spec_coverage(
    task_ids: Sequence[str],
    commands: Sequence[CommandRow],
    specs: Mapping[str, ToolSpec],
) -> tuple[dict[str, ToolSpec], dict[str, Any]]:
    invocations = Counter()
    tasks: dict[str, set[str]] = defaultdict(set)
    for command in commands:
        for clause in command.clauses:
            for tool, spec in specs.items():
                if (
                    interpret_argv(
                        spec,
                        clause.bin,
                        clause.argv,
                        spec.documented_version,
                    )
                    is not None
                ):
                    invocations[tool] += 1
                    tasks[tool].add(command.task_id)
    report = {
        tool: {
            "invocations": invocations[tool],
            "tasks": len(tasks[tool]),
            "passed": invocations[tool] >= 50 and len(tasks[tool]) >= 10,
        }
        for tool in specs
    }
    enabled = {tool: spec for tool, spec in specs.items() if report[tool]["passed"]}
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("coverage task order contains duplicates")
    return enabled, report


def _evaluation_inputs(
    *,
    scored_task_limit: int | None,
) -> tuple[
    list[Row],
    list[str],
    list[Row],
    list[CommandRow],
    dict[str, ToolSpec],
    dict[str, Sequence[Any]],
    dict[str, Any],
]:
    specs, generation = _generated_specs()
    protocol = json.loads(
        (_GENERATION_SOURCES / "protocol.json").read_text(encoding="utf-8")
    )
    task_ids, clauses, commands, events, split = _load_development_stream()
    specs, coverage = _spec_coverage(task_ids, commands, specs)
    if not specs:
        raise ValueError("no generated tool passes structural and coverage gates")
    warmup_count = len(split["cohorts"]["sqlglot"]["development_warmup"])
    if scored_task_limit is not None:
        if not 0 < scored_task_limit <= len(task_ids) - warmup_count:
            raise ValueError("profile scored-task limit is invalid")
        task_ids = task_ids[: warmup_count + scored_task_limit]
        selected = set(task_ids)
        clauses = [row for row in clauses if row.task_id in selected]
        commands = [row for row in commands if row.task_id in selected]
        events = {task_id: events[task_id] for task_id in task_ids}
    public_paths = [_REPO_ROOT / path for path in protocol["public_telemetry"]]
    public = [row for path in public_paths for row in load_rows(path)]
    reserved_repos = {REPOSITORIES[name].replace("/", "__") for name in REPOSITORIES}
    raw_public_count = len(public)
    public = [row for row in public if row.repo not in reserved_repos]
    if not public or {row.task_id for row in public} & set(task_ids):
        raise ValueError("public evidence is empty or overlaps a reserved task")
    provenance = {
        "generation_artifact": str(
            (_GENERATION_OUTPUT / "generation-artifact.json").resolve()
        ),
        "generation_preregistration_commit": generation["preregistration_commit"],
        "split_manifest": str(_SPLIT_MANIFEST.resolve()),
        "public_telemetry": [str(path.resolve()) for path in public_paths],
        "public_rows_before_reserved_repo_filter": raw_public_count,
        "public_rows_after_reserved_repo_filter": len(public),
        "enabled_tools": sorted(specs),
        "structural_coverage": coverage,
    }
    return public, task_ids, clauses, commands, specs, events, provenance


def _run_evaluation(args: argparse.Namespace, *, profile: bool) -> None:
    inputs = _evaluation_inputs(
        scored_task_limit=args.scored_tasks if profile else None,
    )
    public, task_ids, clauses, commands, specs, events, provenance = inputs
    warmup_count = len(
        json.loads(_SPLIT_MANIFEST.read_text(encoding="utf-8"))["cohorts"]["sqlglot"][
            "development_warmup"
        ]
    )
    started = time.monotonic()
    result, rows = evaluate_doc_semantics(
        public,
        task_ids,
        clauses,
        commands,
        specs,
        warmup_task_count=warmup_count,
        provenance=provenance,
        allow_unverified_versions=True,
        events_by_task=events,
    )
    elapsed = time.monotonic() - started
    if profile:
        print(
            json.dumps(
                {
                    "profile_only": True,
                    "scored_tasks": args.scored_tasks,
                    "scored_commands": len(rows),
                    "evaluation_seconds": elapsed,
                    "enabled_tools": sorted(specs),
                    "structural_coverage": provenance["structural_coverage"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result["cost"] = {"evaluation_seconds": elapsed}
    result["generation"] = {
        "enabled_tools": sorted(specs),
        "structural_coverage": provenance["structural_coverage"],
    }
    result_path = _GENERATION_OUTPUT / "result.json"
    rows_path = _GENERATION_OUTPUT / "rows.jsonl"
    if result_path.exists() or rows_path.exists():
        raise FileExistsError("development evaluation output already exists")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_attempts(
    run_dir: Path, versions: Mapping[str, str]
) -> list[tuple[str, str, Path]]:
    attempts: list[tuple[str, str, Path]] = []
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = row.get("instance_id")
            attempt_value = row.get("attempt_dir")
            if (
                row.get("success") is not True
                or not isinstance(task_id, str)
                or not isinstance(attempt_value, str)
            ):
                raise ValueError(f"{run_dir}: result is not a successful final attempt")
            attempt = Path(attempt_value)
            if not attempt.is_absolute():
                attempt = run_dir / attempt
            attempt = attempt.resolve()
            if not attempt.is_relative_to(run_dir.resolve()):
                raise ValueError(f"{run_dir}: attempt escapes its run directory")
            attempts.append((task_id, versions[task_id], attempt))
    return attempts


def _freeze(args: argparse.Namespace) -> None:
    tasks = json.loads(args.tasks.read_text())
    if not isinstance(tasks, list):
        raise ValueError("task manifest is not an array")
    versions = {
        row["instance_id"]: str(row["version"])
        for row in tasks
        if isinstance(row, Mapping)
        and isinstance(row.get("instance_id"), str)
        and row.get("version") is not None
    }
    attempts = [
        attempt
        for run_dir in args.sqlglot_run
        for attempt in _run_attempts(run_dir.resolve(), versions)
    ]
    development_ids = {task_id for task_id, _version, _attempt in attempts}
    known_ids = {
        row["instance_id"]
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("instance_id"), str)
    }
    result = build_split_manifest(
        tasks,
        traced_task_ids(args.trace_root, known_ids),
        sqlglot_development_ids=development_ids,
    )
    result["development_coverage"] = census_attempts(attempts)
    result["sources"] = {
        "tasks": str(args.tasks.resolve()),
        "trace_root": str(args.trace_root.resolve()),
        "sqlglot_runs": [str(path.resolve()) for path in args.sqlglot_run],
    }
    if args.out.exists():
        raise FileExistsError("split output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--tasks", type=Path, required=True)
    freeze.add_argument("--trace-root", type=Path, required=True)
    freeze.add_argument("--sqlglot-run", type=Path, action="append", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    subparsers.add_parser("generate")
    for name in ("profile", "evaluate"):
        evaluate = subparsers.add_parser(name)
        if name == "profile":
            evaluate.add_argument("--scored-tasks", type=int, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        _freeze(args)
    elif args.command == "generate":
        _generate_specs()
    elif args.command == "profile":
        _run_evaluation(args, profile=True)
    elif args.command == "evaluate":
        _run_evaluation(args, profile=False)


if __name__ == "__main__":
    main()
