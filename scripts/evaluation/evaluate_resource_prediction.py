#!/usr/bin/env python3
"""Evaluate the registered peak-container-memory command-prefix comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource.labels import ResourceCallSample, load_resource_corpus  # noqa: E402
from tool_resource.metrics import ecdf_quantile, pinball_loss  # noqa: E402
from tool_resource.prior import (  # noqa: E402
    adapt_resource_rows,
    build_resource_prior,
    resource_prior_hierarchy,
    validate_resource_profile_eval_disjoint,
)
from tool_time.command import make_row_command_prefix_keys  # noqa: E402
from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.statistics import (  # noqa: E402
    permutation_simultaneous_labels,
    resample_task_totals,
)


_QUANTILE = 0.90
_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_REPLICATES = 50_000
_PERMUTATION_DRAWS = 50_000
_SEED = 0
_FOLD_COUNT = 5
_COMMAND_FIELD = "command"
_MAX_PREFIX_DEPTH = 4
_MIN_TOOL_HISTORY = 1
_MIN_PROFILE_TASKS = 1


def evaluate_peak_memory(
    fit_samples: Sequence[ResourceCallSample],
    eval_samples: Sequence[ResourceCallSample],
) -> dict[str, Any]:
    """Score prefix, tool-name, and global ECDFs on eligible peak labels."""

    fit_rows = [
        sample.to_json_obj() for sample in fit_samples if sample.peak_memory_mb_eligible
    ]
    eval_rows = [
        sample.to_json_obj()
        for sample in eval_samples
        if sample.peak_memory_mb_eligible
    ]
    if not fit_rows or not eval_rows:
        raise ValueError("peak-memory evaluation requires eligible fit and eval rows")

    row_group_keys = make_row_command_prefix_keys(
        _COMMAND_FIELD, max_depth=_MAX_PREFIX_DEPTH
    )
    prior = build_resource_prior(
        fit_rows,
        value_field="peak_memory_mb",
        row_group_keys=row_group_keys,
    )
    adapted_eval = adapt_resource_rows(eval_rows, "peak_memory_mb")
    validate_resource_profile_eval_disjoint(adapted_eval, prior=prior)

    task_ids = sorted({str(row["task_id"]) for row in adapted_eval})
    fold_count = min(_FOLD_COUNT, len(task_ids))
    folds = balanced_task_folds(adapted_eval, fold_count=fold_count)
    task_position = {task_id: index for index, task_id in enumerate(task_ids)}
    loss_by_task = np.zeros((len(task_ids), 3), dtype=float)
    count_by_task = np.zeros(len(task_ids), dtype=float)
    coverage = np.zeros(3, dtype=float)
    selected_sources: Counter[str] = Counter()

    for row in adapted_eval:
        prefix_hierarchy = resource_prior_hierarchy(
            prior,
            str(row["tool_name"]),
            row_group_keys(row),
            min_tool_history=_MIN_TOOL_HISTORY,
            min_profile_tasks=_MIN_PROFILE_TASKS,
        )
        tool_hierarchy = resource_prior_hierarchy(
            prior,
            str(row["tool_name"]),
            (),
            min_tool_history=_MIN_TOOL_HISTORY,
            min_profile_tasks=_MIN_PROFILE_TASKS,
        )
        nodes = (prefix_hierarchy[-1], tool_hierarchy[-1], prefix_hierarchy[0])
        observation = float(row["latency_ms"])
        predictions = [ecdf_quantile(node.values, _QUANTILE) for node in nodes]
        losses = [
            pinball_loss(observation, prediction, _QUANTILE)
            for prediction in predictions
        ]
        position = task_position[str(row["task_id"])]
        loss_by_task[position] += losses
        count_by_task[position] += 1.0
        coverage += [observation <= prediction for prediction in predictions]
        selected_sources[prefix_hierarchy[-1].source] += 1

    totals = loss_by_task.sum(axis=0)
    if totals[1] <= 0.0:
        raise ValueError("tool-name baseline has zero total p90 pinball loss")
    bootstrap_totals = resample_task_totals(
        loss_by_task[:, :2], replicates=_BOOTSTRAP_REPLICATES, seed=_SEED
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        skill_draws = 1.0 - bootstrap_totals[:, 0] / bootstrap_totals[:, 1]
    finite_draws = skill_draws[np.isfinite(skill_draws)]
    if finite_draws.size == 0:
        raise ValueError("all bootstrap skill draws are undefined")
    alpha = 1.0 - _CONFIDENCE_LEVEL
    ci_low, ci_high = np.quantile(
        finite_draws,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )

    paired_contributions = (loss_by_task[:, 1] - loss_by_task[:, 0])[:, None]
    permutation = permutation_simultaneous_labels(
        paired_contributions,
        paired_contributions.sum(axis=0),
        confidence_level=_CONFIDENCE_LEVEL,
        draws=_PERMUTATION_DRAWS,
        seed=_SEED,
    )
    row_count = int(count_by_task.sum())
    names = ("command_prefix", "tool_name", "global")
    losses = {
        name: {
            "total_pinball_loss_mb": float(totals[index]),
            "mean_pinball_loss_mb": float(totals[index] / row_count),
            "p90_coverage": float(coverage[index] / row_count),
        }
        for index, name in enumerate(names)
    }
    return {
        "eligible_eval_row_count": row_count,
        "eligible_eval_task_count": len(task_ids),
        "eligible_fit_row_count": len(fit_rows),
        "eligible_fit_task_count": len({str(row["task_id"]) for row in fit_rows}),
        "balanced_folds": [
            {
                "task_count": len(fold),
                "row_count": int(
                    sum(count_by_task[task_position[task_id]] for task_id in fold)
                ),
            }
            for fold in folds
        ],
        "selected_prefix_node_sources": dict(sorted(selected_sources.items())),
        "absolute_metrics": losses,
        "prefix_vs_tool_p90_skill": {
            "point": float(1.0 - totals[0] / totals[1]),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "confidence_level": _CONFIDENCE_LEVEL,
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "seed": _SEED,
        },
        "paired_sign_flip": permutation,
    }


def repo_cluster_key(task_id: str) -> str:
    """Collapse issue-level task IDs to their repository cluster."""

    return re.sub(r"-\d+$", "", task_id)


def build_ambient_residual_ecdfs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[float]], list[float]]:
    """Build sorted per-tool and global peak-minus-ambient residual ECDFs."""

    residuals_by_tool: dict[str, list[float]] = {}
    global_residuals: list[float] = []
    for row in rows:
        tool_name = str(row["tool_name"])
        peak_memory_mb = float(row["peak_memory_mb"])
        ambient_before_mb = float(row["ambient_before_mb"])
        if not all(
            math.isfinite(value) for value in (peak_memory_mb, ambient_before_mb)
        ):
            raise ValueError("peak and pre-call ambient memory must be finite")
        residual = peak_memory_mb - ambient_before_mb
        residuals_by_tool.setdefault(tool_name, []).append(residual)
        global_residuals.append(residual)
    if not global_residuals:
        raise ValueError("ambient residual prior requires at least one fit row")
    for values in residuals_by_tool.values():
        values.sort()
    global_residuals.sort()
    return residuals_by_tool, global_residuals


def ambient_residual_prediction(
    tool_name: str,
    ambient_before_mb: float,
    residuals_by_tool: Mapping[str, Sequence[float]],
    global_residuals: Sequence[float],
) -> tuple[float, str]:
    """Predict p90 peak memory from the causal ambient anchor and residual ECDF."""

    values = residuals_by_tool.get(tool_name)
    source = "tool_residual" if values else "global_residual"
    return ambient_before_mb + ecdf_quantile(
        values or global_residuals, _QUANTILE
    ), source


def evaluate_stage_a_memory_anchor(
    fit_samples: Sequence[ResourceCallSample],
    eval_samples: Sequence[ResourceCallSample],
) -> dict[str, Any]:
    """Score the registered ambient-anchored residual model against tool ECDFs."""

    baseline_fit_rows = [
        sample.to_json_obj() for sample in fit_samples if sample.peak_memory_mb_eligible
    ]
    residual_fit_rows = [
        row for row in baseline_fit_rows if row["ambient_before_mb"] is not None
    ]
    peak_eval_rows = [
        sample.to_json_obj()
        for sample in eval_samples
        if sample.peak_memory_mb_eligible
    ]
    scoring_rows = [
        row for row in peak_eval_rows if row["ambient_before_mb"] is not None
    ]
    if not baseline_fit_rows or not residual_fit_rows or not scoring_rows:
        raise ValueError("Stage A requires peak labels and pre-call ambient memory")

    absolute_prior = build_resource_prior(
        baseline_fit_rows, value_field="peak_memory_mb"
    )
    adapted_eval = adapt_resource_rows(scoring_rows, "peak_memory_mb")
    validate_resource_profile_eval_disjoint(adapted_eval, prior=absolute_prior)
    residuals_by_tool, global_residuals = build_ambient_residual_ecdfs(
        residual_fit_rows
    )

    losses_by_task: dict[str, np.ndarray] = {}
    losses_by_repo: dict[str, np.ndarray] = {}
    model_sources: Counter[str] = Counter()
    baseline_sources: Counter[str] = Counter()
    for row in adapted_eval:
        task_id = str(row["task_id"])
        tool_name = str(row["tool_name"])
        observation = float(row["latency_ms"])
        model_prediction, model_source = ambient_residual_prediction(
            tool_name,
            float(row["ambient_before_mb"]),
            residuals_by_tool,
            global_residuals,
        )
        baseline_node = resource_prior_hierarchy(
            absolute_prior,
            tool_name,
            (),
            min_tool_history=_MIN_TOOL_HISTORY,
            min_profile_tasks=_MIN_PROFILE_TASKS,
        )[-1]
        baseline_prediction = ecdf_quantile(baseline_node.values, _QUANTILE)
        losses = np.asarray(
            [
                pinball_loss(observation, model_prediction, _QUANTILE),
                pinball_loss(observation, baseline_prediction, _QUANTILE),
            ],
            dtype=float,
        )
        losses_by_task.setdefault(task_id, np.zeros(2, dtype=float))[:] += losses
        repo_id = repo_cluster_key(task_id)
        losses_by_repo.setdefault(repo_id, np.zeros(2, dtype=float))[:] += losses
        model_sources[model_source] += 1
        baseline_sources[baseline_node.source] += 1

    task_skill = _clustered_skill(losses_by_task)
    repo_skill = _clustered_skill(losses_by_repo)
    totals = np.sum(list(losses_by_task.values()), axis=0)
    row_count = len(scoring_rows)
    return {
        "eligibility": {
            "fit_peak_eligible_count": len(baseline_fit_rows),
            "fit_residual_eligible_count": len(residual_fit_rows),
            "fit_excluded_missing_ambient_before_count": len(baseline_fit_rows)
            - len(residual_fit_rows),
            "eval_total_call_count": len(eval_samples),
            "eval_peak_eligible_count": len(peak_eval_rows),
            "eval_scoring_eligible_count": row_count,
            "eval_excluded_peak_ineligible_count": len(eval_samples)
            - len(peak_eval_rows),
            "eval_excluded_missing_ambient_before_count": len(peak_eval_rows)
            - row_count,
        },
        "absolute_metrics": {
            "ambient_anchored_tool_residual": {
                "total_pinball_loss_mb": float(totals[0]),
                "mean_pinball_loss_mb": float(totals[0] / row_count),
            },
            "tool_name_absolute": {
                "total_pinball_loss_mb": float(totals[1]),
                "mean_pinball_loss_mb": float(totals[1] / row_count),
            },
        },
        "model_prediction_sources": dict(sorted(model_sources.items())),
        "baseline_prediction_sources": dict(sorted(baseline_sources.items())),
        "repo_clustered_skill": repo_skill,
        "task_clustered_skill_sensitivity": task_skill,
    }


def _clustered_skill(losses_by_cluster: Mapping[str, np.ndarray]) -> dict[str, Any]:
    keys = sorted(losses_by_cluster)
    contributions = np.vstack([losses_by_cluster[key] for key in keys])
    totals = contributions.sum(axis=0)
    if totals[1] <= 0.0:
        raise ValueError("baseline has zero total p90 pinball loss")
    bootstrap_totals = resample_task_totals(
        contributions, replicates=_BOOTSTRAP_REPLICATES, seed=_SEED
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = 1.0 - bootstrap_totals[:, 0] / bootstrap_totals[:, 1]
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.size == 0:
        raise ValueError("all bootstrap skill draws are undefined")
    alpha = 1.0 - _CONFIDENCE_LEVEL
    low, high = np.quantile(
        finite_draws,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return {
        "point": float(1.0 - totals[0] / totals[1]),
        "ci_low": float(low),
        "ci_high": float(high),
        "cluster_count": len(keys),
        "confidence_level": _CONFIDENCE_LEVEL,
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "seed": _SEED,
    }


def _label_counts(samples: Sequence[ResourceCallSample]) -> dict[str, Any]:
    uncensored = [sample for sample in samples if not sample.censored]
    cpu_kinds = Counter(sample.cpu_core_seconds_kind for sample in samples)
    return {
        "tool_call_count": len(samples),
        "censored_count": sum(sample.censored for sample in samples),
        "cpu_core_seconds_kinds": dict(sorted(cpu_kinds.items())),
        "peak_cpu_eligible_count": sum(
            sample.peak_cpu_cores_eligible for sample in samples
        ),
        "peak_cpu_clipped_sample_count": sum(
            sample.peak_cpu_clipped_sample_count for sample in samples
        ),
        "peak_memory_eligible_count": sum(
            sample.peak_memory_mb_eligible for sample in samples
        ),
        "ambient_memory_only_count": sum(
            sample.ambient_memory_mb_eligible for sample in samples
        ),
        "memory_missing_count": sum(
            not sample.peak_memory_mb_eligible and not sample.ambient_memory_mb_eligible
            for sample in uncensored
        ),
    }


def _flatten(
    samples_by_task: dict[str, list[ResourceCallSample]], task_ids: Sequence[str]
) -> list[ResourceCallSample]:
    return [sample for task_id in task_ids for sample in samples_by_task[task_id]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path, help="Profile trace root")
    parser.add_argument("--eval", required=True, type=Path, help="Eval trace root")
    parser.add_argument(
        "--fit-tasks",
        type=Path,
        default=Path("configs/corpora/swe-100.json"),
        help="JSON manifest supplying fit task_ids",
    )
    parser.add_argument(
        "--eval-tasks",
        type=Path,
        default=Path("configs/corpora/swe-277.json"),
        help="JSON manifest supplying eval task_ids",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        help="Use only the first N fit and eval tasks for a non-evidentiary smoke",
    )
    parser.add_argument(
        "--registered",
        choices=("stage_a_memory_anchor",),
        help="Run a named registered comparison instead of the original prefix path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_tasks is not None and args.limit_tasks < 1:
        raise ValueError("--limit-tasks must be positive")
    fit_by_task, fit_task_ids = load_resource_corpus(
        args.fit, args.fit_tasks, limit_tasks=args.limit_tasks
    )
    eval_by_task, eval_task_ids = load_resource_corpus(
        args.eval, args.eval_tasks, limit_tasks=args.limit_tasks
    )
    task_overlap = sorted(set(fit_task_ids) & set(eval_task_ids))
    if task_overlap:
        raise ValueError(f"fit and eval task_ids overlap: {task_overlap}")
    fit_samples = _flatten(fit_by_task, fit_task_ids)
    eval_samples = _flatten(eval_by_task, eval_task_ids)
    smoke = args.limit_tasks is not None
    if args.registered == "stage_a_memory_anchor":
        comparison = evaluate_stage_a_memory_anchor(fit_samples, eval_samples)
        result = {
            "status": (
                "SMOKE ONLY - limited tasks; metrics are plumbing checks, not findings"
                if smoke
                else "FULL REGISTERED - Stage A memory anchor comparison"
            ),
            "config": {
                "fit": str(args.fit.resolve()),
                "eval": str(args.eval.resolve()),
                "fit_tasks": str(args.fit_tasks.resolve()),
                "eval_tasks": str(args.eval_tasks.resolve()),
                "limit_tasks": args.limit_tasks,
                "registered": args.registered,
                "target": "peak_container_memory_mb",
                "quantile": _QUANTILE,
                "repo_cluster_key": "task_id with trailing -<digits> stripped",
            },
            "fit_labels": _label_counts(fit_samples),
            "eval_labels": _label_counts(eval_samples),
            "comparison": comparison,
            "registered_criterion": {
                "definition": (
                    "repo-clustered prefix-free ambient residual vs tool absolute "
                    "p90 skill bootstrap CI lower bound > 0"
                ),
                "confidence_level": _CONFIDENCE_LEVEL,
                "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
                "seed": _SEED,
                "evaluated": not smoke,
                "passed": (
                    comparison["repo_clustered_skill"]["ci_low"] > 0.0
                    if not smoke
                    else None
                ),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return

    comparison = evaluate_peak_memory(fit_samples, eval_samples)
    result = {
        "status": (
            "SMOKE ONLY - limited tasks; metrics are plumbing checks, not findings"
            if smoke
            else "FULL PRIMARY - registered peak-memory comparison"
        ),
        "config": {
            "fit": str(args.fit.resolve()),
            "eval": str(args.eval.resolve()),
            "fit_tasks": str(args.fit_tasks.resolve()),
            "eval_tasks": str(args.eval_tasks.resolve()),
            "limit_tasks": args.limit_tasks,
            "target": "peak_container_memory_mb",
            "quantile": _QUANTILE,
            "command_field": _COMMAND_FIELD,
            "max_prefix_depth": _MAX_PREFIX_DEPTH,
            "min_tool_history": _MIN_TOOL_HISTORY,
            "min_profile_tasks": _MIN_PROFILE_TASKS,
        },
        "fit_labels": _label_counts(fit_samples),
        "eval_labels": _label_counts(eval_samples),
        "comparison": comparison,
        "registered_criterion": {
            "definition": "prefix-vs-tool p90 skill bootstrap CI lower bound > 0",
            "evaluated": not smoke,
            "passed": (
                comparison["prefix_vs_tool_p90_skill"]["ci_low"] > 0.0
                if not smoke
                else None
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
