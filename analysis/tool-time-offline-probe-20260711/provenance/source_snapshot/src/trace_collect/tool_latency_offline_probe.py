"""Nested task-OOF calibration for a three-region utility-clock guard."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from trace_collect.command_features import make_row_command_prefix_keys
from trace_collect.latency_validation import (
    normalized_positive_floats,
    required_nonnegative_float,
    required_text,
)
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
from trace_collect.tool_latency_profiled import (
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
    validate_profile_eval_disjoint,
)
from trace_collect.tool_latency_utility_clock import (
    evaluate_utility_clock_policy,
    trigger_policy_utility_ms,
)


POLICY_TRIGGER_FIELDS = {
    "deadline_only": "deadline_trigger_ms",
    "mean_hazard": "mean_hazard_trigger_ms",
    "robust_clock": "robust_trigger_ms",
    "offline_probe_guard": "offline_probe_trigger_ms",
}


@dataclass(frozen=True)
class MeanClockRegionStats:
    """Profile-only utility and three-region statistics for one clock."""

    trigger_ms: float
    normalized_margin: float
    normalized_band_gain: float
    normalized_short_penalty: float
    survivor_count: int
    probability_short_given_survival: float | None
    probability_band_given_survival: float | None
    probability_far_given_survival: float | None


def mean_clock_region_stats(
    values: Sequence[float],
    *,
    threshold_ms: float,
    kv_cost_ms: float,
) -> MeanClockRegionStats:
    """Project one empirical latency prior into decision-relevant regions."""
    if not values:
        raise ValueError("mean-clock region statistics require profile samples")
    if not math.isfinite(threshold_ms) or threshold_ms <= 0.0:
        raise ValueError("threshold_ms must be positive and finite")
    if not math.isfinite(kv_cost_ms) or kv_cost_ms <= 0.0:
        raise ValueError("kv_cost_ms must be positive and finite")
    samples = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("profile latency samples must be finite and non-negative")

    trigger_ms = hazard_recheck_ms(
        samples,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
    )
    band_gain_ms = 0.0
    short_penalty_ms = 0.0
    survivors: list[float] = []
    for latency_ms in samples:
        candidate_utility = trigger_policy_utility_ms(
            latency_ms,
            trigger_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
        )
        deadline_utility = trigger_policy_utility_ms(
            latency_ms,
            threshold_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
        )
        delta_ms = candidate_utility - deadline_utility
        if threshold_ms < latency_ms < threshold_ms + kv_cost_ms:
            if delta_ms < -1e-9:
                raise AssertionError("candidate loses utility on a boundary call")
            band_gain_ms += delta_ms
        elif latency_ms <= threshold_ms:
            if delta_ms > 1e-9:
                raise AssertionError("candidate gains utility on a short call")
            short_penalty_ms -= delta_ms
        elif not math.isclose(delta_ms, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError("far-tail candidate delta must be zero")
        if latency_ms > trigger_ms:
            survivors.append(latency_ms)

    denominator = len(samples) * kv_cost_ms
    normalized_band_gain = band_gain_ms / denominator
    normalized_short_penalty = short_penalty_ms / denominator
    normalized_margin = normalized_band_gain - normalized_short_penalty
    if normalized_margin < -1e-12:
        raise AssertionError("mean-utility trigger cannot underperform the deadline")
    normalized_margin = max(0.0, normalized_margin)

    survivor_count = len(survivors)
    if survivor_count:
        short_count = sum(value <= threshold_ms for value in survivors)
        band_count = sum(
            threshold_ms < value < threshold_ms + kv_cost_ms for value in survivors
        )
        far_count = sum(value >= threshold_ms + kv_cost_ms for value in survivors)
        if short_count + band_count + far_count != survivor_count:
            raise AssertionError("three latency regions do not partition survivors")
        p_short = short_count / survivor_count
        p_band = band_count / survivor_count
        p_far = far_count / survivor_count
    else:
        p_short = p_band = p_far = None
    return MeanClockRegionStats(
        trigger_ms=trigger_ms,
        normalized_margin=normalized_margin,
        normalized_band_gain=normalized_band_gain,
        normalized_short_penalty=normalized_short_penalty,
        survivor_count=survivor_count,
        probability_short_given_survival=p_short,
        probability_band_given_survival=p_band,
        probability_far_given_survival=p_far,
    )


def select_probe_guard(
    probe_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a dimensionless margin guard by cross-fitted probe utility."""
    eligible: list[tuple[float, float, str]] = []
    probe_tasks: set[str] = set()
    for index, row in enumerate(probe_decisions):
        source = f"probe decision {index}"
        task_id = required_text(row, "task_id", source=source)
        probe_tasks.add(task_id)
        score = float(row["probe_margin_normalized"])
        candidate_ms = float(row["probe_candidate_trigger_ms"])
        threshold_ms = float(row["threshold_ms"])
        cost_ms = float(row["kv_cost_ms"])
        latency_ms = float(row["latency_ms"])
        if not all(
            math.isfinite(value)
            for value in (score, candidate_ms, threshold_ms, cost_ms, latency_ms)
        ):
            raise ValueError(f"{source} contains non-finite values")
        if score < 0.0 or not 0.0 <= candidate_ms <= threshold_ms or cost_ms <= 0.0:
            raise ValueError(f"{source} contains an invalid score/trigger/cost")
        if candidate_ms >= threshold_ms or score <= 0.0:
            continue
        delta_normalized = (
            trigger_policy_utility_ms(
                latency_ms,
                candidate_ms,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
            )
            - trigger_policy_utility_ms(
                latency_ms,
                threshold_ms,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
            )
        ) / cost_ms
        eligible.append((score, delta_normalized, task_id))

    candidate_guards = sorted({0.0, *(score for score, _, _ in eligible)})
    best_guard: float | None = None
    best_objective = 0.0
    for guard in candidate_guards:
        objective = math.fsum(delta for score, delta, _ in eligible if score > guard)
        if objective > best_objective + 1e-12:
            best_guard = guard
            best_objective = objective
        elif math.isclose(objective, best_objective, rel_tol=0.0, abs_tol=1e-12):
            if best_guard is not None and guard > best_guard:
                best_guard = guard

    accepted = [
        (delta, task_id)
        for score, delta, task_id in eligible
        if best_guard is not None and score > best_guard
    ]
    normalized_delta_by_task: dict[str, float] = defaultdict(float)
    for delta, task_id in accepted:
        normalized_delta_by_task[task_id] += delta
    return {
        "selected_guard_normalized": best_guard,
        "probe_objective_normalized": best_objective,
        "probe_decision_count": len(probe_decisions),
        "probe_task_count": len(probe_tasks),
        "eligible_early_decision_count": len(eligible),
        "accepted_early_decision_count": len(accepted),
        "candidate_guard_count": len(candidate_guards) + 1,
        "accepted_task_count": len(normalized_delta_by_task),
        "worst_accepted_task_normalized_delta": (
            min(normalized_delta_by_task.values()) if normalized_delta_by_task else None
        ),
    }


def evaluate_offline_probe_clock(
    eval_rows: Iterable[dict[str, Any]],
    *,
    profile_rows: Iterable[dict[str, Any]],
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    inner_folds: int,
    min_tool_history: int = 1,
    min_profile_tasks: int = 1,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
) -> dict[str, Any]:
    """Learn a profile-only guard and evaluate it on disjoint outer tasks."""
    kv_costs = normalized_positive_floats(kv_costs_ms, label="kv cost")
    _validate_config(
        guard_ms=guard_ms,
        inner_folds=inner_folds,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        max_prefix_depth=max_prefix_depth,
    )
    profile_list = list(profile_rows)
    eval_list = list(eval_rows)
    profile_folds = _balanced_task_folds(profile_list, fold_count=inner_folds)
    probe_decisions: list[dict[str, Any]] = []
    all_profile_tasks = set().union(*profile_folds)
    for held_out_tasks in profile_folds:
        inner_profile = [
            row for row in profile_list if str(row["task_id"]) not in held_out_tasks
        ]
        inner_eval = [
            row for row in profile_list if str(row["task_id"]) in held_out_tasks
        ]
        probe_decisions.extend(
            _score_mean_clock_rows(
                inner_eval,
                profile_rows=inner_profile,
                kv_costs=kv_costs,
                guard_ms=guard_ms,
                min_tool_history=min_tool_history,
                min_profile_tasks=min_profile_tasks,
                command_field=command_field,
                max_prefix_depth=max_prefix_depth,
                skip_leading_cd=skip_leading_cd,
            )
        )
    if {str(row["task_id"]) for row in profile_list} != all_profile_tasks:
        raise AssertionError("inner folds do not cover every profile task")
    calibration = select_probe_guard(probe_decisions)

    baseline = evaluate_utility_clock_policy(
        eval_list,
        profile_rows=profile_list,
        kv_costs_ms=kv_costs,
        guard_ms=guard_ms,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    scored_rows = _score_mean_clock_rows(
        eval_list,
        profile_rows=profile_list,
        kv_costs=kv_costs,
        guard_ms=guard_ms,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    scored_by_key = {
        (str(row["sample_id"]), float(row["kv_cost_ms"])): row for row in scored_rows
    }
    selected_guard = calibration["selected_guard_normalized"]
    decisions: list[dict[str, Any]] = []
    for baseline_row in baseline["decisions"]:
        key = (
            str(baseline_row["sample_id"]),
            float(baseline_row["kv_cost_ms"]),
        )
        scored = scored_by_key.pop(key)
        if not math.isclose(
            float(baseline_row["mean_hazard_trigger_ms"]),
            float(scored["probe_candidate_trigger_ms"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AssertionError("scored candidate differs from mean-hazard trigger")
        if (
            baseline_row["prior_source"] != scored["probe_prior_source"]
            or baseline_row["prior_group_key"] != scored["probe_prior_group_key"]
            or baseline_row["prior_task_count"] != scored["probe_prior_task_count"]
        ):
            raise AssertionError("scored candidate selected a different prior node")
        candidate_ms = float(scored["probe_candidate_trigger_ms"])
        threshold_ms = float(scored["threshold_ms"])
        margin = float(scored["probe_margin_normalized"])
        use_candidate = (
            selected_guard is not None
            and candidate_ms < threshold_ms
            and margin > selected_guard
        )
        decisions.append(
            {
                **baseline_row,
                **{
                    key: value
                    for key, value in scored.items()
                    if key.startswith("probe_")
                },
                "offline_probe_guard_normalized": selected_guard,
                "offline_probe_trigger_ms": (
                    candidate_ms if use_candidate else threshold_ms
                ),
            }
        )
    if scored_by_key:
        raise AssertionError("scored decisions were not matched to baseline rows")

    summary = summarize_offline_probe_decisions(decisions)
    return {
        "policies": list(POLICY_TRIGGER_FIELDS),
        "kv_costs_ms": kv_costs,
        "guard_ms": guard_ms,
        "inner_folds": inner_folds,
        "min_tool_history": min_tool_history,
        "min_profile_tasks": min_profile_tasks,
        "command_field": command_field,
        "max_prefix_depth": max_prefix_depth,
        "skip_leading_cd": skip_leading_cd,
        "profile_row_count": len(profile_list),
        "profile_task_count": len(all_profile_tasks),
        "row_count": len(eval_list),
        "calibration": calibration,
        "points": summary["points"],
        "decisions": decisions,
    }


def load_and_evaluate_offline_probe_clock(
    eval_path: Path,
    *,
    profile_path: Path,
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    inner_folds: int,
    min_tool_history: int = 1,
    min_profile_tasks: int = 1,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
) -> dict[str, Any]:
    """Load latency JSONLs and evaluate the offline-probe clock."""
    return evaluate_offline_probe_clock(
        read_tool_latency_jsonl(eval_path),
        profile_rows=read_tool_latency_jsonl(profile_path),
        kv_costs_ms=kv_costs_ms,
        guard_ms=guard_ms,
        inner_folds=inner_folds,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )


def summarize_offline_probe_decisions(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute pooled policy utility from raw offline-probe decisions."""
    if not decisions:
        raise ValueError("offline-probe decisions must be non-empty")
    by_cost: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, float]] = set()
    for row in decisions:
        cost = float(row["kv_cost_ms"])
        key = (str(row["sample_id"]), cost)
        if key in seen:
            raise ValueError(f"duplicate offline-probe sample/cost: {key}")
        seen.add(key)
        by_cost[cost].append(row)
    points = {
        str(cost): _summarize_cost(rows) for cost, rows in sorted(by_cost.items())
    }
    return {
        "sample_count": len({str(row["sample_id"]) for row in decisions}),
        "task_count": len({str(row["task_id"]) for row in decisions}),
        "costs_ms": sorted(by_cost),
        "points": points,
    }


def aggregate_offline_probe_cv(
    cv_root: Path,
    *,
    expected_fold_count: int,
) -> dict[str, Any]:
    """Pool fold decisions and learned guards from one corpus directory."""
    if expected_fold_count < 1:
        raise ValueError("expected_fold_count must be positive")
    expected_summary_names = {
        f"f{fold}_summary.json" for fold in range(1, expected_fold_count + 1)
    }
    expected_decision_names = {
        f"f{fold}_decisions.jsonl" for fold in range(1, expected_fold_count + 1)
    }
    actual_summary_names = {path.name for path in cv_root.glob("f*_summary.json")}
    actual_decision_names = {path.name for path in cv_root.glob("f*_decisions.jsonl")}
    if actual_summary_names != expected_summary_names:
        raise ValueError(
            "CV summary folds do not match the expected set: "
            f"actual={sorted(actual_summary_names)}, "
            f"expected={sorted(expected_summary_names)}"
        )
    if actual_decision_names != expected_decision_names:
        raise ValueError(
            "CV decision folds do not match the expected set: "
            f"actual={sorted(actual_decision_names)}, "
            f"expected={sorted(expected_decision_names)}"
        )
    decisions: list[dict[str, Any]] = []
    fold_calibrations: list[dict[str, Any]] = []
    common_config: dict[str, Any] | None = None
    outer_tasks: set[str] = set()
    for fold in range(1, expected_fold_count + 1):
        summary_path = cv_root / f"f{fold}_summary.json"
        decision_path = cv_root / f"f{fold}_decisions.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"fold f{fold} summary is not a JSON object")
        with decision_path.open("r", encoding="utf-8") as handle:
            fold_decisions = [json.loads(line) for line in handle if line.strip()]
        fold_config = _validate_fold_output(
            summary,
            fold_decisions,
            fold=f"f{fold}",
        )
        if common_config is None:
            common_config = fold_config
        elif fold_config != common_config:
            raise ValueError(f"fold f{fold} configuration differs from earlier folds")
        fold_tasks = {str(row["task_id"]) for row in fold_decisions}
        overlap = outer_tasks & fold_tasks
        if overlap:
            raise ValueError(
                "outer eval tasks must be disjoint across folds; overlap: "
                f"{sorted(overlap)}"
            )
        outer_tasks.update(fold_tasks)
        fold_calibrations.append(
            {
                "fold": f"f{fold}",
                **summary["calibration"],
            }
        )
        decisions.extend(fold_decisions)
    pooled = summarize_offline_probe_decisions(decisions)
    if common_config is None:
        raise AssertionError("validated CV folds produced no configuration")
    if pooled["costs_ms"] != common_config["kv_costs_ms"]:
        raise AssertionError("pooled cost panel differs from validated fold config")
    pooled["fold_count"] = expected_fold_count
    pooled["fold_calibrations"] = fold_calibrations
    pooled["policies"] = list(POLICY_TRIGGER_FIELDS)
    pooled["config"] = common_config
    return pooled


def _validate_fold_output(
    summary: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    fold: str,
) -> dict[str, Any]:
    config_fields = (
        "kv_costs_ms",
        "guard_ms",
        "inner_folds",
        "min_tool_history",
        "min_profile_tasks",
        "command_field",
        "max_prefix_depth",
        "skip_leading_cd",
    )
    missing = [field for field in config_fields if field not in summary]
    if missing:
        raise ValueError(f"{fold} summary is missing config fields: {missing}")
    config = {field: summary[field] for field in config_fields}
    costs = [float(cost) for cost in config["kv_costs_ms"]]
    if not costs or len(costs) != len(set(costs)):
        raise ValueError(f"{fold} summary has invalid kv_costs_ms")
    if summary.get("policies") != list(POLICY_TRIGGER_FIELDS):
        raise ValueError(f"{fold} summary policy list is inconsistent")
    if not isinstance(summary.get("calibration"), dict):
        raise ValueError(f"{fold} summary is missing calibration")
    calibration = summary["calibration"]
    profile_row_count = int(summary.get("profile_row_count", -1))
    profile_task_count = int(summary.get("profile_task_count", -1))
    if profile_row_count < 1 or profile_task_count < 1:
        raise ValueError(f"{fold} summary has invalid profile counts")
    _validate_calibration(
        calibration,
        fold=fold,
        expected_probe_decision_count=profile_row_count * len(costs),
        expected_probe_task_count=profile_task_count,
    )
    selected_guard = calibration["selected_guard_normalized"]
    if not decisions:
        raise ValueError(f"{fold} decisions are empty")

    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_sample[str(row["sample_id"])].append(row)
    if len(by_sample) != int(summary.get("row_count", -1)):
        raise ValueError(f"{fold} summary row_count differs from raw decisions")
    if len(decisions) != len(by_sample) * len(costs):
        raise ValueError(f"{fold} decision count is not sample_count * cost_count")
    expected_costs = set(costs)
    for sample_id, panel in by_sample.items():
        panel_costs = [float(row["kv_cost_ms"]) for row in panel]
        if (
            len(panel_costs) != len(set(panel_costs))
            or set(panel_costs) != expected_costs
        ):
            raise ValueError(
                f"{fold} sample {sample_id!r} has an incomplete cost panel"
            )
        reference = panel[0]
        metadata = (
            str(reference["task_id"]),
            str(reference["tool_name"]),
            float(reference["latency_ms"]),
        )
        for row in panel:
            if (
                str(row["task_id"]),
                str(row["tool_name"]),
                float(row["latency_ms"]),
            ) != metadata:
                raise ValueError(
                    f"{fold} sample {sample_id!r} metadata changes across costs"
                )
            threshold_ms = float(row["threshold_ms"])
            latency_ms = float(row["latency_ms"])
            if bool(row["label_exceeds_threshold"]) != (latency_ms > threshold_ms):
                raise ValueError(f"{fold} sample {sample_id!r} has an invalid label")
            if row.get("offline_probe_guard_normalized") != selected_guard:
                raise ValueError(f"{fold} decision guard differs from its summary")

    recomputed = summarize_offline_probe_decisions(decisions)
    if recomputed["costs_ms"] != costs:
        raise ValueError(f"{fold} recomputed costs differ from its summary")
    if recomputed["points"] != summary.get("points"):
        raise ValueError(f"{fold} summary points differ from raw recomputation")
    return config


def _validate_calibration(
    calibration: Mapping[str, Any],
    *,
    fold: str,
    expected_probe_decision_count: int,
    expected_probe_task_count: int,
) -> None:
    fields = {
        "selected_guard_normalized",
        "probe_objective_normalized",
        "probe_decision_count",
        "probe_task_count",
        "eligible_early_decision_count",
        "accepted_early_decision_count",
        "candidate_guard_count",
        "accepted_task_count",
        "worst_accepted_task_normalized_delta",
    }
    missing = sorted(fields - calibration.keys())
    if missing:
        raise ValueError(f"{fold} calibration is missing fields: {missing}")
    guard = calibration["selected_guard_normalized"]
    if guard is not None and (
        not isinstance(guard, (int, float))
        or isinstance(guard, bool)
        or not math.isfinite(float(guard))
        or float(guard) < 0.0
    ):
        raise ValueError(f"{fold} calibration guard is invalid")
    objective = calibration["probe_objective_normalized"]
    if (
        not isinstance(objective, (int, float))
        or isinstance(objective, bool)
        or not math.isfinite(float(objective))
        or float(objective) < 0.0
    ):
        raise ValueError(f"{fold} calibration objective is invalid")

    count_fields = (
        "probe_decision_count",
        "probe_task_count",
        "eligible_early_decision_count",
        "accepted_early_decision_count",
        "candidate_guard_count",
        "accepted_task_count",
    )
    counts: dict[str, int] = {}
    for field in count_fields:
        value = calibration[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{fold} calibration {field} is invalid")
        counts[field] = value
    if counts["probe_decision_count"] < 1 or counts["probe_task_count"] < 1:
        raise ValueError(f"{fold} calibration probe counts must be positive")
    if counts["probe_decision_count"] != expected_probe_decision_count:
        raise ValueError(f"{fold} calibration probe decision count is inconsistent")
    if counts["probe_task_count"] != expected_probe_task_count:
        raise ValueError(f"{fold} calibration probe task count is inconsistent")
    if counts["candidate_guard_count"] < 2:
        raise ValueError(f"{fold} calibration must include zero and never guards")
    if counts["eligible_early_decision_count"] > counts["probe_decision_count"]:
        raise ValueError(f"{fold} calibration eligible count exceeds probe count")
    if (
        counts["accepted_early_decision_count"]
        > counts["eligible_early_decision_count"]
    ):
        raise ValueError(f"{fold} calibration accepted count exceeds eligible count")
    if counts["accepted_task_count"] > counts["probe_task_count"]:
        raise ValueError(f"{fold} calibration accepted task count exceeds probe tasks")
    if counts["accepted_task_count"] > counts["accepted_early_decision_count"]:
        raise ValueError(
            f"{fold} calibration accepted task count exceeds accepted decisions"
        )
    if (counts["accepted_early_decision_count"] == 0) != (
        counts["accepted_task_count"] == 0
    ):
        raise ValueError(
            f"{fold} calibration accepted decision/task zero states differ"
        )

    worst = calibration["worst_accepted_task_normalized_delta"]
    if counts["accepted_task_count"] == 0:
        if worst is not None:
            raise ValueError(f"{fold} calibration has a worst task without acceptance")
    elif (
        not isinstance(worst, (int, float))
        or isinstance(worst, bool)
        or not math.isfinite(float(worst))
    ):
        raise ValueError(f"{fold} calibration worst accepted task is invalid")
    if guard is None and (
        counts["accepted_early_decision_count"] != 0
        or not math.isclose(float(objective), 0.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{fold} null guard must disable early probe decisions")
    if guard is not None and (
        float(objective) <= 1e-12
        or counts["accepted_early_decision_count"] == 0
        or counts["accepted_task_count"] == 0
    ):
        raise ValueError(
            f"{fold} non-null guard requires positive accepted probe utility"
        )


def _score_mean_clock_rows(
    eval_rows: Sequence[dict[str, Any]],
    *,
    profile_rows: Sequence[dict[str, Any]],
    kv_costs: Sequence[float],
    guard_ms: float,
    min_tool_history: int,
    min_profile_tasks: int,
    command_field: str | None,
    max_prefix_depth: int,
    skip_leading_cd: bool,
) -> list[dict[str, Any]]:
    row_group_keys = (
        make_row_command_prefix_keys(
            command_field,
            max_depth=max_prefix_depth,
            skip_leading_cd=skip_leading_cd,
        )
        if command_field is not None
        else None
    )
    prior = build_latency_prior(profile_rows, row_group_keys=row_group_keys)
    if len(prior.task_ids) < min_profile_tasks:
        raise ValueError("inner profile has fewer tasks than min_profile_tasks")
    task_by_sample = validate_profile_eval_disjoint(eval_rows, prior=prior)
    cache: dict[tuple[int, float, float], MeanClockRegionStats] = {}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(eval_rows):
        source = f"scored eval row {index}"
        sample_id = required_text(row, "sample_id", source=source)
        if sample_id in seen:
            raise ValueError(f"duplicate scored eval sample_id: {sample_id!r}")
        seen.add(sample_id)
        tool_name = required_text(row, "tool_name", source=source)
        latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
        group_keys = row_group_keys(row) if row_group_keys is not None else ()
        hierarchy = latency_prior_hierarchy(
            prior,
            tool_name,
            group_keys,
            min_tool_history=min_tool_history,
            min_profile_tasks=min_profile_tasks,
        )
        selected = hierarchy[-1]
        for cost_ms in kv_costs:
            threshold_ms = cost_ms + guard_ms
            key = (id(selected.values), threshold_ms, cost_ms)
            stats = cache.get(key)
            if stats is None:
                stats = mean_clock_region_stats(
                    selected.values,
                    threshold_ms=threshold_ms,
                    kv_cost_ms=cost_ms,
                )
                cache[key] = stats
            output.append(
                {
                    "sample_id": sample_id,
                    "task_id": task_by_sample[sample_id],
                    "tool_name": tool_name,
                    "latency_ms": latency_ms,
                    "kv_cost_ms": cost_ms,
                    "threshold_ms": threshold_ms,
                    "probe_candidate_trigger_ms": stats.trigger_ms,
                    "probe_margin_normalized": stats.normalized_margin,
                    "probe_band_gain_normalized": stats.normalized_band_gain,
                    "probe_short_penalty_normalized": (stats.normalized_short_penalty),
                    "probe_survivor_count": stats.survivor_count,
                    "probe_probability_short_given_survival": (
                        stats.probability_short_given_survival
                    ),
                    "probe_probability_band_given_survival": (
                        stats.probability_band_given_survival
                    ),
                    "probe_probability_far_given_survival": (
                        stats.probability_far_given_survival
                    ),
                    "probe_prior_source": selected.source,
                    "probe_prior_group_key": selected.group_key,
                    "probe_prior_task_count": len(selected.values_by_task),
                }
            )
    return output


def _balanced_task_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_count: int,
) -> list[set[str]]:
    rows_by_task: dict[str, int] = Counter()
    for index, row in enumerate(rows):
        task_id = required_text(row, "task_id", source=f"profile row {index}")
        rows_by_task[task_id] += 1
    if len(rows_by_task) < fold_count:
        raise ValueError(
            f"inner_folds={fold_count} exceeds profile task count {len(rows_by_task)}"
        )
    folds = [set() for _ in range(fold_count)]
    fold_rows = [0] * fold_count
    for task_id, row_count in sorted(
        rows_by_task.items(), key=lambda item: (-item[1], item[0])
    ):
        fold_index = min(
            range(fold_count),
            key=lambda index: (fold_rows[index], len(folds[index]), index),
        )
        folds[fold_index].add(task_id)
        fold_rows[fold_index] += row_count
    if any(not fold for fold in folds):
        raise AssertionError("balanced task split produced an empty fold")
    return folds


def _summarize_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cost_ms = float(rows[0]["kv_cost_ms"])
    threshold_ms = float(rows[0]["threshold_ms"])
    if any(
        float(row["kv_cost_ms"]) != cost_ms
        or float(row["threshold_ms"]) != threshold_ms
        for row in rows
    ):
        raise ValueError("cost point contains inconsistent cost/threshold values")
    positive_count = sum(float(row["latency_ms"]) > threshold_ms for row in rows)
    band_rows = [
        row
        for row in rows
        if threshold_ms < float(row["latency_ms"]) < threshold_ms + cost_ms
    ]
    oracle_ms = positive_count * cost_ms
    headroom_ms = math.fsum(
        2.0 * (cost_ms - (float(row["latency_ms"]) - threshold_ms)) for row in band_rows
    )
    policies = {
        policy: _summarize_policy(
            rows,
            trigger_field=trigger_field,
            threshold_ms=threshold_ms,
            cost_ms=cost_ms,
            headroom_ms=headroom_ms,
            oracle_ms=oracle_ms,
        )
        for policy, trigger_field in POLICY_TRIGGER_FIELDS.items()
    }
    deadline_net = policies["deadline_only"]["net_saved_ms"]
    if not math.isclose(
        oracle_ms - deadline_net,
        headroom_ms,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise AssertionError("offline-probe deadline headroom identity failed")
    return {
        "kv_cost_ms": cost_ms,
        "threshold_ms": threshold_ms,
        "call_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "positive_count": positive_count,
        "band_count": len(band_rows),
        "oracle_ms": oracle_ms,
        "deadline_headroom_ms": headroom_ms,
        "policies": policies,
    }


def _summarize_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    trigger_field: str,
    threshold_ms: float,
    cost_ms: float,
    headroom_ms: float,
    oracle_ms: float,
) -> dict[str, Any]:
    net_ms = 0.0
    deadline_net_ms = 0.0
    early_count = 0
    early_short_count = 0
    band_gain_ms = 0.0
    short_penalty_ms = 0.0
    for row in rows:
        latency_ms = float(row["latency_ms"])
        trigger_ms = float(row[trigger_field])
        if not 0.0 <= trigger_ms <= threshold_ms:
            raise ValueError(f"invalid trigger {trigger_ms} in {trigger_field}")
        utility = trigger_policy_utility_ms(
            latency_ms,
            trigger_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=cost_ms,
        )
        deadline_utility = trigger_policy_utility_ms(
            latency_ms,
            threshold_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=cost_ms,
        )
        delta_ms = utility - deadline_utility
        net_ms += utility
        deadline_net_ms += deadline_utility
        if latency_ms > trigger_ms and trigger_ms < threshold_ms:
            early_count += 1
            if latency_ms <= threshold_ms:
                early_short_count += 1
        if threshold_ms < latency_ms < threshold_ms + cost_ms:
            band_gain_ms += delta_ms
        elif latency_ms <= threshold_ms:
            short_penalty_ms -= delta_ms
        elif not math.isclose(delta_ms, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError("far-tail policy delta must be zero")
    delta_ms = net_ms - deadline_net_ms
    if not math.isclose(
        delta_ms,
        band_gain_ms - short_penalty_ms,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise AssertionError("offline-probe policy decomposition failed")
    return {
        "net_saved_ms": net_ms,
        "delta_vs_deadline_ms": delta_ms,
        "net_fraction_of_oracle": net_ms / oracle_ms if oracle_ms else None,
        "captured_fraction_of_deadline_headroom": (
            delta_ms / headroom_ms if headroom_ms else None
        ),
        "early_trigger_count": early_count,
        "early_trigger_on_short_count": early_short_count,
        "band_gain_ms": band_gain_ms,
        "short_exposure_penalty_ms": short_penalty_ms,
    }


def _validate_config(
    *,
    guard_ms: float,
    inner_folds: int,
    min_tool_history: int,
    min_profile_tasks: int,
    max_prefix_depth: int,
) -> None:
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError("guard_ms must be finite and non-negative")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if min_tool_history < 1 or min_profile_tasks < 1 or max_prefix_depth < 1:
        raise ValueError("history/task/depth configuration must be positive")


__all__ = [
    "MeanClockRegionStats",
    "aggregate_offline_probe_cv",
    "evaluate_offline_probe_clock",
    "load_and_evaluate_offline_probe_clock",
    "mean_clock_region_stats",
    "select_probe_guard",
    "summarize_offline_probe_decisions",
]
