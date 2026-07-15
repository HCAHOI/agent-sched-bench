"""Restore-cost sensitivity analyses for fitted trigger policies.

The deployed utility model charges nothing for a swap that completes inside a
call which then finishes before the deadline, although a real system must
swap state back on the critical path. Two analyses stress this:

* Mode A (:func:`analyze_restore_cost_sweep`) re-scores existing decisions —
  triggers stay exactly as fitted at restore cost zero, and only the
  evaluation utility charges each fire on a short call
  ``fraction * kv_cost_ms``. Gains that vanish under realistic fractions
  were artifacts of the free unnecessary swap.
* Mode B (:func:`run_mode_b_refit`) refits every stage — inner probe
  scoring, guard selection, and outer triggers — at each fraction and
  scores at the same fraction, answering whether the method can adapt to an
  honest swap-back cost. It reuses the frozen fold splits and latency
  JSONLs, and asserts that the fraction-zero refit reproduces the frozen
  triggers exactly.

Output payloads keep ``schema_version`` 1 with additive keys only; consumers
must tolerate unknown keys.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from trace_collect.tool_latency_confirmation import (
    _resample_task_totals,
    paired_task_cluster_bootstrap,
)
from trace_collect.tool_latency_context import context_lengths_from_traces
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
from trace_collect.tool_latency_recompute import (
    effective_min_restore_ms,
    recompute_restore_ms,
    validate_recompute_rate,
)
from trace_collect.tool_latency_utility_clock import trigger_policy_utility_ms
from trace_collect.tool_latency_hazard_eval import (
    HAZARD_COMPARISONS,
    HAZARD_ENSEMBLE_COMPARISONS,
    build_survival_feature_spec,
    evaluate_hazard_model_clock,
)
from trace_collect.tool_latency_offline_probe import (
    evaluate_offline_probe_clock,
    select_probe_guard,
)
from trace_collect.tool_latency_within_task import within_task_trigger_rows


_FOLD_DECISIONS_PATTERN = re.compile(r"^f(\d+)_decisions\.jsonl$")

# (name, treatment field, baseline field, enforce gated-treatment invariant).
# The first comparison re-scores the frozen certification contrast; the other
# two measure each early policy against never firing early at all.
COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    ("gated_vs_robust", "offline_gated_robust_trigger_ms", "robust_trigger_ms", True),
    (
        "gated_vs_deadline",
        "offline_gated_robust_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    ("robust_vs_deadline", "robust_trigger_ms", "deadline_trigger_ms", False),
)

_FROZEN_TRIGGER_FIELDS = (
    "deadline_trigger_ms",
    "mean_hazard_trigger_ms",
    "robust_trigger_ms",
    "offline_probe_trigger_ms",
    "offline_gated_robust_trigger_ms",
)


def analyze_restore_cost_sweep(
    confirmation_root: Path,
    *,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Re-run the paired task-cluster bootstrap for each restore fraction."""

    _validate_fractions(restore_cost_fractions)
    decisions, fold_names = load_fold_decisions(confirmation_root)
    costs = sorted({float(row["kv_cost_ms"]) for row in decisions})

    comparisons: dict[str, dict[str, Any]] = {}
    for name, treatment_field, baseline_field, enforce_gated in COMPARISONS:
        by_fraction: dict[str, Any] = {}
        for fraction in restore_cost_fractions:
            by_fraction[fraction_key(fraction)] = paired_task_cluster_bootstrap(
                decisions,
                costs_ms=costs,
                replicates=replicates,
                confidence_level=confidence_level,
                seed=seed,
                baseline_trigger_field=baseline_field,
                treatment_trigger_field=treatment_field,
                restore_cost_fraction=fraction,
                enforce_gated_treatment=enforce_gated,
            )
        comparisons[name] = {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": by_fraction,
        }

    return {
        "schema_version": 1,
        "mode": "rescore_frozen_triggers",
        "confirmation_root": str(confirmation_root.resolve()),
        "fold_count": len(fold_names),
        "fold_names": fold_names,
        "decision_row_count": len(decisions),
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }


def run_mode_b_refit(
    confirmation_root: Path,
    *,
    output_root: Path,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Refit per fold at each fraction, then bootstrap the paired contrasts."""

    _validate_fractions(restore_cost_fractions)
    confirmation_root = confirmation_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")
    manifest = _read_frozen_manifest(confirmation_root)
    fold_count = manifest["fold_count"]
    costs = [float(cost) for cost in manifest["costs_ms"]]

    fold_data: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for fold in range(1, fold_count + 1):
        eval_path = confirmation_root / "data" / f"f{fold}_eval.jsonl"
        profile_path = confirmation_root / "data" / f"f{fold}_profile.jsonl"
        fold_data.append(
            (
                f"f{fold}",
                list(read_tool_latency_jsonl(eval_path)),
                list(read_tool_latency_jsonl(profile_path)),
            )
        )
    frozen_decisions, _ = load_fold_decisions(confirmation_root)
    frozen_triggers = _index_triggers(frozen_decisions)

    output_root.mkdir(parents=True)
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in COMPARISONS
    }
    calibrations_by_fraction: dict[str, list[dict[str, Any]]] = {}
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        fraction_root = output_root / f"rho_{key}"
        fraction_root.mkdir()
        decisions: list[dict[str, Any]] = []
        calibrations: list[dict[str, Any]] = []
        for fold_name, eval_rows, profile_rows in fold_data:
            result = evaluate_offline_probe_clock(
                eval_rows,
                profile_rows=profile_rows,
                kv_costs_ms=costs,
                guard_ms=manifest["guard_ms"],
                inner_folds=manifest["inner_folds"],
                min_tool_history=manifest["min_tool_history"],
                min_profile_tasks=manifest["min_profile_tasks"],
                command_field=manifest["command_field"],
                max_prefix_depth=manifest["max_prefix_depth"],
                skip_leading_cd=manifest["skip_leading_cd"],
                restore_cost_fraction=fraction,
            )
            fold_decisions = result.pop("decisions")
            _write_json(fraction_root / f"{fold_name}_summary.json", result)
            _write_jsonl(fraction_root / f"{fold_name}_decisions.jsonl", fold_decisions)
            calibrations.append(
                {
                    "fold": fold_name,
                    "calibration": result["calibration"],
                    "robust_calibration": result["robust_calibration"],
                }
            )
            decisions.extend(
                {**row, "outer_fold": fold_name} for row in fold_decisions
            )
        if fraction == 0.0:
            _assert_matches_frozen_triggers(decisions, frozen_triggers)
        calibrations_by_fraction[key] = calibrations
        for name, treatment_field, baseline_field, enforce_gated in COMPARISONS:
            comparisons[name]["by_restore_cost_fraction"][key] = (
                paired_task_cluster_bootstrap(
                    decisions,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_field,
                    treatment_trigger_field=treatment_field,
                    restore_cost_fraction=fraction,
                    enforce_gated_treatment=enforce_gated,
                )
            )

    expected_row_count = sum(len(rows) for _, rows, _ in fold_data) * len(costs)
    if len(decisions) != expected_row_count:
        raise AssertionError(
            "refit decision count does not match eval rows x costs: "
            f"{len(decisions)} != {expected_row_count}"
        )
    result = {
        "schema_version": 1,
        "mode": "refit_with_restore_cost",
        "confirmation_root": str(confirmation_root),
        "fold_count": fold_count,
        "decision_row_count": expected_row_count,
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "frozen_zero_fraction_check": (
            "passed" if 0.0 in restore_cost_fractions else "not_run"
        ),
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "calibrations_by_fraction": calibrations_by_fraction,
        "comparisons": comparisons,
    }
    _write_json(output_root / "mode_b_refit.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Restore-cost Mode B refit",
            intro_lines=(
                "Triggers, probe margins, and guards are refit at each",
                "restore fraction; the bootstrap scores at the same fraction.",
            ),
        ),
        encoding="utf-8",
    )
    return result


# Contrasts for the within-task baseline. The first pair keeps the ungated
# B1 (does task-local history alone beat the deadline, and does the cross-task
# gated machinery beat it?); the second pair repeats both against the *gated*
# B1 — the fair control that gets the same cross-fitted margin-guard
# protection as the cross-task method, so the ungated collapse under restore
# cost is not a straw man.
WITHIN_TASK_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "within_task_vs_deadline",
        "within_task_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "gated_vs_within_task",
        "offline_gated_robust_trigger_ms",
        "within_task_trigger_ms",
        False,
    ),
    (
        "gated_within_task_vs_deadline",
        "gated_within_task_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "gated_vs_gated_within_task",
        "offline_gated_robust_trigger_ms",
        "gated_within_task_trigger_ms",
        False,
    ),
)


def run_within_task_baseline(
    confirmation_root: Path,
    *,
    mode_b_root: Path,
    output_root: Path,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Score the within-task baseline against the refit gated policy.

    For each fraction, within-task triggers are computed at that fraction on
    the frozen fold eval rows and merged into the Mode B refit decisions
    (``mode_b_root/rho_<fraction>/f*_decisions.jsonl``), so both policies in
    every contrast are fitted and scored at the same restore cost.
    """

    _validate_fractions(restore_cost_fractions)
    confirmation_root = confirmation_root.resolve()
    mode_b_root = mode_b_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")
    manifest = _read_frozen_manifest(confirmation_root)
    fold_count = manifest["fold_count"]
    costs = [float(cost) for cost in manifest["costs_ms"]]

    output_root.mkdir(parents=True)
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in (
            WITHIN_TASK_COMPARISONS
        )
    }
    history_coverage: dict[str, dict[str, int]] = {}
    within_task_guards: dict[str, list[dict[str, Any]]] = {}
    decision_count = 0
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        decisions: list[dict[str, Any]] = []
        fold_guards: list[dict[str, Any]] = []
        for fold in range(1, fold_count + 1):
            fold_name = f"f{fold}"
            eval_rows = list(
                read_tool_latency_jsonl(
                    confirmation_root / "data" / f"{fold_name}_eval.jsonl"
                )
            )
            profile_rows = list(
                read_tool_latency_jsonl(
                    confirmation_root / "data" / f"{fold_name}_profile.jsonl"
                )
            )
            # Frozen fold construction guarantees this, but the guard is only
            # causally clean if profile (fit) and eval (apply) tasks are
            # disjoint, so verify rather than trust.
            overlap = {str(row["task_id"]) for row in eval_rows} & {
                str(row["task_id"]) for row in profile_rows
            }
            if overlap:
                raise AssertionError(
                    f"fold {fold_name} profile and eval tasks overlap: "
                    f"{sorted(overlap)}"
                )
            within_task_kwargs = dict(
                kv_costs_ms=costs,
                guard_ms=manifest["guard_ms"],
                command_field=manifest["command_field"],
                max_prefix_depth=manifest["max_prefix_depth"],
                skip_leading_cd=manifest["skip_leading_cd"],
                restore_cost_fraction=fraction,
            )
            baseline_rows = within_task_trigger_rows(eval_rows, **within_task_kwargs)
            # Fit the margin guard on the fold's profile tasks only, scoring
            # each profile call's within-task margin against its realized
            # early-fire utility (same select_probe_guard machinery, and the
            # same restore fraction, as the cross-task probe).
            profile_rows_scored = within_task_trigger_rows(
                profile_rows, **within_task_kwargs
            )
            guard_calibration = select_probe_guard(
                profile_rows_scored,
                score_field="within_task_margin_normalized",
                candidate_field="within_task_trigger_ms",
                restore_cost_fraction=fraction,
            )
            fold_guards.append({"fold": fold_name, **guard_calibration})
            decisions.extend(
                _merge_within_task_rows(
                    mode_b_root / f"rho_{key}" / f"{fold_name}_decisions.jsonl",
                    baseline_rows,
                    fold_name=fold_name,
                    guard_normalized=guard_calibration["selected_guard_normalized"],
                )
            )
        _write_jsonl(output_root / f"rho_{key}_decisions.jsonl", decisions)
        if decision_count and len(decisions) != decision_count:
            raise AssertionError(
                "restore fractions produced differing decision counts: "
                f"{len(decisions)} != {decision_count}"
            )
        decision_count = len(decisions)
        within_task_guards[key] = fold_guards
        history_coverage[key] = {
            "with_history": sum(
                row["within_task_source"] != "none" for row in decisions
            ),
            "without_history": sum(
                row["within_task_source"] == "none" for row in decisions
            ),
            "early_within_task_triggers": sum(
                row["within_task_trigger_ms"] < row["threshold_ms"]
                for row in decisions
            ),
            "gated_early_within_task_triggers": sum(
                row["gated_within_task_trigger_ms"] < row["threshold_ms"]
                for row in decisions
            ),
        }
        for name, treatment_field, baseline_field, enforce_gated in (
            WITHIN_TASK_COMPARISONS
        ):
            comparisons[name]["by_restore_cost_fraction"][key] = (
                paired_task_cluster_bootstrap(
                    decisions,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_field,
                    treatment_trigger_field=treatment_field,
                    restore_cost_fraction=fraction,
                    enforce_gated_treatment=enforce_gated,
                )
            )

    result = {
        "schema_version": 1,
        "mode": "within_task_baseline",
        "confirmation_root": str(confirmation_root),
        "mode_b_root": str(mode_b_root),
        "fold_count": fold_count,
        "decision_row_count": decision_count,
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "history_coverage": history_coverage,
        "within_task_guards": within_task_guards,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }
    _write_json(output_root / "within_task_baseline.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Within-task history baseline (B1)",
            intro_lines=(
                "The within-task policy uses only the current task's strictly",
                "earlier calls (deepest prefix context, then tool level) with",
                "the same hazard-recheck estimator. The gated variant adds a",
                "margin guard fitted on each fold's profile tasks and applied",
                "to its disjoint eval tasks, matching the cross-task method's",
                "protection; every policy is fitted and scored at each",
                "restore fraction.",
            ),
        ),
        encoding="utf-8",
    )
    return result


def _merge_within_task_rows(
    decisions_path: Path,
    baseline_rows: list[dict[str, Any]],
    *,
    fold_name: str,
    guard_normalized: float | None,
) -> list[dict[str, Any]]:
    """Attach within-task triggers to the fold's refit decision rows.

    ``guard_normalized`` is the fold's profile-fitted margin guard. Each eval
    trigger is gated exactly as ``evaluate_offline_probe_clock`` gates the
    cross-task candidate: keep the early trigger only when a guard exists, the
    trigger fires before the deadline, and the call's within-task margin
    strictly clears the guard; otherwise fall back to the deadline.
    """

    baseline_by_key = {
        (str(row["sample_id"]), float(row["kv_cost_ms"])): row
        for row in baseline_rows
    }
    merged: list[dict[str, Any]] = []
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        decision = json.loads(line)
        key = (str(decision["sample_id"]), float(decision["kv_cost_ms"]))
        baseline = baseline_by_key.pop(key, None)
        if baseline is None:
            raise ValueError(f"no within-task row for decision {key}")
        for field in ("latency_ms", "threshold_ms"):
            if not math.isclose(
                float(decision[field]),
                float(baseline[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{field} mismatch for decision {key}")
        trigger_ms = float(baseline["within_task_trigger_ms"])
        threshold_ms = float(baseline["threshold_ms"])
        margin_normalized = float(baseline["within_task_margin_normalized"])
        gated_trigger_ms = (
            trigger_ms
            if (
                guard_normalized is not None
                and trigger_ms < threshold_ms
                and margin_normalized > guard_normalized
            )
            else threshold_ms
        )
        merged.append(
            {
                **decision,
                "outer_fold": fold_name,
                "within_task_trigger_ms": baseline["within_task_trigger_ms"],
                "within_task_margin_normalized": margin_normalized,
                "gated_within_task_trigger_ms": gated_trigger_ms,
                "gated_within_task_guard_normalized": guard_normalized,
                "within_task_source": baseline["within_task_source"],
                "within_task_history_count": (
                    baseline["within_task_history_count"]
                ),
            }
        )
    if baseline_by_key:
        raise ValueError(
            f"{len(baseline_by_key)} within-task rows unmatched in {fold_name}"
        )
    return merged


def run_hazard_model_confirmation(
    confirmation_root: Path,
    *,
    mode_b_root: Path,
    gated_b1_root: Path,
    output_root: Path,
    restore_cost_fractions: list[float],
    num_intervals: int,
    replicates: int,
    confidence_level: float,
    seed: int,
    l2_penalty: float | None = None,
    feature_set: str = "full",
    model_family: str = "logistic",
    ensemble_members: int = 0,
) -> dict[str, Any]:
    """Score the learned hazard clock against the refit gated policies.

    ``feature_set`` names the ablation arm: ``full`` (all blocks),
    ``with_within_task`` (no task aggregates), or ``cross_task_only``
    (tool/prefix blocks only). Command parsing config always comes from
    the frozen manifest.

    ``model_family`` selects the estimator: ``logistic`` (the pooled penalized
    discrete-time logistic; ``l2_penalty=None`` selects the penalty per fold)
    or ``gbm`` (a HistGradientBoosting hazard over the same person-period rows;
    ``l2_penalty`` must stay ``None`` and the fold's fits use ``seed``).

    ``ensemble_members`` (0 = off, else ``>= 2``) turns on the bagged arm: each
    fold additionally fits ``M`` leave-one-Mth-out members whose unanimity
    trigger and weakest-member margin import the trie's robustness. When on, the
    four :data:`HAZARD_ENSEMBLE_COMPARISONS` contrasts are added and the
    ensemble triggers are merged into the decisions alongside the single model.

    The structural twin of :func:`run_within_task_baseline`. Folds are the outer
    loop: each fold fits the hazard model exactly once (the fit is
    fraction-independent) and scores every restore fraction from the cached
    masses. Each fraction's per-``(sample, kv_cost)`` triggers are merged into
    the Mode B refit decisions (``mode_b_root/rho_<frac>/f*_decisions.jsonl``,
    source of ``deadline_trigger_ms`` and ``offline_gated_robust_trigger_ms``)
    and the gated-B1 decisions (``gated_b1_root/rho_<frac>_decisions.jsonl``,
    source of ``gated_within_task_trigger_ms``), so every policy in every
    contrast is fit and scored at the same restore cost. The feature spec
    mirrors the frozen trie config from the manifest. ``l2_penalty=None``
    selects the penalty per fold on that fold's profile rows.
    """

    _validate_fractions(restore_cost_fractions)
    if num_intervals < 2:
        raise ValueError("num_intervals must be at least 2")
    confirmation_root = confirmation_root.resolve()
    mode_b_root = mode_b_root.resolve()
    gated_b1_root = gated_b1_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")
    manifest = _read_frozen_manifest(confirmation_root)
    fold_count = manifest["fold_count"]
    costs = [float(cost) for cost in manifest["costs_ms"]]
    if model_family not in ("logistic", "gbm"):
        raise ValueError(
            f"unknown model_family {model_family!r}; expected 'logistic' or 'gbm'"
        )
    if ensemble_members < 0 or ensemble_members == 1:
        raise ValueError(
            f"ensemble_members must be 0 or >= 2, got {ensemble_members}"
        )
    ensemble_on = ensemble_members >= 2
    spec = build_survival_feature_spec(
        feature_set,
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
    )

    active_comparisons = HAZARD_COMPARISONS + (
        HAZARD_ENSEMBLE_COMPARISONS if ensemble_on else ()
    )
    output_root.mkdir(parents=True)
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in active_comparisons
    }
    # One gated-B1 lookup per fraction; each fold pops its samples out, and the
    # residual must be empty once every fold is merged (unmatched both ways).
    gated_lookups: dict[str, dict[tuple[str, float], dict[str, Any]]] = {}
    hazard_guards: dict[str, list[dict[str, Any]]] = {}
    grid_l2_choices: dict[str, list[dict[str, Any]]] = {}
    calibrations_by_fraction: dict[str, list[dict[str, Any]]] = {}
    decisions_by_fraction: dict[str, list[dict[str, Any]]] = {}
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        (output_root / f"rho_{key}").mkdir()
        gated_lookups[key] = _load_gated_within_task(
            gated_b1_root / f"rho_{key}_decisions.jsonl"
        )
        hazard_guards[key] = []
        grid_l2_choices[key] = []
        calibrations_by_fraction[key] = []
        decisions_by_fraction[key] = []

    # Folds outermost so the model is fit exactly once per fold (inner_folds
    # inner fits + 1 outer fit); every fraction reuses that fold's cached masses.
    for fold in range(1, fold_count + 1):
        fold_name = f"f{fold}"
        eval_rows = list(
            read_tool_latency_jsonl(
                confirmation_root / "data" / f"{fold_name}_eval.jsonl"
            )
        )
        profile_rows = list(
            read_tool_latency_jsonl(
                confirmation_root / "data" / f"{fold_name}_profile.jsonl"
            )
        )
        overlap = {str(row["task_id"]) for row in eval_rows} & {
            str(row["task_id"]) for row in profile_rows
        }
        if overlap:
            raise AssertionError(
                f"fold {fold_name} profile and eval tasks overlap: "
                f"{sorted(overlap)}"
            )
        fold_result = evaluate_hazard_model_clock(
            eval_rows,
            profile_rows=profile_rows,
            kv_costs_ms=costs,
            guard_ms=manifest["guard_ms"],
            inner_folds=manifest["inner_folds"],
            spec=spec,
            num_intervals=num_intervals,
            restore_cost_fractions=restore_cost_fractions,
            l2_penalty=l2_penalty,
            model_family=model_family,
            seed=seed,
            ensemble_members=ensemble_members,
        )
        shared = {
            field: value
            for field, value in fold_result.items()
            if field != "by_restore_cost_fraction"
        }
        for fraction in restore_cost_fractions:
            key = fraction_key(fraction)
            per_fraction = fold_result["by_restore_cost_fraction"][key]
            hazard_rows = per_fraction["decisions"]
            summary = {
                **shared,
                "restore_cost_fraction": fraction,
                "calibration_guard": per_fraction["calibration_guard"],
            }
            if "ensemble_calibration_guard" in per_fraction:
                summary["ensemble_calibration_guard"] = per_fraction[
                    "ensemble_calibration_guard"
                ]
            fraction_root = output_root / f"rho_{key}"
            _write_json(fraction_root / f"{fold_name}_summary.json", summary)
            _write_jsonl(
                fraction_root / f"{fold_name}_hazard_decisions.jsonl", hazard_rows
            )
            hazard_guards[key].append(
                {"fold": fold_name, **per_fraction["calibration_guard"]}
            )
            grid_l2_choices[key].append(
                {
                    "fold": fold_name,
                    "l2_penalty": shared["l2_penalty"],
                    "l2_selected_on_full_profile": shared[
                        "l2_selected_on_full_profile"
                    ],
                    "num_intervals": shared["num_intervals"],
                    "grid_edges_ms": shared["outer_grid_edges_ms"],
                }
            )
            calibrations_by_fraction[key].append(
                {"fold": fold_name, "hazard_calibration": shared["hazard_calibration"]}
            )
            decisions_by_fraction[key].extend(
                _merge_hazard_rows(
                    mode_b_root / f"rho_{key}" / f"{fold_name}_decisions.jsonl",
                    hazard_rows,
                    gated_within_task_by_key=gated_lookups[key],
                    fold_name=fold_name,
                )
            )

    decision_count = 0
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        if gated_lookups[key]:
            raise ValueError(
                f"{len(gated_lookups[key])} gated-B1 rows unmatched at "
                f"fraction {key}"
            )
        decisions = decisions_by_fraction[key]
        _write_jsonl(output_root / f"rho_{key}_decisions.jsonl", decisions)
        if decision_count and len(decisions) != decision_count:
            raise AssertionError(
                "restore fractions produced differing decision counts: "
                f"{len(decisions)} != {decision_count}"
            )
        decision_count = len(decisions)
        for name, treatment_field, baseline_field, enforce_gated in active_comparisons:
            comparisons[name]["by_restore_cost_fraction"][key] = (
                paired_task_cluster_bootstrap(
                    decisions,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_field,
                    treatment_trigger_field=treatment_field,
                    restore_cost_fraction=fraction,
                    enforce_gated_treatment=enforce_gated,
                )
            )

    result = {
        "schema_version": 1,
        "mode": "hazard_model_confirmation",
        "feature_set": feature_set,
        "model_family": model_family,
        "ensemble_members": ensemble_members,
        "confirmation_root": str(confirmation_root),
        "mode_b_root": str(mode_b_root),
        "gated_b1_root": str(gated_b1_root),
        "fold_count": fold_count,
        "decision_row_count": decision_count,
        "costs_ms": costs,
        "num_intervals": num_intervals,
        "restore_cost_fractions": restore_cost_fractions,
        "hazard_guards": hazard_guards,
        "grid_l2_choices": grid_l2_choices,
        "calibrations_by_fraction": calibrations_by_fraction,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }
    _write_json(output_root / "hazard_model_confirmation.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Hazard-model confirmation",
            intro_lines=(
                "A learned discrete-time hazard model predicts each call's",
                "latency distribution; its trigger and margin guard replace the",
                "empirical trie behind the same utility-clock seam. Every policy",
                "is fit and scored at each restore fraction, and the hazard gate",
                "reuses the cross-fitted margin guard the trie method has.",
            ),
        ),
        encoding="utf-8",
    )
    return result


def _load_gated_within_task(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    """Index one fraction's gated-B1 decisions by ``(sample_id, kv_cost)``."""

    lookup: dict[tuple[str, float], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["sample_id"]), float(row["kv_cost_ms"]))
        if key in lookup:
            raise ValueError(f"duplicate gated-B1 decision key: {key}")
        lookup[key] = row
    return lookup


def _merge_hazard_rows(
    decisions_path: Path,
    hazard_rows: list[dict[str, Any]],
    *,
    gated_within_task_by_key: dict[tuple[str, float], dict[str, Any]],
    fold_name: str,
) -> list[dict[str, Any]]:
    """Attach hazard triggers and the gated-B1 trigger to Mode B decisions.

    Joins each Mode B refit decision to its hazard row (from the current fold)
    and its gated-B1 row (from the shared per-fraction lookup) by
    ``(sample_id, kv_cost)``, verifying the latency and threshold agree. The
    gate is already applied inside ``evaluate_hazard_model_clock``, so the
    merge only carries ``offline_gated_hazard_trigger_ms`` through. Unmatched
    rows in either direction raise, mirroring ``_merge_within_task_rows``.
    """

    hazard_by_key = {
        (str(row["sample_id"]), float(row["kv_cost_ms"])): row for row in hazard_rows
    }
    merged: list[dict[str, Any]] = []
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        decision = json.loads(line)
        key = (str(decision["sample_id"]), float(decision["kv_cost_ms"]))
        hazard = hazard_by_key.pop(key, None)
        if hazard is None:
            raise ValueError(f"no hazard row for decision {key}")
        gated = gated_within_task_by_key.pop(key, None)
        if gated is None:
            raise ValueError(f"no gated-B1 row for decision {key}")
        for field in ("latency_ms", "threshold_ms"):
            for other, label in ((hazard, "hazard"), (gated, "gated-B1")):
                if not math.isclose(
                    float(decision[field]),
                    float(other[field]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(f"{field} mismatch for decision {key} vs {label}")
        merged_row = {
            **decision,
            "outer_fold": fold_name,
            "hazard_trigger_ms": hazard["hazard_trigger_ms"],
            "hazard_margin_normalized": hazard["hazard_margin_normalized"],
            "offline_gated_hazard_trigger_ms": hazard[
                "offline_gated_hazard_trigger_ms"
            ],
            "offline_gated_hazard_guard_normalized": hazard[
                "offline_gated_hazard_guard_normalized"
            ],
            "gated_within_task_trigger_ms": gated["gated_within_task_trigger_ms"],
        }
        # The bagged ensemble arm is optional; carry its triggers only when the
        # hazard evaluator produced them (ensemble_members >= 2).
        if "ensemble_hazard_trigger_ms" in hazard:
            for field in (
                "ensemble_hazard_trigger_ms",
                "ensemble_hazard_margin_normalized",
                "offline_gated_ensemble_hazard_trigger_ms",
                "offline_gated_ensemble_hazard_guard_normalized",
            ):
                merged_row[field] = hazard[field]
        merged.append(merged_row)
    if hazard_by_key:
        raise ValueError(
            f"{len(hazard_by_key)} hazard rows unmatched in {fold_name}"
        )
    return merged


# The union fires when EITHER gate opens early, at whichever trigger comes
# first. Mechanism analysis (tool-time-mechanism-analysis-20260715) showed the
# two gates fire on largely disjoint calls, so the OR combines complementary
# coverage; it is parameter-free (both triggers are computed offline).
GATE_UNION_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    ("union_vs_deadline", "gate_union_trigger_ms", "deadline_trigger_ms", False),
    (
        "union_vs_gated_hazard",
        "gate_union_trigger_ms",
        "offline_gated_hazard_trigger_ms",
        False,
    ),
    (
        "union_vs_gated_robust",
        "gate_union_trigger_ms",
        "offline_gated_robust_trigger_ms",
        False,
    ),
)


def run_gate_union_analysis(
    hazard_root: Path,
    *,
    output_root: Path,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the OR-union of the trie and hazard gates.

    Re-scores the merged decisions of a hazard confirmation run
    (``hazard_root/rho_<fraction>_decisions.jsonl``) with the derived field
    ``gate_union_trigger_ms = min(offline_gated_hazard_trigger_ms,
    offline_gated_robust_trigger_ms)``. Triggers stay exactly as fitted at
    each fraction; only the combination rule is new, so this is a Mode-A
    style derived re-scoring, not a refit.

    Sensitivity only: the OR rule was itself selected after observing gate
    disjointness on this corpus, so these intervals certify the combiner on
    the corpus that motivated it. A fresh-corpus certification is required
    before any headline claim.
    """

    _validate_fractions(restore_cost_fractions)
    hazard_root = hazard_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")

    output_root.mkdir(parents=True)
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in (
            GATE_UNION_COMPARISONS
        )
    }
    decision_count = 0
    costs: list[float] = []
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        decisions_path = hazard_root / f"rho_{key}_decisions.jsonl"
        decisions: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            decisions_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            source = f"{decisions_path}:{line_number}"
            for field in (
                "offline_gated_hazard_trigger_ms",
                "offline_gated_robust_trigger_ms",
            ):
                if field not in row:
                    raise ValueError(f"{source} lacks {field}")
            row["gate_union_trigger_ms"] = min(
                float(row["offline_gated_hazard_trigger_ms"]),
                float(row["offline_gated_robust_trigger_ms"]),
            )
            decisions.append(row)
        if not decisions:
            raise ValueError(f"no decisions in {decisions_path}")
        costs = sorted({float(row["kv_cost_ms"]) for row in decisions})
        if decision_count and len(decisions) != decision_count:
            raise AssertionError(
                "restore fractions carry differing decision counts: "
                f"{len(decisions)} != {decision_count}"
            )
        decision_count = len(decisions)
        for name, treatment_field, baseline_field, enforce_gated in (
            GATE_UNION_COMPARISONS
        ):
            comparisons[name]["by_restore_cost_fraction"][key] = (
                paired_task_cluster_bootstrap(
                    decisions,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_field,
                    treatment_trigger_field=treatment_field,
                    restore_cost_fraction=fraction,
                    enforce_gated_treatment=enforce_gated,
                )
            )

    result = {
        "schema_version": 1,
        "mode": "gate_union_analysis",
        "confirmation_root": str(hazard_root),
        "decision_row_count": decision_count,
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }
    _write_json(output_root / "gate_union_analysis.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Gate-union combiner (OR of trie and hazard gates)",
            intro_lines=(
                "gate_union_trigger_ms = min of the two gated triggers as",
                "fitted at each fraction; the combination rule is the only",
                "new element (parameter-free derived re-scoring). Sensitivity",
                "only: the OR rule was selected after observing gate",
                "disjointness on this same frozen corpus, so these intervals",
                "carry selection optimism; fresh-corpus certification is",
                "required before any headline claim.",
            ),
        ),
        encoding="utf-8",
    )
    return result


# The candidate gates the certified union may OR: the empirical trie gate and
# the learned hazard gate, each named by its gated (guard-protected) trigger.
CERTIFIED_UNION_GATES: tuple[tuple[str, str], ...] = (
    ("trie", "offline_gated_robust_trigger_ms"),
    ("hazard", "offline_gated_hazard_trigger_ms"),
)

_CERTIFIED_UNION_REQUIRED_FIELDS = (
    "outer_fold",
    "sample_id",
    "task_id",
    "latency_ms",
    "kv_cost_ms",
    "threshold_ms",
    "deadline_trigger_ms",
    "offline_gated_hazard_trigger_ms",
    "offline_gated_robust_trigger_ms",
)

_CERTIFIED_INCLUSION_CRITERIA = ("loo_point", "loo_lcb")

# certified_union_trigger_ms fires when EITHER *certified* gate opens early, at
# whichever fitted trigger comes first — the naive union restricted to the gates
# that beat the deadline on the fitting partition. The naive union
# (gate_union_trigger_ms = min of both gates unconditionally) is the fourth
# baseline so the certified rule is measured against the combiner it corrects.
CERTIFIED_UNION_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "certified_union_vs_deadline",
        "certified_union_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "certified_union_vs_gated_hazard",
        "certified_union_trigger_ms",
        "offline_gated_hazard_trigger_ms",
        False,
    ),
    (
        "certified_union_vs_gated_robust",
        "certified_union_trigger_ms",
        "offline_gated_robust_trigger_ms",
        False,
    ),
    (
        "certified_union_vs_naive_union",
        "certified_union_trigger_ms",
        "gate_union_trigger_ms",
        False,
    ),
)


def run_certified_union_analysis(
    hazard_root: Path,
    *,
    output_root: Path,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
    inclusion_criterion: str = "loo_point",
) -> dict[str, Any]:
    """Bootstrap the certified OR-union of the trie and hazard gates.

    Re-scores the merged decisions of a hazard confirmation run
    (``hazard_root/rho_<fraction>_decisions.jsonl``, the same source as
    :func:`run_gate_union_analysis`). Where the naive union ORs both gates
    unconditionally, the certified union ORs only the gates that certifiably
    beat the deadline on the fitting partition — decided WITHOUT looking at the
    rows being scored.

    Cross-fitted inclusion (the leakage-free core): the decisions span every
    outer fold (``outer_fold`` field). For each outer fold ``f`` and candidate
    gate ``g``, inclusion is decided using ONLY the rows with
    ``outer_fold != f`` (leave-fold-out): the gate is included for fold ``f``
    iff its total utility advantage over the deadline on those other folds'
    rows passes the criterion. Because fold ``f``'s own rows never enter its
    inclusion decision, the eval partition is never used to certify the rule it
    is then scored under — no leakage into the scored partition.

    ``inclusion_criterion`` selects that test: ``loo_point`` (default) includes
    a gate iff its leave-fold-out total delta is strictly positive;
    ``loo_lcb`` (conservative) includes it iff a task-clustered bootstrap lower
    bound of that delta is strictly positive.

    The certified-union trigger for each row of fold ``f`` is the min over the
    gates included for ``f`` (always ``<= threshold``), or the deadline if no
    gate is included. Triggers stay exactly as fitted at each fraction; only the
    combination rule is new (a Mode-A style derived re-scoring, not a refit).

    Sensitivity only: the certified-union rule family was itself selected after
    observing gate behavior on these corpora, so these intervals certify the
    combiner on the corpora that motivated it. A fresh-corpus certification is
    required before any headline claim.
    """

    _validate_fractions(restore_cost_fractions)
    if inclusion_criterion not in _CERTIFIED_INCLUSION_CRITERIA:
        raise ValueError(
            f"unknown inclusion_criterion {inclusion_criterion!r}; expected one of "
            f"{_CERTIFIED_INCLUSION_CRITERIA}"
        )
    hazard_root = hazard_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")

    output_root.mkdir(parents=True)
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in (
            CERTIFIED_UNION_COMPARISONS
        )
    }
    inclusion_by_fraction: dict[str, dict[str, dict[str, Any]]] = {}
    decision_count = 0
    costs: list[float] = []
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        decisions = _load_certified_union_decisions(
            hazard_root / f"rho_{key}_decisions.jsonl"
        )
        fraction_costs = sorted({float(row["kv_cost_ms"]) for row in decisions})
        if decision_count and (
            len(decisions) != decision_count or fraction_costs != costs
        ):
            raise AssertionError(
                "restore fractions carry differing decision counts or cost "
                f"sets: {len(decisions)}/{fraction_costs} != "
                f"{decision_count}/{costs}"
            )
        costs = fraction_costs
        decision_count = len(decisions)

        fold_names = sorted({str(row["outer_fold"]) for row in decisions})
        if len(fold_names) < 2:
            raise ValueError(
                "certified union needs >= 2 outer folds for leave-fold-out "
                f"inclusion, found {fold_names}"
            )
        inclusion = {
            fold: _fold_gate_inclusion(
                [row for row in decisions if str(row["outer_fold"]) != fold],
                restore_cost_fraction=fraction,
                inclusion_criterion=inclusion_criterion,
                replicates=replicates,
                confidence_level=confidence_level,
                seed=seed,
            )
            for fold in fold_names
        }
        inclusion_by_fraction[key] = inclusion

        for row in decisions:
            fold = str(row["outer_fold"])
            included_triggers = [
                float(row[gate_field])
                for gate_name, gate_field in CERTIFIED_UNION_GATES
                if inclusion[fold][gate_name]["included"]
            ]
            row["certified_union_trigger_ms"] = (
                min(included_triggers)
                if included_triggers
                else float(row["threshold_ms"])
            )
        _write_jsonl(output_root / f"rho_{key}_decisions.jsonl", decisions)

        for name, treatment_field, baseline_field, enforce_gated in (
            CERTIFIED_UNION_COMPARISONS
        ):
            comparisons[name]["by_restore_cost_fraction"][key] = (
                paired_task_cluster_bootstrap(
                    decisions,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_field,
                    treatment_trigger_field=treatment_field,
                    restore_cost_fraction=fraction,
                    enforce_gated_treatment=enforce_gated,
                )
            )

    result = {
        "schema_version": 1,
        "mode": "certified_union_analysis",
        "confirmation_root": str(hazard_root),
        "inclusion_criterion": inclusion_criterion,
        "decision_row_count": decision_count,
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "gate_inclusion_by_restore_cost_fraction": inclusion_by_fraction,
        "comparisons": comparisons,
    }
    _write_json(output_root / "certified_union_analysis.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Certified-union combiner (OR of certified trie/hazard gates)",
            intro_lines=(
                "certified_union_trigger_ms ORs only the gates that certifiably",
                "beat the deadline on the FITTING partition: for each outer fold,",
                "a gate is included using ONLY the OTHER folds' rows (cross-fitted",
                "leave-fold-out), so a fold's own rows never certify the rule they",
                "are scored under. The trigger is the min over included gates, or",
                f"the deadline if none is included (criterion: {inclusion_criterion}).",
                "Sensitivity only: the certified-union rule family was itself",
                "selected after observing gate behavior on these corpora, so these",
                "intervals carry selection optimism; fresh-corpus certification is",
                "required before any headline claim.",
            ),
        ),
        encoding="utf-8",
    )
    return result


def _load_certified_union_decisions(path: Path) -> list[dict[str, Any]]:
    """Load merged decisions and attach the naive-union trigger inline.

    Requires the leave-fold-out partition (``outer_fold``) and both gate
    triggers on every row; ``gate_union_trigger_ms`` is the unconditional
    min-of-both baseline the certified rule is compared against.
    """

    decisions: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        source = f"{path}:{line_number}"
        for field in _CERTIFIED_UNION_REQUIRED_FIELDS:
            if field not in row:
                raise ValueError(f"{source} lacks {field}")
        row["gate_union_trigger_ms"] = min(
            float(row["offline_gated_hazard_trigger_ms"]),
            float(row["offline_gated_robust_trigger_ms"]),
        )
        decisions.append(row)
    if not decisions:
        raise ValueError(f"no decisions in {path}")
    return decisions


def _fold_gate_inclusion(
    leave_fold_out_rows: list[dict[str, Any]],
    *,
    restore_cost_fraction: float,
    inclusion_criterion: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Decide each gate's inclusion for one fold from the other folds' rows.

    For each candidate gate, sums the per-task utility advantage of the gate's
    trigger over the deadline across the leave-fold-out rows. ``loo_point``
    includes the gate iff that total is strictly positive; ``loo_lcb`` requires
    a task-clustered bootstrap lower bound of the total to be strictly positive.

    The advantage is aggregated across all kv-cost columns into ONE inclusion
    decision per (fold, gate), re-decided per ``restore_cost_fraction``. A gate
    net-positive on the cost-aggregate but locally harmful at a single cost is
    still included at that cost; this matches the total-utility framing and
    keeps inclusion a single per-fold rule rather than a per-cost one.
    """

    records: dict[str, dict[str, Any]] = {}
    for gate_name, gate_field in CERTIFIED_UNION_GATES:
        task_deltas: dict[str, float] = defaultdict(float)
        for row in leave_fold_out_rows:
            cost = float(row["kv_cost_ms"])
            threshold = float(row["threshold_ms"])
            latency = float(row["latency_ms"])
            restore_cost_ms = restore_cost_fraction * cost
            gate_utility = trigger_policy_utility_ms(
                latency,
                float(row[gate_field]),
                threshold_ms=threshold,
                kv_cost_ms=cost,
                restore_cost_ms=restore_cost_ms,
            )
            deadline_utility = trigger_policy_utility_ms(
                latency,
                float(row["deadline_trigger_ms"]),
                threshold_ms=threshold,
                kv_cost_ms=cost,
                restore_cost_ms=restore_cost_ms,
            )
            task_deltas[str(row["task_id"])] += gate_utility - deadline_utility
        total_delta = sum(task_deltas.values())
        record: dict[str, Any] = {"leave_fold_out_delta_ms": total_delta}
        if inclusion_criterion == "loo_point":
            record["included"] = total_delta > 0.0
        else:
            lcb = _task_cluster_lower_bound(
                list(task_deltas.values()),
                replicates=replicates,
                confidence_level=confidence_level,
                seed=seed,
            )
            record["leave_fold_out_lcb_ms"] = lcb
            record["included"] = lcb > 0.0
        records[gate_name] = record
    return records


def _task_cluster_lower_bound(
    task_totals: list[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> float:
    """Lower confidence bound of a scalar total by resampling task clusters.

    Reuses the confirmation bootstrap's task-cluster resampler on a single
    (across-cost) contribution column, so the LCB uses the same PCG64 draw and
    percentile convention as :func:`paired_task_cluster_bootstrap`.
    """

    contributions = np.asarray(task_totals, dtype=float).reshape(-1, 1)
    bootstrap_totals = _resample_task_totals(
        contributions, replicates=replicates, seed=seed
    )
    alpha = 1.0 - confidence_level
    return float(np.quantile(bootstrap_totals[:, 0], alpha / 2.0, method="linear"))


# P2 reload-vs-recompute restore choice. All three contrasts keep the gated
# trigger fitted at restore cost zero (Mode A); only the restore cost charged
# on fires-on-short changes. C1 isolates the mechanism increment: the same
# gated trigger scored under min(swap-in, recompute) vs swap-in only, so its
# paired delta is non-negative by construction (min <= swap). C2/C3 place the
# min-restore and swap-only policies against the never-early deadline.
RECOMPUTE_RESTORE_COMPARISONS: tuple[tuple[str, str, str, str, str, bool], ...] = (
    (
        "min_restore_vs_swap_restore",
        "offline_gated_robust_trigger_ms",
        "offline_gated_robust_trigger_ms",
        "p2_min_restore_ms",
        "p2_swap_restore_ms",
        True,
    ),
    (
        "min_restore_gated_vs_deadline",
        "offline_gated_robust_trigger_ms",
        "deadline_trigger_ms",
        "p2_min_restore_ms",
        "p2_min_restore_ms",
        False,
    ),
    (
        "swap_restore_gated_vs_deadline",
        "offline_gated_robust_trigger_ms",
        "deadline_trigger_ms",
        "p2_swap_restore_ms",
        "p2_swap_restore_ms",
        False,
    ),
)

# Default recompute-rate grid (ms/token). Chosen to bracket the swap/recompute
# crossover context (restore_cost_fraction * kv_cost_ms / rate) across the
# swept kv grid and the observed context range, not tuned to any outcome; it
# is a stand-in for a later measured H100 prefill-vs-context curve, exactly as
# rho was swept before it was measured. See tool_latency_recompute.
DEFAULT_RECOMPUTE_RATES_MS_PER_TOKEN: tuple[float, ...] = (0.05, 0.15, 0.5, 1.5)


def rate_key(rate: float) -> str:
    return str(float(rate))


def _trace_paths_from_decisions(decisions: list[dict[str, Any]]) -> list[Path]:
    """Recover the raw trace paths embedded in decision sample ids."""

    paths: set[str] = set()
    for row in decisions:
        sample_id = str(row["sample_id"])
        # sample_id == f"{trace_path}:{agent_id}:{iteration}:{action_id}"
        paths.add(sample_id.rsplit(":", 3)[0])
    resolved: list[Path] = []
    for path_str in sorted(paths):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"trace referenced by decisions is missing: {path}")
        resolved.append(path)
    return resolved


def _attach_restore_fields(
    decisions: list[dict[str, Any]],
    context_by_sample: dict[str, int],
    *,
    restore_cost_fraction: float,
    recompute_rate_ms_per_token: float,
) -> list[dict[str, Any]]:
    """Attach per-row swap, recompute, and min restore costs for one grid cell."""

    attached: list[dict[str, Any]] = []
    for row in decisions:
        sample_id = str(row["sample_id"])
        context_length = context_by_sample.get(sample_id)
        if context_length is None:
            raise ValueError(f"no context length for decision sample {sample_id!r}")
        cost = float(row["kv_cost_ms"])
        swap_restore_ms = restore_cost_fraction * cost
        recompute_ms = recompute_restore_ms(
            context_length, recompute_rate_ms_per_token
        )
        attached.append(
            {
                **row,
                "context_length_tokens": context_length,
                "p2_swap_restore_ms": swap_restore_ms,
                "p2_recompute_restore_ms": recompute_ms,
                "p2_min_restore_ms": effective_min_restore_ms(
                    swap_restore_ms, recompute_ms
                ),
            }
        )
    return attached


def run_recompute_restore_sweep(
    confirmation_root: Path,
    *,
    restore_cost_fractions: list[float],
    recompute_rates_ms_per_token: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Re-score frozen gated triggers under the reload-vs-recompute choice.

    For every ``(restore_cost_fraction, recompute_rate)`` cell the gated
    trigger stays exactly as fitted (Mode A); the restore charged on fires on
    short calls becomes ``min(fraction * kv_cost_ms, rate * context_length)``,
    where ``context_length`` is the call's resident KV token count recovered
    from the raw traces. The paired task-cluster bootstrap then certifies the
    mechanism increment (:data:`RECOMPUTE_RESTORE_COMPARISONS`).

    Sensitivity only: the recompute-rate grid is a stand-in for a measured
    prefill curve and the triggers were fit under swap-only restore, so this
    lower-bounds the value of a recompute-aware policy. Fresh-corpus
    certification at a measured rate is required before any headline claim.
    """

    _validate_fractions(restore_cost_fractions)
    _validate_recompute_rates(recompute_rates_ms_per_token)
    decisions, fold_names = load_fold_decisions(confirmation_root)
    costs = sorted({float(row["kv_cost_ms"]) for row in decisions})
    context_by_sample = context_lengths_from_traces(
        _trace_paths_from_decisions(decisions)
    )
    context_values = sorted(
        context_by_sample[sample_id]
        for sample_id in {str(row["sample_id"]) for row in decisions}
    )
    context_stats = {
        "sample_count": len(context_values),
        "min": context_values[0],
        "median": context_values[len(context_values) // 2],
        "max": context_values[-1],
    }

    by_recompute_rate: dict[str, Any] = {}
    for rate in recompute_rates_ms_per_token:
        comparisons: dict[str, dict[str, Any]] = {
            name: {
                "treatment_trigger_field": treatment_trigger,
                "baseline_trigger_field": baseline_trigger,
                "treatment_restore_cost_ms_field": treatment_restore,
                "baseline_restore_cost_ms_field": baseline_restore,
                "enforce_gated_treatment": enforce_gated,
                "by_restore_cost_fraction": {},
            }
            for (
                name,
                treatment_trigger,
                baseline_trigger,
                treatment_restore,
                baseline_restore,
                enforce_gated,
            ) in RECOMPUTE_RESTORE_COMPARISONS
        }
        for fraction in restore_cost_fractions:
            attached = _attach_restore_fields(
                decisions,
                context_by_sample,
                restore_cost_fraction=fraction,
                recompute_rate_ms_per_token=rate,
            )
            for (
                name,
                treatment_trigger,
                baseline_trigger,
                treatment_restore,
                baseline_restore,
                enforce_gated,
            ) in RECOMPUTE_RESTORE_COMPARISONS:
                comparisons[name]["by_restore_cost_fraction"][
                    fraction_key(fraction)
                ] = paired_task_cluster_bootstrap(
                    attached,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_trigger,
                    treatment_trigger_field=treatment_trigger,
                    restore_cost_fraction=fraction,
                    baseline_restore_cost_ms_field=baseline_restore,
                    treatment_restore_cost_ms_field=treatment_restore,
                    enforce_gated_treatment=enforce_gated,
                )
        by_recompute_rate[rate_key(rate)] = {"comparisons": comparisons}

    return {
        "schema_version": 1,
        "mode": "reload_vs_recompute_restore",
        "confirmation_root": str(confirmation_root.resolve()),
        "fold_count": len(fold_names),
        "fold_names": fold_names,
        "decision_row_count": len(decisions),
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "recompute_rates_ms_per_token": recompute_rates_ms_per_token,
        "context_length_stats": context_stats,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "by_recompute_rate": by_recompute_rate,
    }


def render_recompute_restore_markdown(result: dict[str, Any]) -> str:
    """Summarize the reload-vs-recompute sweep as one table per recompute rate."""

    lines = [
        "# Reload-vs-recompute restore choice (P2)",
        "",
        f"Source: `{result['confirmation_root']}`",
        "",
        "Gated triggers stay fitted at restore cost zero (Mode A); the restore",
        "charged on fires on short calls is min(swap-in, recompute), with",
        "recompute = rate * per-call context length recovered from traces.",
        "Labels use the Bonferroni-corrected simultaneous intervals over all kv",
        "costs within one comparison-fraction cell.",
        "",
        "Context length (tokens): "
        f"min {result['context_length_stats']['min']}, "
        f"median {result['context_length_stats']['median']}, "
        f"max {result['context_length_stats']['max']}.",
        "",
    ]
    for rate in result["recompute_rates_ms_per_token"]:
        rate_block = result["by_recompute_rate"][rate_key(rate)]
        lines.append(f"## recompute rate {rate} ms/token")
        lines.append("")
        for name, comparison in rate_block["comparisons"].items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                "| restore fraction | positive | inconclusive | harmful "
                "| total delta (ms) | worst simultaneous LCB (ms) |"
            )
            lines.append("|---|---|---|---|---|---|")
            for fraction in result["restore_cost_fractions"]:
                payload = comparison["by_restore_cost_fraction"][fraction_key(fraction)]
                points = payload["points"].values()
                labels = [point["simultaneous_label"] for point in points]
                total_delta = sum(point["paired_delta_ms"] for point in points)
                worst_lcb = min(
                    point["simultaneous_interval_ms"]["low"] for point in points
                )
                lines.append(
                    f"| {fraction} | {labels.count('positive')} "
                    f"| {labels.count('inconclusive')} | {labels.count('harmful')} "
                    f"| {total_delta:.1f} | {worst_lcb:.1f} |"
                )
            lines.append("")
    return "\n".join(lines)


def _validate_recompute_rates(recompute_rates_ms_per_token: list[float]) -> None:
    if not recompute_rates_ms_per_token:
        raise ValueError("recompute_rates_ms_per_token must be non-empty")
    if len(set(recompute_rates_ms_per_token)) != len(recompute_rates_ms_per_token):
        raise ValueError("recompute_rates_ms_per_token must be unique")
    for rate in recompute_rates_ms_per_token:
        validate_recompute_rate(rate)


def render_summary_markdown(
    result: dict[str, Any],
    *,
    title: str = "Restore-cost re-scoring sweep",
    intro_lines: tuple[str, ...] = (
        "Triggers are frozen as fitted at restore cost zero; only the",
        "evaluation utility charges fires on short calls.",
    ),
) -> str:
    """Summarize simultaneous labels and total deltas per fraction."""

    source = result.get("confirmation_root") or result.get("trace_root")
    lines = [
        f"# {title}",
        "",
        f"Source: `{source}`",
        "",
        *intro_lines,
        "Labels use the Bonferroni-corrected simultaneous intervals over",
        "all kv costs within one comparison-fraction cell; they are not",
        "corrected across comparisons or fractions, so read each row as",
        "its own what-if.",
        "",
    ]
    for name, comparison in result["comparisons"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            "| restore fraction | positive | inconclusive | harmful "
            "| total delta (ms) | worst simultaneous LCB (ms) |"
        )
        lines.append("|---|---|---|---|---|---|")
        for fraction in result["restore_cost_fractions"]:
            payload = comparison["by_restore_cost_fraction"][fraction_key(fraction)]
            points = payload["points"].values()
            labels = [point["simultaneous_label"] for point in points]
            total_delta = sum(point["paired_delta_ms"] for point in points)
            worst_lcb = min(
                point["simultaneous_interval_ms"]["low"] for point in points
            )
            lines.append(
                f"| {fraction} | {labels.count('positive')} "
                f"| {labels.count('inconclusive')} | {labels.count('harmful')} "
                f"| {total_delta:.1f} | {worst_lcb:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def load_fold_decisions(
    confirmation_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load cv fold decision rows and stamp each with its outer fold."""

    cv_root = confirmation_root / "cv"
    if not cv_root.is_dir():
        raise ValueError(f"confirmation root lacks a cv directory: {cv_root}")
    decisions: list[dict[str, Any]] = []
    fold_names: list[str] = []
    for path in sorted(cv_root.iterdir()):
        match = _FOLD_DECISIONS_PATTERN.match(path.name)
        if match is None:
            continue
        fold = f"f{match.group(1)}"
        fold_names.append(fold)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: decision must be an object")
            existing_fold = row.get("outer_fold")
            if existing_fold is not None and existing_fold != fold:
                raise ValueError(
                    f"{path}:{line_number}: outer_fold {existing_fold!r} "
                    f"conflicts with fold file {fold}"
                )
            decisions.append({**row, "outer_fold": fold})
    if not decisions:
        raise ValueError(f"no f*_decisions.jsonl rows found under {cv_root}")
    return decisions, fold_names


def fraction_key(fraction: float) -> str:
    return str(float(fraction))


def _validate_fractions(restore_cost_fractions: list[float]) -> None:
    if not restore_cost_fractions:
        raise ValueError("restore_cost_fractions must be non-empty")
    if len(set(restore_cost_fractions)) != len(restore_cost_fractions):
        raise ValueError("restore_cost_fractions must be unique")


_FROZEN_MANIFEST_FIELDS = (
    "fold_count",
    "inner_folds",
    "costs_ms",
    "guard_ms",
    "min_tool_history",
    "min_profile_tasks",
    "command_field",
    "max_prefix_depth",
    "skip_leading_cd",
)


def _read_frozen_manifest(confirmation_root: Path) -> dict[str, Any]:
    return load_config_manifest(confirmation_root / "provenance" / "manifest.json")


def load_config_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read a frozen config manifest json and require every frozen field.

    The same field set the confirmation protocol freezes
    (:data:`_FROZEN_MANIFEST_FIELDS`); callers that only vary the estimator
    (num_intervals, model_family, …) reuse this to source the shared fold and
    command-parsing config.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    missing = [field for field in _FROZEN_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"manifest lacks frozen config fields: {missing}")
    return manifest


def _index_triggers(
    decisions: list[dict[str, Any]],
) -> dict[tuple[str, float], dict[str, float]]:
    indexed: dict[tuple[str, float], dict[str, float]] = {}
    for row in decisions:
        key = (str(row["sample_id"]), float(row["kv_cost_ms"]))
        if key in indexed:
            raise ValueError(f"duplicate frozen decision key: {key}")
        indexed[key] = {
            field: float(row[field]) for field in _FROZEN_TRIGGER_FIELDS
        }
    return indexed


def _assert_matches_frozen_triggers(
    decisions: list[dict[str, Any]],
    frozen_triggers: dict[tuple[str, float], dict[str, float]],
) -> None:
    """The fraction-zero refit must reproduce the frozen triggers exactly."""
    if len(decisions) != len(frozen_triggers):
        raise AssertionError(
            "fraction-zero refit decision count differs from frozen run: "
            f"{len(decisions)} != {len(frozen_triggers)}"
        )
    for row in decisions:
        key = (str(row["sample_id"]), float(row["kv_cost_ms"]))
        frozen = frozen_triggers.get(key)
        if frozen is None:
            raise AssertionError(f"refit decision missing from frozen run: {key}")
        for field, frozen_value in frozen.items():
            if not math.isclose(
                float(row[field]),
                frozen_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise AssertionError(
                    f"fraction-zero refit diverges from frozen run at {key}: "
                    f"{field} {row[field]} != {frozen_value}"
                )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


__all__ = [
    "CERTIFIED_UNION_COMPARISONS",
    "CERTIFIED_UNION_GATES",
    "COMPARISONS",
    "GATE_UNION_COMPARISONS",
    "WITHIN_TASK_COMPARISONS",
    "DEFAULT_RECOMPUTE_RATES_MS_PER_TOKEN",
    "RECOMPUTE_RESTORE_COMPARISONS",
    "analyze_restore_cost_sweep",
    "fraction_key",
    "load_fold_decisions",
    "rate_key",
    "render_recompute_restore_markdown",
    "render_summary_markdown",
    "run_certified_union_analysis",
    "run_recompute_restore_sweep",
    "run_gate_union_analysis",
    "run_hazard_model_confirmation",
    "run_mode_b_refit",
    "run_within_task_baseline",
]
