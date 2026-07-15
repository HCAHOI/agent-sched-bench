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

import json
import math
from pathlib import Path
import re
from typing import Any

from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
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
    "COMPARISONS",
    "GATE_UNION_COMPARISONS",
    "WITHIN_TASK_COMPARISONS",
    "analyze_restore_cost_sweep",
    "fraction_key",
    "load_fold_decisions",
    "render_summary_markdown",
    "run_gate_union_analysis",
    "run_hazard_model_confirmation",
    "run_mode_b_refit",
    "run_within_task_baseline",
]
