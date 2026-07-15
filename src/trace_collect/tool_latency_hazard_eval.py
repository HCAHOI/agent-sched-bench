"""Fold evaluation and calibration for the discrete-time hazard clock.

This is the hazard-model analog of :func:`evaluate_offline_probe_clock`. It
fits the learned survival predictor on profile tasks, learns a margin guard by
an inner task-out-of-fold probe, and scores the deployed
``offline_gated_hazard_trigger_ms`` on disjoint eval tasks. The trigger/gate
seam is identical to the empirical clock (``survival_clock_region_stats`` plays
the role of ``mean_clock_region_stats`` and ``select_probe_guard`` is reused
verbatim), so every downstream contrast stays predictor-agnostic.

**Restore-cost amortization.** The fitted model — grid quantiles, feature
encoder, coefficients, and L2 penalty — is *independent* of the restore-cost
fraction: rho enters only the utility functional (``survival_trigger_ms`` /
``survival_clock_region_stats``) and the guard selection. So the model fits and
the predicted interval masses are computed exactly once per fold
(``inner_folds`` inner fits + 1 outer fit), and every requested fraction reuses
the cached masses to derive its probe margins, guard, triggers, and gating. The
per-fraction results are therefore identical to refitting the model separately
at each fraction (the model never sees rho); see
``test_evaluate_hazard_model_clock_fits_are_fraction_independent``.

Two disclosed simplifications, both eval-blind:

* **Shared L2.** When ``l2_penalty is None`` the penalty is selected once, by
  the task-grouped CV inside :func:`fit_hazard_model` on the *full* profile
  rows, and that single scalar is reused for every inner-fold fit. The scalar
  never sees eval tasks, so the guard stays causally clean with respect to
  eval; it is, however, chosen on profile data that includes each inner-held-out
  fold, so the guard is not fully nested. Full nested selection would multiply
  the number of fits by roughly the L2-grid size times the CV-fold count (~30x)
  for one regularization scalar, which is not worth it for a hyperparameter
  this insensitive.
* **Calibration reporting.** The ``hazard_calibration`` block compares predicted
  survival against realized eval latencies. This touches eval labels for
  *reporting only* — it never feeds any fit or gate decision — and is emitted
  purely as a diagnostic. It is a function of the fitted masses alone, hence
  reported once per fold rather than per fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from trace_collect.latency_validation import normalized_positive_floats
from trace_collect.tool_latency_hazard_model import (
    DEFAULT_CV_FOLDS,
    DEFAULT_L2_GRID,
    DiscreteTimeGrid,
    FittedHazardModel,
    build_log_grid,
    ensemble_survival_trigger_stats,
    fit_hazard_model,
    survival_clock_region_stats,
)
from trace_collect.tool_latency_offline_probe import (
    balanced_task_folds,
    select_probe_guard,
)
from trace_collect.tool_latency_survival_features import (
    SurvivalFeatureSpec,
    iter_causal_row_features,
)
from trace_collect.tool_latency_utility_clock import validate_restore_cost


# (name, treatment trigger field, baseline trigger field, enforce gated
# invariant). The hazard gate is an independent point-margin guard, so its
# trigger is never constrained to equal the robust or within-task baselines;
# every contrast disables ``enforce_gated_treatment``.
HAZARD_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    ("hazard_vs_deadline", "hazard_trigger_ms", "deadline_trigger_ms", False),
    (
        "gated_hazard_vs_deadline",
        "offline_gated_hazard_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "gated_hazard_vs_gated_robust",
        "offline_gated_hazard_trigger_ms",
        "offline_gated_robust_trigger_ms",
        False,
    ),
    (
        "gated_hazard_vs_gated_within_task",
        "offline_gated_hazard_trigger_ms",
        "gated_within_task_trigger_ms",
        False,
    ),
)


# Additional contrasts activated only when the bagged ensemble arm is on
# (``ensemble_members >= 2``). The ensemble's unanimity trigger and weakest-
# member margin apply an M-way jackknife over tasks — the trie's
# leave-one-task-out mechanism at coarser granularity (they coincide only at
# M == task count) — which a single hazard model lacks; these measure it
# against the deadline, the single hazard model, and the gated robust trie.
# Every contrast is an independent point-margin guard, so
# ``enforce_gated_treatment`` stays ``False``.
HAZARD_ENSEMBLE_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "ensemble_hazard_vs_deadline",
        "ensemble_hazard_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "gated_ensemble_vs_deadline",
        "offline_gated_ensemble_hazard_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "gated_ensemble_vs_gated_hazard",
        "offline_gated_ensemble_hazard_trigger_ms",
        "offline_gated_hazard_trigger_ms",
        False,
    ),
    (
        "gated_ensemble_vs_gated_robust",
        "offline_gated_ensemble_hazard_trigger_ms",
        "offline_gated_robust_trigger_ms",
        False,
    ),
)


@dataclass(frozen=True)
class _PredictedCall:
    """One call's rho-independent prediction: cached masses over a fixed grid.

    ``member_predictions`` caches the bagged ensemble arm's extra members for
    this call: each entry is one leave-one-Mth-out member's ``(grid, masses)``
    (its grid can differ from the full model's). It is empty when the ensemble
    arm is off; the full-data model — ``grid``/``masses`` above — is always the
    first ensemble member and is never refit, so an ``M``-member arm caches only
    the ``M`` partial members here.
    """

    sample_id: str
    task_id: str
    tool_name: str
    latency_ms: float
    grid: DiscreteTimeGrid
    masses: np.ndarray
    member_predictions: tuple[tuple[DiscreteTimeGrid, np.ndarray], ...] = ()


def evaluate_hazard_model_clock(
    eval_rows: Iterable[dict[str, Any]],
    *,
    profile_rows: Iterable[dict[str, Any]],
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    inner_folds: int,
    spec: SurvivalFeatureSpec,
    num_intervals: int,
    restore_cost_fractions: Sequence[float],
    l2_penalty: float | None = None,
    cv_folds: int = DEFAULT_CV_FOLDS,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
    model_family: str = "logistic",
    seed: int | None = None,
    ensemble_members: int = 0,
    calibration_edge_stride: int = 1,
    reliability_bin_count: int = 10,
) -> dict[str, Any]:
    """Fit the hazard clock once and score it at each requested restore fraction.

    Mirrors :func:`evaluate_offline_probe_clock`'s discipline. The grid
    quantiles and feature vocabulary are fit strictly on the fitting split
    (inner-train for the probe, full profile for the outer predictions), so no
    held-out task ever informs its own scored masses. The model and its
    predicted masses are fraction-independent, so they are computed once and
    every fraction in ``restore_cost_fractions`` reuses them to derive its
    probe margins, guard, triggers, and gating — producing results identical to
    refitting per fraction (rho never enters any fit).

    Returns ``by_restore_cost_fraction`` keyed by ``str(float(fraction))``,
    each carrying that fraction's per-``(sample, kv_cost)`` ``decisions`` and
    ``calibration_guard`` (the :func:`select_probe_guard` payload), plus the
    shared, rho-independent ``hazard_calibration`` diagnostics (see module
    docstring — eval labels are read for reporting only) and a config echo
    including ``num_intervals`` and the selected ``l2_penalty``. Per-call masses
    are never stored in decisions; the calibration block aggregates them.
    """

    kv_costs = normalized_positive_floats(kv_costs_ms, label="kv cost")
    fractions = list(restore_cost_fractions)
    if not fractions:
        raise ValueError("restore_cost_fractions must be non-empty")
    for fraction in fractions:
        validate_restore_cost(fraction, label="restore_cost_fraction")
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError("guard_ms must be finite and non-negative")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if num_intervals < 2:
        raise ValueError("num_intervals must be at least 2")
    if calibration_edge_stride < 1:
        raise ValueError("calibration_edge_stride must be at least 1")
    if reliability_bin_count < 1:
        raise ValueError("reliability_bin_count must be at least 1")
    # 0 = off; 1 would leave-one-1st-out every task (an empty subset), so the
    # bagged arm needs at least the 2-way leave-one-Mth-out partition.
    if ensemble_members < 0 or ensemble_members == 1:
        raise ValueError("ensemble_members must be 0 or >= 2, got "
                         f"{ensemble_members}")
    ensemble_on = ensemble_members >= 2

    profile_list = list(profile_rows)
    eval_list = list(eval_rows)
    if not profile_list or not eval_list:
        raise ValueError("evaluate_hazard_model_clock requires profile and eval rows")
    profile_tasks = {str(row["task_id"]) for row in profile_list}
    eval_tasks = {str(row["task_id"]) for row in eval_list}
    overlap = profile_tasks & eval_tasks
    if overlap:
        raise AssertionError(f"profile and eval tasks overlap: {sorted(overlap)}")

    if model_family not in ("logistic", "gbm"):
        raise ValueError(
            f"unknown model_family {model_family!r}; expected 'logistic' or 'gbm'"
        )

    # --- Fits (fraction-independent): outer model on all profile rows, l2 once.
    outer_grid = build_log_grid(_row_latencies(profile_list), num_intervals=num_intervals)
    outer_model = fit_hazard_model(
        profile_list,
        spec=spec,
        grid=outer_grid,
        l2_penalty=l2_penalty,
        cv_folds=cv_folds,
        l2_grid=l2_grid,
        model_family=model_family,
        seed=seed,
    )
    selected_l2 = outer_model.l2_penalty

    # The full-data model is ensemble member 0 and is never refit; the bagged
    # arm adds M partial members per fit context (see _fit_ensemble_members).
    outer_members = (
        _fit_ensemble_members(
            profile_list,
            spec=spec,
            num_intervals=num_intervals,
            selected_l2=selected_l2,
            model_family=model_family,
            seed=seed,
            ensemble_members=ensemble_members,
        )
        if ensemble_on
        else []
    )

    inner_predictions = _inner_probe_predictions(
        profile_list,
        spec=spec,
        inner_folds=inner_folds,
        num_intervals=num_intervals,
        selected_l2=selected_l2,
        profile_tasks=profile_tasks,
        model_family=model_family,
        seed=seed,
        ensemble_members=ensemble_members,
    )
    outer_predictions = _outer_eval_predictions(
        eval_list,
        spec=spec,
        outer_model=outer_model,
        outer_grid=outer_grid,
        outer_members=outer_members,
    )

    hazard_calibration = _hazard_calibration(
        outer_grid,
        outer_predictions,
        edge_stride=calibration_edge_stride,
        bin_count=reliability_bin_count,
    )

    # --- Per-fraction scoring reuses the cached masses; no refits.
    by_fraction: dict[str, dict[str, Any]] = {}
    for fraction in fractions:
        probe_decisions = _probe_decisions_at_fraction(
            inner_predictions,
            kv_costs=kv_costs,
            guard_ms=guard_ms,
            restore_cost_fraction=fraction,
            ensemble_on=ensemble_on,
        )
        calibration_guard = select_probe_guard(
            probe_decisions,
            score_field="hazard_margin_normalized",
            candidate_field="hazard_candidate_trigger_ms",
            restore_cost_fraction=fraction,
        )
        ensemble_calibration_guard = (
            select_probe_guard(
                probe_decisions,
                score_field="ensemble_hazard_margin_normalized",
                candidate_field="ensemble_hazard_trigger_ms",
                restore_cost_fraction=fraction,
            )
            if ensemble_on
            else None
        )
        decisions = _eval_decisions_at_fraction(
            outer_predictions,
            kv_costs=kv_costs,
            guard_ms=guard_ms,
            restore_cost_fraction=fraction,
            selected_guard=calibration_guard["selected_guard_normalized"],
            ensemble_on=ensemble_on,
            ensemble_guard=(
                ensemble_calibration_guard["selected_guard_normalized"]
                if ensemble_calibration_guard is not None
                else None
            ),
        )
        fraction_entry = {
            "restore_cost_fraction": fraction,
            "calibration_guard": calibration_guard,
            "decisions": decisions,
        }
        if ensemble_on:
            fraction_entry["ensemble_calibration_guard"] = ensemble_calibration_guard
        by_fraction[_fraction_key(fraction)] = fraction_entry

    result = {
        "by_restore_cost_fraction": by_fraction,
        "hazard_calibration": hazard_calibration,
        "kv_costs_ms": kv_costs,
        "guard_ms": guard_ms,
        "inner_folds": inner_folds,
        "num_intervals": num_intervals,
        "model_family": model_family,
        "l2_penalty": selected_l2,
        "l2_selected_on_full_profile": model_family == "logistic" and l2_penalty is None,
        "cv_folds": cv_folds,
        "l2_grid": [float(value) for value in l2_grid],
        "restore_cost_fractions": fractions,
        "calibration_edge_stride": calibration_edge_stride,
        "reliability_bin_count": reliability_bin_count,
        "spec": _spec_config(spec),
        "profile_row_count": len(profile_list),
        "profile_task_count": len(profile_tasks),
        "row_count": len(eval_list),
        "outer_grid_edges_ms": _finite_edges_list(outer_grid),
    }
    if ensemble_on:
        result["ensemble_members"] = ensemble_members
    return result


def _leave_one_mth_out_tasks(
    task_ids: Sequence[str], ensemble_members: int
) -> list[set[str]]:
    """Tasks left out by each of the ``M`` members, keyed by member index.

    Task ids are sorted deterministically and member ``m`` leaves out those at
    sorted position ``== m (mod M)``. Every task is therefore left out by
    exactly one member (the ``M``-way generalization of leave-one-task-out), and
    the assignment is a pure function of the id set and ``M`` — no randomness.
    """

    ordered = sorted(set(task_ids))
    return [
        {task for index, task in enumerate(ordered) if index % ensemble_members == member}
        for member in range(ensemble_members)
    ]


def _fit_ensemble_members(
    train_rows: Sequence[dict[str, Any]],
    *,
    spec: SurvivalFeatureSpec,
    num_intervals: int,
    selected_l2: float | None,
    model_family: str,
    seed: int | None,
    ensemble_members: int,
) -> list[FittedHazardModel]:
    """Fit the ``M`` leave-one-Mth-out members for one fitting context.

    The context's task ids are sorted deterministically and member ``m`` trains
    on every row whose task index is NOT ``== m (mod M)`` — an ``M``-way
    generalization of leave-one-task-out that drops a disjoint ``1/M`` of tasks
    per member with no randomness. Each member fits its own grid and hazard
    model on its subset (grids may differ), reusing the context's selected L2 and
    the shared ``model_family``/``seed`` exactly as the full model does. The
    full-data model is member 0 and is fit by the caller, so only these ``M``
    partial members are produced here. ``M >= 2`` is enforced upstream, so every
    member keeps at least one task.
    """

    task_ids = sorted({str(row["task_id"]) for row in train_rows})
    left_out_by_member = _leave_one_mth_out_tasks(task_ids, ensemble_members)
    members: list[FittedHazardModel] = []
    for member in range(ensemble_members):
        left_out = left_out_by_member[member]
        subset = [row for row in train_rows if str(row["task_id"]) not in left_out]
        if not subset:
            raise AssertionError(
                f"ensemble member {member} left out every training task"
            )
        member_grid = build_log_grid(_row_latencies(subset), num_intervals=num_intervals)
        members.append(
            fit_hazard_model(
                subset,
                spec=spec,
                grid=member_grid,
                l2_penalty=selected_l2,
                model_family=model_family,
                seed=seed,
            )
        )
    return members


def _member_predictions(
    members: Sequence[FittedHazardModel], feat: Any
) -> tuple[tuple[DiscreteTimeGrid, np.ndarray], ...]:
    """Cache each partial member's ``(grid, masses)`` for one call's features."""

    return tuple(
        (member.grid, member.predict_interval_masses(feat)) for member in members
    )


def _inner_probe_predictions(
    profile_list: Sequence[dict[str, Any]],
    *,
    spec: SurvivalFeatureSpec,
    inner_folds: int,
    num_intervals: int,
    selected_l2: float | None,
    profile_tasks: set[str],
    model_family: str,
    seed: int | None,
    ensemble_members: int,
) -> list[_PredictedCall]:
    """Task-OOF probe fits + mass predictions, computed once (rho-independent).

    Each inner fold fits the grid quantiles, feature vocabulary, and hazard
    model on the inner-train rows only, so a task present only in the held-out
    fold cannot inform its own probe masses (an unseen tool encodes to an
    all-zero block). The logistic penalty selected once on the full profile
    (``selected_l2``, ``None`` for the GBM) is reused for every inner fit;
    ``model_family`` and ``seed`` are the same as the outer fit. Masses are
    cached per held-out call for reuse across every restore fraction and kv
    cost.
    """

    profile_folds = balanced_task_folds(profile_list, fold_count=inner_folds)
    covered = set().union(*profile_folds)
    if covered != profile_tasks:
        raise AssertionError("inner folds do not cover every profile task")
    predictions: list[_PredictedCall] = []
    for held_out in profile_folds:
        inner_train = [
            row for row in profile_list if str(row["task_id"]) not in held_out
        ]
        inner_eval = [
            row for row in profile_list if str(row["task_id"]) in held_out
        ]
        inner_grid = build_log_grid(
            _row_latencies(inner_train), num_intervals=num_intervals
        )
        inner_model = fit_hazard_model(
            inner_train,
            spec=spec,
            grid=inner_grid,
            l2_penalty=selected_l2,
            model_family=model_family,
            seed=seed,
        )
        inner_members = (
            _fit_ensemble_members(
                inner_train,
                spec=spec,
                num_intervals=num_intervals,
                selected_l2=selected_l2,
                model_family=model_family,
                seed=seed,
                ensemble_members=ensemble_members,
            )
            if ensemble_members >= 2
            else []
        )
        inner_task_by_sample = {
            str(row["sample_id"]): str(row["task_id"]) for row in inner_eval
        }
        for feat in iter_causal_row_features(inner_eval, spec=spec):
            predictions.append(
                _PredictedCall(
                    sample_id=feat.sample_id,
                    task_id=inner_task_by_sample[feat.sample_id],
                    tool_name=feat.tool_name,
                    latency_ms=feat.latency_ms,
                    grid=inner_grid,
                    masses=inner_model.predict_interval_masses(feat),
                    member_predictions=_member_predictions(inner_members, feat),
                )
            )
    return predictions


def _outer_eval_predictions(
    eval_list: Sequence[dict[str, Any]],
    *,
    spec: SurvivalFeatureSpec,
    outer_model: FittedHazardModel,
    outer_grid: DiscreteTimeGrid,
    outer_members: Sequence[FittedHazardModel],
) -> list[_PredictedCall]:
    """Outer-model mass predictions over eval calls, computed once.

    When the bagged arm is on, ``outer_members`` are the leave-one-Mth-out
    partial members fit on the full profile; each call caches their masses
    alongside the full model's.
    """

    eval_task_by_sample = {
        str(row["sample_id"]): str(row["task_id"]) for row in eval_list
    }
    predictions: list[_PredictedCall] = []
    for feat in iter_causal_row_features(eval_list, spec=spec):
        predictions.append(
            _PredictedCall(
                sample_id=feat.sample_id,
                task_id=eval_task_by_sample[feat.sample_id],
                tool_name=feat.tool_name,
                latency_ms=feat.latency_ms,
                grid=outer_grid,
                masses=outer_model.predict_interval_masses(feat),
                member_predictions=_member_predictions(outer_members, feat),
            )
        )
    return predictions


def _ensemble_stats_for_call(
    call: _PredictedCall,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float,
) -> tuple[float, float]:
    """Ensemble unanimity trigger + weakest-member margin for one cached call.

    The full model (``call.grid``/``call.masses``) is always member 0; the
    cached partial members follow. Each member is passed as bare ``(reps,
    masses)`` because member grids may differ.
    """

    members = [(call.grid.reps, call.masses)]
    members.extend((grid.reps, masses) for grid, masses in call.member_predictions)
    stats = ensemble_survival_trigger_stats(
        members,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    return stats.trigger_ms, stats.normalized_advantage


def _probe_decisions_at_fraction(
    inner_predictions: Sequence[_PredictedCall],
    *,
    kv_costs: Sequence[float],
    guard_ms: float,
    restore_cost_fraction: float,
    ensemble_on: bool,
) -> list[dict[str, Any]]:
    """Derive inner-probe candidate triggers + margins at one restore fraction."""

    probe_decisions: list[dict[str, Any]] = []
    for call in inner_predictions:
        for cost_ms in kv_costs:
            threshold_ms = cost_ms + guard_ms
            restore_cost_ms = restore_cost_fraction * cost_ms
            stats = survival_clock_region_stats(
                call.grid,
                call.masses,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
                restore_cost_ms=restore_cost_ms,
            )
            decision = {
                "sample_id": call.sample_id,
                "task_id": call.task_id,
                "latency_ms": call.latency_ms,
                "kv_cost_ms": cost_ms,
                "threshold_ms": threshold_ms,
                "hazard_candidate_trigger_ms": stats.trigger_ms,
                "hazard_margin_normalized": stats.normalized_margin,
            }
            if ensemble_on:
                trigger_ms, margin = _ensemble_stats_for_call(
                    call,
                    threshold_ms=threshold_ms,
                    kv_cost_ms=cost_ms,
                    restore_cost_ms=restore_cost_ms,
                )
                decision["ensemble_hazard_trigger_ms"] = trigger_ms
                decision["ensemble_hazard_margin_normalized"] = margin
            probe_decisions.append(decision)
    return probe_decisions


def _eval_decisions_at_fraction(
    outer_predictions: Sequence[_PredictedCall],
    *,
    kv_costs: Sequence[float],
    guard_ms: float,
    restore_cost_fraction: float,
    selected_guard: float | None,
    ensemble_on: bool,
    ensemble_guard: float | None,
) -> list[dict[str, Any]]:
    """Derive gated eval triggers at one restore fraction from cached masses."""

    decisions: list[dict[str, Any]] = []
    for call in outer_predictions:
        for cost_ms in kv_costs:
            threshold_ms = cost_ms + guard_ms
            restore_cost_ms = restore_cost_fraction * cost_ms
            stats = survival_clock_region_stats(
                call.grid,
                call.masses,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
                restore_cost_ms=restore_cost_ms,
            )
            candidate_ms = stats.trigger_ms
            margin = stats.normalized_margin
            use_candidate = (
                selected_guard is not None
                and candidate_ms < threshold_ms
                and margin > selected_guard
            )
            decision = {
                "sample_id": call.sample_id,
                "task_id": call.task_id,
                "tool_name": call.tool_name,
                "latency_ms": call.latency_ms,
                "kv_cost_ms": cost_ms,
                "threshold_ms": threshold_ms,
                "deadline_trigger_ms": threshold_ms,
                "hazard_trigger_ms": candidate_ms,
                "hazard_margin_normalized": margin,
                "offline_gated_hazard_trigger_ms": (
                    candidate_ms if use_candidate else threshold_ms
                ),
                "offline_gated_hazard_guard_normalized": selected_guard,
            }
            if ensemble_on:
                ensemble_trigger_ms, ensemble_margin = _ensemble_stats_for_call(
                    call,
                    threshold_ms=threshold_ms,
                    kv_cost_ms=cost_ms,
                    restore_cost_ms=restore_cost_ms,
                )
                use_ensemble = (
                    ensemble_guard is not None
                    and ensemble_trigger_ms < threshold_ms
                    and ensemble_margin > ensemble_guard
                )
                decision["ensemble_hazard_trigger_ms"] = ensemble_trigger_ms
                decision["ensemble_hazard_margin_normalized"] = ensemble_margin
                decision["offline_gated_ensemble_hazard_trigger_ms"] = (
                    ensemble_trigger_ms if use_ensemble else threshold_ms
                )
                decision["offline_gated_ensemble_hazard_guard_normalized"] = (
                    ensemble_guard
                )
            decisions.append(decision)
    return decisions


def _hazard_calibration(
    grid: DiscreteTimeGrid,
    predictions: Sequence[_PredictedCall],
    *,
    edge_stride: int,
    bin_count: int,
) -> dict[str, Any]:
    """Predicted-vs-observed survival diagnostics pooled over eval calls.

    For each checkpoint edge (every finite grid edge, subsampled by
    ``edge_stride``) reports the mean predicted survival ``S(edge)`` and the
    observed fraction of eval calls with latency past that edge. Reliability
    bins partition the predicted survival at the median grid edge into
    ``bin_count`` equal-width bins and report the bin mean predicted survival,
    observed exceedance rate, and count. Rho-independent. All values are
    JSON-serializable.
    """

    latencies = np.array([call.latency_ms for call in predictions], dtype=float)
    masses = np.array([call.masses for call in predictions], dtype=float)
    # suffix[:, j] = P(latency >= edges[j]) = sum of interval masses at or above j.
    suffix = np.flip(np.cumsum(np.flip(masses, axis=1), axis=1), axis=1)
    finite_edges = grid.edges[:-1]  # drop the +inf sentinel
    num_edges = len(finite_edges)

    checkpoints: list[dict[str, Any]] = []
    for j in range(0, num_edges, edge_stride):
        edge_ms = float(finite_edges[j])
        checkpoints.append(
            {
                "edge_ms": edge_ms,
                "predicted_survival_mean": float(suffix[:, j].mean()),
                "observed_exceed_fraction": float(np.mean(latencies > edge_ms)),
                "count": int(latencies.size),
            }
        )

    median_index = num_edges // 2
    median_edge_ms = float(finite_edges[median_index])
    predicted_at_median = suffix[:, median_index]
    observed_at_median = latencies > median_edge_ms
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    assignments = np.clip(
        np.digitize(predicted_at_median, bin_edges[1:-1], right=False),
        0,
        bin_count - 1,
    )
    reliability_bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        mask = assignments == index
        count = int(np.count_nonzero(mask))
        reliability_bins.append(
            {
                "bin_index": index,
                "predicted_survival_mean": (
                    float(predicted_at_median[mask].mean()) if count else None
                ),
                "observed_exceed_rate": (
                    float(np.mean(observed_at_median[mask])) if count else None
                ),
                "count": count,
            }
        )
    return {
        "checkpoints": checkpoints,
        "median_edge_ms": median_edge_ms,
        "reliability_bins": reliability_bins,
        "edge_stride": edge_stride,
        "bin_count": bin_count,
    }


def _fraction_key(fraction: float) -> str:
    """Match ``restore_cost_analysis.fraction_key`` so the driver can index."""

    return str(float(fraction))


def _row_latencies(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(row["latency_ms"]) for row in rows]


def _finite_edges_list(grid: DiscreteTimeGrid) -> list[float]:
    return [float(edge) for edge in grid.edges[:-1]]


def _spec_config(spec: SurvivalFeatureSpec) -> dict[str, Any]:
    return {
        "command_field": spec.command_field,
        "max_prefix_depth": spec.max_prefix_depth,
        "skip_leading_cd": spec.skip_leading_cd,
        "use_tool_identity": spec.use_tool_identity,
        "use_command_prefix": spec.use_command_prefix,
        "use_within_task_history": spec.use_within_task_history,
        "use_task_aggregates": spec.use_task_aggregates,
    }


__all__ = [
    "HAZARD_COMPARISONS",
    "HAZARD_ENSEMBLE_COMPARISONS",
    "evaluate_hazard_model_clock",
]
