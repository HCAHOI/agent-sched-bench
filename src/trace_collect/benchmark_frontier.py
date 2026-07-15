"""Within-benchmark GBM-vs-trie frontier replicated on a second trace corpus.

The frozen certification and the hazard-model confirmation both rest on a single
corpus (swe-rebench-100): the whole GBM-vs-trie head-to-head lives there. This
harness replicates that *within-benchmark* comparison on any other trace corpus.
It builds task folds from a trace root (or a pre-extracted latency JSONL), and
for every fold at every restore fraction it fits and scores BOTH policies on the
same eval/profile split:

* the gated empirical trie via :func:`evaluate_offline_probe_clock` (whose fits
  are not rho-amortized, so it is called once per fraction, as in
  :func:`trace_collect.restore_cost_analysis.run_mode_b_refit`), and
* the gated learned hazard model via :func:`evaluate_hazard_model_clock`
  (rho-amortized, so one call per fold covers every fraction).

The two per-``(sample, kv_cost)`` decision sets are joined (mirroring
``_merge_hazard_rows``: latency/threshold must agree, unmatched rows in either
direction raise) and the head-to-head contrasts are bootstrapped by task.

This is a *sensitivity* replication, not a fresh certification: the first target
(Terminal-Bench) was dev-exposed during method development, so a positive result
is not independent evidence. The caller MUST state the corpus's exposure status
in ``exposure_note``; there is no default, and it is stamped into the payload and
the summary.

Output payloads keep ``schema_version`` 1 with additive keys only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from trace_collect.restore_cost_analysis import (
    _validate_fractions,
    _write_json,
    _write_jsonl,
    fraction_key,
    render_summary_markdown,
)
from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap
from trace_collect.tool_latency_hazard_eval import (
    build_survival_feature_spec,
    evaluate_hazard_model_clock,
)
from trace_collect.tool_latency_offline_probe import evaluate_offline_probe_clock
from trace_collect.tool_latency_transfer import _load_eval_rows


# (name, treatment field, baseline field, enforce gated-treatment invariant).
# The two gated policies are each measured against never firing early (the fixed
# deadline), then head-to-head. Both gates are independent point-margin guards,
# so the gated-treatment invariant stays off for every contrast.
FRONTIER_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "gated_robust_vs_deadline",
        "offline_gated_robust_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
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
)


# Extra contrasts activated only with the tool-name-only trie arm
# (``tool_name_trie=True``): the tool-name trie (Continuum's tool-identity prior
# with cold-start back-off, i.e. ``command_field=None``) against the deadline,
# and the two full-conditioning gated policies against it. The head-to-heads
# quantify what command-prefix conditioning (full trie) and the learned model
# (GBM) buy over tool-name conditioning alone. As in every frontier contrast the
# gates are independent point-margin guards, so the gated-treatment invariant
# stays off.
FRONTIER_TOOL_NAME_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "gated_tool_name_vs_deadline",
        "offline_gated_tool_name_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    (
        "gated_robust_vs_gated_tool_name",
        "offline_gated_robust_trigger_ms",
        "offline_gated_tool_name_trigger_ms",
        False,
    ),
    (
        "gated_hazard_vs_gated_tool_name",
        "offline_gated_hazard_trigger_ms",
        "offline_gated_tool_name_trigger_ms",
        False,
    ),
)


# Extra contrasts activated only with the bagged ensemble arm (``M >= 2``): the
# gated ensemble against the deadline, the single hazard model, and the gated
# trie.
FRONTIER_ENSEMBLE_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
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


def run_benchmark_frontier(
    trace_root: Path | None = None,
    *,
    output_root: Path,
    fold_count: int,
    inner_folds: int,
    kv_costs_ms: list[float],
    guard_ms: float,
    min_tool_history: int,
    min_profile_tasks: int,
    command_field: str | None,
    max_prefix_depth: int,
    skip_leading_cd: bool,
    num_intervals: int,
    model_family: str,
    seed: int,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    exposure_note: str,
    eval_latencies: Path | None = None,
    feature_set: str = "full",
    ensemble_members: int = 0,
    tool_name_trie: bool = False,
) -> dict[str, Any]:
    """Fold a target corpus and bootstrap the gated trie vs gated hazard frontier.

    Exactly one of ``trace_root`` (canonical ``trace.jsonl`` files to extract)
    or ``eval_latencies`` (a pre-extracted tool-latency JSONL) supplies the
    corpus. Tasks are split into ``fold_count`` outer folds by sorted-id index
    modulo ``fold_count`` (the frozen protocol); each fold's eval tasks are
    scored against a disjoint profile. Both policies are fit and scored at every
    restore fraction, the per-fold decisions are joined, and the contrasts in
    :data:`FRONTIER_COMPARISONS` (plus :data:`FRONTIER_ENSEMBLE_COMPARISONS`
    when ``ensemble_members >= 2``) are bootstrapped per fraction.
    """

    _validate_fractions(restore_cost_fractions)
    if not isinstance(exposure_note, str) or not exposure_note.strip():
        raise ValueError("exposure_note must be a non-empty string")
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if num_intervals < 2:
        raise ValueError("num_intervals must be at least 2")
    if model_family not in ("logistic", "gbm"):
        raise ValueError(
            f"unknown model_family {model_family!r}; expected 'logistic' or 'gbm'"
        )
    if ensemble_members < 0 or ensemble_members == 1:
        raise ValueError(
            f"ensemble_members must be 0 or >= 2, got {ensemble_members}"
        )
    ensemble_on = ensemble_members >= 2
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")

    costs = [float(cost) for cost in kv_costs_ms]
    rows, eval_source = _load_eval_rows(
        eval_trace_root=trace_root,
        eval_latencies=eval_latencies,
    )
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_task.setdefault(str(row["task_id"]), []).append(row)
    task_ids = sorted(rows_by_task)
    if len(task_ids) < fold_count:
        raise ValueError(
            f"corpus has {len(task_ids)} tasks, fewer than fold_count={fold_count}"
        )

    spec = build_survival_feature_spec(
        feature_set,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    active_comparisons = (
        FRONTIER_COMPARISONS
        + (FRONTIER_TOOL_NAME_COMPARISONS if tool_name_trie else ())
        + (FRONTIER_ENSEMBLE_COMPARISONS if ensemble_on else ())
    )

    output_root.mkdir(parents=True)
    folds_root = output_root / "folds"
    folds_root.mkdir()
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in active_comparisons
    }
    trie_calibrations_by_fraction: dict[str, list[dict[str, Any]]] = {}
    hazard_guards: dict[str, list[dict[str, Any]]] = {}
    ensemble_guards: dict[str, list[dict[str, Any]]] = {}
    decisions_by_fraction: dict[str, list[dict[str, Any]]] = {}
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        (output_root / f"rho_{key}").mkdir()
        trie_calibrations_by_fraction[key] = []
        hazard_guards[key] = []
        ensemble_guards[key] = []
        decisions_by_fraction[key] = []

    fold_task_sets: list[dict[str, Any]] = []
    for fold in range(1, fold_count + 1):
        fold_name = f"f{fold}"
        eval_tasks = {
            task_id
            for index, task_id in enumerate(task_ids)
            if index % fold_count == fold - 1
        }
        profile_tasks = set(task_ids) - eval_tasks
        # Modulo folds guarantee disjointness, but the guard is only causally
        # clean if profile (fit) and eval (apply) tasks are disjoint, so verify.
        overlap = eval_tasks & profile_tasks
        if overlap:
            raise AssertionError(
                f"fold {fold_name} profile and eval tasks overlap: {sorted(overlap)}"
            )
        if not eval_tasks or not profile_tasks:
            raise AssertionError(
                f"fold {fold_name} has an empty eval or profile split"
            )
        _write_task_set(folds_root / f"{fold_name}_eval.txt", eval_tasks)
        _write_task_set(folds_root / f"{fold_name}_profile.txt", profile_tasks)
        fold_task_sets.append(
            {
                "fold": fold_name,
                "eval_tasks": sorted(eval_tasks),
                "profile_tasks": sorted(profile_tasks),
            }
        )
        eval_rows = _rows_for_tasks(rows_by_task, eval_tasks)
        profile_rows = _rows_for_tasks(rows_by_task, profile_tasks)

        # Hazard model: one rho-amortized fit per fold covers every fraction.
        fold_hazard = evaluate_hazard_model_clock(
            eval_rows,
            profile_rows=profile_rows,
            kv_costs_ms=costs,
            guard_ms=guard_ms,
            inner_folds=inner_folds,
            spec=spec,
            num_intervals=num_intervals,
            restore_cost_fractions=restore_cost_fractions,
            model_family=model_family,
            seed=seed,
            ensemble_members=ensemble_members,
        )
        hazard_shared = {
            field: value
            for field, value in fold_hazard.items()
            if field != "by_restore_cost_fraction"
        }
        for fraction in restore_cost_fractions:
            key = fraction_key(fraction)
            fraction_root = output_root / f"rho_{key}"
            # Empirical trie: refit per fraction (its fits carry rho).
            trie_result = evaluate_offline_probe_clock(
                eval_rows,
                profile_rows=profile_rows,
                kv_costs_ms=costs,
                guard_ms=guard_ms,
                inner_folds=inner_folds,
                min_tool_history=min_tool_history,
                min_profile_tasks=min_profile_tasks,
                command_field=command_field,
                max_prefix_depth=max_prefix_depth,
                skip_leading_cd=skip_leading_cd,
                restore_cost_fraction=fraction,
            )
            trie_decisions = trie_result.pop("decisions")
            per_fraction = fold_hazard["by_restore_cost_fraction"][key]
            hazard_rows = per_fraction["decisions"]
            merged = _merge_trie_hazard_rows(
                trie_decisions,
                hazard_rows,
                fold_name=fold_name,
                ensemble_on=ensemble_on,
            )
            tool_name_result: dict[str, Any] | None = None
            if tool_name_trie:
                # Second empirical trie on the SAME fold split, conditioned on
                # tool identity only (command_field=None disables prefix
                # grouping, leaving tool + global back-off nodes) — Continuum's
                # P(tau, f). Refit per fraction like the full trie.
                tool_name_result = evaluate_offline_probe_clock(
                    eval_rows,
                    profile_rows=profile_rows,
                    kv_costs_ms=costs,
                    guard_ms=guard_ms,
                    inner_folds=inner_folds,
                    min_tool_history=min_tool_history,
                    min_profile_tasks=min_profile_tasks,
                    command_field=None,
                    max_prefix_depth=max_prefix_depth,
                    skip_leading_cd=skip_leading_cd,
                    restore_cost_fraction=fraction,
                )
                merged = _merge_tool_name_rows(
                    merged,
                    tool_name_result.pop("decisions"),
                    fold_name=fold_name,
                )
            decisions_by_fraction[key].extend(merged)

            _write_json(fraction_root / f"{fold_name}_trie_summary.json", trie_result)
            if tool_name_result is not None:
                _write_json(
                    fraction_root / f"{fold_name}_tool_name_trie_summary.json",
                    tool_name_result,
                )
            hazard_summary = {
                **hazard_shared,
                "restore_cost_fraction": fraction,
                "calibration_guard": per_fraction["calibration_guard"],
            }
            if "ensemble_calibration_guard" in per_fraction:
                hazard_summary["ensemble_calibration_guard"] = per_fraction[
                    "ensemble_calibration_guard"
                ]
            _write_json(
                fraction_root / f"{fold_name}_hazard_summary.json", hazard_summary
            )
            _write_jsonl(fraction_root / f"{fold_name}_decisions.jsonl", merged)
            trie_calibrations_by_fraction[key].append(
                {
                    "fold": fold_name,
                    "calibration": trie_result["calibration"],
                    "robust_calibration": trie_result["robust_calibration"],
                }
            )
            hazard_guards[key].append(
                {"fold": fold_name, **per_fraction["calibration_guard"]}
            )
            if ensemble_on:
                ensemble_guards[key].append(
                    {"fold": fold_name, **per_fraction["ensemble_calibration_guard"]}
                )

    expected_row_count = len(rows) * len(costs)
    decision_count = 0
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        decisions = decisions_by_fraction[key]
        _write_jsonl(output_root / f"rho_{key}_decisions.jsonl", decisions)
        if len(decisions) != expected_row_count:
            raise AssertionError(
                "merged decision count does not match corpus rows x costs: "
                f"{len(decisions)} != {expected_row_count}"
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
        "mode": "benchmark_frontier",
        "feature_set": feature_set,
        "model_family": model_family,
        "ensemble_members": ensemble_members,
        "tool_name_trie": tool_name_trie,
        "trace_root": str(eval_source),
        "exposure_note": exposure_note,
        "fold_count": fold_count,
        "inner_folds": inner_folds,
        "task_count": len(task_ids),
        "row_count": len(rows),
        "decision_row_count": decision_count,
        "costs_ms": costs,
        "num_intervals": num_intervals,
        "restore_cost_fractions": restore_cost_fractions,
        "fold_task_sets": fold_task_sets,
        "trie_calibrations_by_fraction": trie_calibrations_by_fraction,
        "hazard_guards": hazard_guards,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }
    if ensemble_on:
        result["ensemble_guards"] = ensemble_guards
    _write_json(output_root / "benchmark_frontier.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Within-benchmark frontier (gated trie vs gated hazard)",
            intro_lines=(
                "Both the empirical trie and the learned hazard model are fit",
                "and scored on the same fold splits at each restore fraction;",
                "every contrast is a within-benchmark head-to-head. "
                + exposure_note,
            ),
        ),
        encoding="utf-8",
    )
    return result


def _merge_trie_hazard_rows(
    trie_rows: list[dict[str, Any]],
    hazard_rows: list[dict[str, Any]],
    *,
    fold_name: str,
    ensemble_on: bool,
) -> list[dict[str, Any]]:
    """Join the trie and hazard decision sets by ``(sample_id, kv_cost)``.

    The trie decision is the base (it carries ``deadline_trigger_ms`` and
    ``offline_gated_robust_trigger_ms``); the hazard row contributes the gated
    hazard trigger (and, when the ensemble arm is on, the gated ensemble
    trigger). Latency and threshold must agree between the two, and unmatched
    rows in either direction raise, mirroring
    :func:`trace_collect.restore_cost_analysis._merge_hazard_rows`.
    """

    hazard_by_key = {
        (str(row["sample_id"]), float(row["kv_cost_ms"])): row for row in hazard_rows
    }
    merged: list[dict[str, Any]] = []
    for trie in trie_rows:
        key = (str(trie["sample_id"]), float(trie["kv_cost_ms"]))
        hazard = hazard_by_key.pop(key, None)
        if hazard is None:
            raise ValueError(f"no hazard row for trie decision {key}")
        for field in ("latency_ms", "threshold_ms"):
            if not math.isclose(
                float(trie[field]),
                float(hazard[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{field} mismatch for decision {key}")
        merged_row = {
            **trie,
            "outer_fold": fold_name,
            "hazard_trigger_ms": hazard["hazard_trigger_ms"],
            "hazard_margin_normalized": hazard["hazard_margin_normalized"],
            "offline_gated_hazard_trigger_ms": hazard[
                "offline_gated_hazard_trigger_ms"
            ],
            "offline_gated_hazard_guard_normalized": hazard[
                "offline_gated_hazard_guard_normalized"
            ],
        }
        if ensemble_on:
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


def _merge_tool_name_rows(
    merged_rows: list[dict[str, Any]],
    tool_name_rows: list[dict[str, Any]],
    *,
    fold_name: str,
) -> list[dict[str, Any]]:
    """Join the tool-name trie's gated trigger onto the merged trie/hazard rows.

    The tool-name trie is a second :func:`evaluate_offline_probe_clock` run with
    ``command_field=None`` on the identical fold split, so every
    ``(sample_id, kv_cost)`` panel matches exactly. Its
    ``offline_gated_robust_trigger_ms`` (the gated tool-identity clock) enters as
    ``offline_gated_tool_name_trigger_ms``; latency, threshold, and the shared
    deadline must agree, and unmatched rows in either direction raise — the same
    join discipline as :func:`_merge_trie_hazard_rows`.
    """

    tool_name_by_key = {
        (str(row["sample_id"]), float(row["kv_cost_ms"])): row
        for row in tool_name_rows
    }
    result: list[dict[str, Any]] = []
    for base in merged_rows:
        key = (str(base["sample_id"]), float(base["kv_cost_ms"]))
        tool_name = tool_name_by_key.pop(key, None)
        if tool_name is None:
            raise ValueError(f"no tool-name row for decision {key}")
        for field in ("latency_ms", "threshold_ms", "deadline_trigger_ms"):
            if not math.isclose(
                float(base[field]),
                float(tool_name[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{field} mismatch for tool-name decision {key}")
        result.append(
            {
                **base,
                "offline_gated_tool_name_trigger_ms": tool_name[
                    "offline_gated_robust_trigger_ms"
                ],
                "offline_gated_tool_name_guard_normalized": tool_name[
                    "offline_gated_robust_guard_normalized"
                ],
                "tool_name_prior_source": tool_name["probe_robust_source"],
                "tool_name_prior_group_key": tool_name["probe_robust_group_key"],
            }
        )
    if tool_name_by_key:
        raise ValueError(
            f"{len(tool_name_by_key)} tool-name rows unmatched in {fold_name}"
        )
    return result


def _rows_for_tasks(
    rows_by_task: dict[str, list[dict[str, Any]]],
    task_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for task_id in sorted(task_ids) for row in rows_by_task[task_id]]


def _write_task_set(path: Path, task_ids: set[str]) -> None:
    path.write_text(
        "".join(f"{task_id}\n" for task_id in sorted(task_ids)),
        encoding="utf-8",
    )


__all__ = [
    "FRONTIER_COMPARISONS",
    "FRONTIER_ENSEMBLE_COMPARISONS",
    "FRONTIER_TOOL_NAME_COMPARISONS",
    "run_benchmark_frontier",
]
