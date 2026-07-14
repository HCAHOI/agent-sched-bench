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

    lines = [
        f"# {title}",
        "",
        f"Source: `{result['confirmation_root']}`",
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


def _read_frozen_manifest(confirmation_root: Path) -> dict[str, Any]:
    manifest_path = confirmation_root / "provenance" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    required = (
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
    missing = [field for field in required if field not in manifest]
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
    "WITHIN_TASK_COMPARISONS",
    "analyze_restore_cost_sweep",
    "fraction_key",
    "load_fold_decisions",
    "render_summary_markdown",
    "run_mode_b_refit",
    "run_within_task_baseline",
]
