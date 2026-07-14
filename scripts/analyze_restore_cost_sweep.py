#!/usr/bin/env python3
"""Re-score frozen trigger decisions under a swap-back (restore) cost sweep.

The deployed utility model charges nothing for a swap that completes inside a
call which then finishes before the deadline, although a real system must swap
state back on the critical path. This script stress-tests existing decisions:
triggers stay exactly as fitted at restore cost zero, and only the evaluation
utility charges each fire on a short call ``fraction * kv_cost_ms``. Gains
that vanish under realistic fractions were artifacts of the free unnecessary
swap, not of latency prediction. Refitting triggers with the restore cost in
the objective is a separate (Mode B) analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap


_FOLD_DECISIONS_PATTERN = re.compile(r"^f(\d+)_decisions\.jsonl$")

# (name, treatment field, baseline field, enforce gated-treatment invariant).
# The first comparison re-scores the frozen certification contrast; the other
# two measure each early policy against never firing early at all.
_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    ("gated_vs_robust", "offline_gated_robust_trigger_ms", "robust_trigger_ms", True),
    (
        "gated_vs_deadline",
        "offline_gated_robust_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    ("robust_vs_deadline", "robust_trigger_ms", "deadline_trigger_ms", False),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score fitted trigger decisions under restore costs."
    )
    parser.add_argument(
        "--confirmation-root",
        required=True,
        type=Path,
        help="Confirmation output directory containing cv/f*_decisions.jsonl",
    )
    parser.add_argument(
        "--restore-cost-fractions",
        type=comma_separated_floats("restore cost fraction"),
        default=[0.0, 0.25, 0.5, 1.0],
        help="Swap-back cost per fired-on-short call, as fractions of kv cost",
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_restore_cost_sweep(
        args.confirmation_root,
        restore_cost_fractions=args.restore_cost_fractions,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            render_summary_markdown(result),
            encoding="utf-8",
        )
    print(
        f"Re-scored {result['decision_row_count']} decisions from "
        f"{result['fold_count']} folds under {len(result['restore_cost_fractions'])} "
        f"restore fractions -> {args.output}"
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

    if not restore_cost_fractions:
        raise ValueError("restore_cost_fractions must be non-empty")
    if len(set(restore_cost_fractions)) != len(restore_cost_fractions):
        raise ValueError("restore_cost_fractions must be unique")
    decisions, fold_names = _load_fold_decisions(confirmation_root)
    costs = sorted({float(row["kv_cost_ms"]) for row in decisions})

    comparisons: dict[str, dict[str, Any]] = {}
    for name, treatment_field, baseline_field, enforce_gated in _COMPARISONS:
        by_fraction: dict[str, Any] = {}
        for fraction in restore_cost_fractions:
            by_fraction[_fraction_key(fraction)] = paired_task_cluster_bootstrap(
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


def render_summary_markdown(result: dict[str, Any]) -> str:
    """Summarize simultaneous labels and total deltas per fraction."""

    lines = [
        "# Restore-cost re-scoring sweep",
        "",
        f"Source: `{result['confirmation_root']}`",
        "",
        "Triggers are frozen as fitted at restore cost zero; only the",
        "evaluation utility charges fires on short calls. Labels use the",
        "Bonferroni-corrected simultaneous intervals over all kv costs",
        "within one comparison-fraction cell; they are not corrected across",
        "comparisons or fractions, so read each row as its own what-if.",
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
            payload = comparison["by_restore_cost_fraction"][_fraction_key(fraction)]
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


def _load_fold_decisions(
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


def _fraction_key(fraction: float) -> str:
    return str(float(fraction))


if __name__ == "__main__":
    main()
