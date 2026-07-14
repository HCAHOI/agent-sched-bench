#!/usr/bin/env python3
"""Score the within-task history baseline (B1) against the refit gated policy.

The strongest simple competitor from the formulation critique: predict each
call from the current task's own earlier calls only (deepest command-prefix
context, then tool level), using the same hazard-recheck estimator and the
same restore-cost-aware utility. Runs two variants side by side: the ungated
B1 and a gated B1 whose per-fold margin guard is fitted on that fold's profile
tasks and applied to its disjoint eval tasks — the same cross-fitted guard
protection the cross-task method has, so the ungated collapse under restore
cost is compared against a fair control. Both merge into the Mode B refit
decisions so every contrast shares fitting and scoring restore fractions.
See trace_collect.restore_cost_analysis.run_within_task_baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import run_within_task_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare within-task baseline triggers under restore costs."
    )
    parser.add_argument(
        "--confirmation-root",
        required=True,
        type=Path,
        help="Frozen confirmation output with provenance/manifest.json and "
        "data/f*_eval.jsonl",
    )
    parser.add_argument(
        "--mode-b-root",
        required=True,
        type=Path,
        help="Mode B refit output with rho_*/f*_decisions.jsonl",
    )
    parser.add_argument(
        "--restore-cost-fractions",
        type=comma_separated_floats("restore cost fraction"),
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_within_task_baseline(
        args.confirmation_root,
        mode_b_root=args.mode_b_root,
        output_root=args.output_root,
        restore_cost_fractions=args.restore_cost_fractions,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    print(
        f"Scored within-task baseline over {result['decision_row_count']} decisions "
        f"at {len(result['restore_cost_fractions'])} fractions -> {args.output_root}"
    )


if __name__ == "__main__":
    main()
