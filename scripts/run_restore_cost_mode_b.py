#!/usr/bin/env python3
"""Refit offline-gated trigger policies with the restore cost in the objective.

Mode B counterpart to scripts/analyze_restore_cost_sweep.py: every fit stage
(inner probe scoring, guard selection, outer triggers) runs at each swept
``restore_cost_fraction`` and the paired bootstrap scores at the same
fraction. Inputs are the frozen fold splits and latency JSONLs written by
scripts/run_offline_gated_robust_confirmation.py, so data, folds, and config
match the certified run; the fraction-zero refit is asserted to reproduce the
frozen triggers exactly. See trace_collect.restore_cost_analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import run_mode_b_refit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refit offline-gated policies under restore costs."
    )
    parser.add_argument(
        "--confirmation-root",
        required=True,
        type=Path,
        help="Frozen confirmation output with provenance/manifest.json, "
        "data/f*_{eval,profile}.jsonl, and cv/f*_decisions.jsonl",
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
    result = run_mode_b_refit(
        args.confirmation_root,
        output_root=args.output_root,
        restore_cost_fractions=args.restore_cost_fractions,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    print(
        f"Refit {len(result['restore_cost_fractions'])} fractions over "
        f"{result['fold_count']} folds -> {args.output_root}"
    )


if __name__ == "__main__":
    main()
