#!/usr/bin/env python3
"""Re-score frozen trigger decisions under a swap-back (restore) cost sweep.

Mode A stress test: triggers stay exactly as fitted at restore cost zero and
only the evaluation utility charges each fire on a short call
``fraction * kv_cost_ms``. See trace_collect.restore_cost_analysis for the
analysis itself and scripts/run_restore_cost_mode_b.py for the refit (Mode B)
counterpart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import (
    analyze_restore_cost_sweep,
    render_summary_markdown,
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


if __name__ == "__main__":
    main()
