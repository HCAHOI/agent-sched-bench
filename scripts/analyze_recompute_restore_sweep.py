#!/usr/bin/env python3
"""Re-score frozen gated triggers under the reload-vs-recompute restore choice.

P2 (second action, first M(t,S|P,A) step). The gated trigger stays exactly as
fitted at restore cost zero (Mode A); the restore charged on a fire on a short
call becomes ``min(fraction * kv_cost_ms, rate * context_length)``, where the
per-call context length is the resident KV token count recovered from the raw
traces (the same-iteration llm_call prompt_tokens). The paired task-cluster
bootstrap certifies the mechanism increment over swap-only restore.

Sensitivity only: the recompute-rate grid is a stand-in for a measured H100
prefill-vs-context curve, and triggers were fit under swap-only restore, so
this lower-bounds a recompute-aware policy's value. See
trace_collect.tool_latency_recompute and trace_collect.restore_cost_analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import (
    DEFAULT_RECOMPUTE_RATES_MS_PER_TOKEN,
    render_recompute_restore_markdown,
    run_recompute_restore_sweep,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score gated triggers under min(swap-in, recompute) restore."
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
        help="Swap-in cost per fired-on-short call, as fractions of kv cost",
    )
    parser.add_argument(
        "--recompute-rates-ms-per-token",
        type=comma_separated_floats("recompute rate"),
        default=list(DEFAULT_RECOMPUTE_RATES_MS_PER_TOKEN),
        help="Recompute (prefill) restore rate grid in ms per context token",
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_recompute_restore_sweep(
        args.confirmation_root,
        restore_cost_fractions=args.restore_cost_fractions,
        recompute_rates_ms_per_token=args.recompute_rates_ms_per_token,
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
            render_recompute_restore_markdown(result),
            encoding="utf-8",
        )
    print(
        f"Re-scored {result['decision_row_count']} decisions from "
        f"{result['fold_count']} folds under "
        f"{len(result['restore_cost_fractions'])} restore fractions x "
        f"{len(result['recompute_rates_ms_per_token'])} recompute rates "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
