#!/usr/bin/env python3
"""Evaluate the deadline-triggered KV swap policy against t0-only decisions.

For each KV cost, reports side-by-side cost accounting for the plain t=0
threshold decision and the deadline policy that additionally starts a swap
once a running tool outlives the threshold (a proven-label late swap; see
trace_collect.swap_deadline_policy). Profile and eval latency files must
come from disjoint task-level trace splits. KV costs come either from an
explicit list or from a profiled KV swap cost file; experiment runs must
use measurements from the actual serving engine's swap mechanism.

Usage:
  uv run python scripts/evaluate_swap_deadline_policy.py \
    --profile-latencies profile_split.jsonl \
    --eval-latencies eval_split.jsonl \
    --kv-costs-ms 500,1000,2000 --guard-ms 0 \
    --command-field command --skip-leading-cd \
    --output deadline_policy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats, resolve_kv_costs
from trace_collect.swap_deadline_policy import load_and_evaluate_deadline_policy

_PREDICTORS = ["prior_only", "online_only", "blended"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare t0-only vs deadline-recheck KV swap policies."
    )
    parser.add_argument(
        "--profile-latencies",
        required=True,
        type=Path,
        help="Latency JSONL extracted from the profile trace split",
    )
    parser.add_argument(
        "--eval-latencies",
        required=True,
        type=Path,
        help="Latency JSONL extracted from the held-out eval trace split",
    )
    costs_source = parser.add_mutually_exclusive_group(required=True)
    costs_source.add_argument(
        "--kv-costs-ms",
        type=comma_separated_floats("kv cost"),
        default=None,
        help="Comma-separated explicit KV swap costs, e.g. 500,1000,2000",
    )
    costs_source.add_argument(
        "--kv-profile",
        type=Path,
        default=None,
        help="KV swap profile JSONL to derive KV costs from",
    )
    parser.add_argument(
        "--quantile",
        default="p95",
        choices=["p50", "p90", "p95", "p99"],
        help="Profile quantile used as KV cost (default: p95)",
    )
    parser.add_argument("--engine", default=None, help="Optional profile engine filter")
    parser.add_argument("--direction", default=None, help="Optional direction filter")
    parser.add_argument(
        "--guard-ms",
        type=float,
        default=0.0,
        help="Guard milliseconds added to each KV cost for the decision threshold (default: 0)",
    )
    parser.add_argument(
        "--predictor",
        default="prior_only",
        choices=_PREDICTORS,
        help="Survival predictor for the t=0 decision (default: prior_only)",
    )
    parser.add_argument(
        "--probability-cutoff",
        type=float,
        default=0.5,
        help="Swap at t=0 when survival estimate >= cutoff (default: 0.5)",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=None,
        help="Pseudo-observation weight of the prior for blended (default: raw pooling)",
    )
    parser.add_argument(
        "--min-tool-history",
        type=int,
        default=1,
        help="Minimum same-group/tool samples before using that rate (default: 1)",
    )
    parser.add_argument(
        "--command-field",
        default=None,
        help="tool_args field holding a shell command; enables prefix-tree grouping",
    )
    parser.add_argument(
        "--max-prefix-depth",
        type=int,
        default=4,
        help="Maximum command prefix-tree depth in tokens (default: 4)",
    )
    parser.add_argument(
        "--skip-leading-cd",
        action="store_true",
        help="Drop leading 'cd <dir> &&' segments before building prefix keys",
    )
    parser.add_argument(
        "--segment-costs",
        action="store_true",
        help=(
            "Fit an additive per-segment cost model on the profile split and "
            "key rows by their dominant segment (needs --command-field)"
        ),
    )
    parser.add_argument(
        "--segment-fit",
        default="nnls",
        choices=["nnls", "lad"],
        help="Segment-cost estimator: nnls (squared error) or lad (median regression)",
    )
    parser.add_argument(
        "--recheck",
        default="threshold",
        choices=["threshold", "hazard"],
        help=(
            "Re-check time for t0-declined calls: 'threshold' (k=T, late swaps "
            "never wrong) or 'hazard' (per-node expected-cost-optimal k<=T)"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Write summary JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_and_evaluate_deadline_policy(
        args.eval_latencies,
        profile_path=args.profile_latencies,
        kv_costs_ms=resolve_kv_costs(args),
        guard_ms=args.guard_ms,
        predictor=args.predictor,
        probability_cutoff=args.probability_cutoff,
        prior_strength=args.prior_strength,
        min_tool_history=args.min_tool_history,
        command_field=args.command_field,
        max_prefix_depth=args.max_prefix_depth,
        skip_leading_cd=args.skip_leading_cd,
        segment_costs=args.segment_costs,
        segment_fit=args.segment_fit,
        recheck=args.recheck,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            f"Evaluated {len(summary['points'])} kv points over "
            f"{summary['row_count']} rows -> {args.output}"
        )
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
