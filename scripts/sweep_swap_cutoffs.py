#!/usr/bin/env python3
"""Sweep swap-decision probability cutoffs against observed tool latencies.

Produces the risk-coverage trade-off for the causal survival predictor: per
(KV cost, cutoff) point, coverage, stall rate, and exposed / absorbed /
missed milliseconds. KV costs come either from an explicit list or from a
profiled KV swap cost file. The profile file may be a tiny fixture for
plumbing tests, but experiment runs must use measurements from the actual
serving engine's KV swap mechanism.

Usage:
  uv run python scripts/sweep_swap_cutoffs.py \
    --latencies tool_latencies.jsonl \
    --kv-profile kv_profile.jsonl --quantile p95 --guard-ms 50 \
    --cutoffs 0.1,0.3,0.5,0.7,0.9 \
    --output cutoff_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from trace_collect.kv_profile_sweep import filter_profiles, load_profile
from trace_collect.swap_cutoff_sweep import load_and_evaluate_swap_cutoff_sweep
from trace_collect.tool_latency_bucket import bucket_edges_from_profile


def _comma_floats(label: str) -> Callable[[str], list[float]]:
    def parse(value: str) -> list[float]:
        parsed: list[float] = []
        for raw in value.split(","):
            text = raw.strip()
            if not text:
                continue
            try:
                parsed.append(float(text))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid {label} {text!r}") from exc
        if not parsed:
            raise argparse.ArgumentTypeError(f"at least one {label} is required")
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep swap-decision probability cutoffs for tool latency."
    )
    parser.add_argument("--latencies", required=True, type=Path, help="Latency JSONL")
    costs_source = parser.add_mutually_exclusive_group(required=True)
    costs_source.add_argument(
        "--kv-costs-ms",
        type=_comma_floats("kv cost"),
        default=None,
        help="Comma-separated explicit KV swap costs, e.g. 100,200,500",
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
        "--cutoffs",
        required=True,
        type=_comma_floats("cutoff"),
        help="Comma-separated probability cutoffs in [0, 1], e.g. 0.1,0.5,0.9",
    )
    parser.add_argument(
        "--min-tool-history",
        type=int,
        default=1,
        help="Minimum completed same-tool history before using same-tool rate (default: 1)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write summary JSON")
    return parser


def resolve_kv_costs(args: argparse.Namespace) -> list[float]:
    if args.kv_costs_ms is not None:
        return args.kv_costs_ms
    profiles = filter_profiles(
        load_profile(args.kv_profile),
        engine=args.engine,
        direction=args.direction,
    )
    # With zero guard this is exactly the deduped profiled costs per entry.
    return bucket_edges_from_profile(profiles, quantile=args.quantile, guard_ms=0.0)


def main() -> None:
    args = build_parser().parse_args()
    summary = load_and_evaluate_swap_cutoff_sweep(
        args.latencies,
        kv_costs_ms=resolve_kv_costs(args),
        guard_ms=args.guard_ms,
        probability_cutoffs=args.cutoffs,
        min_tool_history=args.min_tool_history,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            f"Swept {len(summary['sweep'])} (kv cost, cutoff) points over "
            f"{summary['row_count']} rows -> {args.output}"
        )
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
