#!/usr/bin/env python3
"""Evaluate causal bucket predictions for observed tool latency.

Bucket edges come either from an explicit list or from a profiled KV swap
cost file (one edge per profiled entry: quantile cost + guard). The profile
file may be a tiny fixture for plumbing tests, but experiment runs must use
measurements from the actual serving engine's KV swap mechanism.

Usage:
  uv run python scripts/evaluate_tool_latency_buckets.py \
    --latencies tool_latencies.jsonl \
    --kv-profile kv_profile.jsonl --quantile p95 --guard-ms 50 \
    --output bucket_summary.json --decisions-output bucket_decisions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.kv_profile_sweep import filter_profiles, load_profile
from trace_collect.tool_latency_bucket import (
    bucket_edges_from_profile,
    load_and_evaluate_latency_buckets,
    write_bucket_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate causal empirical bucket predictions for tool latency."
    )
    parser.add_argument("--latencies", required=True, type=Path, help="Latency JSONL")
    edges_source = parser.add_mutually_exclusive_group(required=True)
    edges_source.add_argument(
        "--bucket-edges-ms",
        type=comma_separated_floats("bucket edge"),
        default=None,
        help="Comma-separated explicit bucket edges, e.g. 100,200,500",
    )
    edges_source.add_argument(
        "--kv-profile",
        type=Path,
        default=None,
        help="KV swap profile JSONL to derive bucket edges from",
    )
    parser.add_argument(
        "--quantile",
        default="p95",
        choices=["p50", "p90", "p95", "p99"],
        help="Profile quantile used as KV cost (default: p95)",
    )
    parser.add_argument(
        "--guard-ms",
        type=float,
        default=0.0,
        help="Guard milliseconds added to each profiled KV cost (default: 0)",
    )
    parser.add_argument("--engine", default=None, help="Optional profile engine filter")
    parser.add_argument("--direction", default=None, help="Optional direction filter")
    parser.add_argument(
        "--min-tool-history",
        type=int,
        default=1,
        help="Minimum completed same-tool history before using same-tool rate (default: 1)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write summary JSON")
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=None,
        help="Optional per-row bucket decisions JSONL",
    )
    return parser


def resolve_bucket_edges(args: argparse.Namespace) -> list[float]:
    if args.bucket_edges_ms is not None:
        return args.bucket_edges_ms
    profiles = filter_profiles(
        load_profile(args.kv_profile),
        engine=args.engine,
        direction=args.direction,
    )
    return bucket_edges_from_profile(
        profiles,
        quantile=args.quantile,
        guard_ms=args.guard_ms,
    )


def main() -> None:
    args = build_parser().parse_args()
    bucket_edges_ms = resolve_bucket_edges(args)
    summary = load_and_evaluate_latency_buckets(
        args.latencies,
        bucket_edges_ms=bucket_edges_ms,
        min_tool_history=args.min_tool_history,
    )
    write_bucket_outputs(
        summary,
        summary_path=args.output,
        decisions_path=args.decisions_output,
    )
    payload = {key: value for key, value in summary.items() if key != "decisions"}
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Evaluated {summary['row_count']} rows into "
            f"{summary['bucket_count']} buckets -> {args.output}"
        )


if __name__ == "__main__":
    main()
