#!/usr/bin/env python3
"""Sweep profiled KV-swap thresholds against observed tool-gap windows.

Usage:
  uv run python scripts/exploration/sweep_kv_profile.py \
    --profile path/to/kv_profile.jsonl \
    --gaps path/to/tool_gaps.jsonl \
    --quantile p95 \
    --guards-ms 0,50,100 \
    --output sweep_results.json

The profile file may be a tiny fixture for plumbing tests, but experiment runs
must use measurements from the actual serving engine's KV swap mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trace_collect.kv_profile_sweep import (
    evaluate_profile_sweep,
    filter_profiles,
    load_profile,
    load_tool_gaps,
)


def _nonnegative_floats(value: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("guards must be comma-separated numbers") from exc
    if not parsed or any(item < 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("guards must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate profiled KV swap costs against observed tool gaps."
    )
    parser.add_argument("--profile", required=True, type=Path, help="KV profile JSONL")
    parser.add_argument("--gaps", required=True, type=Path, help="Tool-gap JSONL")
    parser.add_argument(
        "--quantile",
        default="p95",
        choices=["p50", "p90", "p95", "p99"],
        help="Profile quantile used as KV cost (default: p95)",
    )
    parser.add_argument(
        "--guards-ms",
        type=_nonnegative_floats,
        default=[0.0],
        help="Comma-separated guard milliseconds added to KV cost (default: 0)",
    )
    parser.add_argument("--engine", default=None, help="Optional profile engine filter")
    parser.add_argument("--direction", default=None, help="Optional direction filter")
    parser.add_argument("--kv-size-mb", type=float, default=None, help="Optional KV size filter")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    return parser


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    profiles = filter_profiles(
        load_profile(args.profile),
        engine=args.engine,
        direction=args.direction,
        kv_size_mb=args.kv_size_mb,
    )
    gaps = load_tool_gaps(args.gaps)
    return evaluate_profile_sweep(
        profiles,
        gaps,
        quantile=args.quantile,
        guard_ms_values=args.guards_ms,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run(args)
    payload = json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {len(results)} sweep rows -> {args.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
