#!/usr/bin/env python3
"""Compute exact metric-v1 deadline-band headroom from OOF decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.tool_latency_headroom import (
    analyze_utility_headroom,
    load_decision_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze deadline-band headroom and policy capture."
    )
    parser.add_argument("decisions", nargs="+", type=Path)
    parser.add_argument(
        "--expected-costs-ms",
        required=True,
        type=comma_separated_floats("expected cost"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_decision_rows(args.decisions)
    result = analyze_utility_headroom(
        rows,
        expected_costs_ms=args.expected_costs_ms,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    result["input_paths"] = [str(path.resolve()) for path in args.decisions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Analyzed {result['sample_count']} samples across "
        f"{len(result['points'])} cost points -> {args.output}"
    )


if __name__ == "__main__":
    main()
