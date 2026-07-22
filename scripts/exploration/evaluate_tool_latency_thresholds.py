#!/usr/bin/env python3
"""Evaluate causal threshold decisions for observed tool latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.tool_latency_threshold import (
    load_and_evaluate_latency_thresholds,
    write_threshold_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate causal empirical survival decisions for tool latency thresholds."
    )
    parser.add_argument("--latencies", required=True, type=Path, help="Latency JSONL")
    parser.add_argument(
        "--thresholds-ms",
        required=True,
        type=comma_separated_floats("threshold"),
        help="Comma-separated latency thresholds, e.g. 100,200,500",
    )
    parser.add_argument(
        "--probability-cutoff",
        type=float,
        default=0.5,
        help="Predict exceeds-threshold when empirical survival >= cutoff (default: 0.5)",
    )
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
        help="Optional per-row threshold decisions JSONL",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_and_evaluate_latency_thresholds(
        args.latencies,
        thresholds_ms=args.thresholds_ms,
        probability_cutoff=args.probability_cutoff,
        min_tool_history=args.min_tool_history,
    )
    write_threshold_outputs(
        summary,
        summary_path=args.output,
        decisions_path=args.decisions_output,
    )
    payload = {key: value for key, value in summary.items() if key != "decisions"}
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Evaluated {summary['decision_count']} decisions over "
            f"{summary['row_count']} rows -> {args.output}"
        )


if __name__ == "__main__":
    main()
