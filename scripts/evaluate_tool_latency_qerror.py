#!/usr/bin/env python3
"""Evaluate causal q-error for observed tool latency prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.tool_latency_qerror import (
    load_and_evaluate_latency_qerror,
    write_qerror_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate causal historical-quantile q-error for tool latency."
    )
    parser.add_argument("--latencies", required=True, type=Path, help="Latency JSONL")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Historical quantile predictor, e.g. 0.5 for median (default: 0.5)",
    )
    parser.add_argument(
        "--epsilon-ms",
        type=float,
        default=1.0,
        help="Positive q-error denominator floor for near-zero latencies (default: 1.0)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write summary JSON")
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Optional per-row predictions JSONL",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_and_evaluate_latency_qerror(
        args.latencies,
        quantile=args.quantile,
        epsilon_ms=args.epsilon_ms,
    )
    write_qerror_outputs(
        summary,
        summary_path=args.output,
        predictions_path=args.predictions_output,
    )
    payload = {key: value for key, value in summary.items() if key != "predictions"}
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Evaluated {summary['evaluated_count']} / {summary['row_count']} rows "
            f"-> {args.output}"
        )


if __name__ == "__main__":
    main()
