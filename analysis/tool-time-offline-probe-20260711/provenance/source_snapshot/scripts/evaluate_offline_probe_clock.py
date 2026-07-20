#!/usr/bin/env python3
"""Evaluate a nested task-OOF calibrated utility-clock guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.tool_latency_offline_probe import (
    load_and_evaluate_offline_probe_clock,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a utility guard from offline sampled tool calls."
    )
    parser.add_argument("--profile-latencies", required=True, type=Path)
    parser.add_argument("--eval-latencies", required=True, type=Path)
    parser.add_argument(
        "--kv-costs-ms",
        required=True,
        type=comma_separated_floats("kv cost"),
    )
    parser.add_argument("--guard-ms", type=float, default=0.0)
    parser.add_argument("--inner-folds", type=int, required=True)
    parser.add_argument("--min-tool-history", type=int, default=1)
    parser.add_argument("--min-profile-tasks", type=int, default=1)
    parser.add_argument("--command-field", default=None)
    parser.add_argument("--max-prefix-depth", type=int, default=4)
    parser.add_argument("--skip-leading-cd", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decisions-output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_and_evaluate_offline_probe_clock(
        args.eval_latencies,
        profile_path=args.profile_latencies,
        kv_costs_ms=args.kv_costs_ms,
        guard_ms=args.guard_ms,
        inner_folds=args.inner_folds,
        min_tool_history=args.min_tool_history,
        min_profile_tasks=args.min_profile_tasks,
        command_field=args.command_field,
        max_prefix_depth=args.max_prefix_depth,
        skip_leading_cd=args.skip_leading_cd,
    )
    decisions = summary.pop("decisions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.decisions_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in decisions
        ),
        encoding="utf-8",
    )
    print(
        f"Calibrated guard={summary['calibration']['selected_guard_normalized']} "
        f"and evaluated {summary['row_count']} rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
