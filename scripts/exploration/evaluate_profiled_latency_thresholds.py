#!/usr/bin/env python3
"""Compare profiled-prior latency predictors on a held-out trace split.

Runs the prior_only / online_only / blended survival predictors over the
same eval rows and thresholds so they can be compared directly. The profile
and eval latency files must be extracted from disjoint task-level trace
splits (enforced via source_trace and logical task_id overlap checks).

Usage:
  uv run python scripts/exploration/evaluate_profiled_latency_thresholds.py \
    --profile-latencies profile_split.jsonl \
    --eval-latencies eval_split.jsonl \
    --thresholds-ms 100,200,500 \
    --abstain-confidence 0.95 \
    --output profiled_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.tool_latency_profiled import (
    load_and_evaluate_profiled_latency_thresholds,
)

_ALL_PREDICTORS = ["prior_only", "online_only", "blended"]


def _comma_predictors(value: str) -> list[str]:
    names = [text.strip() for text in value.split(",") if text.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one predictor is required")
    for name in names:
        if name not in _ALL_PREDICTORS:
            raise argparse.ArgumentTypeError(
                f"unknown predictor {name!r}; choose from {', '.join(_ALL_PREDICTORS)}"
            )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare profiled-prior survival predictors on held-out traces."
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
    parser.add_argument(
        "--thresholds-ms",
        required=True,
        type=comma_separated_floats("threshold"),
        help="Comma-separated latency thresholds, e.g. 100,200,500",
    )
    parser.add_argument(
        "--predictors",
        type=_comma_predictors,
        default=_ALL_PREDICTORS,
        help="Comma-separated predictors to run (default: all three)",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=None,
        help="Pseudo-observation weight of the prior for blended (default: raw pooling)",
    )
    parser.add_argument(
        "--probability-cutoff",
        type=float,
        default=0.5,
        help="Predict exceeds-threshold when survival estimate >= cutoff (default: 0.5)",
    )
    parser.add_argument(
        "--abstain-confidence",
        type=float,
        default=None,
        help="Wilson abstain-band confidence level, e.g. 0.95 (default: no abstain band)",
    )
    parser.add_argument(
        "--min-tool-history",
        type=int,
        default=1,
        help="Minimum same-group/tool samples before using that rate (default: 1)",
    )
    parser.add_argument(
        "--min-profile-tasks",
        type=int,
        default=1,
        help=(
            "Minimum distinct profile tasks before using a group/tool prior "
            "node (default: 1)"
        ),
    )
    parser.add_argument(
        "--prior-aggregation",
        choices=["call", "task"],
        default="call",
        help=(
            "Prior ECDF weighting: pooled calls or equal weight per task; "
            "task is supported only with prior_only (default: call)"
        ),
    )
    parser.add_argument(
        "--command-field",
        default=None,
        help=(
            "tool_args field holding a shell command (e.g. 'command'); enables "
            "data-derived command prefix-tree grouping (default: off)"
        ),
    )
    parser.add_argument(
        "--max-prefix-depth",
        type=int,
        default=4,
        help=(
            "Maximum command prefix-tree depth in tokens; deeper nodes need "
            "enough samples before being used. Only takes effect together "
            "with --command-field (default: 4)"
        ),
    )
    parser.add_argument(
        "--skip-leading-cd",
        action="store_true",
        help=(
            "Drop leading 'cd <dir> &&' segments before building prefix keys "
            "so the depth budget indexes the workload (needs --command-field)"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Write summary JSON")
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=None,
        help="Optional per-row decisions JSONL, tagged with predictor",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    by_predictor: dict[str, dict[str, object]] = {}
    for predictor in args.predictors:
        by_predictor[predictor] = load_and_evaluate_profiled_latency_thresholds(
            args.eval_latencies,
            profile_path=args.profile_latencies,
            thresholds_ms=args.thresholds_ms,
            predictor=predictor,
            prior_strength=args.prior_strength,
            probability_cutoff=args.probability_cutoff,
            abstain_confidence=args.abstain_confidence,
            min_tool_history=args.min_tool_history,
            min_profile_tasks=args.min_profile_tasks,
            prior_aggregation=args.prior_aggregation,
            command_field=args.command_field,
            max_prefix_depth=args.max_prefix_depth,
            skip_leading_cd=args.skip_leading_cd,
        )

    summary_payload = {
        "predictors": args.predictors,
        "by_predictor": {
            predictor: {
                key: value for key, value in summary.items() if key != "decisions"
            }
            for predictor, summary in by_predictor.items()
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if args.decisions_output is not None:
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.decisions_output.open("w", encoding="utf-8") as fh:
            for predictor in args.predictors:
                for row in by_predictor[predictor]["decisions"]:
                    fh.write(
                        json.dumps(
                            {"predictor": predictor, **row},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    fh.write("\n")

    if args.output is None:
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        total_decisions = sum(
            summary["decision_count"] for summary in by_predictor.values()
        )
        print(
            f"Evaluated {len(args.predictors)} predictors, "
            f"{total_decisions} decisions -> {args.output}"
        )


if __name__ == "__main__":
    main()
