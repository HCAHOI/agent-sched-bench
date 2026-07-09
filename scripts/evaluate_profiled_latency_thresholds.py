#!/usr/bin/env python3
"""Compare profiled-prior latency predictors on a held-out trace split.

Runs the prior_only / online_only / blended survival predictors over the
same eval rows and thresholds so they can be compared directly. The profile
and eval latency files must be extracted from disjoint task-level trace
splits (enforced via source_trace overlap check).

Usage:
  uv run python scripts/evaluate_profiled_latency_thresholds.py \
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

from trace_collect.tool_latency_profiled import (
    load_and_evaluate_profiled_latency_thresholds,
)

_ALL_PREDICTORS = ["prior_only", "online_only", "blended"]


def _comma_floats(value: str) -> list[float]:
    parsed: list[float] = []
    for raw in value.split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            parsed.append(float(text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid threshold {text!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return parsed


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
        type=_comma_floats,
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
        help="Minimum same-tool samples before using the per-tool rate (default: 1)",
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
        )

    summary_payload = {
        "predictors": args.predictors,
        "by_predictor": {
            predictor: {key: value for key, value in summary.items() if key != "decisions"}
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
