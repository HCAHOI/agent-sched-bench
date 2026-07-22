#!/usr/bin/env python3
"""Transfer frozen offline-gated triggers from SWE-ReBench to a target corpus.

E2 cross-benchmark transfer harness (see trace_collect.tool_latency_transfer).
The whole frozen SWE-ReBench profile fits the offline-probe guard; a disjoint
target benchmark supplies the evaluation tasks. Provide the target either as a
trace-root of canonical trace.jsonl files (--eval-trace-root) or as a
pre-extracted tool-latency JSONL (--eval-latencies).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.tool_latency_transfer import run_transfer_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen trigger transfer onto a target benchmark."
    )
    parser.add_argument(
        "--confirmation-root",
        required=True,
        type=Path,
        help="Frozen confirmation output with provenance/manifest.json and "
        "data/all.jsonl (the full SWE-ReBench profile).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--eval-trace-root",
        type=Path,
        help="Directory of target trace.jsonl files to extract.",
    )
    source.add_argument(
        "--eval-latencies",
        type=Path,
        help="Pre-extracted target tool-latency JSONL.",
    )
    parser.add_argument(
        "--restore-cost-fractions",
        type=comma_separated_floats("restore cost fraction"),
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_transfer_evaluation(
        args.confirmation_root,
        output_root=args.output_root,
        restore_cost_fractions=args.restore_cost_fractions,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        eval_trace_root=args.eval_trace_root,
        eval_latencies=args.eval_latencies,
    )
    print(
        f"Transferred {len(result['restore_cost_fractions'])} fractions onto "
        f"{result['eval_task_count']} target tasks -> {args.output_root}"
    )


if __name__ == "__main__":
    main()
