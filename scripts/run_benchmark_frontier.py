#!/usr/bin/env python3
"""Replicate the gated trie vs gated hazard frontier on a second trace corpus.

Benchmark-local frontier harness (see trace_collect.benchmark_frontier). The
shared fold and command-parsing config is sourced from a frozen manifest json
(--config-manifest); only the estimator knobs (num_intervals, model_family,
feature_set, ensemble_members) and the bootstrap are set here. Provide the
target corpus either as a trace-root of canonical trace.jsonl files
(--trace-root) or as a pre-extracted tool-latency JSONL (--eval-latencies), and
state its exposure status explicitly with --exposure-note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.benchmark_frontier import run_benchmark_frontier
from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import load_config_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replicate the within-benchmark frontier on a target corpus."
    )
    parser.add_argument(
        "--config-manifest",
        required=True,
        type=Path,
        help="Frozen manifest json supplying fold_count, inner_folds, costs_ms, "
        "guard_ms, min_tool_history, min_profile_tasks, command_field, "
        "max_prefix_depth, and skip_leading_cd.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trace-root",
        type=Path,
        help="Directory of target trace.jsonl files to extract.",
    )
    source.add_argument(
        "--eval-latencies",
        type=Path,
        help="Pre-extracted target tool-latency JSONL.",
    )
    parser.add_argument(
        "--exposure-note",
        required=True,
        help="Explicit statement of the corpus's exposure status (recorded in "
        "the payload); e.g. whether it was seen during method development.",
    )
    parser.add_argument("--num-intervals", type=int, default=16)
    parser.add_argument(
        "--model-family", choices=("logistic", "gbm"), default="gbm"
    )
    parser.add_argument(
        "--feature-set",
        choices=("full", "with_within_task", "cross_task_only"),
        default="full",
    )
    parser.add_argument("--ensemble-members", type=int, default=0)
    parser.add_argument(
        "--tool-name-trie",
        action="store_true",
        help="Also fit a tool-identity-only trie (command_field=None, "
        "Continuum's P(tau, f)) on the same folds and add the tool-name-vs-full, "
        "tool-name-vs-GBM, and tool-name-vs-deadline contrasts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--restore-cost-fractions",
        type=comma_separated_floats("restore cost fraction"),
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--output-root", required=True, type=Path)
    shard = parser.add_mutually_exclusive_group()
    shard.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="Run ONLY this outer fold's fit/eval (write its folds/f{N}_* and "
        "rho_*/f{N}_* files) and exit before any cross-fold aggregation. Lets K "
        "folds run as K concurrent processes into one output root.",
    )
    shard.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip the fit phase; read the folds/f*_* and rho_*/f*_* files the "
        "fold processes wrote, then run the cross-fold aggregation + bootstrap "
        "and write the top-level frontier results.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_config_manifest(args.config_manifest)
    result = run_benchmark_frontier(
        args.trace_root,
        output_root=args.output_root,
        fold_count=manifest["fold_count"],
        inner_folds=manifest["inner_folds"],
        kv_costs_ms=[float(cost) for cost in manifest["costs_ms"]],
        guard_ms=manifest["guard_ms"],
        min_tool_history=manifest["min_tool_history"],
        min_profile_tasks=manifest["min_profile_tasks"],
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
        num_intervals=args.num_intervals,
        model_family=args.model_family,
        seed=args.seed,
        restore_cost_fractions=args.restore_cost_fractions,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        exposure_note=args.exposure_note,
        eval_latencies=args.eval_latencies,
        feature_set=args.feature_set,
        ensemble_members=args.ensemble_members,
        tool_name_trie=args.tool_name_trie,
        only_fold=args.only_fold,
        aggregate_only=args.aggregate_only,
    )
    if args.only_fold is not None:
        return
    print(
        f"Frontier over {result['task_count']} tasks x "
        f"{len(result['restore_cost_fractions'])} fractions -> {args.output_root}"
    )


if __name__ == "__main__":
    main()
