#!/usr/bin/env python3
"""Paired sign-flip permutation certificate for a frontier decision file.

Compares a treatment trigger against a baseline trigger over the KV-cost family
using ``paired_task_cluster_bootstrap`` and writes the per-cost permutation
labels/p-values plus the simultaneous percentile band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--treatment-field", required=True)
    parser.add_argument("--baseline-field", required=True)
    parser.add_argument("--restore-cost-fraction", required=True, type=float)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [
        json.loads(line)
        for line in args.decisions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no decision rows in {args.decisions}")
    costs = sorted({float(row["kv_cost_ms"]) for row in rows})
    result = paired_task_cluster_bootstrap(
        rows,
        costs_ms=costs,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        baseline_trigger_field=args.baseline_field,
        treatment_trigger_field=args.treatment_field,
        restore_cost_fraction=args.restore_cost_fraction,
        enforce_gated_treatment=False,
        permutation_draws=args.replicates,
    )
    per_cost = {
        cost: {
            "permutation_label": point["permutation_label"],
            "permutation_p_positive": point["permutation_p_positive"],
            "permutation_p_harmful": point["permutation_p_harmful"],
            "paired_delta_ms": point["paired_delta_ms"],
            "simultaneous_interval_ms": point["simultaneous_interval_ms"],
            "simultaneous_label": point["simultaneous_label"],
        }
        for cost, point in result["points"].items()
    }
    payload = {
        "decisions": str(args.decisions),
        "treatment_field": args.treatment_field,
        "baseline_field": args.baseline_field,
        "restore_cost_fraction": args.restore_cost_fraction,
        "replicates": args.replicates,
        "confidence_level": args.confidence_level,
        "seed": args.seed,
        "costs_ms": costs,
        "per_cost": per_cost,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote frontier permutation certificate -> {args.output}")


if __name__ == "__main__":
    main()
