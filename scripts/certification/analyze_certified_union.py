#!/usr/bin/env python3
"""Bootstrap the certified OR-union of the trie and hazard gates.

Derived re-scoring of an existing hazard-confirmation run: the union fires
when either *certified* gate opens early, at whichever fitted trigger comes
first. A gate is certified per outer fold using ONLY the other folds' rows
(cross-fitted leave-fold-out), so a fold's own rows never certify the rule
they are scored under. See
trace_collect.restore_cost_analysis.run_certified_union_analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import run_certified_union_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap the certified OR-union of the trie and hazard gates."
    )
    parser.add_argument(
        "--hazard-root",
        required=True,
        type=Path,
        help="Hazard confirmation output with rho_*_decisions.jsonl",
    )
    parser.add_argument(
        "--restore-cost-fractions",
        type=comma_separated_floats("restore cost fraction"),
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--inclusion-criterion",
        choices=("loo_point", "loo_lcb"),
        default="loo_point",
        help=(
            "loo_point (default): include a gate iff its leave-fold-out total "
            "delta is positive; loo_lcb: require a task-clustered bootstrap "
            "lower bound of that delta to be positive."
        ),
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_certified_union_analysis(
        args.hazard_root,
        output_root=args.output_root,
        restore_cost_fractions=args.restore_cost_fractions,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        inclusion_criterion=args.inclusion_criterion,
    )
    print(
        f"Bootstrapped certified union over {result['decision_row_count']} "
        f"decisions at {len(result['restore_cost_fractions'])} fractions "
        f"({result['inclusion_criterion']}) -> {args.output_root}"
    )


if __name__ == "__main__":
    main()
