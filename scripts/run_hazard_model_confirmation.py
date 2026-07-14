#!/usr/bin/env python3
"""Score the learned discrete-time hazard clock against the refit policies.

The hazard model predicts each call's latency distribution and drives the same
utility-clock trigger/gate seam as the empirical trie, so its deployed
``offline_gated_hazard_trigger_ms`` can be compared head-to-head against the
deadline, the gated robust trie, and the gated within-task baseline. Per frozen
fold and per restore fraction the model is fit and scored at that fraction, then
merged into the Mode B refit and gated-B1 decisions so every contrast shares
fitting and scoring restore costs. See
trace_collect.restore_cost_analysis.run_hazard_model_confirmation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.cli_helpers import comma_separated_floats
from trace_collect.restore_cost_analysis import run_hazard_model_confirmation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirm the hazard-model clock under restore costs."
    )
    parser.add_argument(
        "--confirmation-root",
        required=True,
        type=Path,
        help="Frozen confirmation output with provenance/manifest.json and "
        "data/f*_{eval,profile}.jsonl",
    )
    parser.add_argument(
        "--mode-b-root",
        required=True,
        type=Path,
        help="Mode B refit output with rho_*/f*_decisions.jsonl "
        "(deadline and gated-robust triggers)",
    )
    parser.add_argument(
        "--gated-b1-root",
        required=True,
        type=Path,
        help="Within-task baseline output with rho_*_decisions.jsonl "
        "(gated_within_task_trigger_ms)",
    )
    parser.add_argument(
        "--restore-cost-fractions",
        type=comma_separated_floats("restore cost fraction"),
        default=[0.0, 0.25, 0.5, 1.0],
    )
    # Grid resolution swept in the plan at {20, 40, 80}; 40 balances tail
    # resolution against per-interval support on the ~9k-call corpus.
    parser.add_argument("--num-intervals", type=int, default=40)
    parser.add_argument(
        "--feature-set",
        choices=("full", "with_within_task", "cross_task_only"),
        default="full",
        help="Ablation arm; command parsing config always comes from the manifest",
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_hazard_model_confirmation(
        args.confirmation_root,
        mode_b_root=args.mode_b_root,
        gated_b1_root=args.gated_b1_root,
        output_root=args.output_root,
        restore_cost_fractions=args.restore_cost_fractions,
        num_intervals=args.num_intervals,
        feature_set=args.feature_set,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    print(
        f"Scored hazard-model confirmation over {result['decision_row_count']} "
        f"decisions at {len(result['restore_cost_fractions'])} fractions "
        f"-> {args.output_root}"
    )


if __name__ == "__main__":
    main()
