"""Phase 0: harden the certification gate on the frozen corpus.

Three leakage-free re-analyses of committed certified-union decisions, all at
the operating-point restore proxy (rho=1.0 proxy for the measured rho=0.94; the
low-rho files are the closed sensitivity probe and are NOT read here):

(a) Gate calibration -- the false-certification rate of the gate's own
    procedure (task-cluster Bonferroni-percentile bootstrap -> simultaneous
    label) under a paired sign-flip null. This measures whether "certified" is
    a property with near-nominal type-I error, not just a label. Sign-flipping
    each task's whole paired-delta vector is the exact paired randomization
    null: under H0 (policy == deadline) the sign of a task's advantage is
    exchangeable. The bootstrap CI is recomputed on each null exactly as the
    real gate computes it, so this tests the estimator as deployed, on the real
    heavy-tailed per-task distribution.

(b) Repo-clustered CI -- SWE-ReBench task_ids share GitHub repos
    (owner__repo-<instance>). Task-clustered CIs assume task exchangeability; if
    same-repo tasks are correlated the CI is anticonservative. Re-resample at
    the repo level and compare CI width + certified labels against the
    task-level certificate.

(c) Censoring audit -- tool timeouts cap the upper tail. A cap pileup biases
    every trigger aggressive (a censored-long call looks like a completed call
    at the cap). Scan the latency distribution for mass at candidate caps.

Reuses the deployed certification numerics (paired_task_cluster_bootstrap and
its PCG64 task resampler) rather than reimplementing them, so (a)/(b) test the
same code path that produces headline certificates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from trace_collect.restore_cost_analysis import _load_certified_union_decisions
from trace_collect.tool_latency_confirmation import (
    _resample_task_totals,
    paired_task_cluster_bootstrap,
)

# owner__repo-<instance>; the trailing -<int> is the benchmark instance number.
_INSTANCE_SUFFIX = re.compile(r"-\d+$")


def _repo_of(task_id: str) -> str:
    stripped = _INSTANCE_SUFFIX.sub("", task_id)
    if not stripped or stripped == task_id:
        raise ValueError(f"task_id {task_id!r} does not match owner__repo-<instance>")
    return stripped


def _labels_from_totals(
    bootstrap_totals: np.ndarray,
    observed: np.ndarray,
    *,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Bonferroni-percentile simultaneous labels, matching the deployed gate."""
    cost_count = bootstrap_totals.shape[1]
    alpha = 1.0 - confidence_level
    family_tail = alpha / (2.0 * cost_count)
    lows = np.quantile(bootstrap_totals, family_tail, axis=0, method="linear")
    highs = np.quantile(bootstrap_totals, 1.0 - family_tail, axis=0, method="linear")
    out: list[dict[str, Any]] = []
    for col in range(cost_count):
        low, high = float(lows[col]), float(highs[col])
        if low > 0.0:
            label = "positive"
        elif high < 0.0:
            label = "harmful"
        else:
            label = "inconclusive"
        out.append(
            {
                "observed": float(observed[col]),
                "low": low,
                "high": high,
                "width": high - low,
                "label": label,
            }
        )
    return out


def _contribution_matrix(
    bootstrap_result: dict[str, Any],
) -> tuple[list[str], list[float], np.ndarray]:
    """Per-task x per-cost paired-delta matrix from a bootstrap result."""
    costs = [float(c) for c in bootstrap_result["costs_ms"]]
    cost_keys = [str(c) for c in costs]
    task_rows = bootstrap_result["task_contributions"]
    task_ids = [str(row["task_id"]) for row in task_rows]
    matrix = np.array(
        [
            [float(row["paired_delta_ms_by_cost"][k]) for k in cost_keys]
            for row in task_rows
        ],
        dtype=float,
    )
    return task_ids, costs, matrix


def calibration_under_signflip(
    contributions: np.ndarray,
    *,
    confidence_level: float,
    inner_replicates: int,
    null_draws: int,
    bootstrap_seed: int,
    signflip_seed: int,
) -> dict[str, Any]:
    """False-certification rate of the gate under a paired sign-flip null.

    For each null draw, flip each task's whole delta vector by an independent
    +/-1, then run the deployed task-cluster bootstrap and Bonferroni labels.
    Report how often the null wrongly certifies a positive (one-sided) or any
    non-inconclusive (two-sided) cost cell -- the gate's empirical type-I.
    """
    task_count, cost_count = contributions.shape
    alpha = 1.0 - confidence_level
    signflip_rng = np.random.Generator(np.random.PCG64(signflip_seed))
    any_positive = 0
    any_harmful = 0
    any_nonnull = 0
    per_cost_positive = np.zeros(cost_count, dtype=int)
    for draw in range(null_draws):
        signs = signflip_rng.choice(np.array([-1.0, 1.0]), size=task_count)
        null_matrix = signs[:, None] * contributions
        observed = null_matrix.sum(axis=0)
        # Distinct bootstrap seed per draw so inner resamples are independent
        # across nulls but reproducible; PCG64 as in the deployed resampler.
        totals = _resample_task_totals(
            null_matrix,
            replicates=inner_replicates,
            seed=bootstrap_seed + draw,
        )
        labels = _labels_from_totals(
            totals, observed, confidence_level=confidence_level
        )
        cell_labels = [cell["label"] for cell in labels]
        pos = [i for i, lab in enumerate(cell_labels) if lab == "positive"]
        harm = any(lab == "harmful" for lab in cell_labels)
        any_positive += 1 if pos else 0
        any_harmful += 1 if harm else 0
        any_nonnull += 1 if (pos or harm) else 0
        for i in pos:
            per_cost_positive[i] += 1
    return {
        "null_draws": null_draws,
        "inner_replicates": inner_replicates,
        "confidence_level": confidence_level,
        "nominal_family_alpha": alpha,
        "nominal_one_sided_family_alpha": alpha / 2.0,
        "empirical_family_positive_rate": any_positive / null_draws,
        "empirical_family_harmful_rate": any_harmful / null_draws,
        "empirical_family_any_cert_rate": any_nonnull / null_draws,
        "per_cost_positive_rate": (per_cost_positive / null_draws).tolist(),
    }


def certificate(
    contributions: np.ndarray,
    task_ids: list[str],
    *,
    cluster: str,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Simultaneous certificate resampling either tasks or repos as clusters."""
    if cluster == "task":
        matrix = contributions
        unit_count = contributions.shape[0]
    elif cluster == "repo":
        repo_index: dict[str, int] = {}
        rows: list[np.ndarray] = []
        repo_sizes: dict[str, int] = {}
        for task_row, task_id in zip(contributions, task_ids):
            repo = _repo_of(task_id)
            repo_sizes[repo] = repo_sizes.get(repo, 0) + 1
            if repo not in repo_index:
                repo_index[repo] = len(rows)
                rows.append(np.zeros(contributions.shape[1], dtype=float))
            rows[repo_index[repo]] += task_row
        matrix = np.array(rows, dtype=float)
        unit_count = matrix.shape[0]
    else:
        raise ValueError(f"unknown cluster {cluster!r}")
    observed = matrix.sum(axis=0)
    totals = _resample_task_totals(matrix, replicates=replicates, seed=seed)
    labels = _labels_from_totals(totals, observed, confidence_level=confidence_level)
    result: dict[str, Any] = {
        "cluster": cluster,
        "cluster_count": unit_count,
        "cells": labels,
    }
    if cluster == "repo":
        multi = sum(1 for n in repo_sizes.values() if n > 1)
        result["repo_structure"] = {
            "repo_count": unit_count,
            "task_count": len(task_ids),
            "repos_with_multiple_tasks": multi,
            "max_tasks_per_repo": max(repo_sizes.values()),
            # Near-singleton clusters => repo resampling ~= task resampling;
            # this test has little power to detect within-repo correlation,
            # because there is almost no within-repo replication to carry it.
            "singleton_repo_count": unit_count - multi,
        }
    return result


def censoring_audit(
    decisions: list[dict[str, Any]],
    *,
    cap_overshoot_fraction: float = 0.02,
) -> dict[str, Any]:
    """Scan the per-call latency distribution for timeout-cap pileups.

    A censored (timed-out) call lands just ABOVE the nominal timeout by the
    teardown overhead, so caps must be counted with a one-sided UPWARD window
    ``[cap, cap*(1+overshoot)]``, not a symmetric tolerance. We also expose the
    raw upper tail (top latencies + the largest tail gap) so a structural
    pileup is visible even at a cap value we did not enumerate. Constant-cap
    detection is a LOWER bound on censoring: this branch's exec timeout is
    resource-integrated/stall-based (see src/trace_collect/CLAUDE.md), which
    censors at a varying wall-clock value and leaves no constant pileup.
    """
    # latency_ms is per sample (identical across the cost panel); dedup by sample.
    latency_by_sample: dict[str, float] = {}
    for row in decisions:
        latency_by_sample[str(row["sample_id"])] = float(row["latency_ms"])
    latencies = np.array(sorted(latency_by_sample.values()), dtype=float)
    max_latency = float(latencies[-1])
    candidate_caps_ms = [30_000, 60_000, 120_000, 300_000, 600_000]
    cap_hits = {
        str(cap): int(((latencies >= cap) & (latencies <= cap * (1.0 + cap_overshoot_fraction))).sum())
        for cap in candidate_caps_ms
    }
    censored_suspect = sum(cap_hits.values())
    # Largest multiplicative gap in the top tail flags a cap cluster we did not
    # enumerate: a big jump then a tight group above it is the timeout plateau.
    tail = latencies[latencies > np.percentile(latencies, 99.0)]
    gap_ratios = tail[1:] / tail[:-1]
    largest_gap_index = int(np.argmax(gap_ratios)) if gap_ratios.size else -1
    return {
        "sample_count": len(latencies),
        "max_latency_ms": max_latency,
        "quantiles_ms": {
            q: float(np.percentile(latencies, q)) for q in (50, 90, 99, 99.9, 100)
        },
        "top_latencies_ms": latencies[-12:].tolist(),
        "candidate_cap_hit_counts_upward_window": cap_hits,
        "cap_overshoot_fraction": cap_overshoot_fraction,
        "censored_suspect_count": censored_suspect,
        "censored_suspect_fraction": censored_suspect / len(latencies),
        "largest_top_tail_gap_ratio": (
            float(gap_ratios[largest_gap_index]) if largest_gap_index >= 0 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "analysis/tool-time-certified-union-swe-rebench-20260715/"
            "rho_1.0_decisions.jsonl"
        ),
        help="Certified-union decisions at the operating-point restore proxy.",
    )
    parser.add_argument("--restore-cost-fraction", type=float, default=1.0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-draws", type=int, default=2000)
    parser.add_argument(
        "--null-inner-replicates",
        type=int,
        default=None,
        help="Bootstrap replicates per null draw; defaults to --replicates so "
        "the null measures the certificate at deployed Monte-Carlo resolution "
        "(a coarser null inflates the measured type-I).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    null_inner_replicates = (
        args.null_inner_replicates
        if args.null_inner_replicates is not None
        else args.replicates
    )

    decisions = _load_certified_union_decisions(args.decisions)
    costs = sorted({float(row["kv_cost_ms"]) for row in decisions})

    # Deployed certificate: certified union vs deadline at the operating proxy.
    real = paired_task_cluster_bootstrap(
        decisions,
        costs_ms=costs,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        baseline_trigger_field="deadline_trigger_ms",
        treatment_trigger_field="certified_union_trigger_ms",
        restore_cost_fraction=args.restore_cost_fraction,
        enforce_gated_treatment=False,
    )
    task_ids, cost_list, contributions = _contribution_matrix(real)

    # Anchor: reproduce the library's simultaneous labels via our resampler.
    anchor = certificate(
        contributions,
        task_ids,
        cluster="task",
        confidence_level=args.confidence_level,
        replicates=args.replicates,
        seed=args.seed,
    )
    lib_labels = [real["points"][str(c)]["simultaneous_label"] for c in cost_list]
    our_labels = [cell["label"] for cell in anchor["cells"]]
    if lib_labels != our_labels:
        raise AssertionError(
            f"anchor mismatch: library {lib_labels} != reproduction {our_labels}"
        )

    repo_cert = certificate(
        contributions,
        task_ids,
        cluster="repo",
        confidence_level=args.confidence_level,
        replicates=args.replicates,
        seed=args.seed,
    )
    calibration = calibration_under_signflip(
        contributions,
        confidence_level=args.confidence_level,
        inner_replicates=null_inner_replicates,
        null_draws=args.null_draws,
        bootstrap_seed=args.seed + 1,
        signflip_seed=args.seed + 7,
    )
    censoring = censoring_audit(decisions)

    result = {
        "decisions_path": str(args.decisions),
        "restore_cost_fraction": args.restore_cost_fraction,
        "operating_point_note": (
            "rho=1.0 committed proxy for the measured rho=0.94; low-rho files "
            "are the closed sensitivity probe and are not read here"
        ),
        "task_count": len(task_ids),
        "repo_count": repo_cert["cluster_count"],
        "costs_ms": cost_list,
        "deployed_certificate_task_cluster": {
            str(c): {
                "label": lib_labels[i],
                "paired_delta_ms": real["points"][str(c)]["paired_delta_ms"],
                "simultaneous_interval_ms": real["points"][str(c)][
                    "simultaneous_interval_ms"
                ],
            }
            for i, c in enumerate(cost_list)
        },
        "repo_structure": repo_cert["repo_structure"],
        "repo_cluster_certificate": {
            str(cost_list[i]): repo_cert["cells"][i] for i in range(len(cost_list))
        },
        "calibration_signflip_null": calibration,
        "censoring_audit": censoring,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
