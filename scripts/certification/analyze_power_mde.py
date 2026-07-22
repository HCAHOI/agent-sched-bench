"""Phase 0b: power / minimum-detectable-effect sizing for the fresh corpus.

Pilot-based sample-size analysis (standard: use the dev corpus to SIZE a future
fresh study; not method tuning, not leakage). The one-shot fresh-corpus run is
what converts every sensitivity result to certified, so it must not be
collected underpowered -- an ambiguous null would burn the only untouched data.

Uses the coverage-valid permutation certificate (Phase 0's fix) and the observed
per-task paired-delta distribution of the primary contrast (certified-union vs
deadline, at the operating-point restore proxy rho=1.0) as the empirical
data-generating process. Two deliverables:

(1) Power vs n -- model a fresh corpus of n tasks as n draws (with replacement)
    of whole task delta-vectors from the observed 100 (preserving the real
    heavy tails and cross-cost correlation AND the real effect size), run the
    permutation certificate on each simulated corpus, and report the
    re-certification rate. Tells us whether n=100 suffices and, if not, what n
    does.

(2) MDE at fixed n -- the smallest true mean effect a corpus of n tasks can
    certify at a target power. Hold the real residual (noise) shape fixed and
    vary only the mean, so the MDE reflects the heavy-tailed noise the fresh
    run will actually face. Reported per cost cell (in workload-total ms and
    normalized by kv), at the Bonferroni family bar alpha/(2m).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trace_collect.restore_cost_analysis import _load_certified_union_decisions
from trace_collect.tool_latency_confirmation import (
    _permutation_simultaneous_labels,
    paired_task_cluster_bootstrap,
)


def contribution_matrix(
    decisions: list[dict[str, Any]],
    costs: list[float],
    *,
    restore_cost_fraction: float,
) -> tuple[list[float], np.ndarray]:
    """Per-task x per-cost paired-delta matrix for certified-union vs deadline.

    Reuses the deployed bootstrap purely to obtain its per-task contribution
    vectors (the exact paired deltas at the operating restore cost), so the DGP
    is the same quantity every campaign certificate is computed from.
    """
    result = paired_task_cluster_bootstrap(
        decisions,
        costs_ms=costs,
        replicates=1,
        confidence_level=0.95,
        seed=0,
        baseline_trigger_field="deadline_trigger_ms",
        treatment_trigger_field="certified_union_trigger_ms",
        restore_cost_fraction=restore_cost_fraction,
        enforce_gated_treatment=False,
    )
    cost_list = [float(c) for c in result["costs_ms"]]
    cost_keys = [str(c) for c in cost_list]
    matrix = np.array(
        [
            [float(row["paired_delta_ms_by_cost"][k]) for k in cost_keys]
            for row in result["task_contributions"]
        ],
        dtype=float,
    )
    return cost_list, matrix


def _column_p_positive(
    column: np.ndarray, *, draws: int, rng: np.random.Generator
) -> float:
    """One-sided sign-flip randomization p-value for a single cost column."""
    observed = float(column.sum())
    n = column.shape[0]
    ge = 0
    batch = 8192
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, n))
        null_totals = signs @ column
        ge += int(np.sum(null_totals >= observed - 1e-9))
    return (1.0 + ge) / (draws + 1.0)


def power_curve(
    matrix: np.ndarray,
    cost_list: list[float],
    *,
    n_grid: list[int],
    corpora: int,
    inner_draws: int,
    confidence_level: float,
    seed: int,
    with_replacement: bool = True,
) -> list[dict[str, Any]]:
    """Re-certification power of the permutation certificate vs corpus size n.

    ``with_replacement`` draws n task-vectors from the pilot with replacement
    (the only option for n > pilot size; duplicates violate the sign-flip null's
    independence and bias power cell-dependently). Without replacement draws n
    DISTINCT pilot tasks -- closer to a genuinely fresh corpus, but only defined
    for n <= pilot size, which is exactly why n beyond the pilot cannot be
    reliably sized from the pilot alone.
    """
    task_count, cost_count = matrix.shape
    alpha = 1.0 - confidence_level
    family_tail = alpha / (2.0 * cost_count)
    rng = np.random.Generator(np.random.PCG64(seed))
    perm_rng = np.random.Generator(np.random.PCG64(seed + 1))
    out: list[dict[str, Any]] = []
    for n in n_grid:
        if not with_replacement and n > task_count:
            continue
        family_cert = 0
        per_cost_cert = np.zeros(cost_count, dtype=int)
        for _ in range(corpora):
            if with_replacement:
                idx = rng.integers(0, task_count, size=n)
            else:
                idx = rng.permutation(task_count)[:n]
            corpus = matrix[idx]
            observed = corpus.sum(axis=0)
            labels = _permutation_simultaneous_labels(
                corpus,
                observed,
                confidence_level=confidence_level,
                draws=inner_draws,
                seed=int(perm_rng.integers(0, 2**31 - 1)),
            )["points"]
            pos = [i for i, c in enumerate(labels) if c["permutation_label"] == "positive"]
            family_cert += 1 if pos else 0
            for i in pos:
                per_cost_cert[i] += 1
        out.append(
            {
                "n": n,
                "family_power": family_cert / corpora,
                "per_cost_power": {
                    str(cost_list[i]): float(per_cost_cert[i] / corpora)
                    for i in range(cost_count)
                },
                "min_achievable_p_value": max(
                    2.0 ** (-min(n, task_count)), 1.0 / (inner_draws + 1.0)
                ),
                "family_tail_probability": family_tail,
                "with_replacement": with_replacement,
            }
        )
    return out


def mde_for_cell(
    residual: np.ndarray,
    *,
    n: int,
    kv_cost_ms: float,
    family_tail: float,
    target_power: float,
    corpora: int,
    inner_draws: int,
    seed: int,
    max_mean_ms: float,
) -> dict[str, Any]:
    """Smallest per-task mean effect certifiable at target power, one cost cell.

    Holds the observed residual (mean-subtracted) shape fixed and bisects the
    added per-task mean until the permutation certificate certifies the cell (at
    the Bonferroni family tail) with >= target_power. Workload-total MDE = n*mean.
    """
    draw_rng = np.random.Generator(np.random.PCG64(seed))
    perm_rng = np.random.Generator(np.random.PCG64(seed + 1))

    def power_at(mean_ms: float) -> float:
        certified = 0
        for _ in range(corpora):
            idx = draw_rng.integers(0, residual.shape[0], size=n)
            column = residual[idx] + mean_ms
            p_pos = _column_p_positive(column, draws=inner_draws, rng=perm_rng)
            certified += 1 if p_pos <= family_tail else 0
        return certified / corpora

    # Bisect on the per-task mean. Upper bracket must reach target power.
    lo, hi = 0.0, max_mean_ms
    power_hi = power_at(hi)
    if power_hi < target_power:
        return {
            "n": n,
            "kv_cost_ms": kv_cost_ms,
            "target_power": target_power,
            "mde_workload_ms": None,
            "note": f"target power not reached by max per-task mean {max_mean_ms} ms",
            "power_at_max_mean": power_hi,
        }
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if power_at(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    mde_mean = hi
    return {
        "n": n,
        "kv_cost_ms": kv_cost_ms,
        "target_power": target_power,
        "mde_per_task_mean_ms": mde_mean,
        "mde_workload_ms": n * mde_mean,
        "mde_workload_normalized_by_kv": n * mde_mean / kv_cost_ms,
    }


def effect_concentration(matrix: np.ndarray, cost_list: list[float]) -> dict[str, Any]:
    """Share of each cell's positive effect carried by its top tasks.

    Heavy concentration means the permutation certificate's power (and any
    'some cell certified' family result) hinges on a few tasks being resampled
    -- a fragility the fresh-corpus sizing must account for.
    """
    out: dict[str, Any] = {}
    for i, cost in enumerate(cost_list):
        column = matrix[:, i]
        positive_mass = float(column[column > 0].sum())
        top = np.sort(column)[::-1]
        out[str(cost)] = {
            "positive_mass_ms": positive_mass,
            "net_effect_ms": float(column.sum()),
            "top1_share_of_positive": (
                float(top[0] / positive_mass) if positive_mass > 0 else None
            ),
            "top3_share_of_positive": (
                float(top[:3].sum() / positive_mass) if positive_mass > 0 else None
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "analysis/tool-time-certified-union-swe-rebench-20260715/"
            "rho_1.0_decisions.jsonl"
        ),
    )
    parser.add_argument("--restore-cost-fraction", type=float, default=1.0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--corpora", type=int, default=500)
    parser.add_argument("--inner-draws", type=int, default=2000)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-grid",
        type=int,
        nargs="+",
        default=[50, 75, 100, 150, 200, 300, 500],
    )
    parser.add_argument(
        "--mde-n",
        type=int,
        nargs="+",
        default=[100, 200],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decisions = _load_certified_union_decisions(args.decisions)
    costs = sorted({float(row["kv_cost_ms"]) for row in decisions})
    cost_list, matrix = contribution_matrix(
        decisions, costs, restore_cost_fraction=args.restore_cost_fraction
    )
    task_count, cost_count = matrix.shape
    alpha = 1.0 - args.confidence_level
    family_tail = alpha / (2.0 * cost_count)
    observed_effect = {
        str(cost_list[i]): float(matrix[:, i].sum()) for i in range(cost_count)
    }

    curve = power_curve(
        matrix,
        cost_list,
        n_grid=args.n_grid,
        corpora=args.corpora,
        inner_draws=args.inner_draws,
        confidence_level=args.confidence_level,
        seed=args.seed,
        with_replacement=True,
    )
    # Distinct-task control, defined only for n <= pilot size: quantifies the
    # with-replacement bias (duplicates violate the sign-flip null's row
    # independence, inflating power cell-dependently). n beyond the pilot has no
    # such control -- it cannot be reliably sized from the pilot.
    curve_distinct = power_curve(
        matrix,
        cost_list,
        n_grid=[n for n in args.n_grid if n <= task_count],
        corpora=args.corpora,
        inner_draws=args.inner_draws,
        confidence_level=args.confidence_level,
        seed=args.seed,
        with_replacement=False,
    )

    # MDE per cost cell at each requested n. Residual = mean-subtracted column;
    # bracket the search at 8x the observed per-task mean magnitude so a real
    # effect the size of what we saw is comfortably inside the bracket.
    residuals = matrix - matrix.mean(axis=0, keepdims=True)
    observed_per_task_mean = matrix.mean(axis=0)
    mde: dict[str, dict[str, Any]] = {}
    for n in args.mde_n:
        mde[str(n)] = {}
        for i, cost in enumerate(cost_list):
            max_mean = 8.0 * abs(float(observed_per_task_mean[i])) + 1.0
            mde[str(n)][str(cost)] = mde_for_cell(
                residuals[:, i],
                n=n,
                kv_cost_ms=cost,
                family_tail=family_tail,
                target_power=args.target_power,
                corpora=args.corpora,
                inner_draws=args.inner_draws,
                seed=args.seed + 100 + n + i,
                max_mean_ms=max_mean,
            )

    result = {
        "decisions_path": str(args.decisions),
        "restore_cost_fraction": args.restore_cost_fraction,
        "operating_point_note": (
            "rho=1.0 committed proxy for measured rho=0.94; low-rho probe files "
            "not read"
        ),
        "certificate": "paired_signflip_randomization",
        "pilot_task_count": task_count,
        "cost_count": cost_count,
        "family_tail_probability": family_tail,
        "corpora": args.corpora,
        "inner_draws": args.inner_draws,
        "observed_effect_workload_ms": observed_effect,
        "effect_concentration": effect_concentration(matrix, cost_list),
        "power_curve": curve,
        "power_curve_distinct_tasks": curve_distinct,
        "mde_by_n": mde,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "power_curve": curve,
                "power_curve_distinct_tasks": curve_distinct,
                "mde_by_n": mde,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
