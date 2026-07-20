#!/usr/bin/env python3
"""A2 -- pre-restore offline accounting over certified latency priors.

Pre-registered in ``analysis/rolling-survival-design-20260720.md`` ("A2 --
Pre-restore offline accounting", the first real experiment after A0 confirmed
the k>1 re-check collapse). Offline accounting ONLY: it replays the frozen
fresh-277 corpus through the existing cost functional; GPU live validation
remains W10-11 and is labeled as such.

The mechanism priced. Today, when a swapped-out call completes, the agent pays
the full restore cost ``R = rho * kv`` on the critical path before resuming.
Pre-restore starts the swap-back-in at elapsed ``s`` while the call is still
running, so the restore overlaps the call's tail. The swap trigger is held
FIXED (same as the certified policy); the ONLY thing that changes between
baseline and treatment is how the restore is paid, so the swap-trigger costs
(hidden/exposed/short-fire restore) are identical in both arms and cancel. The
paired per-call quantity is the pre-restore differential alone.

Accounting identity (stated explicitly; the docstring is the contract the test
pins). For a call with observed latency ``L``, a pre-restore start ``s`` that is
never earlier than the swap trigger ``g`` (nothing to restore before the swap),
and restore work ``R``:

* ``L <= s``  -> ``(hidden=0, wasted=0)``. The call finished before pre-restore
  initiated (this also covers a call never swapped, ``L <= g <= s``): the
  restore is paid at ``L`` on the critical path exactly as in the baseline, so
  the differential is zero -- a call that was never swapped pays and gains
  nothing.
* ``s < L <= s + R`` -> ``(hidden = L - s, wasted = 0)``. The restore ran during
  ``[s, L]`` and the resume consumes it; the lead time hidden off the critical
  path is ``L - s == min(L - s, R)`` (``L - s <= R`` on this branch).
* ``L > s + R`` -> ``(hidden = 0, wasted = R)``. The call OUTLIVED the restore
  window: the swap-in completed at ``s + R`` with the call still running, so the
  KV was brought back too early, must be re-swapped, and the real restore is
  redone at ``L`` on the critical path. The speculative swap-in bought nothing
  and is charged in full. (This is an all-or-nothing re-swap model, hence the
  ``+R -> -R`` cliff at ``L = s + R``; it is the honest conservative reading of
  "the KV must be re-swapped or the restore redone" and introduces no cost
  outside the existing ``R = rho * kv`` structure.)

Per-call utility ``= hidden - wasted``, i.e. lead-time hidden minus wasted
restore, charged at par -- the same convention the swap-trigger functional uses
(``hidden_on_long - exposed - restore``; ``tool_latency_utility_clock``).

Optimizer (amendment 5: an EXACT piecewise stopping-time optimizer analogous to
``hazard_recheck_ms``, NOT runtime curve queries). ``E[u(s)]`` over a node's fit
sample set is piecewise linear in ``s`` with the only jumps at ``{L - R}`` per
sample ``L`` (each term is ``-R`` below ``L - R``, jumps to ``+R`` at ``L - R``,
then decays linearly to ``0`` at ``L``). A linear piece is maximised at an
endpoint and every jump point is a candidate, so maximising over
``{g} u {L - R > g} u {L > g}`` is exact -- a precomputed elapsed-only scalar,
no runtime curve, no tuning parameter. The no-pre-restore option (utility 0) is
the floor: pre-restore is adopted only when some ``s`` is STRICTLY positive, so
degenerate/thin nodes (``< 2`` logical tasks, mirroring the robust clock's
existing guard) and ``R = 0`` inherit conservative no-pre-restore behaviour.

Cross-fit. The optimal start ``s*`` and the swap trigger ``g`` are computed from
FIT-fold node samples only (the deepest prior node ``latency_prior_hierarchy(
...)[-1]`` the certified policy selects, ``hazard_recheck_ms`` for ``g`` exactly
as A0 treats the certified k=1 trigger); the held-out eval call's latency is
scored against them. Every kv cell is evaluated at the certified operating point
(guard 0 so threshold == kv, rho = 0.94).

Statistics. Per (task, kv) net-utility contributions feed the SAME certified
engine as the H1 / WTN Stage-2 certificate -- ``_resample_task_totals`` (paired
task-cluster percentile bootstrap) and ``_permutation_simultaneous_labels``
(task-clustered sign-flip randomization, Bonferroni over the full cost family)
at replicates 50000, confidence 0.95, seed 0. We feed a precomputed
contributions matrix because the engine's ``paired_task_cluster_bootstrap``
wrapper hardwires the swap-trigger utility; the resampler and permutation core
are reused verbatim, no reimplementation.

Pre-registered kill (FROZEN, amendment 4): at rho=0.94, net seconds per 277
tasks (lead-time hidden minus wasted-restore charged) POSITIVE with a
task-clustered CI excluding zero in >= 1 headline kv cell (3500/5000), Bonferroni
over the full cost panel. SURVIVE (pre-restore ships) iff met; otherwise
pre-restore ships as an honest negative table.

EXPLORATORY until ``--final``. Emits JSON + MD to ``analysis/`` (``-PARTIAL``
unless ``--final``).

Usage (full corpus -- run by the main session, not the smoke):
  uv run python scripts/analyze_prerestore_accounting.py \
    --manifest analysis/fresh-corpus-certification-20260717/\
offline-gated-robust/manifest.json --final
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

# Allow direct `python scripts/analyze_prerestore_accounting.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.adjudicate_k2_recheck import (  # noqa: E402
    _CERT_GUARD_MS,
    _CERT_RESTORE_COST_FRACTION,
    _banner,
    _git_sha,
    _load_manifest_corpus,
    _row_group_keys,
)
from trace_collect.tool_latency_dataset import ToolLatencySample  # noqa: E402
from trace_collect.tool_latency_profiled import (  # noqa: E402
    LatencyPriorNode,
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
)

# Reuse the certified task-cluster bootstrap + sign-flip permutation core
# verbatim (same replicate/seed/Bonferroni discipline as H1). We supply a
# precomputed contributions matrix instead of routing through
# ``paired_task_cluster_bootstrap`` because that wrapper's per-call utility is
# hardwired to the swap-trigger functional; the statistical engine is identical.
from trace_collect.tool_latency_confirmation import (  # noqa: E402
    _permutation_simultaneous_labels,
    _resample_task_totals,
)

# Headline cells for the pre-registered kill (spec: 3500/5000). The FULL cost
# family still drives the Bonferroni correction, matching H1.
_HEADLINE_COSTS_MS = (3500.0, 5000.0)


# --------------------------------------------------------------------------- #
# Pre-restore per-call accounting and stopping-time optimizer.
# --------------------------------------------------------------------------- #
def prerestore_components(
    latency_ms: float, start_ms: float | None, *, restore_cost_ms: float
) -> tuple[float, float]:
    """Return ``(hidden_lead, wasted_restore)`` for one call (identity above).

    ``start_ms is None`` means no pre-restore (baseline): both terms zero.
    Utility is ``hidden_lead - wasted_restore``.
    """

    if start_ms is None or latency_ms <= start_ms:
        return 0.0, 0.0
    lead = latency_ms - start_ms
    if lead <= restore_cost_ms:
        return lead, 0.0  # == min(lead, R); lead <= R on this branch
    return 0.0, restore_cost_ms


def _expected_prerestore_utility(
    samples: np.ndarray, start_ms: float, restore_cost_ms: float
) -> float:
    """Mean per-call pre-restore utility of a start over a node sample set."""

    lead = samples - start_ms
    fires = lead > 0.0
    hidden = np.where(fires & (lead <= restore_cost_ms), lead, 0.0)
    wasted = np.where(fires & (lead > restore_cost_ms), restore_cost_ms, 0.0)
    return float(np.mean(hidden - wasted))


def prerestore_start_ms(
    node: LatencyPriorNode, *, swap_trigger_ms: float, restore_cost_ms: float
) -> float | None:
    """Exact-optimal pre-restore start ``s* >= swap_trigger_ms``, or ``None``.

    ``None`` = no pre-restore (the conservative floor): returned when there is
    nothing to overlap (``R <= 0``), the node is thin/degenerate (``< 2`` logical
    tasks, mirroring ``robust_utility_trigger_stats``), or no start beats the
    zero-utility no-fire baseline. The candidate grid ``{g} u {L - R > g} u
    {L > g}`` is exact for the piecewise-linear objective (module docstring);
    ties resolve to the LATEST (smallest-window, least-speculative) start.
    """

    if restore_cost_ms <= 0.0:
        return None
    if len(node.values_by_task) < 2:
        return None
    values = node.values
    if not values:
        return None
    samples = np.asarray(values, dtype=float)
    candidates = {float(swap_trigger_ms)}
    for latency in values:
        edge = latency - restore_cost_ms
        if edge > swap_trigger_ms:
            candidates.add(float(edge))
        if latency > swap_trigger_ms:
            candidates.add(float(latency))
    best_start: float | None = None
    best_utility = 0.0  # no-pre-restore floor
    for start in sorted(candidates):
        utility = _expected_prerestore_utility(samples, start, restore_cost_ms)
        if utility > 0.0 and utility >= best_utility:
            best_utility = utility
            best_start = start
    return best_start


# --------------------------------------------------------------------------- #
# Per-eval-call scoring over the certified folds.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrerestoreConfig:
    fold_count: int
    command_field: str
    max_prefix_depth: int
    skip_leading_cd: bool
    min_tool_history: int
    min_profile_tasks: int
    costs_ms: tuple[float, ...]
    guard_ms: float
    restore_cost_fraction: float
    replicates: int
    confidence_level: float
    seed: int


def _score_eval_call(
    node: LatencyPriorNode,
    latency_ms: float,
    *,
    kv_cost_ms: float,
    guard_ms: float,
    restore_cost_fraction: float,
    trigger_cache: dict[tuple[int, float], tuple[float, float | None]],
) -> dict[str, Any]:
    """Score one held-out call at one kv cell against its fit-fold node."""

    threshold_ms = kv_cost_ms + guard_ms
    restore_cost_ms = restore_cost_fraction * kv_cost_ms
    # id(node.values) keys distinct fit nodes: safe because the cache is per-fold
    # and the prior's node lists stay alive for the fold (same pattern as A0).
    cache_key = (id(node.values), kv_cost_ms)
    cached = trigger_cache.get(cache_key)
    if cached is None:
        swap_trigger_ms = hazard_recheck_ms(
            node.values,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
            restore_cost_ms=restore_cost_ms,
        )
        start_ms = prerestore_start_ms(
            node, swap_trigger_ms=swap_trigger_ms, restore_cost_ms=restore_cost_ms
        )
        trigger_cache[cache_key] = (swap_trigger_ms, start_ms)
    else:
        swap_trigger_ms, start_ms = cached
    hidden_ms, wasted_ms = prerestore_components(
        latency_ms, start_ms, restore_cost_ms=restore_cost_ms
    )
    fired = start_ms is not None and latency_ms > start_ms
    return {
        "kv_cost_ms": kv_cost_ms,
        "threshold_ms": threshold_ms,
        "restore_cost_ms": restore_cost_ms,
        "swap_trigger_ms": swap_trigger_ms,
        "prerestore_start_ms": start_ms,
        "hidden_lead_ms": hidden_ms,
        "wasted_restore_ms": wasted_ms,
        "utility_ms": hidden_ms - wasted_ms,
        "fired": fired,
    }


def score_decisions(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: PrerestoreConfig,
) -> list[dict[str, Any]]:
    """Cross-fitted per-(call, cost) pre-restore decision rows over the folds."""

    row_group_keys = _row_group_keys(
        cfg.command_field,
        max_prefix_depth=cfg.max_prefix_depth,
        skip_leading_cd=cfg.skip_leading_cd,
    )
    declared = list(task_ids)
    decisions: list[dict[str, Any]] = []
    for fold in range(1, cfg.fold_count + 1):
        eval_tasks = {
            task_id
            for index, task_id in enumerate(declared)
            if index % cfg.fold_count == fold - 1
        }
        profile_tasks = set(declared) - eval_tasks
        profile_rows = [
            sample.to_json_obj()
            for task_id in sorted(profile_tasks)
            for sample in samples_by_task[task_id]
        ]
        prior = build_latency_prior(profile_rows, row_group_keys=row_group_keys)
        trigger_cache: dict[tuple[int, float], tuple[float, float | None]] = {}
        for task_id in sorted(eval_tasks):
            for sample in samples_by_task[task_id]:
                row = sample.to_json_obj()
                node = latency_prior_hierarchy(
                    prior,
                    str(row["tool_name"]),
                    row_group_keys(row),
                    min_tool_history=cfg.min_tool_history,
                    min_profile_tasks=cfg.min_profile_tasks,
                )[-1]
                latency_ms = float(row["latency_ms"])
                for kv_cost_ms in cfg.costs_ms:
                    scored = _score_eval_call(
                        node,
                        latency_ms,
                        kv_cost_ms=kv_cost_ms,
                        guard_ms=cfg.guard_ms,
                        restore_cost_fraction=cfg.restore_cost_fraction,
                        trigger_cache=trigger_cache,
                    )
                    decisions.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "task_id": task_id,
                            "tool_name": str(row["tool_name"]),
                            "outer_fold": f"f{fold}",
                            "latency_ms": latency_ms,
                            "prior_source": node.source,
                            "prior_group_key": node.group_key,
                            **scored,
                        }
                    )
    return decisions


# --------------------------------------------------------------------------- #
# Aggregation, certificate, verdict.
# --------------------------------------------------------------------------- #
def _contributions_matrix(
    decisions: Sequence[dict[str, Any]], costs_ms: Sequence[float]
) -> tuple[np.ndarray, list[str]]:
    """Per (task, kv) summed pre-restore utility (ms). Rows are logical tasks."""

    task_ids = sorted({str(row["task_id"]) for row in decisions})
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    cost_index = {cost: index for index, cost in enumerate(costs_ms)}
    contributions = np.zeros((len(task_ids), len(costs_ms)), dtype=float)
    for row in decisions:
        contributions[
            task_index[str(row["task_id"])], cost_index[float(row["kv_cost_ms"])]
        ] += float(row["utility_ms"])
    return contributions, task_ids


def _certificate(
    contributions: np.ndarray, cfg: PrerestoreConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse the H1 engine: percentile CI + sign-flip permutation labels."""

    observed = np.sum(contributions, axis=0)
    bootstrap_totals = _resample_task_totals(
        contributions, replicates=cfg.replicates, seed=cfg.seed
    )
    alpha = 1.0 - cfg.confidence_level
    family_tail = alpha / (2.0 * len(cfg.costs_ms))
    point_quantiles = np.quantile(
        bootstrap_totals, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0, method="linear"
    )
    simultaneous_quantiles = np.quantile(
        bootstrap_totals, [family_tail, 1.0 - family_tail], axis=0, method="linear"
    )
    permutation = _permutation_simultaneous_labels(
        contributions,
        observed,
        confidence_level=cfg.confidence_level,
        draws=cfg.replicates,
        seed=cfg.seed,
    )
    return observed, point_quantiles, simultaneous_quantiles, permutation


def summarize(
    decisions: Sequence[dict[str, Any]], cfg: PrerestoreConfig, *, task_count: int
) -> dict[str, Any]:
    """Per-cell net-seconds table, task-clustered CI, and the kill/survive readout."""

    missing = set(_HEADLINE_COSTS_MS) - set(cfg.costs_ms)
    if missing:
        raise ValueError(
            f"cost panel is missing headline kv cell(s) {sorted(missing)}; the kill "
            "criterion is undefined -- refusing to emit a vacuous KILL"
        )
    contributions, task_ids = _contributions_matrix(decisions, cfg.costs_ms)
    observed, point_q, simul_q, permutation = _certificate(contributions, cfg)

    rows_by_cost: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        rows_by_cost[float(row["kv_cost_ms"])].append(row)

    cells: list[dict[str, Any]] = []
    survive_costs: list[float] = []
    for column, cost in enumerate(cfg.costs_ms):
        rows = rows_by_cost[cost]
        per_task_totals = contributions[:, column]
        hidden_total = sum(float(r["hidden_lead_ms"]) for r in rows)
        wasted_total = sum(float(r["wasted_restore_ms"]) for r in rows)
        fired_count = sum(1 for r in rows if r["fired"])
        net_ms = float(observed[column])
        # Accounting identity holds cell by cell: net == hidden - wasted.
        assert abs(net_ms - (hidden_total - wasted_total)) < 1e-6, (
            f"net {net_ms} != hidden {hidden_total} - wasted {wasted_total}"
        )
        label = permutation["points"][column]["permutation_label"]
        simul_low = float(simul_q[0, column])
        is_headline = cost in _HEADLINE_COSTS_MS
        if is_headline and label == "positive":
            survive_costs.append(cost)
        cells.append(
            {
                "kv_cost_ms": cost,
                "threshold_ms": cost + cfg.guard_ms,
                "restore_cost_ms": cfg.restore_cost_fraction * cost,
                "headline": is_headline,
                "net_ms_per_277": net_ms,
                "net_seconds_per_277": net_ms / 1000.0,
                "mean_ms_per_task": net_ms / task_count,
                "p90_ms_per_task": float(np.percentile(per_task_totals, 90)),
                "hidden_lead_ms_total": hidden_total,
                "wasted_restore_ms_total": wasted_total,
                "prerestore_fire_fraction": (fired_count / len(rows)) if rows else 0.0,
                "call_count": len(rows),
                "pointwise_interval_ms": {
                    "low": float(point_q[0, column]),
                    "high": float(point_q[1, column]),
                },
                "simultaneous_interval_ms": {
                    "low": simul_low,
                    "high": float(simul_q[1, column]),
                },
                "simultaneous_label": (
                    "positive"
                    if simul_low > 0.0
                    else "harmful"
                    if float(simul_q[1, column]) < 0.0
                    else "inconclusive"
                ),
                **permutation["points"][column],
            }
        )

    verdict = "SURVIVE" if survive_costs else "KILL"
    return {
        "verdict": verdict,
        "survive_positive_headline_costs_ms": survive_costs,
        "kill_criterion": (
            "net positive with a task-clustered permutation CI excluding zero "
            "(Bonferroni over the full cost family) in >=1 headline kv cell "
            f"({', '.join(f'{c:.0f}' for c in _HEADLINE_COSTS_MS)})"
        ),
        "task_count": task_count,
        "call_count": len({str(r["sample_id"]) for r in decisions}),
        "headline_costs_ms": list(_HEADLINE_COSTS_MS),
        "cells": cells,
        "permutation_config": permutation["config"],
    }


def run_prerestore_accounting(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: PrerestoreConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every fold's held-out calls and apply the frozen kill readout."""

    decisions = score_decisions(samples_by_task, task_ids, cfg)
    summary = summarize(decisions, cfg, task_count=len(task_ids))
    return summary, decisions


# --------------------------------------------------------------------------- #
# Rendering / CLI.
# --------------------------------------------------------------------------- #
def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# A2 pre-restore offline accounting")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY, OFFLINE ACCOUNTING. Restore lead-time overlap priced on "
        "original-trace latencies via the frozen manifest "
        f"({provenance['collection_id']}); GPU live validation remains W10-11. "
        f"Generated {provenance['generated']} (git {provenance['git_sha']})."
    )
    lines.append("")
    lines.append(f"**Verdict: {results['verdict']}**")
    lines.append("")
    lines.append(
        "> Swap-trigger asymmetry: the swap trigger `g` here is "
        "`hazard_recheck_ms`, the earlier-firing optimistic analog of the "
        "SHIPPED robust clock (which fires later or falls back to the deadline). "
        "An earlier `g` only widens the pre-restore window, so a **KILL is "
        "conservative/strong** (pre-restore fails even given its best shot), "
        "whereas any **SURVIVE must be re-confirmed under the shipped robust-clock "
        "`g` (W10-11 GPU validation) before pre-restore is acted on**."
    )
    lines.append("")
    lines.append(f"Kill criterion (frozen): {results['kill_criterion']}.")
    if results["verdict"] == "SURVIVE":
        costs = ", ".join(f"{c:.0f}" for c in results["survive_positive_headline_costs_ms"])
        lines.append("")
        lines.append(f"Met at headline kv cell(s): {costs}.")
    else:
        lines.append("")
        lines.append("Not met -- pre-restore ships as an honest negative table.")
    lines.append("")
    lines.append(
        f"{results['task_count']} tasks, {results['call_count']} calls, rho="
        f"{provenance['restore_cost_fraction']}, guard {provenance['guard_ms']:.0f}ms "
        f"(threshold==kv). Permutation: sign-flip, "
        f"{results['permutation_config']['draws']} draws, Bonferroni over "
        f"{results['permutation_config']['simultaneous_family_size']} costs."
    )
    lines.append("")
    lines.append("## Net seconds per 277 tasks (lead-time hidden - wasted restore)")
    lines.append("")
    lines.append(
        "| kv | net s/277 | mean ms/task | P90 ms/task | fire frac | hidden s | "
        "wasted s | perm label | simul CI ms |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["cells"]:
        marker = " (H)" if cell["headline"] else ""
        simul = cell["simultaneous_interval_ms"]
        lines.append(
            f"| {cell['kv_cost_ms']:.0f}{marker} | "
            f"{cell['net_seconds_per_277']:.2f} | "
            f"{cell['mean_ms_per_task']:.2f} | "
            f"{cell['p90_ms_per_task']:.2f} | "
            f"{cell['prerestore_fire_fraction']:.3f} | "
            f"{cell['hidden_lead_ms_total'] / 1000.0:.2f} | "
            f"{cell['wasted_restore_ms_total'] / 1000.0:.2f} | "
            f"{cell['permutation_label']} | "
            f"[{simul['low'] / 1000.0:.2f}, {simul['high'] / 1000.0:.2f}] |"
        )
    lines.append("")
    lines.append("(H) = headline cell. CI columns in seconds per 277 tasks.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "analysis/fresh-corpus-certification-20260717/"
            "offline-gated-robust/manifest.json"
        ),
    )
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Smoke only: cap tasks (subsets folds consistently). Rejected with "
        "--final.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/prerestore-accounting-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    samples_by_task, task_ids, manifest = _load_manifest_corpus(
        args.manifest, limit_tasks=args.limit_tasks, final=args.final
    )
    cfg = PrerestoreConfig(
        fold_count=manifest["fold_count"],
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
        min_tool_history=manifest["min_tool_history"],
        min_profile_tasks=manifest["min_profile_tasks"],
        costs_ms=tuple(float(cost) for cost in manifest["costs_ms"]),
        guard_ms=float(manifest.get("guard_ms", _CERT_GUARD_MS)),
        restore_cost_fraction=_CERT_RESTORE_COST_FRACTION,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    summary, decisions = run_prerestore_accounting(samples_by_task, task_ids, cfg)
    provenance = {
        "exploratory": True,
        "offline_accounting": True,
        "live_validation": "W10-11 (GPU)",
        "final": bool(args.final),
        "manifest": str(args.manifest),
        "collection_id": manifest["collection_id"],
        "task_count": len(task_ids),
        "limit_tasks": args.limit_tasks,
        "guard_ms": cfg.guard_ms,
        "restore_cost_fraction": cfg.restore_cost_fraction,
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    payload = {
        "provenance": provenance,
        "config": cfg.__dict__,
        **summary,
        "decisions": decisions,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(summary, provenance), encoding="utf-8")

    print(
        f"verdict={summary['verdict']} "
        f"survive_costs={summary['survive_positive_headline_costs_ms']}"
    )
    for cell in summary["cells"]:
        if cell["headline"]:
            print(
                f"  kv{cell['kv_cost_ms']:.0f}: net {cell['net_seconds_per_277']:.2f}s/277 "
                f"perm={cell['permutation_label']} fire={cell['prerestore_fire_fraction']:.3f}"
            )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
