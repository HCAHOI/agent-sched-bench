#!/usr/bin/env python3
"""A0 adjudication: does a k=2 re-check DP collapse onto the k=1 optimum?

Pre-registered in ``analysis/rolling-survival-design-20260720.md`` ("A0 --
Adjudication check"). The claim under test (the debate's decisive finding):
under the existing evaluation functional -- ``hazard_recheck_ms``
(``trace_collect.tool_latency_profiled``) as the exact k=1 optimizer and the
``utility_matrix`` per-call accounting that ``_accumulate_trigger_policy``
sums in replay -- any elapsed-only k-check policy collapses to a single
stopping time. A correctly-formulated k=2 dynamic program must therefore
attain EXACTLY the k=1 optimal value, and with a per-check overhead priced it
is (weakly) dominated. A0 tests that structural claim against implementation
reality (e.g. restore charged on short-call fires at the deadline boundary).

The policy space, faithfully. The tool's only runtime observable is "call
still alive at elapsed t"; the swap action is irreversible (fire once). A k=2
policy is a pair ``0 <= t1 < t2 <= T`` plus, at ``t1``, a defer/swap choice.
At ``t1`` every call still running looks identical (aliveness is the only
signal), so that choice is a single scalar precomputed from the fit sample
set -- NOT sample-conditional. Honest two-stage value on the fit samples:

* ``swap`` at ``t1``  -> the policy fires at ``t1`` for every call alive at
  ``t1``; value == single-trigger utility at ``t1`` (a call finishing before
  ``t1`` never fires).
* ``defer`` to ``t2`` -> no fire at ``t1``; every call still alive at ``t2``
  fires at ``t2`` (calls finishing in ``(t1, t2]`` never fire); value ==
  single-trigger utility at ``t2``.

We do NOT assume this reduction: ``k2_stage_value`` re-simulates the two-stage
policy call-by-call through the SAME functional (``trigger_policy_utility_ms``,
one fire per call), and a test asserts it equals the single-trigger utility at
the fired checkpoint. Given that (validated) reduction, the DP value of a pair
is ``max(u(t1), u(t2))`` and the k=2 optimum over the candidate grid is
``max_k u(k)`` -- so brute force is unnecessary once the reduction is checked.

Candidate-grid sufficiency (the exactness argument). ``u(k)`` is piecewise
linear in ``k`` with breakpoints only at ``{0, T}`` and, per sample latency
``L``, at ``L`` (the fire boundary) and ``L - kv`` (the exposed-term kink) --
exactly the candidate set ``hazard_recheck_ms`` enumerates. A piecewise-linear
function is maximized at a breakpoint, so optimizing over that grid is exact
for both k=1 and the k=2 stage values. ``_hazard_candidates`` mirrors the
optimizer's own set so the two policies are optimized over one grid.

Nodes. The certified offline-gated-robust clock feeds ``hazard_recheck_ms`` the
deepest prior node selected per eval call (``latency_prior_hierarchy(...)[-1]``
in ``tool_latency_offline_probe._score_clock_rows`` via
``mean_clock_region_stats``). We reproduce the frozen manifest's outer folds
and enumerate exactly those selected node sample sets -- original-trace
latencies via the manifest (what the certified policy consumes; the segment
corpus would answer a different, replayed-residual question). Every kv cell in
the cost panel is evaluated at the certified operating point (guard 0 so
threshold == kv, rho = 0.94 restore fraction).

Readout (binding, per the design spec):

* COLLAPSE CONFIRMED iff, over ALL nodes x cells, the value gap
  ``|u_k2_opt - u_k1|`` is within ``--tolerance-ms`` AND every induced-decision
  difference is a value-tie (paired delta ``u(t_k2) - u(t_k1)`` exactly 0
  within tolerance). k>1 stays dead.
* COLLAPSE FALSIFIED otherwise, listing offending nodes (key, gap, triggers).
  Per the spec this reinstates the D2 replay experiment.

Domination (reported alongside, one positive-overhead run): with a per-check
overhead priced, ``priced_u_k2_opt <= priced_u_k1_opt`` for every node/cell
(k=2 never strictly better), and a genuine two-check (defer) policy is strictly
below the k=1 optimum wherever an earlier check carries mass.

EXPLORATORY until ``--final`` (cheap: closed-form over node sample sets, no
bootstrap). Emits JSON + MD to ``analysis/`` (``-PARTIAL`` unless ``--final``).

Usage:
  uv run python scripts/certification/adjudicate_k2_recheck.py \
    --manifest analysis/results/prequential-task-update-20260721/inputs/\
swe-277.json --final
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import functools
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

# Allow direct `python scripts/certification/adjudicate_k2_recheck.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trace_collect.cli_helpers import resolve_worker_count  # noqa: E402
from trace_collect.tool_latency_dataset import (  # noqa: E402
    ToolLatencySample,
    discover_trace_files,
    extract_many_tool_latency_samples,
    read_tool_latency_corpus_manifest,
    require_explicit_trace_task_ids,
)
from trace_collect.tool_latency_profiled import (  # noqa: E402
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
)
from trace_collect.tool_latency_utility_clock import (  # noqa: E402
    trigger_policy_utility_ms,
    utility_matrix,
)

# Fixed operating point (retained Fresh-277 manifest +
# scripts/serving/export_trigger_table.py): guard 0 (threshold == kv) and the measured
# restore-cost fraction rho = 0.94. The kv-cost panel is read from the manifest.
_CERT_GUARD_MS = 0.0
_CERT_RESTORE_COST_FRACTION = 0.94


def _hazard_candidates(
    values: Sequence[float], *, threshold_ms: float, kv_cost_ms: float
) -> np.ndarray:
    """Breakpoint grid of the piecewise-linear per-trigger utility.

    Mirrors the candidate set ``hazard_recheck_ms`` enumerates ({0, T} plus, per
    sample L, the interior points L and L - kv). Kept in lockstep with the
    optimizer so the k=1 and k=2 policies are maximized over one identical grid;
    the sufficiency argument (module docstring) makes the grid maximum exact.
    """

    candidates = {0.0, float(threshold_ms)}
    for value in values:
        if 0.0 < value < threshold_ms:
            candidates.add(float(value))
        edge = value - kv_cost_ms
        if 0.0 < edge < threshold_ms:
            candidates.add(float(edge))
    return np.asarray(sorted(candidates), dtype=float)


def single_trigger_utilities(
    samples: np.ndarray,
    candidates: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float,
) -> np.ndarray:
    """Mean per-call utility of a single trigger at each candidate.

    Uses the shared ``utility_matrix`` functional (hidden_on_long - exposed -
    restore), i.e. exactly what ``_accumulate_trigger_policy`` sums in replay
    and what ``hazard_recheck_ms`` maximizes internally.
    """

    matrix = utility_matrix(
        samples,
        candidates,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    return matrix.mean(axis=0)


def k2_stage_value(
    samples: np.ndarray,
    *,
    t1_ms: float,
    t2_ms: float,
    branch: str,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float,
) -> float:
    """Honest two-stage value: re-simulate the fire-once policy call-by-call.

    ``branch='swap'`` fires at ``t1`` for calls alive at ``t1``; ``branch=
    'defer'`` fires at ``t2`` for calls alive at ``t2``. No reduction assumed;
    each call is scored through ``trigger_policy_utility_ms`` (one fire). A test
    asserts this equals ``single_trigger_utilities`` at the fired checkpoint --
    validating the reduction the optimizer then relies on.
    """

    if not (0.0 <= t1_ms < t2_ms):
        raise ValueError(f"require 0 <= t1 < t2, got t1={t1_ms}, t2={t2_ms}")
    if branch not in ("swap", "defer"):
        raise ValueError(f"branch must be 'swap' or 'defer', got {branch!r}")
    fire_ms = t1_ms if branch == "swap" else t2_ms
    total = 0.0
    for latency_ms in samples:
        total += trigger_policy_utility_ms(
            float(latency_ms),
            fire_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
            restore_cost_ms=restore_cost_ms,
        )
    return total / len(samples)


@dataclass(frozen=True)
class NodeCellResult:
    """k=1 vs k=2 adjudication for one node sample set at one kv cell."""

    node_key: str
    fold: str
    prior_source: str
    prior_group_key: str | None
    sample_count: int
    kv_cost_ms: float
    threshold_ms: float
    restore_cost_ms: float
    k1_trigger_ms: float
    k1_value: float
    k2_trigger_ms: float
    k2_branch: str
    k2_value: float
    value_gap: float
    paired_delta: float
    priced_k1_value: float
    priced_k2_value: float
    priced_best_defer_value: float | None
    priced_domination_gap: float  # priced_k2 - priced_k1; <= 0 == dominated

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "node_key": self.node_key,
            "fold": self.fold,
            "prior_source": self.prior_source,
            "prior_group_key": self.prior_group_key,
            "sample_count": self.sample_count,
            "kv_cost_ms": self.kv_cost_ms,
            "threshold_ms": self.threshold_ms,
            "restore_cost_ms": self.restore_cost_ms,
            "k1_trigger_ms": self.k1_trigger_ms,
            "k1_value": self.k1_value,
            "k2_trigger_ms": self.k2_trigger_ms,
            "k2_branch": self.k2_branch,
            "k2_value": self.k2_value,
            "value_gap": self.value_gap,
            "paired_delta": self.paired_delta,
            "priced_k1_value": self.priced_k1_value,
            "priced_k2_value": self.priced_k2_value,
            "priced_best_defer_value": self.priced_best_defer_value,
            "priced_domination_gap": self.priced_domination_gap,
        }


def adjudicate_node_cell(
    values: Sequence[float],
    *,
    node_key: str,
    fold: str,
    prior_source: str,
    prior_group_key: str | None,
    kv_cost_ms: float,
    guard_ms: float,
    restore_cost_fraction: float,
    overhead_ms: float,
) -> NodeCellResult:
    """Adjudicate one node sample set at one kv cell (unpriced + priced)."""

    threshold_ms = kv_cost_ms + guard_ms
    restore_cost_ms = restore_cost_fraction * kv_cost_ms
    samples = np.asarray(values, dtype=float)
    candidates = _hazard_candidates(
        values, threshold_ms=threshold_ms, kv_cost_ms=kv_cost_ms
    )

    # k=1: the certified optimizer itself (never reimplemented), then scored.
    k1_trigger_ms = hazard_recheck_ms(
        list(values),
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    k1_value = float(
        utility_matrix(
            samples,
            np.asarray([k1_trigger_ms], dtype=float),
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
            restore_cost_ms=restore_cost_ms,
        ).mean()
    )

    benefit = single_trigger_utilities(
        samples,
        candidates,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    # k=2 optimum. Per-pair DP value == max(benefit[i], benefit[j]) (validated
    # reduction), so the optimum over the grid of pairs is the grid maximum. The
    # collapsed single stopping time is the argmax candidate; its branch is
    # 'defer' when any earlier candidate exists (the DP defers past it), else
    # 'swap' (the earliest candidate can only be reached by swapping at it).
    best_index = int(np.argmax(benefit))
    k2_value = float(benefit[best_index])
    k2_trigger_ms = float(candidates[best_index])
    k2_branch = "defer" if best_index > 0 else "swap"

    value_gap = k2_value - k1_value
    paired_delta = float(
        utility_matrix(
            samples,
            np.asarray([k2_trigger_ms], dtype=float),
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
            restore_cost_ms=restore_cost_ms,
        ).mean()
    ) - k1_value

    # Priced (per-check overhead). A check at candidate c is performed only for
    # calls still alive at c: fraction P(L > c), non-increasing in c.
    fire_fraction = np.asarray(
        [float(np.mean(samples > c)) for c in candidates], dtype=float
    )
    priced_benefit = benefit - overhead_ms * fire_fraction
    priced_k1_value = float(priced_benefit.max())
    # swap branch of any pair (i, j>i) yields priced_benefit[i]: reachable for
    # every i except the last candidate (needs a strictly later t2).
    priced_swap_best = (
        float(priced_benefit[:-1].max()) if len(candidates) > 1 else -np.inf
    )
    # defer branch (i<j): benefit[j] - overhead*(fire[i] + fire[j]); for fixed j
    # the cheapest earlier check is the nearest one (fire non-increasing), i.e.
    # i = j-1.
    defer_values = [
        float(benefit[j] - overhead_ms * (fire_fraction[j - 1] + fire_fraction[j]))
        for j in range(1, len(candidates))
    ]
    priced_best_defer_value = max(defer_values) if defer_values else None
    priced_k2_value = max(
        [v for v in (priced_swap_best, priced_best_defer_value) if v is not None]
    )
    priced_domination_gap = priced_k2_value - priced_k1_value

    return NodeCellResult(
        node_key=node_key,
        fold=fold,
        prior_source=prior_source,
        prior_group_key=prior_group_key,
        sample_count=len(values),
        kv_cost_ms=kv_cost_ms,
        threshold_ms=threshold_ms,
        restore_cost_ms=restore_cost_ms,
        k1_trigger_ms=k1_trigger_ms,
        k1_value=k1_value,
        k2_trigger_ms=k2_trigger_ms,
        k2_branch=k2_branch,
        k2_value=k2_value,
        value_gap=value_gap,
        paired_delta=paired_delta,
        priced_k1_value=priced_k1_value,
        priced_k2_value=priced_k2_value,
        priced_best_defer_value=priced_best_defer_value,
        priced_domination_gap=priced_domination_gap,
    )


@dataclass(frozen=True)
class SelectedNode:
    """A distinct prior node sample set the certified policy feeds hazard."""

    node_key: str
    fold: str
    prior_source: str
    prior_group_key: str | None
    values: tuple[float, ...]


def enumerate_selected_nodes(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    *,
    fold_count: int,
    command_field: str,
    max_prefix_depth: int,
    skip_leading_cd: bool,
    min_tool_history: int,
    min_profile_tasks: int,
) -> list[SelectedNode]:
    """Enumerate the deepest prior nodes selected per eval call, per outer fold.

    Reproduces the frozen manifest's outer folds and the certified node
    selection (``latency_prior_hierarchy(...)[-1]``). Nodes are de-duplicated by
    object identity within a fold: that is exactly the set of distinct sample
    sets ``hazard_recheck_ms`` is invoked on (each cached by ``id(values)`` in
    the production path), so the adjudication covers every node/cell the
    certified policy actually consumes -- no more, no less.
    """

    declared = list(task_ids)
    nodes: list[SelectedNode] = []
    for fold in range(1, fold_count + 1):
        eval_tasks = {
            task_id
            for index, task_id in enumerate(declared)
            if index % fold_count == fold - 1
        }
        profile_tasks = set(declared) - eval_tasks
        profile_rows = [
            sample.to_json_obj()
            for task_id in sorted(profile_tasks)
            for sample in samples_by_task[task_id]
        ]
        eval_rows = [
            sample.to_json_obj()
            for task_id in sorted(eval_tasks)
            for sample in samples_by_task[task_id]
        ]
        row_group_keys = _row_group_keys(
            command_field,
            max_prefix_depth=max_prefix_depth,
            skip_leading_cd=skip_leading_cd,
        )
        prior = build_latency_prior(profile_rows, row_group_keys=row_group_keys)
        seen_ids: set[int] = set()
        for row in eval_rows:
            tool_name = str(row["tool_name"])
            group_keys = row_group_keys(row)
            selected = latency_prior_hierarchy(
                prior,
                tool_name,
                group_keys,
                min_tool_history=min_tool_history,
                min_profile_tasks=min_profile_tasks,
            )[-1]
            marker = id(selected.values)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            group_label = selected.group_key if selected.group_key is not None else "-"
            nodes.append(
                SelectedNode(
                    node_key=(
                        f"f{fold}:{selected.source}:{group_label}:"
                        f"n{len(selected.values)}"
                    ),
                    fold=f"f{fold}",
                    prior_source=selected.source,
                    prior_group_key=selected.group_key,
                    values=tuple(selected.values),
                )
            )
    return nodes


def _row_group_keys(
    command_field: str, *, max_prefix_depth: int, skip_leading_cd: bool
):
    from trace_collect.command_features import make_row_command_prefix_keys

    return make_row_command_prefix_keys(
        command_field,
        max_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )


@dataclass
class AdjudicationConfig:
    fold_count: int
    command_field: str
    max_prefix_depth: int
    skip_leading_cd: bool
    min_tool_history: int
    min_profile_tasks: int
    costs_ms: tuple[float, ...]
    guard_ms: float
    restore_cost_fraction: float
    overhead_ms: float
    tolerance_ms: float


@dataclass
class AdjudicationResult:
    cells: list[NodeCellResult] = field(default_factory=list)


def _adjudicate_node(
    node: SelectedNode, *, cfg: AdjudicationConfig
) -> list[NodeCellResult]:
    return [
        adjudicate_node_cell(
            node.values,
            node_key=node.node_key,
            fold=node.fold,
            prior_source=node.prior_source,
            prior_group_key=node.prior_group_key,
            kv_cost_ms=kv_cost_ms,
            guard_ms=cfg.guard_ms,
            restore_cost_fraction=cfg.restore_cost_fraction,
            overhead_ms=cfg.overhead_ms,
        )
        for kv_cost_ms in cfg.costs_ms
    ]


def run_adjudication(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: AdjudicationConfig,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Adjudicate every certified node x cell and apply the binding readout."""

    workers = resolve_worker_count(workers)
    nodes = enumerate_selected_nodes(
        samples_by_task,
        task_ids,
        fold_count=cfg.fold_count,
        command_field=cfg.command_field,
        max_prefix_depth=cfg.max_prefix_depth,
        skip_leading_cd=cfg.skip_leading_cd,
        min_tool_history=cfg.min_tool_history,
        min_profile_tasks=cfg.min_profile_tasks,
    )
    worker = functools.partial(_adjudicate_node, cfg=cfg)
    if workers == 1 or len(nodes) < 2:
        per_node = list(map(worker, nodes))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(nodes))) as pool:
            per_node = list(pool.map(worker, nodes, chunksize=8))
    return summarize([cell for cells in per_node for cell in cells], cfg)


def summarize(cells: Sequence[NodeCellResult], cfg: AdjudicationConfig) -> dict[str, Any]:
    """Apply the binding COLLAPSE readout and the domination check."""

    tol = cfg.tolerance_ms
    max_value_gap = max((abs(c.value_gap) for c in cells), default=0.0)
    max_paired_delta = max((abs(c.paired_delta) for c in cells), default=0.0)
    # Offending == a real value gap OR a paired-delta that is not a value-tie.
    offenders = [
        c
        for c in cells
        if abs(c.value_gap) > tol or abs(c.paired_delta) > tol
    ]
    collapse = not offenders
    # Value-tied churn: k1 and k2 triggers differ but the paired delta is zero.
    churn = [
        c
        for c in cells
        if abs(c.k1_trigger_ms - c.k2_trigger_ms) > tol and abs(c.paired_delta) <= tol
    ]
    max_domination_gap = max((c.priced_domination_gap for c in cells), default=0.0)
    domination_holds = max_domination_gap <= tol
    genuine_defers = [
        c
        for c in cells
        if c.priced_best_defer_value is not None
        and c.priced_best_defer_value < c.priced_k1_value - tol
    ]
    return {
        "verdict": "COLLAPSE CONFIRMED" if collapse else "COLLAPSE FALSIFIED",
        "collapse_confirmed": collapse,
        "node_count": len({c.node_key for c in cells}),
        "cell_count": len(cells),
        "max_value_gap": max_value_gap,
        "max_abs_paired_delta": max_paired_delta,
        "tolerance_ms": tol,
        "value_tied_churn_cell_count": len(churn),
        "offending_cells": [c.to_json_obj() for c in offenders],
        "domination": {
            "overhead_ms": cfg.overhead_ms,
            "holds": domination_holds,
            "max_priced_domination_gap": max_domination_gap,
            "genuine_defer_dominated_cell_count": len(genuine_defers),
        },
        "cells": [c.to_json_obj() for c in cells],
    }


# --------------------------------------------------------------------------- #
# Rendering / CLI.
# --------------------------------------------------------------------------- #
def _git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _banner(final: bool) -> str:
    if final:
        return "FINAL - complete corpus"
    return (
        "PARTIAL - not final: validated against a subset; numbers are for "
        "script validation only, not findings."
    )


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# A0 adjudication: k=2 re-check DP vs certified k=1")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Original-trace latencies via the frozen manifest "
        f"({provenance['collection_id']}). Generated {provenance['generated']} "
        f"(git {provenance['git_sha']})."
    )
    lines.append("")
    lines.append(f"**Verdict: {results['verdict']}**")
    lines.append("")
    lines.append(
        f"Nodes: {results['node_count']} distinct selected prior sample sets x "
        f"{provenance['cost_count']} kv cells = {results['cell_count']} cells at "
        f"guard {provenance['guard_ms']:.0f}ms (threshold==kv), rho="
        f"{provenance['restore_cost_fraction']}."
    )
    lines.append("")
    lines.append("## Value equivalence (unpriced)")
    lines.append("")
    lines.append(
        f"- max |k2_opt - k1| value gap: {results['max_value_gap']:.3e} ms "
        f"(tolerance {results['tolerance_ms']:.1e})"
    )
    lines.append(
        f"- max |paired delta| (induced decisions): "
        f"{results['max_abs_paired_delta']:.3e} ms"
    )
    lines.append(
        f"- value-tied churn cells (different trigger, zero value delta): "
        f"{results['value_tied_churn_cell_count']}"
    )
    lines.append(f"- offending cells: {len(results['offending_cells'])}")
    lines.append("")
    if results["offending_cells"]:
        lines.append("### Offending nodes/cells")
        lines.append("")
        lines.append("| node | kv | k1 trig | k2 trig | value gap | paired delta |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in results["offending_cells"]:
            lines.append(
                f"| {c['node_key']} | {c['kv_cost_ms']:.0f} | "
                f"{c['k1_trigger_ms']:.1f} | {c['k2_trigger_ms']:.1f} | "
                f"{c['value_gap']:.3e} | {c['paired_delta']:.3e} |"
            )
        lines.append("")
    dom = results["domination"]
    lines.append("## Per-check overhead: domination")
    lines.append("")
    lines.append(
        f"At overhead {dom['overhead_ms']:.3g} ms/check: k=2 never strictly "
        f"beats k=1 (max priced gap {dom['max_priced_domination_gap']:.3e} ms, "
        f"holds={dom['holds']}). Genuine two-check (defer) policies strictly "
        f"dominated in {dom['genuine_defer_dominated_cell_count']} cells."
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "analysis/results/prequential-task-update-20260721/inputs/"
            "swe-277.json"
        ),
    )
    parser.add_argument(
        "--overhead-ms",
        type=float,
        default=1.0,
        help="Per-check overhead priced in the domination run (a small positive "
        "cost; the unpriced value-equivalence test always runs at 0).",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=1e-6,
        help="Numerical tolerance for value gaps and paired deltas.",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Smoke only: cap tasks (subsets folds consistently). Rejected with "
        "--final.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=resolve_worker_count(),
        help="CPU processes for deterministic per-node adjudication; 1 is sequential.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/certification/adjudication-k2-recheck-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def _load_manifest_corpus(
    manifest_path: Path, *, limit_tasks: int | None, final: bool
) -> tuple[dict[str, list[ToolLatencySample]], list[str], dict[str, Any]]:
    """Load fresh-277 samples grouped by task, matching the frozen manifest."""

    repo_root = Path(__file__).resolve().parents[2]
    manifest = read_tool_latency_corpus_manifest(manifest_path.resolve(), repo_root=repo_root)
    trace_root = Path(manifest["trace_root"])
    trace_paths = discover_trace_files([trace_root])
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {trace_root}")
    task_by_trace = require_explicit_trace_task_ids(trace_paths)
    samples = extract_many_tool_latency_samples(trace_paths)
    samples_by_task: dict[str, list[ToolLatencySample]] = defaultdict(list)
    for sample in samples:
        expected = task_by_trace.get(str(Path(sample.source_trace).resolve()))
        if expected is None or sample.task_id != expected:
            raise ValueError(
                "extracted sample task_id differs from explicit trace metadata: "
                f"{sample.source_trace}: {sample.task_id!r} != {expected!r}"
            )
        samples_by_task[sample.task_id].append(sample)

    task_ids = list(manifest["task_ids"])
    if len(task_ids) != manifest["expected_task_count"]:
        raise ValueError(
            "manifest expected_task_count differs from pinned task_ids: "
            f"{manifest['expected_task_count']} != {len(task_ids)}"
        )
    if set(samples_by_task) != set(task_ids):
        raise ValueError(
            "extracted logical tasks differ from pinned task_ids: "
            f"missing={sorted(set(task_ids) - set(samples_by_task))}, "
            f"unexpected={sorted(set(samples_by_task) - set(task_ids))}"
        )
    if limit_tasks is not None:
        if final:
            raise ValueError("--limit-tasks is a smoke knob; not allowed with --final")
        if limit_tasks < manifest["fold_count"]:
            raise ValueError(
                f"--limit-tasks must be >= fold_count ({manifest['fold_count']})"
            )
        task_ids = task_ids[:limit_tasks]
    return dict(samples_by_task), task_ids, manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    samples_by_task, task_ids, manifest = _load_manifest_corpus(
        args.manifest, limit_tasks=args.limit_tasks, final=args.final
    )
    cfg = AdjudicationConfig(
        fold_count=manifest["fold_count"],
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
        min_tool_history=manifest["min_tool_history"],
        min_profile_tasks=manifest["min_profile_tasks"],
        costs_ms=tuple(float(cost) for cost in manifest["costs_ms"]),
        guard_ms=float(manifest.get("guard_ms", _CERT_GUARD_MS)),
        restore_cost_fraction=_CERT_RESTORE_COST_FRACTION,
        overhead_ms=args.overhead_ms,
        tolerance_ms=args.tolerance_ms,
    )
    results = run_adjudication(samples_by_task, task_ids, cfg, workers=args.workers)
    provenance = {
        "exploratory": True,
        "final": bool(args.final),
        "manifest": str(args.manifest),
        "collection_id": manifest["collection_id"],
        "task_count": len(task_ids),
        "limit_tasks": args.limit_tasks,
        "workers": args.workers,
        "cost_count": len(cfg.costs_ms),
        "guard_ms": cfg.guard_ms,
        "restore_cost_fraction": cfg.restore_cost_fraction,
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    payload = {"provenance": provenance, "config": cfg.__dict__, **results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(results, provenance), encoding="utf-8")

    dom = results["domination"]
    print(
        f"verdict={results['verdict']} max_value_gap={results['max_value_gap']:.3e} "
        f"max_paired_delta={results['max_abs_paired_delta']:.3e} "
        f"domination_holds={dom['holds']} "
        f"max_priced_gap={dom['max_priced_domination_gap']:.3e}"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
