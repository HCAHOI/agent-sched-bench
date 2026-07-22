#!/usr/bin/env python3
"""WTN Stage-2: certified decision replay of a joint policy vs the frozen cert.

Pre-registered in ``analysis/CLOSED-QUESTIONS.md``, section
"Stage-2 pre-registration". Drives THREE arms through the EXISTING offline-probe
confirmation machinery (``trace_collect.tool_latency_offline_probe``) on the
frozen fresh-277 manifest, differing ONLY by configuration -- no pipeline fork,
no parallel reimplementation:

* **P_new**: production prior + screen-learned key normalization (the WTN screen
  re-learned per fit fold on ORIGINAL tool-latency durations) + node evidence
  gate 5.
* **P_0**: the exact frozen certified config (``skip_leading_cd=False``,
  ``min_tool_history=1``, no normalization) -- the H1-certified policy.
* **oracle**: hardcoded cd-only strip + node evidence gate 5. Reported alongside
  but NEVER certified or shipped (standing directive, 2026-07-19: the hardcoded
  cd-skip rule is an oracle-baseline row, never a method).

All arms run the certified robust clock (``offline_gated_robust_trigger_ms``) at
rho=0.94. The comparison is a paired per-task trigger-policy utility at rho=0.94
with a task-clustered sign-flip permutation certificate per kv cell
(``paired_task_cluster_bootstrap``) at the SAME replicate/Bonferroni discipline
as the H1 certification (replicates 50000, confidence 0.95, seed 0; the full kv
cost family drives the Bonferroni correction, kv3500/kv5000 are the headline
cells).

The transparent set is learned by the review-APPROVED Stage-1 screen
(``analyze_wrapper_transparency.screen_transparency``) on each fit fold's
ORIGINAL tool-latency rows -- per-call totals + command text only, no segment
telemetry -- cross-fitted: learned on the fit (profile) fold, applied only to
that fold's predictions. No token spelling appears in method logic; ``cd`` is
the K2 positive control only.

Pre-registered KILL:

* no kv cell where P_new and P_0 DIVERGE certifies POSITIVE (permutation), or
* any headline cell certifies HARMFUL (P_new significantly worse), or
* the original-durations screen does not recover ``cd`` transparent in every
  fold (contradicting Stage-1) -- reported as the finding, not papered over.

No synthetic data: original traces from the frozen manifest. EXPLORATORY until
``--final``.

Usage (full corpus -- run by the main session, not the smoke):
  uv run python scripts/exploration/run_wtn_stage2.py \
    --manifest analysis/fresh-corpus-certification-20260717/\
offline-gated-robust/manifest.json --final
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
import datetime as _dt
import functools
import json
from pathlib import Path
import sys
from typing import Any, Sequence

# Allow direct `python scripts/exploration/run_wtn_stage2.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploration.analyze_segment_variance import Chain, _git_sha  # noqa: E402
from scripts.exploration.analyze_wrapper_transparency import (  # noqa: E402
    ScreenConfig,
    ScreenResult,
    screen_transparency,
)
from scripts.certification.run_offline_gated_robust_confirmation import (  # noqa: E402
    _read_manifest,
    _read_task_ids,
    _require_explicit_trace_task_ids,
    resolve_worker_count,
)
from trace_collect.tool_latency_confirmation import (  # noqa: E402
    paired_task_cluster_bootstrap,
)
from trace_collect.tool_latency_dataset import (  # noqa: E402
    ToolLatencySample,
    discover_trace_files,
    extract_many_tool_latency_samples,
)
from trace_collect.tool_latency_offline_probe import (  # noqa: E402
    evaluate_offline_probe_clock,
)

# The ONE method-adjacent token spelling, allowed by the integrity rule solely
# as the K2 positive control (a generic POSIX builtin whose transparency is
# known a-priori). It NEVER selects, drops, or weights anything -- only asserts
# the data-driven screen recovers it on original durations.
_K2_POSITIVE_CONTROL_CLASS = "cd"

# The certified policy trigger every arm is compared on (the H1 gate).
_CERT_TRIGGER_FIELD = "offline_gated_robust_trigger_ms"

# Headline cells for the verdict (spec: "per kv cell (kv3500, kv5000)"). The
# FULL cost family still drives the Bonferroni correction, matching H1.
_HEADLINE_COSTS_MS = (3500.0, 5000.0)

# Pre-registered P_new node evidence gate (Stage-1 grid's min_evidence=5 arm).
_PNEW_MIN_TOOL_HISTORY_DEFAULT = 5


def _chains_from_samples(
    samples: Sequence[ToolLatencySample], *, command_field: str
) -> list[Chain]:
    """Build screen-input Chains from ORIGINAL tool-latency rows.

    The screen needs only per-call totals (``latency_ms``) and command text
    (``tool_args[command_field]``); ``segments`` is left empty because the
    transparency screen never reads it (candidate extraction is purely from the
    command string). Rows without a usable command keep an empty command, so
    they group at the tool level exactly as the production prior does.
    """

    chains: list[Chain] = []
    for sample in samples:
        command = ""
        if sample.tool_args is not None:
            value = sample.tool_args.get(command_field)
            if isinstance(value, str):
                command = value
        chains.append(
            Chain(
                task_id=sample.task_id,
                source_trace=sample.source_trace,
                action_id=sample.action_id,
                tool_name=sample.tool_name,
                parent_command=command,
                parent_total_ms=sample.latency_ms,
                parent_raw_total_ms=None,
                segments=(),
            )
        )
    return chains


def _run_arm(
    eval_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    *,
    costs_ms: Sequence[float],
    guard_ms: float,
    inner_folds: int,
    min_tool_history: int,
    min_profile_tasks: int,
    command_field: str,
    max_prefix_depth: int,
    skip_leading_cd: bool,
    transparent_wrappers: frozenset[str],
    restore_cost_fraction: float,
) -> list[dict[str, Any]]:
    """One arm = one offline-probe evaluation; only the config differs."""

    result = evaluate_offline_probe_clock(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=costs_ms,
        guard_ms=guard_ms,
        inner_folds=inner_folds,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        transparent_wrappers=transparent_wrappers,
        restore_cost_fraction=restore_cost_fraction,
    )
    return result["decisions"]


def _merge_fold_arms(
    p0: list[dict[str, Any]],
    pnew: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    *,
    fold: str,
) -> list[dict[str, Any]]:
    """Join the three arms per (sample, cost) into paired-trigger decision rows.

    The three arms score the SAME held-out eval rows over the SAME cost panel,
    so latency/threshold/task must agree per key; only the cert trigger differs
    (that is the whole comparison). The merged row carries all three triggers
    for ``paired_task_cluster_bootstrap`` to difference.
    """

    def index(decisions: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
        out: dict[tuple[str, float], dict[str, Any]] = {}
        for decision in decisions:
            key = (str(decision["sample_id"]), float(decision["kv_cost_ms"]))
            if key in out:
                raise ValueError(f"duplicate (sample, cost) in arm: {key}")
            out[key] = decision
        return out

    p0_index, pnew_index, oracle_index = index(p0), index(pnew), index(oracle)
    if set(p0_index) != set(pnew_index) or set(p0_index) != set(oracle_index):
        raise ValueError("arms produced different (sample, cost) panels")

    rows: list[dict[str, Any]] = []
    for key, base in p0_index.items():
        arm_new = pnew_index[key]
        arm_oracle = oracle_index[key]
        for other in (arm_new, arm_oracle):
            if (
                str(other["task_id"]) != str(base["task_id"])
                or float(other["latency_ms"]) != float(base["latency_ms"])
                or float(other["threshold_ms"]) != float(base["threshold_ms"])
            ):
                raise ValueError(f"arms disagree on eval metadata for {key}")
        rows.append(
            {
                "sample_id": base["sample_id"],
                "task_id": base["task_id"],
                "tool_name": base["tool_name"],
                "kv_cost_ms": base["kv_cost_ms"],
                "threshold_ms": base["threshold_ms"],
                "latency_ms": base["latency_ms"],
                "outer_fold": fold,
                "p0_trigger_ms": base[_CERT_TRIGGER_FIELD],
                "pnew_trigger_ms": arm_new[_CERT_TRIGGER_FIELD],
                "oracle_trigger_ms": arm_oracle[_CERT_TRIGGER_FIELD],
            }
        )
    return rows


def _divergence_by_cost(
    merged: Sequence[dict[str, Any]],
    *,
    treatment_field: str,
    baseline_field: str,
) -> dict[float, int]:
    """Per cost: number of samples where treatment and baseline triggers differ."""

    diverged: dict[float, int] = defaultdict(int)
    for row in merged:
        if abs(float(row[treatment_field]) - float(row[baseline_field])) > 1e-9:
            diverged[float(row["kv_cost_ms"])] += 1
    return dict(diverged)


def _certificate(
    merged: Sequence[dict[str, Any]],
    *,
    treatment_field: str,
    baseline_field: str,
    costs_ms: Sequence[float],
    replicates: int,
    confidence_level: float,
    seed: int,
    restore_cost_fraction: float,
) -> dict[str, Any]:
    """Paired task-cluster utility + sign-flip permutation (H1 machinery)."""

    return paired_task_cluster_bootstrap(
        merged,
        costs_ms=costs_ms,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        baseline_trigger_field=baseline_field,
        treatment_trigger_field=treatment_field,
        restore_cost_fraction=restore_cost_fraction,
        enforce_gated_treatment=False,
        permutation_draws=replicates,
    )


def _screen_consistency(fold_screens: Sequence[tuple[int, ScreenResult]]) -> dict[str, Any]:
    """K2 on ORIGINAL durations: cd must emerge transparent in every fold."""

    per_fold = [
        {
            "fold": fold,
            "degenerate": screen.degenerate,
            "marginal_transparent": sorted(screen.marginal),
            "applied_transparent": sorted(screen.applied),
            "cd_marginal": _K2_POSITIVE_CONTROL_CLASS in screen.marginal,
        }
        for fold, screen in fold_screens
    ]
    non_degenerate = [row for row in per_fold if not row["degenerate"]]
    cd_is_candidate = any(
        _K2_POSITIVE_CONTROL_CLASS in screen.universe for _, screen in fold_screens
    )
    cd_every_fold = bool(non_degenerate) and all(
        row["cd_marginal"] for row in non_degenerate
    )
    return {
        "positive_control_class": _K2_POSITIVE_CONTROL_CLASS,
        "candidate_present": cd_is_candidate,
        "cd_transparent_every_fold": cd_every_fold,
        "per_fold": per_fold,
    }


def compute_verdict(
    pnew_cert: dict[str, Any],
    divergence: dict[float, int],
    screen_consistency: dict[str, Any],
    *,
    headline_costs_ms: Sequence[float],
) -> dict[str, Any]:
    """Apply the pre-registered kill criteria to the certificate.

    CERTIFIED iff (a) cd emerges transparent every fold on original durations,
    AND (b) at least one HEADLINE cell where P_new and P_0 diverge certifies
    POSITIVE by permutation, AND (c) no headline cell certifies HARMFUL.
    """

    points = pnew_cert["points"]
    headline: list[dict[str, Any]] = []
    diverging_positive: list[float] = []
    any_harmful = False
    for cost in headline_costs_ms:
        point = points.get(str(cost))
        if point is None:
            raise ValueError(f"certificate is missing headline cost {cost}")
        label = point["permutation_label"]
        diverged = divergence.get(cost, 0) > 0
        if label == "harmful":
            any_harmful = True
        if diverged and label == "positive":
            diverging_positive.append(cost)
        headline.append(
            {
                "kv_cost_ms": cost,
                "permutation_label": label,
                "permutation_p_positive": point["permutation_p_positive"],
                "permutation_p_harmful": point["permutation_p_harmful"],
                "paired_delta_ms": point["paired_delta_ms"],
                "diverged_sample_count": divergence.get(cost, 0),
            }
        )

    cd_ok = bool(screen_consistency["cd_transparent_every_fold"])
    killed_reasons: list[str] = []
    if not cd_ok:
        killed_reasons.append(
            "screen did not recover cd transparent in every fold (Stage-1 contradiction)"
        )
    if not diverging_positive:
        killed_reasons.append(
            "no headline kv cell where P_new and P_0 diverge certified positive"
        )
    if any_harmful:
        killed_reasons.append("a headline kv cell certified P_new harmful")
    verdict = "KILL" if killed_reasons else "CERTIFIED"
    return {
        "verdict": verdict,
        "killed_reasons": killed_reasons,
        "cd_transparent_every_fold": cd_ok,
        "diverging_positive_costs_ms": diverging_positive,
        "any_headline_harmful": any_harmful,
        "headline_cells": headline,
    }


@dataclass(frozen=True)
class _FoldReplayConfig:
    fold_count: int
    screen: ScreenConfig
    costs_ms: tuple[float, ...]
    guard_ms: float
    inner_folds: int
    min_profile_tasks: int
    command_field: str
    max_prefix_depth: int
    restore_cost_fraction: float
    cert_min_tool_history: int
    cert_skip_leading_cd: bool
    pnew_min_tool_history: int


def _run_fold_replay(
    fold: int,
    *,
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: _FoldReplayConfig,
) -> tuple[int, ScreenResult, list[dict[str, Any]]]:
    eval_tasks = {
        task_id
        for index, task_id in enumerate(task_ids)
        if index % cfg.fold_count == fold - 1
    }
    profile_tasks = set(task_ids) - eval_tasks
    eval_samples = [
        row for task_id in sorted(eval_tasks) for row in samples_by_task[task_id]
    ]
    profile_samples = [
        row for task_id in sorted(profile_tasks) for row in samples_by_task[task_id]
    ]
    screen = screen_transparency(
        _chains_from_samples(profile_samples, command_field=cfg.command_field),
        cfg.screen,
    )
    eval_rows = [sample.to_json_obj() for sample in eval_samples]
    profile_rows = [sample.to_json_obj() for sample in profile_samples]
    arm_kwargs = dict(
        costs_ms=cfg.costs_ms,
        guard_ms=cfg.guard_ms,
        inner_folds=cfg.inner_folds,
        min_profile_tasks=cfg.min_profile_tasks,
        command_field=cfg.command_field,
        max_prefix_depth=cfg.max_prefix_depth,
        restore_cost_fraction=cfg.restore_cost_fraction,
    )
    p0 = _run_arm(
        eval_rows,
        profile_rows,
        min_tool_history=cfg.cert_min_tool_history,
        skip_leading_cd=cfg.cert_skip_leading_cd,
        transparent_wrappers=frozenset(),
        **arm_kwargs,
    )
    pnew = _run_arm(
        eval_rows,
        profile_rows,
        min_tool_history=cfg.pnew_min_tool_history,
        skip_leading_cd=False,
        transparent_wrappers=screen.applied,
        **arm_kwargs,
    )
    oracle = _run_arm(
        eval_rows,
        profile_rows,
        min_tool_history=cfg.pnew_min_tool_history,
        skip_leading_cd=True,
        transparent_wrappers=frozenset(),
        **arm_kwargs,
    )
    return fold, screen, _merge_fold_arms(p0, pnew, oracle, fold=f"f{fold}")


def run_stage2(
    manifest_path: Path,
    *,
    screen_cfg: ScreenConfig,
    pnew_min_tool_history: int,
    restore_cost_fraction: float,
    replicates: int,
    confidence_level: float,
    seed: int,
    limit_tasks: int | None,
    final: bool,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the Stage-2 replay and return the full result payload."""
    workers = resolve_worker_count(workers)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = _read_manifest(manifest_path.resolve(), repo_root=repo_root)
    trace_root = Path(manifest["trace_root"])
    task_ids_path = Path(manifest["task_ids_file"])
    costs_ms = list(manifest["costs_ms"])
    fold_count = manifest["fold_count"]
    inner_folds = manifest["inner_folds"]
    guard_ms = manifest["guard_ms"]
    command_field = manifest["command_field"]
    max_prefix_depth = manifest["max_prefix_depth"]
    min_profile_tasks = manifest["min_profile_tasks"]
    cert_min_tool_history = manifest["min_tool_history"]
    cert_skip_leading_cd = manifest["skip_leading_cd"]

    trace_paths = discover_trace_files([trace_root])
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {trace_root}")
    task_by_trace = _require_explicit_trace_task_ids(trace_paths)
    samples = extract_many_tool_latency_samples(trace_paths)
    samples_by_task: dict[str, list[ToolLatencySample]] = {}
    for sample in samples:
        expected = task_by_trace.get(str(Path(sample.source_trace).resolve()))
        if expected is None or sample.task_id != expected:
            raise ValueError(
                "extracted sample task_id differs from explicit trace metadata: "
                f"{sample.source_trace}: {sample.task_id!r} != {expected!r}"
            )
        samples_by_task.setdefault(sample.task_id, []).append(sample)

    task_ids = _read_task_ids(task_ids_path)
    if len(task_ids) != manifest["expected_task_count"]:
        raise ValueError(
            "manifest expected_task_count differs from frozen task_ids_file: "
            f"{manifest['expected_task_count']} != {len(task_ids)}"
        )
    if set(samples_by_task) != set(task_ids):
        raise ValueError(
            "extracted logical tasks differ from frozen task_ids_file: "
            f"missing={sorted(set(task_ids) - set(samples_by_task))}, "
            f"unexpected={sorted(set(samples_by_task) - set(task_ids))}"
        )
    if limit_tasks is not None:
        if final:
            raise ValueError("--limit-tasks is a smoke knob; not allowed with --final")
        if limit_tasks < fold_count:
            raise ValueError(f"--limit-tasks must be >= fold_count ({fold_count})")
        task_ids = task_ids[:limit_tasks]
    fold_cfg = _FoldReplayConfig(
        fold_count=fold_count,
        screen=screen_cfg,
        costs_ms=tuple(float(cost) for cost in costs_ms),
        guard_ms=guard_ms,
        inner_folds=inner_folds,
        min_profile_tasks=min_profile_tasks,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        restore_cost_fraction=restore_cost_fraction,
        cert_min_tool_history=cert_min_tool_history,
        cert_skip_leading_cd=cert_skip_leading_cd,
        pnew_min_tool_history=pnew_min_tool_history,
    )
    worker = functools.partial(
        _run_fold_replay,
        samples_by_task=samples_by_task,
        task_ids=task_ids,
        cfg=fold_cfg,
    )
    folds = range(1, fold_count + 1)
    if workers == 1:
        per_fold = list(map(worker, folds))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, fold_count)) as pool:
            per_fold = list(pool.map(worker, folds))
    merged: list[dict[str, Any]] = []
    fold_screens: list[tuple[int, ScreenResult]] = []
    for fold, screen, rows in per_fold:
        fold_screens.append((fold, screen))
        merged.extend(rows)

    cert_kwargs = dict(
        costs_ms=costs_ms,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        restore_cost_fraction=restore_cost_fraction,
    )
    pnew_vs_p0 = _certificate(
        merged,
        treatment_field="pnew_trigger_ms",
        baseline_field="p0_trigger_ms",
        **cert_kwargs,
    )
    oracle_vs_p0 = _certificate(
        merged,
        treatment_field="oracle_trigger_ms",
        baseline_field="p0_trigger_ms",
        **cert_kwargs,
    )
    pnew_divergence = _divergence_by_cost(
        merged, treatment_field="pnew_trigger_ms", baseline_field="p0_trigger_ms"
    )
    oracle_divergence = _divergence_by_cost(
        merged, treatment_field="oracle_trigger_ms", baseline_field="p0_trigger_ms"
    )
    screen_consistency = _screen_consistency(fold_screens)
    verdict = compute_verdict(
        pnew_vs_p0,
        pnew_divergence,
        screen_consistency,
        headline_costs_ms=_HEADLINE_COSTS_MS,
    )

    return {
        "provenance": {
            "exploratory": True,
            "final": bool(final),
            "replayed_on": "our_hardware",
            "manifest": str(manifest_path),
            "trace_root": str(trace_root),
            "collection_id": manifest["collection_id"],
            "task_count": len(task_ids),
            "limit_tasks": limit_tasks,
            "workers": workers,
            "git_sha": _git_sha(),
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        "config": {
            "fold_count": fold_count,
            "inner_folds": inner_folds,
            "guard_ms": guard_ms,
            "command_field": command_field,
            "max_prefix_depth": max_prefix_depth,
            "min_profile_tasks": min_profile_tasks,
            "p0_min_tool_history": cert_min_tool_history,
            "p0_skip_leading_cd": cert_skip_leading_cd,
            "pnew_min_tool_history": pnew_min_tool_history,
            "restore_cost_fraction": restore_cost_fraction,
            "costs_ms": costs_ms,
            "headline_costs_ms": list(_HEADLINE_COSTS_MS),
            "bootstrap": {
                "replicates": replicates,
                "confidence_level": confidence_level,
                "seed": seed,
                "permutation_draws": replicates,
            },
            "screen": {
                "tolerance": screen_cfg.tolerance,
                "min_tasks": screen_cfg.min_tasks,
                "min_chains": screen_cfg.min_chains,
                "inner_folds": screen_cfg.inner_folds,
                "min_evidence": screen_cfg.min_evidence,
            },
        },
        "screen_consistency": screen_consistency,
        "divergence": {
            "pnew_vs_p0_by_cost": {str(k): v for k, v in sorted(pnew_divergence.items())},
            "oracle_vs_p0_by_cost": {
                str(k): v for k, v in sorted(oracle_divergence.items())
            },
        },
        "certificate_pnew_vs_p0": pnew_vs_p0,
        "certificate_oracle_vs_p0": oracle_vs_p0,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _banner(final: bool) -> str:
    if final:
        return "FINAL - complete corpus"
    return (
        "PARTIAL - not final: validated against a subset; numbers are for "
        "script validation only, not findings."
    )


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(results: dict[str, Any]) -> str:
    prov = results["provenance"]
    cfg = results["config"]
    verdict = results["verdict"]
    screen = results["screen_consistency"]
    lines: list[str] = []
    lines.append("# WTN Stage-2: certified decision replay (joint policy vs cert)")
    lines.append("")
    lines.append(f"> **{_banner(prov['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Original-trace replay on our own hardware "
        f"({prov['replayed_on']}); collection {prov['collection_id']}. "
        f"Generated {prov['generated']} (git {prov['git_sha']})."
    )
    lines.append("")
    lines.append(f"**Verdict: {verdict['verdict']}**")
    lines.append("")
    if verdict["killed_reasons"]:
        for reason in verdict["killed_reasons"]:
            lines.append(f"- KILL: {reason}")
        lines.append("")
    lines.append(
        f"Arms at rho={cfg['restore_cost_fraction']}: P_0 "
        f"(min_tool_history={cfg['p0_min_tool_history']}, no normalization), "
        f"P_new (min_tool_history={cfg['pnew_min_tool_history']}, screen-learned "
        f"normalization), oracle (cd-only strip, min_tool_history="
        f"{cfg['pnew_min_tool_history']}; reported, never certified). "
        f"{prov['task_count']} tasks, {cfg['fold_count']} folds."
    )
    lines.append("")

    lines.append("## Screen consistency (K2 on original durations)")
    lines.append("")
    lines.append(
        f"cd is a candidate: {screen['candidate_present']}; cd transparent every "
        f"fold: {screen['cd_transparent_every_fold']}."
    )
    lines.append("")
    lines.append("| fold | degenerate | marginal transparent | applied transparent |")
    lines.append("| --- | --- | --- | --- |")
    for row in screen["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['degenerate']} | "
            f"{row['marginal_transparent']} | {row['applied_transparent']} |"
        )
    lines.append("")

    lines.append("## Headline permutation cells (P_new vs P_0)")
    lines.append("")
    lines.append(
        "Positive = P_new beats P_0 (task-clustered sign-flip permutation, "
        f"Bonferroni over {len(cfg['costs_ms'])} costs)."
    )
    lines.append("")
    lines.append(
        "| kv cost | label | p(positive) | p(harmful) | paired delta ms | diverged samples |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cell in verdict["headline_cells"]:
        lines.append(
            f"| {cell['kv_cost_ms']:.0f} | {cell['permutation_label']} | "
            f"{_fmt(cell['permutation_p_positive'])} | "
            f"{_fmt(cell['permutation_p_harmful'])} | "
            f"{_fmt(cell['paired_delta_ms'])} | {cell['diverged_sample_count']} |"
        )
    lines.append("")

    lines.append("## Oracle row (cd-only + me5 vs P_0; reported, never certified)")
    lines.append("")
    oracle_points = results["certificate_oracle_vs_p0"]["points"]
    oracle_div = results["divergence"]["oracle_vs_p0_by_cost"]
    lines.append("| kv cost | label | paired delta ms | diverged samples |")
    lines.append("| --- | --- | --- | --- |")
    for cost in cfg["headline_costs_ms"]:
        point = oracle_points[str(cost)]
        lines.append(
            f"| {cost:.0f} | {point['permutation_label']} | "
            f"{_fmt(point['paired_delta_ms'])} | {oracle_div.get(str(cost), 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
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
    parser.add_argument("--screen-tolerance", type=float, default=0.05)
    parser.add_argument("--screen-min-tasks", type=int, default=5)
    parser.add_argument("--screen-min-chains", type=int, default=20)
    parser.add_argument("--screen-inner-folds", type=int, default=5)
    parser.add_argument(
        "--pnew-min-tool-history",
        type=int,
        default=_PNEW_MIN_TOOL_HISTORY_DEFAULT,
        help="P_new node evidence gate (pre-registered = 5); P_0 uses the "
        "manifest's frozen cert value.",
    )
    parser.add_argument("--restore-cost-fraction", type=float, default=0.94)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=resolve_worker_count(),
        help="CPU processes for independent folds; 1 is sequential.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/offline/wrapper-transparency-stage2-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    screen_cfg = ScreenConfig(
        tolerance=args.screen_tolerance,
        min_tasks=args.screen_min_tasks,
        min_chains=args.screen_min_chains,
        inner_folds=args.screen_inner_folds,
    )
    results = run_stage2(
        args.manifest,
        screen_cfg=screen_cfg,
        pnew_min_tool_history=args.pnew_min_tool_history,
        restore_cost_fraction=args.restore_cost_fraction,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        limit_tasks=args.limit_tasks,
        final=args.final,
        workers=args.workers,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(results), encoding="utf-8")

    verdict = results["verdict"]
    print(f"verdict={verdict['verdict']}")
    if verdict["killed_reasons"]:
        for reason in verdict["killed_reasons"]:
            print(f"  KILL: {reason}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
