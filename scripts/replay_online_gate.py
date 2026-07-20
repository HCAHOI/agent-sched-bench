#!/usr/bin/env python3
"""W3-4 -- replay-validate the anytime-valid online gate against the offline one.

Roadmap block W3-4 (``analysis/ROADMAP-mlsys2027-20260717.md``): "online gate:
confidence sequence over task-clustered paired regret; replay-validate against
the offline permutation gate". Serves C2 in ``analysis/PAPER-SKELETON-20260720.md``
-- certification decides what conditioning ships.

WHAT THIS DOES AND DOES NOT SHOW. It shows that an anytime-valid gate applied to
the frozen cross-fitted deltas reproduces the offline verdict, and reaches it
after N tasks rather than only at the end of the corpus. It does NOT show a
prospective deployment. The deltas are cross-fitted: fold ``f``'s eval tasks are
scored against a prior fitted on the OTHER folds, so the delta for the task at
replay position 0 was produced using tasks that arrive later in that same replay
order. Task ``i``'s delta is therefore not computable at the moment task ``i``
arrives. The type-I guarantee is unaffected -- the gate's bets are predictable
with respect to the delta sequence it is fed, under the same sign-symmetry null
on the same cross-fitted quantity the offline gate already assumes -- but the
"could have been decided online" reading requires a PREQUENTIAL variant (prior
frozen on a burn-in prefix, never refit on later tasks). That is future work and
is not claimed here.

What is compared. For each corpus and each kv cell, the identical cross-fitted
per-task paired deltas (SHIPPED gated-robust clock minus the robust-clock
baseline, at the certified operating point) are handed to two gates:

* OFFLINE: ``paired_task_cluster_bootstrap`` with the task-clustered sign-flip
  permutation certificate, Bonferroni over the full cost family. One verdict,
  at the end of the corpus. This is the existing certified machinery, called
  unmodified.
* ONLINE: ``replay_online_gate`` -- a sign-symmetry test martingale with
  predictable bets over the running mean of the same deltas, read after EVERY
  task, at the SAME Bonferroni one-sided family tail ``alpha / (2m)``. Both
  gates therefore test the same null at the same level; only the stopping rule
  differs, which is precisely what the replay is meant to isolate.

and the table reports agreement, detection lag (tasks until the online gate
first certifies, against the offline gate's single end-of-corpus verdict) and
revocation lag (tasks between a certification and its revocation).

Reading the agreement column. The kv cells are NOT independent: they are the
same tasks re-priced at ten cost scalings, so their deltas are near-collinear
and the cells tend to agree or disagree en bloc. "10/10" is closer to one
effective observation than to ten, and the artifact reports the DIRECTION of any
disagreement, which is the informative part. The tests are level-matched but
not nested: the offline gate thresholds a terminal sign-flip total, while the
online gate thresholds an order-dependent adaptive e-process. Either can call a
cell the other leaves inconclusive, so disagreements are reported neutrally
unless an independent invariant fails.

Task order. A frozen corpus has no arrival order -- collection ran tasks
concurrently -- so no ordering can claim to be "the" natural one. The
deterministic reference order is Unicode-lexicographic task id, matching the
certified cross-fit convention. Because an anytime gate's detection lag depends
on order, ``--order-seeds`` replays those sorted deltas under seeded
permutations and the artifact reports the lag DISTRIBUTION across seeds. Read
the seed spread, not the reference-order trajectory, as the lag result.

Corpus provenance. Every corpus pins a frozen task-id list, so "what was
analysed" never means "whatever is in the directory today". The strong form is a
committed manifest (trace_root + task_ids_file + expected_task_count +
freshness attestation); the weak form is a trace root plus a task-id list
recovered from a committed analysis, which the artifact labels as weaker.

``excluded_trace_roots`` is SCOPED to the manifest carrying it: it names data
withheld from THAT corpus for being development-exposed. Being on such a list is
what marks a root as development data, not what makes it unusable -- treating it
as a global ban would forbid development-corpus comparison, i.e. this entire
deliverable. The rule with teeth is supersession
(``_reject_superseded_roots``): reading a raw root for a benchmark that already
has a manifest-defined corpus is how a superseded 50-trace SWE-ReBench root was
once used in place of the canonical 100-task one.

Protocol. Every corpus is scored under ONE frozen protocol config (fold counts,
prefix keying, cost panel, guard, rho) read from the fresh-corpus manifest, so
only the corpus changes between rows. No estimator is modified and no constant
is per-dataset. The gate's level is inherited from the offline gate rather than
chosen here, and its bets are predictable by construction, so there is nothing
in the online gate that could be tuned to flatter the agreement column.

Non-final runs score only the two development corpora and never open
fresh-277 outcomes. ``--final`` adds the pinned fresh-277 certified-reference
row. Emits JSON + MD to ``analysis/`` (``-PARTIAL`` unless ``--final``), plus a
gitignored zstd per-task decisions sidecar.

Usage (full corpus -- run by the main session, not the smoke):
  uv run python scripts/replay_online_gate.py --final
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import datetime as _dt
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

# Allow direct `python scripts/replay_online_gate.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.adjudicate_k2_recheck import (  # noqa: E402
    _CERT_GUARD_MS,
    _CERT_RESTORE_COST_FRACTION,
    _banner,
    _git_sha,
    _load_manifest_corpus,
)
from scripts.analyze_prerestore_accounting import (  # noqa: E402
    _write_decisions_zst,
)
from scripts.run_offline_gated_robust_confirmation import (  # noqa: E402
    _read_manifest,
    _require_explicit_trace_task_ids,
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
from trace_collect.tool_latency_online_gate import (  # noqa: E402
    BET_RULE,
    CERTIFIED,
    CONTINUE,
    HARMFUL,
    replay_online_gate,
)

# Incremental C2 conditioning comparison: the offline-gated robust clock versus
# the ungated robust clock. This is NOT the banked H1 policy-vs-deadline pair.
_BASELINE_TRIGGER_FIELD = "robust_trigger_ms"
_TREATMENT_TRIGGER_FIELD = "offline_gated_robust_trigger_ms"
_FROZEN_MANIFEST = Path(
    "analysis/fresh-corpus-certification-20260717/offline-gated-robust/manifest.json"
)
_CERTIFIED_REFERENCE_TASK_COUNT = 277

# Offline permutation label -> online gate label, so "agreement" is well defined.
_OFFLINE_TO_ONLINE = {
    "positive": CERTIFIED,
    "harmful": HARMFUL,
    "inconclusive": CONTINUE,
}


@dataclass(frozen=True)
class DevCorpus:
    """A development corpus and the provenance backing its task set.

    Exactly one of ``manifest`` or ``trace_root`` is set. A manifest pins
    trace_root + task_ids_file + expected_task_count and is the stronger form.
    A ``trace_root`` corpus must still name a ``task_ids_file`` so the task set
    is a frozen list rather than "whatever is on disk today", but it carries no
    expected-count attestation or freshness record -- record that weakness in
    the artifact rather than letting it pass as equivalent provenance.
    """

    manifest: Path | None = None
    trace_root: Path | None = None
    task_ids_file: Path | None = None
    provenance_note: str = ""

    def __post_init__(self) -> None:
        if (self.manifest is None) == (self.trace_root is None):
            raise ValueError("set exactly one of manifest or trace_root")
        if self.trace_root is not None and self.task_ids_file is None:
            raise ValueError("a trace_root corpus must pin a task_ids_file")


# The development corpora. ScienceAgentBench is RETIRED (benchmark removed,
# corpus deleted, roadmap updated in e3b2d56).
#
# On excluded_trace_roots: that list is SCOPED to the manifest carrying it -- it
# names data withheld from THAT corpus because it is development-exposed. Both
# roots below appear on the fresh-certification manifest's list precisely
# BECAUSE they are development corpora; that is what makes them dev data, not
# what makes them unusable. A blanket "never read an excluded root" would forbid
# development-corpus comparison altogether, i.e. this entire deliverable. The
# rule that does bite is supersession (see _reject_superseded_roots).
_DEV_CORPORA = {
    "swe-rebench-100": DevCorpus(
        manifest=Path(
            "analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100"
            "-20260713/manifest.json"
        ),
    ),
    "terminal-bench": DevCorpus(
        trace_root=Path("traces/terminal-bench/zai-org-GLM-5.2/20260709T171830"),
        # No TB manifest exists anywhere in analysis/. The task set is instead
        # pinned to the 5-fold eval partition of the committed TB frontier
        # analysis -- the union of f{1..5}_eval.txt, 83 ids, which is exactly
        # the set every committed TB result was computed on. Verified identical
        # to what extraction from the root yields today, so pinning it changes
        # no number; it converts a silent dependency on directory contents into
        # a checked one. The root holds 100 traces: 17 contribute no tool
        # latency samples, which is why 83 and not 100 is the corpus.
        task_ids_file=Path("analysis/tool-time-frontier-terminal-bench-20260715/folds"),
        provenance_note=(
            "WEAKER PROVENANCE than swe-rebench-100: no committed manifest, so "
            "no expected_task_count or freshness attestation. Task set pinned "
            "to the union of f{1..5}_eval.txt from the committed TB frontier "
            "analysis (83 ids), the set every committed TB result used."
        ),
    ),
}

# --final may not ship at a weakened statistical discipline (the pattern from
# analyze_prior_calibration._require_certified_discipline). The certified values
# ARE the argparse defaults, so there is no second copy to drift.
_CERTIFIED_KNOBS = (
    "replicates",
    "confidence_level",
    "seed",
    "order_seeds",
)


@dataclass(frozen=True)
class ReplayConfig:
    fold_count: int
    inner_folds: int
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
    order_seeds: tuple[int, ...]


# --------------------------------------------------------------------------- #
# Corpus loading.
# --------------------------------------------------------------------------- #
def committed_manifest_trace_roots(repo_root: Path) -> dict[Path, Path]:
    """``{trace_root: manifest_path}`` over every committed manifest."""

    roots: dict[Path, Path] = {}
    for manifest_path in sorted((repo_root / "analysis").rglob("manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = payload.get("trace_root")
        if raw is None:
            continue
        path = Path(raw)
        roots[(path if path.is_absolute() else repo_root / path).resolve()] = (
            manifest_path
        )
    return roots


def read_pinned_task_ids(task_ids_file: Path) -> list[str]:
    """Frozen task-id list, from a file or from a fold directory's eval split.

    A directory is read as the ``f*_eval.txt`` partition a committed analysis
    left behind: their union is the task set that analysis ran on. Duplicates
    across folds mean it is not a partition and the list cannot be trusted, so
    that raises rather than silently deduplicating.
    """

    if task_ids_file.is_dir():
        fold_files = sorted(task_ids_file.glob("f*_eval.txt"))
        if not fold_files:
            raise ValueError(f"no f*_eval.txt fold splits under {task_ids_file}")
        task_ids: list[str] = []
        for fold_file in fold_files:
            task_ids.extend(fold_file.read_text(encoding="utf-8").split())
        duplicates = len(task_ids) - len(set(task_ids))
        if duplicates:
            raise ValueError(
                f"{task_ids_file} fold eval splits overlap by {duplicates} ids; "
                "they are not a partition, so the task set is ambiguous"
            )
        return sorted(task_ids)
    return sorted(task_ids_file.read_text(encoding="utf-8").split())


def load_pinned_root_corpus(
    trace_root: Path, task_ids_file: Path
) -> tuple[dict[str, list[ToolLatencySample]], list[str], int]:
    """Load a corpus from a trace root, pinned to a frozen task-id list.

    For a corpus with no committed manifest. The frozen list is the provenance:
    if the root ever gains, loses, or renames a task, the extracted set stops
    matching and this raises instead of quietly analysing a different corpus.
    """

    trace_paths = discover_trace_files([trace_root])
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {trace_root}")
    task_by_trace = _require_explicit_trace_task_ids(trace_paths)
    samples_by_task: dict[str, list[ToolLatencySample]] = defaultdict(list)
    for sample in extract_many_tool_latency_samples(trace_paths):
        expected = task_by_trace.get(str(Path(sample.source_trace).resolve()))
        if expected is None or sample.task_id != expected:
            raise ValueError(
                "extracted sample task_id differs from explicit trace metadata: "
                f"{sample.source_trace}: {sample.task_id!r} != {expected!r}"
            )
        samples_by_task[sample.task_id].append(sample)

    pinned = read_pinned_task_ids(task_ids_file)
    extracted = sorted(samples_by_task)
    if extracted != pinned:
        raise ValueError(
            f"{trace_root} no longer matches its pinned task list "
            f"{task_ids_file}: missing={sorted(set(pinned) - set(extracted))[:5]}, "
            f"unexpected={sorted(set(extracted) - set(pinned))[:5]}"
        )
    return dict(samples_by_task), pinned, len(trace_paths)


def _reject_superseded_roots(trace_roots: dict[str, Path], repo_root: Path) -> None:
    """Refuse a raw root when a manifest-defined corpus supersedes it.

    The scoped rule (CLAUDE.md, 037d81c). ``excluded_trace_roots`` is scoped to
    the manifest carrying it and says "development-exposed", not "forbidden", so
    global exclusion membership is NOT the test -- it would reject every
    development corpus and with it this whole deliverable. What is genuinely
    wrong is reading a raw root for a benchmark+model that already HAS a
    curated, manifest-defined corpus: that is how a superseded 50-trace
    SWE-ReBench root got used in place of the canonical 100-task one.

    Applies only to corpora configured by raw root. A manifest-defined corpus is
    canonical by construction and cannot be superseded by a sibling manifest
    (the fresh-277 and SWE-100 corpora legitimately share a parent directory).
    """

    canonical = committed_manifest_trace_roots(repo_root)
    offenders: dict[str, tuple[Path, Path]] = {}
    for name, root in trace_roots.items():
        parent = (repo_root / root).resolve().parent
        for canonical_root, manifest_path in canonical.items():
            if (
                canonical_root.parent == parent
                and canonical_root != (repo_root / root).resolve()
            ):
                offenders[name] = (root, manifest_path)
                break
    if offenders:
        detail = ", ".join(
            f"{name} -> {root} superseded by {manifest}"
            for name, (root, manifest) in sorted(offenders.items())
        )
        raise ValueError(
            "dev corpus reads a raw trace root where a manifest-defined corpus "
            f"already exists for that benchmark: {detail}. Point the entry at "
            "that manifest instead of the raw root."
        )


# --------------------------------------------------------------------------- #
# Cross-fitted decisions (the certified pipeline, called unmodified).
# --------------------------------------------------------------------------- #
def cross_fitted_decisions(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: ReplayConfig,
) -> list[dict[str, Any]]:
    """Per (call, cost) gated-robust decisions over the outer folds.

    Same outer-fold construction as the certified confirmation and
    ``analyze_prerestore_accounting``: fold ``f`` evaluates the tasks at stride
    positions ``f - 1`` and profiles on the rest, so no task is ever scored
    against a prior fitted on itself.
    """

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
        eval_rows = [
            sample.to_json_obj()
            for task_id in sorted(eval_tasks)
            for sample in samples_by_task[task_id]
        ]
        result = evaluate_offline_probe_clock(
            eval_rows,
            profile_rows=profile_rows,
            kv_costs_ms=cfg.costs_ms,
            guard_ms=cfg.guard_ms,
            inner_folds=cfg.inner_folds,
            min_tool_history=cfg.min_tool_history,
            min_profile_tasks=cfg.min_profile_tasks,
            command_field=cfg.command_field,
            max_prefix_depth=cfg.max_prefix_depth,
            skip_leading_cd=cfg.skip_leading_cd,
            restore_cost_fraction=cfg.restore_cost_fraction,
        )
        for row in result["decisions"]:
            decisions.append({**row, "outer_fold": f"f{fold}"})
    return decisions


# --------------------------------------------------------------------------- #
# The two gates, on identical deltas.
# --------------------------------------------------------------------------- #
def _ordered_task_deltas(
    offline: dict[str, Any], task_ids: Sequence[str], cost: float
) -> np.ndarray:
    """Per-task deltas at one kv cell, in Unicode-lexicographic task-id order."""

    by_task = {
        str(entry["task_id"]): entry["paired_delta_ms_by_cost"][str(cost)]
        for entry in offline["task_contributions"]
    }
    missing = [task_id for task_id in task_ids if task_id not in by_task]
    if missing:
        raise ValueError(f"offline contributions omit tasks {missing[:5]}")
    return np.asarray([float(by_task[task_id]) for task_id in task_ids], dtype=float)


def _order_sensitivity(
    deltas: np.ndarray, *, one_sided_tail: float, seeds: Sequence[int]
) -> dict[str, Any]:
    """Detection / revocation lag under seeded permutations of the task order."""

    certify_lags: list[int] = []
    harmful_lags: list[int] = []
    never_harmful = 0
    never_certified = 0
    revoked_seeds = 0
    lifecycle_labels: list[str] = []
    instantaneous_labels: list[str] = []
    for seed in seeds:
        rng = np.random.Generator(np.random.PCG64(seed))
        permuted = deltas[rng.permutation(deltas.size)]
        replay = replay_online_gate(permuted, one_sided_tail=one_sided_tail)
        if replay["first_certified_at_task"] is None:
            never_certified += 1
        else:
            certify_lags.append(int(replay["first_certified_at_task"]))
        if replay["first_harmful_at_task"] is None:
            never_harmful += 1
        else:
            harmful_lags.append(int(replay["first_harmful_at_task"]))
        revoked_seeds += int(replay["revoked"])
        lifecycle_labels.append(str(replay["final_lifecycle_label"]))
        instantaneous_labels.append(str(replay["instantaneous_final_label"]))
    return {
        "seeds": list(seeds),
        "never_certified_seed_count": never_certified,
        "first_certified_at_task": {
            "min": min(certify_lags) if certify_lags else None,
            "median": float(np.median(certify_lags)) if certify_lags else None,
            "max": max(certify_lags) if certify_lags else None,
        },
        "never_harmful_seed_count": never_harmful,
        "first_harmful_at_task": {
            "min": min(harmful_lags) if harmful_lags else None,
            "median": float(np.median(harmful_lags)) if harmful_lags else None,
            "max": max(harmful_lags) if harmful_lags else None,
        },
        "revoked_seed_count": revoked_seeds,
        # Lifecycle verdicts, not instantaneous evidence, decide whether a
        # deployed policy remains active after an uncalibrated lapse.
        "final_lifecycle_label_counts": {
            label: lifecycle_labels.count(label)
            for label in sorted(set(lifecycle_labels))
        },
        "instantaneous_final_label_counts": {
            label: instantaneous_labels.count(label)
            for label in sorted(set(instantaneous_labels))
        },
    }


def compare_gates(
    decisions: Sequence[dict[str, Any]],
    task_ids: Sequence[str],
    cfg: ReplayConfig,
) -> dict[str, Any]:
    """Offline permutation verdict vs online CS verdict, per kv cell."""

    offline = paired_task_cluster_bootstrap(
        decisions,
        costs_ms=cfg.costs_ms,
        replicates=cfg.replicates,
        confidence_level=cfg.confidence_level,
        seed=cfg.seed,
        baseline_trigger_field=_BASELINE_TRIGGER_FIELD,
        treatment_trigger_field=_TREATMENT_TRIGGER_FIELD,
        restore_cost_fraction=cfg.restore_cost_fraction,
        permutation_draws=cfg.replicates,
    )
    family_size = len(cfg.costs_ms)
    # Identical to the offline gate's Bonferroni one-sided family tail
    # (`alpha / (2m)` in _permutation_simultaneous_labels), so the two gates
    # carry the same simultaneous guarantee and agreement is like-for-like.
    one_sided_tail = (1.0 - cfg.confidence_level) / (2.0 * family_size)

    cells: list[dict[str, Any]] = []
    for cost in cfg.costs_ms:
        deltas = _ordered_task_deltas(offline, task_ids, cost)
        replay = replay_online_gate(deltas, one_sided_tail=one_sided_tail)
        offline_label = str(offline["points"][str(cost)]["permutation_label"])
        instantaneous_label = str(replay["instantaneous_final_label"])
        lifecycle_label = str(replay["final_lifecycle_label"])
        # Tasks whose paired delta is EXACTLY zero contribute nothing to either
        # gate: the policies took the same action on every call in them. They
        # inflate the nominal task count without adding evidence, so the
        # effective n -- and hence whether an e-value of 1/tail is reachable at
        # all -- is the nonzero count.
        nonzero = int(np.count_nonzero(deltas))
        cells.append(
            {
                "kv_cost_ms": cost,
                "task_count": int(deltas.size),
                "effective_task_count": nonzero,
                "zero_delta_task_count": int(deltas.size) - nonzero,
                "total_delta_ms": float(np.sum(deltas)),
                "offline_permutation_label": offline_label,
                "offline_permutation_p_positive": float(
                    offline["points"][str(cost)]["permutation_p_positive"]
                ),
                "offline_permutation_p_harmful": float(
                    offline["points"][str(cost)]["permutation_p_harmful"]
                ),
                "online_instantaneous_final_label": instantaneous_label,
                "online_lifecycle_label": lifecycle_label,
                "agree": _OFFLINE_TO_ONLINE[offline_label] == lifecycle_label,
                "online_first_certified_at_task": replay["first_certified_at_task"],
                "online_first_harmful_at_task": replay["first_harmful_at_task"],
                "online_ever_certified": replay["ever_certified"],
                # Ville-controlled revocation (certified, then crossed to
                # harmful) vs the uncalibrated lapse -- see
                # tool_latency_online_gate.Lapse.
                "online_revoked": replay["revoked"],
                "online_revoked_at_task": replay["revoked_at_task"],
                "online_revocation_lag_tasks": replay["revocation_lag_tasks"],
                "online_lapse_count": replay["lapse_count"],
                "online_lapses": replay["lapses"],
                "online_max_lapse_lag_tasks": replay["max_lapse_lag_tasks"],
                "online_log_e_positive": replay["final_log_e_positive"],
                "online_log_e_harmful": replay["final_log_e_harmful"],
                "online_log_e_threshold": float(np.log(1.0 / one_sided_tail)),
                # Proven upper bound: the first nonzero delta receives a zero
                # bet, and each later nonzero delta contributes strictly less
                # than log 2. Falling at or below the threshold proves
                # certification impossible; clearing it proves nothing about
                # reachability under the predictable truncated bets.
                "online_log_e_upper_bound": float(max(nonzero - 1, 0) * np.log(2.0)),
                "online_certification_ruled_out_by_upper_bound": bool(
                    max(nonzero - 1, 0) * np.log(2.0) <= np.log(1.0 / one_sided_tail)
                ),
                "order_sensitivity": _order_sensitivity(
                    deltas, one_sided_tail=one_sided_tail, seeds=cfg.order_seeds
                ),
                # Per-task trace: stripped into the zstd sidecar by the driver so
                # the committed JSON stays small (same split as the other lanes).
                "_replay_trace": {
                    "task_delta_ms": [float(value) for value in deltas],
                    "labels": replay["labels"],
                    "lifecycle_labels": replay["lifecycle_labels"],
                    "log_e_positive": replay["log_e_positive"],
                    "log_e_harmful": replay["log_e_harmful"],
                },
            }
        )

    agreements = [cell["agree"] for cell in cells]
    disagreements = [
        f"offline={cell['offline_permutation_label']}/"
        f"online={cell['online_lifecycle_label']}"
        for cell in cells
        if not cell["agree"]
    ]
    # Agreement is split by WHAT the two gates agreed on. Both gates saying
    # nothing is not the same evidence as both certifying, and collapsing them
    # into one count makes a corpus where nothing certified look like a
    # validated match. Only the positive/harmful rows are informative.
    concordant = {"positive": 0, "harmful": 0, "null": 0}
    for cell in cells:
        if not cell["agree"]:
            continue
        label = cell["offline_permutation_label"]
        concordant["null" if label == "inconclusive" else label] += 1
    return {
        "alpha_family": 1.0 - cfg.confidence_level,
        "one_sided_tail": one_sided_tail,
        "family_size": family_size,
        "task_count": int(offline["task_count"]),
        "sample_count": int(offline["sample_count"]),
        "agreement_count": int(sum(agreements)),
        "concordant_positive": concordant["positive"],
        "concordant_harmful": concordant["harmful"],
        "concordant_null": concordant["null"],
        # The cells are the same tasks at ten cost scalings, so this is closer
        # to one effective observation than to ten -- see the module docstring.
        "cells_are_independent": False,
        "disagreement_directions": {
            direction: disagreements.count(direction)
            for direction in sorted(set(disagreements))
        },
        "min_effective_task_count": min(cell["effective_task_count"] for cell in cells),
        "max_effective_task_count": max(cell["effective_task_count"] for cell in cells),
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# Rendering / CLI.
# --------------------------------------------------------------------------- #
def _format_lag_spread(summary: dict[str, int | float | None]) -> str:
    """Compact min/median/max task lag for one crossing direction."""

    if summary["median"] is None:
        return "never"
    return f"{summary['min']}/{summary['median']:.0f}/{summary['max']}"


def _format_crossing_sweep(
    sensitivity: dict[str, Any], *, summary_key: str, never_key: str
) -> str:
    total = len(sensitivity["seeds"])
    detected = total - int(sensitivity[never_key])
    return f"{detected}/{total}; {_format_lag_spread(sensitivity[summary_key])}"


def _format_lifecycle_counts(sensitivity: dict[str, Any]) -> str:
    counts = sensitivity["final_lifecycle_label_counts"]
    return (
        f"{counts.get(CERTIFIED, 0)}/{counts.get(HARMFUL, 0)}/{counts.get(CONTINUE, 0)}"
    )


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# W3-4 online gate vs offline permutation gate (replay)")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> Anytime-valid sign-symmetry test martingale (Ville; Ramdas et al. "
        "2023) over task-clustered paired regret, replayed against the FROZEN "
        "offline permutation gate on identical cross-fitted deltas. Both gates "
        "test the SAME null (sign symmetry of the paired deltas). "
        f"Generated {provenance['generated']} (git {provenance['git_sha']})."
    )
    lines.append("")
    lines.append(
        f"Directional bet rule: `{provenance['bet_rule']}`. The positive "
        "e-process uses the absolute magnitude of the truncated signed Kelly "
        "plug-in; the harmful e-process uses its negative."
    )
    lines.append("")
    lines.append(
        f"One-sided tail {provenance['one_sided_tail']:.4f} "
        f"(= {provenance['alpha_family']:.2f} / (2 x {provenance['family_size']} kv "
        "cells), identical to the offline gate's Bonferroni family tail; "
        f"e-value threshold {1.0 / provenance['one_sided_tail']:.0f}), rho="
        f"{provenance['restore_cost_fraction']}, guard "
        f"{provenance['guard_ms']:.0f}ms (threshold==kv)."
    )
    lines.append("")
    lines.append(
        "Comparison: treatment "
        f"`{provenance['treatment_trigger_field']}` vs baseline "
        f"`{provenance['baseline_trigger_field']}`. This incremental C2 "
        "conditioning comparison is not the banked H1 policy-vs-deadline pair."
    )
    lines.append("")
    lines.append(
        "Reference replay order: "
        f"`{provenance['task_order']}`; seeded permutations start from that order."
    )
    lines.append("")

    for corpus in results["corpora"]:
        lines.append(f"## {corpus['name']} — {corpus.get('role', 'unspecified role')}")
        lines.append("")
        if corpus.get("unavailable"):
            lines.append(
                f"**NOT RUN** -- {corpus['unavailable']}. This corpus is one the "
                "roadmap names; the deliverable is incomplete until it is "
                "restored and this row is filled."
            )
            lines.append("")
            continue
        comparison = corpus["comparison"]
        lines.append(
            f"Corpus provenance: {corpus.get('provenance', 'unknown')} -- "
            f"`{corpus['trace_root']}`"
            + (
                f", manifest `{corpus['manifest']}`, expected task count "
                f"{corpus['expected_task_count']}."
                if corpus.get("manifest")
                else f", {corpus.get('trace_file_count', '?')} trace files, task "
                f"set pinned by `{corpus.get('task_ids_file')}`."
            )
        )
        if corpus.get("provenance_note"):
            lines.append("")
            lines.append(f"> {corpus['provenance_note']}")
        lines.append("")
        lines.append(
            f"{comparison['task_count']} tasks, {comparison['sample_count']} calls. "
            f"Agreement {comparison['agreement_count']}/{comparison['family_size']} "
            f"kv cells, of which **{comparison['concordant_positive']} concordant-"
            f"positive, {comparison['concordant_harmful']} concordant-harmful, "
            f"{comparison['concordant_null']} concordant-null** (both gates said "
            "nothing)."
        )
        if comparison["concordant_positive"] + comparison["concordant_harmful"] == 0:
            lines.append("")
            lines.append(
                "> **This row has no informative concordance.** Its agreement "
                "count contains only cells where both gates declined to decide. "
                "Read the one-way `ruled out` upper-bound diagnostic and achieved "
                "`logE+`; clearing the upper bound does not prove certification "
                "was attainable."
            )
        lines.append("")
        lines.append(
            f"Effective n per cell (tasks with a NONZERO paired delta): "
            f"{comparison['min_effective_task_count']}-"
            f"{comparison['max_effective_task_count']} of "
            f"{comparison['task_count']}."
        )
        if comparison["disagreement_directions"]:
            directions = ", ".join(
                f"{direction} x{count}"
                for direction, count in comparison["disagreement_directions"].items()
            )
            lines.append("")
            lines.append(f"Disagreement directions: {directions}.")
        lines.append("")
        lines.append(
            "| kv | eff n | logE+ / logE- (thr) | upper bound | ruled out | "
            "total ms | offline | online lifecycle (instant) | agree | "
            "first + / - @task | revoked | lapses | seed + hit/n; min/med/max | "
            "seed - hit/n; min/med/max | seed final C/H/N |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- | --- | --- | --- |"
        )
        for cell in comparison["cells"]:
            sensitivity = cell["order_sensitivity"]
            positive_spread = _format_crossing_sweep(
                sensitivity,
                summary_key="first_certified_at_task",
                never_key="never_certified_seed_count",
            )
            harmful_spread = _format_crossing_sweep(
                sensitivity,
                summary_key="first_harmful_at_task",
                never_key="never_harmful_seed_count",
            )
            lifecycle_counts = _format_lifecycle_counts(sensitivity)
            first_positive = cell["online_first_certified_at_task"]
            first_harmful = cell["online_first_harmful_at_task"]
            revoke_lag = cell["online_revocation_lag_tasks"]
            lapse_lag = cell["online_max_lapse_lag_tasks"]
            lines.append(
                f"| {cell['kv_cost_ms']:.0f} | "
                f"{cell['effective_task_count']} | "
                f"{cell['online_log_e_positive']:.2f} / "
                f"{cell['online_log_e_harmful']:.2f} "
                f"({cell['online_log_e_threshold']:.2f}) | "
                f"{cell['online_log_e_upper_bound']:.1f} | "
                f"{'yes' if cell['online_certification_ruled_out_by_upper_bound'] else 'no'} | "
                f"{cell['total_delta_ms']:.0f} | "
                f"{cell['offline_permutation_label']} | "
                f"{cell['online_lifecycle_label']} "
                f"({cell['online_instantaneous_final_label']}) | "
                f"{'yes' if cell['agree'] else 'NO'} | "
                f"{first_positive if first_positive is not None else '-'} / "
                f"{first_harmful if first_harmful is not None else '-'} | "
                f"{f'yes (lag {revoke_lag})' if cell['online_revoked'] else 'no'} | "
                f"{cell['online_lapse_count']}"
                f"{f' (lag {lapse_lag})' if lapse_lag else ''} | "
                f"{positive_spread} | "
                f"{harmful_spread} |"
                f" {lifecycle_counts} |"
            )
        lines.append("")
    lines.append(
        "Detection lag is tasks until the online gate FIRST crosses in either "
        "direction; the offline gate has no analogue (it speaks once, at the "
        "end of the corpus). Seed columns report crossings/total before the "
        "conditional lag spread and final lifecycle counts as "
        "certified/harmful/continue. Read them, not the reference-order columns, "
        "as the order-sensitivity result."
    )
    lines.append("")
    lines.append(
        "**Revocation vs lapse.** A REVOCATION is a certified cell whose "
        "harmful martingale later crossed. Ville bounds that crossing under the "
        "conditional sign-symmetry null used by both gates; it is not a "
        "guarantee for every distribution with nonnegative mean. A LAPSE is a "
        "certified cell whose instantaneous evidence merely fell below the "
        "threshold. Crossing back carries no type-I guarantee and is common "
        "under sustained positive effects in the validity tests, so a lapse "
        "never withdraws the policy. Revocation lag is exposure measured from "
        "first certification, not detection latency from when truth degraded."
    )
    lines.append("")
    lines.append(
        "**Agreement is not ten independent checks.** The kv cells re-price the "
        "same tasks at ten cost scalings, so the cells are near-collinear and "
        "move together; read the count with the disagreement direction, not as "
        "a rate out of ten. And a concordant-NULL cell is not a validated "
        "match: both gates declining to decide is agreement about nothing."
    )
    lines.append("")
    lines.append(
        "**Effective n and its one-way upper bound.** A task with paired delta "
        "exactly 0.0 moves neither gate. The first nonzero delta receives a zero "
        "bet, and every later nonzero delta contributes strictly less than "
        "log 2, so `(eff n - 1) x log 2` is a proven log-e upper bound. If that "
        "bound does not clear the threshold, certification is ruled out. If it "
        "does clear, reachability remains unknown under the predictable "
        "truncated bets; only the achieved `logE+`/`logE-` is observed evidence."
    )
    lines.append("")
    lines.append(
        "**Lifecycle state vs instantaneous evidence.** `agree` uses the sticky "
        "lifecycle state: certification remains deployed through a mere lapse, "
        "and changes only when the opposite martingale crosses; a later "
        "certification can deploy again. The parenthesized table value and "
        "`online_instantaneous_final_label` retain the terminal evidence "
        "diagnostic without letting an uncalibrated lapse withdraw policy."
    )
    lines.append("")
    lines.append(
        "**Scope.** The deltas are cross-fitted, so this replay shows an "
        "anytime-valid gate reproducing the offline verdict earlier, NOT a "
        "prospective online deployment -- a prequential variant (prior frozen "
        "on a burn-in prefix) is future work."
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--order-seeds",
        type=int,
        default=50,
        help="Number of seeded task-order permutations for the lag sensitivity "
        "sweep (seeds 0..n-1).",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Smoke only: cap tasks per corpus. Rejected with --final.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _require_certified_discipline(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """``--final`` may only ship at the certified statistical discipline."""

    off = {
        name: (getattr(args, name), parser.get_default(name))
        for name in _CERTIFIED_KNOBS
        if getattr(args, name) != parser.get_default(name)
    }
    if off:
        detail = ", ".join(
            f"--{name.replace('_', '-')}={got!r} (certified {want!r})"
            for name, (got, want) in sorted(off.items())
        )
        raise ValueError(
            f"--final requires the certified discipline; got {detail}. Drop the "
            "override or drop --final."
        )


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/online-gate-replay-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def _limit(task_ids: list[str], limit_tasks: int | None, fold_count: int) -> list[str]:
    if limit_tasks is None:
        return task_ids
    if limit_tasks < fold_count:
        raise ValueError(f"--limit-tasks must be >= fold_count ({fold_count})")
    return task_ids[:limit_tasks]


def _score_corpus(
    corpus: dict[str, Any], cfg: ReplayConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one pinned corpus and split public summary from replay trace."""

    task_ids = corpus["task_ids"]
    decisions = cross_fitted_decisions(corpus["samples_by_task"], task_ids, cfg)
    comparison = compare_gates(decisions, task_ids, cfg)
    rendered = {
        key: value
        for key, value in corpus.items()
        if key not in {"samples_by_task", "task_ids"}
    } | {"task_count": len(task_ids), "comparison": comparison}
    sidecar = {
        "corpus": corpus["name"],
        "task_ids": list(task_ids),
        "cells": [
            {
                "kv_cost_ms": cell["kv_cost_ms"],
                "offline_permutation_label": cell["offline_permutation_label"],
                "offline_permutation_p_positive": cell[
                    "offline_permutation_p_positive"
                ],
                "offline_permutation_p_harmful": cell["offline_permutation_p_harmful"],
                **cell.pop("_replay_trace"),
            }
            for cell in comparison["cells"]
        ],
    }
    return rendered, sidecar


def _staging_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        return Path(handle.name)


def _write_artifacts(
    out_json: Path,
    out_md: Path,
    decisions_path: Path,
    payload: dict[str, Any],
    markdown: str,
    sidecar: list[dict[str, Any]],
) -> None:
    """Stage every artifact; publish the JSON completion anchor last."""

    if len({out_json.resolve(), out_md.resolve(), decisions_path.resolve()}) != 3:
        raise ValueError("JSON, Markdown, and decisions sidecar paths must be distinct")
    staged_json = _staging_path(out_json)
    staged_md = _staging_path(out_md)
    staged_decisions = _staging_path(decisions_path)
    try:
        staged_json.write_text(
            json.dumps(payload, indent=2, default=list), encoding="utf-8"
        )
        staged_md.write_text(markdown, encoding="utf-8")
        _write_decisions_zst(staged_decisions, sidecar)
        # Remove any prior completion anchor before publishing a new sidecar.
        # A later rename failure can then leave partial files, but never a JSON
        # anchor that falsely joins artifacts from two runs.
        out_json.unlink(missing_ok=True)
        os.replace(staged_decisions, decisions_path)
        os.replace(staged_md, out_md)
        os.replace(staged_json, out_json)
    finally:
        staged_json.unlink(missing_ok=True)
        staged_md.unlink(missing_ok=True)
        staged_decisions.unlink(missing_ok=True)


def _publish_artifacts_and_results(
    out_json: Path,
    out_md: Path,
    decisions_path: Path,
    payload: dict[str, Any],
    markdown: str,
    sidecar: list[dict[str, Any]],
    rendered: list[dict[str, Any]],
) -> None:
    """Publish artifacts first; only then disclose comparison results."""

    _write_artifacts(out_json, out_md, decisions_path, payload, markdown, sidecar)
    for corpus in rendered:
        comparison = corpus["comparison"]
        print(
            f"{corpus['name']}: agreement {comparison['agreement_count']}/"
            f"{comparison['family_size']} cells"
        )


def _preflight_fresh_pin(manifest: dict[str, Any]) -> list[str]:
    """Validate the exact fresh-277 identity before opening any trace outcome."""

    task_ids = read_pinned_task_ids(Path(manifest["task_ids_file"]))
    expected = manifest["expected_task_count"]
    if (
        expected != _CERTIFIED_REFERENCE_TASK_COUNT
        or len(task_ids) != _CERTIFIED_REFERENCE_TASK_COUNT
        or len(set(task_ids)) != _CERTIFIED_REFERENCE_TASK_COUNT
    ):
        raise ValueError(
            "--final requires the frozen "
            f"{_CERTIFIED_REFERENCE_TASK_COUNT}-task reference corpus; "
            f"manifest expects {expected} and pins {len(task_ids)} task IDs"
        )
    return task_ids


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.final:
        _require_certified_discipline(args, parser)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / _FROZEN_MANIFEST
    # Protocol metadata is safe to read in every mode; fresh outcomes are loaded
    # only after both development corpora score successfully in final mode.
    manifest = _read_manifest(manifest_path, repo_root=repo_root)
    cfg = ReplayConfig(
        fold_count=manifest["fold_count"],
        inner_folds=manifest["inner_folds"],
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
        order_seeds=tuple(range(args.order_seeds)),
    )

    corpora: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    # Guard BEFORE loading: refuse a raw root that a manifest-defined corpus
    # supersedes. Only raw-root corpora can be superseded.
    _reject_superseded_roots(
        {
            name: corpus.trace_root
            for name, corpus in _DEV_CORPORA.items()
            if corpus.trace_root is not None
        },
        repo_root,
    )
    for name, corpus in _DEV_CORPORA.items():
        if corpus.manifest is not None:
            # _load_manifest_corpus reconciles the extracted task set against the
            # manifest's task_ids_file and expected_task_count, so an unexpected
            # count raises there rather than silently changing n.
            dev_by_task, dev_task_ids, dev_manifest = _load_manifest_corpus(
                repo_root / corpus.manifest,
                limit_tasks=args.limit_tasks,
                final=args.final,
            )
            corpora.append(
                {
                    "name": name,
                    "role": "development corpus",
                    "provenance": "committed manifest",
                    "manifest": str(corpus.manifest),
                    "trace_root": dev_manifest["trace_root"],
                    "collection_id": dev_manifest["collection_id"],
                    "expected_task_count": dev_manifest["expected_task_count"],
                    "samples_by_task": dev_by_task,
                    "task_ids": dev_task_ids,
                }
            )
            continue
        assert corpus.trace_root is not None and corpus.task_ids_file is not None
        dev_by_task, dev_task_ids, trace_count = load_pinned_root_corpus(
            repo_root / corpus.trace_root,
            repo_root / corpus.task_ids_file,
        )
        corpora.append(
            {
                "name": name,
                "role": "development corpus",
                "provenance": "pinned trace root (no manifest)",
                "provenance_note": corpus.provenance_note,
                "trace_root": str(corpus.trace_root),
                "task_ids_file": str(corpus.task_ids_file),
                "trace_file_count": trace_count,
                "samples_by_task": dev_by_task,
                "task_ids": _limit(dev_task_ids, args.limit_tasks, cfg.fold_count),
            }
        )

    rendered: list[dict[str, Any]] = []
    for corpus in corpora:
        print(f"{corpus['name']}: scoring {len(corpus['task_ids'])} tasks ...")
        rendered_row, sidecar_row = _score_corpus(corpus, cfg)
        rendered.append(rendered_row)
        sidecar.append(sidecar_row)

    if args.final:
        preflight_task_ids = _preflight_fresh_pin(manifest)
        fresh_by_task, fresh_task_ids, fresh_manifest = _load_manifest_corpus(
            manifest_path, limit_tasks=args.limit_tasks, final=True
        )
        if (
            len(fresh_task_ids) != _CERTIFIED_REFERENCE_TASK_COUNT
            or fresh_manifest["expected_task_count"] != _CERTIFIED_REFERENCE_TASK_COUNT
            or fresh_task_ids != preflight_task_ids
        ):
            raise ValueError(
                "--final requires the frozen "
                f"{_CERTIFIED_REFERENCE_TASK_COUNT}-task reference corpus; "
                f"manifest expects {fresh_manifest['expected_task_count']} and "
                f"loaded {len(fresh_task_ids)}"
            )
        fresh = {
            "name": f"fresh-277 ({fresh_manifest['collection_id']})",
            "role": "reused certified-reference corpus",
            "provenance": "frozen committed manifest",
            "provenance_note": (
                "Previously used for H1 certification and later analyses; this "
                "is not newly unseen W3-4 validation data. It was not opened by "
                "development/profile runs in this lane."
            ),
            "manifest": str(_FROZEN_MANIFEST),
            "trace_root": fresh_manifest["trace_root"],
            "task_ids_file": fresh_manifest["task_ids_file"],
            "collection_id": fresh_manifest["collection_id"],
            "expected_task_count": fresh_manifest["expected_task_count"],
            "samples_by_task": fresh_by_task,
            "task_ids": fresh_task_ids,
        }
        print(f"{fresh['name']}: scoring {len(fresh_task_ids)} tasks ...")
        rendered_row, sidecar_row = _score_corpus(fresh, cfg)
        rendered.append(rendered_row)
        sidecar.append(sidecar_row)

    results = {"corpora": rendered}
    provenance = {
        "exploratory": not args.final,
        "final": bool(args.final),
        "manifest": str(_FROZEN_MANIFEST),
        "collection_id": manifest["collection_id"],
        "limit_tasks": args.limit_tasks,
        "task_order": "unicode_lexicographic_task_id",
        "guard_ms": cfg.guard_ms,
        "restore_cost_fraction": cfg.restore_cost_fraction,
        "baseline_trigger_field": _BASELINE_TRIGGER_FIELD,
        "treatment_trigger_field": _TREATMENT_TRIGGER_FIELD,
        "alpha_family": 1.0 - cfg.confidence_level,
        "family_size": len(cfg.costs_ms),
        "one_sided_tail": (1.0 - cfg.confidence_level) / (2.0 * len(cfg.costs_ms)),
        "order_seed_count": args.order_seeds,
        "cs_method": "sign_symmetry_test_martingale",
        "cs_reference": "Ville 1939; Ramdas et al. Statist. Sci. 2023",
        "bet_rule": BET_RULE,
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    decisions_path = out_json.with_name(out_json.stem + "-decisions.json.zst")
    payload = {
        "provenance": provenance,
        "config": cfg.__dict__,
        "decisions_sidecar": decisions_path.name,
        **results,
    }
    _publish_artifacts_and_results(
        out_json,
        out_md,
        decisions_path,
        payload,
        render_markdown(results, provenance),
        sidecar,
        rendered,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"wrote {decisions_path}")


if __name__ == "__main__":
    main()
