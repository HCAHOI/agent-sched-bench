#!/usr/bin/env python3
"""Candidate A Stage-1 kill test: session-state-conditioned duration CV.

Pre-registered in ``analysis/CLOSED-QUESTIONS.md`` (Candidate A,
"Session-state-conditioned nodes") INCLUDING the section "Candidate A Stage-1
amendments (pre-registered 2026-07-20)" which overrides the original sketch on
two points and is implemented verbatim here:

1. **Emergent state vocabulary.** No heavy verb is named in method logic. The
   state vocabulary is derived per fit fold: the top-k verb classes by fit-fold
   median duration above a configured action-relevance floor (``--top-k`` /
   ``--floor-ms``), plus a structural cwd-changed bit. A class enters the
   evaluated set only if it is selected in EVERY non-degenerate fit fold
   (cross-fold unanimity), so the choice of classes is honest and never a
   function of the CV-reduction outcome. The selection criterion is median
   MAGNITUDE, orthogonal to the tested quantity (variance reduction), so
   selecting on it is not the same as selecting on the outcome. (Ordering of a
   task's history is ALWAYS temporal -- never by duration or outcome; only the
   heavy-class *screen* ranks by median, as the amendment prescribes.)
2. **Dual duration-source consistency.** The test runs on BOTH duration sources
   and KILLs unless the CV reduction survives the paired task-clustered
   bootstrap on BOTH. A signal on only one source is a hardware artifact, not
   session-state information.

   * **Replayed segment corpus** (``--traces-dir``, e.g.
     ``traces/fresh-277-segtimeline``): a call is one exec chain; the observed
     duration of a verb class is the per-segment ``segment_ms`` of that atom.
     Verb classes come from segment atoms (``analyze_segment_variance.atom_key``).
   * **Original trace durations** (``--manifest``, the frozen fresh-277
     certification manifest): a call is one exec; the observed duration is the
     whole-call ``latency_ms`` attributed to the call's primary verb (the head
     of its LAST sequential segment -- the workload the leading wrappers wrap).
     Verb classes come from the production command tokenizer
     (``command_features``).

   State is computed identically on both sources from each source's OWN history.

Falsifiable claim: the same command class is a different random variable
depending on what the task already did. Test: conditioning each heavy verb
class's duration on a small session-state hash -- computed from the task's OWN
earlier calls only (information legally available at decision time) -- reduces
its cross-task coefficient of variation vs unconditional.

State-hash (per call, from the task's calls STRICTLY BEFORE it, ordered by
timestamp then action id):

* for each fit-selected top-k class: a "completed at least once before" bit;
* a count bucket (``--count-bucket-edges``) of prior completions of the single
  top-1 heavy class (pytest-run-count generalized, no token named);
* a structural cwd-changed-since-last-call bit (delegated to the PRODUCTION
  leading-cd stripper ``shell_command_prefix_tokens(skip_leading_cd=True)`` so
  ``cd`` is treated exactly as production treats it -- a generic builtin, not a
  named method class; pure ``cd X`` calls the production stripper keeps are a
  documented ceiling, not counted as a change).

Total vocabulary is asserted <= ``--max-state-bits`` (default 8) at config time.

Metrics (pre-registered): per heavy verb class and per source, unconditional
cross-task CV of per-task median durations (the ``atom_stability`` quantity)
vs conditional CV = support-floored, task-weighted mean of the WITHIN
(class, state) cross-task CV. Reduction = unconditional - conditional (positive
= state helps). Primary statistic: the task-clustered paired bootstrap CI of
the mean reduction across the selected classes. A within-class state-shuffle
negative control is reported as a diagnostic (an uninformative split must not
reduce CV) -- it strengthens the "conditioning trivially reduces CV" rebuttal
but is NOT part of the verdict, which stays exactly as pre-registered.

KILL readout: per-source SURVIVE iff the aggregate reduction CI excludes zero
(CI low > 0); combined SURVIVE iff BOTH sources survive. One-source-only signal
=> KILL.

No synthetic data. EXPLORATORY, replayed on our own hardware. Prints a PARTIAL
banner unless ``--final``.

Usage (full corpus -- run by the main session, not the smoke):
  uv run python scripts/analyze_session_state_cv.py \
    --traces-dir traces/fresh-277-segtimeline \
    --manifest analysis/fresh-corpus-certification-20260717/\
offline-gated-robust/manifest.json --final
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

import numpy as np

# Allow direct `python scripts/analyze_session_state_cv.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_segment_variance import (  # noqa: E402
    _git_sha,
    atom_key,
    build_chains,
)
from scripts.run_offline_gated_robust_confirmation import (  # noqa: E402
    _read_manifest,
    _read_task_ids,
    _require_explicit_trace_task_ids,
)
from trace_collect.command_features import (  # noqa: E402
    shell_command_prefix_tokens,
    shell_command_segments,
)
from trace_collect.tool_latency_dataset import (  # noqa: E402
    ToolLatencySample,
    discover_trace_files,
    extract_many_tool_latency_samples,
    extract_segment_latency_samples,
    extract_tool_latency_samples,
)
from trace_collect.tool_latency_offline_probe import balanced_task_folds  # noqa: E402

# The two duration sources, named once as labels only (never method logic).
_SOURCE_REPLAYED = "replayed_segment"
_SOURCE_ORIGINAL = "original_trace"


# --------------------------------------------------------------------------- #
# Call abstraction (both sources normalize onto this).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Call:
    """One tool call in a task's timeline, source-agnostic.

    ``order_key`` is a strictly-temporal sort key (timestamp then action id) --
    NEVER a function of duration or outcome, so state features read only what
    was legally knowable before the call ran.
    ``history_verbs`` are the verb classes the call exercises (for the
    "seen before" membership + count bucket); ``observations`` are the
    (verb_class, duration_ms) pairs whose cross-task CV the test measures.
    """

    task_id: str
    order_key: tuple[float, str]
    cwd_changed: bool
    history_verbs: frozenset[str]
    observations: tuple[tuple[str, float], ...]


def call_changes_cwd(command: str) -> bool:
    """Structural cwd-change bit: does the production leading-cd stripper fire?

    Delegates entirely to ``shell_command_prefix_tokens(skip_leading_cd=True)``:
    the call changes the working directory iff production would strip a leading
    ``cd`` segment from its key. No token spelling appears here -- ``cd`` is
    handled by the same production normalization the shipped prior uses, which
    documents it as a generic builtin (not a command class). Ceiling: a command
    that is ONLY ``cd X`` the production stripper keeps unchanged, so a bare
    directory change with no follow-on command reads as no-change (documented).
    """

    if not command:
        return False
    full = shell_command_prefix_tokens(command)
    stripped = shell_command_prefix_tokens(command, skip_leading_cd=True)
    return stripped != full


def _primary_verb(command: str) -> str | None:
    """Verb the whole-call duration is attributed to: head of the LAST segment.

    The last sequential segment is the workload the leading wrappers wrap
    (same spirit as leading-cd stripping); its head verb, from the production
    tokenizer, labels the call. Untokenizable commands yield no verb.
    """

    segments = shell_command_segments(command)
    return segments[-1][0] if segments and segments[-1] else None


def _segment_head_verbs(command: str) -> frozenset[str]:
    """All sequential-segment head verbs of a command (history membership)."""

    return frozenset(seg[0] for seg in shell_command_segments(command) if seg)


# --------------------------------------------------------------------------- #
# Source ingestion.
# --------------------------------------------------------------------------- #
def load_replayed_calls(traces_dir: Path, glob: str, *, limit: int | None) -> list[Call]:
    """Replayed segment corpus -> Calls (per-segment atom durations).

    Chains are ordered within a task by the parent exec's ``ts_start`` (read
    from the SAME trace files via the tool-latency extractor); atoms come from
    each segment; duration of an atom is its ``segment_ms``. Pipe/loop parents
    are excluded at extraction (``skip_concurrent=True``) per the
    duration-validity ceiling.
    """

    files = sorted(traces_dir.glob(glob))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"no trace files under {traces_dir}/{glob}")

    ts_start: dict[tuple[str, str], float] = {}
    all_segments = []
    for path in files:
        for sample in extract_tool_latency_samples(path):
            ts_start[(str(path), sample.action_id)] = sample.tool_ts_start
        all_segments.extend(
            extract_segment_latency_samples(path, skip_concurrent=True)
        )
    if not all_segments:
        raise ValueError(f"no segment latency samples under {traces_dir}/{glob}")

    chains = build_chains(all_segments)
    calls: list[Call] = []
    for chain in chains:
        key = (chain.source_trace, chain.action_id)
        order_ts = ts_start.get(key, math.inf)  # unordered chains sort last, stably
        observations = tuple(
            (seg.atom, seg.duration_ms) for seg in chain.segments
        )
        history_verbs = frozenset(seg.atom for seg in chain.segments)
        calls.append(
            Call(
                task_id=chain.task_id,
                order_key=(order_ts, chain.action_id),
                cwd_changed=call_changes_cwd(chain.parent_command),
                history_verbs=history_verbs,
                observations=observations,
            )
        )
    return calls


def load_original_calls(
    manifest_path: Path, *, limit_tasks: int | None, final: bool
) -> list[Call]:
    """Original fresh-277 traces -> Calls (whole-call latency durations).

    Loads exactly the frozen manifest task set (same guards as
    ``run_wtn_stage2``): explicit per-trace task ids, extracted-vs-declared task
    reconciliation. A call's duration is ``latency_ms`` attributed to its
    primary verb (last-segment head); its history verbs are all segment heads.
    """

    repo_root = Path(__file__).resolve().parents[1]
    manifest = _read_manifest(manifest_path.resolve(), repo_root=repo_root)
    trace_root = Path(manifest["trace_root"])
    command_field = manifest["command_field"]

    trace_paths = discover_trace_files([trace_root])
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {trace_root}")
    task_by_trace = _require_explicit_trace_task_ids(trace_paths)
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

    task_ids = _read_task_ids(Path(manifest["task_ids_file"]))
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
        task_ids = task_ids[:limit_tasks]

    calls: list[Call] = []
    for task_id in task_ids:
        for sample in samples_by_task[task_id]:
            command = ""
            if sample.tool_args is not None:
                value = sample.tool_args.get(command_field)
                if isinstance(value, str):
                    command = value
            primary = _primary_verb(command)
            observations = (
                ((primary, sample.latency_ms),) if primary is not None else ()
            )
            calls.append(
                Call(
                    task_id=task_id,
                    order_key=(sample.tool_ts_start, sample.action_id),
                    cwd_changed=call_changes_cwd(command),
                    history_verbs=_segment_head_verbs(command),
                    observations=observations,
                )
            )
    return calls


# --------------------------------------------------------------------------- #
# Fit-fold heavy-class selection (cross-fold unanimity).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SelectionConfig:
    top_k: int
    floor_ms: float
    fold_count: int
    count_bucket_edges: tuple[float, ...]
    max_state_bits: int
    min_class_tasks: int
    min_cell_tasks: int


def _class_task_medians(calls: Sequence[Call]) -> dict[str, dict[str, list[float]]]:
    """verb_class -> task -> list of observed durations (for that class)."""

    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for call in calls:
        for verb, duration in call.observations:
            out[verb][call.task_id].append(duration)
    return out


def _fold_heavy_classes(calls: Sequence[Call], cfg: SelectionConfig) -> list[str]:
    """Top-k classes by fit-fold median duration above the floor, one fold's fit."""

    by_class = _class_task_medians(calls)
    ranked: list[tuple[float, str]] = []
    for verb, task_map in by_class.items():
        task_medians = [statistics.median(v) for v in task_map.values()]
        median = statistics.median(task_medians)
        if median >= cfg.floor_ms:
            ranked.append((median, verb))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [verb for _, verb in ranked[: cfg.top_k]]


def select_heavy_classes(calls: Sequence[Call], cfg: SelectionConfig) -> dict[str, Any]:
    """Cross-fitted heavy-class selection: unanimous across all fit folds.

    Task-grouped folds; per fold the heavy set is chosen on the OTHER folds'
    calls (fit side). The evaluated set is the intersection over folds -- a
    class survives only if every fit split ranks it heavy, so the choice never
    hinges on one lucky split. The top-1 class (count-bucket target) is the
    unanimous class with the highest mean fit-fold median.
    """

    rows = [{"task_id": call.task_id} for call in calls]
    folds = balanced_task_folds(rows, fold_count=cfg.fold_count)
    task_to_fold = {task: idx for idx, fold in enumerate(folds) for task in fold}

    per_fold: list[list[str]] = []
    fold_rank_median: dict[str, list[float]] = defaultdict(list)
    for test_index in range(cfg.fold_count):
        fit_calls = [c for c in calls if task_to_fold[c.task_id] != test_index]
        if not fit_calls:
            continue
        heavy = _fold_heavy_classes(fit_calls, cfg)
        per_fold.append(heavy)
        # cache each selected class's fit-fold median for the top-1 tie-break
        medians = _class_task_medians(fit_calls)
        for verb in heavy:
            task_medians = [statistics.median(v) for v in medians[verb].values()]
            fold_rank_median[verb].append(statistics.median(task_medians))

    if not per_fold:
        return {"selected": [], "top1": None, "per_fold": per_fold, "unanimous": []}
    unanimous = sorted(set.intersection(*(set(f) for f in per_fold)))
    top1 = (
        max(unanimous, key=lambda v: statistics.mean(fold_rank_median[v]))
        if unanimous
        else None
    )
    return {
        "selected": unanimous,
        "top1": top1,
        "per_fold": per_fold,
        "unanimous": unanimous,
    }


# --------------------------------------------------------------------------- #
# State-hash construction (strictly-prior history only).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateVocab:
    selected: tuple[str, ...]
    top1: str | None
    count_bucket_edges: tuple[float, ...]

    def bits(self) -> int:
        """Bit budget: one per selected class + count bucket + one cwd bit."""

        n_buckets = len(self.count_bucket_edges) + 1
        bucket_bits = int(math.ceil(math.log2(n_buckets))) if n_buckets > 1 else 0
        return len(self.selected) + bucket_bits + 1


def assert_state_budget(vocab: StateVocab, max_bits: int) -> None:
    used = vocab.bits()
    if used > max_bits:
        raise ValueError(
            f"state vocabulary needs {used} bits > max_state_bits {max_bits}: "
            f"{len(vocab.selected)} class bits + count bucket + cwd bit"
        )


def _count_bucket(count: int, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), count, side="right"))


@dataclass(frozen=True)
class StateRecord:
    task_id: str
    verb_class: str
    duration_ms: float
    state: tuple[Any, ...]


def compute_state_records(calls: Sequence[Call], vocab: StateVocab) -> list[StateRecord]:
    """Attach a session-state hash to every observation of a selected class.

    State uses ONLY calls strictly before the current one (verified by temporal
    order_key). For each task the calls are walked in order; the state emitted
    for a call is a pure function of the accumulated prior history:
      * seen bit per selected class (did it appear in any earlier call);
      * count bucket of prior completions of the top-1 class;
      * whether the immediately preceding call changed the working directory.
    History is only THEN updated with the current call -- no leakage.
    """

    selected = vocab.selected
    by_task: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_task[call.task_id].append(call)

    records: list[StateRecord] = []
    for task_calls in by_task.values():
        task_calls.sort(key=lambda c: c.order_key)
        seen: set[str] = set()
        top1_count = 0
        prev_changed_cwd = False
        for call in task_calls:
            seen_bits = tuple(verb in seen for verb in selected)
            bucket = _count_bucket(top1_count, vocab.count_bucket_edges)
            state = (seen_bits, bucket, prev_changed_cwd)
            for verb, duration in call.observations:
                if verb in selected:
                    records.append(StateRecord(call.task_id, verb, duration, state))
            # advance history AFTER emitting this call's state
            seen |= call.history_verbs
            if vocab.top1 is not None and vocab.top1 in call.history_verbs:
                top1_count += 1
            prev_changed_cwd = call.cwd_changed
    return records


# --------------------------------------------------------------------------- #
# Conditional-vs-unconditional CV + task-clustered paired bootstrap.
# --------------------------------------------------------------------------- #
def _cv(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = statistics.mean(values)
    if mean <= 0:
        return float("nan")
    return statistics.pstdev(values) / mean


@dataclass
class ClassCV:
    """Per-class task-median caches so the bootstrap only re-indexes tasks."""

    verb_class: str
    task_all_median: dict[str, float]
    cell_task_median: dict[tuple[Any, ...], dict[str, float]]
    cell_weight: dict[tuple[Any, ...], int]

    def reduction(self, tasks: Sequence[str]) -> float:
        """CV reduction (unconditional - conditional) over a task multiset."""

        uncond_vals = [self.task_all_median[t] for t in tasks if t in self.task_all_median]
        uncond = _cv(uncond_vals)
        if uncond != uncond:
            return float("nan")
        weighted_sum = 0.0
        weight_total = 0.0
        for cell, task_median in self.cell_task_median.items():
            cell_vals = [task_median[t] for t in tasks if t in task_median]
            cvc = _cv(cell_vals)
            if cvc != cvc:
                continue
            w = self.cell_weight[cell]
            weighted_sum += w * cvc
            weight_total += w
        if weight_total == 0:
            return float("nan")
        cond = weighted_sum / weight_total
        return uncond - cond


def build_class_cvs(
    records: Sequence[StateRecord], cfg: SelectionConfig
) -> dict[str, ClassCV]:
    """Build per-class CV caches; keep only classes/cells clearing the support
    floors so an underpowered cell can never manufacture a spurious reduction.
    """

    by_class: dict[str, list[StateRecord]] = defaultdict(list)
    for rec in records:
        by_class[rec.verb_class].append(rec)

    out: dict[str, ClassCV] = {}
    for verb, recs in by_class.items():
        all_by_task: dict[str, list[float]] = defaultdict(list)
        cell_by_task: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in recs:
            all_by_task[rec.task_id].append(rec.duration_ms)
            cell_by_task[rec.state][rec.task_id].append(rec.duration_ms)
        if len(all_by_task) < cfg.min_class_tasks:
            continue
        task_all_median = {t: statistics.median(v) for t, v in all_by_task.items()}
        cell_task_median: dict[tuple[Any, ...], dict[str, float]] = {}
        cell_weight: dict[tuple[Any, ...], int] = {}
        for cell, task_map in cell_by_task.items():
            if len(task_map) < cfg.min_cell_tasks:
                continue
            cell_task_median[cell] = {
                t: statistics.median(v) for t, v in task_map.items()
            }
            cell_weight[cell] = len(task_map)
        if not cell_task_median:
            continue
        out[verb] = ClassCV(
            verb_class=verb,
            task_all_median=task_all_median,
            cell_task_median=cell_task_median,
            cell_weight=cell_weight,
        )
    return out


def aggregate_reduction(class_cvs: dict[str, ClassCV], tasks: Sequence[str]) -> float:
    """Mean CV reduction across evaluable classes over a task multiset."""

    values = [cv.reduction(tasks) for cv in class_cvs.values()]
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")


def paired_task_cv_bootstrap(
    class_cvs: dict[str, ClassCV],
    tasks: Sequence[str],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Task-clustered percentile CI of the aggregate CV reduction.

    Clusters are tasks; each replicate resamples tasks with replacement and
    recomputes the aggregate mean reduction. Paired because unconditional and
    conditional CV are computed on the SAME resampled task set. SURVIVE iff the
    CI lower bound excludes zero from above (reduction is positive).
    """

    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    task_list = list(tasks)
    if len(task_list) < 2:
        raise ValueError("need >= 2 tasks to bootstrap")
    point = aggregate_reduction(class_cvs, task_list)
    rng = np.random.default_rng(seed)
    n = len(task_list)
    reps: list[float] = []
    for _ in range(replicates):
        idx = rng.integers(0, n, n)
        sample = [task_list[i] for i in idx]
        value = aggregate_reduction(class_cvs, sample)
        if value == value:
            reps.append(value)
    if not reps:
        raise ValueError("all bootstrap replicates degenerate")
    arr = np.asarray(reps)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point_reduction": point,
        "ci_low": float(np.quantile(arr, alpha)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        "confidence": confidence,
        "replicates_used": len(reps),
        "tasks": n,
    }


def shuffle_control(
    records: Sequence[StateRecord], cfg: SelectionConfig, *, seed: int
) -> float:
    """Negative control: point reduction after shuffling state labels WITHIN
    each class (breaks the state<->duration link, preserves cell-size profile).

    Diagnostic only -- NOT part of the verdict. An informative split's reduction
    should collapse to ~0 here; a large residual would flag that the metric
    rewards partitioning per se rather than session-state information.
    """

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[StateRecord]] = defaultdict(list)
    for rec in records:
        by_class[rec.verb_class].append(rec)
    shuffled: list[StateRecord] = []
    for verb, recs in by_class.items():
        states = [rec.state for rec in recs]
        perm = rng.permutation(len(states))
        for rec, j in zip(recs, perm):
            shuffled.append(
                StateRecord(rec.task_id, rec.verb_class, rec.duration_ms, states[j])
            )
    class_cvs = build_class_cvs(shuffled, cfg)
    tasks = sorted({rec.task_id for rec in shuffled})
    if len(tasks) < 2 or not class_cvs:
        return float("nan")
    return aggregate_reduction(class_cvs, tasks)


# --------------------------------------------------------------------------- #
# Per-source evaluation.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int
    confidence: float
    seed: int


def evaluate_source(
    calls: Sequence[Call],
    sel_cfg: SelectionConfig,
    boot_cfg: BootstrapConfig,
    *,
    source: str,
) -> dict[str, Any]:
    """Full Stage-1 pipeline on one duration source; returns its verdict block."""

    selection = select_heavy_classes(calls, sel_cfg)
    selected = selection["selected"]
    if not selected:
        return {
            "source": source,
            "selection": selection,
            "degenerate": True,
            "degenerate_reason": "no heavy class selected unanimously across folds",
            "survive": False,
        }
    vocab = StateVocab(
        selected=tuple(selected),
        top1=selection["top1"],
        count_bucket_edges=sel_cfg.count_bucket_edges,
    )
    assert_state_budget(vocab, sel_cfg.max_state_bits)
    records = compute_state_records(calls, vocab)
    class_cvs = build_class_cvs(records, sel_cfg)
    if not class_cvs:
        return {
            "source": source,
            "selection": selection,
            "state_bits": vocab.bits(),
            "degenerate": True,
            "degenerate_reason": "no class cleared the task/cell support floors",
            "survive": False,
        }
    tasks = sorted({rec.task_id for rec in records})
    boot = paired_task_cv_bootstrap(
        class_cvs,
        tasks,
        replicates=boot_cfg.replicates,
        confidence=boot_cfg.confidence,
        seed=boot_cfg.seed,
    )
    per_class = []
    for verb, cv in sorted(class_cvs.items()):
        point = cv.reduction(tasks)
        uncond = _cv([cv.task_all_median[t] for t in cv.task_all_median])
        per_class.append(
            {
                "verb_class": verb,
                "tasks": len(cv.task_all_median),
                "cells": len(cv.cell_task_median),
                "unconditional_cv": uncond,
                "reduction": point,
            }
        )
    survive = boot["ci_low"] > 0.0
    return {
        "source": source,
        "selection": selection,
        "state_bits": vocab.bits(),
        "state_vocab": {
            "selected": list(vocab.selected),
            "top1": vocab.top1,
            "count_bucket_edges": list(vocab.count_bucket_edges),
        },
        "degenerate": False,
        "n_calls": len(calls),
        "n_records": len(records),
        "evaluable_classes": len(class_cvs),
        "per_class": per_class,
        "bootstrap": boot,
        "shuffle_control_reduction": shuffle_control(
            records, sel_cfg, seed=boot_cfg.seed
        ),
        "survive": survive,
    }


# --------------------------------------------------------------------------- #
# Config + orchestration.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    selection: SelectionConfig
    bootstrap: BootstrapConfig


def run(
    traces_dir: Path,
    manifest: Path,
    cfg: Config,
    *,
    glob: str,
    limit: int | None,
    limit_tasks: int | None,
    final: bool,
) -> dict[str, Any]:
    replayed_calls = load_replayed_calls(traces_dir, glob, limit=limit)
    original_calls = load_original_calls(manifest, limit_tasks=limit_tasks, final=final)

    replayed = evaluate_source(
        replayed_calls, cfg.selection, cfg.bootstrap, source=_SOURCE_REPLAYED
    )
    original = evaluate_source(
        original_calls, cfg.selection, cfg.bootstrap, source=_SOURCE_ORIGINAL
    )

    combined_survive = bool(replayed["survive"] and original["survive"])
    verdict = "SURVIVE" if combined_survive else "KILL"
    kill_reasons: list[str] = []
    for block in (replayed, original):
        if not block["survive"]:
            if block.get("degenerate"):
                kill_reasons.append(
                    f"{block['source']}: {block.get('degenerate_reason', 'degenerate')}"
                )
            else:
                kill_reasons.append(
                    f"{block['source']}: reduction CI does not exclude zero "
                    f"(low={block['bootstrap']['ci_low']:.4f})"
                )

    return {
        "provenance": {
            "exploratory": True,
            "final": bool(final),
            "replayed_on": "our_hardware",
            "traces_dir": str(traces_dir),
            "manifest": str(manifest),
            "git_sha": _git_sha(),
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        "config": {
            "top_k": cfg.selection.top_k,
            "floor_ms": cfg.selection.floor_ms,
            "fold_count": cfg.selection.fold_count,
            "count_bucket_edges": list(cfg.selection.count_bucket_edges),
            "max_state_bits": cfg.selection.max_state_bits,
            "min_class_tasks": cfg.selection.min_class_tasks,
            "min_cell_tasks": cfg.selection.min_cell_tasks,
            "bootstrap": {
                "replicates": cfg.bootstrap.replicates,
                "confidence": cfg.bootstrap.confidence,
                "seed": cfg.bootstrap.seed,
            },
        },
        "sources": {_SOURCE_REPLAYED: replayed, _SOURCE_ORIGINAL: original},
        "verdict": {
            "verdict": verdict,
            "combined_survive": combined_survive,
            "replayed_survive": replayed["survive"],
            "original_survive": original["survive"],
            "kill_reasons": kill_reasons,
        },
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
        return f"{value:.4f}"
    return str(value)


def _render_source(block: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"### Source: {block['source']}")
    lines.append("")
    if block.get("degenerate"):
        lines.append(
            f"DEGENERATE ({block.get('degenerate_reason', 'n/a')}) -> cannot survive."
        )
        lines.append(f"Selection per fold: {block['selection']['per_fold']}")
        lines.append("")
        return lines
    vocab = block["state_vocab"]
    boot = block["bootstrap"]
    lines.append(
        f"Selected heavy classes (unanimous): {vocab['selected']}; "
        f"top-1 (count target): {vocab['top1']}; state bits {block['state_bits']}."
    )
    lines.append(
        f"{block['n_records']} class observations, "
        f"{block['evaluable_classes']} evaluable classes."
    )
    lines.append("")
    lines.append("| verb class | tasks | cells | uncond CV | reduction |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in block["per_class"]:
        lines.append(
            f"| `{row['verb_class']}` | {row['tasks']} | {row['cells']} | "
            f"{_fmt(row['unconditional_cv'])} | {_fmt(row['reduction'])} |"
        )
    lines.append("")
    lines.append(
        f"Aggregate reduction {_fmt(boot['point_reduction'])} "
        f"(CI [{_fmt(boot['ci_low'])}, {_fmt(boot['ci_high'])}], "
        f"{int(boot['confidence'] * 100)}%, {boot['replicates_used']} reps, "
        f"{boot['tasks']} tasks) -> "
        f"{'SURVIVE' if block['survive'] else 'no CI exclusion'}."
    )
    lines.append(
        f"Shuffle negative control (diagnostic, expect ~0): "
        f"{_fmt(block['shuffle_control_reduction'])}."
    )
    lines.append("")
    return lines


def render_markdown(results: dict[str, Any]) -> str:
    prov = results["provenance"]
    verdict = results["verdict"]
    lines: list[str] = []
    lines.append("# Candidate A Stage-1: session-state-conditioned duration CV")
    lines.append("")
    lines.append(f"> **{_banner(prov['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Durations replayed on our own hardware "
        f"({prov['replayed_on']}). Generated {prov['generated']} "
        f"(git {prov['git_sha']})."
    )
    lines.append("")
    lines.append(f"**Verdict: {verdict['verdict']}**")
    lines.append("")
    if verdict["kill_reasons"]:
        for reason in verdict["kill_reasons"]:
            lines.append(f"- KILL: {reason}")
        lines.append("")
    lines.append(
        "Combined SURVIVE requires the aggregate CV-reduction CI to exclude zero "
        "on BOTH duration sources (a one-source-only signal is a hardware "
        "artifact -> KILL)."
    )
    lines.append("")
    lines.append("## Per-source results")
    lines.append("")
    for source in (_SOURCE_REPLAYED, _SOURCE_ORIGINAL):
        lines.extend(_render_source(results["sources"][source]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.wave_*.worker_*.jsonl")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "analysis/fresh-corpus-certification-20260717/"
            "offline-gated-robust/manifest.json"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Heavy verb classes to condition on (fit-fold median rank). "
        "Bounded so the state vocabulary stays within --max-state-bits.",
    )
    parser.add_argument(
        "--floor-ms",
        type=float,
        default=1000.0,
        help="Action-relevance floor: a class is heavy only if its median "
        "duration is at or above this many ms (default 1s -- sub-second calls "
        "are not scheduling-relevant). Wall-clock threshold, not a corpus "
        "constant.",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--count-bucket-edges",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="Right-open edges bucketing the prior top-1-class completion count "
        "(default 1 2 4 -> buckets {0},{1},{2,3},{4+}).",
    )
    parser.add_argument(
        "--max-state-bits",
        type=int,
        default=8,
        help="Hard cap on the state vocabulary bit budget (memo: <= ~8).",
    )
    parser.add_argument(
        "--min-class-tasks",
        type=int,
        default=10,
        help="A class needs this many tasks with an observation to be evaluated.",
    )
    parser.add_argument(
        "--min-cell-tasks",
        type=int,
        default=5,
        help="A (class, state) cell needs this many tasks to enter the "
        "conditional-CV average (support floor against spurious reductions).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="Smoke only: cap replayed wave files."
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Smoke only: cap original-source tasks. Rejected with --final.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/session-state-cv-stage1-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    cfg = Config(
        selection=SelectionConfig(
            top_k=args.top_k,
            floor_ms=args.floor_ms,
            fold_count=args.fold_count,
            count_bucket_edges=tuple(args.count_bucket_edges),
            max_state_bits=args.max_state_bits,
            min_class_tasks=args.min_class_tasks,
            min_cell_tasks=args.min_cell_tasks,
        ),
        bootstrap=BootstrapConfig(
            replicates=args.bootstrap_replicates,
            confidence=args.bootstrap_confidence,
            seed=args.bootstrap_seed,
        ),
    )
    results = run(
        args.traces_dir,
        args.manifest,
        cfg,
        glob=args.glob,
        limit=args.limit,
        limit_tasks=args.limit_tasks,
        final=args.final,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(results), encoding="utf-8")

    verdict = results["verdict"]
    print(f"verdict={verdict['verdict']}")
    for source in (_SOURCE_REPLAYED, _SOURCE_ORIGINAL):
        block = results["sources"][source]
        if block.get("degenerate"):
            print(f"  {source}: DEGENERATE ({block.get('degenerate_reason')})")
        else:
            boot = block["bootstrap"]
            print(
                f"  {source}: reduction {boot['point_reduction']:.4f} "
                f"CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] "
                f"-> {'survive' if block['survive'] else 'kill'}"
            )
    for reason in verdict["kill_reasons"]:
        print(f"  KILL: {reason}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
