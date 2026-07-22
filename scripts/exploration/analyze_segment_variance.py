#!/usr/bin/env python3
"""Atom-vs-chain variance decomposition for segment-timeline tool durations.

Scientific question (deliverable for the advisor): does decomposing chained
shell commands into their sequential *atoms* explain tool-call duration
variance well enough that an atom-level latency predictor deserves a rematch
against the chain-prefix trie? The additive segment-cost model was rejected
when atom costs had to be *deconvolved* from chain totals (see
``trace_collect.segment_cost_model``). segment_timeline v2 removes that
objection by observing each atom's duration directly; the remaining question
is whether atoms + their arguments carry the variance, or whether
chain-context / state-sharing dominates.

Five out-of-sample models predict a chain's ``parent_total_ms`` on
task-grouped train/test folds:

* ``atom_identity``     - global overhead + sum of per-atom TRAIN median
  durations (the additive model's dream case, now with direct observation).
* ``atom_plus_args``    - same, conditioned additionally on a coarse
  per-segment argument-size bin (token count).
* ``atom_trie``         - the model the advisor actually proposed: a PER-ATOM
  command-prefix trie. It applies the chain trie's conditioning one level
  down - each segment's own token stream yields nested prefix keys
  (``verb``, ``verb arg1``, ...) up to ``--atom-depth`` (default 4); per key
  node it collects observed SEGMENT durations, gated by the exploratory
  ``--min-prefix-evidence`` knob (like ``chain_prefix_cdskip``, NOT a cert
  literal). Each atom takes its deepest evidence-passing node's TRAIN median,
  backing off deeper -> shallower -> verb -> global-atom-median; the chain
  prediction sums the per-atom costs and adds the same train-residual overhead
  intercept the other additive models get, for fairness. A per-model
  diagnostic reports mean matched depth per atom and the verb-level fallback
  fraction, exposing whether argument thinness stops evidence from
  conditioning below the verb regardless of the MAE outcome.
* ``chain_prefix_cert`` - the trie's conditioning unit, EXACTLY as certified:
  TRAIN median ``parent_total_ms`` at the deepest depth-capped command-prefix
  key with enough evidence, backing off to shallower keys, the tool, then
  global, with ``skip_leading_cd=False``, ``max_prefix_depth=4`` and
  ``min_evidence=1`` matching the frozen FRESH-CERT offline-fit config (see
  ``scripts/serving/export_trigger_table.py`` and the certified manifest at
  ``analysis/fresh-corpus-certification-20260717/offline-gated-robust/
  manifest.json``, whose ``min_tool_history``/``max_prefix_depth`` gate every
  backoff level - ``trace_collect/tool_latency_profiled.py:255,775,791``).
  These three values are LITERALS in the model registry, independent of the
  ``--prefix-depth``/``--min-prefix-evidence`` sweep knobs, so a sweep run can
  never silently degrade the cert baseline's resolution. This is the faithful
  rematch baseline; ``chain_prefix_cdskip`` below is a variant, not the cert.
* ``chain_prefix_cdskip`` - identical fitter with ``skip_leading_cd=True``
  (drops leading ``cd`` segments from the prefix key), depth and evidence gate
  taken from the sweep config (``--prefix-depth``/``--min-prefix-evidence``).
  Deviates from the certified config on all three axes; included because it
  is precisely H2 case-study fix candidate #2, so this run doubles as
  evidence for that pending decision.

DURATION-VALIDITY CEILING (trace_collect/CLAUDE.md): per-segment durations are
trustworthy ONLY for sequential operators (``&&``, ``;``, newline). Parents
containing a pipe or a loop keyword are EXCLUDED from all duration analysis and
counted; the excluded fraction is reported up front, never hidden.

No synthetic data: durations come only from the replay corpus. Results are
EXPLORATORY and replayed on our own hardware. Run prints a "PARTIAL - not
final" banner unless ``--final`` is passed (the trigger is the full-corpus
replay finishing).

Usage:
  uv run python scripts/exploration/analyze_segment_variance.py \
    --traces-dir traces/fresh-277-segtimeline \
    --fold-count 5 --prefix-depth 4 --final
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from trace_collect.command_features import (
    command_has_concurrent_segments,
    command_prefix_keys,
    segment_prefix_keys,
    shell_command_heads,
    shell_command_prefix_tokens,
)
from trace_collect.tool_latency_dataset import (
    SegmentLatencySample,
    extract_many_segment_latency_samples,
)
from trace_collect.tool_latency_offline_probe import balanced_task_folds
from trace_collect.trace_data import TraceData

_UNPARSED_ATOM = "__unparsed__"


# --------------------------------------------------------------------------- #
# Pure logic (atom keying, models, metrics) - imported by tests.
# --------------------------------------------------------------------------- #
def atom_key(segment_command: str) -> str:
    """Normalize a segment command to its atom key (leading command head).

    Mirrors ``command_features`` head-verb logic: the path-basenamed first
    head of the segment. ``cd`` keeps its own head, so preamble atoms form
    their own class naturally. Untokenizable segments map to a shared
    ``__unparsed__`` class (the designed tool-level fallback, not an error).
    """

    heads = shell_command_heads(segment_command)
    return heads[0] if heads else _UNPARSED_ATOM


def segment_token_count(segment_command: str) -> int:
    """Coarse argument-size feature: normalized token count of the segment."""

    return len(shell_command_prefix_tokens(segment_command))


@dataclass(frozen=True)
class Segment:
    atom: str
    duration_ms: float
    token_count: int
    # Normalized per-segment token stream (same tokenizer command_prefix_keys
    # uses, applied PER ATOM) - feeds the atom_trie model's prefix keys.
    # Defaults empty so non-trie fixtures need not supply it.
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chain:
    """One exec tool call reconstructed from its per-atom segment samples."""

    task_id: str
    source_trace: str
    action_id: str
    tool_name: str
    parent_command: str
    parent_total_ms: float
    parent_raw_total_ms: float | None
    segments: tuple[Segment, ...]

    @property
    def family(self) -> str:
        """Ordered atom-identity signature, e.g. ``cd>>python3``."""

        return ">>".join(seg.atom for seg in self.segments)

    @property
    def segment_sum_ms(self) -> float:
        return sum(seg.duration_ms for seg in self.segments)


def _build_segment(sample: SegmentLatencySample) -> Segment:
    """One segment sample with its atom, duration and normalized token stream.

    The token stream is tokenized once (same tokenizer as the chain trie) and
    reused for both ``token_count`` and the atom_trie prefix keys.
    """

    tokens = tuple(shell_command_prefix_tokens(sample.segment_command))
    return Segment(
        atom=atom_key(sample.segment_command),
        duration_ms=sample.segment_ms,
        token_count=len(tokens),
        tokens=tokens,
    )


def build_chains(samples: Sequence[SegmentLatencySample]) -> list[Chain]:
    """Group per-segment samples into chains keyed by (trace, action)."""

    grouped: dict[tuple[str, str], list[SegmentLatencySample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.source_trace, sample.action_id)].append(sample)
    chains: list[Chain] = []
    for (source_trace, action_id), rows in grouped.items():
        rows.sort(key=lambda s: s.segment_index)
        first = rows[0]
        segments = tuple(_build_segment(s) for s in rows)
        chains.append(
            Chain(
                task_id=first.task_id,
                source_trace=source_trace,
                action_id=action_id,
                tool_name=first.tool_name,
                parent_command=first.parent_chain_command or "",
                parent_total_ms=first.parent_total_ms,
                parent_raw_total_ms=first.parent_raw_total_ms,
                segments=segments,
            )
        )
    return chains


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _token_bin_edges(token_counts: Sequence[int], *, bin_count: int) -> list[float]:
    """Quantile edges splitting token counts into ``bin_count`` bins.

    Edges are the interior quantiles of the TRAIN token-count distribution, so
    bins are data-driven (no hardcoded size thresholds). Fewer distinct values
    than bins collapses to whatever the quantiles yield - harmless, the model
    just sees coarser bins.
    """

    if bin_count < 1:
        raise ValueError(f"bin_count must be >= 1, got {bin_count}")
    if bin_count == 1:
        return []
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    return [float(np.quantile(token_counts, q)) for q in quantiles]


def _token_bin(token_count: int, edges: Sequence[float]) -> int:
    return int(np.searchsorted(edges, token_count, side="right"))


def fit_atom_identity(train: Sequence[Chain]) -> Callable[[Chain], float]:
    """Global overhead + sum of per-atom TRAIN median segment durations.

    The single global overhead constant absorbs shell/preamble/gap time that
    lives in ``parent_total_ms`` but in no segment (``raw_total`` = sum of
    segments + gaps); without it every additive prediction carries the same
    level bias, confounding the variance question this model exists to answer.
    Estimated as the median residual (parent_total - sum of atom medians) over
    train chains. Unseen atoms fall back to the global segment median.
    """

    by_atom: dict[str, list[float]] = defaultdict(list)
    all_durations: list[float] = []
    for chain in train:
        for seg in chain.segments:
            by_atom[seg.atom].append(seg.duration_ms)
            all_durations.append(seg.duration_ms)
    atom_median = {atom: _median(vals) for atom, vals in by_atom.items()}
    global_median = _median(all_durations)

    def sum_atom_medians(chain: Chain) -> float:
        return sum(atom_median.get(seg.atom, global_median) for seg in chain.segments)

    overhead = _median(
        [chain.parent_total_ms - sum_atom_medians(chain) for chain in train]
    )

    def predict(chain: Chain) -> float:
        return overhead + sum_atom_medians(chain)

    return predict


def fit_atom_plus_args(
    train: Sequence[Chain], *, bin_count: int
) -> Callable[[Chain], float]:
    """Additive model conditioned on (atom, coarse token-count bin)."""

    token_counts = [seg.token_count for chain in train for seg in chain.segments]
    edges = _token_bin_edges(token_counts, bin_count=bin_count)
    by_atom_bin: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_atom: dict[str, list[float]] = defaultdict(list)
    all_durations: list[float] = []
    for chain in train:
        for seg in chain.segments:
            key = (seg.atom, _token_bin(seg.token_count, edges))
            by_atom_bin[key].append(seg.duration_ms)
            by_atom[seg.atom].append(seg.duration_ms)
            all_durations.append(seg.duration_ms)
    atom_bin_median = {key: _median(v) for key, v in by_atom_bin.items()}
    atom_median = {atom: _median(v) for atom, v in by_atom.items()}
    global_median = _median(all_durations)

    def seg_cost(seg: Segment) -> float:
        key = (seg.atom, _token_bin(seg.token_count, edges))
        if key in atom_bin_median:
            return atom_bin_median[key]
        if seg.atom in atom_median:
            return atom_median[seg.atom]
        return global_median

    def sum_costs(chain: Chain) -> float:
        return sum(seg_cost(seg) for seg in chain.segments)

    overhead = _median([chain.parent_total_ms - sum_costs(chain) for chain in train])

    def predict(chain: Chain) -> float:
        return overhead + sum_costs(chain)

    return predict


@dataclass(frozen=True)
class AtomTrieModel:
    """Per-atom trie predictor plus a match-depth probe for the diagnostic."""

    predict: Callable[[Chain], float]
    match_depth: Callable[[Chain, Segment], int]


def fit_atom_trie(
    train: Sequence[Chain], *, max_depth: int, min_evidence: int
) -> AtomTrieModel:
    """Per-atom command-prefix trie: the chain trie applied ONE LEVEL DOWN.

    Where ``fit_chain_prefix`` keys the whole chain command, this keys each
    segment's own token stream: nested prefix keys ``verb``, ``verb arg1``, ...
    up to ``max_depth``, each node collecting observed SEGMENT durations gated
    by ``min_evidence`` (the exploratory sweep knob, like ``chain_prefix_cdskip``
    - not a cert literal). Prediction per atom is the deepest evidence-passing
    node's TRAIN median, backing off deeper -> shallower -> verb, then the
    global segment median. The chain prediction sums the per-atom costs and
    adds the same train-residual overhead intercept the other additive models
    use (``fit_atom_identity``/``fit_atom_plus_args``), so the level bias is
    treated identically across models - apples-to-apples.

    Returns both the predictor and a ``match_depth(chain, seg)`` probe (the
    depth of the node that served an atom: 1..max_depth for a prefix node, 0
    for the global-atom fallback) for the headwind-#2 diagnostic.
    """

    by_key: dict[str, list[float]] = defaultdict(list)
    all_durations: list[float] = []
    for chain in train:
        for seg in chain.segments:
            for key in segment_prefix_keys(
                chain.tool_name, list(seg.tokens), max_depth=max_depth
            ):
                by_key[key].append(seg.duration_ms)
            all_durations.append(seg.duration_ms)
    key_median = {k: _median(v) for k, v in by_key.items() if len(v) >= min_evidence}
    global_median = _median(all_durations)

    def seg_cost_depth(chain: Chain, seg: Segment) -> tuple[float, int]:
        keys = segment_prefix_keys(
            chain.tool_name, list(seg.tokens), max_depth=max_depth
        )
        for depth in range(len(keys), 0, -1):  # deepest key first
            median = key_median.get(keys[depth - 1])
            if median is not None:
                return median, depth
        return global_median, 0

    def sum_costs(chain: Chain) -> float:
        return sum(seg_cost_depth(chain, seg)[0] for seg in chain.segments)

    overhead = _median([chain.parent_total_ms - sum_costs(chain) for chain in train])

    def predict(chain: Chain) -> float:
        return overhead + sum_costs(chain)

    def match_depth(chain: Chain, seg: Segment) -> int:
        return seg_cost_depth(chain, seg)[1]

    return AtomTrieModel(predict=predict, match_depth=match_depth)


def fit_chain_prefix(
    train: Sequence[Chain],
    *,
    max_depth: int,
    min_evidence: int,
    skip_leading_cd: bool,
    key_fn: Callable[[Chain], Sequence[str]] | None = None,
) -> Callable[[Chain], float]:
    """Trie conditioning unit: TRAIN median parent_total_ms at the deepest
    depth-capped command-prefix key with >= ``min_evidence`` chains, backing
    off to shallower keys, then tool, then global.

    To reproduce the certified FRESH-CERT trie exactly, callers must pass
    ``skip_leading_cd=False``, ``max_depth=4`` and ``min_evidence=1`` (the
    frozen ``min_tool_history`` gate that applies at every backoff level per
    ``trace_collect/tool_latency_profiled.py:255,775,791`` - see the
    certified manifest at
    ``analysis/fresh-corpus-certification-20260717/offline-gated-robust/
    manifest.json``). The ``chain_prefix_cert`` model registry entry below
    passes these three as literals, not the ``--prefix-depth``/
    ``--min-prefix-evidence`` sweep knobs, so it cannot drift off the cert.
    ``chain_prefix_cdskip`` is a deliberate deviation on all three axes,
    evaluated separately, not the certified one.

    ``key_fn`` overrides the command-prefix key derivation for a whole run
    (fit AND predict use the same callable, so backoff stays consistent). It
    exists so the wrapper-transparency study can plug a data-driven key
    normalization into this exact fitter instead of forking the backoff logic;
    when ``None`` the default is the production ``command_prefix_keys`` at the
    given ``max_depth``/``skip_leading_cd`` (the cert/cdskip behaviour above).
    """

    if key_fn is None:

        def key_fn(chain: Chain) -> Sequence[str]:
            return command_prefix_keys(
                chain.tool_name,
                chain.parent_command,
                max_depth=max_depth,
                skip_leading_cd=skip_leading_cd,
            )

    by_key: dict[str, list[float]] = defaultdict(list)
    by_tool: dict[str, list[float]] = defaultdict(list)
    all_totals: list[float] = []
    for chain in train:
        for key in key_fn(chain):
            by_key[key].append(chain.parent_total_ms)
        by_tool[chain.tool_name].append(chain.parent_total_ms)
        all_totals.append(chain.parent_total_ms)
    key_median = {k: _median(v) for k, v in by_key.items() if len(v) >= min_evidence}
    tool_median = {t: _median(v) for t, v in by_tool.items()}
    global_median = _median(all_totals)

    def predict(chain: Chain) -> float:
        for key in reversed(list(key_fn(chain))):  # deepest first
            if key in key_median:
                return key_median[key]
        return tool_median.get(chain.tool_name, global_median)

    return predict


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _metrics_block(
    y_true: np.ndarray, y_pred: np.ndarray, *, tail_percentile: float
) -> dict[str, Any]:
    if y_true.size == 0:
        return {"count": 0}
    tail_threshold = float(np.percentile(y_true, tail_percentile))
    tail_mask = y_true >= tail_threshold
    mid_mask = ~tail_mask
    block: dict[str, Any] = {
        "count": int(y_true.size),
        "r2": r2_score(y_true, y_pred),
        "mae_ms": mae(y_true, y_pred),
        "tail_percentile": tail_percentile,
        "tail_threshold_ms": tail_threshold,
        "tail_count": int(tail_mask.sum()),
        "tail_mae_ms": mae(y_true[tail_mask], y_pred[tail_mask])
        if tail_mask.any()
        else None,
        "tail_r2": r2_score(y_true[tail_mask], y_pred[tail_mask])
        if tail_mask.sum() > 1
        else None,
        "middle_mae_ms": mae(y_true[mid_mask], y_pred[mid_mask])
        if mid_mask.any()
        else None,
    }
    return block


# --------------------------------------------------------------------------- #
# Cross-validated evaluation.
# --------------------------------------------------------------------------- #
# FRESH-CERT frozen offline-fit config, as literals (never the sweep knobs):
# max_prefix_depth=4 and min_tool_history=1 from the certified manifest at
# analysis/fresh-corpus-certification-20260717/offline-gated-robust/
# manifest.json; min_tool_history gates every backoff level per
# trace_collect/tool_latency_profiled.py:255,775,791. A --prefix-depth or
# --min-prefix-evidence sweep must never change the cert baseline's
# resolution - CERT_MIN_EVIDENCE=1 in particular keeps every depth-1..4 node
# with >=1 observed chain, matching the cert's exhaustive backoff.
_CERT_MAX_PREFIX_DEPTH = 4
_CERT_MIN_EVIDENCE = 1

_MODEL_FITTERS = {
    "atom_identity": lambda train, cfg: fit_atom_identity(train),
    "atom_plus_args": lambda train, cfg: fit_atom_plus_args(
        train, bin_count=cfg.token_bin_count
    ),
    # The advisor's proposal: per-atom command-prefix trie. Exploratory, so it
    # uses the sweep knobs (cfg.atom_depth / cfg.min_prefix_evidence), NOT cert
    # literals - same footing as chain_prefix_cdskip.
    "atom_trie": lambda train, cfg: fit_atom_trie(
        train, max_depth=cfg.atom_depth, min_evidence=cfg.min_prefix_evidence
    ).predict,
    # Certified trie, exactly: skip_leading_cd=False, max_depth and
    # min_evidence are the frozen cert literals above (never cfg) - the
    # faithful rematch baseline for the atom-vs-chain question.
    "chain_prefix_cert": lambda train, cfg: fit_chain_prefix(
        train,
        max_depth=_CERT_MAX_PREFIX_DEPTH,
        min_evidence=_CERT_MIN_EVIDENCE,
        skip_leading_cd=False,
    ),
    # cd-normalization variant (deviates from the cert config on all three
    # axes; uses the sweep knobs since it's exploratory). Doubles as
    # evidence for pending H2 case-study fix candidate #2.
    "chain_prefix_cdskip": lambda train, cfg: fit_chain_prefix(
        train,
        max_depth=cfg.prefix_depth,
        min_evidence=cfg.min_prefix_evidence,
        skip_leading_cd=True,
    ),
}


def _task_folds(
    chains: Sequence[Chain], cfg: "Config"
) -> Iterator[tuple[list[Chain], list[Chain]]]:
    """Yield (train, test) chain splits for each task-grouped fold.

    Empty-train/empty-test folds are skipped. Shared by the CV predictions and
    the atom_trie diagnostic so both see exactly the same fold split.
    """

    rows = [{"task_id": chain.task_id} for chain in chains]
    folds = balanced_task_folds(rows, fold_count=cfg.fold_count)
    task_to_fold = {task: index for index, fold in enumerate(folds) for task in fold}
    for test_index in range(cfg.fold_count):
        train = [c for c in chains if task_to_fold[c.task_id] != test_index]
        test = [c for c in chains if task_to_fold[c.task_id] == test_index]
        if train and test:
            yield train, test


def cross_validated_predictions(
    chains: Sequence[Chain], cfg: "Config"
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Task-grouped out-of-sample (y_true, y_pred, family) per model."""

    collected: dict[str, tuple[list[float], list[float], list[str]]] = {
        name: ([], [], []) for name in _MODEL_FITTERS
    }
    for train, test in _task_folds(chains, cfg):
        for name, fitter in _MODEL_FITTERS.items():
            predict = fitter(train, cfg)
            yt, yp, fam = collected[name]
            for chain in test:
                yt.append(chain.parent_total_ms)
                yp.append(predict(chain))
                fam.append(chain.family)
    return {
        name: (np.asarray(yt), np.asarray(yp), fam)
        for name, (yt, yp, fam) in collected.items()
    }


def atom_trie_diagnostics(chains: Sequence[Chain], cfg: "Config") -> dict[str, Any]:
    """Out-of-sample match-depth accounting for the atom_trie model only.

    For every test-fold atom, records the trie depth that served its prediction
    (1..atom_depth for a prefix node, 0 for the global-atom fallback), using the
    same folds as the CV predictions. Reports per-atom mean matched depth and
    the verb-level fallback fraction (depth == 1): the direct test of whether
    argument thinness (headwind #2) stops evidence from conditioning below the
    verb, independent of the MAE outcome.
    """

    by_atom: dict[str, list[int]] = defaultdict(list)
    for train, test in _task_folds(chains, cfg):
        model = fit_atom_trie(
            train, max_depth=cfg.atom_depth, min_evidence=cfg.min_prefix_evidence
        )
        for chain in test:
            for seg in chain.segments:
                by_atom[seg.atom].append(model.match_depth(chain, seg))
    per_atom: list[dict[str, Any]] = []
    all_depths: list[int] = []
    for atom, depths in by_atom.items():
        all_depths.extend(depths)
        per_atom.append(
            {
                "atom": atom,
                "count": len(depths),
                "mean_matched_depth": float(np.mean(depths)),
                "verb_level_fraction": float(np.mean([d == 1 for d in depths])),
                "global_fallback_fraction": float(np.mean([d == 0 for d in depths])),
            }
        )
    per_atom.sort(key=lambda row: -row["count"])
    overall = {
        "atom_prediction_count": len(all_depths),
        "atom_depth": cfg.atom_depth,
        "min_evidence": cfg.min_prefix_evidence,
        "mean_matched_depth": float(np.mean(all_depths)) if all_depths else float("nan"),
        "verb_level_fraction": float(np.mean([d == 1 for d in all_depths]))
        if all_depths
        else float("nan"),
        "global_fallback_fraction": float(np.mean([d == 0 for d in all_depths]))
        if all_depths
        else float("nan"),
    }
    return {"overall": overall, "per_atom": per_atom}


# --------------------------------------------------------------------------- #
# Corpus census (coverage / exclusion / reconciliation accounting).
# --------------------------------------------------------------------------- #
@dataclass
class Census:
    trace_files: int
    tool_exec_count: int
    timeline_present: int
    telemetry_absent: int
    concurrent_excluded: int

    def to_json(self) -> dict[str, Any]:
        analysed = self.timeline_present - self.concurrent_excluded
        return {
            "trace_files": self.trace_files,
            "tool_exec_count": self.tool_exec_count,
            "timeline_present": self.timeline_present,
            "telemetry_absent": self.telemetry_absent,
            "telemetry_coverage_fraction": (
                self.timeline_present / self.tool_exec_count
                if self.tool_exec_count
                else 0.0
            ),
            "concurrent_excluded": self.concurrent_excluded,
            "concurrent_excluded_fraction": (
                self.concurrent_excluded / self.timeline_present
                if self.timeline_present
                else 0.0
            ),
            "duration_analysable": analysed,
        }


def census_corpus(files: Sequence[Path]) -> Census:
    """Action-level coverage/exclusion accounting (fail-fast preserved).

    Counts every ``tool_exec``, whether a segment_timeline is present, absent
    (``telemetry_absent``), or a concurrent (pipe/loop) parent to be excluded.
    Genuine schema surprises still fail fast in the extractor pass; this pass
    only classifies presence/absence/exclusion.
    """

    tool_exec = present = absent = excluded = 0
    for path in files:
        trace = TraceData.load(path)
        for action in trace.actions:
            if action.get("action_type") != "tool_exec":
                continue
            tool_exec += 1
            data = action.get("data") or {}
            timeline = data.get("segment_timeline")
            if timeline is None:
                continue
            if isinstance(timeline, dict) and timeline.get("telemetry_absent"):
                absent += 1
                continue
            present += 1
            command = _parent_command(data)
            if command is not None and command_has_concurrent_segments(command):
                excluded += 1
    return Census(
        trace_files=len(files),
        tool_exec_count=tool_exec,
        timeline_present=present,
        telemetry_absent=absent,
        concurrent_excluded=excluded,
    )


def _parent_command(data: dict[str, Any]) -> str | None:
    tool_args = data.get("tool_args")
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return None
    if not isinstance(tool_args, dict):
        return None
    inner = tool_args.get("exec")
    if isinstance(inner, dict):
        tool_args = inner
    command = tool_args.get("command")
    return command if isinstance(command, str) else None


def reconciliation_stats(chains: Sequence[Chain]) -> dict[str, Any]:
    """sum-of-segments+gaps vs raw_total_ms: report the gap-ratio tails.

    gap_ratio = (raw_total - sum_segments) / raw_total. Positive gaps are the
    untraced preamble/shell overhead; large negative ratios flag chains whose
    segment durations overshoot the measured wall time (a residual telemetry
    artefact worth surfacing, not hiding).
    """

    ratios: list[float] = []
    for chain in chains:
        raw = chain.parent_raw_total_ms
        if raw is None or raw <= 0.0:
            continue
        ratios.append((raw - chain.segment_sum_ms) / raw)
    if not ratios:
        return {"count": 0}
    arr = np.asarray(ratios)
    return {
        "count": int(arr.size),
        "gap_ratio_median": float(np.median(arr)),
        "gap_ratio_p10": float(np.percentile(arr, 10)),
        "gap_ratio_p90": float(np.percentile(arr, 90)),
        "negative_gap_count": int((arr < 0).sum()),
        "negative_gap_fraction": float((arr < 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Atom vocabulary + within-chain structure.
# --------------------------------------------------------------------------- #
def atom_vocabulary(chains: Sequence[Chain]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for chain in chains:
        for seg in chain.segments:
            counts[seg.atom] += 1
    return [
        {"atom": atom, "count": count}
        for atom, count in counts.most_common()
    ]


def atom_stability(
    chains: Sequence[Chain], *, min_count: int
) -> list[dict[str, Any]]:
    """Per-atom cross-task coefficient of variation of segment duration.

    CV is computed over per-task median durations (not raw samples) so a few
    heavy tasks do not dominate: the direct test of whether atoms are more
    stable than the chains they compose. Only atoms observed in >= min_count
    tasks are reported.
    """

    by_atom_task: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for chain in chains:
        for seg in chain.segments:
            by_atom_task[seg.atom][chain.task_id].append(seg.duration_ms)
    out: list[dict[str, Any]] = []
    for atom, task_map in by_atom_task.items():
        task_medians = [_median(v) for v in task_map.values()]
        if len(task_medians) < min_count:
            continue
        mean = statistics.mean(task_medians)
        cv = (
            statistics.pstdev(task_medians) / mean
            if mean > 0 and len(task_medians) > 1
            else float("nan")
        )
        out.append(
            {
                "atom": atom,
                "task_count": len(task_medians),
                "median_ms": _median(task_medians),
                "cv_across_tasks": cv,
            }
        )
    out.sort(key=lambda row: row["cv_across_tasks"])
    return out


def family_variance_shares(
    chains: Sequence[Chain], *, min_count: int
) -> list[dict[str, Any]]:
    """Per family: which atom position carries the segment-duration variance,
    and the family's own cross-task CV of parent_total_ms (the chain baseline
    the atoms are tested against).
    """

    by_family: dict[str, list[Chain]] = defaultdict(list)
    for chain in chains:
        by_family[chain.family].append(chain)
    out: list[dict[str, Any]] = []
    for family, members in by_family.items():
        if len(members) < min_count:
            continue
        n_pos = len(members[0].segments)
        pos_var = []
        for pos in range(n_pos):
            durations = [m.segments[pos].duration_ms for m in members]
            pos_var.append(statistics.pvariance(durations) if len(durations) > 1 else 0.0)
        total_var = sum(pos_var)
        shares = [v / total_var if total_var > 0 else 0.0 for v in pos_var]
        dominant = int(np.argmax(pos_var)) if total_var > 0 else 0
        # chain-level cross-task CV of parent_total_ms
        by_task: dict[str, list[float]] = defaultdict(list)
        for m in members:
            by_task[m.task_id].append(m.parent_total_ms)
        task_medians = [_median(v) for v in by_task.values()]
        mean_total = statistics.mean(task_medians)
        chain_cv = (
            statistics.pstdev(task_medians) / mean_total
            if mean_total > 0 and len(task_medians) > 1
            else float("nan")
        )
        out.append(
            {
                "family": family,
                "chain_count": len(members),
                "atoms": [members[0].segments[i].atom for i in range(n_pos)],
                "dominant_position": dominant,
                "dominant_atom": members[0].segments[dominant].atom,
                "variance_shares": shares,
                "chain_cv_across_tasks": chain_cv,
            }
        )
    out.sort(key=lambda row: -row["chain_count"])
    return out


# --------------------------------------------------------------------------- #
# Config + orchestration.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    fold_count: int
    prefix_depth: int
    min_prefix_evidence: int
    atom_depth: int
    token_bin_count: int
    min_atom_count: int
    min_family_count: int
    tail_percentile: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.wave_*.worker_*.jsonl")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--prefix-depth", type=int, default=4)
    parser.add_argument("--min-prefix-evidence", type=int, default=5)
    parser.add_argument(
        "--atom-depth",
        type=int,
        default=4,
        help="Per-atom prefix depth budget for the atom_trie model (default 4).",
    )
    parser.add_argument("--token-bin-count", type=int, default=3)
    parser.add_argument("--min-atom-count", type=int, default=10)
    parser.add_argument("--min-family-count", type=int, default=20)
    parser.add_argument("--tail-percentile", type=float, default=90.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: analysis/offline/segment-atom-study-<date>[-PARTIAL].json "
        "(the -PARTIAL suffix is dropped only with --final).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Default: analysis/offline/segment-atom-study-<date>[-PARTIAL].md "
        "(the -PARTIAL suffix is dropped only with --final).",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Assert the run is on the complete corpus (drops PARTIAL banner "
        "and the -PARTIAL default output suffix).",
    )
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    """Default (json, md) output paths; PARTIAL runs get a -PARTIAL suffix

    so a partial file can never masquerade as final on disk.
    """

    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/offline/segment-atom-study-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config(
        fold_count=args.fold_count,
        prefix_depth=args.prefix_depth,
        min_prefix_evidence=args.min_prefix_evidence,
        atom_depth=args.atom_depth,
        token_bin_count=args.token_bin_count,
        min_atom_count=args.min_atom_count,
        min_family_count=args.min_family_count,
        tail_percentile=args.tail_percentile,
    )
    files = sorted(args.traces_dir.glob(args.glob))
    if not files:
        raise ValueError(f"no trace files under {args.traces_dir}/{args.glob}")

    census = census_corpus(files)
    # skip_concurrent=True drops concurrent (pipe/loop) parents before bounds
    # validation - but only when parent_chain_command parses; a concurrent
    # parent behind unparseable tool_args still reaches bounds validation and
    # can still fail-fast the whole run on a reversed t_end_ms < t_start_ms
    # (the fictitious-bounds artifact this flag exists to route around). No
    # behavior change here - documented contract, not a gap fix.
    samples = extract_many_segment_latency_samples(files, skip_concurrent=True)
    all_chains = build_chains(samples)
    multi = [c for c in all_chains if len(c.segments) >= 2]

    predictions = cross_validated_predictions(multi, cfg)
    model_metrics: dict[str, Any] = {}
    per_family_metrics: dict[str, Any] = {}
    for name, (y_true, y_pred, fams) in predictions.items():
        model_metrics[name] = _metrics_block(
            y_true, y_pred, tail_percentile=cfg.tail_percentile
        )
        fam_counts = Counter(fams)
        family_block: dict[str, Any] = {}
        fams_arr = np.asarray(fams)
        for family, count in fam_counts.items():
            if count < cfg.min_family_count:
                continue
            mask = fams_arr == family
            family_block[family] = _metrics_block(
                y_true[mask], y_pred[mask], tail_percentile=cfg.tail_percentile
            )
        per_family_metrics[name] = family_block

    results = {
        "provenance": {
            "exploratory": True,
            "final": bool(args.final),
            "replayed_on": "our_hardware",
            "traces_dir": str(args.traces_dir),
            "corpus_file_count": len(files),
            "git_sha": _git_sha(),
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        "config": cfg.__dict__,
        "census": census.to_json(),
        "reconciliation": reconciliation_stats(multi),
        "chain_counts": {
            "all_chains": len(all_chains),
            "multi_segment_chains": len(multi),
            "tasks": len({c.task_id for c in multi}),
        },
        "atom_vocabulary": atom_vocabulary(all_chains),
        "model_metrics": model_metrics,
        "per_family_metrics": per_family_metrics,
        "atom_trie_diagnostics": atom_trie_diagnostics(multi, cfg),
        "atom_stability": atom_stability(multi, min_count=cfg.min_atom_count),
        "family_variance": family_variance_shares(
            multi, min_count=cfg.min_family_count
        ),
    }
    return results


def _git_sha() -> str | None:
    """Best-effort commit SHA of the repo this script lives in.

    Anchored to this file's directory (not the caller's cwd) so provenance is
    correct regardless of where the script is invoked from.
    """

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
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _banner(final: bool) -> str:
    if final:
        return "FINAL - complete corpus"
    return (
        "PARTIAL - not final: validated against an in-progress replay; "
        "numbers are for script validation only, not findings."
    )


def render_markdown(results: dict[str, Any]) -> str:
    prov = results["provenance"]
    census = results["census"]
    recon = results["reconciliation"]
    counts = results["chain_counts"]
    lines: list[str] = []
    lines.append("# Segment atom-vs-chain variance study")
    lines.append("")
    lines.append(f"> **{_banner(prov['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Durations replayed on our own hardware "
        f"({prov['replayed_on']}); segment_timeline v2 telemetry. "
        f"Generated {prov['generated']}."
    )
    lines.append("")
    lines.append("## Exclusion accounting (read first)")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(f"| trace files | {census['trace_files']} |")
    lines.append(f"| tool_exec calls | {census['tool_exec_count']} |")
    lines.append(
        f"| segment_timeline present | {census['timeline_present']} "
        f"({census['telemetry_coverage_fraction']:.1%} coverage) |"
    )
    lines.append(f"| telemetry_absent | {census['telemetry_absent']} |")
    lines.append(
        f"| **pipe/loop excluded** | {census['concurrent_excluded']} "
        f"({census['concurrent_excluded_fraction']:.1%} of present) |"
    )
    lines.append(f"| duration-analysable | {census['duration_analysable']} |")
    lines.append("")
    lines.append(
        f"Reconciliation (raw_total vs sum-of-segments), gap ratio: "
        f"median {recon.get('gap_ratio_median', float('nan')):.3f}, "
        f"p10 {recon.get('gap_ratio_p10', float('nan')):.3f}, "
        f"p90 {recon.get('gap_ratio_p90', float('nan')):.3f}; "
        f"negative-gap tail {recon.get('negative_gap_fraction', 0.0):.1%} "
        f"({recon.get('negative_gap_count', 0)} chains)."
    )
    lines.append("")
    lines.append(
        f"Chains: {counts['all_chains']} total, "
        f"{counts['multi_segment_chains']} multi-segment "
        f"across {counts['tasks']} tasks."
    )
    lines.append("")
    lines.append("## Core decomposition (out-of-sample, task-grouped folds)")
    lines.append("")
    lines.append(
        "Target: chain `parent_total_ms`. tail = "
        f"P{results['config']['tail_percentile']:.0f}+ of true values."
    )
    lines.append("")
    lines.append("| model | R^2 | MAE ms | tail MAE ms | tail R^2 | mid MAE ms |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, block in results["model_metrics"].items():
        lines.append(
            f"| {name} | {_fmt(block.get('r2'))} | {_fmt(block.get('mae_ms'))} | "
            f"{_fmt(block.get('tail_mae_ms'))} | {_fmt(block.get('tail_r2'))} | "
            f"{_fmt(block.get('middle_mae_ms'))} |"
        )
    lines.append("")
    diag = results["atom_trie_diagnostics"]
    overall = diag["overall"]
    lines.append("## atom_trie match-depth diagnostic (headwind #2: argument thinness)")
    lines.append("")
    lines.append(
        f"Per-atom prefix depth budget {overall['atom_depth']}, evidence gate "
        f"{overall['min_evidence']}. Matched depth = trie level serving each "
        "atom (0 = global-atom fallback, 1 = verb-only node, higher = "
        "argument-conditioned). Over "
        f"{overall['atom_prediction_count']} out-of-sample atom predictions: "
        f"mean matched depth {_fmt(overall['mean_matched_depth'])}, verb-level "
        f"{overall['verb_level_fraction']:.1%}, global fallback "
        f"{overall['global_fallback_fraction']:.1%}."
    )
    lines.append("")
    lines.append("| atom | atoms | mean depth | verb-level | global fallback |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in diag["per_atom"]:
        if row["count"] < results["config"]["min_atom_count"]:
            continue
        lines.append(
            f"| {row['atom']} | {row['count']} | "
            f"{_fmt(row['mean_matched_depth'])} | "
            f"{row['verb_level_fraction']:.1%} | "
            f"{row['global_fallback_fraction']:.1%} |"
        )
    lines.append("")
    lines.append("## Atom-duration stability (lower CV = more stable across tasks)")
    lines.append("")
    lines.append("| atom | tasks | median ms | CV across tasks |")
    lines.append("| --- | --- | --- | --- |")
    for row in results["atom_stability"]:
        lines.append(
            f"| {row['atom']} | {row['task_count']} | "
            f"{_fmt(row['median_ms'])} | {_fmt(row['cv_across_tasks'])} |"
        )
    lines.append("")
    lines.append("## Within-chain variance (top families by chain count)")
    lines.append("")
    lines.append(
        "| family | chains | dominant atom | variance shares | chain CV/tasks |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for row in results["family_variance"][:20]:
        shares = ", ".join(f"{s:.2f}" for s in row["variance_shares"])
        lines.append(
            f"| `{row['family']}` | {row['chain_count']} | "
            f"{row['dominant_atom']} | {shares} | "
            f"{_fmt(row['chain_cv_across_tasks'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    if args.out_json is None:
        args.out_json = default_json
    if args.out_md is None:
        args.out_md = default_md
    print(_banner(args.final))
    results = run(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(results), encoding="utf-8")
    census = results["census"]
    print(
        f"tool_exec={census['tool_exec_count']} "
        f"present={census['timeline_present']} "
        f"pipe/loop-excluded={census['concurrent_excluded']} "
        f"({census['concurrent_excluded_fraction']:.1%}) "
        f"analysable={census['duration_analysable']}"
    )
    for name, block in results["model_metrics"].items():
        print(
            f"  {name:16s} R2={_fmt(block.get('r2'))} "
            f"MAE={_fmt(block.get('mae_ms'))}ms "
            f"tailMAE={_fmt(block.get('tail_mae_ms'))}ms"
        )
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
