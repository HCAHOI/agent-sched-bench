#!/usr/bin/env python3
"""WTN Stage-1 harness: emergent wrapper-transparency normalization.

Pre-registered in ``analysis/wrapper-transparency-design-20260719.md`` (FINAL,
post-debate). The question: cd-skip ("strip the literal leading ``cd X &&``")
is a hardcoded rule, not a method. WTN generalizes it -- candidate wrapper
positions come from shell STRUCTURE (every non-final segment class, keyed by
its verb token), transparency is decided by DATA on fit-side nested folds
(does pooling over presence/absence of the class degrade prediction of
``parent_total_ms``?), and the normalization is certified per workload. No
token spelling appears in the method logic; the string ``cd`` appears ONLY in
the K2 positive-control assertion and the cd-only grid arm, both of which are
evaluation harness, not method (integrity rule).

The harness reuses the five-model atom study
(``scripts.analyze_segment_variance``) verbatim: its wave parsing / chain
building (``build_chains``), its task-grouped fold split (``_task_folds``),
its metrics block (``_metrics_block``) and -- crucially -- its chain-prefix
fitter (``fit_chain_prefix``), which now accepts a ``key_fn`` so the WTN key
normalization plugs into the EXACT same backoff logic instead of a forked
trie. cd-only is that same fitter with ``skip_leading_cd=True`` (the existing
production code path).

Three questions, one run (spec Stage 1):

* **Step 0 -- reachable-mass census (emitted BEFORE any accuracy number).**
  Non-final wrapper candidates by verb class; corpus mass a normalization can
  touch. Fixes the contribution framing before accuracy numbers exist.
* **Transparency screen (sole gate).** Per candidate class, a fit-side NESTED
  task-grouped predictive-equivalence test: does dropping the class from the
  command-prefix key degrade OOS prediction of ``parent_total_ms`` beyond a
  pre-set relative-MAE tolerance? Marginal per class, then a joint
  non-degradation check on all passing classes. Underpowered classes (below
  configured task AND chain support) default to NON-transparent.
* **Knob-matched grid.** skip in {off, cd-only, WTN} x min_evidence in {1, 5},
  depth 4 fixed, same folds. OOS MAE + P90+ tail MAE per cell, plus
  task-clustered paired bootstrap CIs for the pre-registered pairwise deltas
  (WTN vs off, WTN vs cd-only, at matched min_evidence).

Pre-registered kill readout (printed explicitly):

* **K1 (reproduce the win, knob-matched).** KILL unless WTN beats knob-matched
  off on MAE with a paired CI excluding zero at the primary min_evidence, AND
  WTN is not worse than knob-matched cd-only beyond paired noise.
* **K2 (emergence, positive control, no retuning).** The one known
  ground-truth-transparent class (``cd``) must EMERGE transparent in every
  fold's marginal screen. Screen thresholds are frozen before this check and
  never retuned on the outcome.
* **K3 (stability, mass-weighted).** Fraction of normalization-affected chain
  mass whose class-transparency decision is unanimous across folds. Raw
  Jaccard on the tiny transparent set is degenerate and reported as a
  diagnostic only.

No synthetic data: durations come only from the replay corpus. EXPLORATORY,
replayed on our own hardware. Prints a PARTIAL banner unless ``--final``.

Usage:
  uv run python scripts/analyze_wrapper_transparency.py \
    --traces-dir traces/fresh-277-segtimeline --fold-count 5 --final
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np

# Allow `python scripts/analyze_wrapper_transparency.py ...` to import the
# sibling study module as a package (pytest adds the repo root itself).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_segment_variance import (  # noqa: E402
    Chain,
    Config,
    _git_sha,
    _metrics_block,
    _task_folds,
    build_chains,
    fit_chain_prefix,
)
from trace_collect.command_features import (  # noqa: E402
    shell_command_prefix_tokens,
    shell_command_segments,
)
from trace_collect.tool_latency_dataset import (  # noqa: E402
    extract_many_segment_latency_samples,
)

# The ONE method-adjacent appearance of a token spelling in this file, allowed
# by the integrity rule solely as the K2 positive control: `cd` is the single
# class whose transparency is known a-priori (a generic POSIX builtin with
# negligible cost that never defines the workload). It is NEVER used to select,
# drop, or weight anything -- only to assert the data-driven screen recovers it.
_K2_POSITIVE_CONTROL_CLASS = "cd"

# Fixed structural knobs (spec: "depth 4 fixed"). The grid varies only the two
# declared knobs (skip mode and min_evidence); depth is held at the cert value.
_PREFIX_DEPTH = 4
_GRID_MIN_EVIDENCE = (1, 5)

# --------------------------------------------------------------------------- #
# Structural candidate extraction + data-driven key normalization (no token
# names). A candidate is any non-final segment's verb; normalization drops the
# non-final segments whose verb is in a data-selected transparent set (applied
# by the production key derivation in command_features).
# --------------------------------------------------------------------------- #
def candidate_classes(command: str) -> list[str]:
    """Verb classes of the NON-FINAL segments of a sequential chain command.

    The verb is the segment's first (already path-basenamed) token, the same
    unit ``command_prefix_keys`` builds nodes from. The final segment is never
    a candidate (a wrapper wraps something), so a single-segment command yields
    no candidates. Verbs may repeat (e.g. ``cd .. && cd sub && make``).

    Uses ``shell_command_segments`` (which drops grouping parens) so a bare
    ``(`` can never surface as a spurious verb class; key reconstruction is
    delegated to ``command_features.shell_command_prefix_tokens`` (the
    production drop), which preserves paren tokens verbatim.
    """

    verbs = [seg[0] for seg in shell_command_segments(command) if seg]
    return verbs[:-1] if len(verbs) > 1 else []


def beyond_leading_candidates(command: str) -> list[str]:
    """Non-final candidates sitting AFTER the first segment (span index >= 1).

    Structural proxy for "reachable beyond a leading-only stripper": cd-skip
    strips only the leading run, so a mid-chain wrapper (index >= 1) is mass a
    leading-only rule cannot reach. Slightly generous for repeated identical
    leading verbs (``cd .. && cd sub``), a known census-granularity caveat; it
    is a framing diagnostic, never method logic, and names no token.
    """

    segments = [seg for seg in shell_command_segments(command) if seg]
    if len(segments) <= 1:
        return []
    non_final = segments[:-1]
    return [seg[0] for seg in non_final[1:]]  # drop the leading (index 0) span


def normalized_key_tokens(command: str, transparent: frozenset[str]) -> list[str]:
    """Token stream with non-final segments whose verb is transparent removed.

    Delegates to the PRODUCTION key normalization
    (``command_features.shell_command_prefix_tokens(transparent_wrappers=...)``)
    so the screen learns transparency against the exact same key derivation the
    shipped prior uses - no forked segment-split logic to drift. Empty
    ``transparent`` (or an untokenizable command) returns the production token
    stream unchanged, so the WTN key is IDENTICAL to the no-normalization key on
    the degenerate path.
    """

    return shell_command_prefix_tokens(command, transparent_wrappers=transparent)


def wtn_prefix_keys(
    tool_name: str, command: str, *, max_depth: int, transparent: frozenset[str]
) -> tuple[str, ...]:
    """Nested prefix keys of the transparency-normalized command."""

    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    tokens = normalized_key_tokens(command, transparent)
    if not tokens:
        return ()
    depth = min(len(tokens), max_depth)
    return tuple(
        f"{tool_name}:{' '.join(tokens[:length])}" for length in range(1, depth + 1)
    )


def make_wtn_key_fn(
    transparent: frozenset[str], *, max_depth: int
) -> Callable[[Chain], tuple[str, ...]]:
    """A ``fit_chain_prefix`` key function applying the transparent set."""

    def key_fn(chain: Chain) -> tuple[str, ...]:
        return wtn_prefix_keys(
            chain.tool_name,
            chain.parent_command,
            max_depth=max_depth,
            transparent=transparent,
        )

    return key_fn


# --------------------------------------------------------------------------- #
# Step 0 census.
# --------------------------------------------------------------------------- #
def census(chains: Sequence[Chain]) -> dict[str, Any]:
    """Reachable-mass census: candidates by verb class, corpus mass affected.

    Emitted before any accuracy number. Per non-final verb class: chains and
    tasks in which it is a wrapper. The HEADLINE framing quantity is chains
    (and parent_total_ms mass) with a wrapper BEYOND the leading segment -- the
    mass a leading-only stripper (cd-skip) cannot reach, i.e. WTN's novel
    territory. "Chains with any non-final candidate" is near-vacuous (every
    multi-segment chain trivially has one) and is kept only as an upper bound.
    """

    per_class_chains: Counter[str] = Counter()
    per_class_tasks: dict[str, set[str]] = defaultdict(set)
    affected_chains = 0
    affected_total_ms = 0.0
    beyond_leading_chains = 0
    beyond_leading_total_ms = 0.0
    newline_only_chains = 0
    total_ms = sum(c.parent_total_ms for c in chains)
    for chain in chains:
        classes = set(candidate_classes(chain.parent_command))
        for verb in classes:
            per_class_chains[verb] += 1
            per_class_tasks[verb].add(chain.task_id)
        if classes:
            affected_chains += 1
            affected_total_ms += chain.parent_total_ms
        if beyond_leading_candidates(chain.parent_command):
            beyond_leading_chains += 1
            beyond_leading_total_ms += chain.parent_total_ms
        # Population-mismatch row (#4): xtrace split this into >=2 segments
        # (it is in `chains`), but token-level extraction sees < 2 sequential
        # segments -- newline-separated or otherwise unsplit commands invisible
        # to candidate extraction. Surfaced so the denominator is honest.
        if len(shell_command_segments(chain.parent_command)) < 2:
            newline_only_chains += 1
    per_class = [
        {
            "verb_class": verb,
            "chains": count,
            "tasks": len(per_class_tasks[verb]),
        }
        for verb, count in per_class_chains.most_common()
    ]
    n = len(chains)
    return {
        "multi_segment_chains": n,
        "candidate_verb_classes": len(per_class_chains),
        # Headline framing quantity (beyond a leading-only stripper's reach).
        "chains_beyond_leading_segment": beyond_leading_chains,
        "chains_beyond_leading_fraction": beyond_leading_chains / n if n else 0.0,
        "beyond_leading_mass_fraction": (
            beyond_leading_total_ms / total_ms if total_ms > 0 else 0.0
        ),
        # Upper bound incl. leading cd: near-vacuous, kept only as a ceiling.
        "chains_with_candidate_upper_bound": affected_chains,
        "chains_with_candidate_upper_bound_fraction": (
            affected_chains / n if n else 0.0
        ),
        "affected_mass_fraction_upper_bound": (
            affected_total_ms / total_ms if total_ms > 0 else 0.0
        ),
        "newline_only_chains": newline_only_chains,
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# Transparency screen (sole gate), fit-side nested task-grouped folds.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScreenConfig:
    """Documented screen thresholds, applied to FIT-fold statistics only.

    ``tolerance`` -- a class is transparent iff dropping it does not raise the
    fit-side out-of-sample MAE by more than ``tolerance`` * (fit-side off MAE).
    A small positive slack (default 5%) because an exactly-zero tolerance would
    reject a class over pure fold noise; the sensitivity grid reports the
    selection at other tolerances.
    ``sensitivity_tolerances`` -- pre-declared points reported alongside the
    primary (the primary drives the applied normalization; sensitivity is
    re-thresholded from the SAME cached deltas, no refit, no retuning).
    ``min_tasks`` / ``min_chains`` -- a class below this fit-fold support is
    NON-transparent by default (an underpowered equivalence test must not pool;
    this protects rare heavy wrappers from silent over-pooling).
    ``inner_folds`` -- nested task-grouped folds within each outer train split
    (the screen NEVER sees the outer eval fold).
    ``min_evidence`` -- the prefix-node evidence gate the screen fits at; held
    at the cert value 1 so the screen judges the key at deployed resolution.
    """

    tolerance: float = 0.05
    sensitivity_tolerances: tuple[float, ...] = (0.02, 0.05, 0.10)
    min_tasks: int = 5
    min_chains: int = 20
    inner_folds: int = 5
    min_evidence: int = 1


@dataclass(frozen=True)
class ScreenResult:
    marginal: frozenset[str]
    applied: frozenset[str]
    joint_ok: bool
    joint_stat: float | None
    fit_side_mae: float
    universe: list[str]
    per_class: list[dict[str, Any]]
    degenerate: bool

    def marginal_at(self, tolerance: float) -> frozenset[str]:
        """Re-threshold the cached per-class stats at a sensitivity tolerance."""

        bound = tolerance * self.fit_side_mae
        return frozenset(
            row["verb_class"]
            for row in self.per_class
            if row["support_ok"] and row["stat_ms"] <= bound
        )


def _class_support(chains: Sequence[Chain]) -> dict[str, tuple[int, int]]:
    """Per verb class: (#tasks, #chains) in which it is a non-final wrapper."""

    tasks: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for chain in chains:
        for verb in set(candidate_classes(chain.parent_command)):
            tasks[verb].add(chain.task_id)
            counts[verb] += 1
    return {verb: (len(tasks[verb]), counts[verb]) for verb in counts}


def _abs_errors(
    predict: Callable[[Chain], float], test: Sequence[Chain]
) -> list[tuple[str, float]]:
    return [(c.task_id, abs(c.parent_total_ms - predict(c))) for c in test]


def _task_weighted_mean(deltas_by_task: dict[str, list[float]]) -> float:
    """Mean over tasks of each task's mean per-chain delta (equal task weight).

    The pre-registered statistic: "paired per-task mean of per-chain
    absolute-error deltas". Equal per-task weight so a few heavy tasks cannot
    dominate the equivalence decision.
    """

    per_task = [float(np.mean(v)) for v in deltas_by_task.values() if v]
    return float(np.mean(per_task)) if per_task else float("nan")


def _inner_config(cfg: ScreenConfig) -> Config:
    return Config(
        fold_count=cfg.inner_folds,
        prefix_depth=_PREFIX_DEPTH,
        min_prefix_evidence=cfg.min_evidence,
        atom_depth=_PREFIX_DEPTH,
        token_bin_count=3,
        min_atom_count=10,
        min_family_count=20,
        tail_percentile=90.0,
    )


def screen_transparency(train: Sequence[Chain], cfg: ScreenConfig) -> ScreenResult:
    """Cross-fitted transparency screen on one outer train split.

    Nested task-grouped folds within ``train``: for each inner fold the off
    (no-drop) arm is the shared reference; per candidate class the pool arm
    drops that class and its paired per-chain absolute-error delta vs off is
    accumulated. A class is transparent iff it has enough support AND pooling
    does not raise MAE beyond ``tolerance`` * (fit-side off MAE). The joint arm
    then drops all marginally-transparent classes at once and must clear the
    same bar; the applied set is the joint set only if that joint check passes.
    """

    universe = sorted({v for c in train for v in candidate_classes(c.parent_command)})
    support = _class_support(train)
    try:
        inner_folds = list(_task_folds(train, _inner_config(cfg)))
    except (ValueError, AssertionError):
        inner_folds = []
    if not inner_folds or not universe:
        return ScreenResult(
            marginal=frozenset(),
            applied=frozenset(),
            joint_ok=False,
            joint_stat=None,
            fit_side_mae=float("nan"),
            universe=universe,
            per_class=[],
            degenerate=True,
        )

    off_errors: list[float] = []
    deltas: dict[str, dict[str, list[float]]] = {v: defaultdict(list) for v in universe}
    for inner_train, inner_test in inner_folds:
        off_predict = fit_chain_prefix(
            inner_train,
            max_depth=_PREFIX_DEPTH,
            min_evidence=cfg.min_evidence,
            skip_leading_cd=False,
        )
        off_err = _abs_errors(off_predict, inner_test)
        off_errors.extend(err for _, err in off_err)
        for verb in universe:
            pool_predict = fit_chain_prefix(
                inner_train,
                max_depth=_PREFIX_DEPTH,
                min_evidence=cfg.min_evidence,
                skip_leading_cd=False,
                key_fn=make_wtn_key_fn(frozenset({verb}), max_depth=_PREFIX_DEPTH),
            )
            pool_err = _abs_errors(pool_predict, inner_test)
            for (task, e_pool), (_, e_off) in zip(pool_err, off_err):
                deltas[verb][task].append(e_pool - e_off)

    fit_side_mae = float(np.mean(off_errors)) if off_errors else float("nan")
    bound = cfg.tolerance * fit_side_mae
    per_class: list[dict[str, Any]] = []
    for verb in universe:
        n_tasks, n_chains = support.get(verb, (0, 0))
        support_ok = n_tasks >= cfg.min_tasks and n_chains >= cfg.min_chains
        stat = _task_weighted_mean(deltas[verb])
        transparent = support_ok and stat == stat and stat <= bound
        per_class.append(
            {
                "verb_class": verb,
                "stat_ms": stat,
                "support_tasks": n_tasks,
                "support_chains": n_chains,
                "support_ok": support_ok,
                "transparent": transparent,
            }
        )
    per_class.sort(key=lambda row: (not row["transparent"], row["stat_ms"]))
    marginal = frozenset(row["verb_class"] for row in per_class if row["transparent"])

    joint_stat = None
    joint_ok = False
    if marginal:
        joint_deltas: dict[str, list[float]] = defaultdict(list)
        for inner_train, inner_test in inner_folds:
            off_predict = fit_chain_prefix(
                inner_train,
                max_depth=_PREFIX_DEPTH,
                min_evidence=cfg.min_evidence,
                skip_leading_cd=False,
            )
            joint_predict = fit_chain_prefix(
                inner_train,
                max_depth=_PREFIX_DEPTH,
                min_evidence=cfg.min_evidence,
                skip_leading_cd=False,
                key_fn=make_wtn_key_fn(marginal, max_depth=_PREFIX_DEPTH),
            )
            off_err = _abs_errors(off_predict, inner_test)
            joint_err = _abs_errors(joint_predict, inner_test)
            for (task, e_joint), (_, e_off) in zip(joint_err, off_err):
                joint_deltas[task].append(e_joint - e_off)
        joint_stat = _task_weighted_mean(joint_deltas)
        joint_ok = joint_stat == joint_stat and joint_stat <= bound

    applied = marginal if joint_ok else frozenset()
    return ScreenResult(
        marginal=marginal,
        applied=applied,
        joint_ok=joint_ok,
        joint_stat=joint_stat,
        fit_side_mae=fit_side_mae,
        universe=universe,
        per_class=per_class,
        degenerate=False,
    )


# --------------------------------------------------------------------------- #
# Task-clustered paired bootstrap (statistic = task-weighted mean of deltas).
# --------------------------------------------------------------------------- #
def paired_task_bootstrap(
    deltas_by_task: dict[str, list[float]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """Point + percentile CI of the task-weighted mean, resampling tasks.

    Clusters are tasks; each bootstrap replicate resamples tasks with
    replacement and recomputes the task-weighted mean (matching
    ``_task_weighted_mean``). Paired because each delta is a within-chain
    (arm A error - arm B error) difference on the SAME held-out chain.
    """

    tasks = [np.asarray(v, dtype=float) for v in deltas_by_task.values() if len(v)]
    if not tasks:
        raise ValueError("no per-task deltas to bootstrap")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    point = float(np.mean([t.mean() for t in tasks]))
    rng = np.random.default_rng(seed)
    n = len(tasks)
    reps = np.empty(replicates, dtype=float)
    for b in range(replicates):
        idx = rng.integers(0, n, n)
        reps[b] = float(np.mean([tasks[i].mean() for i in idx]))
    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(reps, alpha)), float(np.quantile(reps, 1.0 - alpha))


def _mean_pairwise_jaccard(sets: Sequence[frozenset[str]]) -> float:
    """Mean Jaccard over set pairs; 1.0 for identical (or all-empty) sets."""

    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    if not pairs:
        return 1.0
    scores = []
    for a, b in pairs:
        union = a | b
        scores.append(1.0 if not union else len(a & b) / len(union))
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Grid + kill readout.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridConfig:
    fold_count: int
    screen: ScreenConfig
    primary_min_evidence: int
    k3_mass_agreement_floor: float
    bootstrap_replicates: int
    bootstrap_confidence: float
    bootstrap_seed: int


@dataclass
class _CellAccumulator:
    y_true: list[float] = field(default_factory=list)
    y_pred: list[float] = field(default_factory=list)
    task: list[str] = field(default_factory=list)


def _skip_predict(
    train: Sequence[Chain],
    *,
    min_evidence: int,
    skip: str,
    wtn_key_fn: Callable[[Chain], tuple[str, ...]],
) -> Callable[[Chain], float]:
    """Fit one grid cell's predictor. The three skip modes differ ONLY in the
    key derivation (the declared knob), everything else is the shared fitter.
    """

    if skip == "off":
        return fit_chain_prefix(
            train, max_depth=_PREFIX_DEPTH, min_evidence=min_evidence, skip_leading_cd=False
        )
    if skip == "cd-only":
        # Harness ablation, not method: the existing production leading-cd strip.
        return fit_chain_prefix(
            train, max_depth=_PREFIX_DEPTH, min_evidence=min_evidence, skip_leading_cd=True
        )
    if skip == "WTN":
        return fit_chain_prefix(
            train,
            max_depth=_PREFIX_DEPTH,
            min_evidence=min_evidence,
            skip_leading_cd=False,
            key_fn=wtn_key_fn,
        )
    raise ValueError(f"unknown skip mode {skip!r}")


_SKIP_MODES = ("off", "cd-only", "WTN")


def run_grid(chains: Sequence[Chain], cfg: GridConfig) -> dict[str, Any]:
    """Cross-fitted knob-matched grid + transparency screen + kill readout."""

    multi = [c for c in chains if len(c.segments) >= 2 and c.parent_command]
    cells: dict[tuple[str, int], _CellAccumulator] = {
        (skip, me): _CellAccumulator()
        for skip in _SKIP_MODES
        for me in _GRID_MIN_EVIDENCE
    }
    fold_screens: list[ScreenResult] = []

    for train, test in _task_folds(multi, _outer_config(cfg)):
        screen = screen_transparency(train, cfg.screen)
        fold_screens.append(screen)
        wtn_key_fn = make_wtn_key_fn(screen.applied, max_depth=_PREFIX_DEPTH)
        for me in _GRID_MIN_EVIDENCE:
            predictors = {
                skip: _skip_predict(
                    train, min_evidence=me, skip=skip, wtn_key_fn=wtn_key_fn
                )
                for skip in _SKIP_MODES
            }
            for chain in test:
                for skip in _SKIP_MODES:
                    acc = cells[(skip, me)]
                    acc.y_true.append(chain.parent_total_ms)
                    acc.y_pred.append(predictors[skip](chain))
                    acc.task.append(chain.task_id)

    cell_metrics = {
        f"{skip}@me{me}": _metrics_block(
            np.asarray(cells[(skip, me)].y_true),
            np.asarray(cells[(skip, me)].y_pred),
            tail_percentile=90.0,
        )
        for skip in _SKIP_MODES
        for me in _GRID_MIN_EVIDENCE
    }

    pairwise = _pairwise_deltas(cells, cfg)
    k1 = _k1(pairwise, cfg.primary_min_evidence)
    k2 = _k2(fold_screens)
    k3 = _k3(multi, fold_screens, cfg.k3_mass_agreement_floor)
    killed = k1["killed"] or k2["killed"] or k3["killed"]

    return {
        "grid": {
            "min_evidence_values": list(_GRID_MIN_EVIDENCE),
            "prefix_depth": _PREFIX_DEPTH,
            "cell_metrics": cell_metrics,
            "pairwise_deltas": pairwise,
        },
        "screen": _screen_report(fold_screens, cfg.screen),
        "kill_readout": {
            "K1_reproduce_win": k1,
            "K2_emergence_positive_control": k2,
            "K3_stability_mass_weighted": k3,
            "verdict": "KILL" if killed else "SURVIVE",
        },
    }


def _outer_config(cfg: GridConfig) -> Config:
    return Config(
        fold_count=cfg.fold_count,
        prefix_depth=_PREFIX_DEPTH,
        min_prefix_evidence=cfg.primary_min_evidence,
        atom_depth=_PREFIX_DEPTH,
        token_bin_count=3,
        min_atom_count=10,
        min_family_count=20,
        tail_percentile=90.0,
    )


def _pairwise_deltas(
    cells: dict[tuple[str, int], _CellAccumulator], cfg: GridConfig
) -> dict[str, Any]:
    """Task-clustered paired bootstrap CIs for WTN vs off and WTN vs cd-only.

    Statistic per pair = task-weighted mean of per-chain (|y - WTN| - |y -
    other|). Negative => WTN has lower error. Cells share the same test chains
    in the same order per fold, so the arrays align chain-for-chain.
    """

    out: dict[str, Any] = {}
    for me in _GRID_MIN_EVIDENCE:
        wtn = cells[("WTN", me)]
        for other in ("off", "cd-only"):
            oth = cells[(other, me)]
            deltas_by_task: dict[str, list[float]] = defaultdict(list)
            for task, y, p_wtn, p_oth in zip(wtn.task, wtn.y_true, wtn.y_pred, oth.y_pred):
                deltas_by_task[task].append(abs(y - p_wtn) - abs(y - p_oth))
            point, lo, hi = paired_task_bootstrap(
                deltas_by_task,
                replicates=cfg.bootstrap_replicates,
                confidence=cfg.bootstrap_confidence,
                seed=cfg.bootstrap_seed,
            )
            out[f"WTN_vs_{other}@me{me}"] = {
                "mean_delta_ms": point,
                "ci_low": lo,
                "ci_high": hi,
                "confidence": cfg.bootstrap_confidence,
                "tasks": len(deltas_by_task),
            }
    return out


def _k1(pairwise: dict[str, Any], primary_me: int) -> dict[str, Any]:
    """K1: WTN beats off (CI < 0) AND is not worse than cd-only (CI not > 0).

    Evaluated at the primary min_evidence (the cert operating point). Both
    matched-knob deltas are reported for every min_evidence in ``grid`` above;
    the verdict uses the primary point.
    """

    vs_off = pairwise[f"WTN_vs_off@me{primary_me}"]
    vs_cd = pairwise[f"WTN_vs_cd-only@me{primary_me}"]
    beats_off = vs_off["ci_high"] < 0.0  # WTN error strictly below off
    not_worse_than_cd = vs_cd["ci_low"] <= 0.0  # not significantly worse than cd-only
    killed = not (beats_off and not_worse_than_cd)
    return {
        "primary_min_evidence": primary_me,
        "wtn_vs_off": vs_off,
        "wtn_vs_cd_only": vs_cd,
        "beats_off_ci_excludes_zero": beats_off,
        "not_worse_than_cd_only": not_worse_than_cd,
        "killed": killed,
    }


def _k2(fold_screens: Sequence[ScreenResult]) -> dict[str, Any]:
    """K2: the cd positive-control class emerges transparent in every fold.

    The screen thresholds are frozen in ScreenConfig BEFORE any fold runs and
    are never a function of this outcome -- the assertion below makes that
    structural: emergence is read off the already-computed marginal sets, and
    ``_K2_POSITIVE_CONTROL_CLASS`` is a module constant, not a fitted value.
    """

    assert isinstance(_K2_POSITIVE_CONTROL_CLASS, str)
    per_fold = [_K2_POSITIVE_CONTROL_CLASS in s.marginal for s in fold_screens]
    non_degenerate = [s for s in fold_screens if not s.degenerate]
    # cd must appear as a candidate somewhere, else the control is untested.
    cd_is_candidate = any(
        _K2_POSITIVE_CONTROL_CLASS in s.universe for s in fold_screens
    )
    emerges_every_fold = bool(non_degenerate) and all(
        _K2_POSITIVE_CONTROL_CLASS in s.marginal for s in non_degenerate
    )
    killed = not (cd_is_candidate and emerges_every_fold)
    return {
        "positive_control_class": _K2_POSITIVE_CONTROL_CLASS,
        "candidate_present": cd_is_candidate,
        "per_fold_transparent": per_fold,
        "emerges_every_fold": emerges_every_fold,
        "killed": killed,
    }


def _k3(
    chains: Sequence[Chain],
    fold_screens: Sequence[ScreenResult],
    floor: float,
) -> dict[str, Any]:
    """K3: fraction of affected chain MASS whose class decisions are unanimous.

    A candidate class is unanimous if its marginal-transparency decision is the
    same across every (non-degenerate) fold (transparent in all, or in none). A
    chain is stable if every candidate class it contains is unanimous, so its
    normalized key would be identical regardless of fold assignment. The
    pre-registered (mass-weighted) statistic is stable/affected chain mass by
    ``parent_total_ms``; the unweighted chain-count fraction is a secondary
    diagnostic. Raw Jaccard on the tiny transparent set is degeneracy-prone and
    reported as a diagnostic only.
    """

    marginals = [s.marginal for s in fold_screens if not s.degenerate]
    all_candidates = sorted({v for s in fold_screens for v in s.universe})
    unanimous: dict[str, bool] = {}
    for verb in all_candidates:
        votes = [verb in m for m in marginals]
        unanimous[verb] = bool(marginals) and (all(votes) or not any(votes))

    affected = 0
    stable = 0
    affected_mass = 0.0
    stable_mass = 0.0
    for chain in chains:
        classes = set(candidate_classes(chain.parent_command))
        if not classes:
            continue
        affected += 1
        affected_mass += chain.parent_total_ms
        if all(unanimous.get(v, False) for v in classes):
            stable += 1
            stable_mass += chain.parent_total_ms
    mass_fraction = stable_mass / affected_mass if affected_mass > 0 else None
    count_fraction = stable / affected if affected else None
    jaccard = _mean_pairwise_jaccard(marginals) if marginals else 1.0
    killed = mass_fraction is None or mass_fraction < floor
    return {
        "mass_agreement_floor": floor,
        "affected_chains": affected,
        "stable_chains": stable,
        "affected_mass_ms": affected_mass,
        "stable_mass_ms": stable_mass,
        "mass_agreement_fraction": mass_fraction,
        "count_agreement_fraction": count_fraction,
        "mean_pairwise_jaccard_diagnostic": jaccard,
        "killed": killed,
    }


def _screen_report(
    fold_screens: Sequence[ScreenResult], cfg: ScreenConfig
) -> dict[str, Any]:
    """Marginal/joint/applied per fold + the sensitivity-grid selections."""

    folds = []
    for index, s in enumerate(fold_screens):
        folds.append(
            {
                "fold": index,
                "degenerate": s.degenerate,
                "universe": s.universe,
                "marginal_transparent": sorted(s.marginal),
                "joint_ok": s.joint_ok,
                "joint_stat_ms": s.joint_stat,
                "applied_transparent": sorted(s.applied),
                "fit_side_mae_ms": s.fit_side_mae,
                "per_class": s.per_class,
                "sensitivity_marginal": {
                    f"{tol:.3f}": sorted(s.marginal_at(tol))
                    for tol in cfg.sensitivity_tolerances
                },
            }
        )
    return {
        "config": {
            "tolerance": cfg.tolerance,
            "sensitivity_tolerances": list(cfg.sensitivity_tolerances),
            "min_tasks": cfg.min_tasks,
            "min_chains": cfg.min_chains,
            "inner_folds": cfg.inner_folds,
            "min_evidence": cfg.min_evidence,
        },
        "per_fold": folds,
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _banner(final: bool) -> str:
    if final:
        return "FINAL - complete corpus"
    return (
        "PARTIAL - not final: validated against a subset; numbers are for "
        "script validation only, not findings."
    )


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    cen = results["census"]
    grid = results["grid"]
    kill = results["kill_readout"]
    lines: list[str] = []
    lines.append("# WTN Stage-1: wrapper-transparency normalization")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Durations replayed on our own hardware "
        f"({provenance['replayed_on']}); segment_timeline v2. Generated "
        f"{provenance['generated']}."
    )
    lines.append("")
    lines.append(f"**Verdict: {kill['verdict']}**")
    lines.append("")

    # Step 0 census FIRST, before any accuracy number (spec).
    lines.append("## Step 0 - reachable-mass census (read before accuracy)")
    lines.append("")
    lines.append(
        f"{cen['multi_segment_chains']} multi-segment chains; "
        f"{cen['candidate_verb_classes']} candidate verb classes. "
        f"**Framing quantity -- chains with a wrapper BEYOND the leading "
        f"segment (mass a leading-only stripper cannot reach): "
        f"{cen['chains_beyond_leading_segment']} "
        f"({cen['chains_beyond_leading_fraction']:.1%} of chains, "
        f"{cen['beyond_leading_mass_fraction']:.1%} of parent_total_ms mass).** "
        f"Upper bound incl. leading cd (near-vacuous -- every multi-segment "
        f"chain has a non-final segment): {cen['chains_with_candidate_upper_bound']} "
        f"({cen['chains_with_candidate_upper_bound_fraction']:.1%})."
    )
    lines.append("")
    lines.append("| verb class | chains | tasks |")
    lines.append("| --- | --- | --- |")
    for row in cen["per_class"]:
        lines.append(
            f"| `{row['verb_class']}` | {row['chains']} | {row['tasks']} |"
        )
    lines.append("")
    lines.append(
        f"> Footnote (population mismatch): {cen['newline_only_chains']} "
        "multi-segment chains use newline-separated (or otherwise unsplit) "
        "commands that xtrace splits into segments but token-level candidate "
        "extraction sees as a single segment. They sit in the denominator with "
        "an empty candidate set; the beyond-leading framing quantity is "
        "unaffected (it requires a detected wrapper)."
    )
    lines.append("")

    lines.append("## Knob-matched grid (out-of-sample, task-grouped folds)")
    lines.append("")
    lines.append(
        f"Target: chain `parent_total_ms`. Depth {grid['prefix_depth']} fixed; "
        "cells differ only in skip mode and min_evidence. tail = P90+."
    )
    lines.append("")
    lines.append("| cell | R^2 | MAE ms | tail MAE ms | count |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, block in grid["cell_metrics"].items():
        lines.append(
            f"| {name} | {_fmt(block.get('r2'))} | {_fmt(block.get('mae_ms'))} | "
            f"{_fmt(block.get('tail_mae_ms'))} | {block.get('count')} |"
        )
    lines.append("")
    lines.append("### Pre-registered pairwise deltas (paired task-clustered bootstrap)")
    lines.append("")
    lines.append(
        "Statistic: task-weighted mean of per-chain (|y-WTN| - |y-other|); "
        "negative = WTN lower error."
    )
    lines.append("")
    lines.append("| pair | mean delta ms | CI low | CI high | tasks |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, block in grid["pairwise_deltas"].items():
        lines.append(
            f"| {name} | {_fmt(block['mean_delta_ms'])} | {_fmt(block['ci_low'])} "
            f"| {_fmt(block['ci_high'])} | {block['tasks']} |"
        )
    lines.append("")

    lines.append("## Transparency screen (sole gate)")
    lines.append("")
    scfg = results["screen"]["config"]
    lines.append(
        f"Primary tolerance {scfg['tolerance']:.0%} of fit-side MAE; min support "
        f"{scfg['min_tasks']} tasks AND {scfg['min_chains']} chains; "
        f"{scfg['inner_folds']} nested inner folds; evidence gate "
        f"{scfg['min_evidence']}. Sensitivity tolerances: "
        f"{scfg['sensitivity_tolerances']}."
    )
    lines.append("")
    for fold in results["screen"]["per_fold"]:
        lines.append(
            f"- fold {fold['fold']}: marginal={fold['marginal_transparent']} "
            f"joint_ok={fold['joint_ok']} applied={fold['applied_transparent']} "
            f"(fit-side MAE {_fmt(fold['fit_side_mae_ms'])} ms)"
        )
    lines.append("")

    lines.append("## Kill readout")
    lines.append("")
    k1 = kill["K1_reproduce_win"]
    k2 = kill["K2_emergence_positive_control"]
    k3 = kill["K3_stability_mass_weighted"]
    lines.append(
        f"- **K1** (reproduce win, me={k1['primary_min_evidence']}): "
        f"beats off (CI<0)={k1['beats_off_ci_excludes_zero']}, "
        f"not worse than cd-only={k1['not_worse_than_cd_only']} -> "
        f"{'KILL' if k1['killed'] else 'pass'}"
    )
    lines.append(
        f"- **K2** (cd emergence positive control): candidate_present="
        f"{k2['candidate_present']}, per-fold transparent="
        f"{k2['per_fold_transparent']} -> {'KILL' if k2['killed'] else 'pass'}"
    )
    lines.append(
        f"- **K3** (mass-weighted stability): agreement "
        f"{_fmt(k3['mass_agreement_fraction'])} vs floor "
        f"{k3['mass_agreement_floor']:.0%} (Jaccard diagnostic "
        f"{_fmt(k3['mean_pairwise_jaccard_diagnostic'])}) -> "
        f"{'KILL' if k3['killed'] else 'pass'}"
    )
    lines.append("")
    lines.append(f"**Verdict: {kill['verdict']}**")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.wave_*.worker_*.jsonl")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=None, help="Smoke only: cap wave files loaded."
    )
    parser.add_argument(
        "--screen-tolerance",
        type=float,
        default=0.05,
        help="Transparency tolerance as a fraction of fit-side MAE (default 0.05).",
    )
    parser.add_argument(
        "--screen-min-tasks",
        type=int,
        default=5,
        help="Min fit-fold tasks for a class to be poolable (default 5).",
    )
    parser.add_argument(
        "--screen-min-chains",
        type=int,
        default=20,
        help="Min fit-fold chains for a class to be poolable (default 20).",
    )
    parser.add_argument("--screen-inner-folds", type=int, default=5)
    parser.add_argument(
        "--primary-min-evidence",
        type=int,
        default=1,
        help="min_evidence the K1 verdict is read at (cert operating point, 1).",
    )
    parser.add_argument(
        "--k3-mass-agreement-floor",
        type=float,
        default=0.9,
        help="K3 kills below this fraction of affected mass with unanimous "
        "cross-fold class decisions (default 0.9).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/wrapper-transparency-stage1-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    files = sorted(args.traces_dir.glob(args.glob))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"no trace files under {args.traces_dir}/{args.glob}")
    samples = extract_many_segment_latency_samples(files, skip_concurrent=True)
    chains = build_chains(samples)
    multi = [c for c in chains if len(c.segments) >= 2 and c.parent_command]

    # Screen config is frozen HERE, before any fold or emergence check runs
    # (K2 integrity: thresholds are never a function of the results).
    screen_cfg = ScreenConfig(
        tolerance=args.screen_tolerance,
        min_tasks=args.screen_min_tasks,
        min_chains=args.screen_min_chains,
        inner_folds=args.screen_inner_folds,
    )
    grid_cfg = GridConfig(
        fold_count=args.fold_count,
        screen=screen_cfg,
        primary_min_evidence=args.primary_min_evidence,
        k3_mass_agreement_floor=args.k3_mass_agreement_floor,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )

    census_block = census(multi)
    grid_results = run_grid(multi, grid_cfg)
    provenance = {
        "exploratory": True,
        "final": bool(args.final),
        "replayed_on": "our_hardware",
        "traces_dir": str(args.traces_dir),
        "corpus_file_count": len(files),
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    # census FIRST in the payload dict (emitted before any accuracy number).
    payload = {
        "provenance": provenance,
        "config": {
            "fold_count": grid_cfg.fold_count,
            "prefix_depth": _PREFIX_DEPTH,
            "grid_min_evidence": list(_GRID_MIN_EVIDENCE),
            "primary_min_evidence": grid_cfg.primary_min_evidence,
            "k3_mass_agreement_floor": grid_cfg.k3_mass_agreement_floor,
        },
        "census": census_block,
        **grid_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(payload, provenance), encoding="utf-8")

    kill = grid_results["kill_readout"]
    print(
        f"verdict={kill['verdict']} "
        f"census beyond-leading: {census_block['chains_beyond_leading_segment']} "
        f"chains ({census_block['chains_beyond_leading_fraction']:.1%}, "
        f"{census_block['beyond_leading_mass_fraction']:.1%} mass), "
        f"{census_block['candidate_verb_classes']} classes"
    )
    print(
        f"  K1={'KILL' if kill['K1_reproduce_win']['killed'] else 'pass'} "
        f"K2={'KILL' if kill['K2_emergence_positive_control']['killed'] else 'pass'} "
        f"K3={'KILL' if kill['K3_stability_mass_weighted']['killed'] else 'pass'}"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
