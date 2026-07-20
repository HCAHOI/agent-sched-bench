#!/usr/bin/env python3
"""Footprint-aware pricing headroom screen: is one fixed swap price leaving value?

Pre-registered in ``analysis/pressure-headroom-design-20260720.md``. The screen
bounds the value of EVERY pressure-aware swap policy before any GPU hour is
committed to the direction.

The structural observation it rests on (verified in
``tool_latency_utility_clock.evaluate_utility_clock_policy``): ``threshold_ms =
kv_cost_ms + guard_ms`` and ``restore_cost_ms = restore_cost_fraction *
kv_cost_ms``. So ``kv_cost_ms`` IS the price of the swap action, and the
existing kv sweep is already a STATIC pressure sweep. A pressure-conditioned
policy is the same policy with ``kv_cost_ms`` replaced by a time-varying
``lambda(t)``; its ceiling is therefore measurable offline.

WHICH lambda -- and why NOT the spec's first choice
---------------------------------------------------
The spec's primary lambda was replayed concurrent occupancy. Measured over the
frozen fresh-277 corpus, that signal DOES NOT EXIST here and must not be used:

* Within-task call overlap is a timestamp artifact, not concurrency: 336
  adjacent overlapping pairs, max 22ms, median 1.2ms, 100% under 50ms, all on
  back-to-back fast reads. The agent is strictly sequential.
* Cross-task co-occupancy is exactly the collection harness's worker count
  (peak simultaneously-active tasks 2, median 2). It encodes a ``--concurrency``
  flag, not a workload property.
* The apparent occupancy tail (3+) is the same tick artifact: those calls have
  mean latency ~2ms and contain ZERO calls above the smallest headline
  threshold.
* Collection ran against a cloud provider -- no shared KV cache existed, so no
  memory contention could have been recorded.

Pricing off that would dress a harness flag as physics on a near-binary signal
and would manufacture a near-zero headroom, i.e. a DROP caused by the corpus
lacking pressure variation rather than by pressure-conditioning lacking value.

THE LAMBDA USED INSTEAD is the call's own resident KV footprint at its decision
instant, read from ``llm_call.data.prompt_tokens`` and joined to each tool call
on ``(source_trace, iteration)``. This is not a proxy: ``kv_cost_ms`` is the
time to move the KV cache, KV bytes are linear in resident tokens, so the swap
price is linear in ``prompt_tokens`` by construction. In this corpus it covers
13410/13410 calls and spans 1912..85189 tokens, growing a median 13.4x within
every one of the 277 tasks.

Map (config, documented, NOT fitted to outcomes)::

    lambda_i = kv_cost_ms * tokens_i / reference_tokens

``reference_tokens`` is the FIT-FOLD mean footprint, so each kv panel cell is a
mean pressure level with real per-call variation around it and the eval fold
never touches the anchor. The map is strictly increasing in tokens and depends
on no outcome.

Arms (the ONLY difference is what price the policy BELIEVED)
------------------------------------------------------------
Every call's realized utility is always scored at its TRUE price ``lambda_i``
-- that is physics, the swap really costs that. What differs is the trigger:

* **Footprint-priced:** the trigger is optimized for the call's own ``lambda_i``.
* **Fixed-price (status quo):** the trigger is optimized for ONE constant
  ``lambda_bar``, selected on FIT FOLDS ONLY by maximizing the same certified
  utility functional (``hazard_recheck_ms`` for the trigger,
  ``trigger_policy_utility_ms`` / ``utility_matrix`` for the score) that the
  shipped policy uses. This is what the certified policy does TODAY.

``headroom = footprint_priced - fixed_price``, in seconds per 277 tasks, with a
task-clustered CI and a permutation label from the certified engine.

WHERE THE NON-NEGATIVITY GUARANTEE ACTUALLY LIVES (stated, not assumed). The
footprint_priced trigger maximizes the node-EXPECTED utility at the call's true
price, so in node-expectation it dominates the trigger induced by ANY constant
believed price -- that is the construction that makes the headroom a ceiling.
It is NOT a pointwise guarantee on realized held-out latencies, and the
best-fixed arm is deliberately given the stronger deal: ``lambda_bar`` is chosen
to maximize the REALIZED fit-fold total, not a node expectation. That
asymmetry is intentional and conservative -- it makes the no-pressure arm as
strong as the fit folds allow, so the reported headroom is a lower bound on the
true gap and a small NEGATIVE headroom is a legitimate outcome meaning a
constant price is already effectively optimal.

WHAT THIS ACTUALLY TESTS (reframe, team-lead ruling 2026-07-20)
---------------------------------------------------------------
Because ``lambda_i`` is known at call start and constant for the call's
duration, this is NOT a test of time-varying pressure. It is a test of PER-CALL
FOOTPRINT-AWARE PRICING. The certified policy today prices every call at ONE
fixed ``kv_cost_ms`` while real footprints vary 44x across the corpus and grow
by roughly an order of magnitude WITHIN a single task (the mechanism figure
``footprint_growth`` is computed per run and reported alongside the verdict).
So a positive result here is not merely "headroom exists for some future
pressure policy" -- it is a directly shippable policy iteration.

Correspondingly the arm status is: the footprint-priced arm consumes NO
hindsight, so calling it an oracle would be inaccurate. It is a TIGHT and
ATTAINABLE bound. The binding limitation is the AXIS: this screens the
SELF-FOOTPRINT axis only. The multi-tenant CONTENTION axis (my price rising
because co-tenants are resident) is a STRUCTURAL NEGATIVE on this corpus -- see
above -- and no result here, positive or null, says anything about it.

A SURVIVE HERE DOES NOT LICENSE DEPLOYMENT. This is a screen. A positive
headroom triggers a SEPARATE certified decision replay -- paired against the
frozen certified policy at rho=0.94, permutation per kv cell, full H1 discipline
-- exactly as A2 required its robust-clock re-confirmation. The number this
script prints must NEVER be quoted as a certified gain.

Pre-registered kill criterion (FROZEN before code, spec section "Pre-registered
kill criterion") and POWER RULE (pre-registered 2026-07-20, before any
full-corpus number existed). Three-way verdict at the headline kv cell against
the banked pre-restore effect (``--banked-seconds-per-277``, default 156.0
s/277 at kv3500), read off the task-clustered SIMULTANEOUS CI:

* **PROCEED** iff the CI LOWER bound exceeds the bar and the permutation CI
  excludes zero -- headroom materially beats what is already held.
* **DROP** (direction CLOSED, becomes a C3 sentence) iff the CI UPPER bound is
  BELOW the bar. Only then has the screen actually demonstrated the headroom is
  under the bar.
* **UNDERPOWERED** (direction NOT closed) iff the point estimate is below the
  bar but the CI spans it. Non-inferiority framing: a direction may only be
  closed if the screen could have detected the effect it is compared against.
  Such a result MUST NOT be cited as evidence of absent headroom and MUST NOT be
  recorded as a closed axis -- it is future work contingent on a corpus with
  more heavy-call mass.

A panel-coherence diagnostic (sign flips of the headroom across the kv panel)
is reported as a named secondary indicator supporting an UNDERPOWERED reading
when it occurs. It is DESCRIPTIVE ONLY -- the CI rule binds, so the verdict
cannot be argued either way after the fact.

Mandatory reporting in every state: the footprint-priced gap in seconds, the
fire fraction it scales against, and the fixed-price lambda selected.

EXPLORATORY until ``--final``. Emits JSON + MD to ``analysis/`` (``-PARTIAL``
unless ``--final``).

Usage (full corpus):
  uv run python scripts/analyze_pressure_headroom.py \
    --manifest analysis/fresh-corpus-certification-20260717/\
offline-gated-robust/manifest.json --final
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

# Allow direct `python scripts/analyze_pressure_headroom.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.adjudicate_k2_recheck import (  # noqa: E402
    _CERT_GUARD_MS,
    _CERT_RESTORE_COST_FRACTION,
    _banner,
    _git_sha,
    _load_manifest_corpus,
    _row_group_keys,
)
from trace_collect.tool_latency_dataset import (  # noqa: E402
    ToolLatencySample,
    discover_trace_files,
)
from trace_collect.tool_latency_profiled import (  # noqa: E402
    LatencyPriorNode,
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
)
from trace_collect.tool_latency_utility_clock import utility_matrix  # noqa: E402

# Certified statistics engine, reused verbatim (same replicates/seed/Bonferroni
# discipline as H1 and A2). We feed a precomputed contributions matrix because
# ``paired_task_cluster_bootstrap`` hardwires the swap-trigger utility.
from trace_collect.tool_latency_confirmation import (  # noqa: E402
    _permutation_simultaneous_labels,
    _resample_task_totals,
)

# Headline cell for the frozen kill readout: the banked pre-restore effect the
# spec names is quoted at kv3500, so the comparison must be made in that cell.
_HEADLINE_COST_MS = 3500.0

# The already-banked pre-restore effect the direction must beat to be worth
# chasing (spec: "~156 s/277 at kv3500"). Exposed as a flag so the frozen number
# is auditable, never silently redefined.
_BANKED_PRERESTORE_SECONDS_PER_277 = 156.0

# SECONDARY, NON-BINDING readout. The verdict compares 3500-to-3500; a reader
# will still ask what kv5000 looks like, so the answer belongs in the artifact
# rather than being inferred. Never feeds the verdict.
_SECONDARY_COST_MS = 5000.0
_SECONDARY_BANKED_SECONDS_PER_277 = 317.9


# --------------------------------------------------------------------------- #
# lambda(t): per-call resident KV footprint.
# --------------------------------------------------------------------------- #
def load_kv_footprint_tokens(trace_paths: Sequence[Path]) -> dict[tuple[str, int], float]:
    """Map ``(resolved trace path, iteration) -> prompt_tokens`` from llm_calls.

    ``prompt_tokens`` is the context resident in the KV cache when the agent
    emitted that iteration's tool call, i.e. the footprint a swap would move.
    It is recorded at call START and is unrelated to the call's outcome.
    """

    footprints: dict[tuple[str, int], float] = {}
    for path in trace_paths:
        key_path = str(path.resolve())
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                action = json.loads(line)
                if action.get("action_type") != "llm_call":
                    continue
                iteration = action.get("iteration")
                tokens = (action.get("data") or {}).get("prompt_tokens")
                if iteration is None or tokens is None:
                    continue
                tokens = float(tokens)
                if not np.isfinite(tokens) or tokens <= 0.0:
                    raise ValueError(
                        f"{path}: iteration {iteration} has non-positive "
                        f"prompt_tokens {tokens}"
                    )
                # First llm_call of an iteration is the one that emitted the
                # tool call; malformed-retry re-issues must not overwrite it.
                footprints.setdefault((key_path, int(iteration)), tokens)
    if not footprints:
        raise ValueError("no llm_call prompt_tokens found; cannot build a lambda")
    return footprints


def sample_footprint_tokens(
    sample: ToolLatencySample, footprints: dict[tuple[str, int], float]
) -> float:
    """Resident KV tokens for one tool call. Fails fast when unjoinable."""

    key = (str(Path(sample.source_trace).resolve()), int(sample.iteration))
    tokens = footprints.get(key)
    if tokens is None:
        raise ValueError(
            f"no prompt_tokens footprint for sample {sample.sample_id!r} "
            f"(trace={sample.source_trace}, iteration={sample.iteration}); "
            "the lambda trajectory would be incomplete"
        )
    return tokens


def footprint_growth_stats(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    footprints: dict[tuple[str, int], float],
) -> dict[str, Any]:
    """Mechanism figure: how much the KV footprint varies, within and across tasks.

    This is WHY footprint-aware pricing could matter at all -- a single fixed
    ``kv_cost_ms`` is charged across this entire spread. Reported next to the
    verdict. Computed from the corpus every run; nothing here is a constant.
    """

    ratios: list[float] = []
    all_tokens: list[float] = []
    for task_id in task_ids:
        tokens = [
            sample_footprint_tokens(sample, footprints)
            for sample in samples_by_task[task_id]
        ]
        all_tokens.extend(tokens)
        if len(tokens) >= 2:
            ratios.append(max(tokens) / min(tokens))
    if not all_tokens:
        raise ValueError("no footprints to summarize")
    corpus = np.asarray(all_tokens, dtype=float)
    growth = np.asarray(ratios, dtype=float) if ratios else np.zeros(0)
    return {
        "corpus_min_tokens": float(corpus.min()),
        "corpus_max_tokens": float(corpus.max()),
        "corpus_spread_ratio": float(corpus.max() / corpus.min()),
        "within_task_growth_tasks": int(growth.size),
        "within_task_growth_median": float(np.median(growth)) if growth.size else None,
        "within_task_growth_p90": (
            float(np.percentile(growth, 90)) if growth.size else None
        ),
        "within_task_growth_max": float(growth.max()) if growth.size else None,
    }


def pressure_price_ms(
    tokens: float, *, kv_cost_ms: float, reference_tokens: float
) -> float:
    """Swap price at a decision instant, given the resident KV footprint.

    Linear because KV bytes are linear in resident tokens and swap time is
    linear in bytes -- this is the physical price, not a fitted curve. Strictly
    increasing in ``tokens`` and independent of every outcome.
    ``reference_tokens`` is a FIT-FOLD anchor, so each kv panel cell reads as
    the price at the mean footprint.
    """

    if reference_tokens <= 0.0:
        raise ValueError(f"reference_tokens must be positive, got {reference_tokens}")
    if tokens <= 0.0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    return kv_cost_ms * tokens / reference_tokens


# --------------------------------------------------------------------------- #
# Config.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PressureHeadroomConfig:
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
    # Resolution of the fit-fold quantile grid the constant lambda is chosen
    # over. A resolution knob, not a tuned hyperparameter: the panel cell itself
    # is always a candidate, so a coarser grid can only weaken the FOOTPRINT-PRICED
    # side of the comparison, never inflate the headroom.
    fixed_lambda_grid: int = 21
    banked_seconds_per_277: float = _BANKED_PRERESTORE_SECONDS_PER_277
    secondary_banked_seconds_per_277: float = _SECONDARY_BANKED_SECONDS_PER_277

    def __post_init__(self) -> None:
        if self.fixed_lambda_grid < 2:
            raise ValueError(
                f"fixed_lambda_grid must be >= 2, got {self.fixed_lambda_grid}"
            )


# --------------------------------------------------------------------------- #
# Triggers and per-call scoring under a believed price.
# --------------------------------------------------------------------------- #
def believed_trigger_ms(
    node: LatencyPriorNode,
    *,
    believed_price_ms: float,
    guard_ms: float,
    restore_cost_fraction: float,
) -> float:
    """Certified k=1 optimal trigger for a policy that believes the price.

    Reuses ``hazard_recheck_ms`` verbatim under the certified price semantics
    (``threshold = price + guard``, ``restore = rho * price``). Empty/thin nodes
    fall back to the deadline inside the optimizer -- conservative by
    construction, no special-casing here.
    """

    return hazard_recheck_ms(
        node.values,
        threshold_ms=believed_price_ms + guard_ms,
        kv_cost_ms=believed_price_ms,
        restore_cost_ms=restore_cost_fraction * believed_price_ms,
    )


def realized_utilities_ms(
    latency_ms: float,
    triggers_ms: np.ndarray,
    *,
    true_price_ms: float,
    guard_ms: float,
    restore_cost_fraction: float,
) -> np.ndarray:
    """Score one call's triggers at its TRUE price via the shared functional.

    Vectorized over candidate triggers. ``utility_matrix`` is the public
    passthrough to the same ``hidden_on_long - exposed - rho*restore``
    accounting the certified policies use -- no reimplementation.
    """

    return utility_matrix(
        np.asarray([latency_ms], dtype=float),
        np.asarray(triggers_ms, dtype=float),
        threshold_ms=true_price_ms + guard_ms,
        kv_cost_ms=true_price_ms,
        restore_cost_ms=restore_cost_fraction * true_price_ms,
    )[0]


# --------------------------------------------------------------------------- #
# Fit-fold selection of the single constant lambda.
# --------------------------------------------------------------------------- #
def fixed_lambda_candidates(
    fit_prices_ms: np.ndarray, *, kv_cost_ms: float, grid: int
) -> np.ndarray:
    """Constant-price candidates: fit-fold price quantiles plus the panel cell.

    Including ``kv_cost_ms`` guarantees the best-fixed arm is never worse than
    the SHIPPED constant-price policy, so the reported headroom is a
    conservative lower bound on what a constant price already achieves.
    """

    quantiles = np.linspace(0.0, 1.0, grid)
    candidates = np.quantile(fit_prices_ms, quantiles, method="linear")
    candidates = np.append(candidates, float(kv_cost_ms))
    candidates = candidates[candidates > 0.0]
    return np.unique(np.round(candidates, 6))


def select_fixed_lambda_ms(
    fit_calls: Sequence[tuple[LatencyPriorNode, float, float]],
    *,
    candidates: np.ndarray,
    kv_cost_ms: float,
    guard_ms: float,
    restore_cost_fraction: float,
) -> tuple[float, float]:
    """Pick the constant price maximizing FIT-FOLD utility. Returns (lambda, total).

    ``fit_calls`` are ``(node, latency_ms, true_price_ms)`` triples from the
    PROFILE folds only -- the eval fold never enters this selection, which is
    the cross-fit isolation the certified pipeline requires.

    The objective is the same utility the shipped policy is optimized under:
    each fit call is triggered by the believed constant price and then scored at
    its own true price. Ties resolve to the candidate nearest the panel cell
    ``kv_cost_ms``, i.e. the status quo is preferred unless strictly beaten --
    which makes the best-fixed arm as strong as possible and the headroom
    conservative.
    """

    trigger_cache: dict[tuple[int, float], float] = {}
    totals = np.zeros(len(candidates), dtype=float)
    for node, latency_ms, true_price_ms in fit_calls:
        triggers = np.empty(len(candidates), dtype=float)
        for index, believed in enumerate(candidates):
            key = (id(node.values), float(believed))
            trigger = trigger_cache.get(key)
            if trigger is None:
                trigger = believed_trigger_ms(
                    node,
                    believed_price_ms=float(believed),
                    guard_ms=guard_ms,
                    restore_cost_fraction=restore_cost_fraction,
                )
                trigger_cache[key] = trigger
            triggers[index] = trigger
        totals += realized_utilities_ms(
            latency_ms,
            triggers,
            true_price_ms=true_price_ms,
            guard_ms=guard_ms,
            restore_cost_fraction=restore_cost_fraction,
        )
    best = float(np.max(totals))
    tied = np.flatnonzero(totals >= best - 1e-9)
    winner = tied[int(np.argmin(np.abs(candidates[tied] - kv_cost_ms)))]
    return float(candidates[winner]), float(totals[winner])


# --------------------------------------------------------------------------- #
# Cross-fitted scoring over the certified folds.
# --------------------------------------------------------------------------- #
def score_decisions(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    footprints: dict[tuple[str, int], float],
    cfg: PressureHeadroomConfig,
) -> list[dict[str, Any]]:
    """Per-(call, cost) footprint_priced vs best-fixed rows over the certified folds."""

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

        def node_for(sample: ToolLatencySample) -> LatencyPriorNode:
            row = sample.to_json_obj()
            return latency_prior_hierarchy(
                prior,
                str(row["tool_name"]),
                row_group_keys(row),
                min_tool_history=cfg.min_tool_history,
                min_profile_tasks=cfg.min_profile_tasks,
            )[-1]

        # Fit-fold anchor and selection set: PROFILE tasks only.
        fit_samples = [
            sample for task_id in sorted(profile_tasks) for sample in samples_by_task[task_id]
        ]
        fit_tokens = np.asarray(
            [sample_footprint_tokens(s, footprints) for s in fit_samples], dtype=float
        )
        reference_tokens = float(np.mean(fit_tokens))
        fit_nodes = [node_for(s) for s in fit_samples]

        for kv_cost_ms in cfg.costs_ms:
            fit_prices = np.asarray(
                [
                    pressure_price_ms(
                        float(tok),
                        kv_cost_ms=kv_cost_ms,
                        reference_tokens=reference_tokens,
                    )
                    for tok in fit_tokens
                ],
                dtype=float,
            )
            candidates = fixed_lambda_candidates(
                fit_prices, kv_cost_ms=kv_cost_ms, grid=cfg.fixed_lambda_grid
            )
            fixed_lambda_ms, _ = select_fixed_lambda_ms(
                list(zip(fit_nodes, (s.latency_ms for s in fit_samples), fit_prices)),
                candidates=candidates,
                kv_cost_ms=kv_cost_ms,
                guard_ms=cfg.guard_ms,
                restore_cost_fraction=cfg.restore_cost_fraction,
            )
            fixed_trigger_cache: dict[int, float] = {}
            for task_id in sorted(eval_tasks):
                for sample in samples_by_task[task_id]:
                    node = node_for(sample)
                    tokens = sample_footprint_tokens(sample, footprints)
                    true_price_ms = pressure_price_ms(
                        tokens,
                        kv_cost_ms=kv_cost_ms,
                        reference_tokens=reference_tokens,
                    )
                    footprint_priced_trigger = believed_trigger_ms(
                        node,
                        believed_price_ms=true_price_ms,
                        guard_ms=cfg.guard_ms,
                        restore_cost_fraction=cfg.restore_cost_fraction,
                    )
                    fixed_trigger = fixed_trigger_cache.get(id(node.values))
                    if fixed_trigger is None:
                        fixed_trigger = believed_trigger_ms(
                            node,
                            believed_price_ms=fixed_lambda_ms,
                            guard_ms=cfg.guard_ms,
                            restore_cost_fraction=cfg.restore_cost_fraction,
                        )
                        fixed_trigger_cache[id(node.values)] = fixed_trigger
                    footprint_priced_ms, fixed_ms = realized_utilities_ms(
                        sample.latency_ms,
                        np.asarray([footprint_priced_trigger, fixed_trigger], dtype=float),
                        true_price_ms=true_price_ms,
                        guard_ms=cfg.guard_ms,
                        restore_cost_fraction=cfg.restore_cost_fraction,
                    )
                    decisions.append(
                        {
                            "sample_id": sample.sample_id,
                            "task_id": task_id,
                            "tool_name": sample.tool_name,
                            "outer_fold": f"f{fold}",
                            "latency_ms": sample.latency_ms,
                            "prior_source": node.source,
                            "prior_group_key": node.group_key,
                            "kv_cost_ms": kv_cost_ms,
                            "footprint_tokens": tokens,
                            "reference_tokens": reference_tokens,
                            "true_price_ms": true_price_ms,
                            "fixed_lambda_ms": fixed_lambda_ms,
                            "footprint_priced_trigger_ms": footprint_priced_trigger,
                            "fixed_trigger_ms": fixed_trigger,
                            "footprint_priced_utility_ms": float(footprint_priced_ms),
                            "fixed_utility_ms": float(fixed_ms),
                            "headroom_ms": float(footprint_priced_ms - fixed_ms),
                            "footprint_priced_fired": sample.latency_ms > footprint_priced_trigger,
                            "fixed_fired": sample.latency_ms > fixed_trigger,
                            # EARLY fires (strictly before the call's deadline)
                            # are the decisions the policy actually adds over
                            # deadline_only -- the rate the headroom scales
                            # against, and the one the spec quotes.
                            "footprint_priced_fired_early": (
                                sample.latency_ms > footprint_priced_trigger
                                and footprint_priced_trigger < true_price_ms + cfg.guard_ms
                            ),
                            "fixed_fired_early": (
                                sample.latency_ms > fixed_trigger
                                and fixed_trigger < true_price_ms + cfg.guard_ms
                            ),
                        }
                    )
    return decisions


# --------------------------------------------------------------------------- #
# Aggregation, certificate, frozen kill readout.
# --------------------------------------------------------------------------- #
def _contributions_matrix(
    decisions: Sequence[dict[str, Any]], costs_ms: Sequence[float], field: str
) -> tuple[np.ndarray, list[str]]:
    """Per (task, kv) summed ``field`` in ms. Rows are logical tasks."""

    task_ids = sorted({str(row["task_id"]) for row in decisions})
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    cost_index = {cost: index for index, cost in enumerate(costs_ms)}
    contributions = np.zeros((len(task_ids), len(costs_ms)), dtype=float)
    for row in decisions:
        contributions[
            task_index[str(row["task_id"])], cost_index[float(row["kv_cost_ms"])]
        ] += float(row[field])
    return contributions, task_ids


def _certificate(
    contributions: np.ndarray, cfg: PressureHeadroomConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Certified engine: percentile CI + sign-flip permutation labels."""

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


def _panel_coherence(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """DESCRIPTIVE ONLY: how ragged is the headroom across the kv panel?

    A coherent panel moves smoothly with kv; sign flips from cell to cell
    indicate the fixed-lambda selection is noise-dominated. This SUPPORTS an
    UNDERPOWERED reading when it occurs but never decides the verdict -- the CI
    rule binds, so the verdict cannot be argued either way after the fact.
    """

    values = [float(cell["headroom_seconds_per_277"]) for cell in cells]
    flips = sum(
        1 for a, b in zip(values, values[1:]) if (a > 0.0) != (b > 0.0)
    )
    return {
        "binding": False,
        "sign_flips_across_panel": flips,
        "negative_cells": sum(1 for v in values if v < 0.0),
        "cell_count": len(values),
        "incoherent": flips >= 2,
    }


def summarize(
    decisions: Sequence[dict[str, Any]], cfg: PressureHeadroomConfig, *, task_count: int
) -> dict[str, Any]:
    """Per-cell headroom table, task-clustered CI, and the frozen kill readout."""

    if _HEADLINE_COST_MS not in cfg.costs_ms:
        raise ValueError(
            f"cost panel is missing the headline kv cell {_HEADLINE_COST_MS}; the "
            "banked-effect comparison is undefined -- refusing to emit a vacuous "
            "verdict"
        )
    contributions, _ = _contributions_matrix(decisions, cfg.costs_ms, "headroom_ms")
    observed, point_q, simul_q, permutation = _certificate(contributions, cfg)

    rows_by_cost: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        rows_by_cost[float(row["kv_cost_ms"])].append(row)

    cells: list[dict[str, Any]] = []
    for column, cost in enumerate(cfg.costs_ms):
        rows = rows_by_cost[cost]
        headroom_ms = float(observed[column])
        footprint_priced_total = sum(float(r["footprint_priced_utility_ms"]) for r in rows)
        fixed_total = sum(float(r["fixed_utility_ms"]) for r in rows)
        assert abs(headroom_ms - (footprint_priced_total - fixed_total)) < 1e-6, (
            f"headroom {headroom_ms} != footprint_priced {footprint_priced_total} - fixed "
            f"{fixed_total}"
        )
        selected = sorted({float(r["fixed_lambda_ms"]) for r in rows})
        cells.append(
            {
                "kv_cost_ms": cost,
                "headline": cost == _HEADLINE_COST_MS,
                "headroom_ms_per_277": headroom_ms,
                "headroom_seconds_per_277": headroom_ms / 1000.0,
                "footprint_priced_seconds_per_277": footprint_priced_total / 1000.0,
                "best_fixed_seconds_per_277": fixed_total / 1000.0,
                "fixed_lambda_ms_by_fold": selected,
                "mean_fixed_lambda_ms": float(np.mean(selected)),
                "footprint_priced_fire_fraction": (
                    sum(1 for r in rows if r["footprint_priced_fired"]) / len(rows)
                    if rows
                    else 0.0
                ),
                "fixed_fire_fraction": (
                    sum(1 for r in rows if r["fixed_fired"]) / len(rows)
                    if rows
                    else 0.0
                ),
                "footprint_priced_early_fire_fraction": (
                    sum(1 for r in rows if r.get("footprint_priced_fired_early")) / len(rows)
                    if rows
                    else 0.0
                ),
                "fixed_early_fire_fraction": (
                    sum(1 for r in rows if r.get("fixed_fired_early")) / len(rows)
                    if rows
                    else 0.0
                ),
                "mean_true_price_ms": float(
                    np.mean([float(r["true_price_ms"]) for r in rows])
                ),
                "call_count": len(rows),
                "pointwise_interval_ms": {
                    "low": float(point_q[0, column]),
                    "high": float(point_q[1, column]),
                },
                "simultaneous_interval_ms": {
                    "low": float(simul_q[0, column]),
                    "high": float(simul_q[1, column]),
                },
                **permutation["points"][column],
            }
        )

    headline = next(cell for cell in cells if cell["headline"])
    headroom_s = headline["headroom_seconds_per_277"]
    ci_excludes_zero = headline["permutation_label"] == "positive"
    exceeds_banked = headroom_s > cfg.banked_seconds_per_277
    # POWER RULE (pre-registered 2026-07-20, before any full-corpus number
    # existed). Non-inferiority framing: a direction may only be CLOSED if the
    # screen could actually have detected the effect it is compared against, so
    # DROP requires the CI upper bound to sit BELOW the bar -- a point estimate
    # below the bar with a CI spanning it is UNDERPOWERED, not evidence of
    # absence.
    bar_s = cfg.banked_seconds_per_277
    ci_low_s = headline["simultaneous_interval_ms"]["low"] / 1000.0
    ci_high_s = headline["simultaneous_interval_ms"]["high"] / 1000.0
    if ci_low_s > bar_s and ci_excludes_zero:
        verdict = "PROCEED"
    elif ci_high_s < bar_s:
        verdict = "DROP"
    else:
        verdict = "UNDERPOWERED"
    secondary = next(
        (c for c in cells if c["kv_cost_ms"] == _SECONDARY_COST_MS), None
    )
    return {
        "verdict": verdict,
        # NON-BINDING context only; deliberately not part of the verdict.
        "secondary_readout": (
            {
                "binding": False,
                "kv_cost_ms": _SECONDARY_COST_MS,
                "headroom_seconds_per_277": secondary["headroom_seconds_per_277"],
                "banked_seconds_per_277": cfg.secondary_banked_seconds_per_277,
                "exceeds_banked": (
                    secondary["headroom_seconds_per_277"]
                    > cfg.secondary_banked_seconds_per_277
                ),
                "permutation_label": secondary["permutation_label"],
            }
            if secondary is not None
            else None
        ),
        "kill_criterion": (
            "POWER RULE at the headline kv cell "
            f"({_HEADLINE_COST_MS:.0f}) against the banked pre-restore effect "
            f"({cfg.banked_seconds_per_277:.0f} s/277), using the task-clustered "
            "simultaneous CI (Bonferroni over the full cost family). PROCEED iff "
            "the CI LOWER bound exceeds the bar with the permutation CI "
            "excluding zero. DROP (direction closed) iff the CI UPPER bound is "
            "BELOW the bar. Otherwise UNDERPOWERED: the point estimate is below "
            "the bar but the CI spans it, so the corpus cannot resolve an effect "
            "of the size we care about and the direction is NOT closed"
        ),
        "power_rule": {
            "bar_seconds_per_277": bar_s,
            "ci_low_seconds_per_277": ci_low_s,
            "ci_high_seconds_per_277": ci_high_s,
            "ci_upper_below_bar": ci_high_s < bar_s,
            "ci_lower_above_bar": ci_low_s > bar_s,
            "direction_closed": verdict == "DROP",
        },
        "panel_coherence": _panel_coherence(cells),
        "headline_cost_ms": _HEADLINE_COST_MS,
        "banked_seconds_per_277": cfg.banked_seconds_per_277,
        "headline_headroom_seconds_per_277": headroom_s,
        # Mandatory reporting (spec): the footprint_priced gap, the fire fraction it
        # scales against, and the selected best-fixed lambda.
        "headline_footprint_priced_seconds_per_277": headline["footprint_priced_seconds_per_277"],
        "headline_best_fixed_seconds_per_277": headline["best_fixed_seconds_per_277"],
        "headline_footprint_priced_early_fire_fraction": headline[
            "footprint_priced_early_fire_fraction"
        ],
        "headline_fixed_lambda_ms": headline["mean_fixed_lambda_ms"],
        "headline_exceeds_banked": exceeds_banked,
        "headline_ci_excludes_zero": ci_excludes_zero,
        "lambda_signal": "resident KV footprint (llm_call.prompt_tokens)",
        "pressure_axis": "self-footprint only; multi-tenant contention unscreenable",
        "task_count": task_count,
        "call_count": len({str(r["sample_id"]) for r in decisions}),
        "cells": cells,
        "permutation_config": permutation["config"],
    }


def run_pressure_headroom(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    footprints: dict[tuple[str, int], float],
    cfg: PressureHeadroomConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every fold's held-out calls and apply the frozen kill readout."""

    decisions = score_decisions(samples_by_task, task_ids, footprints, cfg)
    summary = summarize(decisions, cfg, task_count=len(task_ids))
    summary["footprint_growth"] = footprint_growth_stats(
        samples_by_task, task_ids, footprints
    )
    return summary, decisions


# --------------------------------------------------------------------------- #
# Rendering / CLI.
# --------------------------------------------------------------------------- #
_ARM_STATUS_CAVEAT = (
    "> **Arm status (read with the verdict).** The footprint-priced arm prices "
    "each call at its OWN resident KV footprint at its decision instant. An "
    "agent's context is frozen while a tool call runs, so that footprint is "
    "known at call start and equals the footprint at the swap instant: the arm "
    "consumes NO hindsight and is an ATTAINABLE bound, NOT an oracle and not "
    "unshippable. Because the price is known at call start and constant for the "
    "call, this is not a test of time-varying pressure -- it is a test of "
    "**per-call footprint-aware pricing**, i.e. a directly shippable policy "
    "iteration over a certified policy that currently charges ONE fixed "
    "`kv_cost_ms` to every call."
)

_AXIS_NEGATIVE = (
    "> **Multi-tenant contention: STRUCTURAL NEGATIVE on this corpus.** This "
    "screen bounds the SELF-FOOTPRINT axis only and does NOT bound the "
    "contention axis. The spec's original lambda (replayed concurrent occupancy) "
    "does not exist here, and pricing off it was refused. The measurement, "
    "verbatim: within-task call overlap is a timestamp artifact (336 adjacent "
    "pairs, max 22ms, median 1.2ms, 100% under 50ms, all on back-to-back fast "
    "reads -- the agent is strictly sequential); cross-task co-occupancy is "
    "exactly the collection harness's worker count (peak simultaneously-active "
    "tasks 2, median 2, mean 1.59); the apparent occupancy tail (3..9) is the "
    "same tick artifact, those calls averaging ~2ms latency and containing ZERO "
    "calls above any headline threshold; and collection ran against a cloud "
    "provider, so no shared KV cache existed and no contention could have been "
    "recorded. Pricing lambda off that would dress a `--concurrency 2` flag as "
    "physics on a near-binary signal and would manufacture a near-zero headroom "
    "-- a DROP caused by the corpus lacking pressure variation rather than by "
    "the direction lacking value. That is a false negative dressed as an "
    "empirical screen, and it is why this artifact reports the contention axis "
    "as unscreenable instead of reporting a number for it."
)

_NO_DEPLOYMENT_LICENSE = (
    "> **A SURVIVE here does NOT license deployment.** This is a screen, not a "
    "certificate. A positive headroom triggers a SEPARATE certified decision "
    "replay -- paired against the frozen certified policy at rho=0.94, "
    "permutation per kv cell, full H1 discipline -- exactly as A2 required its "
    "robust-clock re-confirmation. The numbers in this artifact must NEVER be "
    "quoted as a certified gain."
)


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Footprint-aware pricing headroom screen")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY, OFFLINE. Per-call swap price (resident KV footprint) "
        f"replayed over the frozen manifest ({provenance['collection_id']}); no "
        f"GPU. Generated {provenance['generated']} (git {provenance['git_sha']})."
    )
    lines.append("")
    lines.append(f"**Verdict: {results['verdict']}**")
    lines.append("")
    lines.append(_ARM_STATUS_CAVEAT)
    lines.append("")
    lines.append(_AXIS_NEGATIVE)
    lines.append("")
    lines.append(_NO_DEPLOYMENT_LICENSE)
    lines.append("")
    lines.append(f"Kill criterion (frozen before code): {results['kill_criterion']}.")
    lines.append("")
    lines.append(
        f"Headline kv{results['headline_cost_ms']:.0f}: headroom "
        f"**{results['headline_headroom_seconds_per_277']:.2f} s/277** vs banked "
        f"{results['banked_seconds_per_277']:.0f} s/277 "
        f"(exceeds={results['headline_exceeds_banked']}, "
        f"CI excludes zero={results['headline_ci_excludes_zero']})."
    )
    lines.append("")
    secondary = results.get("secondary_readout")
    if secondary:
        lines.append(
            f"Secondary (NON-BINDING, not part of the verdict) kv"
            f"{secondary['kv_cost_ms']:.0f}: headroom "
            f"{secondary['headroom_seconds_per_277']:.2f} s/277 vs the kv5000 "
            f"banked figure {secondary['banked_seconds_per_277']:.1f} s/277 "
            f"(exceeds={secondary['exceeds_banked']}, perm="
            f"{secondary['permutation_label']}). The verdict compares "
            "3500-to-3500; this row is here so the kv5000 question is answered "
            "in the artifact rather than inferred."
        )
        lines.append("")
    power = results["power_rule"]
    lines.append(
        f"Power rule: simultaneous CI [{power['ci_low_seconds_per_277']:.2f}, "
        f"{power['ci_high_seconds_per_277']:.2f}] s/277 against the bar "
        f"{power['bar_seconds_per_277']:.0f} s/277 "
        f"(upper below bar={power['ci_upper_below_bar']}, "
        f"lower above bar={power['ci_lower_above_bar']})."
    )
    lines.append("")
    if results["verdict"] == "DROP":
        lines.append(
            "Footprint-aware pricing DROPS and the direction is CLOSED to one "
            "future-work sentence. The CI upper bound sits BELOW the banked bar, "
            "so the screen genuinely demonstrated the headroom is smaller than "
            "what is already held. Read with the axis limitation above -- this is "
            "a null for self-footprint pricing, NOT for contention-aware policies."
        )
    elif results["verdict"] == "PROCEED":
        lines.append(
            "Footprint-aware pricing PROCEEDS -- to the certified decision replay "
            "described above, NOT to deployment."
        )
    else:
        lines.append(
            "**UNDERPOWERED -- the direction is NOT closed.** The point estimate "
            "is below the banked bar but the CI spans it, so this corpus cannot "
            "resolve an effect of the size we care about. Under the "
            "pre-registered non-inferiority rule a direction may only be closed "
            "when the screen could actually have detected the effect it is "
            "compared against, and this run could not."
        )
        lines.append("")
        lines.append(
            "> **This result MUST NOT be cited as evidence of absent headroom, "
            "and MUST NOT be recorded as a C3 closed axis.** It is future work, "
            "contingent on a corpus with more heavy-call mass."
        )
    lines.append("")
    coherence = results["panel_coherence"]
    if coherence["incoherent"]:
        lines.append(
            f"> Panel-coherence diagnostic (DESCRIPTIVE ONLY, non-binding): the "
            f"headroom changes sign {coherence['sign_flips_across_panel']} times "
            f"across {coherence['cell_count']} kv cells "
            f"({coherence['negative_cells']} negative). A coherent panel moves "
            "smoothly with kv, so this supports the UNDERPOWERED reading -- the "
            "fixed-lambda selection is noise-dominated. The CI rule above is what "
            "binds; this check cannot change the verdict either way."
        )
        lines.append("")
    growth = results.get("footprint_growth")
    if growth:
        lines.append("## Mechanism: how much does the KV footprint actually vary?")
        lines.append("")
        lines.append(
            "This is why the effect could be large at all -- the certified policy "
            "charges ONE fixed `kv_cost_ms` across this entire spread."
        )
        lines.append("")
        lines.append(
            f"- Corpus footprint range: {growth['corpus_min_tokens']:.0f} .. "
            f"{growth['corpus_max_tokens']:.0f} tokens "
            f"({growth['corpus_spread_ratio']:.1f}x spread)."
        )
        if growth["within_task_growth_median"] is not None:
            lines.append(
                "- Within-task growth (max/min footprint), over "
                f"{growth['within_task_growth_tasks']} tasks: median "
                f"{growth['within_task_growth_median']:.1f}x, p90 "
                f"{growth['within_task_growth_p90']:.1f}x, max "
                f"{growth['within_task_growth_max']:.1f}x."
            )
        lines.append("")
    lines.append(
        f"{results['task_count']} tasks, {results['call_count']} calls, rho="
        f"{provenance['restore_cost_fraction']}, guard "
        f"{provenance['guard_ms']:.0f}ms. Permutation: sign-flip, "
        f"{results['permutation_config']['draws']} draws, Bonferroni over "
        f"{results['permutation_config']['simultaneous_family_size']} costs."
    )
    lines.append("")
    lines.append(
        "## Headroom = footprint-priced - fixed-price status quo (seconds per 277 "
        "tasks)"
    )
    lines.append("")
    lines.append(
        "| kv | headroom s/277 | footprint-priced s | fixed-price s | fixed lambda ms | "
        "footprint early fire | fixed early fire | footprint any fire | perm label | "
        "simul CI s |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["cells"]:
        marker = " (H)" if cell["headline"] else ""
        simul = cell["simultaneous_interval_ms"]
        lines.append(
            f"| {cell['kv_cost_ms']:.0f}{marker} | "
            f"{cell['headroom_seconds_per_277']:.2f} | "
            f"{cell['footprint_priced_seconds_per_277']:.2f} | "
            f"{cell['best_fixed_seconds_per_277']:.2f} | "
            f"{cell['mean_fixed_lambda_ms']:.0f} | "
            f"{cell['footprint_priced_early_fire_fraction']:.4f} | "
            f"{cell['fixed_early_fire_fraction']:.4f} | "
            f"{cell['footprint_priced_fire_fraction']:.4f} | "
            f"{cell['permutation_label']} | "
            f"[{simul['low'] / 1000.0:.2f}, {simul['high'] / 1000.0:.2f}] |"
        )
    lines.append("")
    lines.append(
        "(H) = headline cell carrying the frozen kill readout. `fixed lambda ms` "
        "is the fit-fold-selected constant price, averaged over folds. EARLY "
        "fire = a swap strictly before the call's own deadline, i.e. the "
        "decisions the policy adds over deadline_only -- that is the rate the "
        "headroom scales against. `any fire` additionally counts deadline fires."
    )
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
    parser.add_argument(
        "--banked-seconds-per-277",
        type=float,
        default=_BANKED_PRERESTORE_SECONDS_PER_277,
        help="Already-banked pre-restore effect the direction must beat "
        "(frozen: 156 s/277 at kv3500).",
    )
    parser.add_argument(
        "--secondary-banked-seconds-per-277",
        type=float,
        default=_SECONDARY_BANKED_SECONDS_PER_277,
        help="Banked figure for the NON-BINDING kv5000 secondary row "
        "(317.9 s/277). Never feeds the verdict.",
    )
    parser.add_argument(
        "--fixed-lambda-grid",
        type=int,
        default=21,
        help="Fit-fold quantile resolution for the constant-lambda candidate set "
        "(the panel cell is always included).",
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
    stem = f"analysis/pressure-headroom-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def _write_decisions_zst(path: Path, decisions: list[dict[str, Any]]) -> None:
    """Write per-call decisions to a zstd sidecar (kept local, not committed)."""

    import subprocess

    data = json.dumps(decisions, default=list).encode("utf-8")
    # ponytail: shell out to the zstd CLI (no `zstandard` dep installed); "-"
    # reads the JSON from stdin.
    subprocess.run(["zstd", "-q", "-f", "-o", str(path), "-"], input=data, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    samples_by_task, task_ids, manifest = _load_manifest_corpus(
        args.manifest, limit_tasks=args.limit_tasks, final=args.final
    )
    footprints = load_kv_footprint_tokens(
        discover_trace_files([Path(manifest["trace_root"])])
    )
    cfg = PressureHeadroomConfig(
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
        fixed_lambda_grid=args.fixed_lambda_grid,
        banked_seconds_per_277=args.banked_seconds_per_277,
        secondary_banked_seconds_per_277=args.secondary_banked_seconds_per_277,
    )
    summary, decisions = run_pressure_headroom(
        samples_by_task, task_ids, footprints, cfg
    )
    provenance = {
        "exploratory": True,
        "offline_accounting": True,
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
    decisions_path = out_json.with_name(out_json.stem + "-decisions.json.zst")
    payload = {
        "provenance": provenance,
        "config": cfg.__dict__,
        "decisions_sidecar": decisions_path.name,
        **summary,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(summary, provenance), encoding="utf-8")
    _write_decisions_zst(decisions_path, decisions)

    print(f"verdict={summary['verdict']}")
    for cell in summary["cells"]:
        if cell["headline"]:
            print(
                f"  kv{cell['kv_cost_ms']:.0f}: headroom "
                f"{cell['headroom_seconds_per_277']:.2f}s/277 "
                f"perm={cell['permutation_label']} "
                f"fixed_lambda={cell['mean_fixed_lambda_ms']:.0f}ms "
                f"footprint_early_fire={cell['footprint_priced_early_fire_fraction']:.4f}"
            )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"wrote {decisions_path}")


if __name__ == "__main__":
    main()
