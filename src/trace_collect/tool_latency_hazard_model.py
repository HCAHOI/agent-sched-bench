"""Discrete-time hazard predictor behind the utility-clock trigger/gate seam.

The empirical trie produces, per call, a list of latency samples that stand
in for ``P(latency | call context)``; ``hazard_recheck_ms`` and
``mean_clock_region_stats`` then apply the swap utility functional to that
list. This module replaces the sample list with a *learned* discrete survival
curve ``S(t | x)`` on a fixed log-spaced grid plus its interval masses
``p_j``. The utility functional is untouched: it is applied to the interval
representatives weighted by the masses instead of an equal-weight sample list,
so an equal-weight mass vector reproduces the empirical policy exactly (see
``survival_trigger_ms`` and the anchor tests).

The hazard model is a pooled penalized logistic on person-period data
(Singer & Willett / Tutz discrete-time survival): every training call with
latency ``L`` is expanded into one row per grid interval it is at risk in,
with a binary "did the call end in this interval" target. Each interval has
its own unpenalized intercept (the baseline hazard) and the encoder features
share one L2-penalized coefficient vector. ``S(t | x)`` is independent of the
KV cost, threshold, and restore cost, which enter only the downstream utility
functional, so the model is fit once per training set and every cost/rho panel
is a cheap functional over the cached masses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from trace_collect.tool_latency_offline_probe import (
    MeanClockRegionStats,
    balanced_task_folds,
)
from trace_collect.tool_latency_utility_clock import (
    utility_matrix,
    validate_restore_cost,
)

if TYPE_CHECKING:  # runtime import is deferred inside fit_hazard_model
    from trace_collect.tool_latency_survival_features import (
        CausalRowFeatures,
        FittedFeatureEncoder,
        SurvivalFeatureSpec,
    )


# Documented defaults for the inner L2 selection. Geometric grid so the
# regularization strength spans several orders of magnitude with a constant
# ratio; K-fold count matches the profile-side inner-CV default. Both are
# configurable arguments of ``fit_hazard_model`` and are never tuned on eval
# data (selection uses grouped log-loss on the training rows only).
DEFAULT_L2_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_CV_FOLDS: int = 5


@dataclass(frozen=True)
class DiscreteTimeGrid:
    """A fixed latency grid: ascending edges plus a representative per interval.

    ``edges`` has ``J + 1`` entries with ``edges[0] == 0.0`` and
    ``edges[-1] == +inf``; interval ``j`` is the half-open ``[edges[j],
    edges[j + 1])`` and the unbounded top interval ``[edges[-2], +inf)``
    absorbs the tail. ``reps`` holds one representative latency per interval
    (length ``J``), used as the sample value fed to the utility functional.
    """

    edges: np.ndarray
    reps: np.ndarray

    @property
    def num_intervals(self) -> int:
        return len(self.reps)


def build_log_grid(
    train_latencies: Iterable[float],
    *,
    num_intervals: int,
    low_quantile: float = 0.01,
    high_quantile: float = 0.995,
    rep: str = "conditional_mean",
) -> DiscreteTimeGrid:
    """Build a log-spaced discrete-time grid from TRAIN latencies only.

    Interior edges are log-spaced between the ``low_quantile`` and
    ``high_quantile`` train quantiles; ``edges[0]`` is pinned to ``0.0`` and a
    ``+inf`` sentinel closes the unbounded top interval that absorbs the tail.
    ``rep="conditional_mean"`` keeps the representative of each interval
    data-driven: the mean of the train latencies that fall in it.

    Fallbacks for reps:

    * An empty *interior* interval uses the geometric midpoint
      ``sqrt(edges[j] * edges[j + 1])`` of its (finite, positive) edges.
    * The *unbounded top* interval has no finite upper edge, so a
      lower-edge-times-a-factor rule would be an unjustified magic number.
      Its representative is the conditional mean of the train tail; if that
      tail were empty it falls back to the maximum train latency (which always
      lies in the top interval, since ``high_quantile <= 1`` places at least
      the sample maximum there, so this branch is defensive).

    ``num_intervals`` is the swept grid resolution (e.g. {20, 40, 80}); it
    must be at least 2 so there is at least one free-hazard interval below the
    absorbing top. Quantiles/edges are computed on TRAIN latencies only.
    """

    latencies = np.asarray(list(train_latencies), dtype=float)
    if latencies.size == 0:
        raise ValueError("build_log_grid requires at least one train latency")
    if not np.all(np.isfinite(latencies)) or np.any(latencies <= 0.0):
        raise ValueError("train latencies must be finite and strictly positive")
    # num_intervals == 2 would leave a single interior edge (q_low only) and
    # silently ignore high_quantile, contradicting the span contract.
    if num_intervals < 3:
        raise ValueError(f"num_intervals must be >= 3, got {num_intervals}")
    if not (0.0 <= low_quantile < high_quantile <= 1.0):
        raise ValueError(
            "require 0 <= low_quantile < high_quantile <= 1, got "
            f"{low_quantile} and {high_quantile}"
        )
    if rep != "conditional_mean":
        raise ValueError(f"unsupported rep policy {rep!r}")

    q_low = float(np.quantile(latencies, low_quantile))
    q_high = float(np.quantile(latencies, high_quantile))
    if not 0.0 < q_low < q_high:
        raise ValueError(
            "degenerate grid quantiles; train latencies must span a positive "
            f"range (q_low={q_low}, q_high={q_high})"
        )
    interior = np.logspace(math.log10(q_low), math.log10(q_high), num=num_intervals - 1)
    edges = np.concatenate(([0.0], interior, [np.inf]))
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("grid edges are not strictly increasing")

    interval_of = np.searchsorted(edges, latencies, side="right") - 1
    reps = np.empty(num_intervals, dtype=float)
    for j in range(num_intervals):
        members = latencies[interval_of == j]
        if members.size:
            reps[j] = float(members.mean())
        elif j == num_intervals - 1:
            reps[j] = float(latencies.max())
        else:
            reps[j] = math.sqrt(edges[j] * edges[j + 1])
    return DiscreteTimeGrid(edges=edges, reps=reps)


@dataclass(frozen=True)
class FittedHazardModel:
    """A fitted discrete-time hazard model over one grid and feature encoder.

    Two estimator families share the same survival/mass pipeline behind a
    single ``predict_interval_masses`` entry point, so the downstream
    trigger/gate code never learns which family produced the masses:

    * ``"logistic"`` (default): ``coef`` concatenates ``J - 1`` unpenalized
      per-interval intercepts (one per free-hazard interval, i.e. every
      interval below the absorbing top) with the shared L2-penalized feature
      coefficients; ``estimator`` is ``None``.
    * ``"gbm"``: ``estimator`` is a fitted
      ``HistGradientBoostingClassifier`` over the encoder features plus the
      interval index as one ordinal column (the tree interacts context with
      time, replacing the per-interval intercepts). ``coef`` and
      ``l2_penalty`` are ``None``.
    """

    encoder: FittedFeatureEncoder
    grid: DiscreteTimeGrid
    coef: np.ndarray | None
    l2_penalty: float | None
    model_family: str = "logistic"
    estimator: Any | None = None

    def predict_interval_masses(self, feats: CausalRowFeatures) -> np.ndarray:
        """Return the interval mass vector ``p_j`` for one call (sums to 1).

        Per-interval hazards ``h_i`` are formed for the ``J - 1`` free
        intervals (from the logistic linear predictor or the GBM's
        ``predict_proba``); survival ``S_j = prod_{i<=j}(1 - h_i)`` is monotone
        non-increasing by construction. Masses are the survival differences
        ``p_j = S_{j-1} - S_j`` for the free intervals and the absorbing top
        interval receives all remaining survivors ``p_top = S_{J-2}``, so the
        vector sums to 1 exactly.
        """

        x = np.asarray(self.encoder.transform(feats), dtype=float)
        num_free = self.grid.num_intervals - 1
        if self.model_family == "gbm":
            hazards = self._gbm_interval_hazards(x, num_free)
        else:
            intercepts = self.coef[:num_free]
            beta = self.coef[num_free:]
            logits = intercepts + float(x @ beta)
            hazards = expit(logits)
        return _masses_from_hazards(hazards)

    def _gbm_interval_hazards(self, x: np.ndarray, num_free: int) -> np.ndarray:
        """Per-interval GBM hazards for one call's fixed feature vector.

        The context ``x`` is held constant across the ``J - 1`` free intervals
        and the interval index sweeps ``0 .. J - 2`` in the appended ordinal
        column (matching the fit-time layout), so the tree can vary the hazard
        with time exactly as the per-interval intercepts do in the logistic
        family.
        """

        interval_col = np.arange(num_free, dtype=float).reshape(-1, 1)
        design = np.hstack((np.tile(x, (num_free, 1)), interval_col))
        return self.estimator.predict_proba(design)[:, 1]


def _masses_from_hazards(hazards: np.ndarray) -> np.ndarray:
    """Convert ``J - 1`` free-interval hazards into a length-``J`` mass vector.

    ``S_j = prod_{i<=j}(1 - h_i)`` is monotone non-increasing since every
    ``h_i in (0, 1)``; ``p_j = S_{j-1} - S_j`` for the free intervals and the
    absorbing top interval takes all remaining survivors ``S_{J-2}``, so the
    result sums to 1 exactly regardless of estimator family.
    """

    survival = np.cumprod(1.0 - hazards)
    prev_survival = np.concatenate(([1.0], survival[:-1]))
    free_masses = prev_survival - survival
    top_mass = survival[-1]
    return np.concatenate((free_masses, [top_mass]))


def fit_hazard_model(
    train_rows: Iterable[Mapping[str, Any]],
    *,
    spec: SurvivalFeatureSpec,
    grid: DiscreteTimeGrid,
    l2_penalty: float | None = None,
    max_iter: int = 200,
    cv_folds: int = DEFAULT_CV_FOLDS,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
    model_family: str = "logistic",
    seed: int | None = None,
) -> FittedHazardModel:
    """Fit a discrete-time hazard model on TRAIN rows.

    The causal feature encoder and the grid are fit on the training split
    only. Each train call with latency ``L`` is person-period expanded: it
    emits one row per interval it is at risk in, with target 1 in the interval
    containing ``L`` (see ``_expand_person_periods``) and 0 for earlier
    survived intervals; a call landing in the absorbing top interval emits
    survived-only rows. This person-period design is shared by both families.

    ``model_family="logistic"`` (default) fits a pooled penalized logistic:
    one unpenalized intercept per free interval and one L2-penalized
    coefficient vector shared across intervals, optimized by scipy L-BFGS-B on
    the mean regularized log-loss (scikit-learn's ``LogisticRegression``
    penalizes every coefficient uniformly and cannot exempt the per-interval
    intercepts, so the scipy path is used). ``l2_penalty=None`` selects the
    penalty by ``cv_folds``-fold task-grouped CV log-loss over ``l2_grid`` on
    the TRAINING rows only (folds group by ``task_id`` so calls from one task
    never straddle a CV split); ties prefer the larger penalty.

    ``model_family="gbm"`` fits a ``HistGradientBoostingClassifier`` on the
    same person-period rows, with the interval index appended as one ordinal
    feature column (so the tree interacts context with time, replacing the
    per-interval intercepts). ``l2_penalty`` is logistic-only and must be
    ``None`` for the GBM (raised otherwise); ``seed`` is required to make the
    GBM's internal validation split and any subsampling deterministic. It is
    unused by the logistic path, which stays byte-identical regardless of
    ``seed``. No eval data enters any stage.
    """

    if model_family not in ("logistic", "gbm"):
        raise ValueError(
            f"unknown model_family {model_family!r}; expected 'logistic' or 'gbm'"
        )
    if model_family == "gbm":
        if l2_penalty is not None:
            raise ValueError(
                "l2_penalty is logistic-only; pass l2_penalty=None with "
                "model_family='gbm'"
            )
        if seed is None:
            raise ValueError("model_family='gbm' requires an explicit seed")

    from trace_collect.tool_latency_survival_features import (
        fit_feature_encoder,
        iter_causal_row_features,
    )

    rows = list(train_rows)
    if not rows:
        raise ValueError("fit_hazard_model requires at least one train row")
    task_by_sample: dict[str, str] = {}
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        task_id = str(row["task_id"])
        existing = task_by_sample.get(sample_id)
        if existing is not None and existing != task_id:
            raise ValueError(
                f"train row {index}: sample_id {sample_id!r} maps to conflicting "
                f"task_id values {existing!r} and {task_id!r}"
            )
        task_by_sample[sample_id] = task_id

    encoder = fit_feature_encoder(rows, spec=spec)
    feats = list(iter_causal_row_features(rows, spec=spec))
    if not feats:
        raise ValueError("no causal feature rows were produced from train rows")

    num_intervals = grid.num_intervals
    num_free = num_intervals - 1
    features = np.stack([np.asarray(encoder.transform(feat), dtype=float) for feat in feats])
    latencies = np.array([feat.latency_ms for feat in feats], dtype=float)
    if not np.all(np.isfinite(latencies)) or np.any(latencies < 0.0):
        raise ValueError("train feature rows must carry finite non-negative latencies")
    call_tasks = [task_by_sample[feat.sample_id] for feat in feats]
    event_intervals = np.searchsorted(grid.edges, latencies, side="right") - 1

    call_index, interval_index, targets = _expand_person_periods(
        event_intervals, num_intervals
    )

    if model_family == "gbm":
        estimator = _fit_gbm(features[call_index], interval_index, targets, seed=seed)
        return FittedHazardModel(
            encoder=encoder,
            grid=grid,
            coef=None,
            l2_penalty=None,
            model_family="gbm",
            estimator=estimator,
        )

    design = _design_matrix(interval_index, features[call_index], num_free)
    if l2_penalty is None:
        l2_penalty = _select_l2_by_cv(
            design,
            targets,
            call_index=call_index,
            call_tasks=call_tasks,
            num_free=num_free,
            cv_folds=cv_folds,
            l2_grid=l2_grid,
            max_iter=max_iter,
        )
    if not math.isfinite(l2_penalty) or l2_penalty < 0.0:
        raise ValueError(f"l2_penalty must be finite and non-negative, got {l2_penalty}")
    coef = _fit_penalized_logistic(design, targets, num_free, l2_penalty, max_iter)
    return FittedHazardModel(
        encoder=encoder,
        grid=grid,
        coef=coef,
        l2_penalty=float(l2_penalty),
        model_family="logistic",
    )


def survival_trigger_ms(
    grid: DiscreteTimeGrid,
    masses: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> float:
    """Expected-utility-optimal swap re-check time for one predicted call.

    The expected utility ``u(k) = sum_j p_j * util(rep_j, k)`` is piecewise
    linear in ``k`` with breakpoints only at ``{rep_j, rep_j - kv_cost}``
    intersected with ``(0, threshold)`` plus ``{0, threshold}``, so the argmax
    over that finite candidate set is exact for the discretized distribution
    (the continuous-law error is bounded by grid resolution inside the
    decision band). This mirrors ``hazard_recheck_ms``: it reuses the same
    ``utility_matrix`` functional and the same tie rule (equal expected
    utility resolves to the LATEST ``k``), so an equal-weight mass vector over
    a rep list reproduces ``hazard_recheck_ms`` on that list exactly.
    """

    reps = _validated_reps_and_masses(grid, masses)
    _validate_utility_inputs(threshold_ms, kv_cost_ms, restore_cost_ms)
    candidates = _trigger_candidates(reps, threshold_ms, kv_cost_ms)
    utility = utility_matrix(
        reps,
        candidates,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    expected = masses @ utility
    best_k = threshold_ms
    best_utility = -math.inf
    for index in range(len(candidates)):
        value = float(expected[index])
        # ``>=`` walking ascending candidates keeps the latest tied k, exactly
        # as hazard_recheck_ms resolves ties.
        if value >= best_utility:
            best_utility = value
            best_k = float(candidates[index])
    return best_k


def survival_clock_region_stats(
    grid: DiscreteTimeGrid,
    masses: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> MeanClockRegionStats:
    """Project a predicted mass vector into decision-relevant regions.

    The empirical analog ``mean_clock_region_stats`` sums the candidate-vs-
    deadline utility delta over samples and normalizes by ``n * kv``; here the
    same delta is taken per interval representative and weighted by its mass,
    so the normalizer is ``kv`` (the masses already sum to 1). The per-rep sign
    properties survive the weighting because they hold pointwise for every
    latency at any trigger ``<= threshold``: band calls never lose, short calls
    never gain, and far-tail calls are exactly neutral. ``normalized_margin``
    stays non-negative because ``survival_trigger_ms`` maximizes the same
    expected-utility objective over a candidate set that includes the deadline.
    Survivor statistics are probability-weighted analogs over the intervals
    whose representative survives the trigger.
    """

    reps = _validated_reps_and_masses(grid, masses)
    # Boolean masking below requires an ndarray; accept any validated sequence.
    masses = np.asarray(masses, dtype=float)
    _validate_utility_inputs(threshold_ms, kv_cost_ms, restore_cost_ms)
    trigger_ms = survival_trigger_ms(
        grid,
        masses,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    columns = utility_matrix(
        reps,
        np.array([trigger_ms, threshold_ms], dtype=float),
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    deltas = columns[:, 0] - columns[:, 1]

    band_gain = 0.0
    short_penalty = 0.0
    for rep, mass, delta in zip(reps, masses, deltas, strict=True):
        rep_value = float(rep)
        weighted = float(mass) * float(delta)
        if threshold_ms < rep_value < threshold_ms + kv_cost_ms:
            if delta < -1e-9:
                raise AssertionError("candidate loses utility on a boundary interval")
            band_gain += weighted
        elif rep_value <= threshold_ms:
            if delta > 1e-9:
                raise AssertionError("candidate gains utility on a short interval")
            short_penalty -= weighted
        elif not math.isclose(float(delta), 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError("far-tail candidate delta must be zero")

    normalized_band_gain = band_gain / kv_cost_ms
    normalized_short_penalty = short_penalty / kv_cost_ms
    normalized_margin = normalized_band_gain - normalized_short_penalty
    if normalized_margin < -1e-12:
        raise AssertionError("survival trigger cannot underperform the deadline")
    normalized_margin = max(0.0, normalized_margin)

    survives = reps > trigger_ms
    survivor_mass = float(masses[survives].sum())
    survivor_count = int(np.count_nonzero(survives))
    if survivor_count and survivor_mass > 0.0:
        surviving_reps = reps[survives]
        surviving_mass = masses[survives]
        short_mask = surviving_reps <= threshold_ms
        band_mask = (surviving_reps > threshold_ms) & (
            surviving_reps < threshold_ms + kv_cost_ms
        )
        far_mask = surviving_reps >= threshold_ms + kv_cost_ms
        if int(np.count_nonzero(short_mask | band_mask | far_mask)) != survivor_count:
            raise AssertionError("three latency regions do not partition survivors")
        p_short = float(surviving_mass[short_mask].sum()) / survivor_mass
        p_band = float(surviving_mass[band_mask].sum()) / survivor_mass
        p_far = float(surviving_mass[far_mask].sum()) / survivor_mass
    else:
        survivor_count = 0
        p_short = p_band = p_far = None
    return MeanClockRegionStats(
        trigger_ms=trigger_ms,
        normalized_margin=normalized_margin,
        normalized_band_gain=normalized_band_gain,
        normalized_short_penalty=normalized_short_penalty,
        survivor_count=survivor_count,
        probability_short_given_survival=p_short,
        probability_band_given_survival=p_band,
        probability_far_given_survival=p_far,
    )


def _expand_person_periods(
    event_intervals: np.ndarray,
    num_intervals: int,
    *,
    censored: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Person-period expand event intervals into (call, interval, target) rows.

    The top interval ``num_intervals - 1`` is absorbing (its hazard is 1 by
    construction), so only the ``num_intervals - 1`` free intervals get rows.
    A call whose event interval is ``j`` emits rows for intervals ``0..j`` with
    target 1 at ``j`` and 0 before (``j`` survived-0 rows + 1 event row); a
    call landing in the absorbing top interval survives every free interval and
    emits survived-0 rows only. A censored call likewise emits survived-0 rows
    only. Returns parallel arrays indexing the source call, the interval, and
    the binary target for each person-period row.
    """

    intervals = np.asarray(event_intervals, dtype=int)
    if np.any(intervals < 0) or np.any(intervals >= num_intervals):
        raise ValueError("event intervals must lie within the grid")
    num_free = num_intervals - 1
    if censored is None:
        censored = np.zeros(intervals.shape, dtype=bool)
    call_index: list[int] = []
    interval_index: list[int] = []
    targets: list[int] = []
    for call, event in enumerate(intervals):
        is_censored = bool(censored[call])
        is_absorbing = event >= num_free
        last_at_risk = num_free - 1 if is_absorbing else int(event)
        for interval in range(last_at_risk + 1):
            call_index.append(call)
            interval_index.append(interval)
            targets.append(
                1 if (interval == event and not is_absorbing and not is_censored) else 0
            )
    return (
        np.asarray(call_index, dtype=int),
        np.asarray(interval_index, dtype=int),
        np.asarray(targets, dtype=float),
    )


def _design_matrix(
    interval_index: np.ndarray,
    row_features: np.ndarray,
    num_free: int,
) -> np.ndarray:
    """Assemble one-hot interval intercepts followed by the shared features."""

    num_rows = interval_index.shape[0]
    design = np.zeros((num_rows, num_free + row_features.shape[1]), dtype=float)
    design[np.arange(num_rows), interval_index] = 1.0
    design[:, num_free:] = row_features
    return design


def _fit_penalized_logistic(
    design: np.ndarray,
    targets: np.ndarray,
    num_free: int,
    l2_penalty: float,
    max_iter: int,
) -> np.ndarray:
    """Minimize mean log-loss with L2 on the feature coefficients only."""

    num_rows = design.shape[0]

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coef
        # -[y log p + (1-y) log(1-p)] = softplus(logit) - y * logit; logaddexp
        # keeps it numerically stable for large-magnitude logits.
        log_loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
        beta = coef[num_free:]
        penalty = 0.5 * l2_penalty * float(beta @ beta)
        gradient = design.T @ (expit(logits) - targets) / num_rows
        gradient[num_free:] += l2_penalty * beta
        return log_loss + penalty, gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    return result.x


def _fit_gbm(
    row_features: np.ndarray,
    interval_index: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> Any:
    """Fit a HistGradientBoosting hazard on the person-period rows.

    The design is the encoder features with the interval index appended as one
    ordinal column, so a single tree ensemble models the interval hazard as a
    function of context and time together (replacing the logistic family's
    per-interval intercepts). Hyperparameters are scikit-learn defaults with
    two documented exceptions: ``early_stopping=True`` forces the internal
    validation-based capacity control on regardless of corpus size (the default
    ``'auto'`` only enables it above 10k samples, so per-fold behavior would
    otherwise flip with fold size), and ``random_state=seed`` makes that
    validation split and any subsampling deterministic. ``max_leaf_nodes=31``
    and ``learning_rate=0.1`` are left at their defaults (no grid, no
    eval-tuning); ``max_iter`` capacity is handled by early stopping.

    Disclosure: scikit-learn's internal early-stopping validation split is
    row-level (class-stratified), not task-grouped, so person-period rows of
    one call or task can straddle the internal train/validation boundary.
    The effect is confined to capacity control inside the training partition
    (mildly optimistic stopping loss, possibly a few extra trees); the
    outer/inner train-vs-eval TASK split is fully respected, so no eval
    information enters the fit.
    """

    from sklearn.ensemble import HistGradientBoostingClassifier

    design = np.column_stack((row_features, interval_index.astype(float)))
    estimator = HistGradientBoostingClassifier(
        early_stopping=True,
        random_state=seed,
    )
    estimator.fit(design, targets.astype(int))
    return estimator


def _select_l2_by_cv(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    call_index: np.ndarray,
    call_tasks: Sequence[str],
    num_free: int,
    cv_folds: int,
    l2_grid: Sequence[float],
    max_iter: int,
) -> float:
    """Pick the L2 penalty by task-grouped CV log-loss on the training rows."""

    grid_values = sorted(float(value) for value in l2_grid)
    if not grid_values or any(value < 0.0 for value in grid_values):
        raise ValueError("l2_grid must be non-empty and non-negative")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")
    folds = balanced_task_folds(
        [{"task_id": task} for task in call_tasks], fold_count=cv_folds
    )
    row_task = np.array([call_tasks[call] for call in call_index], dtype=object)

    best_l2: float | None = None
    best_loss = math.inf
    for l2_penalty in grid_values:
        total_loss = 0.0
        total_rows = 0
        for held_out in folds:
            val_mask = np.array([task in held_out for task in row_task])
            train_mask = ~val_mask
            if not val_mask.any() or not train_mask.any():
                continue
            coef = _fit_penalized_logistic(
                design[train_mask], targets[train_mask], num_free, l2_penalty, max_iter
            )
            logits = design[val_mask] @ coef
            fold_loss = float(
                np.sum(np.logaddexp(0.0, logits) - targets[val_mask] * logits)
            )
            total_loss += fold_loss
            total_rows += int(val_mask.sum())
        if total_rows == 0:
            raise AssertionError("CV produced no held-out person-period rows")
        mean_loss = total_loss / total_rows
        if best_l2 is None or mean_loss < best_loss - 1e-12:
            best_l2, best_loss = l2_penalty, mean_loss
        elif abs(mean_loss - best_loss) <= 1e-12 and l2_penalty > best_l2:
            best_l2 = l2_penalty
    assert best_l2 is not None
    return best_l2


def _trigger_candidates(
    reps: np.ndarray,
    threshold_ms: float,
    kv_cost_ms: float,
) -> np.ndarray:
    candidates = {0.0, float(threshold_ms)}
    for rep in reps:
        value = float(rep)
        if 0.0 < value < threshold_ms:
            candidates.add(value)
        edge = value - kv_cost_ms
        if 0.0 < edge < threshold_ms:
            candidates.add(edge)
    return np.asarray(sorted(candidates), dtype=float)


def _validated_reps_and_masses(
    grid: DiscreteTimeGrid,
    masses: np.ndarray,
) -> np.ndarray:
    reps = np.asarray(grid.reps, dtype=float)
    mass_array = np.asarray(masses, dtype=float)
    if mass_array.shape != reps.shape:
        raise ValueError(
            f"masses length {mass_array.shape} does not match grid reps {reps.shape}"
        )
    if not np.all(np.isfinite(mass_array)) or np.any(mass_array < -1e-12):
        raise ValueError("interval masses must be finite and non-negative")
    if not math.isclose(float(mass_array.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"interval masses must sum to 1, got {float(mass_array.sum())}")
    if not np.all(np.isfinite(reps)) or np.any(reps < 0.0):
        raise ValueError("grid reps must be finite and non-negative")
    return reps


def _validate_utility_inputs(
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float,
) -> None:
    if not math.isfinite(threshold_ms) or threshold_ms <= 0.0:
        raise ValueError(f"threshold_ms must be finite and positive, got {threshold_ms}")
    if not math.isfinite(kv_cost_ms) or kv_cost_ms <= 0.0:
        raise ValueError(f"kv_cost_ms must be finite and positive, got {kv_cost_ms}")
    validate_restore_cost(restore_cost_ms)


__all__ = [
    "DEFAULT_CV_FOLDS",
    "DEFAULT_L2_GRID",
    "DiscreteTimeGrid",
    "FittedHazardModel",
    "build_log_grid",
    "fit_hazard_model",
    "survival_clock_region_stats",
    "survival_trigger_ms",
]
