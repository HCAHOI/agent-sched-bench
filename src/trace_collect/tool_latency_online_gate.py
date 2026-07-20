"""Anytime-valid online gate over task-clustered paired regret.

The offline permutation gate (``tool_latency_confirmation``) decides ONCE, on a
frozen corpus, whether the candidate policy beats the frozen baseline on the
task-clustered paired utility difference. This module decides the same question
ONLINE: it runs an anytime-valid test of "no effect" against the same per-task
paired differences, so the verdict may be read after every task without a
peeking penalty, and a policy that stops paying can be REVOKED.

Estimand. Exactly the offline one, per kv cell: the per-task paired utility
difference ``treatment - baseline`` (ms), one scalar per logical task, at the
certified operating point. Both gates test the SAME null at the SAME level; only
the stopping rule differs. No estimator is redefined here.

NO INTERVAL IS REPORTED, deliberately. An earlier version inverted the
martingale over a grid of ``mu`` and published the result as a confidence
interval for the MEAN. That was wrong: inverting a SIGN-SYMMETRY test targets
the symmetry CENTRE, which equals the mean only for symmetric laws. On a
right-skewed population with true mean +0.5 the measured coverage was 0.9975 at
277 tasks but 0.3200 at 4000 against a stated 0.9950 -- it converges to the
centre (~0.56), not the mean, and a shifted-symmetric test law cannot detect
this because the two coincide there. The "convex hull is conservative" defence
was also false as implemented, since the hull was taken over grid points
(``hull(S & grid)`` can be a strict SUBSET of ``hull(S)``): on real data
terminal-bench kv=500 emitted [6.82, 581.53], excluding zero, while the same
construction at ``mu = 0`` sat at ``logE+ = -3.92`` against a threshold of 5.99,
nowhere near rejection. An interval is outside the W3-4 deliverable, so it is
simply gone. Do not re-add one without an estimator that targets the mean.

CONSTRUCTION: a test martingale with predictable bets under SIGN SYMMETRY --
the anytime-valid analogue of the offline gate's sign-flip randomization test,
and the e-process the roadmap's online-gate validity paragraph specifies. For a
hypothesised symmetry centre ``mu``, write ``Y_i = X_i - mu``. Under the
null that each ``Y_i`` is symmetric about zero given the past, for ANY
predictable bet ``lam_i`` (a function of ``Y_1..Y_{i-1}`` only)::

    E[ exp(lam_i * Y_i) / cosh(lam_i * Y_i) | F_{i-1} ] = 1

because ``E[exp(lam*eps*|Y|)] = cosh(lam*|Y|)`` for a Rademacher sign ``eps``.
So ``M_t(mu) = prod_{i<=t} exp(lam_i Y_i) / cosh(lam_i Y_i)`` is a NONNEGATIVE
MARTINGALE started at 1, and Ville's inequality gives, EXACTLY and for all
stopping rules at once, ``P(exists t: M_t >= 1/alpha) <= alpha``.

The implemented directional rule starts from the signed plug-in
``mean(Y_<i) / meansq(Y_<i)``, truncates it to one past scale unit, and uses
its absolute magnitude for ``M_plus`` and the negative magnitude for
``M_minus``. Thus each direction has its own predictable bet; the sign of the
historical plug-in cannot silently swap the identities of the two processes.

Why this and not the alternatives (fixed a priori, not chosen after seeing
agreement):

* It tests the SAME null as the offline gate. ``_permutation_simultaneous_labels``
  is exact under sign symmetry of the paired deltas -- a sharper null than
  ``E[delta] = 0`` -- so an online gate built on the identical null makes the
  replay comparison a like-for-like validation rather than two tests of two
  different hypotheses that happen to share a name.
* It is EXACT and distribution-free: no boundedness, no variance proxy, no
  moment assumption. That matters here for a reason this project has already
  paid for once. Per-task summed paired regret is heavy-tailed at n = 50-277 --
  the regime where a construction's nominal level and its actual level part
  company -- and on exactly this quantity the percentile bootstrap measured ~2.7x
  anticonservative, which is why the offline gate was moved onto permutation
  (analysis/tool-time-gate-robustness-swe-rebench-20260716). An asymptotic
  confidence sequence would re-import that risk: its guarantee is a limit
  statement, and nothing in this corpus certifies that the limit has been
  reached at 50 tasks. Ville's inequality needs no such argument.
* Betting / empirical-Bernstein CSs (Waudby-Smith & Ramdas, JRSS-B 2024) need
  bounded observations; the only a priori bound here is
  ``n_calls * kv * (1 + rho)``, orders of magnitude too loose to certify.

References: Ville (1939); Ramdas, Grunwald, Vovk & Shafer, "Game-theoretic
statistics and safe anytime-valid inference" (Statist. Sci. 2023); Waudby-Smith
& Ramdas (JRSS-B 2024) for the Kelly/GROW plug-in bet.

Bets are predictable and fixed by one documented rule -- the approximate-GROW
(Kelly) plug-in ``lam_i = mean(Y_{<i}) / meansq(Y_{<i})``, truncated to one past
scale unit and zero on the first task. Truncation is numerical conditioning
only: validity holds for ANY predictable bet, so no choice here can inflate the
type-I error, and none of it may be tuned to make agreement look good.

Family-wise discipline. The offline gate is Bonferroni-simultaneous over the kv
cost family at one-sided tail ``alpha / (2m)``. This module takes that same
per-side, per-cell tail and thresholds each one-sided martingale at
``1 / tail``, so the two gates carry an identical simultaneous guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

CERTIFIED = "certified"
HARMFUL = "harmful"
CONTINUE = "continue"
BET_RULE = "directional_abs_kelly_plug_in_truncated_one_past_scale_unit"

# Bet truncation, in units of the running root-mean-square of past deltas. Fixed
# a priori; affects tightness only, never validity (any predictable bet is
# valid), so it is not a tunable knob in the statistical sense.
_BET_TRUNCATION = 1.0


def _log_cosh(values: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(cosh(x))`` (naive cosh overflows past ~710)."""

    magnitude = np.abs(values)
    return magnitude + np.log1p(np.exp(-2.0 * magnitude)) - math.log(2.0)


def _kelly_bets(deltas: np.ndarray) -> np.ndarray:
    """Predictable approximate-GROW bets: past mean over past mean-square.

    Entry ``i`` uses ``deltas[:i]`` ONLY, so the bet is fixed before the task's
    own delta resolves -- the predictability that makes the product a martingale
    and lets the gate be read (and acted on) at arbitrary data-dependent times.
    """

    count = deltas.size
    past_counts = np.arange(count, dtype=float)  # 0, 1, ..., count - 1
    past_sum = np.concatenate([[0.0], np.cumsum(deltas)[:-1]])
    past_sumsq = np.concatenate([[0.0], np.cumsum(deltas**2)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(past_counts > 0, past_sum / past_counts, 0.0)
        meansq = np.where(past_counts > 0, past_sumsq / past_counts, 0.0)
        bets = np.where(meansq > 0.0, mean / meansq, 0.0)
        # Truncate to one past scale unit; also kills the first task's bet,
        # where there is no past to learn from.
        limit = np.where(meansq > 0.0, _BET_TRUNCATION / np.sqrt(meansq), 0.0)
    return np.clip(np.nan_to_num(bets), -limit, limit)


def symmetry_log_martingales(
    deltas: Sequence[float] | np.ndarray, *, mu: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Running ``(log M_plus, log M_minus)`` for symmetry centre ``mu``.

    ``M_plus`` bets on values above ``mu`` and ``M_minus`` below it; each is a
    nonnegative martingale under conditional sign symmetry about ``mu``, so each
    may be thresholded at ``1 / one_sided_tail`` by Ville. Element ``i`` uses
    ``i + 1`` deltas, so the caller can read a verdict after every task.

    The signed approximate-GROW plug-in is truncated first. ``M_plus`` then
    uses its absolute magnitude and ``M_minus`` its negative magnitude.
    """

    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("deltas must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("deltas must all be finite")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")

    centered = values - mu
    magnitude = np.abs(_kelly_bets(centered))
    increments_plus = magnitude * centered
    log_plus = np.cumsum(increments_plus - _log_cosh(increments_plus))
    log_minus = np.cumsum(-increments_plus - _log_cosh(increments_plus))
    return log_plus, log_minus


def _label(log_plus: float, log_minus: float, log_threshold: float) -> str:
    if log_plus >= log_threshold:
        return CERTIFIED
    if log_minus >= log_threshold:
        return HARMFUL
    return CONTINUE


@dataclass(frozen=True)
class Lapse:
    """One CERTIFIED -> not-CERTIFIED transition, and how long it took.

    A LAPSE IS NOT A FINDING. Ville's inequality bounds the probability of ever
    CROSSING the threshold; it says nothing about crossing back, so a martingale
    falling below threshold is not evidence for the null and carries no type-I
    guarantee. Under a genuine, sustained positive effect lapses are the norm
    rather than the exception at the tail this project runs at -- see
    ``test_lapse_is_common_under_a_sustained_true_effect``, which pins the
    measured rate. Treat a lapse as an uncalibrated monitoring signal: a prompt
    to keep watching, never as a reason to withdraw a policy.

    The calibrated event is REVOCATION -- a certified policy whose HARMFUL
    martingale later crosses its own threshold -- reported separately by
    ``replay_online_gate``.
    """

    certified_at_task: int
    lapsed_at_task: int
    lapsed_to: str

    @property
    def lag_tasks(self) -> int:
        return self.lapsed_at_task - self.certified_at_task

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "certified_at_task": self.certified_at_task,
            "lapsed_at_task": self.lapsed_at_task,
            "lapsed_to": self.lapsed_to,
            "lag_tasks": self.lag_tasks,
        }


def replay_online_gate(
    deltas: Sequence[float] | np.ndarray,
    *,
    one_sided_tail: float,
) -> dict[str, Any]:
    """Feed per-task paired deltas in order, reading the gate after every task.

    ``one_sided_tail`` is the per-side, per-cell tail probability. Pass the
    offline gate's ``alpha / (2 * family_size)`` to carry an identical
    Bonferroni-simultaneous guarantee over the kv cost family.

    Two distinct post-certification events are reported and must not be
    conflated. ``revoked`` / ``revocation_lag_tasks`` is the CALIBRATED one
    under the conditional sign-symmetry model: the gate certified and the
    harmful martingale later crossed, which Ville bounds at ``one_sided_tail``
    under that null. It is not a guarantee for every distribution with
    nonnegative mean. ``lapse_count`` counts the uncalibrated event of a
    certified run merely ending -- see ``Lapse``. The deliverable's revocation
    column is the former.

    ``revocation_lag_tasks`` is an EXPOSURE DURATION, not a detection latency:
    it counts tasks from the FIRST certification to the harmful crossing, so it
    includes however long the policy was genuinely good before the truth
    degraded. "Revoked after 517 tasks" therefore does NOT mean "harm went
    undetected for 517 tasks". It is anchored on the first certification rather
    than on whichever certified run was in force, because ``Lapse`` forbids
    acting on a lapse -- so there is only ever one deployment event to measure
    from, and anchoring elsewhere would contradict that doctrine (it is also the
    conservative choice, reporting the longer exposure).

    Task indices are 1-based: ``first_certified_at_task = 12`` means the gate
    first certified once the twelfth task's delta had arrived.
    """

    if not math.isfinite(one_sided_tail) or not 0.0 < one_sided_tail < 1.0:
        raise ValueError("one_sided_tail must be finite and in (0, 1)")
    values = np.asarray(deltas, dtype=float)
    log_plus, log_minus = symmetry_log_martingales(values, mu=0.0)
    log_threshold = math.log(1.0 / one_sided_tail)
    labels = [
        _label(float(plus), float(minus), log_threshold)
        for plus, minus in zip(log_plus, log_minus)
    ]

    first_certified: int | None = None
    first_harmful: int | None = None
    lapses: list[Lapse] = []
    open_certified_at: int | None = None
    lifecycle_labels: list[str] = []
    lifecycle_state = CONTINUE
    for index, label in enumerate(labels, start=1):
        if label != CONTINUE:
            lifecycle_state = label
        lifecycle_labels.append(lifecycle_state)
        if label == CERTIFIED:
            if first_certified is None:
                first_certified = index
            if open_certified_at is None:
                open_certified_at = index
            continue
        if label == HARMFUL and first_harmful is None:
            first_harmful = index
        if open_certified_at is not None:
            lapses.append(
                Lapse(
                    certified_at_task=open_certified_at,
                    lapsed_at_task=index,
                    lapsed_to=label,
                )
            )
            open_certified_at = None

    # REVOCATION (the calibrated event): the gate certified, and the HARMFUL
    # martingale later crossed its own threshold. Classifying by the label at
    # the lapse step alone would miss this, because a reversing run typically
    # passes through CONTINUE for many tasks before harmful evidence
    # accumulates -- so the harmful crossing is looked for anywhere after the
    # first certification, not only at the transition.
    # Scans for the first harmful crossing AFTER the certification rather than
    # comparing against the global first_harmful: a harmful -> certified ->
    # harmful run would otherwise report revoked=False despite a real
    # post-certification crossing. (Unreachable under this bet rule -- once the
    # harmful martingale crosses hard the positive one cannot recover at these
    # tails -- but the correct scan costs the same line.)
    revoked_at = next(
        (
            index
            for index, label in enumerate(labels, start=1)
            if label == HARMFUL
            and first_certified is not None
            and index > first_certified
        ),
        None,
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "method": "sign_symmetry_test_martingale",
        "reference": "Ville 1939; Ramdas et al. Statist. Sci. 2023",
        "bet_rule": BET_RULE,
        "one_sided_tail": float(one_sided_tail),
        "e_value_threshold": float(1.0 / one_sided_tail),
        "task_count": int(values.size),
        "instantaneous_final_label": labels[-1],
        "final_lifecycle_label": lifecycle_labels[-1],
        "final_mean_ms": float(np.mean(values)),
        "final_total_ms": float(np.sum(values)),
        "final_log_e_positive": float(log_plus[-1]),
        "final_log_e_harmful": float(log_minus[-1]),
        # Detection lag against the offline gate, which only ever speaks at the
        # end of the corpus: how many tasks the online gate needed to get there.
        "first_certified_at_task": first_certified,
        "first_harmful_at_task": first_harmful,
        "ever_certified": first_certified is not None,
        "ever_harmful": first_harmful is not None,
        # Ville-controlled: certified, then the HARMFUL side crossed.
        "revoked": revoked_at is not None,
        "revoked_at_task": revoked_at,
        "revocation_lag_tasks": (
            None if revoked_at is None else revoked_at - first_certified
        ),
        # Uncalibrated: the certified run merely ended. Common under a true
        # effect; never report as evidence against the policy. See Lapse.
        "lapse_count": len(lapses),
        "lapses": [lapse.to_json_obj() for lapse in lapses],
        "max_lapse_lag_tasks": (
            max(lapse.lag_tasks for lapse in lapses) if lapses else None
        ),
        "labels": labels,
        "lifecycle_labels": lifecycle_labels,
        "log_e_positive": [float(value) for value in log_plus],
        "log_e_harmful": [float(value) for value in log_minus],
    }
    return result


__all__ = [
    "BET_RULE",
    "CERTIFIED",
    "CONTINUE",
    "HARMFUL",
    "Lapse",
    "replay_online_gate",
    "symmetry_log_martingales",
]
