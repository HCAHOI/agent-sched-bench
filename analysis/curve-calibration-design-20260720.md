# Design spec — calibration of the certified latency priors

> **Status: FINAL (methodologically standard; no debate lane).** Asked
> by the project lead: "how do we evaluate the precision of the
> survival curve?" Answer: calibration (are the probabilities honest)
> and sharpness (are they informative), measured out-of-sample, plus a
> decision-band metric because this campaign has already measured one
> case (WTN) where global accuracy improved while decision utility
> regressed.

**Date:** 2026-07-20 · **Estimator:** frozen H1-certified config,
UNCHANGED — this lane MEASURES the shipped prior, it does not tune it.
**Data:** fresh-277 original-trace latencies via the frozen manifest,
cross-fitted exactly as A0/A2 (deepest `latency_prior_hierarchy` node,
fit-fold samples only). Zero GPU.

## Metrics (all out-of-sample, task-grouped folds, task-clustered CIs)

1. **PIT calibration.** For each held-out call with duration y and its
   node's ECDF F: u = F(y) (mid-rank for ties). Honest ⇒ u ~ U[0,1].
   Report the PIT histogram + Cramér–von Mises distance to uniform.
   Also the ROLLING form, which is the object actually consumed: at
   elapsed t, PIT of the renormalized residual curve against y − t,
   evaluated on a documented t-grid (quantiles of the node's own
   support, so the grid is not corpus-specific).
2. **Quantile coverage, tail-weighted.** Empirical exceedance of the
   predicted P50/P90/P95/P99 vs nominal, with CIs. Tail coverage is
   the one that matters: decisions live there.
3. **CRPS with skill scores.** Closed-form CRPS for ECDF forecasts,
   reported as skill vs TWO baselines — the pooled unconditional curve
   and the tool-name-level curve. Beating the pool is what proves the
   conditioning earns its keep; a calibrated-but-uninformative
   forecaster returns the marginal and must score ~0 skill.
4. **Decision-band Brier (the bridge metric).** Brier score of the
   exceedance probability P(L > kv | elapsed) exactly where the policy
   reads it (trigger region, per kv cell). This is the metric the WTN
   lesson demands: a curve can improve globally and degrade here.

## Censoring rule (correctness, not decoration)

Protocol-guard timeouts are RIGHT-CENSORED, not completions. They
enter coverage/PIT as "y exceeded the bound" (contributing to the
upper tail only) and are excluded from CRPS with their count and mass
reported. Silently treating them as exact completions would bias
precisely the tail these metrics exist to measure.

## Pre-registered readout (descriptive, not pass/fail on the estimator)

No KILL criterion — the estimator is frozen and certified; this lane
cannot kill it. Instead a pre-declared reporting standard, fixed
before the run: report P90 coverage with CI and state plainly whether
nominal lies inside; report CvM distance and both CRPS skill scores
with CIs; report decision-band Brier per kv cell. Any coverage miss or
negative skill is reported as a limitation in the paper, not tuned
away — the estimator does not change based on this lane's output.
(A finding of miscalibration would motivate a SEPARATE, pre-registered
estimator lane with its own decision gate.)

## Integrity rules

Cross-fit identical to A0/A2 (no new fold logic). No estimator knobs.
No dataset constants; the t-grid is defined from each node's own
support. Reuse the loaders and stats engine verbatim. Degenerate nodes
(single sample, no support beyond t) are excluded with counts
reported, never silently dropped.
