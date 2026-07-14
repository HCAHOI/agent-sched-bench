# Learned hazard model vs empirical trie — findings (2026-07-14/15)

Primary run of the hazard-model phase (docs/hazard-model-phase-plan-20260714.md):
a pooled penalized discrete-time logistic (person-period expansion, log grid
of 40 intervals, per-interval intercepts unpenalized, task-grouped CV for
l2, scipy L-BFGS-B) predicts each call's latency distribution; its
utility-optimal trigger and margin guard replace the empirical trie behind
the identical utility-clock seam. Frozen swe-rebench-100 folds; fits
amortized across rho (model is rho-independent, pinned by test); Step-4
review APPROVE before running. Three feature arms:

- `full` (this dir): tool + prefix + within-task history + task aggregates
- `with_within_task` (`analysis/tool-time-hazard-model-within-task-20260714/`)
- `cross_task_only` (`analysis/tool-time-hazard-model-cross-task-20260714/`)

## Result: the trie wins at this data scale — decisively

gated_hazard_vs_gated_robust (head-to-head, deployed policies), total ms:

| rho | full | with_within_task | cross_task_only |
|---|---|---|---|
| 0.0  | −235.3 s | −189.8 s | −134.2 s |
| 0.25 | −119.5 s | −119.5 s | −148.4 s |
| 0.5  | −150.7 s | −150.7 s | −165.4 s |
| 1.0  | −118.0 s | −118.0 s | −118.0 s |

No arm certifies a single positive cell in any contrast at any fraction.
The gated hazard clock collapses to the deadline (delta exactly 0) at all
rho ≥ 0.25 in the full and with_within_task arms — the cross-fitted guard
refuses to certify the model's out-of-fold margins — while the ungated
hazard clock is outright harmful under rho (−669 s, 3 harmful cells at
rho=1.0, full arm). More features make the rho=0 linear model *worse*
(full −83.5 s vs cross_task_only +17.7 s against the deadline): feature
dilution, not signal gain.

## Why: calibrated but not sharp

The calibration diagnostics localize the failure. Marginal calibration is
acceptable (pooled predicted S(t) tracks observed exceedance within a few
points across the grid), but the reliability bins show strong shrinkage
toward the mean: calls with predicted survival 0.55 actually survive 78%
of the time; predicted 0.84 → observed 96%; and mostly-short calls get
over-predicted survival (predicted 0.14–0.25 → observed 6–8%). The trigger
utility pays for conditional sharpness — knowing that *this* node is
almost-surely long or almost-surely short — and an L2-penalized linear
model over one-hot context blocks smooths exactly that away. The trie's
memorized per-node empirical distributions are sharp wherever support
exists, which is what the guard can certify.

Two positive notes:

1. **The gate again worked perfectly.** Across all three arms the gated
   hazard variant has zero certified-harmful cells vs the deadline — the
   guard bounded a genuinely weaker predictor at the deadline instead of
   letting it fire (the ungated arm shows what it prevented). Sixth
   consecutive experiment in which the certification layer is the
   component that behaves correctly.
2. **The harness did its job.** The predictor swapped in behind the seam
   with zero changes to the utility, gate, or bootstrap code, and the
   ablation grid took one flag.

## Interpretation for the research direction

- The critique's Q1 recommendation ("factor the trigger through an
  explicit survival model") is now *empirically qualified*: the
  factorization is architecturally right (amortization across costs/rho
  worked exactly as predicted), but a linear pooled hazard is the wrong
  estimator at ~7k-call scale. The empirical trie IS a survival estimator
  — a maximally sharp, unsmoothed one — and it wins.
- The B1/gated-B1 within-task signal did not transfer into the linear
  model (with_within_task ≈ full ≤ cross_task_only at rho=0). The hybrid
  hypothesis is NOT refuted — it says the signal needs trie-style
  conditioning (a within-task node level), not regression smoothing.
- Next candidates, in order of promise: (a) trie-with-within-task-level
  (extend latency_prior_hierarchy with a task-local node — no learning,
  keeps sharpness); (b) the approved sklearn HistGradientBoosting arm
  (nonlinear, can recover sharpness; one more recorded run on the same
  harness); (c) calibration-preserving recalibration (isotonic on OOF) is
  NOT promising — the failure is sharpness, not calibration.

## Caveats

Same frozen corpus as the whole campaign (sensitivity, not fresh
certification); one model class and one grid family tested; l2-scalar
reuse disclosed in the evaluator docstring (reviewer-assessed acceptable);
per-fold decision JSONLs (~420 MB/arm) uncommitted but reproducible via
scripts/run_hazard_model_confirmation.py.
