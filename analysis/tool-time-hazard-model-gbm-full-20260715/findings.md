# GBM hazard arm — findings (2026-07-15)

The sharpness test: identical features, identical grid/expansion/seam,
identical gate — only the estimator changes from the pooled penalized
logistic to HistGradientBoosting (interval index as one ordinal column,
sklearn defaults, seed 0; review-gated APPROVE incl. an early-stopping
split disclosure). Frozen swe-rebench-100 folds, rho in {0, 0.25, 0.5,
1.0}, 50k task-clustered bootstrap.

## Result: sharpness was the binding constraint — confirmed

gated_hazard(GBM) totals (vs the linear full arm in brackets):

| rho | vs deadline | vs gated trie | vs gated-B1 |
|---|---|---|---|
| 0.0  | **+322.1 s**, 2 certified [was −83.5] | **+170.3 s**, 1 certified [was −235.3] | +116.7 s |
| 0.25 | **+180.3 s**, 3 certified [was 0] | +60.8 s [was −119.5] | +111.9 s |
| 0.5  | +56.4 s, 1 certified [was 0] | −94.2 s | +209.6 s, 1 certified |
| 1.0  | +23.6 s [was 0] | −94.4 s (1 harmful: 4500 ms) | +23.6 s |

Calibration diagnostics show exactly the predicted mechanism: the linear
model's shrunken middle (predicted 0.55 → observed 0.78) is replaced by
confident extremes (632 calls at predicted 0.011 / observed 0.013; 66 at
0.933 / 1.000). The utility functional pays for that sharpness directly.

## The frontier after seven experiments

- **rho ≤ 0.25**: the GBM hazard clock is the best deployed policy tested —
  it beats the plain deadline with certified cells (e.g. {1000, 5000} ms at
  rho=0; {2500, 4500, 5000} ms at rho=0.25), certifiably beats the gated
  trie at 1000 ms/rho=0 (+74.2 s), and is the first policy to beat the
  gated within-task control at every fraction — the nonlinear model
  finally converts the B1 within-task signal (the linear model could not).
- **rho ≥ 0.5**: the trie's LOTO-unanimity robustness still wins
  (GBM −94 s head-to-head; one certified-harmful cell at 4500 ms/rho=1.0).
  Sharp point predictions carry no across-task stability estimate; the
  trie's unanimity check is doing work the GBM's single margin cannot.
- No policy dominates the rho range. The deployment decision reduces to
  the measured swap-in/swap-out cost ratio of the target system — which
  was already the campaign's top unmeasured quantity.

## Next candidates

1. Ensemble/LOTO analog for the GBM (bagged fits over task subsets;
   unanimity across members) — imports the trie's robustness mechanism
   into the sharp estimator; plausibly dominates both at all rho.
2. Trie-native within-task node level — the no-learning route to the same
   hybrid signal.
3. Measured rho on the target host to select the operating regime.

## Caveats

Same frozen corpus as the whole campaign (seventh analysis on it —
sensitivity, not certification; a fresh-corpus certification is mandatory
before any headline claim). One seed (determinism pinned by test, but
GBM variance across seeds unmeasured). Early-stopping's internal
validation split is row-level, not task-grouped (disclosed in the
docstring; capacity-control-only). Decision JSONLs (~420 MB) uncommitted,
reproducible via scripts/run_hazard_model_confirmation.py
--model-family gbm --seed 0.
