# Fine-Grained Utility-Clock Threshold Sweep Results

Date: 2026-07-11

Status: exploratory mechanism analysis on analyst-exposed corpora. This report
does not select or recommend an operating threshold.

## Integrity Checks

- All 138 corpus/threshold points completed and passed the exact headroom,
  band-gain minus short-penalty, and zero far-tail-delta checks.
- At all ten shared 500 ms grid points, every raw decision row exactly matches
  the preceding sweep across all 15 corpus/fold files.
- The initial 52-file result/provenance inventory and both PNG/PDF figure pairs
  verified before report generation.
- After relocation into `analysis/`, the regenerated 54-file final inventory
  also verified, including this report and the independent review record.

## Fine-Grid Structure

Point-estimate robust-capture sign regions on the 100 ms grid:

| Corpus | Negative-capture regions | Lowest point | Pointwise delta intervals wholly below zero |
|---|---|---:|---|
| SWE | 1200-1600; 1800-1900 | -7.07% at 1400 | none |
| Terminal | 500-600; 1600-3300; 3500; 3800-4400; 4700; 4900 | -42.37% at 600 | none |
| SAB | 1500-2100; 2400-2700 | -28.44% at 1700 | 1600, 1700, 1800 |

These are descriptive point-estimate runs, not confidence sets for a change
point. The pointwise task-bootstrap intervals are non-simultaneous and are not
paired tests between neighboring thresholds.

Exact headroom remains material where capture is poor. The observed `rho`
ranges are:

- SWE: 7.49% at 1,800 ms to 38.15% at 600 ms.
- Terminal: 13.38% at 700 ms to 73.70% at 1,400 ms.
- SAB: 15.03% at 500 ms to 50.03% at 1,400 ms.

## Discrete Policy Cliffs

The fine grid shows that the learned clock is not a smooth function of action
cost. Selected post-hoc transition diagnostics are retained here only to
explain the curve, not to choose thresholds.

| Corpus | Transition | Capture | Net delta | Main realized change |
|---|---:|---:|---:|---|
| SWE | 600 -> 700 | 40.57% -> 2.48% | +77.63s -> +2.08s | Band gain falls 88.46s -> 6.00s; `cd /testbed && python3` changes +64.64s -> -1.12s |
| SWE | 1100 -> 1200 | 11.75% -> -5.86% | +13.57s -> -6.06s | Band gain falls 32.42s -> 13.05s while penalty stays near 19s |
| Terminal | 500 -> 600 | -3.32% -> -42.37% | -2.73s -> -21.82s | Early-short count rises 14 -> 47; penalty rises 6.65s -> 25.06s |
| Terminal | 600 -> 700 | -42.37% -> 1.42% | -21.82s -> +0.67s | Early-short count falls 47 -> 2; `cd /app &&` changes -22.22s -> -0.83s |
| Terminal | 3700 -> 3800 | 2.62% -> -22.89% | +10.00s -> -89.90s | `python3 -c` changes +15.40s -> -73.47s as band gain collapses |
| Terminal | 4400 -> 4500 | -19.80% -> 0.36% | -98.44s -> +1.77s | Short penalty falls 124.60s -> 18.83s; `python3 -c` loss disappears |
| SAB | 1200 -> 1300 | 42.49% -> 14.12% | +90.08s -> +34.58s | Band gain falls 166.57s -> 118.51s |
| SAB | 1400 -> 1500 | 9.12% -> -10.76% | +25.49s -> -30.03s | Band gain falls 119.42s -> 78.02s while penalty rises to 108.04s |
| SAB | 4500 -> 4600 | 22.96% -> 8.96% | +81.19s -> +27.33s | Band gain falls 115.09s -> 52.77s |

The largest adjacent capture changes are 43.79 percentage points for Terminal
600-700, 38.09 points for SWE 600-700, and 28.37 points for SAB 1200-1300.

## Interpretation

1. The 500 ms sweep did not merely undersample a smooth trend. The empirical
   utility clock has real discrete regime changes: candidate knots and robust
   unanimity comparisons change with cost, which can switch whole nodes
   between early action and deadline fallback.
2. This makes post-hoc threshold tuning particularly fragile. A threshold that
   looks favorable can have a neighboring 100 ms point with a different early
   action set and substantially different exposure.
3. The Terminal 600 ms event reinforces the earlier task-correlation finding:
   one node can suddenly admit many correlated short calls. Same-task causal
   adaptation remains a better next probe than a special-case cost rule.
4. A future continuous-cost policy should be evaluated for stability under a
   measured action-cost uncertainty interval. Such robustness must use real
   cost variance; arbitrary smoothing or monotonicity constraints would not be
   justified by this analysis.

## Figures and Data

- `figures/threshold_sweep_overview.png` and `.pdf`: all 46 `rho` and robust
  capture points per corpus.
- `figures/robust_capture_decomposition.png` and `.pdf`: all 46 robust band
  gain, short penalty, and net-capture points per corpus.
- `sweep_results.json`: complete 138-point metrics, pointwise task-bootstrap
  intervals, both learned-policy decompositions, and the repeated 2,000/5,000
  ms case tables.

## Limitations

- Fine-grid local extrema and sign runs were inspected after generation and
  cannot be used as preregistered threshold-selection evidence.
- Pointwise intervals do not quantify the significance of adjacent jumps.
- The decomposition localizes realized utility changes but does not separate
  features, estimator, control rule, or domain shift as causal factors.
