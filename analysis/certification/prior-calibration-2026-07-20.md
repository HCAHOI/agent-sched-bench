# Calibration of the fixed latency priors

> **FINAL - complete corpus**
>
> DESCRIPTIVE lane, NO kill criterion. The estimator is frozen; this lane measures it and cannot change it. Original-trace latencies via the retained Fresh-277 manifest, cross-fitted as A0/A2. Generated 2026-07-20T10:30:20 (git 06f5a5f0dc9f89ae79850bbc3c8e86ca9181952c).

277 tasks, 13410 calls (14 right-censored), guard 0ms (threshold==kv), rho=0.94. Task-clustered percentile CIs at 0.95.

## Pre-declared readout

- **P90 coverage**: empirical exceedance 0.1035 (nominal 0.1000), CI [0.0950, 0.1125] -- nominal lies **INSIDE** the interval.
- **CvM distance to uniform** (marginal PIT): omega2=0.00003, CI [0.00002, 0.00018] over 13238 calls. omega2 is the N-NORMALISED statistic `W2 / N == integral (F_N(u) - u)^2 du` (here W2=0.423); it is quoted as the headline because it is free of sample size and so is comparable across the differently-sized rolling-grid subsets below.
- **CRPS skill vs pooled curve**: 0.1800, CI [0.1521, 0.2113].
- **CRPS skill vs tool-name curve**: 0.1541, CI [0.1257, 0.1862].
- **Decision-band Brier**: per kv cell, table below.

> **Replicate discipline (non-uniform, stated deliberately).** Coverage, CRPS skill and Brier CIs use 50000 bootstrap replicates; the CvM intervals (marginal and rolling, wherever omega2 appears) use 2000. All are task-clustered at confidence 0.95, seed 0. The split is cost, not convenience: the first three are ratios of sums and go through the certified `_resample_task_totals` in one vectorised pass, whereas CvM is not a ratio of sums and each replicate costs O(calls). At the certified discipline (50000 and 2000) the CvM percentile intervals are stable to ~1e-4. `--final` REJECTS any override of these four knobs, so a FINAL artifact always carries the certified values; only a non-final run can lower them. All four are recorded in the JSON provenance (`replicates`, `cvm_replicates`, `seed`, `confidence_level`).

> Any coverage miss or negative skill is a reported LIMITATION. The estimator does not change based on this lane; a finding of miscalibration would motivate a separate, pre-registered estimator lane with its own decision gate.

## Quantile coverage vs nominal (tail is what matters)

| quantile | nominal exceed | empirical exceed | CI | nominal inside | calls | node-unresolvable | indeterminate censored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P50 | 0.5000 | 0.4992 | [0.4858, 0.5124] | yes | 13252 | 0 | 0 |
| P90 | 0.1000 | 0.1035 | [0.0950, 0.1125] | yes | 12835 | 417 | 0 |
| P95 | 0.0500 | 0.0528 | [0.0462, 0.0602] | yes | 12605 | 647 | 0 |
| P99 | 0.0100 | 0.0109 | [0.0085, 0.0135] | yes | 11406 | 1846 | 0 |

> **Direction of the censoring bias (disclosure).** Right-censored rows whose bound falls BELOW the predicted quantile cannot decide the exceedance, and this lane scores them as non-exceedances. That biases empirical exceedance **LOW** -- i.e. toward the FLATTERING direction, since under-coverage (exceeding more often than nominal) is the dangerous mode for a swap policy. The magnitude is bounded by the censored count (14 of 13410 calls) and is 0 on this corpus: every censored bound sits far above every predicted quantile, so each one resolves as a DEFINITE exceedance and the bias does not bind here. It is disclosed because the bound, not the observed value, is what guarantees that.

## PIT calibration

Marginal PIT over 13238 calls, mean 0.4978 (uniform expects 0.5).

| bin | count | fraction | expected |
| --- | --- | --- | --- |
| [0.0, 0.1) | 1439 | 0.1087 | 0.1000 |
| [0.1, 0.2) | 1273 | 0.0962 | 0.1000 |
| [0.2, 0.3) | 1352 | 0.1021 | 0.1000 |
| [0.3, 0.4) | 1284 | 0.0970 | 0.1000 |
| [0.4, 0.5) | 1287 | 0.0972 | 0.1000 |
| [0.5, 0.6) | 1349 | 0.1019 | 0.1000 |
| [0.6, 0.7) | 1287 | 0.0972 | 0.1000 |
| [0.7, 0.8) | 1268 | 0.0958 | 0.1000 |
| [0.8, 0.9) | 1303 | 0.0984 | 0.1000 |
| [0.9, 1.0) | 1396 | 0.1055 | 0.1000 |

### Rolling PIT (the consumed object) on node-support quantiles

The t-grid is each node's OWN support quantile at the listed fraction, so no corpus-specific time constant enters.

| support fraction | median t (ms) | calls | mean PIT | CvM omega2 | CI |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 3.8 | 11740 | 0.5001 | 0.00002 | [0.00001, 0.00017] |
| 0.20 | 4.1 | 10468 | 0.4992 | 0.00003 | [0.00002, 0.00020] |
| 0.30 | 4.4 | 9126 | 0.5011 | 0.00004 | [0.00002, 0.00021] |
| 0.40 | 4.7 | 7801 | 0.5019 | 0.00003 | [0.00002, 0.00024] |
| 0.50 | 5.6 | 6513 | 0.5037 | 0.00005 | [0.00002, 0.00037] |
| 0.60 | 6.3 | 5187 | 0.5045 | 0.00005 | [0.00003, 0.00040] |
| 0.70 | 6.9 | 3886 | 0.5043 | 0.00007 | [0.00004, 0.00048] |
| 0.80 | 7.4 | 2595 | 0.5013 | 0.00015 | [0.00009, 0.00082] |
| 0.90 | 10.7 | 1279 | 0.5104 | 0.00026 | [0.00013, 0.00183] |

## CRPS and skill (sharpness)

Mean CRPS over 13238 uncensored calls: model 1008.1 ms, pooled baseline 1229.4 ms, tool-name baseline 1191.8 ms. Censored rows excluded: 14 calls carrying 3601.1 s of observed latency.

## Decision-band Brier (per kv cell, at the certified trigger)

Trigger is `hazard_recheck_ms` (the same k=1 optimizer A0 adjudicated and A2 uses as its `hazard` source). Region = calls alive at the trigger; outcome = `L > threshold`. A degenerate region (trigger == threshold) makes the outcome constant, hence the base rate and reference columns.

> **Effective sample (quotable limitation).** Every reported Brier statistic is PER kv CELL -- nothing pools across cells -- so the honest effective sample is the per-cell one: between 669 and 2377 calls fall inside the trigger region in any single cell (across 10 cells), against 13238 calls scoring the marginal PIT and 13238 scoring CRPS. A further 73877 call-by-kv pairs had no usable region (fewer than 2 fit-fold samples surviving past the trigger) and were excluded. The cause is structural, not a data defect: most tool calls last milliseconds, so a node carries little support above a 500-5000 ms trigger. Conclusions from this family are correspondingly weaker than from PIT, coverage, or CRPS.

| kv | region calls | median trigger (ms) | base rate | Brier | CI | reference | skill | degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 2377 | 491.6 | 0.9769 | 0.0220 | [0.0157, 0.0288] | 0.0226 | 0.0277 | 243 |
| 1000 | 1779 | 986.1 | 0.9646 | 0.0302 | [0.0218, 0.0395] | 0.0342 | 0.1168 | 216 |
| 1500 | 1432 | 1495.5 | 0.9455 | 0.0494 | [0.0367, 0.0634] | 0.0515 | 0.0405 | 115 |
| 2000 | 1173 | 1969.3 | 0.9616 | 0.0335 | [0.0232, 0.0451] | 0.0369 | 0.0908 | 104 |
| 2500 | 1028 | 2479.1 | 0.9660 | 0.0306 | [0.0176, 0.0468] | 0.0329 | 0.0700 | 5 |
| 3000 | 932 | 2942.5 | 0.9539 | 0.0400 | [0.0274, 0.0540] | 0.0440 | 0.0908 | 5 |
| 3500 | 841 | 3419.6 | 0.9655 | 0.0300 | [0.0189, 0.0428] | 0.0333 | 0.0982 | 4 |
| 4000 | 760 | 3860.8 | 0.9697 | 0.0299 | [0.0166, 0.0455] | 0.0293 | -0.0176 | 3 |
| 4500 | 720 | 4206.0 | 0.9542 | 0.0433 | [0.0267, 0.0630] | 0.0437 | 0.0106 | 3 |
| 5000 | 669 | 4919.3 | 0.9387 | 0.0561 | [0.0394, 0.0755] | 0.0575 | 0.0256 | 3 |

## Excluded (degenerate nodes and censoring), never silently dropped

- `pit_empty_node`: 0
- `pit_singleton_node`: 158
- `pit_censored_excluded`: 14
- `crps_censored_excluded`: 14
- `crps_censored_excluded_mass_ms`: 3601104.79927063
- `rolling_no_support_beyond_t`: 2122
- `rolling_censored_excluded`: 126
- `coverage_unresolvable_by_quantile`: {'p99': 1846, 'p90': 417, 'p95': 647}
- `coverage_indeterminate_censored_by_quantile`: {}
- `brier_empty_region`: 73877
- `brier_indeterminate_censored`: 0
