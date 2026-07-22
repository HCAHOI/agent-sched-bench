# A2 pre-restore offline accounting

> **FINAL - complete corpus**
>
> EXPLORATORY, OFFLINE ACCOUNTING. Restore lead-time overlap priced on original-trace latencies via the frozen manifest (swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277); GPU live validation remains required before deployment. Generated 2026-07-20T09:39:34 (git ed5c3d8e107d6a4cba18ed67a79896f7774a47d4).

**Verdict: SURVIVE** (trigger source: `robust`)

> Swap trigger `g` here is the SHIPPED offline-gated robust clock (`offline_gated_robust_trigger_ms`), reproduced via the certified `evaluate_offline_probe_clock` pipeline; deadline fallbacks inherit conservatively (no earlier trigger invented). This is the re-confirmation run: a **KILL here renders any hazard-source SURVIVE optimistic-screen-only**.

> Pre-registered decision rule: pre-restore is ACTIONABLE offline only under SURVIVE from BOTH trigger sources (hazard optimistic screen AND robust shipped-clock re-confirmation). GPU live validation remains required before deployment.

Kill criterion (frozen): net positive with a task-clustered permutation CI excluding zero (Bonferroni over the full cost family) in >=1 headline kv cell (3500, 5000).

Met at headline kv cell(s): 3500, 5000.

277 tasks, 13410 calls, rho=0.94, guard 0ms (threshold==kv). Permutation: sign-flip, 50000 draws, Bonferroni over 10 costs.

## Net seconds per 277 tasks (lead-time hidden - wasted restore)

| kv | net s/277 | mean ms/task | P90 ms/task | fire frac | hidden s | wasted s | perm label | simul CI ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | -19.45 | -70.21 | 146.12 | 0.010 | 16.27 | 35.72 | inconclusive | [-46.42, 0.45] |
| 1000 | -6.43 | -23.22 | 672.37 | 0.012 | 55.61 | 62.04 | inconclusive | [-45.96, 28.87] |
| 1500 | 6.17 | 22.26 | 1003.29 | 0.012 | 83.72 | 77.55 | inconclusive | [-38.82, 51.39] |
| 2000 | 27.96 | 100.92 | 1299.78 | 0.011 | 108.80 | 80.84 | inconclusive | [-23.38, 82.68] |
| 2500 | 78.22 | 282.37 | 1902.16 | 0.013 | 179.27 | 101.05 | positive | [11.78, 149.14] |
| 3000 | 124.73 | 450.27 | 2300.14 | 0.014 | 234.71 | 109.98 | positive | [43.93, 211.28] |
| 3500 (H) | 156.23 | 564.01 | 2614.93 | 0.014 | 281.25 | 125.02 | positive | [60.41, 256.62] |
| 4000 | 196.25 | 708.50 | 2841.53 | 0.014 | 331.61 | 135.36 | positive | [88.60, 308.98] |
| 4500 | 246.84 | 891.11 | 3269.27 | 0.014 | 394.89 | 148.05 | positive | [122.74, 374.20] |
| 5000 (H) | 317.86 | 1147.51 | 3792.58 | 0.014 | 472.96 | 155.10 | positive | [181.02, 460.44] |

(H) = headline cell. CI columns in seconds per 277 tasks.
