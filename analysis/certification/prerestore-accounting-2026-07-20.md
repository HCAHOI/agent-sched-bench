# A2 pre-restore offline accounting

> **FINAL - complete corpus**
>
> EXPLORATORY, OFFLINE ACCOUNTING. Restore lead-time overlap priced on original-trace latencies via the frozen manifest (swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277); GPU live validation remains required before deployment. Generated 2026-07-20T06:32:24 (git 4b2332b520a3ed91b845df8921ad073a9435e151).

**Verdict: SURVIVE**

> Swap-trigger asymmetry: the swap trigger `g` here is `hazard_recheck_ms`, the earlier-firing optimistic analog of the SHIPPED robust clock (which fires later or falls back to the deadline). An earlier `g` only widens the pre-restore window, so a **KILL is conservative/strong** (pre-restore fails even given its best shot), whereas any **SURVIVE must be re-confirmed under the shipped robust-clock `g` and then validated live on GPU before pre-restore is acted on**.

Kill criterion (frozen): net positive with a task-clustered permutation CI excluding zero (Bonferroni over the full cost family) in >=1 headline kv cell (3500, 5000).

Met at headline kv cell(s): 3500, 5000.

277 tasks, 13410 calls, rho=0.94, guard 0ms (threshold==kv). Permutation: sign-flip, 50000 draws, Bonferroni over 10 costs.

## Net seconds per 277 tasks (lead-time hidden - wasted restore)

| kv | net s/277 | mean ms/task | P90 ms/task | fire frac | hidden s | wasted s | perm label | simul CI ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | -19.76 | -71.32 | 146.12 | 0.010 | 16.43 | 36.19 | inconclusive | [-46.75, 0.36] |
| 1000 | -19.25 | -69.49 | 610.59 | 0.012 | 51.25 | 70.50 | inconclusive | [-72.93, 21.02] |
| 1500 | 7.88 | 28.45 | 1080.59 | 0.014 | 98.12 | 90.24 | inconclusive | [-41.53, 56.76] |
| 2000 | 27.85 | 100.56 | 1389.66 | 0.012 | 114.33 | 86.48 | inconclusive | [-24.93, 84.98] |
| 2500 | 87.17 | 314.70 | 1951.46 | 0.014 | 192.92 | 105.75 | positive | [19.33, 159.38] |
| 3000 | 133.32 | 481.29 | 2395.78 | 0.015 | 248.94 | 115.62 | positive | [50.82, 221.86] |
| 3500 (H) | 164.77 | 594.82 | 2681.64 | 0.015 | 289.79 | 125.02 | positive | [67.62, 266.40] |
| 4000 | 197.07 | 711.44 | 2875.50 | 0.015 | 332.43 | 135.36 | positive | [89.35, 309.83] |
| 4500 | 248.66 | 897.69 | 3322.58 | 0.015 | 396.71 | 148.05 | positive | [124.69, 376.52] |
| 5000 (H) | 306.47 | 1106.38 | 3739.27 | 0.015 | 470.97 | 164.50 | positive | [166.57, 449.92] |

(H) = headline cell. CI columns in seconds per 277 tasks.
