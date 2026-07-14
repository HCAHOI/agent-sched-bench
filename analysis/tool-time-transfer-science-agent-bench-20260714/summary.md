# Cross-benchmark transfer (E2)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results`

The offline-probe guard is fitted on the whole frozen
SWE-ReBench profile and applied frozen to a disjoint target
benchmark; both policies in every contrast share one restore
fraction. The target corpora were previously used during method development, so these results measure transfer sensitivity of the frozen fitting rules, not a fresh independent certification.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 12845.4 | -50539.3 |
| 0.25 | 0 | 10 | 0 | 169.1 | -4208.3 |
| 0.5 | 0 | 10 | 0 | -315.7 | -5458.3 |
| 1.0 | 0 | 10 | 0 | -1315.7 | -7958.3 |

## robust_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 11559.6 | -50539.3 |
| 0.25 | 2 | 8 | 0 | -20664.2 | -46459.6 |
| 0.5 | 2 | 8 | 0 | -37943.7 | -56584.6 |
| 1.0 | 2 | 8 | 0 | -51164.1 | -76834.6 |

## mean_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 74657.1 | -122016.3 |
| 0.25 | 0 | 10 | 0 | -63357.2 | -145351.7 |
| 0.5 | 0 | 10 | 0 | -149389.0 | -201062.6 |
| 1.0 | 0 | 9 | 1 | -325156.9 | -318622.7 |
