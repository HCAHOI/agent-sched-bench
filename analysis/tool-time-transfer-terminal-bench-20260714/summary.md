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
| 0.0 | 0 | 10 | 0 | -13937.0 | -131936.1 |
| 0.25 | 1 | 9 | 0 | -86513.9 | -75316.3 |
| 0.5 | 0 | 10 | 0 | -47250.6 | -62945.9 |
| 1.0 | 0 | 10 | 0 | -78906.7 | -85446.0 |

## robust_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -27474.8 | -131936.1 |
| 0.25 | 1 | 9 | 0 | -94708.3 | -77396.8 |
| 0.5 | 1 | 9 | 0 | -60809.7 | -83750.9 |
| 1.0 | 1 | 9 | 0 | -94239.6 | -90862.7 |

## mean_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 9 | 1 | -48859.3 | -280909.0 |
| 0.25 | 0 | 7 | 3 | -244563.1 | -429451.2 |
| 0.5 | 1 | 7 | 2 | -390600.8 | -793353.9 |
| 1.0 | 1 | 5 | 4 | -540764.6 | -1073530.6 |
