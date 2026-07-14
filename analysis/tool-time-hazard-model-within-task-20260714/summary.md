# Hazard-model confirmation

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results`

A learned discrete-time hazard model predicts each call's
latency distribution; its trigger and margin guard replace the
empirical trie behind the same utility-clock seam. Every policy
is fit and scored at each restore fraction, and the hazard gate
reuses the cross-fitted margin guard the trie method has.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -41374.0 | -114846.2 |
| 0.25 | 0 | 10 | 0 | -182282.7 | -141984.5 |
| 0.5 | 0 | 10 | 0 | -343452.0 | -187632.8 |
| 1.0 | 0 | 7 | 3 | -668897.5 | -280323.6 |

## gated_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -37954.8 | -70087.4 |
| 0.25 | 0 | 10 | 0 | 0.0 | 0.0 |
| 0.5 | 0 | 10 | 0 | 0.0 | 0.0 |
| 1.0 | 0 | 10 | 0 | 0.0 | 0.0 |

## gated_hazard_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -189789.6 | -134967.0 |
| 0.25 | 0 | 7 | 3 | -119493.2 | -85006.1 |
| 0.5 | 0 | 8 | 2 | -150668.4 | -101854.9 |
| 1.0 | 0 | 8 | 2 | -118037.9 | -83799.4 |

## gated_hazard_vs_gated_within_task

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 9 | 1 | -243405.8 | -195834.3 |
| 0.25 | 0 | 10 | 0 | -68490.5 | -190561.2 |
| 0.5 | 2 | 8 | 0 | 153160.3 | -81447.7 |
| 1.0 | 0 | 10 | 0 | 0.0 | 0.0 |
