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
| 0.0 | 0 | 10 | 0 | 31646.3 | -91261.9 |
| 0.25 | 0 | 10 | 0 | -163151.1 | -141984.5 |
| 0.5 | 0 | 10 | 0 | -308006.4 | -187632.8 |
| 1.0 | 0 | 7 | 3 | -671453.6 | -280323.6 |

## gated_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 17655.7 | -59695.5 |
| 0.25 | 1 | 9 | 0 | -28871.8 | -80484.2 |
| 0.5 | 0 | 10 | 0 | -14699.1 | -44581.6 |
| 1.0 | 0 | 10 | 0 | 0.0 | 0.0 |

## gated_hazard_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 9 | 1 | -134179.2 | -109303.1 |
| 0.25 | 0 | 10 | 0 | -148365.0 | -86809.1 |
| 0.5 | 0 | 9 | 1 | -165367.5 | -88785.4 |
| 1.0 | 0 | 8 | 2 | -118037.9 | -83799.4 |

## gated_hazard_vs_gated_within_task

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 9 | 1 | -187795.4 | -193816.8 |
| 0.25 | 0 | 9 | 1 | -97362.2 | -167132.3 |
| 0.5 | 2 | 8 | 0 | 138461.1 | -73388.3 |
| 1.0 | 0 | 10 | 0 | 0.0 | 0.0 |
