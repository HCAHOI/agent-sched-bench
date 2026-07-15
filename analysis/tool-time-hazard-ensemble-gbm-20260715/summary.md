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
| 0.0 | 1 | 9 | 0 | 288641.1 | -84084.9 |
| 0.25 | 0 | 10 | 0 | 64142.0 | -106229.3 |
| 0.5 | 0 | 10 | 0 | -167315.6 | -160783.0 |
| 1.0 | 0 | 7 | 3 | -585610.8 | -263395.4 |

## gated_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 2 | 8 | 0 | 322104.1 | -63554.6 |
| 0.25 | 3 | 7 | 0 | 180341.9 | -61308.0 |
| 0.5 | 1 | 9 | 0 | 56423.5 | -78904.7 |
| 1.0 | 0 | 10 | 0 | 23604.4 | -19671.2 |

## gated_hazard_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 170269.2 | -67003.2 |
| 0.25 | 0 | 10 | 0 | 60848.7 | -67998.6 |
| 0.5 | 0 | 10 | 0 | -94244.9 | -93121.1 |
| 1.0 | 0 | 9 | 1 | -94433.5 | -81293.0 |

## gated_hazard_vs_gated_within_task

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 116653.0 | -95642.5 |
| 0.25 | 0 | 10 | 0 | 111851.5 | -89584.9 |
| 0.5 | 1 | 9 | 0 | 209583.7 | -83074.9 |
| 1.0 | 0 | 10 | 0 | 23604.4 | -19671.2 |

## ensemble_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 4 | 6 | 0 | 159901.4 | -42650.2 |
| 0.25 | 2 | 8 | 0 | 67955.4 | -58126.3 |
| 0.5 | 2 | 8 | 0 | 12743.3 | -73572.6 |
| 1.0 | 1 | 9 | 0 | -79172.9 | -104599.6 |

## gated_ensemble_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 4 | 6 | 0 | 162279.6 | -42650.2 |
| 0.25 | 2 | 8 | 0 | 71284.6 | -58126.3 |
| 0.5 | 0 | 10 | 0 | -9513.4 | -34082.6 |
| 1.0 | 1 | 9 | 0 | -13944.8 | -47929.2 |

## gated_ensemble_vs_gated_hazard

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -159824.5 | -132872.9 |
| 0.25 | 0 | 9 | 1 | -109057.3 | -111213.8 |
| 0.5 | 0 | 10 | 0 | -65936.9 | -125789.9 |
| 1.0 | 0 | 10 | 0 | -37549.3 | -71989.0 |

## gated_ensemble_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 2 | 8 | 0 | 10444.7 | -65830.1 |
| 0.25 | 0 | 10 | 0 | -48208.6 | -58972.1 |
| 0.5 | 0 | 9 | 1 | -160181.9 | -91402.5 |
| 1.0 | 0 | 9 | 1 | -131982.7 | -81293.0 |
