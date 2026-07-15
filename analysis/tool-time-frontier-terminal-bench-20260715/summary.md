# Within-benchmark frontier (gated trie vs gated hazard)

Source: `/home/chiyu/workspace/agent-sched-bench/traces/terminal-bench/zai-org-GLM-5.2/20260709T171830`

Both the empirical trie and the learned hazard model are fit
and scored on the same fold splits at each restore fraction;
every contrast is a within-benchmark head-to-head. Terminal-Bench corpus was development-exposed during earlier trie work and used as an E2 transfer eval target; this within-benchmark frontier replication is a sensitivity analysis, not independent certification.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## gated_robust_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -298190.5 | -430157.6 |
| 0.25 | 0 | 10 | 0 | -415750.9 | -564423.2 |
| 0.5 | 0 | 10 | 0 | -9094.2 | -57097.7 |
| 1.0 | 0 | 10 | 0 | -25537.9 | -79597.7 |

## gated_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 55455.6 | -23706.0 |
| 0.25 | 0 | 10 | 0 | 40761.7 | -19574.8 |
| 0.5 | 0 | 10 | 0 | 3946.0 | -11970.2 |
| 1.0 | 0 | 10 | 0 | -11951.7 | -21970.2 |

## gated_hazard_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 353646.1 | -53434.0 |
| 0.25 | 0 | 10 | 0 | 456512.6 | -43087.3 |
| 0.5 | 0 | 10 | 0 | 13040.2 | -32825.0 |
| 1.0 | 0 | 10 | 0 | 13586.2 | -20883.2 |
