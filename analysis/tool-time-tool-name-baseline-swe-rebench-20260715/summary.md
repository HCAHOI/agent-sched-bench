# Within-benchmark frontier (gated trie vs gated hazard)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results/data/all.jsonl`

Both the empirical trie and the learned hazard model are fit
and scored on the same fold splits at each restore fraction;
every contrast is a within-benchmark head-to-head. SWE-ReBench 100-task frozen corpus; dev-exposed during method development (sensitivity, not fresh certification)
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## gated_robust_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 151834.9 | -39436.6 |
| 0.25 | 3 | 7 | 0 | 119493.2 | -16458.6 |
| 0.5 | 2 | 8 | 0 | 150668.4 | -20376.2 |
| 1.0 | 2 | 8 | 0 | 118037.9 | -21376.2 |

## gated_hazard_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 106799.8 | -53005.1 |
| 0.25 | 0 | 10 | 0 | 72117.3 | -57791.5 |
| 0.5 | 0 | 10 | 0 | 19820.6 | -46245.8 |
| 1.0 | 0 | 10 | 0 | -3899.9 | -47740.0 |

## gated_hazard_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | -45035.0 | -118187.1 |
| 0.25 | 0 | 10 | 0 | -47375.9 | -76743.9 |
| 0.5 | 0 | 10 | 0 | -130847.9 | -96335.9 |
| 1.0 | 0 | 10 | 0 | -121937.8 | -82316.8 |

## gated_tool_name_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | -13146.6 | -63574.3 |
| 0.25 | 3 | 7 | 0 | 3559.8 | 0.0 |
| 0.5 | 4 | 6 | 0 | 5106.8 | -24332.2 |
| 1.0 | 4 | 6 | 0 | -40696.6 | -128437.6 |

## gated_robust_vs_gated_tool_name

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 164981.5 | -39436.6 |
| 0.25 | 2 | 8 | 0 | 115933.4 | -16458.6 |
| 0.5 | 1 | 8 | 1 | 145561.6 | -20376.2 |
| 1.0 | 1 | 9 | 0 | 158734.5 | -27792.1 |

## gated_hazard_vs_gated_tool_name

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 119946.4 | -55819.5 |
| 0.25 | 0 | 10 | 0 | 68557.5 | -57791.5 |
| 0.5 | 0 | 9 | 1 | 14713.8 | -46245.8 |
| 1.0 | 0 | 8 | 2 | 36796.7 | -35892.6 |
