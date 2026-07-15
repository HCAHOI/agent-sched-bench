# Gate-union combiner (OR of trie and hazard gates)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-hazard-model-gbm-full-20260715`

gate_union_trigger_ms = min of the two gated triggers as
fitted at each fraction; the combination rule is the only
new element (parameter-free derived re-scoring). Sensitivity
only: the OR rule was selected after observing gate
disjointness on this same frozen corpus, so these intervals
carry selection optimism; fresh-corpus certification is
required before any headline claim.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## union_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 2 | 8 | 0 | 348489.4 | -72586.5 |
| 0.25 | 2 | 8 | 0 | 232154.6 | -58363.3 |
| 0.5 | 2 | 8 | 0 | 162565.1 | -67656.7 |
| 1.0 | 1 | 9 | 0 | 135967.1 | -21376.2 |

## union_vs_gated_hazard

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 26385.3 | -29403.8 |
| 0.25 | 1 | 9 | 0 | 51812.6 | -15518.3 |
| 0.5 | 2 | 8 | 0 | 106141.7 | -20376.2 |
| 1.0 | 2 | 8 | 0 | 112362.7 | -21376.2 |

## union_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 196654.5 | -63682.4 |
| 0.25 | 1 | 9 | 0 | 112661.3 | -60605.5 |
| 0.5 | 1 | 9 | 0 | 11896.7 | -75673.7 |
| 1.0 | 1 | 9 | 0 | 17929.3 | -11165.6 |
