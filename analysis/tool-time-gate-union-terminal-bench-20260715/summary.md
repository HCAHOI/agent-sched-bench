# Gate-union combiner (OR of trie and hazard gates)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-frontier-terminal-bench-20260715`

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
| 0.0 | 0 | 10 | 0 | -257003.7 | -430551.5 |
| 0.25 | 0 | 10 | 0 | -378852.4 | -565778.5 |
| 0.5 | 1 | 9 | 0 | -5148.2 | -57097.7 |
| 1.0 | 0 | 10 | 0 | -37489.6 | -79597.7 |

## union_vs_gated_hazard

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -312459.3 | -430157.6 |
| 0.25 | 0 | 10 | 0 | -419614.1 | -564423.2 |
| 0.5 | 0 | 10 | 0 | -9094.2 | -57097.7 |
| 1.0 | 0 | 10 | 0 | -25537.9 | -79597.7 |

## union_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 41186.8 | -37208.1 |
| 0.25 | 0 | 10 | 0 | 36898.5 | -19574.8 |
| 0.5 | 0 | 10 | 0 | 3946.0 | -11970.2 |
| 1.0 | 0 | 10 | 0 | -11951.7 | -21970.2 |
