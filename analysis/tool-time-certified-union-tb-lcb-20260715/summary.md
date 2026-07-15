# Certified-union combiner (OR of certified trie/hazard gates)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-frontier-terminal-bench-20260715`

certified_union_trigger_ms ORs only the gates that certifiably
beat the deadline on the FITTING partition: for each outer fold,
a gate is included using ONLY the OTHER folds' rows (cross-fitted
leave-fold-out), so a fold's own rows never certify the rule they
are scored under. The trigger is the min over included gates, or
the deadline if none is included (criterion: loo_lcb).
Sensitivity only: the certified-union rule family was itself
selected after observing gate behavior on these corpora, so these
intervals carry selection optimism; fresh-corpus certification is
required before any headline claim.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## certified_union_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 0.0 | 0.0 |
| 0.25 | 0 | 10 | 0 | 0.0 | 0.0 |
| 0.5 | 0 | 10 | 0 | 0.0 | 0.0 |
| 1.0 | 0 | 10 | 0 | 0.0 | 0.0 |

## certified_union_vs_gated_hazard

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -55455.6 | -238021.1 |
| 0.25 | 0 | 10 | 0 | -40761.7 | -199678.7 |
| 0.5 | 0 | 10 | 0 | -3946.0 | -27395.4 |
| 1.0 | 0 | 10 | 0 | 11951.7 | -17122.2 |

## certified_union_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 298190.5 | -52693.3 |
| 0.25 | 0 | 10 | 0 | 415750.9 | -41615.5 |
| 0.5 | 0 | 10 | 0 | 9094.2 | -32825.0 |
| 1.0 | 0 | 10 | 0 | 25537.9 | -20883.2 |

## certified_union_vs_naive_union

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 257003.7 | -181459.6 |
| 0.25 | 0 | 10 | 0 | 378852.4 | -110787.3 |
| 0.5 | 0 | 9 | 1 | 5148.2 | -35822.7 |
| 1.0 | 0 | 10 | 0 | 37489.6 | -20883.2 |
