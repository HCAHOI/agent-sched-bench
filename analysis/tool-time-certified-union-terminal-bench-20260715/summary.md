# Certified-union combiner (OR of certified trie/hazard gates)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-frontier-terminal-bench-20260715`

certified_union_trigger_ms ORs only the gates that certifiably
beat the deadline on the FITTING partition: for each outer fold,
a gate is included using ONLY the OTHER folds' rows (cross-fitted
leave-fold-out), so a fold's own rows never certify the rule they
are scored under. The trigger is the min over included gates, or
the deadline if none is included (criterion: loo_point).
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
| 0.0 | 0 | 10 | 0 | -250807.8 | -438462.7 |
| 0.25 | 0 | 10 | 0 | -380863.9 | -565778.5 |
| 0.5 | 0 | 10 | 0 | -11632.5 | -57097.7 |
| 1.0 | 0 | 10 | 0 | 0.0 | 0.0 |

## certified_union_vs_gated_hazard

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -306263.4 | -438384.4 |
| 0.25 | 0 | 10 | 0 | -421625.6 | -564423.2 |
| 0.5 | 0 | 10 | 0 | -15578.6 | -57097.7 |
| 1.0 | 0 | 10 | 0 | 11951.7 | -17122.2 |

## certified_union_vs_gated_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 47382.8 | -47566.7 |
| 0.25 | 0 | 10 | 0 | 34887.0 | -33450.1 |
| 0.5 | 0 | 10 | 0 | -2538.3 | -30862.1 |
| 1.0 | 0 | 10 | 0 | 25537.9 | -20883.2 |

## certified_union_vs_naive_union

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | 6196.0 | -44119.2 |
| 0.25 | 0 | 10 | 0 | -2011.5 | -33450.1 |
| 0.5 | 0 | 10 | 0 | -6484.4 | -30862.1 |
| 1.0 | 0 | 10 | 0 | 37489.6 | -20883.2 |
