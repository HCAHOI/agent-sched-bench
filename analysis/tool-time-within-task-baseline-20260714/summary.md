# Within-task history baseline (B1)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results`

The within-task policy uses only the current task's strictly
earlier calls (deepest prefix context, then tool level) with
the same hazard-recheck estimator; both it and the refit
gated policy are fitted and scored at each restore fraction.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## within_task_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 3 | 7 | 0 | 280501.5 | -104764.9 |
| 0.25 | 1 | 9 | 0 | 86171.8 | -136050.7 |
| 0.5 | 1 | 7 | 2 | -75778.3 | -165144.4 |
| 1.0 | 0 | 8 | 2 | -397843.0 | -249228.4 |

## gated_vs_within_task

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 8 | 2 | -128666.7 | -183422.9 |
| 0.25 | 0 | 9 | 1 | 33321.4 | -184976.1 |
| 0.5 | 2 | 8 | 0 | 226446.7 | -157630.6 |
| 1.0 | 3 | 7 | 0 | 515880.9 | -126316.2 |
