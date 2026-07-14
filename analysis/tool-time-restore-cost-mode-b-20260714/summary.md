# Restore-cost Mode B refit

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results`

Triggers, probe margins, and guards are refit at each
restore fraction; the bootstrap scores at the same fraction.
Labels use the Bonferroni-corrected simultaneous intervals over
all kv costs within one comparison-fraction cell; they are not
corrected across comparisons or fractions, so read each row as
its own what-if.

## gated_vs_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -22420.4 | -65490.3 |
| 0.25 | 0 | 10 | 0 | -19609.8 | -89960.9 |
| 0.5 | 0 | 10 | 0 | 15678.8 | -69230.7 |
| 1.0 | 0 | 9 | 1 | 43179.1 | -83744.6 |

## gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 151834.9 | -39436.6 |
| 0.25 | 3 | 7 | 0 | 119493.2 | -16458.6 |
| 0.5 | 2 | 8 | 0 | 150668.4 | -20376.2 |
| 1.0 | 2 | 8 | 0 | 118037.9 | -21376.2 |

## robust_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 174255.2 | -68668.1 |
| 0.25 | 0 | 10 | 0 | 139103.0 | -57469.4 |
| 0.5 | 0 | 10 | 0 | 134989.6 | -68400.6 |
| 1.0 | 0 | 10 | 0 | 74858.8 | -107502.3 |
