# Restore-cost re-scoring sweep

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results`

Triggers are frozen as fitted at restore cost zero; only the
evaluation utility charges fires on short calls. Labels use the
Bonferroni-corrected simultaneous intervals over all kv costs
within one comparison-fraction cell; they are not corrected across
comparisons or fractions, so read each row as its own what-if.

## gated_vs_robust

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 0 | 10 | 0 | -22420.4 | -65490.3 |
| 0.25 | 0 | 10 | 0 | 16704.6 | -62125.1 |
| 0.5 | 0 | 10 | 0 | 55829.6 | -58952.3 |
| 1.0 | 0 | 10 | 0 | 134079.6 | -54839.6 |

## gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 151834.9 | -39436.6 |
| 0.25 | 0 | 10 | 0 | 102709.9 | -55218.2 |
| 0.5 | 0 | 10 | 0 | 53584.9 | -71278.2 |
| 1.0 | 0 | 10 | 0 | -44665.1 | -103412.1 |

## robust_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.0 | 1 | 9 | 0 | 174255.2 | -68668.1 |
| 0.25 | 0 | 10 | 0 | 86005.2 | -99948.9 |
| 0.5 | 0 | 10 | 0 | -2244.8 | -131105.0 |
| 1.0 | 0 | 10 | 0 | -178744.8 | -195958.9 |
