# Reload-vs-recompute restore choice (P2)

Source: `/home/chiyu/workspace/agent-sched-bench/analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results`

Gated triggers stay fitted at restore cost zero (Mode A); the restore
charged on fires on short calls is min(swap-in, recompute), with
recompute = rate * per-call context length recovered from traces.
Labels use the Bonferroni-corrected simultaneous intervals over all kv
costs within one comparison-fraction cell.

Context length (tokens): min 1908, median 16030, max 70220.

## recompute rate 0.05 ms/token

### min_restore_vs_swap_restore

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 4 | 6 | 0 | 101779.9 | 0.0 |

### min_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | 68904.8 | -69783.8 |

### swap_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -32875.1 | -99432.5 |

## recompute rate 0.15 ms/token

### min_restore_vs_swap_restore

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | 26877.1 | 0.0 |

### min_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -5998.1 | -94423.3 |

### swap_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -32875.1 | -99432.5 |

## recompute rate 0.5 ms/token

### min_restore_vs_swap_restore

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | 1759.0 | 0.0 |

### min_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -31116.1 | -99432.5 |

### swap_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -32875.1 | -99432.5 |

## recompute rate 1.5 ms/token

### min_restore_vs_swap_restore

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | 0.0 | 0.0 |

### min_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -32875.1 | -99432.5 |

### swap_restore_gated_vs_deadline

| restore fraction | positive | inconclusive | harmful | total delta (ms) | worst simultaneous LCB (ms) |
|---|---|---|---|---|---|
| 0.94 | 0 | 10 | 0 | -32875.1 | -99432.5 |
