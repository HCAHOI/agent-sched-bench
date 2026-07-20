# SWE-ReBench 100 Frozen Confirmation

## Setup

- Collection: `swe-rebench-qwen3.7-max-seed42-offset50-100-complete-v2`
- Tasks: 100 complete logical tasks, disjoint from the 50-task SWE-ReBench development set
- Tool-latency samples: 4,640
- Evaluation: five outer task folds, four inner task folds
- Costs: 500 to 5,000 ms in 500 ms increments
- Uncertainty: 50,000 PCG64 task-cluster bootstrap replicates, seed 0
- Family control: two-sided 95% Bonferroni percentile intervals over ten costs
- Primary estimand: `offline_gated_robust_clock - robust_clock`; positive values favor gating
- Independent review: APPROVE, with no critical or major findings

## Primary Result

| Cost (ms) | Paired delta (ms) | Pointwise 95% interval (ms) | Simultaneous 95% interval (ms) | Label | Early fires | Early short fires |
|---:|---:|---:|---:|:---|---:|---:|
| 500 | -2,347.6 | [-5,867.5, 1,489.5] | [-7,251.5, 3,208.7] | inconclusive | 616 -> 247 | 20 -> 10 |
| 1,000 | -9,198.5 | [-21,013.0, 2,255.4] | [-26,048.8, 6,857.3] | inconclusive | 475 -> 254 | 43 -> 26 |
| 1,500 | 276.2 | [-4,805.6, 6,384.5] | [-6,420.3, 9,426.3] | inconclusive | 383 -> 200 | 12 -> 7 |
| 2,000 | 851.6 | [-4,021.1, 7,549.3] | [-5,288.3, 11,234.9] | inconclusive | 290 -> 110 | 11 -> 8 |
| 2,500 | 3,029.1 | [-25,671.9, 45,279.2] | [-33,640.9, 70,869.7] | inconclusive | 262 -> 116 | 20 -> 7 |
| 3,000 | 8,273.9 | [-1,513.6, 20,414.4] | [-4,538.5, 26,729.9] | inconclusive | 241 -> 94 | 14 -> 9 |
| 3,500 | -8,801.0 | [-22,137.9, 5,154.3] | [-27,791.7, 11,288.8] | inconclusive | 213 -> 104 | 10 -> 8 |
| 4,000 | 5,745.7 | [-23,036.9, 44,849.2] | [-32,355.4, 64,575.8] | inconclusive | 210 -> 114 | 15 -> 8 |
| 4,500 | 6,223.4 | [-6,495.0, 22,444.2] | [-9,812.1, 31,027.2] | inconclusive | 192 -> 106 | 4 -> 1 |
| 5,000 | -26,473.3 | [-52,255.4, -2,654.5] | [-65,490.3, 8,104.4] | inconclusive | 196 -> 92 | 11 -> 6 |

The gated policy consistently fires earlier less often and reduces early fires on short calls. However, its paired utility effect versus `robust_clock` changes sign across costs and folds. Every simultaneous interval crosses zero, so the frozen confirmation protocol finds neither a reliable benefit nor reliable harm at any cost.

At 5,000 ms the pointwise interval is negative, but the simultaneous interval crosses zero. The preregistered family-wise rule therefore requires the `inconclusive` label; the pointwise result cannot override it.

## Integrity Checks

- The 100-task pre-run and post-run trace SHA-256 inventories are identical.
- All provenance input hashes and result hashes validate.
- Every outer fold has 20 evaluation tasks and 80 profile tasks.
- All 100 tasks appear in exactly one outer evaluation fold.
- The actual trace set has zero task-ID and byte-level overlap with the three development roots.
- Focused policy and bootstrap tests: 49 passed.

Machine-readable results are in `results/paired_task_cluster_uncertainty.json` and `results/cv/pooled_results.json`.
