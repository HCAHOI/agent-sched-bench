# Task-jackknife GBM ensemble — findings (2026-07-15)

M=5 leave-one-Mth-of-tasks-out members plus the full-data GBM, unanimity
trigger and weakest-member margin mirroring the trie's robust walk
(bit-exact anchor vs robust_utility_trigger_stats), own cross-fitted
guard. Frozen swe-rebench-100 folds, rho in {0, 0.25, 0.5, 1.0}, seed 0,
50k task-clustered bootstrap. Review-gated APPROVE (commit 44b25a2).

## Result: hypothesis refuted at M=5 — unanimity hedges too hard

| rho | vs deadline | vs single gated GBM | vs gated trie |
|---|---|---|---|
| 0.0  | +162.3 s (4 certified) | **−159.8 s** | +10.4 s (2 certified) |
| 0.25 | +71.3 s (2 certified) | −109.1 s (1 harmful) | −48.2 s |
| 0.5  | −9.5 s | −65.9 s | −160.2 s (1 harmful) |
| 1.0  | −13.9 s (1 certified) | −37.5 s | −132.0 s (1 harmful) |

The ensemble loses to the single GBM at every fraction. It does not close
the trie gap at rho ≥ 0.5 either (worse than the single GBM there). The
mechanism transplant was faithful — the failure is granularity: the trie's
LOTO leaves out one task of ~80 (members barely perturbed, unanimity
cheap), while M=5 removes 20% of tasks per member, making members noisy
enough that unanimity rarely survives early candidates. The reviewer's
wording finding ("M-way jackknife, not LOTO; they coincide only at
M = task count") was empirically prescient.

One genuinely interesting nuance: vs the deadline at rho=0 the ensemble
certifies MORE cells than any policy so far (4, vs the single GBM's 2)
on a smaller total (+162 vs +322 s), with tighter worst-case LCBs
(−42.7 vs −63.6 s). Unanimity trades total gain for per-cell certainty —
potentially valuable if certification count, not total, is the objective.

## Why we stop here rather than tune M

Raising M toward the task count (true LOTO ≈ 80 fits/context) or
switching to lighter subsampling are obvious knobs — and turning them on
this corpus would be exactly the forking-paths overfitting the original
critique warned against. This is the eighth analysis on the frozen
swe-rebench-100 collection. The ensemble-granularity question goes on the
fresh-corpus agenda with a pre-registered M grid, or it doesn't get
answered.

## Standing frontier (unchanged)

- rho ≤ 0.25: single gated GBM (+322 s / +180 s vs deadline, certified).
- rho ≥ 0.5: gated trie (LOTO unanimity over memorized nodes).
- The certified-cell-maximizing variant at rho=0 is the M=5 ensemble.
- Measured rho on the target system remains the deciding measurement.

## Caveats

Eighth analysis on the same frozen corpus (sensitivity only); single M,
single seed; decision JSONLs uncommitted, reproducible via
scripts/run_hazard_model_confirmation.py --model-family gbm
--ensemble-members 5 --seed 0.
