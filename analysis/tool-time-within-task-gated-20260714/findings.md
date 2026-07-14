# Gated within-task baseline — findings (2026-07-14)

Fair control for B1: the within-task baseline given the same cross-fitted
margin-guard protection the cross-task method has (guard fitted per fold on
profile tasks' within-task margins, applied to disjoint eval tasks; both
policies fitted and scored at each rho). Frozen swe-rebench-100 folds; 50k
task-clustered bootstrap; round-4 review APPROVE. Ungated rows reproduce the
earlier B1 run exactly (same seed and inputs).

## Result: the gate cannot rescue the within-task baseline

gated_within_task_vs_deadline:

| rho | total | certified | per-fold guards |
|---|---|---|---|
| 0.0  | +205.5 s | 2 | 0.026–0.148 |
| 0.25 | +68.5 s | 0 | 0.099–0.195 |
| 0.5  | **−153.2 s** | 0 (2 harmful) | two folds "never early" |
| 1.0  | 0.0 (== deadline) | — | all five folds "never early" |

Gated early fires collapse 2056 → 1269 → 674 → 0 across the sweep.

Three sharpened conclusions:

1. **The rho=0 verdict survives the fair control.** Even gated, the
   task-local baseline (+205.5 s, 2 certified) beats the cross-task gated
   method (+151.8 s, 1 certified) at restore-zero. The method's rho=0
   numbers remain non-evidence.
2. **The gate's power comes from cross-task pooling, not from gating per
   se.** The identical guard machinery that keeps the cross-task method
   positive at rho=1.0 collapses to "never fire" when fed within-task
   margins: tiny per-call histories produce margins that do not rank
   fire quality across tasks (at rho=0.5 the gated variant is *worse*
   than ungated, −153.2 vs −75.8 s — the guard kept confidently-wrong
   fires and blocked profitable ones).
3. **Complementarity, not redundancy.** Within-task signal carries most of
   the value when mistakes are cheap; cross-task pooling is what makes
   certified early action possible when mistakes are expensive. This is
   direct evidence for the hybrid (within-task history as the deepest
   context level of the cross-task hierarchy, certified by cross-task
   pooled margins) rather than for either component alone.

gated_vs_gated_within_task confirms the crossover: −53.6 s at rho=0 (2
cells harmful for the method) → +303.8 s at rho=0.5 (1 certified) →
+118.0 s at rho=1.0 (2 certified; gated-B1 is the deadline there).

## Caveats

Same frozen corpus as everything above (sensitivity, not certification);
per-cell labels noisy; one benchmark. The hybrid remains future work — this
control only establishes that neither component alone dominates across the
rho range.
