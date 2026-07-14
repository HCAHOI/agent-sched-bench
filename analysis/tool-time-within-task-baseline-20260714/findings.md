# Within-task history baseline (B1) — findings (2026-07-14)

The critique's strongest simple competitor: trigger each call from the
current task's own completed earlier calls (deepest command-prefix context,
then tool level), using the same restore-aware hazard-recheck estimator and
utility as the cross-task method. No profile data, no gate, no
certification. Compared against the Mode B refit gated policy with both
policies fitted and scored at the same rho on the frozen swe-rebench-100
folds. Causality is completion-gated: a call enters history only once its
tool_ts_end is at or before the scored call's start (4.2% of trace calls
overlap the previous call, so start-ordering would have leaked). Review
gate round 3: APPROVE.

## Result: the baseline wins the free-swap game and loses the honest one

| rho | within-task vs deadline | gated vs within-task |
|---|---|---|
| 0.0  | **+280.5 s** (3 certified) | **−128.7 s** (2 cells harmful for gated) |
| 0.25 | +86.2 s (1 certified) | +33.3 s |
| 0.5  | −75.8 s (2 harmful) | +226.4 s (2 certified) |
| 1.0  | **−397.8 s** (2 harmful) | **+515.9 s** (3 certified: 1500/2000/2500) |

At rho=0 the trivial task-local baseline (+280.5 s) roughly doubles the
gated policy's margin over the deadline (+151.8 s) — under the free-swap
utility, the entire cross-task profile/gate machinery is worse than
"this command took long last time in this task." The critique's suspicion
was correct for the restore-zero setting.

Under honest restore costs the picture inverts. The baseline's triggers do
adapt (fires-on-short decline 416 → 305 across the sweep), but its tiny
histories (median context is a handful of same-task calls; last-value for
many) cannot separate hazardous from safe early fires: at rho=0.5 it turns
net-negative with two certified-harmful cells, at rho=1.0 it loses ~398 s
while the refit gated policy holds ≈ +118 s and certifies over the baseline
in three cost cells.

## Interpretation

1. **Restore-zero results are not publishable evidence for the cross-task
   method.** A ~10-line task-local rule dominates it there. Any headline
   claim must be made at measured, nonzero rho.
2. **The machinery's real contribution is robustness, not raw signal.**
   The LOTO-unanimity trigger and the certification gate are what keep the
   method positive when unnecessary swaps carry their true cost; the
   ungated baseline demonstrates what happens without them.
3. **Task-local signal is real and currently wasted.** The baseline's
   rho=0 strength shows strong within-task autocorrelation that the
   cross-task prior ignores. The obvious next method step is a hybrid:
   within-task history as the deepest context level of the existing
   hierarchy (backing off to cross-task prefix/tool/global), gated by the
   same robust machinery — potentially capturing B1's signal with the
   method's discipline.

## Caveats

- Same frozen data as the certification and Mode B — sensitivity analysis,
  not a fresh-corpus claim; per-cell labels remain noisy across fractions.
- The baseline is deliberately ungated; a gated variant of B1 (certifying
  its early fires with the same DACD-style machinery) is the fair
  follow-up before claiming the hybrid is necessary.
- One benchmark; transfer (E2) still open.
