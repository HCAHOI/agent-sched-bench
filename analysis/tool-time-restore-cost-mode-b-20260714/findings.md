# Mode B restore-cost refit — findings (2026-07-14)

Refit of the offline-gated trigger policies with the swap-back cost in the
objective, on the frozen swe-rebench-100 folds
(`analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713`).
Every fit stage (inner probe margins, guard selection, robust/mean triggers)
and all scoring ran at the same rho in {0, 0.25, 0.5, 1.0} x kv_cost.
Bootstrap: 50k replicates, task-clustered, Bonferroni over the 10-cost
family per cell. The built-in fraction-zero check passed: the rho=0 refit
reproduced the frozen certified triggers exactly. Code review-gated (round
2, APPROVE) before running.

## Headline: the method adapts

Where Mode A (triggers blind to rho) went net-negative at rho=1.0, the
Mode B refit stays positive against the plain deadline at every fraction:

| rho | gated_vs_deadline total | certified-positive costs | fires on short |
|---|---|---|---|
| 0.0  | +151.8 s | {1500} | 90 |
| 0.25 | +119.5 s | {1000, 1500, 5000} | 18 |
| 0.5  | +150.7 s | {1000, 5000} | 15 |
| 1.0  | +118.0 s | {1000, 4500} | 11 |

Mechanism: the refit clock moves triggers past the short-call mass instead
of abandoning early action — early fires stay substantial (696–1437 across
the cost panel) while fires-on-short collapse 90 → 11 as rho grows. The
worst simultaneous LCB also tightens (−39.4 s at rho=0 → −16.5 s at 0.25):
avoiding short-fires removes variance, which is why rho=0.25 certifies
*more* cells than rho=0.

Secondary observations:

- The ungated robust clock never certifies at rho>0 and its totals decay
  (+174 s → +75 s); the DACD-style gate grows in value with rho
  (gated_vs_robust −22 s at rho=0 → +43 s at rho=1.0). One anomaly: at
  rho=1.0, cost 500 ms labels gated-vs-robust "harmful" — at small kv the
  restore penalty is small in absolute ms and the gate blocks fires that
  would have paid off.
- The certified cost set wobbles across fractions ({1500} vs {1000,5000}
  vs {1000,4500}); per-cell labels are noisy — the stable claim is that
  some cells certify at every rho and totals stay ≈ +120–150 s.

## Interpretation

Combined with Mode A (`analysis/tool-time-restore-cost-sweep-20260714/`):
the restore-zero gains were partly artifact — a policy fitted blind to the
swap-back cost is net-harmful at realistic rho. But the formulation itself
survives the correction: once rho enters the utility, the same
trigger machinery re-optimizes to later, more selective triggers and keeps
a positive, partially certified margin over the conservative deadline.
Restore cost must therefore be a first-class action parameter (like kv
cost), not a post-hoc adjustment.

## Caveats / next

1. Sensitivity analysis, not certification: Mode B refits on the same
   frozen data the original certification consumed. A certified rho>0
   claim needs a fresh trace corpus (protocol already exists).
2. One benchmark (SWE-ReBench qwen3.7-max, 100 tasks); transfer (E2) open.
3. rho should be measured on the target system (swap-in vs swap-out
   bandwidth asymmetry) to pick the real operating row.
4. Baselines from the critique (within-task last-value, tool-class
   whitelist) still pending — they must now be run at matched rho.
