# E1 restore-cost sweep — findings (2026-07-14)

Re-scoring of the frozen swe-rebench-100 confirmation decisions
(`analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713`)
under a swap-back cost of {0, 0.25, 0.5, 1.0} x kv_cost charged to fires on
short calls. Triggers remain exactly as fitted at restore cost zero
(Mode A stress test; refit-with-rho is Mode B, not run yet). Bootstrap:
50k replicates, task-clustered, Bonferroni over the 10-cost family per cell.
Code review-gated before running (see
`analysis/tool-time-formulation-critique-20260714.md`, audit section).

## Headline

The critique's F1 concern is confirmed on this dataset:

- **gated_vs_deadline**: at rho=0 the deployed gated policy beats the plain
  deadline by +151.8 s total, with one certified-positive cost (1500 ms,
  +17.1 s). At rho=0.25 the certification is already gone; at rho=1.0
  (swap-in ~ swap-out, the physically expected regime) the point estimate is
  **net-negative** (-44.7 s).
- **robust_vs_deadline** (ungated early clock): worse — certified only at
  cost 5000 ms under rho=0, net-negative from rho=0.5, -178.7 s at rho=1.0.
- **gated_vs_robust**: the DACD-style gate *gains* value as rho grows
  (-22.4 s at rho=0 → +134.1 s at rho=1.0): gating suppresses exactly the
  early fires that the restore cost punishes. The gate is protective, but at
  rho=1.0 it protects toward "never fire early", i.e. the deadline itself.
- Concentration (E4 preview): at rho=0, the top 3 of 100 tasks carry 48% of
  the gated_vs_deadline gain (matchms-212 +28.6 s, rdflib-2112 +25.3 s,
  nf-core tools-3247 +19.4 s).
- For reference, the frozen certification's own contrast (gated_vs_robust at
  rho=0) was all-inconclusive on this collection.

## Interpretation

Under the rho=0 utility, early-fire gains were partly an artifact of free
unnecessary swaps. Once the swap-back is charged at its physically expected
scale, no early-trigger policy variant tested here beats the conservative
deadline on this dataset, and the certified-positive cells disappear.

## Caveats / next

1. Mode A only: triggers were fitted blind to rho. Refitting with the
   restore cost in the objective (Mode B; `evaluate_utility_clock_policy
   --restore-cost-fraction`, offline-probe plumbing pending) moves triggers
   later (unit-tested behavior) and may recover certified gains at
   moderate rho.
2. One benchmark (SWE-ReBench qwen3.7-max, 100 tasks); transfer protocol
   (E2) untouched.
3. rho itself should be measured, not assumed: actual swap-in cost on the
   target system decides which sweep row is the real operating point.
