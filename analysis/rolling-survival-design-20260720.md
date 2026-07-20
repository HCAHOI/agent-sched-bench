# Design spec — priced residual-time policies over certified latency priors

> **A0 OUTCOME (2026-07-20, full fresh-277, binding):** COLLAPSE
> CONFIRMED. 670 certified prior nodes × 10 kv cells: max |k2−k1|
> value gap 0.0 ms (tol 1e-6), induced-decision paired delta 0.0,
> two-stage reduction validated call-by-call (not assumed), k=2
> strictly dominated everywhere at 1ms/check overhead
> (`adjudication-k2-recheck-2026-07-20.md`). Per the binding rule: D2
> stays dead; the completeness subsection (A1) is live; A2 pre-restore
> accounting is the next experiment.

> **Status: FINAL, post-debate (Fable-5 adversarial debate 2026-07-20,
> verdict RESHAPE; all 10 amendments incorporated — the draft's D2
> replay experiment is CUT as structurally degenerate).** Formerly
> titled "rolling survival curves as the system primitive"; the
> debate showed that framing over-claims (nothing queries a runtime
> curve under the frozen estimator) and that the draft's centerpiece
> experiment would have certified an identity.

**Date:** 2026-07-20 · **Estimator:** frozen H1-certified chain-prefix
config, UNCHANGED. **Data:** existing corpora only.

## The debate's decisive finding (recorded as a claim to adjudicate)

Under the existing evaluation functional the policy space is one
irreversible swap action with "call still alive at elapsed t" as the
only runtime observable; the replay accounting reduces every policy to
a scalar `trigger_ms`. Any elapsed-adapted k-check policy is therefore
a deterministic plan computable at call start — it IS a single
stopping time — and `hazard_recheck_ms` already optimizes that scalar
EXACTLY (piecewise-linear argument, breakpoints at {L, L−kv}). Hence
k>1 re-checks have identical optimal value to k=1, and with per-check
overhead priced (mandatory) are strictly dominated. Candidate C's
measured kill closes the other flank: the only filtration enrichment
we can observe (boundary identity) adds no information beyond elapsed
time. **The re-check dimension is closed on both sides — by lemma
(elapsed) and by measurement (boundary).**

## Work items (amended)

- **A0 — Adjudication check (half-day, first move).** Implement the
  k=2 DP over the exact existing functional and verify on fresh-277
  node/decision data that its induced decisions are identical to the
  certified k=1 (or value-tied churn with zero paired delta). This
  tests the collapse argument against implementation reality (e.g.
  restore charging at the deadline boundary). If it FALSIFIES the
  collapse (a real value gap), the original D2 replay experiment is
  reinstated as spec'd in the draft (git history) and runs under the
  H1 replay discipline. If it confirms, no replay is spent.
- **A1 — Completeness subsection (paper artifact, zero data cost).**
  "Within elapsed-adapted policies, one optimally-placed re-check
  captures everything available (exact piecewise optimality); the one
  observable filtration enrichment adds nothing measurable (C:
  72.9% divergence, log-score gain CI [−0.41, +0.20])." Stated in one
  sentence + the lemma; C's numbers are the empirical flank, not a
  statistics-forward display.
- **A2 — Pre-restore offline accounting (the first real experiment;
  was D3).** Formulated per amendment 5 as an EXACT piecewise
  stopping-time optimizer with the restore-lead cost structure (a
  `hazard_recheck_ms` analog), NOT as runtime curve queries — the same
  collapse applies to any elapsed-only consumer. **Pre-registered kill
  (amendment 4, fixed before any code):** at rho=0.94, net seconds per
  277 tasks (restore lead-time hidden minus wasted-restore charged)
  positive with a task-clustered CI excluding zero in ≥1 kv cell
  (3500/5000 headline family, Bonferroni over the cost panel);
  otherwise pre-restore ships as an honest negative table. Offline
  accounting only; GPU live validation remains W10-11 and is labeled
  as such.
- **Internal refactor (was D1, demoted per amendments 6-7):** the
  sorted-sample residual accessor is ordinary code hygiene with a
  property test (exact agreement with direct ECDF recomputation). No
  primitive claim, no micro-benchmark in the paper plan. A runtime
  query surface is justified only by exogenous-state consumers (live
  scheduler admission, memory-pressure-conditioned pre-restore) —
  W10-11, one future-work sentence.

## Framing rules (amendments 8-10)

Lead with **priced residual-time policies over certified latency
priors**: the certified instantiation is the swap trigger; breadth =
the completeness lemma + the pre-restore table. Do NOT claim an
"online distribution prediction service" or a "living curve."
Continuum contrast, corrected wording: conditional-residual
expected-cost pricing vs frozen point-estimate TTL. Presentation is
seconds-saved / X.Xx throughout.

## Integrity rules

Estimator frozen. A2's optimizer uses fit-fold information only,
existing cross-fit discipline, priced overheads (zero-cost actions
forbidden), documented config, no dataset constants, degenerate paths
inherit existing conservative behavior. A0's outcome is binding either
way: confirmation ⇒ D2 stays dead; falsification ⇒ D2 reinstated
verbatim — no third reading.
