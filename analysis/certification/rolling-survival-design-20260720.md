# Priced residual-time policies over certified latency priors

**Date:** 2026-07-20 · Estimator: frozen H1-certified config, unchanged
· Existing data only · Zero GPU for everything below.

## What this lane established

**The re-check dimension is closed on both flanks.** Under the existing
evaluation functional the policy space is one irreversible swap action
with "call still alive at elapsed t" as the only runtime observable,
and the replay accounting reduces every policy to a scalar
`trigger_ms`. So any elapsed-adapted k-check policy is a plan
computable at call start — it IS a single stopping time — and
`hazard_recheck_ms` already optimizes that scalar exactly (piecewise
linear, breakpoints at {L, L−kv}).

Verified rather than argued: an honestly-implemented k=2 dynamic
program attains the k=1 optimum on all 670 certified prior nodes × 10
kv cells, **max value gap 0.0 ms** (tol 1e-6), and is strictly
dominated once per-check overhead is priced. The two-stage reduction
was re-simulated call-by-call rather than assumed.
→ `adjudication-k2-recheck-2026-07-20.md`

The other flank is measured: mid-call boundary identity (which
sub-command finished) moves 72.9% of re-check decisions but its paired
log-score gain is −0.0137 nats, CI [−0.41, +0.20] — churn without
information. → `boundary-evidence-stage1-2026-07-19.md`

**Together these are a completeness result, not an absence:** within
elapsed-adapted policies, one optimally-placed re-check captures
everything available, and the one observable enrichment this setting
offers adds nothing measurable. That is a paper subsection costing a
lemma rather than a corpus.

**Pre-restore works and cleared both of its gates.** Starting the KV
restore before the tool call completes, so restore overlaps the call's
tail, priced as an exact piecewise stopping-time optimizer (a
`hazard_recheck_ms` analog with the restore-lead cost structure):

| trigger source | kv3500 | kv5000 |
|---|---|---|
| hazard (optimistic screen) | +164.8 s/277 | +306.5 s/277 |
| **shipped robust clock** | **+156.2 s/277** (CI [60.4, 256.6]) | **+317.9 s/277** (CI [181.0, 460.4]) |

Both permutation-positive under the full Bonferroni panel; effect
monotone in kv from 2500 up; fires on 1.4% of calls; hidden:wasted ≈
3:1. The pre-registered rule (SURVIVE under both trigger sources) is
met. The near-invariance to trigger source is the load-bearing
robustness fact — the gain comes from tail-overlap structure, not from
an optimistic trigger.
→ `prerestore-accounting-2026-07-20.md`, `-robust-2026-07-20.md`

**Remaining gate:** GPU live validation. Offline accounting cannot
price real transfer contention. Until that runs, pre-restore is
certified offline, not shipped.

## Framing rules for the paper

Lead with **priced residual-time policies over certified latency
priors**. The certified instantiation is the swap trigger; breadth
comes from the completeness result plus the pre-restore table.

Do NOT claim an "online distribution prediction service" or a "living
curve" — under a frozen estimator every elapsed-only consumer is a
precomputed stopping time, and nothing queries a curve at runtime. The
sorted-sample residual accessor is ordinary internal code with a
property test, not a contribution; a runtime query surface is
justified only by exogenous-state consumers (live scheduler admission,
pressure-conditioned pre-restore), which are W10-11 future work.

Continuum contrast, worded accurately: **conditional-residual
expected-cost pricing vs a frozen point-estimate TTL**. Not "we added
load awareness" — Continuum already carries a workload-level
sliding-window load term.

Presentation is seconds-saved / X.Xx throughout.

## Integrity rules

Estimator frozen; any estimator change is a separate pre-registered
lane with its own gate. Optimizers use fit-fold information only under
the existing cross-fit discipline. Overheads are priced — zero-cost
actions are forbidden. Degenerate paths (thin nodes, ties) inherit the
existing conservative behavior (deadline re-check).

## Provenance note

This spec originally proposed a "rolling survival primitive" plus a
k>1 re-check replay experiment. A Fable-5 adversarial debate showed the
k>1 experiment was structurally degenerate before any compute was
spent, and the A0 adjudication then confirmed it on real nodes. Both
were removed rather than annotated; the completeness result above is
what that reasoning produced.
