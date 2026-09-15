# Priced residual-time policies over certified latency priors

This document records frozen offline evidence. It does not claim live serving
benefit.

## Provenance

The pre-registration for this lane is no longer in the working tree. It was
locked at `8d6608cc:analysis/fresh-corpus-preregistration-20260716.md`
(2026-07-16), states that it was committed before any fresh trace was collected
or inspected, and fixes the corpus (`nebius/SWE-rebench`, split `filtered`), the
selection (one `random.Random(42)` shuffle at `skip=150`), the verified
disjointness from the development set, and the collection config; a logged
amendment (`ed4e9d88`, 2026-07-16) extends the sample from 200 to 300 tasks, and
credit exhaustion truncates that extension to the final `N=277`. Its final
amended text is at
`f5478440^:analysis/certification/fresh-corpus-preregistration-20260716.md`,
removed by `f5478440`. The analysis pipeline behind these results, including
`hazard_recheck_ms`, is likewise out of the tree; it is recoverable at
`90b17cdc` and in its last pre-deletion state at `f5478440^`, both under
`scripts/` and `src/trace_collect/`.

## Closed re-check space

Under the existing functional, swap-out is one irreversible action and the only
elapsed-time observation is whether the call is still alive. Every elapsed-only
multi-check plan is therefore computable at call start and reduces to one
stopping time, which `hazard_recheck_ms` already optimizes.

The independent `k=2` dynamic program matched the `k=1` optimum on all 670
certified prior nodes across 10 KV cells. Maximum value gap was `0.0 ms` at
tolerance `1e-6`; any positive per-check overhead makes `k=2` strictly worse.
See [`adjudication-k2-recheck-2026-07-20.md`](adjudication-k2-recheck-2026-07-20.md).

Boundary identity closes the observable-enrichment flank. It changed 72.9% of
re-check decisions, but paired log-score gain was `-0.0137` nats with a
task-clustered 95% CI of `[-0.41, +0.20]`. The compact negative and recovery
reference are in [`../CLOSED-QUESTIONS.md`](../CLOSED-QUESTIONS.md).

## Offline pre-restore result

Pre-restore begins the KV swap-in before the tool call completes so that the
restore overlaps the call tail. It is optimized over the same conditional
residual-time samples and prices both useful overlap and wasted early restore.

| Trigger source | `kv=3500 ms` | `kv=5000 ms` |
|---|---:|---:|
| Hazard clock, optimistic screen | `+164.8 s/277` | `+306.5 s/277` |
| **Robust clock** | **`+156.2 s/277`**, CI `[60.4, 256.6]` | **`+317.9 s/277`**, CI `[181.0, 460.4]` |

Both trigger sources passed the full Bonferroni panel. The effect is monotone
from `kv=2500 ms`, fires on 1.4% of calls, and has approximately 3:1
hidden-to-wasted restore time. The predeclared survival rule required both
trigger sources to pass; it was met. The near-invariance to trigger source is
the evidence that the gain comes from tail-overlap structure rather than an
optimistic trigger.

Authoritative artifacts:

- [`prerestore-accounting-2026-07-20.md`](prerestore-accounting-2026-07-20.md)
- [`prerestore-accounting-robust-2026-07-20.md`](prerestore-accounting-robust-2026-07-20.md)

## Evidence boundary

Offline accounting charges the modeled restore cost but cannot price live PCIe
contention, queue interference, or memory occupancy from an early restore.
Pre-restore remains an offline-supported candidate until the W5 harness measures
those effects.

The paper may describe swap-out and pre-restore as two priced decisions over one
certified residual-time prior. It may not describe a runtime prediction service:
with a frozen estimator, elapsed-only decisions are precomputed stopping times.
A live query surface is justified only by exogenous scheduler state.

## Integrity constraints

- The estimator and operating point remain frozen for these results.
- Every optimizer uses fit-fold information only under the existing cross-fit
  discipline.
- Restore and check overheads are priced; zero-cost actions are not admissible.
- Thin nodes and ties retain the conservative deadline behavior.
- Any estimator change or live pressure-conditioned policy is a new lane with
  its own predeclared gate.
