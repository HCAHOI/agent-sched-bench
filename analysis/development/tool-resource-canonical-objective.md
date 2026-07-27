# Tool-Resource Latency Prediction — Canonical Objective and Implementation Lock

**Effective 2026-07-27.** This document is authoritative for current KB and
predictor work. It replaces the earlier CPU/memory objective, the stale
3500/5000 ms binary objective, boolean clause composition, and separate offline
model paths. Historical artifacts remain evidence of what ran; they are not
current implementation or evaluation contracts.

## 1. Current objective

Predict command latency as one probability mass function over fixed ordered
buckets. Current scope is **latency only**. CPU and memory are not current
prediction or acceptance targets.

The fixed boundaries are, in milliseconds:

```text
500, 1000, 2000, 4000, 8000, 16000, 32000, 64000
```

To match the scheduler questions `T > b`, the buckets are:

```text
[0, 500]
(500, 1000]
(1000, 2000]
(2000, 4000]
(4000, 8000]
(8000, 16000]
(16000, 32000]
(32000, 64000]
(64000, +inf)
```

The predictor returns one normalized bucket PMF. For every boundary `b_i`,

```text
P(T > b_i) = sum(probability of buckets strictly above b_i)
predicted(T > b_i) = P(T > b_i) > 0.5
ground_truth(T > b_i) = observed_latency_ms > b_i
```

The sole prediction score is classification accuracy at each fixed boundary:

```text
accuracy_i = correct(T > b_i) / eligible_examples
```

Reports include `eligible_examples`, positive count/rate, and the four confusion
counts so raw accuracy is interpretable. These are reconciliation/diagnostic
counts, not additional acceptance metrics.
Do not add balanced accuracy, precision/recall, Brier/NLL, bucket MAE, q-error,
or legacy 3500/5000 ms metrics unless the human explicitly changes this lock.
There is no hidden aggregate across boundaries.

## 2. One implementation for offline and online

There is one predictor/KB algorithm. Offline replay and online serving differ
only in their event source and metric sink.

The canonical core exposes the equivalent of:

```python
predict(repo, clauses, ts_start, runtime_context) -> bucket PMF + provenance
observe(completed_clause_observations) -> None
```

Offline chronological replay calls the same `predict` and `observe` methods as
`resource-agentd`. It must not reimplement lookup, canonicalization,
arbitration, update, or bucket semantics. An observation becomes visible only
to a query with `observation.ts_end < query.ts_start`; overlapping calls must be
buffered accordingly, and backdated queries must be rejected.
Causal observations become eligible only after successful trace finalization.

A fitted public state is frozen for one evaluation/deployment version. Repo or
workspace-local state updates causally. Raw eligible observations remain in the
existing store for provenance; the serving state may use bucket counts as its
sufficient statistics.

## 3. Canonical current architecture

```text
externally parsed clauses
        ↓
generic canonical clause representation
        ↓
one ClauseResourceKB / predictor core
        ├── frozen cross-repo public bucket counts
        └── causal repo-local bucket counts
        ↓
support-aware distribution arbitration
        ↓
bucket PMF + derived P(T > boundary) + provenance
```

`resource-agentd` owns parsing/canonicalization, prediction, KB persistence,
snapshot pinning, causal update, and telemetry orchestration. `telemetryd` owns
privileged collection and finalized normalized observations. Clients never
implement prediction logic.

Compound-command bucket composition is out of scope. Sequential clauses can
accumulate while pipeline members overlap; until a physical composition
contract is separately approved, compound commands return an explicit
unavailable result rather than ORing, adding, or maximizing bucket IDs.

## 4. Evidence boundary

Canonical telemetry validity remains call-granular. Downstream consumers use
only calls marked eligible for KB ingestion. Withheld or missing observations
are never negative labels or zero-valued targets.

Legacy SWE-100 and fresh-277 traces are development-exposed diagnostic proxies.
They may be used to implement, replay, and compare this mechanism but cannot
support a canonical resource claim. Untouched Terminal-Bench confirmation data
must remain untouched until the implementation and criterion are frozen.

## 5. Implementation sequence

### P0 — one executable core

1. Keep the current `ClauseResourceKB` path as the canonical algorithm; avoid a
   rename-only refactor.
2. Make online `resource-agentd` and the offline latency evaluator call that
   same core and semantics.
3. Remove executable legacy predictor/evaluator/CLI paths and their dedicated
   tests once current callers are audited. Preserve historical result artifacts.
4. Add one golden test that feeds an identical timestamped event stream through
   offline and online adapters and asserts identical PMFs, provenance, causal
   visibility, and restored state.
5. Correct two real runtime problems: public evidence must be genuinely
   cross-repo rather than prefiltered to the active workspace, and the KB must
   be constructed once per run/snapshot rather than rebuilt from all visible
   observations for every query.

Stop and review after P0. Do not begin a long SWE replay in P0.

### P1 — frozen development baseline

Run the canonical core on the development-exposed SWE fit/replay corpora with
these exact boundaries. Report per-boundary accuracy and the required counts.
This is a diagnostic baseline, not confirmation.
P1 scores each mapped non-structural clause against its aligned segment latency;
outer command bucket composition remains out of scope.
Use manifest task order as a serialized virtual deployment: predict every call
in one source trace before successful finalization releases its observations,
preserve repository state across later traces, and use synthetic monotonic
timestamps only to order sessions rather than represent historical concurrency.

### P2 — representation and arbitration

Retain the trie only as a cheap candidate/backoff index. Introduce the minimum
generic canonical signature justified by P1 diagnostics, then replace hard
single-node repo-first selection with support-aware public/repo distribution
shrinkage. Do not add ANN retrieval, a second model, or target-specific
similarity while latency is the only target.

### Deferred

- Bin-specific plugins are added only when frequency × residual loss × semantic
  extractability justifies them. Plugins extract semantics; they never encode
  resource predictions.
- Neural/MLP predictors are not parallel production paths. If later evidence
  justifies one, it replaces the core scoring mechanism and must still use the
  same offline/online API and causal replay.
- Delta compaction, decay, reindexing, and atomic snapshot swaps wait for
  measured storage or latency pressure. Bucket counts do not require KLL or
  t-digest sketches.

## 6. Task contract

Every result-affecting KB/predictor task must preserve:

```text
Target = latency bucket PMF only.
Boundaries_ms = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000].
Threshold truth = latency_ms > boundary; prediction = cumulative PMF > 0.5.
Score = per-boundary classification accuracy; include n and positive count/rate.
Offline and online use one predictor implementation and identical causal updates.
No compound composition, legacy binary objective, CPU/memory target, or TB confirmation access.
```
