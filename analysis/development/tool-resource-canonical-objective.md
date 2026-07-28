# Tool-Resource Prediction — Canonical Objective and Implementation Lock

**Effective 2026-07-28.** This document is authoritative for current KB and
predictor work. It replaces the earlier CPU/memory objective, the stale
3500/5000 ms binary objective, boolean clause composition, and separate offline
model paths. Historical artifacts remain evidence of what ran; they are not
current implementation or evaluation contracts.

## 1. Current objective

Predict clause latency as one probability mass function over fixed ordered
buckets, plus clause CPU peak, sampled RSS, and Disk I/O as independent
Heavy/Light classifications.

The human-authorized fixed boundaries are, in milliseconds:

```text
2000, 8000
```

The mutually exclusive buckets are:

```text
[0, 2000]
(2000, 8000]
(8000, +inf)
```

The predictor returns one normalized three-bucket PMF:

```text
(P(short), P(middle), P(long))
```

The hard prediction is the highest-probability bucket. An exact probability
tie selects the shorter bucket. The primary development score is exact
three-class accuracy:

```text
three_class_accuracy = correct_bucket / eligible_examples
```

The comparator is the most frequent ground-truth bucket on the exact same
evaluation rows. Reports include `eligible_examples`, per-class counts/rates,
predicted class counts, the 3x3 label-by-prediction confusion matrix, majority
class and accuracy, and deltas from majority and current. These are
reconciliation diagnostics, not additional acceptance metrics.

The old eight-boundary accuracies and nine-bin exact accuracy remain historical
diagnostics and must not select a candidate. Do not use balanced accuracy,
precision/recall, Brier/NLL, bucket MAE, q-error, legacy 3500/5000 ms metrics,
or a hand-selected subset to choose the candidate.

The Heavy thresholds are fixed and strict:

```text
peak_cpu_cores > 2.0 cores
sampled_peak_rss_mb > 500 decimal MB
disk_read_write_bytes_total > 104857600 bytes (100 MiB)
```

For these resource targets only, an explicitly policy-marked observation with a
null value and `latency_ms < 500` is an imputed Light label. A null at or above
500 ms remains unavailable. Reports must separate observed Heavy, observed
Light, short-null-imputed Light, and null-unavailable counts. Primary resource
diagnostics include eligible count, Heavy count/rate, TP/TN/FP/FN, accuracy,
and the majority-Light baseline.

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
one ClauseResourceKB / predictor core
        ├── frozen cross-repo public binary/global evidence
        └── causal repo exact/prefix/binary evidence
        ↓
hard repo-first deepest-nonempty backoff
        ↓
latency bucket PMF or resource P(Heavy) + provenance
```

`resource-agentd` owns parsing/canonicalization, prediction, KB persistence,
snapshot pinning, causal update, and telemetry orchestration. `telemetryd` owns
privileged collection and finalized normalized observations. Clients never
implement prediction logic.

`analysis/development/tool-resource-service-architecture.md` remains in force
for everything this section does not restate: the fixed privilege boundary
(a runtime flag must never change which process is privileged), the one-way
dependency that keeps the daemon import graph free of `trace_collect`, the
bounded per-trace telemetry FIFO with `CloseTrace` as the settlement barrier,
and fail-closed backpressure that withholds evidence without delaying or
changing the workload. Read both documents before changing the runtime.

`src/tool_resource/` is a self-contained directory that imports nothing from
this repository, so it can be copied out and used on its own. The offline lane
that reads this repository's trace formats lives outside it, in
`src/tool_resource_eval/`.

Compound-command bucket composition is out of scope. Sequential clauses can
accumulate while pipeline members overlap; until a physical composition
contract is separately approved, compound commands return an explicit
unavailable result rather than ORing, adding, or maximizing bucket IDs.

## 4. Evidence boundary

Canonical telemetry validity remains call-granular. Downstream consumers use
only calls marked eligible for KB ingestion. Withheld or missing observations
are never negative labels or zero-valued targets, except for the explicit,
per-observation short-null resource-label policy above.

The locked SWE-100 and fresh-277 traces are development-exposed diagnostic
inputs. They may be used in both fit/evaluation orientations to implement,
replay, and compare this mechanism but cannot support a confirmation claim.
No other benchmark is in scope for the current three-bucket development goal.

**2026-07-27 development amendment:** `generic-argv-v1` results were visible
before `generic-argv-v2-shape` was defined. V2 removes plaintext opaque values
from cross-repo keys and preserves numeric order of magnitude. Both versions
remain non-confirmatory diagnostics; the production public layer remains
binary/global until a candidate and criterion are frozen.

The local-vs-frozen-public counterfactual gate also failed: pooled local
support `n <= 4` lost by `1.17 pp` in one exposed orientation and tied in the
other. Do not implement fixed support-aware shrinkage from this evidence; no
runtime arbitration change is selected.

**2026-07-28 resource-class amendment:** the human selected the three thresholds
above and the short-null label policy. The implementation reuses the exact
latency KB hierarchy, strict trace-close visibility, and empirical first-hit
node; no second model was added. The first two development-exposed SWE
orientations show no stable gain over majority-Light (Disk improves only in
one orientation). These results validate plumbing, not predictive skill or a
confirmation claim.

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
the exact `2000/8000` boundaries. Report three-class accuracy, the
same-evaluation majority baseline, the 3x3 confusion matrix, class counts, and
the required provenance. This is a diagnostic baseline, not confirmation.
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

## 6. Amendment record

An open amendment is legitimate; an amendment described as pre-registration is
not. Each entry records the date and what was visible when the criterion moved.

### 2026-07-28 — three-bucket development objective

Visible before this amendment: all earlier eight-boundary/nine-bin SWE
diagnostics, `generic-argv-v1`, `generic-argv-v2-shape`, the failed
`local_n <= 4` arbitration gate, and both SWE resource-class orientations.

The human replaced the eight latency boundaries with `2000/8000` after deciding
that three operational latency regimes—short, uncertain middle, and long—were
sufficient. Exact three-class accuracy against the same-evaluation majority
class is now the primary development decision metric. This is an openly
development-exposed amendment, not a pre-registration or confirmation claim.
CPU, sampled RSS, Disk I/O, short-null policy, evidence eligibility, causal
visibility, and compound-command semantics are unchanged.

### 2026-07-27 — concurrency-one diagnostic

Concurrency one (`c1`) was authorized after the concurrency-two (`c2`) rung-20
results were visible. It tests whether cross-trace concurrency caused unstable
promotion; it is an openly development-exposed diagnostic, not a pre-registered
confirmation. The `c1` replay reproduced the same missing-evidence pattern, so
concurrency was rejected as the cause.

### 2026-07-27 — call-scoped promotion

Before this amendment, four development replays had shown stable call mapping
(82 of 139 calls eligible) but unstable promoted-observation counts
(82, 50, and 81 were observed), including traces with eligible calls and zero
promotion. Inspection then identified a trace-level promotion filter that
discarded every valid call when any call-level telemetry RPC made the trace
status unavailable.

Promotion is amended to retain individually eligible observations whenever the
session collector-health, loss, and cleanup gates passed. Trace/run lifecycle
validity remains reported separately; a failed call contributes no observation
and no longer voids eligible sibling calls. This correction can raise measured
yield, so results produced before and after it are not directly comparable.

The first post-change replay exposed a second lifecycle defect that the new
daemon logs made observable: the SQLite store contained all 82 eligible
observations, but run manifests reported only 81. Resource runs were opened
before the serial task queue started; one healthy closed trace and two not-yet-
started tasks crossed the 1800-second run lease. The earlier TTL hypothesis had
been rejected using trace-open times, which are not run-open times.

Active run/session lifetime is therefore amended to follow the reuse-safe local
client process identity obtained from the Unix socket, not elapsed RPC
inactivity. Eligible observations become visible when trace finalization has
established healthy collector, loss, and cleanup state; `CloseRun` reports the
already-settled observations but does not control their visibility. TTL remains
only for bounded retention of completed results and unacknowledged finalized
observations. Collector loss and cleanup failure remain fail-closed.
`promoted_observation_count` is the sum of store-confirmed promotion row counts;
any mismatch between that count and the eligible observation IDs blocks
settlement.

`traces/terminal-bench/tb-dev10-resource-agentd-50x-20260726-r3` is a pre-fix,
development-exposed baseline. The exact-cohort direct comparison for the
call-scoped promotion correction is
`traces/terminal-bench/tb-dev20-c1-ladder-20260727-r4`.

### 2026-07-27 — command-scoped best-effort attribution

The dev-20 r9 replay and targeted kernel, make-mips, Lean, and FMRI diagnostics
were visible before this amendment. They showed complete failed `execve`
attempts, uniquely identifying capped argv prefixes, and ambiguous runtime
occurrences whose downstream clause observations were identical.

Within one tool call's command only, attribution now admits three additional
evidence forms:

- complete failed attempts whose errno values are all `ENOENT`, or exact
  source/replay shell lookup failures with direct status 127 or a zero status
  proven to come from an in-command `|| true`, produce explicit zero
  target-program latency, CPU, memory, and disk observations;
- a collector-capped argv with no truncated captured word may match through its
  complete captured prefix; and
- equal-cardinality ambiguous candidates may be paired deterministically when
  their downstream observation identity is identical, including the fields
  that control KB eligibility.

No evidence is matched across commands or tool calls. No tolerance for
different downstream identities is defined by this amendment; adding one
requires an explicit result-affecting threshold. These rules can raise yield,
so pre/post attribution results are not directly comparable.

### 2026-07-27 — objective replacement and lineage merge

Visible before this amendment: the telemetry-lifecycle replay ladder through
`tb-fmri-best-effort-attribution-20260727-r4`, and the separately developed KB
and predictor work on `dev/tool-resource-kb-predictor`.

Two consequences, both openly development-exposed:

- **Objective replaced.** The previous right-open `[b_i, b_{i+1})` latency-bucket
  contract is superseded by §1 of this document: left-open `(b_i, b_{i+1}]`
  intervals matched to the scheduler question `T > b`, a normalized bucket PMF,
  and independent CPU-peak / sampled-RSS / Disk-I/O Heavy-Light targets on the
  fixed boundaries recorded there. Bucket-labelled results produced under the
  right-open convention are not comparable to results produced after it, and
  cannot be reinterpreted by relabelling.
- **Lineage merged.** The telemetry-lifecycle branch and the KB/predictor branch
  diverged at commit `c281d48` and were merged on 2026-07-27. Replay numbers
  produced on either branch before the merge are not directly comparable to
  numbers produced after it. The named cohorts above retain their meaning only
  within their own pre-merge lineage.

No claim-bearing evaluation had been read from either lineage when this
amendment was written, and the Terminal-Bench confirmation attempts remain
untouched.

## 7. Task contract

Every result-affecting KB/predictor task must preserve:

```text
Targets = latency bucket PMF plus CPU peak, sampled RSS, and Disk I/O Heavy/Light.
Boundaries_ms = [2000, 8000].
Latency truth = exact bucket in [0,2000], (2000,8000], or (8000,+inf).
Latency prediction = argmax normalized PMF; exact ties select the shorter bucket.
Primary latency score = exact three-class accuracy vs same-evaluation majority.
Resource truth = CPU > 2 cores; RSS > 500 decimal MB; Disk read+write > 100 MiB.
Short-null resource policy = Light only when explicitly marked and latency_ms < 500.
Scores include n, positive rate, TP/TN/FP/FN, accuracy, and majority-Light baseline.
Offline and online use one predictor implementation and identical causal updates.
No compound composition, legacy predictor path, or non-SWE benchmark access.
```
