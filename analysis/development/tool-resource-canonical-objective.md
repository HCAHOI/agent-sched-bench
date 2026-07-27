# Tool-Resource Prediction — Canonical Objective Lock

**Effective 2026-07-26.** This correction is authoritative. It replaces the
stale three-binary-label, 3500/5000 ms balanced-accuracy, and boolean clause-OR
objective. Historical artifacts remain evidence about what was run; they are
not instructions for current optimization.

## Final predictor family and current scope

The final predictor family is **log-spaced bucket prediction** for command
latency, peak CPU, and peak memory.

The current scope implements and evaluates **latency buckets only** while the
new canonical resource-telemetry run finishes. CPU and memory buckets are
intentionally deferred, not removed from the final objective. Do not derive
current CPU or memory bucket claims from legacy `pacct`, cgroup peaks, or
whole-container sampled memory.

## Bucket contract

For an ordered boundary sequence

```text
0 = b_0 < b_1 < ... < b_k
```

bucket `i < k` is the right-open interval `[b_i, b_{i+1})`, and bucket `k` is
`[b_k, +inf)`. The intervals are mutually exclusive and exhaustive for
non-negative latency. A value exactly equal to `b_i` belongs to bucket `i`.

The positive boundaries `b_1 ... b_k` must be:

- finite, strictly increasing, and expressed in milliseconds;
- explicitly supplied to the current API/evaluator, with no implicit default;
- frozen before any claim-bearing evaluation; and
- log-spaced according to an advisor-approved rule recorded with the run.

No authoritative numeric latency boundaries currently exist in this
repository. The deleted historical bucket prototype derived ad hoc thresholds
from KV profiles and used the incompatible interval convention
`(b_i, b_{i+1}]`; it is not a boundary specification. Choosing the exact
boundaries and log-spacing rule is therefore a required decision before
claim-bearing evaluation.

Until that decision is recorded, tests may use explicit fixture boundaries to
validate plumbing and boundary semantics, and development-exposed traces may
be used only for diagnostics. Do not optimize or report the old 3500/5000 ms
balanced accuracy as a success criterion.

## Current latency path

The current path predicts a latency bucket ID before command execution using
only causally available history. Bucket IDs are categorical interval labels:
they cannot be ORed, added, maximized, or otherwise composed as if they were
independent boolean flags.

Parser, clause identity, causal KB lookup, clause bridge, mapping evidence,
and telemetry-integrity work remain useful. Clause-level latency history may
serve as auxiliary evidence. A compound-command bucket composer is not part of
this scope: sequential clauses may accumulate while pipeline members overlap,
so any future composer requires its own explicit physical contract and
evaluation.

The former per-clause three-valued boolean OR and its latency-long targets are
historical diagnostic code only. They are not the current latency API or
evaluation contract.

## Telemetry and evidence boundary

Canonical clause telemetry remains the source for the eventual latency, CPU,
and memory bucket family. Preserve its collector, bridge, attribution,
availability, mapping, loss, and cleanup gates. Do not interrupt or modify an
active telemetry run or its artifacts.

Telemetry validity is call-granular. Attempt artifacts report workload
execution, collector health, and formal mapping completeness separately. A
healthy attempt with withheld calls is `partial`: downstream consumers use
only calls with `eligible_for_kb=true` and never discard its valid calls or
treat withheld clauses as negative observations. Collector failure, loss, or
cleanup failure remains `unavailable` and contributes no KB evidence.

The legacy SWE `pacct` / bash-xtrace replay and fresh-277 segment timeline are
development-exposed diagnostic proxies. They may exercise plumbing, but they
cannot support canonical resource claims. SWE-100/fresh-277 are
development-exposed; the untouched Terminal-Bench confirmation attempts must
remain untouched.

## Amendment record

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

## Runtime architecture lock

The canonical runtime architecture is specified in
[`tool-resource-service-architecture.md`](tool-resource-service-architecture.md).
It defines two independently deployed modules with fixed privilege boundaries:

- privileged `telemetryd` owns cgroup resolution, eBPF lifecycle, attribution,
  loss/cleanup checks, and finalized normalized observations;
- unprivileged `resource-agentd` owns parser/canonicalization, prediction, KB
  persistence, snapshot pinning, causal update, and telemetry orchestration.

These are modules, not permission modes. Trace clients connect only to
`resource-agentd`; raw eBPF events never leave `telemetryd`. After migration,
per-worker privileged auto-start, in-process KB ownership, direct telemetry
observer APIs, and compatibility aliases are deleted unless a current caller is
proved to require them.

The online daemon import graph and wire protocols do not depend on
`trace_collect` or its action schema. A trace runner may adapt its records into
the normalized tool-resource client protocol; that dependency is one-way.
The online client path synchronously performs only local clause parsing and
prediction. Telemetry attachment and call collection run through a bounded
per-trace FIFO; `CloseTrace`, after workload completion, is the settlement
barrier. Backpressure or telemetry failure withholds evidence fail-closed and
never delays or changes the workload. Formal replay joins asynchronous
attachment once, before workload timing begins; this setup barrier is not part
of the online Begin/End path.

## Task contract

Every task that changes tool-resource data, prediction, evaluation, or
scheduler integration must include this lock:

```text
Final family = log-spaced latency/CPU/memory buckets.
Current scope = latency buckets only; CPU/memory buckets are deferred.
Intervals = [b_i, b_{i+1}), final bucket [b_k, +inf).
Numeric latency boundaries have no default and must be frozen before claims.
Bucket IDs are not composed with boolean OR.
Legacy binary and proxy results are noncanonical diagnostics.
```
