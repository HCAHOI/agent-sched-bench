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
