# CPU-Idle Backfill with Predicted RSS Safety

**Status:** run once and recorded on 2026-08-09. The gate failed on two of its
eight conditions; see Outcome below.

## Question

Can the frozen Task-Aware RSS predictor replace hindsight RSS-fit in the
successful CPU-idle FCFS backfill action without creating memory-capacity
exposure?

This is a development evaluation on already exposed SQLGlot tasks. It tests a
new predictor role: safety admission, not candidate ordering. Hindsight
shortest selection is closed and must not be reintroduced.

## Frozen population and action

- Existing 50-task SQLGlot validation cohort.
- Existing 32 deterministic orders, each selecting 40 tasks.
- 8 CPU cores and 16,000 MB RSS.
- One normal command and at most one speculative command.
- FCFS chooses both commands in every backfill arm.
- Normal work has strict CPU priority; speculation receives residual CPU only.
- Commands, task delays, CPU profiles, and source RSS observations are
  identical across arms.
- CPU work must be conserved and physical CPU allocation must not exceed eight
  cores.

All tasks and prediction results are development-exposed. No result is
confirmation or a deployable safety claim.

## Frozen RSS policies

The existing Clause-KB and Task-Aware hard RSS predictions are consumed at the
common causal `BeginCall` point. No prediction, threshold, PMF, provenance
rule, or fallback may be changed.

Hard classes map to reservations as follows:

```text
Low       -> 500 MB
Medium    -> 2,000 MB
High      -> 16,000 MB
Unavailable or unmatched -> 16,000 MB
```

A speculative command starts only when its reservation plus the running normal
command's reservation is at most 16,000 MB. Prediction affects this RSS-fit
decision only. It does not change FCFS order, CPU allocation, priority,
promotion, or command contents.

For safety evaluation, source clause RSS composes with the existing physical
rules: sequential stages use max and pipeline stages use sum. Missing,
ambiguous, unmatched, or structurally invalid source RSS is conservatively
16,000 MB. Predicted admission uses predicted reservations; capacity exposure
uses these source values. No unavailable source row may be silently excluded.

## Frozen arms

1. `serial8`: no speculative command.
2. `oracle_rss_fcfs`: FCFS idle backfill using source RSS for admission. This
   reproduces the action ceiling and is not deployable.
3. `clause_kb_rss_fcfs`: FCFS idle backfill using frozen Clause-KB RSS hard
   predictions.
4. `task_aware_rss_fcfs`: identical action using frozen Task-Aware RSS hard
   predictions.

The existing CPU-idle oracle result is visible and recorded: its FCFS arm
improved mean per-order completion 14.620% versus Serial-8. This protocol is an
open next-stage decision informed by that result, not a preregistration made
before it.

## Frozen metrics and gate

Report per arm and order: mean task completion, makespan, queue time, command
service and inflation versus recorded eight-core service, predicted reserved
RSS, maximum source RSS demand, source-capacity exposure events and commands,
normal/speculative starts and completions, promotions, speculative CPU work,
and CPU-work conservation.

Task-Aware advances to physical `cpu.idle` calibration only if all are true:

1. mean per-order task completion improves at least 5% versus `serial8`;
2. it captures at least 50% of `oracle_rss_fcfs`'s completion reduction versus
   `serial8`;
3. its completion reduction is at least one percentage point larger than
   `clause_kb_rss_fcfs`;
4. the paired order-bootstrap upper endpoint versus `serial8` is below zero;
5. service inflation versus recorded eight-core service is at most 5%;
6. makespan regression versus `serial8` is at most 1%;
7. there are zero source-RSS capacity exposures, including conservative
   unavailable-source exposures; and
8. there are zero predicted-capacity or physical-CPU violations and exact CPU
   work is conserved.

Failure stops predictor-backed CPU-idle admission on exposed SQLGlot. The
prediction-free native action remains retained evidence; failure does not
authorize deleting Clause-KB, Task-Aware prediction, or CPU-idle replay.

## Implementation boundary

Extend the existing idle replay with an explicit per-command RSS reservation
map while retaining source RSS separately for safety accounting. Add one narrow
evaluator that reuses the committed split, task programs, predictions, CPU
profiles, orders, bootstrap, and provenance checks.

Before the single formal run: add focused tests for predicted fit versus source
exposure and conservative unavailable RSS; obtain one bounded independent
review; commit result-affecting code; and measure a one-order smoke. Do not add
runtime cgroup control, telemetry changes, collection, PMF quantiles, alternate
fallbacks, or physical replay in this phase.

## Outcome

Receipt:
[`../results/tool-resource-5-3-3-3-20260804/sqlglot50-cpu-idle-rss-safety-v1/result.json`](../results/tool-resource-5-3-3-3-20260804/sqlglot50-cpu-idle-rss-safety-v1/result.json),
status `development_stop_predictor_backed_cpu_idle`.

**Gate: not met.** Task-Aware failed conditions 3 and 7 of the eight above;
conditions 1, 2, 4, 5, 6, and 8 passed.

| Arm | Completion versus `serial8` | Orders improved | Service inflation | Makespan | Source-RSS exposures |
|---|---:|---:|---:|---:|---:|
| `oracle_rss_fcfs` | -14.620% | 32/32 | 3.073% | -17.226% | 0 |
| `clause_kb_rss_fcfs` | -33.392% | 32/32 | 4.783% | -14.583% | 12,390 |
| `task_aware_rss_fcfs` | -25.984% | 32/32 | 4.561% | -10.525% | 12,439 |

Condition 3 required Task-Aware to beat Clause-KB by at least one percentage
point. It was 7.408 percentage points worse instead. Condition 7 required zero
source-RSS capacity exposures, including conservative unavailable-source
exposures; both learned arms produced roughly twelve thousand. The paired
order-bootstrap interval versus `serial8` was [-3,702.319, -3,437.197] s for
Task-Aware and [-4,738.655, -4,436.349] s for Clause-KB.

Predictor-backed CPU-idle admission is therefore stopped on exposed SQLGlot. As
this protocol states, the failure does not authorize deleting Clause-KB,
Task-Aware prediction, or CPU-idle replay.

One open post-outcome sensitivity followed, recorded in
[`cpu-idle-short-null-amendment.md`](cpu-idle-short-null-amendment.md). Applying
the canonical short-null Low policy to source RSS cut confirmed exposures to
2,702 Task-Aware and 2,638 Clause-KB, still nonzero, so both arms remained
NO-GO on the zero-exposure condition alone. Its receipt is
[`../results/tool-resource-5-3-3-3-20260804/sqlglot50-cpu-idle-short-null-v1/result.json`](../results/tool-resource-5-3-3-3-20260804/sqlglot50-cpu-idle-short-null-v1/result.json).
