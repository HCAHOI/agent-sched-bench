# CPU Feedback with Work-Conserving Borrowing

## Question

Does causal CPU feedback improve command admission when reservation pages are
used only for capacity accounting and admitted commands may borrow idle CPU?

This is a development mechanism test on already exposed SQLGlot traces. It is
not confirmation and does not authorize runtime integration.

## Frozen comparison

- Population: the existing 50-task SQLGlot validation cohort.
- Orders: the existing 32 deterministic seeds, each selecting 40 tasks.
- Capacity: 8 CPU cores and 16,000 MB RSS.
- Queue: FCFS-ready admission with work-conserving backfill.
- Baseline: static Task-Aware CPU/RSS reservations plus equal-weight,
  work-conserving CPU execution.
- Candidate: identical initial reservations and CPU execution, with causal CPU
  feedback changing only the logical CPU reservation.
- Context only: fixed-eight execution and the committed hard-page feedback
  result. They do not determine the incremental verdict.

Every command must have a valid source CPU profile. Every arm uses identical
tasks, commands, source work, delays, RSS reservations, and equal CPU weights.

## Execution and feedback semantics

Logical CPU reservations are 2, 4, or 8 cores and must sum to at most eight
across admitted commands. They control admission only. They are not CPU quotas.

At each instant, runnable source demand is replayed from the existing
eight-core 0.5-second profiles. Physical CPU is allocated by equal-weight
progressive filling, capped by each command's current source demand. Idle CPU
is therefore borrowed automatically. Any service slowdown caused by concurrent
demand is charged.

Every 0.5 seconds, feedback observes only CPU work already served and whether
the command experienced CPU backlog during that interval. After
0.14132007875 seconds it requests:

- 8 logical cores if any backlog was observed; otherwise
- the smallest of 2, 4, or 8 covering mean served CPU rate.

A shrink immediately releases logical admission capacity. An expansion applies
only if logical capacity is free; otherwise it is retried after the next
observation. A denied logical expansion never caps physical CPU borrowing.

The backlog bit is a causal ideal-observation ceiling for per-cgroup CPU PSI:
Linux `cpu.pressure some total` measures time in which at least one cgroup task
is stalled on CPU, as specified by the
[kernel PSI documentation](https://docs.kernel.org/accounting/psi.html).
Existing traces did not record PSI, so a passing simulation
must still pass a separate real-counter calibration before physical workload
evaluation. The simulation may not be described as a deployable controller.

At one timestamp, process command completions, logical shrinks, logical
expansions, feedback observations, then new admissions. Ties retain the current
ready-time, seed-rank, and task-ID order.

## Frozen gate

Report per arm and order: mean task completion, makespan, queue time, command
service, service inflation versus recorded eight-core service, logical reserved
core-seconds, physical CPU work, concurrency, feedback changes, denied logical
expansions, and capacity violations.

Advance to a real PSI calibration only if all are true:

1. candidate mean per-order task-completion reduction versus the static
   borrowing baseline is at least 5%;
2. the paired order-bootstrap upper endpoint is below zero;
3. candidate service inflation versus recorded eight-core service is at most
   5%;
4. CPU/RSS logical capacity is never violated and physical CPU work is
   conserved.

The bootstrap measures workload-order sensitivity, not new-task uncertainty.
Failure stops this feedback-plus-borrowing branch without deleting the retained
feedback or burstable baselines. No threshold, predictor, task selection,
weight, or feedback rule may be changed after outcomes are read.

## Implementation boundary

Extend the existing feedback-admission event loop with one work-conserving
allocation mode and one narrow evaluator. Reuse the reviewed source-profile,
prediction, split, bootstrap, and provenance code. Do not add telemetry fields,
runtime IPC, Docker control, a scheduler framework, or a physical replay in
this phase.
