# CPU Feedback Admission Protocol

## Question

Does causal command-level CPU feedback improve queue completion by changing
which ready commands fit on an eight-core host, beyond a static Task-Aware
prediction?

This is a development mechanism test. All SQLGlot tasks and predictor outputs
used here are already development-exposed. It cannot support a generalization
claim or tune the predictor.

## Frozen comparison

- Population: the existing 50-task SQLGlot validation cohort.
- Workload orders: the existing 32 deterministic seeds, each selecting 40
  tasks exactly as the current admission evaluator does.
- Host capacity: 8 CPU cores and 16,000 MB RSS.
- Scheduling: FCFS-ready command admission with work-conserving backfill.
- Primary baseline: static Task-Aware CPU and RSS requests.
- Candidate: the same initial Task-Aware requests, followed by causal CPU
  feedback. RSS requests never change.
- Context only: fixed-eight admission and the already measured equal-share
  burstable control. They do not determine the incremental feedback verdict.

Every arm uses identical tasks, command order, source CPU work, and initial
delays. A missing prediction uses 8 CPU cores and 16,000 MB RSS.

## Feedback semantics

For commands with a valid 0.5-second CPU timeline collected at eight cores,
the simulator replays source CPU work uniformly within each source interval.
The command begins at its Task-Aware CPU page: 2, 4, or 8 cores.

Every 0.5 seconds of simulated wall time, the controller observes only CPU
work actually served during that interval and whether the command was
throttled. After 0.14132007875 seconds, it requests:

- 8 cores if throttling was observed; otherwise
- the smallest of 2, 4, or 8 cores covering the observed mean CPU rate.

A shrink applies immediately and releases capacity. An expansion applies only
when free capacity is sufficient; otherwise the current page remains active
and the controller makes a new decision after the next observation interval.
Ready commands may start whenever their current CPU and static RSS requests
fit. Queue waiting is outside command service time.

At one timestamp, the simulator processes command completions, then all
shrinks, then expansions, then new admissions. Ties within a class use the
existing ready-order key: ready time, seed rank, and task ID. This gives an
already-running command priority over newly ready work without adding another
priority policy.

Commands without a valid CPU timeline retain their static Task-Aware request
and recorded duration; feedback is disabled for them. This keeps lack of
telemetry from becoming a hindsight feature.

## Outputs and gate

Report, for every arm and workload order:

- mean task completion time, batch makespan, total command queue time;
- total modeled command service time and service inflation versus recorded
  eight-core service;
- reservation core-seconds, maximum concurrent commands, denied expansions,
  and capacity violations.

The primary comparison is candidate minus static Task-Aware mean task
completion time. The mechanism advances to a physical replay only if all are
true:

1. relative reduction of mean task completion, averaged over the 32 fixed
   orders, is at least 5%;
2. the upper endpoint of the paired order-bootstrap interval is below zero;
3. total modeled command service inflation versus recorded eight-core service
   is at most 5%;
4. there are no CPU or RSS capacity violations.

The bootstrap measures sensitivity to workload order, not uncertainty over a
new task population. Failure of this gate stops the physical replay and must
be diagnosed as prediction initialization, expansion pressure, insufficient
release, or service inflation. It is not evidence against resource feedback
on other workloads.

## Implementation boundary

Extend the existing admission event loop and its focused tests. Reuse the
current SQLGlot program builder, Task-Aware prediction maps, feedback constants,
and result conventions. Do not add a scheduler framework, daemon, runtime IPC,
new predictor rule, prompt, dataset-specific parser, or physical controller in
this phase.

After a passing result, design the physical controller separately. Its first
action must be a small smoke that measures runtime and resource cost; a run
estimated above 30 minutes requires explicit approval.
