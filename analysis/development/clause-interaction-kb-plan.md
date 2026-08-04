# Early-Execution Resource Control Research Plan

**Effective:** 2026-08-04

**Status:** development-only decision tree

This plan extends `tool-resource-canonical-objective.md`. The completed
semantic-signature experiments remain frozen evidence; they are not the active
method. The static predictor remains a cold-start baseline.

## 1. Research question

> After a command starts, can its first short interval of real system behavior
> predict the work that remains well enough to justify changing its resources?

Command text cannot expose cache, installation, download, or dependency state
reliably. Early CPU, memory, disk, and process behavior observes the consequence
of that hidden state directly. This direction does not require two tasks to run
similar commands: every prediction begins from the current command's own past.

Proceed only through this decision tree:

1. Check whether existing traces contain time-resolved observations and leave a
   useful interval after a possible decision.
2. Check with hindsight whether any real resource action could have helped.
3. Fit the simplest prefix-based predictor only if that action space exists.
4. Test a real controller only if predicted actions improve over static
   baselines after observation and actuation costs are charged.

All existing SQLGlot, SWE, and Terminal-Bench traces are development-exposed.
They may diagnose this mechanism because it does not learn cross-task semantic
similarity. They cannot support a confirmation claim.

## 2. Phase 0 — observability and decision window (complete: pass)

Use existing traces only. Do not fit a model or add telemetry first.

For each corpus, inventory whether an eligible command has timestamped samples
inside its execution for CPU, RSS, Disk I/O, and process/exec activity. Record
the native sample interval, timestamp domain, command-boundary alignment, and
validity flags. Final aggregate counters do not count as an early signal.

The **observation interval** is the part of the command already executed when a
decision is made. Phase 0 uses one native sample interval only for its coverage
audit; it does not inspect future-work labels.

Continue only if existing valid traces contain:

- at least 100 commands across at least 20 tasks with one complete observation
  interval and at least one equally long interval still remaining; and
- at least 20 commands across at least 10 tasks with a recorded finite CPU
  quota or memory limit and enough time after the decision. This is only a
  trace-side opportunity screen; Phase 1 must verify a real update.

If traces contain only final aggregates, stop and report the exact missing
fields. Do not reconstruct a prefix from final totals.

Observed on the selected successful SQLGlot100 traces: all 1,916 exec calls
have the 0.5-second CPU/network timeline; 567 calls across all 100 tasks retain
another 0.5 seconds after a causal sample and record a finite CPU quota. The
separate task sampler provides a fully in-command CPU/RSS/Disk interval plus an
equally long future interval for 306 calls across all 100 tasks. The coverage
gate passes. RSS and Disk are sampled at about 2.016 seconds and are aligned by
timestamp, not command-exclusive. Persisted eBPF clause observations are
attached only after the exec completes, so they are not an early signal in the
current interface.

## 3. Phase 1 — hindsight actionability (mechanism pass; utility open)

Run only if Phase 0 passes. The candidate action is CPU allocation among task
containers sharing the measured eight-core host. A cgroup limit by itself is
not a consumer: the comparison must name the shared CPU-budget policy whose
decision and cost change.

First measure the supported limits, update latency, and update cost on the real
runtime. Sampling and updating are serial, so the earliest effective action is
the sample endpoint plus the 50 ms availability pad plus measured p95 update
latency. Repeat the Phase 0 coverage gate at that time. Freeze the CPU action
set and costs before inspecting which actions would have helped. Then use
future samples only as a hindsight upper bound; they are never predictor input.

The native smoke passed on the collection host. Twenty alternating CPU updates
between one and two cores had p95 91.3 ms; twenty memory updates between 256 and
384 MiB had p95 99.7 ms. The CPU-bound container remained live and was not OOM
killed. After charging the CPU p95, 548 SQLGlot exec calls across all 100 tasks
still retain one complete 0.5-second future interval, so coverage remains a
pass. This establishes mechanism availability, not benefit.

Memory control stops here. `memory.max` is a cap rather than a reservation, and
the existing cgroup experiment shows that lowering it below current use can
reclaim and then OOM-kill the workload. A future memory experiment would need a
separate `memory.high` or reserving-admission protocol.

The current replay consumes the CPU/network timeline only to preserve
source-equivalent timeout progress; it does not implement a shared CPU-budget
policy. Do not infer benefit from lower quota-time. First name and freeze a real
consumer, baseline, and false/missed-action costs; otherwise Phase 1 stops.

Continue only if the hindsight policy changes an action for at least 20
commands across at least 10 tasks and its benefit exceeds the observation and
resource-update costs. CPU is the only target in this route.

## 4. Phase 2 — early-prefix prediction

Run only if Phase 1 passes. Reuse the existing causal evaluator and trace
loader. Compare exactly four arms on identical command decisions:

- constant action;
- Current static command predictor;
- early observations only; and
- Current plus early observations.

Start with simple summaries from the current command before the decision:
CPU use, RSS level and change, Disk I/O rate, and process/exec count. Do not add
tool-specific parsing, package names, agent-generated adapters, or a model
sweep. Split causally by task; no sample after the decision may enter a feature.

Primary evaluation is the consumer's action cost, not fit quality. Continue
only if Current plus early observations improves the frozen action objective,
changes actions for at least 20 commands across at least 10 tasks, and is better
than both Current and early-only. Report the fixed full-command latency/CPU/RSS/
Disk classifications as secondary diagnostics on their unchanged eligible rows.

## 5. Phase 3 — real control

Run only if Phase 2 passes. Apply the predicted cgroup change during real task
execution and compare against the unchanged static policy with matched tasks and
order. Charge observation, prediction, and actuation overhead. Report task wall
time, command tail latency, CPU-time, memory failures, and any workload failure.

A smoke validates plumbing but is not evidence. Before any run expected to take
over 30 minutes, report the wall-time and memory estimate and wait for explicit
approval. Fresh confirmation data is considered only after a development run
shows an effect large enough to justify consuming it.

## 6. Artifacts and stop rules

- Phase 0 is a read-only schema and coverage audit; it must not change the KB,
  evaluator, telemetry service, or runtime.
- Reuse existing loaders and runtime mechanisms. Do not create another trace
  reader, controller framework, daemon, or dependency.
- Emit one machine-readable result per completed phase and update the existing
  self-contained HTML only when a phase produces scientific evidence.
- Obtain one bounded independent review before results from a substantial new
  evaluator or controller become evidence.
- Commit each completed phase, including a negative result.

Stop the route when time-resolved data is absent, the post-decision action space
is too small, the hindsight upper bound is not useful, or the predictor does not
improve the real action. A failed gate does not authorize changing the window,
thresholds, corpus subset, or action costs after seeing results.
