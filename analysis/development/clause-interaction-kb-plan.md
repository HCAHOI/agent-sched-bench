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

## 2. Phase 0 — observability and decision window

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
- at least 20 commands across at least 10 tasks where CPU or memory could
  physically be changed after that decision.

If traces contain only final aggregates, stop and report the exact missing
fields. Do not reconstruct a prefix from final totals.

## 3. Phase 1 — hindsight actionability

Run only if Phase 0 passes. The named consumer is a task container whose cgroup
CPU or memory limit can be kept, raised, or lowered while a command runs.

First measure the supported limits, update latency, and update cost on the real
runtime. Freeze the final decision time as the greater of the native sample
interval and measured p95 update latency, then repeat the Phase 0 coverage gate.
Freeze the action set and costs before inspecting which actions would have
helped. Then use future samples only as a hindsight upper bound; they are never
predictor input.

Continue only if the hindsight policy changes an action for at least 20
commands across at least 10 tasks and its benefit exceeds the observation and
resource-update costs. Report CPU and memory separately even if the final
controller changes them together.

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
