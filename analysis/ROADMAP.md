# Research frontiers

This is the single compact roadmap. Current metrics, retained evidence, exposed
datasets, and frozen gates remain authoritative in
[`development/tool-resource-canonical-objective.md`](development/tool-resource-canonical-objective.md).

## Question and boundary

Treat an agent as a long-lived job that alternates between GPU inference and
remote CPU tool execution while carrying state on both sides:

> Can phase and state information guide admission, priority, placement, and
> state retention so that agents finish sooner without harming inference tails?

The scheduling unit is an agent session. Actions occur only at LLM/tool
boundaries.

| Kind | In scope |
|---|---|
| Observed state | Current phase; GPU queue; KV/prefix location, size, and reuse cost; tool workspace/container location, version, size, and restore cost; causal clause/eBPF feedback |
| Predicted state | Distribution over tool return time and CPU/RSS/I/O demand |
| Actions | Borrower admission and return priority; replica routing; KV retain/offload/evict; whole-task CPU-worker placement; tool-state keep/park/restore |
| Objectives | Session JCT, makespan, all-request tail TTFT, resource-time, and measured state movement cost |

SSH is deployment machinery; a narrow RPC connects the coordinator and remote
tool workers. Host-local telemetry keeps its UDS privilege boundary. EAR owns
dynamic CPU/RSS elasticity inside workers, so static container right-sizing is
not the contribution.

Keep these meanings of snapshot separate: the existing immutable KB evidence
snapshot is for reproducibility; GPU KV/prefix state and tool workspace/container
state are scheduling state. The first system excludes arbitrary process
checkpointing, per-command workspace migration, online LLM policy generation,
a new RPC framework, and dynamic TP reconfiguration.

## Decision sequence

### F0 — Protect returns while using tool gaps

Run the already frozen PennyLane `fixed / feedback / native-priority+feedback`
comparison without changing its cohort, order, hardware, cost, or gate. It asks
whether tool-gap completion-time gains can coexist with safe TTFT tails. On
failure, do one bounded queueing-versus-KV-loss diagnosis; continue only if the
observed state identifies a concrete action.

### F1 — Test tool-state parking headroom

Before implementing checkpointing, compare `always resident` with a zero-cost
`perfect parking` upper bound on existing PennyLane trajectories. Freeze the
screen before outcome access: it must change a real admission decision and
improve mean task completion by at least 5%, or the tool-snapshot branch closes.
A pass authorizes measuring warm reuse, park/restore, and cold rebuild costs.

### F2 — Place remote tools using state locality

Enter only if F1 still passes after real restore costs are charged. Reuse EAR,
keep each task on one worker by default, and compare fixed placement,
least-loaded placement, and snapshot-locality-plus-capacity placement. The last
must beat both simple baselines physically after preparation, transfer, restore,
and queueing time are included.

### F3 — Route LLM returns using KV locality

With at least two replicas, compare least-loaded routing, KV-local routing, and
one cost rule: leave the local replica only when avoided queueing exceeds the
measured state-reuse loss. Charge scheduler wait and repeated prefill to JCT and
TTFT. No queue-versus-reuse crossover closes this branch.

### F4 — Study RP x TP only after locality works

Confirm whether RP means replica or request parallelism. If it means replicas,
start with fixed-topology `2 x TP1` versus `1 x TP2`; four GPUs permit
`4 x TP1 / 2 x TP2 / 1 x TP4`. Dynamic reconfiguration is considered only if
real phases prefer different layouts long enough to amortize weight and KV
movement.

## Where tool understanding contributes

Clause/eBPF feedback describes current work, the KB/predictor estimates future
tool work, and snapshot metadata prices state reuse. Evaluate them in order:

1. phase feedback only;
2. phase feedback plus observed state locality;
3. state locality plus tool prediction.

Prediction contributes only if the third arm changes actions and improves
physical session outcomes; accuracy alone is insufficient.

The paper arc is:

`tool-gap backfill -> dual-state parking/placement -> RP x TP interaction`

GPU/KV, CPU/RSS, and state-location fragmentation must be reported separately.
Simulation and zero-cost oracles establish headroom only; allocation,
snapshotting, and migration claims require physical execution.
