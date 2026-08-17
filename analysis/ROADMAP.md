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
| Observed state | Current phase; GPU queue; KV/prefix location, size, and reuse cost; causal clause/eBPF feedback |
| Predicted state | Distribution over tool return time and CPU/RSS/I/O demand |
| Actions | Borrower admission and return priority; replica routing; KV retain/offload/evict |
| Objectives | Session JCT, makespan, all-request tail TTFT, resource-time, and measured state movement cost |

SSH and RPC are deployment machinery, not the current contribution. Host-local
telemetry keeps its UDS privilege boundary. EAR owns dynamic CPU/RSS elasticity
inside workers, so static container right-sizing is not the contribution.

Keep snapshot meanings separate: the existing immutable KB evidence snapshot is
for reproducibility; GPU KV/prefix state is scheduling state. Tool-container
parking is closed below. The first system excludes arbitrary process
checkpointing, workspace migration, online LLM policy generation, a new RPC
framework, and dynamic TP reconfiguration.

## Decision sequence

### F0 — Protect returns while using tool gaps

Run the already frozen PennyLane `fixed / feedback / native-priority+feedback`
comparison without changing its cohort, order, hardware, cost, or gate. It asks
whether tool-gap completion-time gains can coexist with safe TTFT tails. On
failure, do one bounded queueing-versus-KV-loss diagnosis; continue only if the
observed state identifies a concrete action.

### F1/F2 — Tool-state parking and remote placement are closed

The zero-cost perfect-parking screen changed starts but worsened mean completion
2.664%: 27 tasks advanced by 113,402 task-seconds while 30 were delayed by
139,700. It removed only 3.178% of CPU-core-time and 1.316% of RSS-time; baseline
RSS utilization was 14.171%. Therefore do not measure restore costs or build the
remote snapshot-RPC branch for this workload. Reopening requires a new workload
with independently demonstrated CPU/RSS pressure.

### F3 — Measure KV-local return placement on two replicas

The current corpus shows frequent long-context returns and substantial source
provider cache reuse, but its original tasks ran serially and contain no replica
identity, vLLM KV hit, or per-replica queue state. The opportunity is therefore
underdetermined. With two replicas, compare least-loaded routing, KV-local
routing, and one cost rule: leave the local replica only when avoided queueing
exceeds measured state-reuse loss. Charge scheduler wait and repeated prefill to
JCT and TTFT. No queue-versus-reuse crossover closes this branch.

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

`tool-gap backfill -> KV-local replica placement -> RP x TP interaction`

GPU/KV, CPU/RSS, and state-location fragmentation must be reported separately.
Simulation and zero-cost oracles establish headroom only; allocation,
snapshotting, and migration claims require physical execution.
