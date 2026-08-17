# PennyLane Perfect Container Parking Protocol

**Status:** frozen before perfect-parking outcomes
**Role:** development-only zero-cost action-space screen

## Question and evidence

Can releasing a task container's sampled CPU/RSS footprint while its agent is
waiting on an LLM change joint admission enough to improve completion time?

- Reuse exactly the 70 ordinary tasks, lexical arrival order, two-second bins,
  capacities, exclusions, and input-tree digest frozen by
  `pennylane-joint-phase-packing-protocol.md`.
- All inputs and the always-resident joint result are development-exposed.
- Do not use command labels, future outcomes beyond the already frozen exact
  phase profiles, or the six source-scaled replay tasks.

## Arms

- `always_resident`: the existing uncapped joint arm, including each task's
  sampled CPU/RSS profile during bins touched by an LLM call.
- `perfect_parking`: the same task profiles and joint admission algorithm, but
  set that task's CPU and RSS to zero in every bin where its LLM occupancy is
  nonzero. Keep its duration, LLM occupancy, task order, and non-LLM CPU/RSS
  unchanged.

Parking is instantaneous, lossless, and free. A partially LLM-covered bin is
fully released. This deliberately optimistic arm is an upper bound, not a
checkpoint implementation or physical performance claim.

## Metrics and gate

Report both arms' mean task completion, makespan, utilization, peaks, capacity
violations, per-task starts/completions, and the number and total amount of
earlier starts. Also report removed CPU-core-seconds and RSS-MiB-seconds.

`GO` requires all of:

1. `perfect_parking` has zero modeled capacity violations;
2. at least one task starts earlier than in `always_resident`;
3. mean task completion improves by at least 5%.

Otherwise the tool-container snapshot and remote snapshot-RPC branch closes.
A `GO` authorizes only measurement of warm-resident, park/restore, and cold
rebuild costs. Those costs must be charged before any distributed implementation
or scheduling claim.

## Interpretation

Use released resource-time, changed starts, and the binding resource profiles
to explain the outcome. Do not reinterpret a failed gate through a favorable
subset. Simulation establishes opportunity only; resource release and restore
claims require physical execution.
