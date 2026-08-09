# CPU-Idle Speculative Backfill Oracle

## Decision question

Can command prediction improve scheduling by choosing one waiting command to
run at idle CPU priority, without slowing the normal command?

This is a development-only oracle test on exposed SQLGlot traces. It tests
whether the action and its selection problem have enough headroom before any
Clause-KB or Task-Aware policy is implemented.

## Why this action is different

Previous policies used predicted CPU classes as hard quotas, shares, or
logical admission pages. Those actions either slowed commands or left too
little incremental benefit after work-conserving borrowing.

Here prediction never caps normal work. One normal command has first claim on
all eight cores. At most one speculative command may use only the CPU capacity
left idle by the normal command. Linux cgroup v2 exposes this physical action
as `cpu.idle=1`, which makes a non-root cgroup very low priority relative to
normal peers, as documented in the
[kernel cgroup v2 CPU interface](https://docs.kernel.org/admin-guide/cgroup-v2.html#cpu-interface-files).
The replay uses strict priority as an optimistic ceiling; a passing result
still requires a real calibration of interference and promotion cost.

## Frozen population and controls

- Population: the existing 50-task SQLGlot validation cohort.
- Orders: the existing 32 deterministic seeds, each selecting 40 tasks.
- Capacity: 8 CPU cores and 16,000 MB RSS.
- Commands, task delays, and source CPU profiles are identical across arms.
- Every command must have a complete CPU profile. Missing or ambiguous RSS
  evidence makes a command ineligible for speculative selection.
- Disk and network receive no modeled benefit because the traces do not contain
  counterfactual contention curves for them.

All evidence is development-exposed. No result can authorize a confirmation
claim or runtime integration.

## Frozen execution semantics

Only commands that are causally ready in the existing task program may run.
At most one command is normal and at most one is speculative.

1. The oldest ready command becomes normal when no normal command is running.
2. The normal command receives up to eight cores, capped by its current source
   demand.
3. A speculative command receives only the remaining physical CPU capacity.
4. When the normal command finishes, the existing speculative command is
   promoted to normal. Promotion preserves completed CPU work.
5. When a speculative command finishes first, another eligible waiting command
   may take the speculative slot.
6. CPU work is never discarded or duplicated. Physical allocation never
   exceeds eight cores.
7. Normal and speculative RSS reservations must sum to at most 16,000 MB.

Commands remain in their original task workspaces and order. The scheduler
does not rewrite, split, retry, or otherwise change a tool call.

## Frozen arms

### Serial-8 control

Run only the oldest ready normal command. It has first claim on all eight
cores. This measures the safe no-speculation operating point.

### FCFS idle backfill

Use the oldest RSS-eligible ready command as the speculative command. This arm
has no predictor and measures the value of the native action alone.

### Oracle-selected idle backfill

Among RSS-eligible ready commands, select the one with the shortest recorded
source duration. RSS eligibility uses the observed command peak and rejects
unavailable peaks. This is hindsight and may only measure selection headroom;
it is never a deployable policy.

Both backfill arms share the same hindsight RSS-fit filter. It is an oracle
safety control, not evidence that FCFS backfill is currently deployable.

No alternative oracle ranking, number of speculative slots, resource filter,
or tie-break may be tried after outcomes are read. Ties retain ready time,
seed rank, then task ID.

## Frozen gates

Report per arm and order: mean task completion, makespan, queue time, command
service, service inflation versus recorded eight-core service, CPU work,
normal/speculative starts and completions, promotions, speculative CPU work,
RSS capacity, and physical CPU capacity.

The action mechanism advances only if FCFS idle backfill versus Serial-8 has:

1. at least 5% mean per-order task-completion reduction;
2. paired order-bootstrap upper endpoint below zero;
3. service inflation at most 5%;
4. makespan regression at most 1%; and
5. zero CPU/RSS violations with exact CPU-work conservation.

Prediction selection has headroom only if Oracle-selected versus FCFS idle
backfill additionally has:

1. at least 10% mean per-order task-completion reduction;
2. paired order-bootstrap upper endpoint below zero; and
3. the same service, makespan, capacity, and work-conservation bounds.

If the action gate passes but the selection gate fails, idle backfill may be a
useful control but Clause-KB and Task-Aware selection are stopped. If both pass,
a separate protocol may compare one frozen Task-Aware selector with the
Clause-KB selector and must require at least 5% improvement over FCFS plus at
least one percentage point over Clause-KB.

## Implementation and stop boundary

Extend the existing event replay with one strict-priority speculative slot and
one narrow evaluator. Reuse the committed task programs, CPU profiles, split,
orders, bootstrap, and provenance checks. Do not add a scheduler framework,
runtime IPC, cgroup control, new collection, or predictor policy in this phase.

Before reading the single formal outcome: add focused allocation, promotion,
RSS-fit, and CPU-work-conservation tests; obtain one bounded independent review;
commit the implementation; then run the frozen evaluator once. Expected replay
cost must be measured by a smoke and must remain below 30 minutes.
