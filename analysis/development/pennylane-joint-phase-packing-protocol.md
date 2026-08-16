# PennyLane Joint Phase-Packing Protocol

**Status:** frozen before joint scheduling outcomes
**Role:** development-only action-space screen; not physical GPU evidence

**Pre-outcome amendment, 2026-08-16:** the initial 28-core capacity was an
unverified host assumption. Input preflight stopped before computing any arm:
37/70 ordinary traces exceeded 28 sampled cores, the maximum was 42.7186, and
their run manifests recorded no CPU limit. CPU capacity is therefore the
conservative observable lower bound `ceil(42.7186) = 43` cores. The cohort,
algorithms, comparisons, and GO gate are unchanged.

## Question and evidence

Does exact coordination of agent LLM phases and host CPU/RSS phases improve
completion beyond the best feasible single-resource task-admission controls?

- Use the 70 ordinary, unaccelerated tasks in
  `pennylane-all76-clean-ebpf-20260816` with non-empty `resources.json`.
- Freeze the 152 consumed `trace.jsonl`/`resources.json` files by the canonical
  tree digest `53d8faa0a6bd98dcfb38bfdf0916c196ff8d54ae6a305828a7b6f8509f3f7f16`.
- Exclude exactly the six `source_scaled` replay tasks with empty task-resource
  samples: 5538, 5582, 5761, 5846, 5851, and 5866.
- All 70 tasks arrive at time zero in lexical task-ID order. All evidence is
  already development-exposed.

## Frozen model and arms

- Discretize at two seconds, the collection sampler's nominal interval.
  CPU/RSS samples are held until the next sample; a bin touched by an LLM call
  consumes one request slot.
- Capacities are four simultaneous LLM requests, 43 CPU cores, and 80,000 MiB
  RSS. Four is the prior A100 reference cap; CPU/RSS are the collection-node
  capacity proxies. The CPU value is the amended observable lower bound above.
- A task may start at a bin boundary. Its recorded internal phase durations and
  resource profile never change, so modeled service inflation is zero.
- `static`: best feasible global active-task cap.
- `gpu_only`: exact future LLM-slot fit plus a global cap; choose the feasible
  cap with lowest mean completion.
- `tool_only`: exact future CPU/RSS fit plus a global cap; choose the feasible
  cap with lowest mean completion.
- `joint`: exact future LLM-slot and CPU/RSS fit with no extra cap.
- Cap searches cover every integer from one through the uncapped arm's realized
  maximum concurrency; ties choose the smaller cap. Feasibility always requires
  zero violations of all three capacities.

## Metrics and gate

Report mean task completion, makespan, starts before first completion, maximum
active tasks, LLM request-slot occupancy, CPU/RSS utilization, peaks, and
capacity violations.

GO only if joint:

1. has zero LLM, CPU, and RSS capacity violations;
2. lowers mean completion by at least 5% versus both best feasible `gpu_only`
   and best feasible `tool_only`;
3. has makespan no higher than either single-resource arm.

The static arm is a deployment baseline, not an additional gate. Recorded
Codex LLM intervals are an occupancy proxy, not an A100 service curve; a GO can
authorize a physical GPU experiment but cannot support a GPU-performance claim.
