# PennyLane Causal Joint Tool-Admission Protocol

**Status:** frozen before candidate outcomes
**Role:** development-exposed causal mechanism test; not physical evidence

## Question and cohort

Can the selected Task-Aware command predictor plus causal resource feedback
queue ready tools safely enough to improve task completion over a
prediction-free serial-tool control, while sharing four LLM request slots?

- Use the 70 ordinary tasks from `pennylane-all76-clean-ebpf-20260816` and the
  frozen input-tree digest
  `53d8faa0a6bd98dcfb38bfdf0916c196ff8d54ae6a305828a7b6f8509f3f7f16`.
- Bind the frozen public predictor inputs to SHA256
  `1441c0113752b1508341a83b303cc0f059d3455cd78debd0a584fa051e93318e`
  (SWE100) and
  `f4dc3e360d6f24c0c2436da956a9d1adf448defaa0a41034bb9373d2f8e6ee7c`
  (SWE277).
- Exclude the same six `source_scaled` replay-only tasks as the joint-ceiling
  protocol.
- Lexically sorted tasks 1–35 supply settled predictor evidence; tasks 36–70
  form the replay queue. All 70 trajectories are already development-exposed,
  so this can select the next mechanism but cannot confirm a paper claim.
- All replay tasks arrive at time zero in lexical order. At most eight tasks
  are active and four LLM requests execute concurrently, matching the existing
  task-8/LLM-4 operating point rather than tuning a new concurrency limit.

## Causal replay model

- Retain the prior two-second held CPU/RSS samples, but cut each task trajectory
  at both sample boundaries and exact action boundaries. This yields
  piecewise-constant segments without rounding short calls into two-second
  commands. The structural preflight found 2,973 exec actions, every one with a
  CPU timeline, and no overlapping actions; every action interval must remain
  represented exactly.
- A task advances one local segment only after its newly ready LLM or exec
  action is admitted. Once an LLM call or exec starts, it is non-preemptive
  through all of that action's segments. Queue waiting shifts later task
  segments and holds the task's latest causally observed RSS.
- A newly admitted task has no completed resource sample, so its held RSS is
  zero until its first segment completes. Any resulting initial exposure is
  counted as a capacity violation; the evaluator may not inspect that first
  segment to prevent it.
- Actual capacity checks use the shifted recorded CPU/RSS values. Fixed service
  time is valid only when an arm has zero actual capacity-violation seconds; any
  violating arm is unsafe rather than credited with free service.
- The model exposes only the current action identity, past completed tasks,
  and resource samples from completed segments. Future action identities, phase
  lengths, resource values, labels, and task completion are scorer-only.
- Capacities remain four LLM request slots, 43 CPU cores, and 80,000 MiB RSS.
  These are the preceding development model's proxies, not the 14-core remote
  evaluator host.

## Frozen arms

1. `fixed4`: four active tasks, four LLM slots, no tool-resource admission.
2. `fixed8`: eight active tasks, four LLM slots, no tool-resource admission.
3. `serial_tool`: eight active tasks and at most one running exec. This is the
   prediction-free safe-side control; LLM calls may overlap tools.
4. `task_aware_static`: eight active tasks. A newly ready exec reserves the
   hard Task-Aware CPU/RSS bucket upper bound for its full execution. CPU pages
   are 2, 4, and 43 cores; RSS pages are 500, 2,000, and 80,000 MiB. Missing
   predictions use the highest page.
5. `task_aware_feedback`: identical initial reservations. After each completed
   sampled segment, a surviving exec's next-segment CPU reservation becomes its
   latest observed CPU rate and its task RSS reservation becomes the latest
   observed RSS. Updates cannot use the current bin before it completes.
6. `joint_oracle_cap8`: the preceding exact whole-task-profile fit at task
   admission, restricted to the replay tasks and the same active-task cap. It
   measures remaining action headroom and cannot select a deployable rule.

Every arm uses identical tasks, task order, local profiles, and LLM limit.
Task-Aware is generated once from the frozen fit/replay order with whole-task
settlement and zero prediction-time LM calls. No parser, predictor, threshold,
bucket, or command-family rule may be changed in this phase.

## Metrics and gate

Report mean task completion, makespan, tool and LLM queue time, starts before
first completion, maximum active tasks, utilization and peaks for all three
resources, actual capacity-violation seconds, prediction availability, and
candidate reservation underprediction seconds.

`task_aware_feedback` advances only if all are true:

1. both `serial_tool` and `task_aware_feedback` complete with zero LLM, CPU,
   and RSS actual capacity-violation seconds;
2. mean task completion at least 5% below `serial_tool`;
3. makespan no higher than `serial_tool`;
4. at least two exec phases overlap in at least 20 replay tasks, so any gain is
   not a one-task artifact.

Failure must be attributed to prediction availability, initial
underprediction, delayed feedback, persistent task RSS, or insufficient safe
overlap. Do not repair this protocol after reading outcomes. A GO authorizes a
separate dynamic task-admission phase; a NO-GO stops that expansion but retains
the predictor, feedback mechanism, and hindsight ceiling.

## Cost and boundary

Implementation and smoke run locally; the unchanged formal evaluator is rerun
on `Ubuntu@216.81.248.69` and must reproduce the result byte-for-byte. No GPU,
new trace collection, LM call, runtime integration, service, or scheduler
framework is part of this phase.
