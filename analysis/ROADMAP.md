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

## Next scheduler: event-driven multi-resource backfill

Use two workload scales. The 12-task balanced PennyLane manifest has two unique
tasks for each of six workload types and is the smallest physical candidate.
The development simulator uses all 70 ordinary PennyLane tasks, without
duplicating traces. Each type has 18 top-quartile tasks in that corpus; labels
overlap and are used only to build workloads and report per-type effects. A
runtime scheduler may not read a type label derived from a completed trace.

At each LLM/tool boundary or completed feedback sample, maintain one causal
estimate per ready phase:

`LLM service and KV reuse; CPU work; RSS; Disk bytes/rate; return-time interval`

The first implementation is an event loop, not an optimizer or MIP. It scans
the ready phases once at each event; 70 tasks are too small to justify a solver.

1. Enforce measured GPU-request, CPU, RSS, and Disk capacity. RSS is a hard
   placement/admission constraint; CPU and Disk sharing must charge any service
   slowdown.
2. Protect the earliest plausible tool return. Admit a non-preemptive LLM
   borrower only when its conservative finish precedes that return window.
3. Backfill a ready phase only when it cannot delay the oldest blocked phase.
   Among safe candidates, choose the one with the lowest resulting maximum
   utilization across the four resources; ties remain FCFS.
4. On multiple replicas, retain state locality when lost-prefix or KV-movement
   cost exceeds avoided queueing; otherwise route to the less-loaded replica.
5. Update remaining work only from completed clauses and causal eBPF/cgroup
   samples. An unsupported prediction disables speculative backfill for that
   phase and falls back to ordinary execution; it never reserves the whole host
   while other agents retain memory.

| Workload pressure | Signal | Efficient action |
|---|---|---|
| LLM-heavy | prompt, cached and output tokens; resident KV | limit overlapping long prefills; prefer state-local or less-loaded GPU |
| CPU-heavy | CPU work and observed throttling | EAR keeps work-conserving shares; scheduler overlaps complementary phases |
| Memory-heavy | held RSS and conservative incremental RSS | hard admission/placement fit; no memory overcommit or process snapshot |
| Disk-heavy | bytes, rate and measured device slowdown | avoid simultaneous high-I/O phases on one device; spread across workers |
| Long tool gap | elapsed tool time and remaining-time interval | lend the idle GPU opportunity to a waiting agent |
| Large LLM return | return interval, prompt size and KV location | stop unsafe long borrowing, retain useful KV, and prioritize the return |

The simulator must replace source API LLM seconds with an A100-calibrated
service model and measure Disk capacity before making either resource a claim.
Its primary baselines are fixed concurrency, reactive phase-only scheduling,
and Agentix-style scheduling after an LLM request arrives. The candidate adds
pre-arrival return protection and joint resource backfill; an exact-future arm
remains an upper bound. Only a non-dominated session-JCT/TTFT result on the full
70-task queue authorizes the 12-task physical replay.

Calibrate A100 service before reading scheduler outcomes. Task 963 supplies 70
sequential requests; every fifth call is validation and the rest fit a
non-negative linear model from source-visible uncached prompt, cached prompt,
and output tokens. TTFT excludes output tokens. Continue only if validation
median absolute percentage error is at most 25% and p90 is at most 50% for both
latency and TTFT. Then run one bounded four-request concurrency check from four
different tasks: 963, 5986, 3483, and 6358, selecting in each the source call
whose prompt is closest to 50,000 tokens and forcing 128 output tokens. Compare
fresh-server sequential and simultaneous cells. The simulator limits active LLM
requests to four and linearly interpolates from one to the matched median
four-request latency and TTFT slowdown; it is fixed from this calibration, not
chosen from scheduler results.

**Calibration amendment (2026-08-19).** The first task-963 extraction used the
replay output's cached-token field, which was zero for all calls. Joining the
original provider cache field still failed TTFT validation (48.83% median,
118.56% p90). A diagnostic reconstructed the exact model-token common prefix;
all 70 prompt hashes matched the physical requests, and 16-token block-aligned
prefix state reduced TTFT error to 24.56% median and 41.73% p90. This exposed
result cannot repair the original gate. Freeze that exact prefix-state feature
now as the longest common prefix with any known-resident earlier request,
rounded down to vLLM's 16-token cache block. Fit task 963 once, and require the
unchanged model to pass the same latency and TTFT limits on a fresh sequential
task-1320 run before the concurrency check or scheduler screen. This state is
valid only while the prefix is known resident; eviction or migration must
invalidate it rather than assume a hit.

The fresh task-1320 transfer is **NO-GO** for this point service model. Latency
passed at 12.39% median and 29.49% p90 error. TTFT passed the median limit at
20.91% but missed the p90 limit at 51.22%. Sixty-nine of 86 TTFT predictions
were conservative overestimates; the largest underestimate was the first cold
request, which took 974 ms versus 169 ms predicted. Exact prefix state is
therefore necessary and useful, but not sufficient to support the frozen tail
claim. Do not run the concurrency probe or full-corpus scheduler from this
model. Any revisit must preregister cold-start state and a decision-aligned
one-sided interval on another fresh calibration task; task 1320 is now exposed.

**State-interval follow-up (frozen after the point-model NO-GO).** Use all
exposed task-963 and task-1320 calls to fit the same non-negative models with
one added `cold_start` indicator. For warm and cold calls separately, multiply
the point estimate by the empirical 90th-percentile `actual / predicted` ratio
(`higher` quantile); the cold factor is therefore the larger of the two exposed
first-call ratios. Validate once on fresh task 1325, selected before execution
as the cheapest unused task with at least 50 LLM calls and a 70k-token prompt.
Latency and TTFT must each cover at least 90% of warm calls, cover the cold call,
and have median upper-bound/actual ratio at most 1.5. Failure stops state-aware
service modeling; success alone authorizes the already frozen concurrency probe.

## Decision sequence

### F0 — Full-corpus causal simulation

Replay the 70 ordinary PennyLane trajectories as one backlog. First establish
the exact-future ceiling with contention-aware service, then replace future
state with causal estimates without changing the scheduler. Stop if the causal
candidate does not improve the JCT/TTFT Pareto frontier over reactive phase-only
scheduling or if its gain is confined to one workload type.

The exact-future screen compares fixed concurrency four, unprotected borrowing,
and return-guarded revocable borrowing. The guarded arm must reduce mean task
completion by at least 10%, lower makespan, keep modeled all-request p99 TTFT at
or below 1.05x fixed, and improve at least four of the six workload groups.
Failure stops this branch before KB or predictor integration.

This branch is currently stopped by the A100 TTFT transfer gate above; no
full-corpus scheduler outcome exists.

### F1/F2 — Tool-state parking and remote placement are closed

The zero-cost perfect-parking screen changed starts but worsened mean completion
2.664%: 27 tasks advanced by 113,402 task-seconds while 30 were delayed by
139,700. It removed only 3.178% of CPU-core-time and 1.316% of RSS-time; baseline
RSS utilization was 14.171%. Therefore do not measure restore costs or build the
remote snapshot-RPC branch for this workload. Reopening requires a new workload
with independently demonstrated CPU/RSS pressure.

### F3 — Physical pre-arrival coordination beyond Agentix

Agentix already prioritizes arrived LLM calls by program-level attained service
and routes long calls to the program's replica while sending short calls to the
least-loaded replica. Native priority or KV-local routing alone is therefore a
baseline, not our contribution.

The distinct hypothesis is earlier coordination: while a tool is still running,
its elapsed time, completed clauses, and causal resource state may identify a
return window before the next LLM request exists. That state could control
backfill admission, return priority, and KV retention or placement. Reopen this
branch only if the full-corpus simulation shows such advance notice changes
actions across workload types. Then use the 12-task balanced batch and compare
against Agentix-style request-arrival scheduling, charging all queueing,
repeated prefill, and state movement to JCT and TTFT. A run longer than 30
minutes still requires an estimate and explicit approval.

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
