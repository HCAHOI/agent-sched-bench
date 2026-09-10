# MILESTONE 2 — Multi-instance LLM serving (2× L40S)

Date: 2026-09-10. Status: exploration phase closed. This document freezes the
mixed56 two-instance results across four directions: task-aware scheduling
(Continuum, ThunderAgent), cache-aware routing (DualMap), cross-instance load
balancing, and prefill/decode disaggregation (PD, PPD). Language is English;
Milestone 1 is in Chinese and remains the single-instance record.

## 1. Conclusions

1. **Unmodified DualMap is the strongest system on this workload.** It beats
   Continuum on mean, P95 and max JCT while also having lower TPOT, higher
   throughput and higher cached-prompt share. It does this without modeling
   agent progress at all.
2. **ThunderAgent's serving efficiency and its long-task starvation are the
   same mechanism.** Its best-in-class TPOT (76.6 ms) comes from keeping
   large-context tasks out of the active set. Every fix that restored
   completion (60 s resume, pending-release accounting, consistent capacity
   decay) lost 37–44% of that TPOT and still finished behind Continuum on JCT.
3. **Adding accumulated-wait priority to DualMap made it worse on every
   metric.** Unrepaid credit lets previously delayed agents starve newly
   admitted ones, and the resulting dispatch order destroys prefix locality.
4. **Temporal load imbalance is not an exploitable opportunity at this scale.**
   Under mixed56 pressure both GPUs are busy over 96% of the time under every
   policy, including plain task-sticky FCFS. Per-request routing that chases
   load (least-requests) is strictly worse than locality-preserving sticky
   routing.
5. **Fixed PD is prefill-bound; public PPD collapses onto the decode worker.**
   PPD's 512-token rule prices appended tokens when the real cost is uncached
   history (median 24K tokens). The 448-group profiling matrix shows the
   local-vs-PD decision is conditional on cache residency and history length,
   not on appended-token count.

The comparison is a single physical run per policy on one workload; the
paired bootstrap over 28 source trajectories gives the uncertainty for the
headline JCT differences (§4). Two independent repeats of Continuum and
DualMap on different hosts differ by 2–3 min in mean JCT, which bounds
run-to-run spread for this workload.

## 2. Setup

- **Hardware:** Vast.ai instance 50152322, 2× NVIDIA L40S. One serving engine
  per GPU, tensor parallel 1. Instance is stopped, not deleted.
- **Model:** Qwen/Qwen3-4B-Instruct-2507-FP8, revision `8591804019c8…`.
- **Engine:** vLLM 0.10.2 with the public Continuum serving overlay for FCFS,
  Continuum, ThunderAgent and DualMap. PD/PPD used vLLM 0.28.0 with NIXL
  1.4.1 push transport (required by the PPD implementation).
- **Shared engine settings:** 8 sequences per instance, 2,048-token batch
  budget, max model length 131,072, GPU memory utilization 0.95, prefix
  caching on. DualMap additionally runs its CPU KV cache at 48 GiB per
  instance.
- **Workload mixed56** (`mixed56-2l40s-concurrency32-v1`): the 28 mixed28
  source trajectories (15 SQLGlot, 13 PennyLane) each replicated twice.
  56 original tasks, 2,470 LLM requests, 2,414 tool executions. All ready at
  t=0; global task concurrency 32. Same recorded prompts and token IDs for
  every method; output length forced to the source length; tool phases
  replayed at 4× recorded speed. Replacement tasks maintain pressure
  (exponential delay, mean 10 s, seed 42) until all originals terminate.
- **Metrics:** JCT is ready-to-terminal wall time including admission and all
  scheduler delays. Engine TPOT covers decode steps while a request is being
  served; time paused by a task scheduler is excluded. Cached-prompt share is
  the API-usage cached fraction over original-task input tokens.

| Method | Implementation |
|---|---|
| FCFS, task-sticky | Continuum-fork `serve-fcfs`; task bound to one instance for its lifetime, round-robin initial placement |
| Continuum | Public fork `316a5879…`, task-sticky routing, program-level FCFS with 2 s KV pinning |
| ThunderAgent | Public `7ddc8610…` proxy, router mode `tr`, 1,800 s forced-resume wait (original); patched variants in §5 |
| DualMap | Public `24816acc…`: shadow-cache routing, admission, cache-aware queueing, CPU KV cache via LMCache |
| PD / PPD | Public PPD implementation, one prefill worker and one decode worker, PCIe NIXL transfer |

## 3. Stage A: mixed28 on two instances was underloaded

The first two-instance runs replayed mixed28 with concurrency 16. Continuum
and sticky FCFS were indistinguishable, and per-request least-requests
routing was worse than sticky routing.

| mixed28, 2 GPUs | Mean JCT, min | P95 | Max |
|---|---:|---:|---:|
| FCFS, task-sticky | 14.53 | 33.74 | 43.86 |
| FCFS, least-requests | 17.44 | 42.98 | 50.42 |
| Continuum, task-sticky | 14.38 | 32.26 | 39.15 |

Two consequences: (a) locality-preserving placement matters more than
balancing dispatched requests, which §7 confirms at higher load; (b) the
workload had to double to expose scheduler differences. mixed56 with
concurrency 32 is the result.

## 4. Stage B: mixed56 main comparison

All completed-run numbers below are from the same host and engine build.
Original ThunderAgent and the FCFS/Continuum 2026-09-06 repeats ran on a
different host with CUPTI enabled; their JCT is comparable, their TPOT carries
a configuration offset.

| Method | Completed | Mean JCT, min | P95 | Max | Engine TPOT, ms | Cached prompt |
|---|---:|---:|---:|---:|---:|---:|
| FCFS, task-sticky | 56/56 | 55.27 | 111.08 | 139.44 | — | — |
| Continuum | 56/56 | 35.27 | 78.47 | 80.75 | 112.70 | 70.1% |
| ThunderAgent, original | 55/56 | ≈9.4 h run, see §5 | — | — | 76.6 | — |
| ThunderAgent, fix 1 | 56/56 | 44.90 | 104.60 | 118.38 | 104.83 | 39.1% |
| ThunderAgent, fix 2 | 56/56 | 46.13 | 103.49 | 113.86 | 110.58 | 34.7% |
| **DualMap, original** | **56/56** | **32.85** | **71.08** | **74.68** | **85.89** | **76.0%** |
| DualMap + agent-progress | 56/56 | 38.69 | 78.12 | 91.03 | 93.79 | 62.2% |
| Fixed PD | 56/56 | 71.47 | 143.51 | 177.06 | ≈42 | — |
| PPD, original | 56/56 | 109.24 | 211.63 | 265.43 | ≈126 | ≈1.7% |

Throughput over the common 74.68-minute window (original plus background):

| Method | LLM steps/min | Output tokens/s |
|---|---:|---:|
| Continuum | 45.73 | 137.33 |
| ThunderAgent, fix 2 | 37.78 | 113.75 |
| **DualMap, original** | **49.60** | **149.98** |
| DualMap + agent-progress | 41.18 | 125.16 |

Paired mean-JCT differences over 28 source trajectories (2,000 bootstrap
draws, seed 42):

| Comparison | Δ mean JCT, s | 95% interval |
|---|---:|---|
| DualMap + agent-progress vs DualMap | +350 | [222, 476] |
| DualMap + agent-progress vs Continuum | +205 | [60, 350] |

Worst tasks are the same under every policy: both replicas of PennyLane-4161,
-5857 and -6049. The policies differ in how long those tasks are held, not in
which tasks are hard.

## 5. ThunderAgent: starvation is the price of its TPOT

**Original.** With the default 1,800 s forced-resume threshold the run lasted
about 9 h 26 min and finished 55/56 tasks and 2,468/2,470 requests. One task
accumulated ≈493 min paused against 27 min decoding. Forced restoration did
not give sustained service: the task returned, was marked for suspension
again, and lost its slot. Matched-step TPOT was 76.6 ms, the best of any
policy, with a historical throughput window of 58.9 steps/min and 176 output
tokens/s, above Continuum.

**Diagnostic: 60 s threshold.** Lowering the wait to 60 s did not help; the
run was stopped at 140.66 min with 54/56 complete. Tasks repeatedly hit the
threshold and were re-suspended. TPOT rose to 110.4 ms.

**Fix 1: pending-release accounting.** The pause decision ignored memory
already scheduled for release by tasks marked for suspension, so it
over-suspended. Crediting pending releases (with the 60 s threshold)
completed all tasks: mean JCT 44.90 min, max 118.38 min, TPOT 104.8 ms
(+36.5% vs original). Pauses fell; engine queueing and prefill rose.

**Fix 2: consistent capacity accounting.** Extending the tool-phase capacity
decay from restoration to admission and suspension completed all tasks
without improving the tradeoff: mean JCT 46.13 min, TPOT 110.6 ms (+44.1%).
Relative to fix 1, scheduler wait fell 8.18 → 7.46 min per task while engine
queueing rose 6.33 → 7.41 min, cached share fell 39.1% → 34.7%, and engine
preemptions rose 263 → 305.

Every step that gave long tasks more service pushed more large-context work
into the engine, which raised queueing, lowered cache hit rate and slowed
decoding for everyone. ThunderAgent's advantage on this workload is not
separable from its exclusion of expensive tasks.

## 6. DualMap: wins by locality, loses when given memory of the past

**Original.** DualMap's routing plus admission plus CPU KV cache gave the
lowest JCT at every quantile, the lowest TPOT of any completed vLLM 0.10.2 run,
the highest throughput and 76% cached-prompt share. It has request-level
queueing tails (worst router wait 519 s) but no task-level starvation.

**Agent-progress modification.** Task-level cumulative waiting was added to
queue priority, keeping public routing and admission. Every metric degraded
(§4). The trace shows why: wait credit persisted until task completion and
was never repaid, so previously delayed agents kept precedence over new
arrivals. The 519 s first-request wait dropped to 2.55 s, and a different
task's first request then waited 454 s while 182 later requests were
dispatched. The change moved the unfairness and, by reordering dispatch away
from cache-resident requests, cut the cached share from 76% to 62% and
throughput by 17%.

Any future priority term in this family must be bounded and repaid, and must
not outrank cache residency at dispatch.

## 7. Load balancing: both GPUs are already busy

Synchronized ~200 ms GPU polling with ~98% paired coverage. "Busy" is
utilization ≥50%, "idle" is ≤10%.

| Method | Mean util, GPU 0 / 1 | Both busy | One busy, other idle | Longest such interval |
|---|---:|---:|---:|---:|
| FCFS, task-sticky | 95.4% / 95.9% | 99.05% | 0.056% | 0.20 s |
| Continuum | 93.1% / 92.0% | 98.24% | 0.174% | 2.82 s |
| DualMap | 88.6% / 88.2% | 96.11% | 0.265% | 0.41 s |

Instantaneous differences exist (DualMap has >20-point utilization gaps
18.9% of the time) but never persist beyond 2.4 s, and the total busy/idle
time across a DualMap run is 11.6 s. Engine telemetry agrees: FCFS keeps
about seven running and seven waiting requests on each engine; Continuum
also keeps both queues populated. Tool phases do not empty a GPU's queue
because other agents fill it.

A scheduler whose premise is reclaiming capacity from imbalance has nothing
to reclaim here. DualMap wins with the lowest utilization of the three, so
utilization is not the objective either. Lighter load or larger clusters
could change this; it is not established by these measurements.

## 8. PD / PPD: the routing rule prices the wrong quantity

| Measurement | Fixed PD | Original PPD |
|---|---:|---:|
| Completed original tasks | 56/56 | 56/56 |
| Mean task JCT | 71.47 min | 109.24 min |
| Replay wall time | 177.76 min | 266.11 min |
| Engine TPOT | ≈42 ms | ≈126 ms |
| Mean D-engine queueing per request | 3.75 s | 82.88 s |
| P-worker GPU utilization | ≈97.9% | ≈0.04% |
| D-worker GPU utilization | ≈62.8% | ≈98.9% |
| Requests through the PD path | 2,470 | 56 |
| Requests prefilled locally on D | 0 | 2,414 |

**Fixed PD** is prefill-bound: P runs flat out while D sits at 63%. Its TPOT
is the best measured because D only decodes, but JCT is twice Continuum's.

**PPD** routes all 2,414 subsequent-turn requests to local prefill on D and
leaves P idle. D then does all prompt processing and decoding, queueing rises
20×, and TPOT triples. The 512-token bypass is a threshold on newly appended
input, but subsequent-turn histories have a median of ~24K tokens (max 107K).
The 962 requests that bypassed lookup averaged 248 appended tokens against
27.8K uncached tokens and 1.23% cached share. Lowering the threshold to 128
still routes every request locally; the calibrated lookup also chooses D.

**Profiling matrix (complete).** 56 length points × PD/local × 2 arrival
rates × 2 seeds = 448 groups, 7,168 requests, 224 paired comparisons, 11.34 h
of accepted execution. History 833–107K tokens, output 30–1,536 tokens.

| Local minus PD, wins for local | TTFT | Decode TPOT | E2E |
|---|---:|---:|---:|
| All 224 pairs | 224 | 74 | 210 |
| History <8K (44) | 44 | 23 | 39 |
| History 8–32K (80) | 80 | 32 | 71 |
| History 32–64K (36) | 36 | 7 | 36 |
| History ≥64K (64) | 64 | 12 | 64 |

Local prefill always wins TTFT and almost always wins E2E in isolation, but
its decode-TPOT win rate falls from 52% below 8K history to 19% above 64K:
long local prefills interfere with concurrent decoding on D. Crucially, the
profiling ran at 90.6% cached-prompt share, versus 1.7% in the online PPD
run. Local execution is cheap when history is resident and expensive when it
is not, and residency depends on concurrent load and interleaving rather than
on conversation continuity. A ~76K-history, 1,536-output case showed full,
partial and zero hits within one group.

The right routing signal is expected uncached prefill work under current D
load, not appended-token count. Integrating the measurements into PPD's
decision tables and validating on mixed56 has not been done.

## 9. What is settled and what the next stage must answer

Settled by this milestone:

- Task-aware scheduling that keeps expensive tasks out of the active set
  buys TPOT with starvation; the two cannot be decoupled by threshold or
  accounting changes to ThunderAgent.
- Cache-aware routing with a CPU KV tier (DualMap) is the baseline to beat.
  Its remaining tail is request-level queueing, not task starvation.
- Load imbalance is not the lever at 2 GPUs under this pressure.
- Fixed PD and public PPD both lose to single-role instances on JCT; PPD's
  failure is a cost-model error, not a tuning problem.

Open for the next stage:

1. A DualMap-side mechanism must improve mean and tail JCT without lowering
   cached share, TPOT or throughput. Bounded, repaid wait credit that never
   outranks cache residency is the only candidate the evidence supports.
2. A PPD policy that routes on expected uncached prefill and D-side load,
   derived from the profiling matrix, validated on mixed56 against DualMap
   and Continuum on the same engine build.
3. Any new method reports the same table as §4: completion, JCT quantiles,
   engine TPOT, cached share, common-window throughput, paired bootstrap.

## 10. Evidence and code

Result directories live in the two worktrees next to this repository.

- Four-way comparison and DualMap modification:
  `../../agent-sched-bench-thunderagent-fairness/results/mixed56-vast-dualmap-agent-progress-20260909-r1/` (`comparison.json`, `protocol.json`)
- ThunderAgent 60 s, fix 1, fix 2:
  `../../agent-sched-bench-thunderagent-fairness/results/mixed56-vast-thunderagent-wait60-{,pending-release-,capacity-consistent-}20260909-r1/`
- Original ThunderAgent: `../../agent-sched-bench-multi-instance/results/mixed56-2l40s-thunderagent-20260906-r2.tar.gz`
- FCFS / Continuum / DualMap GPU timelines:
  `../../agent-sched-bench-multi-instance/results/mixed56-vast-{fcfs-sticky,continuum,dualmap}-gpu-timeline-20260909-r1/gpu-balance-summary.json`
- mixed28 two-instance stage: `../../agent-sched-bench-multi-instance/results/mixed28-2l40s-*-20260906-r*/`
- Fixed PD and PPD: `../../agent-sched-bench-multi-instance/results/mixed56-vast-pd-pcie-20260907-r3/`, `mixed56-vast-ppd-pcie-20260907-r1/`
- Profiling matrix: `../../agent-sched-bench-multi-instance/results/serving-length-profile-vast-20260908-complete/` (`README.md`, `paired.csv`, `requests.csv`, `summary.json`)
- Code, branch `codex/thunderagent-fairness` commit `790b00c` and branch `codex/multi-instance-fcfs` commits `0ac2f04`, `dc7eee7`:
  `scripts/baselines/thunderagent_official_launcher.py` with `thunderagent_pending_release.patch` and `thunderagent_capacity_consistent.patch`;
  `scripts/baselines/dualmap_official_proxy.py` (`--agent-progress`);
  `scripts/baselines/ppd_official_proxy.py`, `ppd_policy.py` and the four PPD patches;
  `scripts/evaluation/profile_serving_lengths.py`, `profile_ppd.py`;
  workload manifest `analysis/development/mixed56-2l40s-concurrency32-v1/`.
