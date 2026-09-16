# MILESTONE 2 — Multi-instance LLM serving (2× L40S)

Date: 2026-09-10. Status: exploration phase closed. This document freezes the
mixed56 two-instance results across four directions: task-aware scheduling
(Continuum, ThunderAgent), cache-aware routing (DualMap), cross-instance load
balancing, and prefill/decode disaggregation (PD, PPD). Language is English;
Milestone 1 is in Chinese and remains the single-instance record.

## 1. Conclusions

1. **DualMap beats Continuum on the same host, but only its tail advantage
   survives pooling across hosts.** On the matched 2026-09-09 pair DualMap
   wins mean, P95 and max JCT with lower TPOT, higher throughput and higher
   cached-prompt share, and it does this without modeling agent progress at
   all. The earlier host points the same way. Pooled over every run in this
   repository, however, the mean-JCT ranges overlap: Continuum's fastest run
   (32.76 min) is faster than DualMap's slowest (33.24 min), because the
   host-to-host spread within each policy is as large as the gap between
   them. Max JCT is the one quantile that separates the two in every pairing;
   the P95 ranges touch (§4).
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
5. **Both disaggregated runs measured a no-residency regime, not the designs
   they were meant to test.** Fixed PD is prefill-bound, and public PPD sends
   almost every subsequent-turn request to local prefill on the decode
   worker; those are the measurements and they stand. Neither run had its
   design's premise. PPD serves turn 2 and later from the decode node's own
   prefix cache, and our decode node reached 1.8% cached share on
   original-task steps; fixed PD needs the history resident on the prefill
   worker, and ours took no cache hits across 101.7M prefilled prompt tokens.
   The inferences those runs were used to draw about the two designs are
   withdrawn (2026-09-15, §8).

Run counts differ by policy on this one workload: DualMap has three physical
runs, FCFS and Continuum two each, everything else one (§4). The paired
bootstrap over 28 source trajectories gives the uncertainty of the difference
*within* a pair of runs, and its own limitation field records that it does
not estimate host-repeat variation. So the DualMap repeat pair (mean JCT
33.24 vs 32.85 min, paired +23 s with interval [-68, 135], cached share
75.7% vs 76.0%) does not bound run-to-run spread, and the measured spreads
are several times wider than it: DualMap 29.63–33.24 min over three runs,
Continuum 32.76–35.27 over two. A separability threshold for mean JCT has to
be read off those ranges, not off one repeat pair; on this evidence a
candidate that moves mean JCT by a few minutes is not yet separated from a
change of host.

## 2. Setup

- **Hardware:** 2× NVIDIA L40S on Vast.ai, one serving engine per GPU,
  tensor parallel 1. Three hosts in sequence: the 2026-09-06 FCFS, Continuum
  and ThunderAgent runs and the 2026-09-07 DualMap run are the first host
  (the two archived tarballs carry matching GPU UUIDs in `hardware.txt`; the
  2026-09-06 directories keep no `hardware.txt` and are placed there by their
  shared launch path and CUPTI declaration); the 2026-09-09 comparisons ran
  on instance 50152322 (gone); the 2026-09-10 DualMap repeat and all later
  runs use container C.50481401 (`millstone/NEXT-SESSION.md` has the
  endpoint).
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

The unmarked vLLM 0.10.2 rows are from the 2026-09-09 host and engine build;
the PD and PPD rows are the 2026-09-07 runs on vLLM 0.28.0 (§8). The marked
rows name their own host: the 2026-09-10 DualMap repeat uses the 2026-09-09
engine build on the third host, and the four first-host rows are the
2026-09-06 FCFS, Continuum and ThunderAgent runs and the 2026-09-07 DualMap
run, with CUPTI sampling declared for the 2026-09-06 pair. Their JCT is
comparable; their TPOT carries a configuration offset, which is why the three
added rows leave TPOT and cached share out. ThunderAgent's 76.6 ms is kept
because §5 turns on it and reads it against that offset.

| Method | Completed | Mean JCT, min | P95 | Max | Engine TPOT, ms | Cached prompt |
|---|---:|---:|---:|---:|---:|---:|
| FCFS, task-sticky | 56/56 | 55.27 | 111.08 | 139.44 | — | — |
| FCFS, task-sticky (2026-09-06, first host) | 56/56 | 57.27 | 121.43 | 163.91 | — | — |
| Continuum | 56/56 | 35.27 | 78.47 | 80.75 | 112.70 | 70.1% |
| Continuum (2026-09-06, first host) | 56/56 | 32.76 | 70.10 | 82.73 | — | — |
| ThunderAgent, original (2026-09-06, first host) | 55/56 | ≈9.4 h run, see §5 | — | — | 76.6 | — |
| ThunderAgent, fix 1 | 56/56 | 44.90 | 104.60 | 118.38 | 104.83 | 39.1% |
| ThunderAgent, fix 2 | 56/56 | 46.13 | 103.49 | 113.86 | 110.58 | 34.7% |
| **DualMap, original** | **56/56** | **32.85** | **71.08** | **74.68** | **85.89** | **76.0%** |
| DualMap, original (2026-09-07, first host) | 56/56 | 29.63 | 62.06 | 68.95 | — | — |
| DualMap, original, repeat (2026-09-10 host) | 56/56 | 33.24 | 70.59 | 75.31 | 87.53 | 75.7% |
| DualMap + agent-progress | 56/56 | 38.69 | 78.12 | 91.03 | 93.79 | 62.2% |
| Fixed PD | 56/56 | 71.47 | 143.51 | 177.06 | ≈42 | — |
| PPD, original | 56/56 | 109.24 | 211.63 | 265.43 | ≈126 | ≈1.8% |

Reading the repeated policies across hosts: DualMap wins mean JCT against
Continuum on both hosts where both ran, by 3.13 min on the first host
(29.63 vs 32.76, a day apart, not a matched pair) and 2.42 min on the
2026-09-09 host (32.85 vs 35.27). The direction is consistent; the margin is
not larger than each policy's own spread across hosts, so the pooled ranges
overlap on mean (DualMap 29.63–33.24, Continuum 32.76–35.27) and touch on
P95 (62.06–71.08 against 70.10–78.47). Max JCT is the exception: DualMap's
68.95–75.31 does not reach Continuum's 80.75–82.73 in any pairing, so
DualMap's tail advantage survives pooling while its mean advantage does not.
FCFS spans 55.27–57.27 over its two runs, far behind both under every
pairing.

Throughput over the common 74.68-minute window (original plus background):

| Method | LLM steps/min | Output tokens/s |
|---|---:|---:|
| Continuum | 45.73 | 137.33 |
| ThunderAgent, fix 2 | 37.78 | 113.75 |
| **DualMap, original** | **49.60** | **149.98** |
| DualMap, original, repeat (2026-09-10 host) | 47.97 | 144.59 |
| DualMap + agent-progress | 41.18 | 125.16 |

Paired mean-JCT differences over 28 source trajectories (2,000 bootstrap
draws, seed 42):

| Comparison | Δ mean JCT, s | 95% interval |
|---|---:|---|
| DualMap + agent-progress vs DualMap | +350 | [222, 476] |
| DualMap + agent-progress vs Continuum | +205 | [60, 350] |
| DualMap repeat (2026-09-10 host) vs DualMap | +23 | [-68, 135] |

Worst tasks are the same under every policy: both replicas of PennyLane-4161
and -5857 always take the top four places, with -6049 and -5831 filling the
next two. The policies differ in how long those tasks are held, not in which
tasks are hard.

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

**Original.** On the 2026-09-09 host DualMap's routing plus admission plus
CPU KV cache gave the lowest JCT at every quantile, the lowest TPOT of any
completed vLLM 0.10.2 run, the highest throughput and 76% cached-prompt
share. Across hosts only the max-JCT lead holds without qualification (§4).
It has request-level queueing tails (worst router wait 519 s) but no
task-level starvation.

**Agent-progress modification.** Task-level cumulative waiting was added to
queue priority, keeping public routing and admission. Every metric degraded
(§4). The trace shows why: wait credit persisted until task completion and
was never repaid, so previously delayed agents kept precedence over new
arrivals. The 519 s first-request wait dropped to 2.55 s, and a different
task's first request then waited 454 s while 182 later requests were
dispatched. The change moved the unfairness and, by reordering dispatch away
from cache-resident requests, cut the cached share from 76% to 62% and
throughput by 17%.

Any future priority term in this family must be bounded and repaid. The
repeat run's router log shows where DualMap's remaining tail sits: of 255
router waits over 60 s, 238 belong to requests with a small but nonzero cache
estimate (under 4K tokens, worth under 0.2 s of prefill) that the public
residency-first heap holds behind high-residency requests; only 14 have zero
estimate and long waits rarely compete with other requests of equal estimate.
A credit that only reorders equal-residency requests therefore cannot reach
this tail; a bounded credit has to be priced in the same seconds as the
cache term.

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

## 8. PD / PPD: both runs paid full prefill on every turn

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
P took no cache hits: over the whole run it prefilled 101.7M prompt tokens
across 4,022 requests at a mean prompt of 25,291 tokens, which is the full
history every turn.

**PPD** routes all 2,414 subsequent-turn requests to local prefill on D and
leaves P idle. D then does all prompt processing and decoding, queueing rises
20×, and TPOT triples. Its 512-token bypass is a threshold on newly appended
input, and subsequent-turn histories have a median of ~24K tokens (max 107K),
so every request clears it: the 962 requests that bypassed lookup averaged
248 appended tokens against 27.8K uncached tokens and 1.23% cached share.
Lowering the threshold to 128 still routes every request locally; the
calibrated lookup also chooses D.

**Withdrawn on 2026-09-15: the reading of these two runs as evidence about
the designs' routing rules**, after that reading had been carried into the
Frontier C closure of 2026-09-11. Upstream PPD serves turn 2 and later from
the decode-capable node's own prefix cache, so routing later turns to local
prefill is its mechanism, not a degeneration of it. Our decode node held 1.8%
of original-task prompt tokens in cache (1.4% over all replayed steps) and
re-prefilled a mean of 25,320 tokens across 3,985 requests, while P saw 114
requests at a mean prompt of 1,194 tokens. The engine holds 273,952 KV tokens
per GPU, and 32 concurrent contexts at that mean need about 2.9× that, so no
routing rule could have kept them resident. Fixed PD had the mirror problem
on the other side. What the two runs put on the host is therefore two ways of
paying full prefill on every turn. The measurements stand as measurements of
that regime, and so does the system-level verdict that both configurations
lose to single-role instances by a wide margin at 2 GPUs on this workload;
what does not stand is any inference from them about how either published
design prices its routing decisions.
`millstone/MILESTONE-3-Frontiers.md` §3 carries the full analysis, the
upstream reading of PPD's four roles, and the two-sided router that followed.

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

The 224 pairs are a valid measurement, but only of the condition they ran
in. **Withdrawn on 2026-09-15: the reading that local prefill wins TTFT and
E2E in isolation.** The prefill worker was saturated throughout this matrix,
so every PD pair paid P's queue; the 2026-09-11 controlled-load profiling
repeated three of these history points with an idle P and burst sizes 8, 16
and 32, and PD won end-to-end in 7 of the 9 cells, losing only where the
history was already resident on D (`millstone/MILESTONE-3-Frontiers.md` §3,
`results/ppd-load-profile-20260911-r3/`). What this matrix does establish is
the decode-side cost: local's decode-TPOT win rate falls from 52% below 8K
history to 19% above 64K, because long local prefills interfere with
concurrent decoding on D. It also ran at 90.6% cached-prompt share against
1.8% in the online PPD run, so local execution here was cheap for a reason
the online run did not have. Residency depends on concurrent load and
interleaving rather than on conversation continuity: a ~76K-history,
1,536-output case showed full, partial and zero hits within one group.

The right routing signal is residency on D weighed against pending prefill
work on each side, not appended-token count. A two-sided router built on
these cost constants was run on mixed56 on 2026-09-11; it did not beat fixed
PD, and Milestone 3 §3 closes the frontier on that result.

## 9. What is settled and what the next stage must answer

Settled by this milestone:

- Task-aware scheduling that keeps expensive tasks out of the active set
  buys TPOT with starvation; the two cannot be decoupled by threshold or
  accounting changes to ThunderAgent.
- Cache-aware routing with a CPU KV tier (DualMap) is the baseline to beat.
  Its remaining tail is request-level queueing, not task starvation. Its
  max-JCT lead over Continuum holds across every host; its mean-JCT lead does
  not survive pooling the runs on three hosts (§4).
- Load imbalance is not the lever at 2 GPUs under this pressure.
- Both disaggregated runs lose to single-role instances on JCT by a wide
  margin, and both did so while paying full prefill on every turn. That is a
  result about running each configuration with no history resident on the
  side that had to reuse it, not about the cost model of either published
  design; §8 records what was withdrawn.

Open for the next stage:

1. A DualMap-side mechanism must improve mean and tail JCT without lowering
   cached share, TPOT or throughput. The bounded, repaid credit (cap 5 s)
   was run on 2026-09-10 and lost on every metric for the same reason
   ThunderAgent starves large tasks; Milestone 3 §1 records the result and
   closes this frontier.
2. A policy that routes on expected uncached prefill and each side's load,
   derived from the profiling matrix, was run on mixed56 on 2026-09-11 and
   lost to fixed PD; Milestone 3 §3 records the result and closes this
   frontier. Reopening it needs measured residency on the side that reuses
   the history, which no run in this milestone had.
3. Any new method reports the same table as §4: completion, JCT quantiles,
   engine TPOT, cached share, common-window throughput, paired bootstrap.

## 10. Evidence and code

Result directories live under `results/` in this repository; provenance branches `codex/thunderagent-fairness` and `codex/multi-instance-fcfs` are archived on GitHub.

- Four-way comparison and DualMap modification:
  `../results/mixed56-vast-dualmap-agent-progress-20260909-r1/` (`comparison.json`, `protocol.json`)
- DualMap repeat on the 2026-09-10 host: `../results/mixed56-vast-dualmap-20260910-r1/`
  (`comparison.json` against the 2026-09-09 DualMap and Continuum runs, produced by
  `scripts/evaluation/compare_two_instance_runs.py`, which reproduces the checklist
  values of the 2026-09-09 `comparison.json` from run directories alone)
- The three first-host rows added to §4, tabulated from each run's own
  `output/throughput_summary.json` (`ready_to_terminal_s` over the 56
  original tasks, the quantity §4 tabulates everywhere):
  `../results/mixed56-2l40s-fcfs-task-sticky-20260906-r1/`,
  `../results/mixed56-2l40s-continuum-task-sticky-20260906-r1/`,
  `../results/mixed56-2l40s-dualmap-20260907-r1.tar.gz` (summary inside the
  tarball). The two 2026-09-06 directories declare 1 s CUPTI sampling in
  `protocol.json`; the DualMap tarball's `hardware.txt` carries the GPU UUIDs
  shared with the archived ThunderAgent run.
- ThunderAgent 60 s, fix 1, fix 2:
  `../results/mixed56-vast-thunderagent-wait60-{,pending-release-,capacity-consistent-}20260909-r1/`
- Original ThunderAgent, same first host: `../results/mixed56-2l40s-thunderagent-20260906-r2.tar.gz`
- FCFS / Continuum / DualMap GPU timelines:
  `../results/mixed56-vast-{fcfs-sticky,continuum,dualmap}-gpu-timeline-20260909-r1/gpu-balance-summary.json`
- mixed28 two-instance stage: `../results/mixed28-2l40s-*-20260906-r*/`
- Fixed PD and PPD: `../results/mixed56-vast-pd-pcie-20260907-r3/`, `mixed56-vast-ppd-pcie-20260907-r1/`
  (cached share and per-worker prompt volumes from `output/*/attempt_*/openclaw_host_replay.jsonl`
  and `server/instance-*/vllm-request-telemetry.jsonl`; the KV capacity is in
  `server/instance-*/vllm.log`). The withdrawal in §8 follows
  `millstone/MILESTONE-3-Frontiers.md` §3.
- Profiling matrix: `../results/serving-length-profile-vast-20260908-complete/` (`README.md`, `paired.csv`, `requests.csv`, `summary.json`),
  superseded on its E2E conclusion by the controlled-load profiling in
  `../results/ppd-load-profile-20260911-r3/` (`load-profile-summary.txt`)
- Code, ported onto the main lineage in commit `f62a7ab` (originally `790b00c`, `0ac2f04`, `dc7eee7` on the archived branches):
  `scripts/baselines/thunderagent_official_launcher.py` with `thunderagent_pending_release.patch` and `thunderagent_capacity_consistent.patch`;
  `scripts/baselines/dualmap_official_proxy.py` (`--agent-progress`);
  `scripts/baselines/ppd_official_proxy.py`, `ppd_policy.py` and the four PPD patches;
  `scripts/evaluation/profile_serving_lengths.py`, `profile_ppd.py`
  (the 2026-09-16 cleanup kept only the two paradigms: `ppd_policy.py`, `profile_serving_lengths.py`
  and three of the patches are gone, `ppd_official_proxy.py` and `profile_ppd.py` remain — M3 §3);
  workload manifest `analysis/development/mixed56-2l40s-concurrency32-v1/`.
