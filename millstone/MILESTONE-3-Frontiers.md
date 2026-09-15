# MILESTONE 3 — Three frontiers on 2× L40S

Date: 2026-09-10. Status: closed 2026-09-11; all three frontiers closed on
2× L40S, and Milestone 4 reinterprets them as one operating point (KV
demand above capacity). Milestone 2 froze the exploration
(unmodified DualMap is the system to beat; load imbalance is not a lever on
mixed56; PD and PPD lose on JCT). This document tracks the three directions
that remain open, one section each. Each section is rewritten when its
frontier moves; it states what is settled, what was just measured, and the
single next run. Metrics, workload and the reporting checklist are those of
Milestone 2 §2 and Milestone 1 §3.

## 0. Common ground

- **Host.** All runs from 2026-09-10 on use Vast container C.50481401
  (2× L40S, vLLM 0.10.2 build for DualMap/Continuum/FCFS). The 2026-09-09
  host is gone. Cross-host DualMap repeat: mean JCT 33.24 vs 32.85 min,
  paired +23 s [-68, 135], cached share 75.7% vs 76.0%, TPOT 87.5 vs 85.9 ms.
  A candidate on this host is compared with the repeat
  (`results/mixed56-vast-dualmap-20260910-r1`), and must move mean JCT by
  more than about 2 min to be separable from a repeat.
- **Comparison tool.** `scripts/evaluation/compare_two_instance_runs.py`
  produces the checklist (completion, JCT quantiles and makespan,
  token-weighted engine TPOT, cached share, common-window steps/min and
  output tokens/s, paired mean-JCT bootstrap over 28 sources, worst tasks)
  from run directories alone and reproduces the 2026-09-09 `comparison.json`.
- **Budget.** No new agent traces can be recorded. Every workload variant is
  built from the 28 mixed28 source trajectories and their replay copies.

## 1. Frontier A — DualMap dispatch fairness

**Settled.** DualMap's public heap dispatches by cached-token estimate first,
arrival second. Its residual tail is request-level queueing, not task
starvation. Of the 255 router waits over 60 s in the baseline repeat, 238
belong to requests with a small but nonzero cache estimate (under 4K tokens,
under 0.2 s of prefill) held behind high-residency requests; only 14 have a
zero estimate, and long waiters rarely compete with requests of equal
estimate. Unbounded, unrepaid wait credit (2026-09-09) lost on every metric.

**Candidate.** Seconds-based dispatch order (arrival − credit − cached
tokens × calibrated prefill seconds per token), with the credit capped at
the DualMap TTFT SLO (5 s) and repaid on every dispatch: only a task's most
recent router wait carries forward
(`scripts/baselines/dualmap_official_proxy.py --wait-credit-cap-s 5`,
`DUALMAP_WAIT_CREDIT_CAP_S=5` in the launcher). One predeclared cap, no
sweep.

**Result (2026-09-10, `results/mixed56-vast-dualmap-credit5-20260910-r1`).**
Worse on every metric. Paired mean JCT +1057 s, 95% [+775, +1377]
(28 sources, 2,000 draws), against Continuum +935 s.

| Metric | Bounded credit (5 s) | DualMap repeat |
|---|---:|---:|
| Completed | 56/56, 2470 requests | 56/56, 2470 |
| Mean JCT, min | 50.85 | 33.24 |
| P95 JCT, min | 107.17 | 70.59 |
| Max JCT, min | 131.68 | 75.31 |
| Makespan, min | 132.37 | 76.02 |
| Engine TPOT, ms | 107.42 | 87.53 |
| Cached prompt share | 31.7% | 75.7% |
| Steps/min, 75.3 min window | 35.25 | 48.07 |
| Output tokens/s | 105.9 | 145.0 |

Worst tasks: both replicas of PennyLane-4161 (131.68 and
129.24 min, vs 70.3 and 73.2 under DualMap), then 5857 and 4697.

**Mechanism (router log, originals only).** Router waits were bounded as
designed: max 109 s vs 270 s overall, and for the 4161 replicas mean 32 s,
max 83 s vs 8.7 s and 249 s. But the shadow-cache estimate at dispatch for
those replicas' steps 50 onward fell from a median of 79K tokens to 272
tokens (a third at zero), rebalance counts per dispatch tripled (mean 170
vs 55), and the run-wide cached share fell from 76% to 32%. Once wait is
priced in seconds, a 100K-token hit is worth about 5.5 s and a 5 s credit
on every other request outranks it; the large-context task is delayed,
loses residency, returns cold and cheaper to delay again, and pays this on
each of its 82 steps. This is the same loop that ThunderAgent enters by
suspending large-context tasks (Milestone 2 §5): delay → eviction → cold
prefill → higher engine cost for everyone → further delay. Unmodified
DualMap avoids it because residency-first is the one order that never
delays the requests whose delay is expensive; its tail is made of requests
whose residency is worth under 0.2 s.

**Frontier closed.** The fairness unit must be recompute cost, not seconds.
A credit is safe only if it never displaces a request whose residency
exceeds the credit's token value, and the requests in DualMap's tail can
only advance by displacing exactly such requests. On this workload at two
GPUs there is no room between residency-first and the starvation loop, so
no further credit variants will be run. Cap 0 (pure seconds order) is not
worth a run: the credit saturated at the cap on 83% of dispatches, so this
run already approximates it.

## 2. Frontier B — Cross-instance load balance

**Settled.** Under mixed56 (56 tasks ready at t=0, concurrency 32,
replacement tasks keep pressure) both GPUs are busy over 96% of the time
under every policy, and per-request least-requests routing loses to sticky
routing (Milestone 2 §7). Nothing to reclaim there.

**Open question.** Does a workload as close to mixed56 as possible, but
without the all-at-t=0 saturation, create persistent imbalance under sticky
placement, and does balancing then pay? The asymmetry that could drive it
is real: per source task, PennyLane vs SQLGlot has 52.8 vs 36.5 LLM steps,
60K vs 34K peak prompt tokens, 1.77M vs 0.72M total prompt tokens, and 1,992
vs 480 s source wall time (tool time 1,647 vs 281 s, so 4× tool replay
compresses PennyLane far more). Sticky placement is least-outstanding with
alternating ties (`scripts/baselines/least_requests_proxy.py`), so a burst
of heavy tasks can land on one engine and stay there.

**Knobs.** Available by manifest edit alone: per-task `arrival_s`
(staggered, Poisson or bursty arrivals; the replacement stream starts after
the last arrival), and composition (extra replicas). Needing a small code
change: disabling replacement or changing tool speed
(`scripts/evaluation/vast_two_instance.py` hardcodes both), and pinning
placement per task (proxy `completion()`).

**Runs (2026-09-10 night chain).** mixed56's 56 tasks with Poisson
arrivals at one task per minute (`analysis/development/mixed56-poisson60-2l40s-concurrency32-v1`,
seed 42, cumulative exponential gaps, first arrival at 0, last at 52.3 min),
concurrency 32, replacement left as is. Little's law puts steady-state active
tasks near the concurrency cap (arrival rate × ~33 min mean JCT), so nominal
load matches mixed56 while the active set is built up gradually and
fluctuates; one run therefore sweeps density from 1 to about 30 active
tasks. Four policies on that manifest: FCFS task-sticky, FCFS
least-requests, DualMap, Continuum task-sticky. Primary comparison:
least-requests vs sticky, with the GPU timeline (both-busy / one-idle
fractions by 10-minute window, `scripts/evaluation/gpu_balance_summary.py`)
and the checklist. Go/no-go for the frontier: a one-busy-other-idle fraction
that is no longer negligible in some density range and a paired JCT
difference in favour of balancing there; if sticky still wins, load balance
is closed at this scale.

**Run 1, FCFS task-sticky (`results/mixed56p60-vast-fcfs-sticky-20260910-r1`).**
56/56 tasks, 2,470 requests; JCT measured from each task's arrival: mean
17.32 min, P95 59.25, max 111.86 (a PennyLane-6049 replica arriving at
minute 52), makespan 164.8 min; engine TPOT 62.1 ms; cached share 72%.
Placement put 28 original tasks on each engine. GPU balance over the whole
window: both busy 88.8%, one busy while the other idle 2.54% (249 s, longest
7.7 s), utilization gap over 20 points 19.1% (longest 39 s). By phase:

| Window | Util GPU0 / GPU1 | Both busy | One busy, other idle | Longest one-idle |
|---|---:|---:|---:|---:|
| 0–10 min | 52 / 47% | 41% | 20.1% | 4.6 s |
| 10–20 | 79 / 86% | 88% | 0.2% | 1.2 s |
| 20–30 | 83 / 67% | 74% | 2.8% | 4.8 s |
| 30–40 | 73 / 68% | 67% | 6.0% | 3.8 s |
| 40–50 | 76 / 59% | 64% | 9.5% | 7.7 s |
| 50–60 | 76 / 83% | 82% | 3.0% | 1.8 s |
| 60–165 | 97 / 97% | ≥99.8% | ≤0.03% | 0.2 s |

Observation: imbalance exists only while the active set is being built
(the 52-minute arrival ramp), reaching a 17-point utilization gap and 9.5%
one-idle time at mid density; once arrivals stop and replacement tasks fill
freed slots, both engines saturate exactly as under mixed56. Under the
mixed56 protocol the ramp phase is 0 s long, which is why Milestone 2 saw
no imbalance. The routing-telemetry checker does not apply to this proxy's
log (dispatch and finish only, 5,277 each, balanced).

**Run 2, FCFS least-requests (`results/mixed56p60-vast-fcfs-least-requests-20260910-r1`),
against run 1.** Per-request balancing removes the ramp imbalance and still
loses on every JCT and efficiency metric:

| Metric | Least-requests | Sticky |
|---|---:|---:|
| Completed | 56/56, 2,470 req | 56/56, 2,470 |
| Mean JCT, min (from arrival) | 21.15 | 17.32 |
| P95 / max JCT, min | 64.94 / 110.85 | 59.25 / 111.86 |
| Engine TPOT, ms | 78.33 | 62.06 |
| Cached prompt share | 52% | 72% |
| Steps/min, 110.9 min window | 34.50 | 35.28 |
| Both busy, whole window | 93.5% | 88.8% |
| One busy other idle, whole window | 1.34% (130 s) | 2.54% (249 s) |
| Ramp 20–50 min: both busy / one idle | 90–92% / ≤0.3% | 64–74% / 2.8–9.5% |

Paired mean JCT +230 s, 95% [+164, +310] against sticky. Worst tasks are the
same replicas (6049-002, 4161-002, 6049-001) under both.

**Verdict.** The GPU-time imbalance that sticky placement creates during the
ramp is real (up to 9.5% one-idle time) and least-requests does reclaim it
(both-busy 90% or more throughout the ramp), but the reclaimed GPU time is
spent on prefix recomputation: cached share drops 20 points and TPOT rises
26%, and mean JCT ends 22% worse. Locality is worth more than the idle time
it costs, at every density this workload passes through. Per-request load
balancing is closed as a lever at 2 GPUs.

**Run 3, DualMap (`results/mixed56p60-vast-dualmap-20260910-r1`).** The
locality-preserving router wins on every metric against both FCFS variants
while leaving the most GPU time idle:

| Metric | DualMap | Sticky | Least-requests |
|---|---:|---:|---:|
| Mean JCT, min (from arrival) | 14.23 | 17.32 | 21.15 |
| P95 / max JCT, min | 40.96 / 66.06 | 59.25 / 111.86 | 64.94 / 110.85 |
| Makespan, min | 118.9 | 164.8 | 163.8 |
| Engine TPOT, ms | 55.64 | 62.06 | 78.33 |
| Cached prompt share | 91% | 72% | 52% |
| Steps/min, 66.1 min window | 43.80 | 42.10 | 40.10 |
| Both busy / one idle, whole window | 80.4% / 4.86% | 88.8% / 2.54% | 93.5% / 1.34% |
| Ramp 20–50 min one idle | 7–18% | 2.8–9.5% | ≤0.3% |

Paired mean JCT −185 s, 95% [−375, −41] against sticky and −415 s
[−630, −244] against least-requests. Worst tasks are the same replicas.

**Frontier B verdict.** Across three policies the ranking of GPU balance is
the reverse of the ranking of JCT: least-requests keeps both GPUs busiest
and finishes last; DualMap idles one GPU 4.9% of the time, more than
double sticky FCFS, and finishes the cohort 46 minutes earlier with 91%
cached share. On this workload family, at 2 GPUs, at every density the
Poisson ramp passes through, GPU idle time is not the resource to
reclaim; prefix residency is. Load balance is closed as a frontier.

**Run 5, Continuum task-sticky (`results/mixed56p60-vast-continuum-sticky-20260911-r1`),
aborted.** All 56 original tasks completed successfully, but at minute 100
one replacement task's request hit a proxy-to-engine connection error
(HTTP 500 from the least-requests proxy, `httpcore.ReadError`, no engine
error), the replacement failed its tool-replay contract, and the simulator
aborts the whole replay on the first replacement failure before writing
`throughput_summary.json`. Per-task summaries for the originals exist in the
run's simulate trace and can be reconstructed if the four-policy table is
wanted; the run is not rerun tonight because it cannot change the verdict.
Two defects to report, not fixed here: a single transport error on a
background request terminates a replay, and the router proxy has no retry
for a dropped engine connection.

## 3. Frontier C — PD/PPD routing

**Closed 2026-09-11.** Fixed PD, public PPD (always-local), and the
two-sided expected-cost router (result below) all lose to DualMap by 2.2×
or more in mean JCT on mixed56 at 2 GPUs; the two-sided run shows why a
per-request cost rule cannot fix it.

**Settled.** Fixed PD is prefill-bound (mean JCT 71.5 min); public PPD
routes 2,414 of 2,470 requests to local prefill on the decode worker
(109.2 min). With predicted output 128 every appended input of 512 tokens
or more classifies as `huge_paste`, and the lookup says local at the QPS
points seen, so the public rule equals always-local on this workload. The
extended-context and state-aware PPD runs prepared on 2026-09-08 were never
executed.

**What the 448-group profiling matrix establishes** (analysis of
`results/serving-length-profile-vast-20260908-complete/`, 2026-09-10;
observation first, inference after):

| Cost constant (measured) | Value |
|---|---|
| Local prefill on D per uncached token | 0.184 ms (+0.03 s), r² 0.97; attention over cached tokens ≈3% of that |
| P prefill per token, cold (prompts >30K) | 0.295 ms, r² 0.98 (0.216 ms overall) |
| KV transfer | 0.0102 ms per token + 22 ms (≈14 GB/s above 16K tokens) |
| PD handoff at low load (transfer + D queue + D prefill + proxy) | 0.33–0.74 s |
| Proxy/HTTP residual, both paths | ≈0.34 s |

- Local wins turn-2 E2E in 210 of 224 pairs by 38.6 s on average. The
  advantage tracks the P worker's queue (0.0, 0.1, 2.5, 27.0, 99.0 s for
  history ≤4K, 4–16K, 16–32K, 32–64K, >64K): in the long-history groups the
  eight concurrent conversations exceed one engine's KV capacity, turn 2
  waits behind other conversations' turn-1 prefills in P's FIFO, and P has
  evicted the history by the time it runs, so it re-prefills the whole
  prompt. Inference: the matrix compares PD with a cold, saturated P
  against local with a warm, idle D. That is the measured condition, not
  the mixed56 condition.
- Local loses decode TPOT in 150 pairs (mean +3.4 ms, up to +16 ms for
  conversations sharing D with a 76K zero-hit local prefill). The
  difference correlates with uncached tokens on D (0.64) and summed local
  prefill time (0.57), not with decode concurrency (−0.01): the PD
  advantage is D being freed from chunked prefill, worth 1–5 ms per token.
- The 14 E2E losses of local are long outputs (10 of 14 with O ≥ 256) at
  1–3K uncached tokens; true output length is not available online.
- Decision reconstruction on decision-time inputs: no rule beats
  always-local on E2E inside the matrix (14 errors, 12.7 s regret); the
  E2E oracle still leaves 261 s of TPOT regret. The declared state guard
  (uncached ≥ threshold and D saturated) never fires because D never
  exceeded 7 active requests in profiling.
- Coverage gap: profiling covers uncached ≤8K at 0–7 active requests on D,
  plus 72 cold requests at ≤7 active. Online mixed56 decisions had median
  30 outstanding requests on D with uncached 8–110K tokens. That regime
  has zero profiling observations, and the PD side was measured only with
  a saturated P.

**Consequence.** The matrix does not support shipping a length-only or
lookup-based rule; it supplies cost constants. The right routing signal is
expected uncached prefill work on each side under each side's current
load. Both sides can be snapshotted the way the decode side already is
(`ppd_policy.cache_snapshot`: cached tokens, running, waiting, KV usage).

**Engine build on this host.** The pinned vLLM 0.28.0 wheel is a CUDA 13
build and needs driver 580 or newer; this host has driver 570 (CUDA 12.8), so
the first profiling launch failed at engine start. PD-family runs from
2026-09-11 on use a separate venv with vLLM 0.28.0+cu129, torch 2.13.0+cu129
and nixl-cu12 1.4.1 (`PPD_CUDA=cu129`, `PPD_VENV=/workspace/venvs/ppd-cu129`),
with the same two patches. JCT stays comparable with the 2026-09-07 PD/PPD
runs under the standing cross-build caveat; TPOT carries a build offset. The
first profiling attempt on that venv stalled because the launcher's default
UCX transport (TCP over loopback) pushed 3.4 GB of KV in 6.9 s and then
stopped after two transfers; the 2026-09-07 runs used `UCX_TLS=all
UCX_NET_DEVICES=all` (GPU-direct, about 30 ms per transfer). The launcher now
defaults both to `all`; `PD_UCX_TLS` / `PD_UCX_NET_DEVICES` still override.

**Runs (2026-09-10 night chain), in order.**

1. *Controlled-load profiling* (`analysis/development/ppd-load-profile-v1/plan.md`,
   harness stage `load`): for the three most frequent long-history cells of
   mixed56 (P17 23K, P29 40K, P35 76K history), build N histories first, wait
   until nothing is outstanding, then release N turn-2 requests together,
   N ∈ {8, 16, 32}, both paths, seed 42, 18 groups. This measures each path
   under a known decode-side burst with an idle prefill worker and fills the
   coverage table's empty cells; cache state follows from N × history and is
   read from the records. Output: the (uncached tokens, D load) region where
   PD wins, and the fraction of mixed56 and Poisson-mixed56 requests inside it.
**Controlled-load profiling result (`results/ppd-load-profile-20260911-r3`,
18 groups, all valid; summary in `load-profile-summary.txt`).** Turn-2 means
per cell, local minus PD, with D's cache state as measured:

| Point (history) | Burst N | Local cached fraction | ΔTTFT s | ΔTPOT ms | ΔE2E s |
|---|---:|---:|---:|---:|---:|
| P17 (23K) | 8 | 0.99 | −0.4 | −0.2 | −0.5 |
| P17 | 16 | 0.49 | −3.9 | +19.8 | −0.3 |
| P17 | 32 | 0.18 | +4.9 | +50.9 | +14.3 |
| P29 (40K) | 8 | 0.33 | +11.2 | +43.5 | +21.1 |
| P29 | 16 | 0.00 | +19.9 | +93.3 | +41.2 |
| P29 | 32 | 0.07 | +27.1 | +98.9 | +49.6 |
| P35 (76K) | 8 | 0.20 | +10.5 | +92.0 | +29.8 |
| P35 | 16 | 0.00 | +49.4 | +117.3 | +74.0 |
| P35 | 32 | 0.02 | +56.8 | +123.9 | +82.8 |

Observation: with an idle P, PD wins end-to-end in 7 of 9 cells; local wins
only when the history is resident on D (23K history, burst 8) and ties at
half residency while paying 20 ms per token. Under the PD path D's decode
TPOT stays at 39–43 ms at every burst size; under local prefill it climbs to
93–163 ms, and the D queue reaches 246 s at burst 32 with 76K histories.
P's queue grows linearly with the burst (191 s at N=32, P35) but never
touches decoding. Inference: the matrix's "local always wins" was the
saturated-P artifact; the real rule is residency on D versus pending
prefill work on each side, which is what the two-sided estimate computes.
The online mixed56 regime (D saturated, 1.7% resident) sits in the PD-wins
region, so all-local's collapse and fixed PD's advantage are both
explained, and a router that sends resident requests to D and cold ones to
P should sit between fixed PD and the residency-first single-role systems.

**Amendment (2026-09-11 04:10 UTC, before any two-sided replay).** The
pre-declared queue term charged one 2,048-token batch per waiting request
(0.4–0.6 s); the profiling measured 1.4–12 s per queued request, because a
queued agent prompt is a full 23K–76K prefill. The snapshot now reports each
engine's pending prefill tokens (prompt tokens not yet computed over waiting
and running requests) and the estimate charges them at that side's per-token
cost. The per-token constants were confirmed by the profiling (D prefill
span within 30% of 0.184 ms × uncached tokens) and are unchanged. Commit
0c6eb91; the two-sided smoke with the amended code passed before C1.

2. *Two-sided expected-cost routing* (`ppd_official_proxy.py --two-sided`,
   `PPD_TWO_SIDED=1`): snapshot both engines per request and route to the
   lower estimated time to first token, local = D queue + 0.184 ms ×
   uncached on D; PD = P queue + 0.295 ms × uncached on P + 0.0102 ms ×
   prompt tokens + 0.4 s handoff; queue = that side's pending prefill tokens
   at its per-token cost (amended, see above). No length threshold, no lookup table.
   Pre-declared criterion: if it cannot beat fixed PD on JCT,
   prefill/decode disaggregation is closed for this workload at 2 GPUs.

**Two-sided result (`results/mixed56-vast-ppd-two-sided-20260911-r1`,
2026-09-11 04:17–07:00 UTC, one run, smoke passed first). Verdict: does not
beat fixed PD; Frontier C is closed.**

| Metric | Two-sided | Fixed PD (09-07 r3) | DualMap (09-10 r1) |
|---|---:|---:|---:|
| Completion | 56/56, 2,470 | 56/56, 2,470 | 56/56, 2,470 |
| Mean JCT, min | 75.0 | 71.5 | 33.2 |
| P95 JCT, min | 154.5 | 143.5 | 70.6 |
| Max JCT / makespan, min | 163.0 / 163.0 | 177.1 / 177.8 | 75.3 / 76.0 |
| Engine TPOT, ms, token-weighted | 96.9 | 42.0 | 87.5 |
| Steps/min, common window | 17.2 (163 min) | 22.9 | 48.1 (75 min; two-sided 27.5) |
| Paired mean JCT vs fixed PD | +212 s, 95% [+56, +382] | — | — |

Routing: 82 first turns to D; of 2,736 later turns 973 (36%) went local
and 1,763 via P. Time to first token by path, later turns: local 40.3 s
mean (P95 67 s), PD 74.0 s (P95 140 s); the fixed-PD run's 71.5 s (P95
113 s). The cached-prompt share (0.60 versus 1.00) is not comparable in PD
mode because transferred KV counts as cached on D.

Mechanism (observation, then inference). The estimator minimized the time
to first token of the request in hand and achieved it for the local
requests, but a local prefill on D is charged nothing for what it does to
D. Measured over the run: D's waiting queue held 13–15 requests throughout
(fixed PD: 0), D's KV usage averaged 75–85% (fixed PD: about 30%), 12.4M
uncached tokens were prefilled on D, and decode TPOT doubled (97 ms versus
42 ms), which the controlled-load profiling had predicted (93–163 ms under
local prefill). Total prefill work is the binding resource at this
concurrency; moving a third of it from P to D relieved P and saturated D,
and the completion curves of the two runs are identical through minute 106
(46 of 56 done in both). Between minutes 106 and 134 D fell into a
preemption livelock: eight running sequences with about 90K-token contexts
exceed the 274K-token KV cache, vLLM preempted and recomputed continuously
(38,250 preemptions in 27 minutes, generation 17–25 tokens/s against about
100), 26 requests took about 1,680 s each, one hit the proxy's 1,800 s
timeout; one original task and 25 replacement requests were affected. The
fixed-PD run recorded zero preemptions on D. The router's decision at that
moment (D at 97% KV, 17 waiting) was PD, so the livelock was built by the
earlier local decisions, not by a wrong call at the point of failure.
Inference: a routing rule that prices only the requesting turn's latency
cannot be safe on the decode side; it needs D's decode TPOT and KV headroom
in the cost, or admission control on D. Neither addresses the 2.3× gap to
DualMap, whose advantage is 76% cache residency with both GPUs decoding,
so the frontier is closed rather than iterated on mixed56.

**Why disaggregation pays elsewhere and not here** (2026-09-11, from this
project's measurements; observation first, inference after).

Published PD/PPD gains (DistServe, Splitwise, Mooncake, the PPD upstream)
come from serving conditions that our runs measured the opposite of:

| Condition where PD wins | What the agent workload measured |
|---|---|
| Independent, short-lived conversations; prompts mostly cold, prefill is real compute that must run somewhere | Each turn shares nearly its whole prompt with the previous turn; DualMap hits 76% of prompt tokens in cache. The scarce resource is where the history lives, not prefill throughput |
| Prompts of a few thousand tokens, outputs of hundreds; decode is a large share of GPU time, so prefill interference on running decodes is the main loss | Prompts 23K–110K, outputs about 150 tokens. Prefill dominates compute; centralizing it on one of two GPUs made that GPU the bottleneck (P queue 71 s TTFT) while D's decode isolation bought only TPOT (42 ms, the best of every policy) |
| Goodput under TTFT/TPOT SLOs | Task JCT over 50–100 sequential steps; step latency is TTFT plus a short decode, so stable TPOT buys nothing |
| Pools of many GPUs with a tunable prefill:decode ratio | Two GPUs, so one role bounds the system whichever way the split goes |
| KV per request small next to compute; fast interconnect | 274K-token KV per L40S holds about three task histories; PD needs the history on P for hits and on D for decode, or re-prefills it on P every turn (the single P thrashed) |
| Requests arrive and leave | Tool gaps of seconds to minutes between turns leave KV idle under eviction pressure |

Inference: the mismatch is the per-request structure (multi-turn,
long-context, high-reuse, short-output, judged by sequential step count),
not the arrival pattern; burst and Poisson arrivals are both realistic and
both fail for the same reason. Where PD could still fit agents: more GPUs
with a prefill-heavy ratio and task-sticky prefill instances so histories
stay hot on P, or heterogeneous hardware with memory-rich decode nodes.
Neither exists at two GPUs, so this frontier does not reopen on the current
host.

**Step 7, two-sided on the Poisson manifest (pre-registered 2026-09-11
07:50 UTC, before launch; `results/mixed56p60-vast-ppd-two-sided-20260911-r1`).**
The chain's condition for this run was that the profiling predicts a win
at moderate density; it does, in the sense that local prefill wins when the
history is resident on D (P17 at burst 8), and residency is higher while
the Poisson arrivals ramp. References are the three Poisson runs of
Frontier B (FCFS sticky 17.3 min, least-requests 21.1 min, DualMap 14.2 min
mean JCT), since no fixed-PD Poisson run exists and one would not change
the frontier verdict. Criterion: 56/56, and mean JCT below FCFS sticky's
17.3 min keeps disaggregation open at moderate density; otherwise the
closure above stands for both arrival patterns. Prediction from the
mixed56 mechanism: the arrivals finish at minute 52 while tasks run about
an hour, so concurrency reaches the mixed56 regime in the second half and
D saturates again; a loss to FCFS sticky is expected.

*Result (run 07:49–10:54 UTC, stopped by the advisor at run minute 185
with 53 of 56 originals finished; `administrative-stop.json` in the run
directory; recorded as a bounded result, not a completed run).* Every
Poisson reference had finished all 56 tasks by minute 165. The 53 finished
tasks average 39.2 min JCT (P95 96 min, max 127 min). The three unfinished
tasks had been running 133, 133 and 144 min at the stop, so the mean over
all 56 is at least 44.4 min against 17.3 min for FCFS sticky, 21.2 min for
least-requests and 14.2 min for DualMap; the same three tasks took 74–112
min under FCFS sticky and 57–66 min under DualMap. Routing at minute 83:
846 local, 1,393 via P, no request over 155 s, so no livelock; the loss is
the same steady-state cost as on mixed56 (all prefill on two roles instead
of two caches). JCTs are reconstructed from trace timestamps as for the
mixed56 run (arrival zero = first action minus 2.6 s, terminal = task
complete plus 2.6 s, anchors from the Poisson references, about ±2 s per
task). The host went away during the artifact pull, so `server/` holds
the engine logs of both instances and instance-0's telemetry but not
`routing.jsonl`, `proxy.log` or instance-1's request telemetry; the
routing counts above come from the minute-83 check on the host. Verdict:
the criterion fails by a factor above 2.5 at moderate
density as well; the Frontier C closure holds for both arrival patterns
without caveat.

Run-record note. The timed-out replacement call surfaced the simulator
defect fixed in commit 2ebe6f4 (a replacement-task failure raised after all
56 originals had finished, and no throughput summary was written).
`output/throughput_summary.json` for this run is reconstructed from the
trace timestamps (first replay action minus 6.1 s as arrival zero; task
terminal as `task_complete` plus 2.8 s; anchors from the three reference
runs, per-task uncertainty about ±2 s) and says so in a `reconstruction`
block; the checklist output is `comparison-reconstructed.{json,txt}`.

## 4. Evidence

- DualMap repeat: `../results/mixed56-vast-dualmap-20260910-r1/` (`comparison.json`)
- Bounded-credit candidate: `../results/mixed56-vast-dualmap-credit5-20260910-r1/`
- Profiling matrix: `../results/serving-length-profile-vast-20260908-complete/`
  (`paired.csv`, `requests.csv`, `plan.md`); PD/PPD runs
  `../results/mixed56-vast-pd-pcie-20260907-r3/`, `../results/mixed56-vast-ppd-pcie-20260907-r1/`
- Poisson-arrival runs: `../results/mixed56p60-vast-*-2026091{0,1}-r1/`
  (`comparison.json`, `gpu-balance-summary.json`); controlled-load profiling
  `../results/ppd-load-profile-20260911-r3/` (`load-profile-summary.txt`;
  r1 failed at engine start, r2 stalled on the TCP transport); two-sided PPD
  `../results/mixed56-vast-ppd-two-sided-20260911-r1/`
  (`comparison-reconstructed.json`, `server/routing.jsonl` with both
  snapshots per decision)
- Chain scripts and shared log: `../results/night-chain-20260910.{sh,log}`,
  `../results/night-chain{2,3,4}-20260911.sh`
