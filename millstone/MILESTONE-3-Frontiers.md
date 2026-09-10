# MILESTONE 3 — Three frontiers on 2× L40S

Date: 2026-09-10. Status: in progress. Milestone 2 froze the exploration
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

**Next run (not started).** mixed56's 56 tasks with Poisson arrivals at one
task per minute (seed 42, cumulative exponential gaps, first arrival at 0),
concurrency 32 and replacement left as is. Little's law puts steady-state
active tasks near the concurrency cap (arrival rate × ~33 min mean JCT), so
nominal load matches mixed56 while the active set now fluctuates and is
built up gradually. Primary comparison: FCFS task-sticky vs FCFS
least-requests on the same manifest, with the GPU timeline
(both-busy / one-idle fractions, longest one-idle interval) and the
checklist. Go/no-go for the frontier: a one-busy-other-idle fraction that
is no longer negligible and a paired JCT difference in favour of balancing;
if sticky still wins with both GPUs saturated, load balance is closed at
this scale. Cost: two runs of about 80 min each.

## 3. Frontier C — PD/PPD routing

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

**Next run (not started).** Two-sided expected-cost routing: query the
cache snapshot on both engines per request and choose the side with the
lower estimated time to first token, cost = engine queue estimate (waiting
× mean prefill in flight) + per-token prefill cost × uncached tokens on
that side (+ transfer for the PD path), using the constants above. One
mixed56 run on the vLLM 0.28.0 build, compared with the existing fixed-PD
and all-local runs as the two endpoints, with the checklist plus per-request
TTFT split by history bucket. Cost: PD-family runs took 3–4.5 h each. If the
two-sided rule cannot beat fixed PD on JCT, prefill/decode disaggregation is
closed for this workload at 2 GPUs.

## 4. Evidence

- DualMap repeat: `../results/mixed56-vast-dualmap-20260910-r1/` (`comparison.json`)
- Bounded-credit candidate: `../results/mixed56-vast-dualmap-credit5-20260910-r1/`
- Profiling matrix: `../results/serving-length-profile-vast-20260908-complete/`
  (`paired.csv`, `requests.csv`, `plan.md`); PD/PPD runs
  `../results/mixed56-vast-pd-pcie-20260907-r3/`, `../results/mixed56-vast-ppd-pcie-20260907-r1/`
- Workload manifests with Poisson arrivals to copy the construction from:
  `analysis/development/mixed128-poisson-v1/manifest.yaml`
