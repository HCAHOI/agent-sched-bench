# MILESTONE 4 — KV pressure as the axis, on 2× RTX Pro 6000

Date: 2026-09-11. Status: in progress. Milestone 3 closed the three L40S
frontiers. Moving the same workload to a host with 2.3× the KV capacity
showed that every Milestone 2 and 3 result was a statement about one
operating point: KV demand above capacity. This document fixes the
operating point as an explicit, pre-registered quantity, records the bridge
runs that exposed it, and carries the decision plan on whether
multi-instance scheduling is a problem at all at realistic pressure.
Metrics and the reporting checklist are those of Milestone 2 §2 and
Milestone 1 §3.

## 0. Host and comparability

- **Host.** gpuhub/AutoDL container, `ssh -p 41548
  root@connect.singapore-a.gpuhub.com`: 2× RTX Pro 6000 Blackwell Server
  Edition, 96 GB each, driver 595 (CUDA 13.2 native), PCIe 5 on one NUMA
  node, cgroup limits 50 cores and 240 GB, 250 GB persistent disk with
  `/workspace` symlinked into it. Bootstrapped 2026-09-11 with
  `benchmark_server.sh --serving-host` after four host fixes: uv and Python
  3.12 installed under `/workspace` (no system Python), CUDA compat package
  skipped on drivers ≥580, PPD patches applied on the physical package path
  (git refuses paths through the symlink), and LMCache 0.3.7 rebuilt from
  source because its wheel has no sm_120 kernels (the first DualMap
  calibration died at the first CPU offload with "no kernel image").
  Kernels without native Blackwell code JIT once (72 s first request) into a
  cache the driver now keeps under `/workspace/.nv/ComputeCache`.
- **Engine facts.** vLLM 0.10.2 with Flash Attention in eager mode, as on
  L40S. GPU KV cache 624,880 tokens per GPU for Qwen3-4B-FP8 (L40S:
  275,008). DualMap prefill calibration 31.3 µs per token (L40S: 55.4 µs).
- **Comparability.** Nothing from the L40S hosts carries over; the L40S
  runs remain the Milestone 2 and 3 record and are cited here only as the
  high-pressure point. New-host candidates compare with the new-host
  baselines in §1. One physical run per policy; the paired bootstrap over
  28 sources is the uncertainty; about 2 min of mean JCT separates a
  candidate from a repeat (Milestone 3 §0).
- **Budget.** No new agent traces. The pool of recorded traces is
  `traces/exports/` (§3); workloads are built from it by selection, not by
  replay copies.

## 1. Bridge runs: the same workload on 2.3× capacity

mixed56 at concurrency 32, both policies, both hosts (`comparison.json` in
each new-host run directory):

| Metric | FCFS sticky, Pro 6000 | DualMap, Pro 6000 | FCFS sticky, L40S | DualMap, L40S |
|---|---:|---:|---:|---:|
| Completion | 56/56 | 56/56 | 56/56 | 56/56 |
| Mean JCT, min | 15.4 | 15.8 | 55.3 | 33.2 |
| P95 / max JCT, min | 40.6 / 44.2 | 36.4 / 37.8 | 111 / 139 | 70.6 / 75.3 |
| Engine TPOT, ms | 37.6 | 39.7 | 131 | 87.5 |
| Cached prompt share | 0.90 | 0.89 | 0.09 | 0.76 |
| GPU prefix hit rate, final | — | 87% | — | 40–46% |
| GPU KV usage, mean / peak | — | 35% / 70% | — | 75% / 100% |
| LMCache CPU-tier tokens retrieved | — | 0 | — | 25 M |

Paired mean JCT: DualMap Pro 6000 vs DualMap L40S −1,047 s [−1,301, −803];
FCFS sticky minus DualMap on Pro 6000 −21 s [−74, +40]; on L40S the same
comparison was a 22-minute DualMap win.

Observation: two effects stack on the new host. Raw speed roughly doubled
(prefill 1.77×, decode TPOT 2.2×). Capacity removed the bottleneck: the
cache never filled, nothing was evicted, the GPU prefix hit rate doubled,
and the CPU tier made 2,223 lookups and stored 17.6 M tokens but retrieved
none. Inference: with residency free, placement policy stops mattering for
the mean; DualMap keeps a small tail advantage. DualMap's 2× on L40S was a
property of the over-capacity regime, not of the workload.

**Tail imbalance at N=2 (`instance_balance_summary.py`, 5-minute windows,
originals only).** Under FCFS sticky, instance 0 ran dry at minute 20 and
stayed idle to the end at minute 45 while instance 1 still carried 12, then
8, then 7 tasks; 4 of the 9 windows have max-over-mean prompt-token
imbalance above 1.5 and the last three are 2.0 (one GPU idle). DualMap
migrated tasks (home-instance share 0.82 against 1.0) and idled less: 1 of
8 windows above 1.5, makespan 38.5 min against 45.0. Mean JCT hides this
because the stranded tasks are the long ones either way; P95 and makespan
show it. On L40S the same FCFS sticky run had no window above 1.5, so the
placement luck of the arrival alternation decides whether the tail strands
one instance. Inference: imbalance exists even at N=2 and it is a tail
phenomenon that sticky routing cannot repair; the step 2 rule therefore
counts only windows with at least as many unfinished tasks as instances,
so a drain that no policy could balance is not read as imbalance, while a
stranded half-run is.

**Replica check.** Each mixed56 copy's system message starts with a unique
"Replay copy: NN" token, so the two replicas of a task diverge at the third
token and cannot share KV. Measured on the L40S and Pro 6000 DualMap runs:
99.9% of cached tokens fall within a task's own history; first-turn hits
are the scaffold shared across different tasks (about 20% of a first
prompt, identical for first- and second-started replicas). The record has
no replica artifact.

## 2. The pressure axis

Define, for a workload and a host, from the traces before any run:

- **R_avg** = concurrency × mean prompt tokens per LLM step (over the
  admitted traces) ÷ total GPU KV tokens. This is time-averaged demand.
- **R_peak** = concurrency × mean peak prompt tokens per trace ÷ total GPU
  KV tokens. An upper bound; reported beside R_avg.

mixed56 at 32: R_avg 1.3 and R_peak 2.3 on the L40S pair (550K tokens);
R_avg 0.57 and R_peak 1.0 on the Pro 6000 pair (1.25 M tokens). Measured
mean KV usage was 75% and 35%, consistent with R_avg once eviction
dynamics are added. Guard: R is computed from traces and concurrency
before a run, never adjusted after seeing which policy wins. The hardware
decides which R values are affordable in wall time; it does not change
the question. No further hardware is needed for this study: the model is
4B and the whole grid R_avg ∈ {0.8, 1.5, 2.5} is reachable on this host by
trace selection and concurrency.

## 3. Decision plan (pre-registered 2026-09-11 14:00 UTC)

**The trace pool.** 868 usable traces under `traces/exports/`, v5 format,
every completed swe-rebench instance has its task record in
`data/swe-rebench/tasks.json`:

| Group | n | Steps p50 / p90 | Peak context p50 / p90 / max | Tool time p50 / p90, s |
|---|---:|---:|---:|---:|
| swe-rebench, gpt-5.6 agent | 270 | 38 / 66 | 33K / 76K / 118K | 266 / 2,096 |
| swe-rebench, qwen3.7 agent | 359 | 42 / 72 | 28K / 54K / 88K | 27 / 152 |
| terminal-bench, gpt-5.6 agent | 239 | 11 / 26 | 8K / 27K / 59K | 6 / 285 |
| mixed56 sources (reference) | 28 | 39 / 64 | 41K / 77K / 101K | 292 / 1,976 |

The mixed56 sources sit at the heavy end of the pool. Only 56 traces exceed
64K peak context, so high R comes from concurrency, not longer tasks.

**Step 1 — decision workload (done 2026-09-11, `analysis/development/pool64-distinct-v1`).**
Built by `scripts/evaluation/build_pool_workload.py`: 64 distinct trajectories,
no replay copies, proportional stratified sampling over (corpus, agent model,
peak-context bucket) with seed 42, one trace per task (some instances were
run by both agents). Result: 47 swe-rebench (37 gpt-5.6, 27 qwen3.7 agents
overall) and 17 terminal-bench; 2,438 LLM steps; median peak context 28.5K;
median tool time 27 s; mean prompt tokens per step 20,713. Pressure at
concurrency 32 (the replay machine's 8 CPUs and 15 GB cap concurrency at
the proven 32, so pressure is set by KV capacity, not by more containers):

| Capacity (total KV tokens) | R_avg | R_peak |
|---|---:|---:|
| N=2 full, 1,249,760 | 0.53 | 0.74 |
| L40S pair (reference), 550,016 | 1.21 | 1.68 |
| capped 400,000 | 1.66 | 2.31 |

Terminal-bench tasks replay with minimal task records (tool replay never
runs the real task; the trace's own instance id resolves the record). The
four-trace `pool4-smoke-v1` manifest checks that path before any full run.

**Step 2 — is multi-instance a real problem (pre-registered 2026-09-11 14:40 UTC; configuration amended 15:05 UTC before any run, see below).**
Launcher generalized to k engines per GPU (commits 8cddf64, 89c2317): engines
of one GPU run under MPS with a fixed 100/k percent SM share each; per-
instance KV is set exactly with `--kv-cache-memory-bytes` and read back from
the engine log.

*Second amendment before launch (15:40 UTC).* The first run on
pool64-distinct-v2 aborted within eight minutes: the Continuum shadow
provider refuses steps that issue more than one tool call, and 217 of the
359 qwen-agent traces contain such steps (no gpt-agent trace does). The
workload is rebuilt as `pool64-distinct-v3` with those traces excluded (588
pool traces remain after both filters): 38 swe-rebench + 26 terminal-bench,
50 gpt-agent and 14 qwen-agent traces, 1,928 steps, mean prompt 19,500 tokens
per step, max peak 55.6K; R_avg 0.50 on the full pair and 1.19 at 524,288
tokens. Parallel tool calls are a real agent behaviour that this harness
cannot yet replay under Continuum-style step signalling; the exclusion is a
harness limitation, recorded here, not a workload choice. Capacities and
decision rule are unchanged. Two earlier harness defects were also fixed on
the way (commits d1b122f, 61f1769 and follow-up): qwen-agent traces record
executor tool-call ids that differ from the model's ids, and the replay now
aliases them; mixed56 was never affected.

*Third amendment before any result (15:45 UTC).* The first run on
pool64-distinct-v3 (`results/pool64v3-pro6000-n2full-fcfs-sticky-20260911-r1`)
aborted after 13 minutes: the terminal-bench trace break-filter-js-from-html
records a second LLM step with zero prompt and zero completion tokens, the
replay therefore asked the engine for `max_tokens=0`, and vLLM rejected the
request (400). The 43 tasks that had started all completed; the run is a
harness abort, not a result, and its numbers were not read. Eight of the 868
pool traces (four after the two earlier filters) carry such a step; the pool
builder now excludes them unconditionally, since a zero-token generation
cannot be replayed. The workload is rebuilt as `pool64-distinct-v4` with the
same sampler and seed: the 38 swe-rebench picks are identical, the
terminal-bench <16K stratum reshuffled (26 terminal-bench traces, 15 of them
new), 1,951 steps, mean prompt 19,355 tokens per step; R_avg 0.50 on the full
pair and 1.18 at 524,288 tokens. Capacities and decision rule are unchanged;
the runs are named `pool64v4-...` (`results/chain10-step2-20260911.sh`).

*Amendment before launch.* The first configuration (N=8 at 50K tokens per
instance, 400K total) is physically ill-posed: vLLM refuses a KV cache that
cannot hold one maximum-length request, and a replica that small could not
serve a 90K-token context in any deployment either. The k=4 smoke failed on
exactly that check. Revised design: the workload's peak context is bounded
at 60K (`pool64-distinct-v2`, superseded by v3 above: same sampler and seed
on the pool traces with peak ≤ 60K), the engine context limit is
65,536 tokens, and every capped instance holds the same total of 524,288
tokens split evenly, so N=8 has exactly one full context per instance:

| Configuration | Per instance | Total KV tokens | R_avg | R_peak | Policies |
|---|---:|---:|---:|---:|---|
| N=2 full | 624,880 | 1,249,760 | 0.50 | 0.65 | FCFS sticky, DualMap |
| N=2 capped | 262,144 | 524,288 | 1.19 | 1.54 | FCFS sticky, DualMap |
| N=8 capped (k=4) | 65,536 | 524,288 | 1.19 | 1.54 | FCFS sticky, DualMap |
| N=4 capped (k=2) | 131,072 | 524,288 | 1.19 | 1.54 | FCFS sticky, DualMap (last, optional) |

The capped rows sit at the L40S pair's pressure for this workload (550K
tokens, R_avg 1.13), so N is the only thing that varies across them; N=2
full is the low-pressure reference. DualMap is calibrated separately at k=2
and k=4 (its prefill constant changes under the SM share) and its CPU tier
stays 96 GB in total (48, 24, 12 GiB per instance). Eight runs on
pool64-distinct-v4 at concurrency 32, about 45 min each. Calibrations:
31.3 µs per token at k=1, 48.9 at k=2, 89.5 at k=4.

Measured (`scripts/evaluation/instance_balance_summary.py`, originals only):
per-instance in-flight load and prompt tokens per 5-minute window, max-over-
mean prompt-token imbalance over windows with at least N unfinished tasks
(drain windows excluded, see §1), share of a task's requests served by its
home instance, JCT with the paired bootstrap over the 64 tasks. Decision
rule: multi-instance stays in scope if at N=8 the imbalance under FCFS
sticky exceeds 1.5 in at least a quarter of the counted windows, or the
DualMap-versus-sticky mean JCT gap at N=8 exceeds the 2-minute separability
bound. Otherwise multi-instance scheduling is closed as a non-problem at
realistic pressure and Milestones 2–3 become a characterization of the
over-capacity regime. Absolute N=8 latencies are not comparable to N=2
(shared memory bandwidth under MPS); the test is about routing behaviour.
Hardware stays at two GPUs through this step.

*N=2 capped result (read 17:45 UTC, before N=8).* FCFS sticky under the
cap: mean JCT 12.88 vs 8.80 min at full (+245 s paired, 95% [+207, +288]),
cached share 0.56 vs 0.96, per-request prefill 0.55 vs 0.08 s, in-engine
queue 5.8 vs 3.3 s, decode 7.9 vs 5.8 s: KV thrash. DualMap under the cap:
9.40 min, cached 0.94, queue 0.7 s, prefill 0.12 s, versus 9.11 min at full.
Its cost is a proxy-side hold of 2.7 s per request that is the same at full
and capped (a fixed throttle), which is why it is 19 s per task slower than
FCFS at full while 209 s faster under the cap. The LMCache CPU tier supplied
about 4% of prompt tokens (1.6M of 37M lookups, 240 retrieves of 39 ms);
the recovery is admission, not the DRAM tier.

*N=4 pair dropped (17:52 UTC).* Optional in the table above, no result read;
the decision rule reads N=8 only. Rerun only if N=8 is anomalous.

**Step 2 result (read 19:01 UTC; rule as pre-registered, N=4 pair dropped).**
Six runs on pool64-distinct-v4, 64/64 tasks and 1,951 original requests each:

| Run | Mean JCT (min) | P95 | Makespan | Cached share | Per-request hold / queue / prefill / decode (s) | Imbalance mean; windows > 1.5 |
|---|---:|---:|---:|---:|---|---|
| N=2 full, FCFS sticky | 8.80 | 15.45 | 26.2 | 0.96 | 0 / 3.3 / 0.08 / 5.8 | 1.05; 0 of 4 |
| N=2 full, DualMap | 9.11 | 15.87 | 24.7 | 0.95 | 2.7 / 0.7 / 0.10 / 6.3 | 1.11; 0 of 4 |
| N=2 capped, FCFS sticky | 12.88 | 26.54 | 37.2 | 0.56 | 0 / 5.8 / 0.55 / 7.9 | 1.41; 2 of 6 |
| N=2 capped, DualMap | 9.40 | 16.83 | 26.0 | 0.94 | 2.7 / 0.7 / 0.12 / 6.3 | 1.24; 1 of 4 |
| N=8 capped, FCFS sticky | 11.74 | 23.20 | 38.2 | 0.58 | 0 / 3.2 / 2.5 / 8.1 | 1.42; **1 of 4** |
| N=8 capped, DualMap | 10.49 | 22.00 | 26.4 | 0.79 | 1.4 / 1.3 / 1.3 / 7.6 | 1.65; 2 of 4 |

Paired mean-JCT differences (64 trajectories, 2,000 draws): DualMap − sticky
at N=8 **−74.6 s, 95% [−133, −16]**; at N=2 capped −209 s [−260, −163]; at
N=2 full +18.8 s [+2, +36]. N=8 sticky − N=2 sticky at equal capacity −69 s
[−116, −17]; N=8 DualMap − N=2 DualMap +65 s [+35, +97].

*Rule.* Prong 1 (sticky imbalance > 1.5 in at least a quarter of counted
windows at N=8): 1 of 4 five-minute windows, share 0.25, **met exactly at
the threshold**. Prong 2 (DualMap − sticky mean JCT gap at N=8 > 2 min):
1.24 min, **not met**. Verdict by the letter of the rule: multi-instance
stays in scope. Sensitivity, recorded and not part of the rule: with 120 s
windows the sticky share is 0.56 (9 windows), with 180 s 0.33 (6 windows);
the single 300 s window that crosses 1.5 is the last counted one, with 18
unfinished tasks and one instance already empty. The verdict therefore rests
on tail-phase stranding, the same effect as on mixed56 (§1), not on a
sustained steady-state imbalance: in the first three windows the share is 0.

*What the pair says beyond the rule.* (1) Routing recovers little mean JCT
at N=8 (75 s) but 11.7 min of makespan (26.4 vs 38.2), because sticky
strands instances in the drain: for a closed batch the tail is where
multi-instance placement matters. (2) DualMap at N=8 keeps only 0.79 cached
share against 0.94 at N=2 with the same total KV, moving 28% of requests off
their home instance; eight 65K caches fragment what two 262K caches hold, so
DualMap is 65 s per task slower at N=8 than at N=2. (3) FCFS sticky is 69 s
per task faster at N=8 than at N=2: 64 running slots against 16 halve the
in-engine queue (3.2 vs 5.8 s) even though each prefill is 4.5× slower under
a quarter of the SMs. The "N=8 not comparable" caveat above cut the other
way. (4) The mechanism finding from the N=2 pair (admission, not the DRAM
tier) holds at N=8: DualMap's CPU tier supplied 2.7M of about 37M prompt
tokens.

*Consequences.* The open-loop (Poisson) N=8 check in `PENDING.md` §4 is
not run: the rule's threshold is met, so it cannot change the verdict.
Whether a verdict carried by one drain window justifies keeping
multi-instance as a research problem, rather than as a tail-placement
detail, is the advisor's call and is listed in `PENDING.md` §8.

**Step 3 — pressure grid (pre-registered 17:56 UTC, before any grid
number; queue and criteria in `PENDING.md` §2).** N=2 capped at concurrency
48 and 64 (R_avg 1.77, 2.36), FCFS sticky and DualMap. Question: does
admission remain the sufficient mechanism at twice the pressure, or does
DualMap's fixed throttle fail. Decision: DualMap near the work-conserving
bound at R 2.4 closes scheduling under KV pressure on this workload;
degradation opens the task-level working-set admission direction
(`PENDING.md` §3). The earlier step 3 text (grid at N=8 or single-instance
mechanisms) is superseded by this; the N=8 verdict still decides whether
multi-instance stays in scope.

*Step 3 result (read 22:52 UTC; `results/chain11-grid-20260911.{sh,log}`).*
N=2 capped (262,144 tokens per instance), pool64-v4, closed loop with
replacement, 64/64 tasks and 1,951 requests in every run:

| Concurrency (R_avg) | Policy | Mean JCT (min) | P95 | Max | Makespan | Cached | Hold / queue / prefill / decode (s) | CPU-tier tokens |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 32 (1.18) | FCFS sticky | 12.88 | 26.5 | 36.6 | 37.2 | 0.56 | 0 / 5.8 / 0.55 / 7.9 | — |
| 32 (1.18) | DualMap | 9.40 | 16.8 | 25.3 | 26.0 | 0.94 | 2.7 / 0.7 / 0.12 / 6.3 | 1.6M |
| 48 (1.77) | FCFS sticky | 16.59 | 37.9 | 42.8 | 43.7 | 0.09 | 0 / 16.8 / 1.0 / 9.0 | — |
| 48 (1.77) | DualMap | 10.00 | 18.3 | 44.0 | 44.9 | 0.95 | 6.8 / 0.7 / 0.12 / 6.9 | 8.1M |
| 64 (2.36) | FCFS sticky | 17.84 | 47.7 | 55.6 | 56.8 | 0.07 | 0 / 23.4 / 1.0 / 7.7 | — |
| 64 (2.36) | DualMap | 9.93 | 17.8 | 25.1 | 26.3 | 0.90 | 8.1 / 0.7 / 0.17 / 6.3 | 7.9M |

Paired DualMap − sticky: −396 s [−504, −285] at c48, −475 s [−631, −330]
at c64. FCFS c64 − c48: +75 s [−10, +158].

*Against the pre-registration.* The FCFS half held: the cache collapses
(0.09, 0.07) and the in-engine queue grows to 17 and 23 s per request. The
DualMap half did not: the hold is not a fixed throttle, it grows with
pressure (2.7 → 6.8 → 8.1 s per request), the in-engine queue stays at 0.7 s
and the cache at ≥ 0.90 at every point, and mean JCT moves from 9.40 to
10.00 to 9.93 min, within 40 s of the c32 value at 2× the pressure. By the
decision criterion (JCT growth ≈ concurrency growth is the work-conserving
bound; DualMap is far under it, cached ≥ 0.9, queue < 1 s), **scheduling
under KV pressure is closed on this workload**: request-level admission with
a DRAM tier already achieves it. The CPU tier grows from 4% of prompt
tokens at c32 to about 22% at c48 and c64, so above R ≈ 1.5 the DRAM tier
is doing real work; that is the §5 32B question.

*Where DualMap does fail: the tail, once.* At c48 the largest-context trace
(install-windows-xp, 54 steps) was held at the proxy for 23.6 min in total
and finished at 44.0 min, against 25.3 min at c32 and 25.1 min at c64; three
other tasks were held over 10 min each. The throttle concentrates waiting on
the heaviest tasks (the rich-get-richer effect the 2026-09-07 meeting notes
§1 assumed), but it did not recur at c64 (max 25.1 min), so it is a
one-run observation, not a characterized failure mode.

*Consequence.* `PENDING.md` §3 (working-set admission) is not pursued: its
gate was DualMap degrading, and it did not. The remaining questions are the
design-space levers in `PENDING.md` §5 and §7 (model size, TP=2 vs DP=2,
memory hierarchy at 32B).

### 3.1 Model size: Qwen3-32B-FP8 (pre-registered `PENDING.md` §5, read 03:02 UTC)

Same pool64-v4, N=2 full memory (236,496 KV tokens per instance, R_avg 1.31
uncapped), concurrency 32, engine context 65,536 via YaRN factor 2 over the
model's native 40,960. Calibrated prefill constant 143 µs per token (4B:
31.3). Chain: `results/chain12-qwen32b-20260911.{sh,log}`; smoke
`fcfs-least-requests-smoke-qwen32b-20260911-r1`, calibration
`dualmap-calibration-pro6000-qwen32b-20260911-r1`.

| Run (64/64, 1,951 requests) | Mean JCT (min) | P95 | Max | Makespan | Cached | TPOT (ms) | Hold / queue / prefill / decode (s) | CPU-tier tokens |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 4B, FCFS sticky (R 0.50) | 8.80 | 15.5 | 25.6 | 26.2 | 0.96 | 26 | 0 / 3.3 / 0.08 / 5.8 | — |
| 4B, DualMap | 9.11 | 15.9 | 24.1 | 24.7 | 0.95 | 28 | 2.7 / 0.7 / 0.10 / 6.3 | — |
| 32B, FCFS sticky (R 1.31) | 57.08 | 107.5 | 125.3 | 125.9 | 0.26 | 156 | 0 / 33.8 / 3.8 / 34.9 | — |
| 32B, DualMap | 29.71 | 49.1 | 70.4 | 71.0 | 0.86 | 90 | 14.0 / 1.5 / 0.8 / 20.2 | 0.87M |

Paired DualMap − sticky at 32B: **−1,642 s per task, 95% [−1,878, −1,418]**.
Wall time: FCFS 2 h 6 min, DualMap 1 h 11 min.

*Against the pre-registration.* The pattern held and is much larger than
expected. FCFS: cached share 0.26, and the recomputed prefill (about 14K
tokens per request at 143 µs) not only costs 3.8 s of prefill but stalls
every co-batched decode under chunked prefill, so TPOT is 156 ms and decode
alone is 35 s per request; five preemptions. DualMap recovers the cache to
0.86 and halves TPOT, at a hold of 14 s per request (4B: 2.7 s).

*What is new at 32B.* (1) The residual pressure cost under DualMap is no
longer small: per step, hold 14 s against 22.5 s of engine time, so about
40% of step time is admission waiting (4B: 2.7 s against 7.1 s). The
work-conserving bound is far away; scheduling under KV pressure is *not*
closed at this model size. (2) The DRAM tier did almost nothing (2% of prompt
tokens, 66K evictions) because 48 GiB per instance holds only 192K tokens at
262 KB per token, less than the instance's own GPU cache; the tier is
mis-sized for the model, not useless. This is the `PENDING.md` §7.2 question
and it has a direct test: the same DualMap run with 96 GiB per instance
(cgroup 240 GB allows it). (3) Imbalance stays modest under both policies
(sticky mean 1.22, 3 of 22 windows above 1.5; DualMap 1.19, none), so at
this scale the multi-instance question is again secondary to KV pressure.

*DRAM tier at 96 GiB per instance (read 06:45 UTC,
`results/pool64v4-pro6000-qwen32b-n2full-dualmap-cpu96-20260912-r1`).*
Same DualMap run with the CPU tier doubled: mean JCT **23.35 min** (48 GiB:
29.71), P95 41.3 (49.1), makespan 53.8 (71.0), cached share 0.95 (0.86),
TPOT 77 ms (90); paired −382 s per task, 95% [−449, −318]. Per request:
hold 9.9 s (14.0), queue 1.6, prefill 0.34 (0.8), decode 17.2 (20.2). The
tier supplied 4.4M prompt tokens (0.87M at 48 GiB), sat full at 99 GB on
both instances and still evicted about 10K times, so capacity is binding and
not yet saturated: two 96 GiB tiers hold about 380K tokens each against a
demand of 32 contexts averaging 19K, growing. Reading: about a third of the
32B residual is DRAM capacity; the remaining hold, 9.9 s of a 29 s step, is
the admission logic. Imbalance unchanged (mean 1.21, 2 of 10 windows).

*Consequence.* The step 3 closure ("scheduling under KV pressure is closed on
this workload") is scoped to the 4B model. At 32B the closed question
reopens with a measured headroom of roughly 40% of step time, and the first
lever to test is memory-hierarchy sizing (DRAM tier), not a new scheduler.

## 4. Evidence

- Smoke and calibration: `../results/fcfs-least-requests-smoke-20260911-r2/`,
  `../results/dualmap-calibration-pro6000-20260911-r2/` (`prefill-calibration.json`)
- Bridge runs: `../results/mixed56-pro6000-dualmap-20260911-r1/`,
  `../results/mixed56-pro6000-fcfs-sticky-20260911-r1/` (`comparison.json`;
  LMCache counters in `server/instance-*/lmcache-metrics-final.prom`)
- Chain script and log: `../results/chain6-pro6000-20260911.{sh,log}`
- Trace pool: `../traces/exports/{swe-rebench-original-flat-644,terminal-bench-original-flat-239}-20260904/`
  with `MANIFEST.jsonl`
