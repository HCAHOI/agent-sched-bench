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

### 3.2 Storage versus admission, one engine (read 15:00 UTC; `results/chain17-gpu0-storage-20260912.{sh,log}`)

Question (PENDING §8b): is DualMap's remaining wait an admission necessity, or
does a right-sized DRAM store make admission unnecessary? Qwen3-4B, one engine
on GPU 0 capped to 262,144 KV tokens, concurrency 16 (R_avg ≈ 1.2), pool64-v4,
64/64 tasks and 1,951 requests in every run. The 12 GiB tier (≈ 85K tokens) is
smaller than the concurrent working set (16 tasks × ≈ 20K), reproducing the 32B
ratio; 48 GiB (≈ 340K tokens) is larger than it.

| Storage tier | Admission | Mean JCT (min) | P95 | Makespan | Cached | TPOT (ms) | Hold / queue / prefill / decode (s) | Tokens served from the tier |
|---|---|---:|---:|---:|---:|---:|---|---:|
| none | none (FCFS sticky) | 25.99 | 42.9 | 55.7 | 0.48 | 40.5 | 0 / — / — / — | — |
| 12 GiB (undersized) | none | 29.68 | 50.3 | 62.3 | 0.41 | 46.9 | 0 / 8.7 / 0.8 / 10.5 | **0** (190K evictions) |
| 12 GiB (undersized) | DualMap | 22.89 | 37.7 | 52.1 | 0.78 | 36.2 | 5.9 / 0.4 / 0.4 / 8.1 | 512 |
| 48 GiB (≥ working set) | none | **20.10** | **32.7** | **42.1** | **0.88** | **31.6** | 0 / 4.6 / 0.2 / 7.1 | 18.2M (49% of prompt tokens; 1,147 retrieves, 48 ms each) |

Paired mean-JCT differences: undersized tier − nothing +222 s [+192, +252];
DualMap + undersized tier − nothing −186 s (from 22.89 vs 25.99; bootstrap not
run on that pair); right-sized tier − nothing −353 s [−395, −310]; right-sized
tier − DualMap + undersized tier −167 s [−203, −132].

Reading. (1) A tier smaller than the working set is pure cost: every request
pays the synchronous store and nothing is ever retrieved before eviction.
This is the 32B situation (48 GiB = 192K tokens against a 600K working set,
2% served). (2) Admission works even with a useless tier: holding requests at
the proxy keeps fewer contexts resident and the GPU cache stops thrashing
(0.41 → 0.78). (3) A tier sized to the working set without any admission beats
admission with an undersized tier by 167 s per task and reaches 0.88 cached
share with a 4.6 s in-engine queue: the store, sized right, does more than the
throttle. (4) The last cell (read 15:47 UTC, chain 18): **DualMap + 48 GiB, 16.91
min**, P95 28.7, makespan 37.5, cached 0.93, TPOT 28.6 ms, hold 3.5 s, queue
0.4 s; −192 s per task against the right-sized tier alone, 95% [−219, −164].
With the store sized, admission still buys 16%, and the tier is consulted far
less (3.7M tokens served against 18.2M) because held requests keep their
contexts on the GPU. Decomposition on this operating point: store sized to the
working set −353 s, admission on top of it −192 s, admission alone −186 s, an
undersized store +222 s. Both mechanisms matter and they are not additive:
the store removes recompute, admission removes cache contention among the
tasks that remain resident.

**32B confirmation (read 17:45 UTC; `results/chain19-gpu0-storage-20260912.{sh,log}`, chain 20 for the
undersized column).** Same protocol at the platform model: Qwen3-32B-FP8, one engine on GPU 0 (GPU KV 236,496
tokens), concurrency 16, working set 16 × 17.9K ≈ 287K tokens mean (540K at the p90 step); 96 GiB ≈ 393K tokens
is sized to it, 24 GiB ≈ 98K tokens covers the mean shortfall and not the peaks. 64/64 tasks in every run.

| Storage tier | Admission | Mean JCT (min) | P95 | Makespan | Cached | TPOT (ms) | Hold / queue / prefill / decode (s) | Tokens served from the tier |
|---|---|---:|---:|---:|---:|---:|---|---:|
| none | none (FCFS sticky) | 111.69 | 189.4 | 211.4 | 0.18 | 170.8 | 0 / 39.7 / 4.0 / 36.2 | — |
| 96 GiB (≥ working set) | none | **53.81** | **93.9** | **107.2** | **0.92** | **93.3** | 0 / 17.7 / 0.6 / 19.6 | 34.1M (75% of prompt tokens; 1,946 retrieves, 93 ms each; 13,346 evictions) |
| 96 GiB | DualMap (r2; r1 aborted at 99% by a replacement-stream harness failure) | **45.80** | **78.4** | **91.4** | **0.95** | **81.7** | 15.1 / 1.2 / 0.4 / 17.7 | 4.9M (11%; 378 retrieves, 68 ms; 8,614 evictions) |
| 24 GiB (undersized) | none | 113.67 | 193.9 | 217.1 | 0.18 | 175.1 | 0 / 40.7 / 4.1 / 37.1 | **0.1M** (0%; 4 retrieves; 44.0M stored, 170,546 evictions) |
| 24 GiB (undersized) | DualMap | 57.83 | 93.9 | 120.9 | 0.85 | 90.8 | 17.6 / 0.9 / 0.9 / 18.8 | 0.1M (0%; 21 retrieves; 37.1M stored, 143,563 evictions) |

Paired mean-JCT differences: sized tier − nothing **−3,473 s [−3,832, −3,093]** per task; undersized tier − nothing
**+119 s [+92, +145]** (read 23:21 UTC, chain 20: 44M tokens stored and evicted, 4 retrieves, cached share unchanged at
0.18, so the 4B reading (1) "a tier smaller than the working set is pure cost" holds at 32B, at a smaller cost
because the 32B store call is a smaller share of a 82 s request); DualMap + undersized tier − nothing **−3,232 s
[−3,618, −2,845]** (read 01:25 UTC): admission alone, with a store that serves nothing, recovers almost the whole
gain (57.83 min against 53.81 for the sized store alone), by holding requests at the proxy (17.6 s per request) until
the resident contexts fit the GPU cache (cached share 0.85, TPOT 91 ms, in-engine queue 0.9 s). The 4B reading (3) holds
with a larger margin at 32B: the sized store alone halves the step (queue 39.7 → 17.7 s, decode 36.2 → 19.6 s) because
the GPU cache stops thrashing (cached share 0.18 → 0.92, TPOT 171 → 93 ms), and prefill nearly disappears (4.0 →
0.6 s). DualMap + sized tier − sized tier alone: **−481 s [−615, −346]** (read 03:12 UTC); DualMap + sized tier − nothing
−3,954 s [−4,391, −3,512]. Decomposition at 32B, mirroring the 4B one: store sized to the working set −3,473 s,
admission on top of it −481 s (a further 15%; 4B: 16%), admission alone −3,232 s, store on top of admission −721 s,
an undersized store +119 s. Used alone the two mechanisms are near-substitutes (both remove the same GPU-cache
thrash) and the store wins by 4 min per task without holding anyone; together they are not additive, and the
combined cell holds each request 15.1 s at the proxy to buy its last 8 min per task. With the store sized, the tier
is consulted far less (4.9M tokens served against 34.1M) because held requests keep their contexts on the GPU.
Pressure axis (chain 22 after the host restart; PENDING §8b "Chain 21"): at concurrency 24 the mean working set
(430K tokens) exceeds the 96 GiB tier alone and the p90-step peaks (810K) exceed tier + GPU. FCFS + 96 GiB at c24
(read 09:27 UTC 2026-09-13): **99.11 min** mean JCT (c16: 53.81), P95 183.9, makespan 198.8, cached share 0.44
(0.92), TPOT 141 ms (93), queue 70.1 s / prefill 3.0 s / decode 31.8 s per request, tier served 18.1M tokens (39% of
prompt tokens; 106,532 evictions); throughput fell to 15.1 steps/min from 23.6 at c16, so the extra concurrency is
lost to cache thrash rather than converted into work. The sized store alone does not carry the pressure once the
working set exceeds it. DualMap + 96 GiB at c24: the first attempt aborted at 97% because DualMap's admission starved one request for the
full 1,800 s step timeout (hold p99 954 s, max 1,782 s over 1,898 dispatched requests; the c16 run's max was 1,778 s);
the rerun with a 3,600 s step timeout (chain 23) failed the same way: one request never dispatched in 88 min
while the others flowed (hold p50 0 s, p90 10 s, max 1,895 s). Three of five DualMap runs at 32B single-engine lost a
request to the scheduler's waiting pool and the two that completed had max holds of ~1,780 s. **DualMap + 96 GiB at
c24: does not complete the workload (indefinite starvation of one request per run).** The pressure-axis comparison
therefore reads: the sized store alone degrades gracefully to 99 min; the official admission scheduler starves.
Our own bounded FIFO admission (arrival order under an in-flight prompt-token budget of 236K, no residency term;
chain 25, read 16:57 UTC 2026-09-13) completes the workload with no wait above 147 s but recovers only 3%: 96.15 min,
cached share 0.44 as without it, the same ~105K tier evictions. The pressure at c24 is store capacity — contexts in
tool gaps are evicted before they return — not dispatch order.

**Capacity confirms it (chain 24b, read 19:20 UTC 2026-09-13).** FCFS + **144 GiB** DRAM at c24 (590K tokens; this
container's cgroup allows ≈ 150 GiB, not the 192 GiB first planned): **51.80 min** mean JCT, P95 93.7, makespan
106.5, cached share **0.91**, TPOT 83 ms, 25.9 steps/min; paired −2,838 s [−3,106, −2,582] against the same
concurrency with 96 GiB, and −120 s [−262, +21] against c16 with 96 GiB — the per-task step time of c16 at 50% more
concurrency, i.e. 11% more throughput. The DRAM tier served 89% of prompt tokens (2,449 retrieves, 91 ms each;
16,248 evictions against 105,684 at 96 GiB). Sizing rule, pre-registered before this number: the DRAM tier must hold
concurrency × mean context (c24 × 17.9K = 430K tokens > 393K at 96 GiB, < 590K at 144 GiB). A trace-driven
simulation of the tier (`scripts/evaluation/dram_tier_simulation.py`) shows the same loss is not an eviction-order
problem: an oracle that knows every task's next arrival equals LRU at every capacity, because tool gaps are p50
0.7 s / p90 2.2 s against LLM steps of 31–75 s; the tier is a capacity, not a policy, problem on this workload.
The knee sits above the mean-context rule: at c20 the same 96 GiB tier already loses part of its hit rate (chain 27,
read 21:55 UTC: 68.38 min, cached 0.74, 45.6K evictions, +874 s [+776, +972] against c16; run concurrently with
chain 26 on the other GPU, so its JCT may include some CPU contention), so the DRAM tier needs ≈ 1.4× the
concurrency × mean-context product (the p90 steps, the replacement stream and chunk granularity). The c24 capacity
curve (read 23:52 UTC): 48 GiB → 124.29 min, cached 0.03, 176K evictions; 96 GiB → 99.11, 0.44; 144 GiB → 51.80,
0.91. Below the working set the tier is worse than none: every prefill pays the store and nothing survives to be
retrieved.
The cause is in the code: DualMap's waiting pool is ordered by cached-prefix length first and arrival time second,
with no aging, so under sustained load a request without a cached prefix (a task's first step) can wait behind a
never-empty stream of cache-affine requests. A one-line aging rule would remove it; that variant would be a modified
baseline and is not run without the advisor's decision.

### 3.3 Pressure axis and DRAM sizing (chains 19–28, 2026-09-12/14; naming: HBM = GPU KV, DRAM = LMCache CPU tier)

One 32B engine (HBM KV 236,496 tokens), FCFS sticky, pool64-v4, 64/64 tasks in every run. The question after §3.2:
what happens when the working set outgrows the sized DRAM tier, and is the fix capacity, dispatch order, or eviction
policy?

| Concurrency | DRAM tier | Mean JCT (min) | Cached share | DRAM evictions | Steps/min | Note |
|---|---|---:|---:|---:|---:|---|
| c16 | 96 GiB | 53.81 | 0.92 | ~13K | 23.6 | reference (§3.2) |
| c16 | 96 GiB + DualMap | 45.80 | 0.95 | 8.6K | 26.0 | admission adds 15% at 15.1 s hold/request |
| c20 | 96 GiB | 68.38 | 0.74 | 45.6K | 19.7 | +874 s [+776, +972] vs c16; tier starts thrashing (concurrent with another run) |
| c24 | 48 GiB | 124.29 | 0.03 | 176K | 10.9 | worse than no tier: every prefill pays the store, nothing survives |
| c24 | 96 GiB | 99.11 | 0.44 | 106K | 15.1 | +2,718 s [+2,432, +3,035] vs c16 |
| c24 | 96 GiB + FIFO working-set admission (236K in-flight tokens) | 96.15 | 0.44 | 106K | 14.0 | −177 s [−205, −151] vs c24/96: dispatch order is not the lever |
| c24 | 96 GiB + DualMap | does not complete | — | — | — | one request starved per run (three attempts; hold p99 954 s, max 1,895 s) |
| c24 | 144 GiB | **51.80** | **0.91** | 16K | 25.9 | −120 s [−262, +21] vs c16: the c16 step time at 50% more load |
| c20 | 144 GiB | (chain 28, pending) | | | | second test of the sizing rule |

Readings. (1) **Capacity, not policy.** The trace-driven tier simulation (`scripts/evaluation/dram_tier_simulation.py`)
shows an oracle that knows every task's next arrival evicts the same contexts as LRU at every capacity, because tool
gaps are p50 0.7 s / p90 2.2 s (5% ≥ 5 s) against LLM steps of 31–75 s: no idle context is idle long enough for the
victim choice to matter. A return-time-aware DRAM policy has no leverage on this trace pool; it would need workloads
with long tool gaps. (2) **Not dispatch order either.** The FIFO gate held in-flight prompt tokens at p50 228K
(FCFS: 408K) and the cached share did not move; what must fit in DRAM is every active task's context, in flight or
in a tool gap, because LMCache stores every prefilled chunk and HBM keeps only what is in flight. (3) **Sizing
rule.** DRAM tokens ≥ ≈ 1.4 × concurrency × mean context (17.9K at 32B, 256 KB per token): c16 needs ≈ 400K (96 GiB
= 393K, fits), c20 ≈ 500K (96 GiB thrashes, 144 GiB = 590K should fit), c24 ≈ 600K (144 GiB fits). The 1.4 is
headroom for p90 steps (33.8K), the replacement stream and 256-token chunk granularity; the plain product already
thrashed at c20. (4) **Residency-first admission starves.** DualMap's waiting pool is ordered by cached-prefix
length with no aging (`double_hash_global_scheduler_utils.py`); CacheWise's released policy is the same family
(fewest new KV blocks first). Starvation and their TPOT gain are the same mechanism (M2 §5, M3 Frontier A), so no
aging variant was run. (5) **Practical ceiling.** The DRAM a rental container can pin is its cgroup limit (240 GB
here, 192 GiB failed, 144 fit), so the rule also says what a box can serve: a 96 GiB tier ≈ 16 agents at 32B, an
80 GiB tier ≈ 13, or ≈ 35 with the 30B-A3B model (98 KB per token).

Next mechanism worth building (not started): **exclusive tiering** — LMCache keeps in-flight contexts in both HBM
and DRAM; storing a context to DRAM only when it leaves HBM would add ≈ 236K tokens of effective DRAM at 32B (+60%
on a 96 GiB tier), enough for c24 by the rule, with no extra memory. The fork (vLLM 0.10.2) has no such connector;
about a day of code, evaluated on the c24 / 96 GiB cell against the 144 GiB reference.

## 4. Evidence

- Smoke and calibration: `../results/fcfs-least-requests-smoke-20260911-r2/`,
  `../results/dualmap-calibration-pro6000-20260911-r2/` (`prefill-calibration.json`)
- Bridge runs: `../results/mixed56-pro6000-dualmap-20260911-r1/`,
  `../results/mixed56-pro6000-fcfs-sticky-20260911-r1/` (`comparison.json`;
  LMCache counters in `server/instance-*/lmcache-metrics-final.prom`)
- Chain script and log: `../results/chain6-pro6000-20260911.{sh,log}`
- Trace pool: `../traces/exports/{swe-rebench-original-flat-644,terminal-bench-original-flat-239}-20260904/`
  with `MANIFEST.jsonl`

## 5. Related work checked (2026-09-13, web + arXiv, recorded in `PENDING.md` at the time)

- Hidden-state prediction for agents predicts categorical targets only — tool identity (SPORK 2607.03333, Speculate-While-You-Reason 2607.25816, Linearly Readable 2605.07990), tool necessity (When2Tool 2605.09252), call errors (2608.27750), parameter correctness (ParamBench 2608.03071), program state (KTH 2607.05188). Output length from hidden states exists only on ≤ 2K-token chat prompts (OUTLETS 2609.01068, ProD 2604.07931, EGTP/PLP 2602.11812), all MAE-only.
- KV offloading for agents: MORI (2606.00866, HBM + DRAM by trailing idleness, Claude Code / SWE-bench Pro), CacheWise (2606.16824, time-to-reuse predictor for in-tier eviction), TokenCake (2510.18586, offload by predicted call duration, DRAM assumed unbounded), Continuum (2511.02230, HBM TTL), ThunderAgent (2602.13692, HBM only), KVFlow / CacheScout / PBKV (workflow DAGs), CachedAttention (chat turns), Agentix (batched swaps). None reports the store-vs-admission decomposition at a sized tier, the starvation mechanism of residency-first admission, or a sandbox-side interface; §3.3 reading (1) says a return-time DRAM policy would not pay on this pool anyway.

