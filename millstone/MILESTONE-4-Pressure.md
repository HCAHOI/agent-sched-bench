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

Question (pre-registered in the queue before the run): is DualMap's remaining wait an admission necessity, or
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
Pressure axis (chain 22 after the host restart, pre-registered as chain 21): at concurrency 24 the mean working set
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
| c20 | 144 GiB | **46.19** | **0.95** | 9.1K | 27.8 | −1,332 s [−1,418, −1,243] vs c20/96; −457 s [−544, −371] vs c16/96; ran alone; rule confirmed |

Readings. (1) **Capacity, not policy.** The trace-driven tier simulation (`scripts/evaluation/dram_tier_simulation.py`)
shows an oracle that knows every task's next arrival evicts the same contexts as LRU at every capacity, because tool
gaps are p50 0.7 s / p90 2.2 s (5% ≥ 5 s) against LLM steps of 31–75 s: no idle context is idle long enough for the
victim choice to matter. A return-time-aware DRAM policy has no leverage on this trace pool; it would need workloads
with long tool gaps. (2) **Not dispatch order either.** The FIFO gate held in-flight prompt tokens at p50 228K
(FCFS: 408K) and the cached share did not move; what must fit in DRAM is every active task's context, in flight or
in a tool gap, because LMCache stores every prefilled chunk and HBM keeps only what is in flight. (3) **Sizing
rule.** DRAM tokens ≥ ≈ 1.4 × concurrency × mean context (17.9K at 32B, 256 KB per token): c16 needs ≈ 400K (96 GiB
= 393K, fits), c20 ≈ 500K (96 GiB thrashes at 0.74; 144 GiB = 590K fits at 0.95, chain 28, read 01:33 UTC 2026-09-14), c24 ≈ 600K (144 GiB fits). The 1.4 is
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

## 6. High per-stream decode regime (2026-09-14/15)

Section 3 measured one 32B dense engine at 8 decode slots, where decode is 93% of the work. On 2026-09-14 the
advisor redirected the work to the regime the field is moving to — 300–1,000 tokens per second per stream — and
the measurements below were taken on one RTX Pro 6000 (single-GPU host, vLLM 0.28.0). Raw curves, lane scripts
and logs: `analysis/results/tpot-curve-20260914/`.

- **Lane 4 DONE 06:41 UTC (user go "gogogo")**: Gemma 4 26B-A4B FP8 + DFlash k=15, KV cache fp8 vs bf16, both on
  TRITON_ATTN (`analysis/results/tpot-curve-20260914/g4-dflash15-triton-*`, `lane4.log`). TPOT median ms / aggregate
  tok/s / accepted per draft:

  | c | agent 16K, bf16 KV | agent 16K, fp8 KV | short 1.2K, bf16 KV | short 1.2K, fp8 KV |
  |---|---|---|---|---|
  | 1 | 4.4 / 136 / 3.3 | 4.2 / 156 / 3.5 | 3.5 / 244 / 2.6 | 3.8 / 250 / 2.4 |
  | 4 | 8.6 / 223 / 2.9 | 7.5 / 329 / 3.2 | 6.0 / 661 / 2.5 | 5.7 / 721 / 2.7 |
  | 8 | 28.5 / 207 / 2.9 | 10.4 / 465 / 3.0 | 6.8 / 918 / 2.5 | 6.6 / 1032 / 2.4 |
  | 16 | 59.8 / 209 / 2.8 | 10.3 / 1064 / 3.1 | 8.5 / 1310 / 2.4 | 7.6 / 1486 / 2.6 |
  | 32 | 97.8 / 215 / 3.0 | **13.9 / 1384 / 2.8** | 10.9 / 1376 / 2.4 | 9.9 / 1504 / 2.3 |

  Reading: with fp8 KV the agent-context curve stops collapsing: c=32 TPOT 13.9 ms (72 tok/s per stream) and 1,384
  tok/s aggregate, 7× and 6.4× over bf16 KV on the same backend, and within 8% of the short-prompt aggregate. The
  bf16 path's c=8→16 cliff (lane 2 reading 6) is therefore a kernel-path property of bf16 KV with Gemma's hybrid
  attention in vLLM 0.28, not the model. On this GPU the frontier point on 16K agent contexts is now **32 agents at
  72 tok/s each, 238 tok/s single-stream**. Acceptance unchanged (2.8–3.5 agent, 2.3–2.7 short).
- **Lane 3 DONE 06:22 UTC (user go "1吧")**: Qwen3-30B-A3B agent prompts, KV cache fp8 vs bf16, both on the TRITON_ATTN
  backend (fp8 KV needs FlashInfer for FlashAttention-class kernels; not installed; lane 1's bf16 curve used
  FlashAttention, so the pair below is the fair one). TPOT median ms / aggregate tok/s:

  | c | bf16 KV (Triton) | fp8 KV (Triton) |
  |---|---|---|
  | 1 | 6.9 / 88 | 6.1 / 95 |
  | 4 | 14.8 / 131 | 12.1 / 134 |
  | 8 | 25.6 / 207 | 19.6 / 210 |
  | 16 | 67.0 / 207 | 25.6 / 535 |
  | 32 | 86.9 / 291 | 40.7 / 641 |

  Reading (3) confirmed: halving the KV bytes halves TPOT at c ≥ 16 (86.9 → 40.7 ms, 2.1×) and doubles the
  aggregate; at c ≤ 8 the gain is 12–25% because expert-weight reads still dominate there. KV bytes per decode step
  are the lever for agent contexts; fp8 KV is the first free half. (Aggregate at c=16 is inflated by prefix hits on
  the reused samples; the TPOT column is the clean number.)
- **TPOT lanes 1+2 DONE (04:17–05:34 UTC)**. Setup: vLLM 0.28.0, CUDA graphs, greedy, 512 output tokens, bf16 KV,
  max-num-seqs 32, chunked prefill 8192. "agent" = real replay prefixes (mean 14–16K tokens); "short" = prefixes
  ≤ 6,000 chars (≈ 1.2K tokens, first steps); "decode-only" = every prefix pre-filled before the level. Cells are
  TPOT median ms / aggregate output tok/s (/ accepted tokens per draft):

  | c | 30B-A3B short | 30B-A3B agent | 30B-A3B agent decode-only | 32B agent | G4 short | G4 agent | G4+DFlash15 short | G4+DFlash15 agent | G4+DFlash8 agent |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | 6.0 / 151 | 6.5 / 55 | 6.5 / 143 | 27.4 / 31 | 5.4 / 169 | 6.2 / 100 | 3.5 / 244 / 2.6 | 4.4 / 127 / 3.3 | 4.6 / 134 / 3.2 |
  | 4 | 10.1 / 342 | 13.3 / 190 | 13.2 / 284 | 34.7 / 78 | 8.3 / 402 | 10.4 / 190 | 5.8 / 618 / 2.5 | 7.9 / 234 / 3.2 | 8.6 / 218 / 2.8 |
  | 8 | 13.0 / 512 | 21.9 / 265 | 26.9 / 244 | 77.3 / 81 | 10.2 / 545 | 16.6 / 278 | 7.1 / 948 / 2.5 | 25.1 / 212 / 3.1 | 30.4 / 206 / 2.8 |
  | 16 | 17.9 / 675 | 42.9 / 322 | 48.3 / 292 | 149.6 / 89 | 12.7 / 766 | 64.2 / 200 | 8.5 / 1339 / 2.5 | 62.0 / 204 / 3.1 | 56.8 / 214 / 2.9 |
  | 32 | 21.1 / 842 | 85.6 / 313 | 92.5 / 312 | 154.2 / 92 | 16.7 / 1005 | 131.0 / 192 | 11.1 / 1538 / 2.4 | 93.8 / 237 / 3.0 | 93.6 / 247 / 2.8 |

  (30B-A3B = Qwen3-30B-A3B-Instruct-2507-FP8, 48 global-attention layers, 98 KB KV/token; 32B = Qwen3-32B-FP8; G4 =
  RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic, 30 layers of which 5 global and 25 sliding-window 1024, draft
  z-lab/gemma-4-26B-A4B-it-DFlash.) Readings: (1) **The 300 tok/s per-stream regime is reproduced on this GPU**:
  Gemma 4 + DFlash 289 tok/s on short prompts, 228 on agent prompts (1.4× over its 160 baseline), 1,538 tok/s
  aggregate at c=32 on short prompts. (2) **Agent context length, not model speed, sets what a GPU can serve**: the
  same models fall to 200–320 aggregate tok/s at c ≥ 8 on 14–16K contexts (short prompts: 840–1,540); TPOT grows
  linearly with concurrency. (3) The growth is decode-side KV reading, not prefill interference: the decode-only
  control reproduces the curve (92.5 vs 85.6 ms at c=32). (4) Speculative decoding pays only at low concurrency on
  agent contexts (DFlash: 1.3–1.4× at c ≤ 4, a loss at c=8, even at c=16, +25% at c=32); draft length 8 vs 15 does
  not matter. (5) Acceptance is higher on later agent steps (3.0–3.3, tool calls copy from context) than on first
  steps (2.4–2.6): the recorded GPT-style history does not lower acceptance. (6) Gemma 4 has a cliff between c=8
  and c=16 on long contexts in vLLM 0.28 (base 16.6 → 64.2 ms) that its sliding-window design should not produce;
  not diagnosed. (7) EAGLE-3 (lmsys SpecForge draft for Qwen3-30B-A3B) accepted 1.0 tokens per draft here; not
  pursued. Caveat: prompts are GPT-recorded trajectories replayed into other models; the step/output distribution of
  Gemma-native trajectories may differ (§4.0).
- **TPOT lane 1 DONE 04:56 UTC** (`analysis/results/tpot-curve-20260914/`, client `scripts/evaluation/tpot_curve.py`;
  vLLM 0.28.0, CUDA graphs, greedy, 512 output tokens, real replay prefixes, mean prompt ≈ 14K tokens; "short" =
  prefixes ≤ 6,000 chars ≈ 1.1K tokens):

  | c | 30B-A3B short: TPOT ms / agg tok/s | 30B-A3B agent 14K: TPOT / agg / TTFT | 32B agent 14K: TPOT / agg / TTFT |
  |---|---|---|---|
  | 1 | 6.0 / 151 | 6.5 / 55 / 2.5 s | 27.4 / 31 / 2.0 s |
  | 4 | 10.1 / 342 | 13.3 / 190 / 0.5 s | 34.7 / 78 / 1.6 s |
  | 8 | 13.0 / 512 | 21.9 / 265 / 0.4 s | 77.3 / 81 / 4.1 s |
  | 16 | 17.9 / 675 | 42.9 / 322 / 0.7 s | 149.6 / 89 / 13.7 s |
  | 32 | 21.1 / 842 | 85.6 / 313 / 2.0 s | 154.2 / 92 / 62.9 s |

  Readings: (1) the MoE model is 4.2× faster per stream than 32B and its aggregate saturates at ≈ 320 tok/s on agent
  prompts but reaches 842 on short prompts at the same concurrency: **agent context length, not model speed, sets
  the per-GPU capacity** (4× TPOT, 2.7× aggregate at c=32). (2) 32B saturates at 90 tok/s aggregate; past c=8 the
  TTFT is prefill queueing (63 s at c=32). (3) EAGLE-3 with the lmsys SpecForge draft on vLLM 0.28: acceptance
  length 1.0 (checkpoint has d2t/t2d; cause not found; not pursued, DFlash/MTP are the frontier). TTFT at c=1 is
  polluted by JIT warm-up; levels reuse leading samples so TTFT at c ≥ 4 has prefix hits.
- **Chain 29 STOPPED 04:14 UTC by the user** (`pool64v4-pro6000-qwen32b-gpu0-c24-fcfs-sticky-tier80-exclusive-20260914-r1`,
  95 min of ≈ 3 h, no verdict on the §4.1 rule). Interim diagnosis at 70 min (same elapsed windows): requests finished
  569/757/911 at 30/50/70 min vs inclusive-96 529/706/869 and 144 856/1387/1861; cached share of original steps
  0.49/0.37/0.31 vs 0.39/0.29/0.23 vs 0.86/0.89/0.90. The mechanism works as coded (copy-back only for the evicted
  part, 1.8% stall) but the "+236K" estimate in M4 §3.3 was wrong: with `--max-num-seqs 8` only ≈ 140K tokens are in
  flight (HBM usage mean 0.67), so exclusive-80 ≈ inclusive-115 GiB, still under the 600K the rule asks at c24.
  Corrected rule: DRAM + (in-flight tokens) ≥ 1.4 × c × context. Recorded, not to be continued (§4.0).
- Hosts, in order: the 2× Pro 6000 box (port 41548) was released on 2026-09-14; a single-GPU box (port 35803, driver
  580.95, cgroup 110 GiB / 22 cores) carried the lanes of 2026-09-14; the current one is
  `ssh -p 36715 root@connect.singapore-a.gpuhub.com`, 1× RTX Pro 6000 Blackwell 96 GB, driver 595, cgroup 120 GiB /
  208 cores, about 50 GB free on `/root/autodl-tmp`. Each move carried `/workspace` over, and each changes the DRAM
  envelope: the 144 GiB tiers of §3.3 were measured under a 240 GB cgroup and cannot start on either single-GPU box.
  Python supervisord has to be started by hand after every restart.

### 6.1 What the regime changes

   on 2026-09-14 (§3): Gemma 4 26B-A4B FP8 + DFlash + fp8 KV gives 289 tok/s single-stream on short prompts, 228 on
   agent prompts, 1,384 tok/s aggregate at c=32 on 16K contexts. The DRAM-capacity line is closed as a sizing result,
   not a paper. What changes in this regime, measured: an agent step is 2–7 s instead of 30–50 s, and because decode
   is amortised over a much larger batch while prefill is not, **prefill stops being a rounding error**. Per step
   (pool64-v4 measured: prompt 18,514 tokens, uncached 944 at cached-share 0.95, output 209 tokens; prefill rates
   from c=1 TTFT, ±30%):

   | Serving configuration (decode batch) | warm prefill | decode GPU time | prefill share | cold prefill | full-context KV transfer |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-32B dense, bf16 KV, batch 8 (the pool64-v4 runs) | 146 ms | 2,069 ms | 7% | 2,767 ms | 323 ms |
   | Qwen3-30B-A3B, fp8 KV, batch 32 | 69 ms | 266 ms | 21% | 1,311 ms | 124 ms |
   | Gemma 4 26B-A4B + DFlash, fp8 KV, batch 32 | 49 ms | 91 ms | 35% | 932 ms | 25 ms |

   Share = warm prefill / (warm prefill + output × TPOT / batch). Counting the cold steps at cached share 0.95
   (5 cold + 95 warm per 100 steps) the totals are 12% prefill at 32B/batch 8 and 51% at Gemma/batch 32. Inference,
   not yet a result: agent serving becomes prefill-bound at high per-stream speed even at a 95% cache hit rate, which
   is the condition industrial PD serving is designed for.

### 6.2 PD and PPD at this operating point

   H200 questions were answered by the simulation in §4.5 and M3 §3 — read that first, this item only adds
   the frontier-regime arithmetic it explicitly does not cover). Frontier C closed PD on
   2 × L40S at 4B (fixed PD 71.5 min, public PPD 109.2, two-sided 75.0, DualMap 33.2; M3 §3). Three things are now
   quantified that were not then. (i) Both rules we ran are blind to residency, in two different ways. The published
   PPD rule decides from the turn number, the tokens appended this turn (a 512-token short-input threshold keeps
   small appends local on D), a predicted output length and the current QPS, against a lookup table from its own
   offline benchmark; on agent traffic almost every append classified as `huge_paste` and the table said local at the
   QPS points reached, so it degenerated to always-local (2,414 of 2,470 requests, 109.2 min). None of its features
   is the variable our measurements say sets the cost: two turns with the same append size and QPS differ by 18K
   tokens of real prefill work depending on whether the task's history is still resident on the decode engine. The
   two-sided expected-cost router is ours, not the paper's (`ppd_policy.two_sided_estimate`), and fails the other
   way: it priced only the requesting turn's TTFT, so when P was busy it sent cold requests back to D — 12.4M
   uncached tokens prefilled on D over 973 "local" requests, 12,700 each. A residency rule (miss → P, hit → D, no
   cost comparison) closes both holes.
   (ii) But its value is bounded by the table in §4.0: at cached share 0.95 the cold steps are half of all prefill
   work, so moving only them off the decode GPU frees **6%** of its time at 32B/batch 8 and **25%** at Gemma/batch 32.
   6% does not buy a second GPU; the sized DRAM tier already removed 95% of the cold prefill that PD would have
   taken away. (iii) For multi-turn agents PD pays either 2× KV memory (history kept on both P and D, which is what
   heterogeneous hardware with a memory-rich decode tier would buy) or 19× prefill (944 → 17,935 uncached tokens per
   step when P starts cold); the KV transfer itself is cheap on modern attention (25 ms for Gemma's 16K context).
   **Pre-registered decision rule, 17:30 UTC 2026-09-15, before the sweep's numbers exist:** from the sweep take the
   largest feasible concurrency B\* and its TPOT, and compute f = cold prefill / (cold prefill + warm prefill +
   output × TPOT / B\*) with the same per-step constants and the run's measured cached share. f ≥ 20% → one 2-GPU
   test of cold-only disaggregation against colocated at B\* is justified (rent, ≈ 4 h); f < 20% → PD/PPD is closed
   for agent workloads at every scale we can reach, and is not raised again. Lit check owed before any novelty claim:
   Mooncake, MemServe, Splitwise heterogeneous, SGLang cache-aware router.

Residency routing was simulated on top of this and the verdict retracted, because the simulator is not
validated for routing policies: `analysis/results/pd-pool-sim-20260915/residency-routing.md`.

## 7. What every run measures

The two policies are instruments, not the object of study. FCFS sticky is
the system as deployed with no help: each task pinned to one engine, vLLM's
own queue, no admission control, no DRAM tier. DualMap is the strongest
existing fix we can run: request-level admission at the proxy, a DRAM KV
tier, and migration. At any operating point (instances N, concurrency c =
number of tasks active at once in the closed loop, model size) two numbers
answer one question: the gap FCFS → DualMap says whether pressure is a
problem that an existing mechanism can fix; the residual DualMap → ideal
(engine time with no hold and no queue) says how much is left to research.

| Operating point | FCFS → DualMap gap | DualMap residual | Reading |
|---|---|---|---|
| 4B, full memory (R 0.5) | none | none | no problem |
| 4B, capped, c32–c64 (R 1.2–2.4) | large | small (hold 3–8 s of a 7 s step) | problem exists, solved by admission |
| 4B, N=8 | small (mean), large (makespan) | small | tail stranding only |
| 32B, full memory (R 1.3) | huge (57 → 30 min) | **large (hold 14 s of a 36 s step)** | problem exists, **not solved**: research goes here |

Chain 13 asks what the 32B residual is made of (DRAM-tier capacity, or the
admission logic) and whether the operating point itself is right (TP=2).
