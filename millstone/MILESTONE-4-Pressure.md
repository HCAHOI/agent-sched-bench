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

*Amendment before launch.* The first configuration (N=8 at 50K tokens per
instance, 400K total) is physically ill-posed: vLLM refuses a KV cache that
cannot hold one maximum-length request, and a replica that small could not
serve a 90K-token context in any deployment either. The k=4 smoke failed on
exactly that check. Revised design: the workload's peak context is bounded
at 60K (`pool64-distinct-v2`: same sampler and seed on the 799 pool traces
with peak ≤ 60K; 45 swe-rebench + 19 terminal-bench, 1,998 steps, mean
prompt 18,024 tokens per step, max peak 55,550), the engine context limit is
65,536 tokens, and every capped instance holds the same total of 524,288
tokens split evenly, so N=8 has exactly one full context per instance:

| Configuration | Per instance | Total KV tokens | R_avg | R_peak | Policies |
|---|---:|---:|---:|---:|---|
| N=2 full | 624,880 | 1,249,760 | 0.46 | 0.66 | FCFS sticky, DualMap |
| N=2 capped | 262,144 | 524,288 | 1.10 | 1.58 | FCFS sticky, DualMap |
| N=8 capped (k=4) | 65,536 | 524,288 | 1.10 | 1.58 | FCFS sticky, DualMap |
| N=4 capped (k=2) | 131,072 | 524,288 | 1.10 | 1.58 | FCFS sticky, DualMap (last, optional) |

The capped rows sit at the L40S pair's pressure for this workload (550K
tokens, R_avg 1.05), so N is the only thing that varies across them; N=2
full is the low-pressure reference. DualMap is calibrated separately at k=2
and k=4 (its prefill constant changes under the SM share) and its CPU tier
stays 96 GB in total (48, 24, 12 GiB per instance). Eight runs on
pool64-distinct-v2 at concurrency 32, about 45 min each.

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

**Step 3 — the study step 2 selects.** If multi-instance stays: the
pressure grid at N=8, R_avg ∈ {0.8, 1.5, 2.5} by concurrency from the same
pool, with FCFS sticky, least-requests and DualMap; the four-GPU question
(N=4 independent as a cross-check) is decided then. If dropped:
single-instance work on the pool, one run per GPU in parallel, the same R
grid on one instance, FCFS against the single-instance mechanisms
Milestone 1 left open (prefill priority, tool-gap-aware KV retention),
judged on mean JCT at each R.

## 4. Evidence

- Smoke and calibration: `../results/fcfs-least-requests-smoke-20260911-r2/`,
  `../results/dualmap-calibration-pro6000-20260911-r2/` (`prefill-calibration.json`)
- Bridge runs: `../results/mixed56-pro6000-dualmap-20260911-r1/`,
  `../results/mixed56-pro6000-fcfs-sticky-20260911-r1/` (`comparison.json`;
  LMCache counters in `server/instance-*/lmcache-metrics-final.prom`)
- Chain script and log: `../results/chain6-pro6000-20260911.{sh,log}`
- Trace pool: `../traces/exports/{swe-rebench-original-flat-644,terminal-bench-original-flat-239}-20260904/`
  with `MANIFEST.jsonl`
