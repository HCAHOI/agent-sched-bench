# Pending: experiment queue and open decisions

Current as of 2026-09-12 16:30 UTC. Rewritten, not appended: this file says
what is queued, why, and what each result decides. Records of finished work
live in the milestone files; this file only points at them.

## 0. What every run measures, and why always FCFS sticky and DualMap

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

## 1. Platform and standing constraints (2026-09-14)

- Host: gpuhub/AutoDL container, 2× RTX Pro 6000 (96 GB HBM each), `ssh -p 41548 root@connect.singapore-a.gpuhub.com`.
  The container cgroup caps memory at 240 GB (the console's 1 TB is the physical host): a pinned DRAM tier of 144 GiB
  starts, 192 GiB does not. The image overwrites `/usr/bin/supervisord` with a Go binary at every restart; after a
  restart run `python3 -m supervisor.supervisord -c /etc/supervisor/supervisord.conf` or every launcher exits silently.
- Platform = one engine per GPU (multi-instance closed, M4 §3). Naming: HBM = GPU KV, DRAM = LMCache CPU tier.
- Two checkouts on the host: `/workspace/agent-sched-bench` (launchers) and `/workspace/agent-sched-bench-b`
  (`--remote-repo`, ship here while a launcher runs). Two single-GPU runs share the host with `--port-base 100
  --tunnel-port 19100` (proxy port fixed 2026-09-13). Output-length lane lives in `/workspace/outlen/` (repo copy, datasets,
  feature caches ≈ 55 GB, results); models ≈ 97 GB under `/workspace/.hf_home`.
- Rules that cost us runs: no GPU run without the user's literal go naming it; `pkill -f` patterns never share a
  shell with text that matches them; results under `results/` are never deleted; runs > 30 min are pre-registered
  in this file before their numbers exist.

## 2. State of each line (the records hold the numbers)

- **Scheduling / KV pressure at 32B** — M4 §3.2 (store vs admission 2×2 at 4B and 32B) and §3.3 (pressure axis,
  DRAM capacity curve, sizing rule ≈ 1.4 × concurrency × mean context, FIFO gate −3%, DualMap starvation, tier
  simulation: capacity not policy). Pending: chain 28 (c20 + 144 GiB, second rule test) result.
- **Output-length prediction** — closed as a prompt-side hidden-state limit; the full ladder and the sampling ceiling are in
  `analysis/development/output-length-prediction-handoff.md`; the hazard (P(remaining ≤ X)) probes are the kept dynamic
  signal for thinking steps.
- **Sandbox interface** — three signals + prefill bound, tables and restore-policy scores in the handoff; restore p50 1.0 s / p90 1.8 s.
- **Baselines** — DualMap/CacheWise/ThunderAgent family starves under saturation (M4 §3.3 (4)); PD/PPD closed (M3).
- **Related work** — M4 §5 (hidden-state tool prediction; agent KV offloading: MORI, CacheWise, TokenCake, Continuum).

## 3. Queue

- GPU 1: chain 28, c20 + 144 GiB DRAM (`results/chain28-gpu1-32b-c20-tier144-20260913.sh`), started 23:51 UTC 2026-09-13,
  pre-registered prediction: cached ≥ 0.85 and mean JCT ≤ 59 min. When it lands: record in M4 §3.3 table, then both GPUs idle.
- GPU 0: idle since 23:50 UTC. Nothing queued.

## 4. Next (proposals; each needs the user's go before any GPU time)

1. **Exclusive tiering** (M4 §3.3 last paragraph): store a context to DRAM only when it leaves HBM, so in-flight
   contexts stop occupying both. Expected +236K effective DRAM tokens at 32B (+60% on 96 GiB), enough for c24 by the
   rule. Design: a connector/LMCache patch in the 0.10.2 fork (`save_kv_layer` currently stores on every prefill;
   `local_cpu_backend` LRU untouched); ≈ 1 day. Pre-registration to write before the run: c24 + 96 GiB exclusive vs the
   inclusive 99.11 min and the 144 GiB reference 51.80; rule: mean JCT ≤ 60 min with cached ≥ 0.85 → the mechanism
   replaces 48 GiB of DRAM; ≥ 85 → the double occupancy was not the binding term.
2. **Sizing rule as online admission**: bound active tasks so Σ contexts ≤ DRAM/1.4 (task-level FIFO, no starvation);
   differs from ThunderAgent/KAIROS program admission by the DRAM criterion. Half a day; run at c24/96.
3. Instance switch: a single-GPU box with ~110 GB DRAM serves ≈ 13 agents at 32B (80 GiB tier) or ≈ 35 with
   30B-A3B; choose the concurrency by the rule. Save first: `/workspace/outlen` feature caches (≈ 55 GB, hours to
   rebuild), models (≈ 97 GB, ~1 h to re-download). Results are all local and committed.
4. Push branch `codex/cleanup-research-dead-code` (≈ 150 commits ahead).

## 5. Decisions waiting on the user

- Exclusive tiering: build or not.
- When to switch instances (after chain 28 both GPUs are free).
- Push the branch.
