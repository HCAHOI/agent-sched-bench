# Pending: experiment queue and open decisions

Current as of 2026-09-14 05:40 UTC. Rewritten, not appended: this file says
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

- Host until 2026-09-14: gpuhub/AutoDL container, 2× RTX Pro 6000 (96 GB HBM each); being replaced by a single-GPU
  instance (~110 GB DRAM). The launcher accepts one GPU with `--single-gpu 0` since 2026-09-14.
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
  simulation: capacity not policy). Chain 28 confirmed the rule at c20 (46.19 min, cached 0.95).
- **Output-length prediction** — closed as a prompt-side hidden-state limit; the full ladder and the sampling ceiling are in
  `analysis/development/output-length-prediction-handoff.md`; the hazard (P(remaining ≤ X)) probes are the kept dynamic
  signal for thinking steps.
- **Sandbox interface** — three signals + prefill bound, tables and restore-policy scores in the handoff; restore p50 1.0 s / p90 1.8 s.
- **Baselines** — DualMap/CacheWise/ThunderAgent family starves under saturation (M4 §3.3 (4)); PD/PPD closed (M3).
- **Related work** — M4 §5 (hidden-state tool prediction; agent KV offloading: MORI, CacheWise, TokenCake, Continuum).

## 3. Queue

- Nothing queued. GPU idle since 05:34 UTC 2026-09-14. Lane 2 done (`analysis/results/tpot-curve-20260914/`, `lane2.log`).
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
- New host: `ssh -p 35803 root@connect.singapore-a.gpuhub.com`, 1× Pro 6000, driver 580.95, cgroup 110 GiB / 22 cores,
  disk 33 GB free; the whole old `/workspace` arrived by cloud transfer (models, venvs, outlen incl. the 50 GB OUTLETS
  caches, 69 launch logs, manifests). Python supervisord started by hand; source shipped at d038f86a.

## 4. Next (proposals; each needs the user's go before any GPU time)

0. **Direction (user, 04:05 UTC 2026-09-14): the frontier is per-stream decode at 300–1000 tok/s** (DeepSeek V4.1
   Flash API ≈ 214 tok/s; Gemma 4 26B-A4B + DFlash 306 tok/s single-stream on H100, 1,957 tok/s aggregate at c16,
   vLLM PR #41703; Qwen with MTP drafts). At that speed an agent step is 1–3 s, prefill of the 17.9K-token context
   (2.6 s at 32B) becomes the larger part of the step, tool time goes from 50% of the trace to 85–95%, and the
   sandbox restore (p50 1.0 s) is the same size as the LLM step. The DRAM-capacity line (§4.1–4.2 below) is closed
   as a result, not a paper: it reduces to sizing. First step = reproduce the regime here (queue), then measure where
   the time goes in an agent step at that speed. DeepSeek V4.1 Flash (552B total, MXFP4) does not fit one 96 GB GPU;
   Gemma 4 26B-A4B (bf16 52 GB) + DFlash/MTP fits but needs the vLLM PR build and a 52 GB download.

1. **Exclusive tiering** — BUILT 2026-09-14, not yet smoked or run; needs the user's go on the
   single-GPU instance. Mechanism (`scripts/serving/exclusive_tier/sitecustomize.py`, `EXCLUSIVE_TIER=1` in the
   launcher, flag recorded in the run's `lmcache.yaml`): the DRAM tier evicts what HBM already holds first. Chunks of
   an in-flight request (just loaded or just stored) go to the evict-first end of LMCache's LRU; at finish vLLM holds
   the blocks one more step (delayed free) while the worker copies back only the part of the context that was
   evicted and promotes the whole context to most recently used. Tool-gap contexts keep plain LRU order; lookups,
   loads, chunking and capacity are unchanged. Costs charged: the finish copy-back (each event logged
   `[exclusive-tier] finish req= tokens= present= stored=`; store D2H measured 36.8 GB/s mean in the c24/96 run, so a
   full 17.9K-token context is ≈ 0.13 s), one step of block hold per request, and preempted requests (24 of 1,951 at
   c24/96) losing their evict-first chunks.
   Pre-registered 2026-09-14 02:15 UTC (no exclusive-tier numbers exist): the mechanism question is whether double
   occupancy is the term that separates c24/96 from c24/144. Primary case = c24, FCFS sticky, one 32B engine,
   pool64-v4, DRAM tier T GiB with exclusive tiering, T = 96 if the instance's cgroup allows engine + 96 GiB (else the
   largest of 80/64 that fits, measured on arrival), ≈ 2–3.5 h. Prediction from the tier simulation
   (`analysis/results/dram-tier-simulation-20260913/`, HBM counted once): miss share at the 0.033 floor, so cached
   share ≈ 0.9 and JCT near the 144 GiB reference (51.80 min). Rule: mean JCT ≤ 60 min and cached ≥ 0.85 → the
   mechanism replaces 48 GiB of DRAM at c24 and the sizing rule becomes DRAM + HBM ≥ 1.4 × c × context; ≥ 85 min →
   double occupancy was not the binding term (diagnose with the copy-back log and `lmcache` eviction counters
   before anything else); between → partial, report both terms. Controls: inclusive c24/96 = 99.11 min (cached
   0.44) and c24/144 = 51.80 (0.91), both measured on the previous 2-GPU host of the same GPU model; if T < 96 the
   inclusive-at-T control does not exist and exclusive-at-T is compared against inclusive-at-96 (a win at less DRAM
   is the stronger statement; a loss is inconclusive and the inclusive-at-T control is the next run). Same-host
   confound: one inclusive control rerun on the new box only if the exclusive result lands within ±5 min of a rule
   boundary. Smoke first (`--smoke --single-gpu 0` with the 4B model, `EXCLUSIVE_TIER=1 CPU_CACHE_GIB=8`): engine log
   shows the patch active and finish lines, KV usage returns to zero after the smoke (no leaked delayed frees).
2. **Sizing rule as online admission**: bound active tasks so Σ contexts ≤ DRAM/1.4 (task-level FIFO, no starvation);
   differs from ThunderAgent/KAIROS program admission by the DRAM criterion. Half a day; run at c24/96.
3. Instance switch: a single-GPU box with ~110 GB DRAM serves ≈ 13 agents at 32B (80 GiB tier) or ≈ 35 with
   30B-A3B; choose the concurrency by the rule. Backup taken 2026-09-14 01:45 UTC into
   `results/gpuhub-host-backup-20260914/` (untracked, local): `outlen/` (datasets, every label set incl. thinking and
   sampled, replay prefixes, small feature caches, hazard features, all probe/OUTLETS results and models, lane scripts
   and logs; the 50 GB OUTLETS per-token caches are excluded, ≈ 3 h to re-extract), `host-logs/` (launch logs,
   manifests), `venvs-and-upstreams.tar` (serving venv with the LMCache sm_120 build, the vLLM 0.28 venv, outlen venv,
   upstreams). Not saved: models (re-download ≈ 97 GB), host copies of run dirs (every run's `server/` is already
   pulled locally). Restore = untar under `/workspace` on a host with the same layout, or rerun the bootstrap.
4. Push branch `codex/cleanup-research-dead-code` (≈ 150 commits ahead).

## 5. Decisions waiting on the user

- Exclusive tiering primary run (§4.1): go, and which T the new box allows.
- Whether the cloud transfer restored `/workspace/outlen` and the venvs (else rebuild via the bootstrap; outlen data
  from the 120 MB partial backup in `results/gpuhub-host-backup-20260914/` plus git).
- Push the branch.
