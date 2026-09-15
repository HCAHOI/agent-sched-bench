# Pending: experiment queue and open decisions

Current as of 2026-09-14 08:56 UTC. Rewritten, not appended: this file says
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
- **Output-length prediction** — prompt-side point prediction remains closed. Full-observation repair is complete;
  subsequent cached-feature exploration selects log-remaining MLP + closing-token signals + EWMA 0.7 on validation.
  Development-test first-alert-within-256 improves 36.30 → 44.83%, mean offline waste 30.93 → 24.64 s, but lateness
  increases 5.77 → 6.25% (48 → 52 requests). The original history/single remains the fixed reference. Ridge,
  eight-position remaining/progress, and signals-only candidates fail validation's lateness/waste constraints.
  Long reasoning still has only 21.05% first-alert hits. Layer-24 full extraction, coverage audit and evaluation
  are complete: 4,215 requests, zero errors after the connection-reuse repair, ~102 min and 8.9 GB.
  Layer 24 loses on validation (within-256 26.97% vs retained 43.68%) and development test (33.41% vs 44.83%,
  waste 28.093 vs 24.639 s; late 6.49% vs 6.25%). More early alerts explain the loss despite slightly better
  position recall. Retain the final-layer candidate; stop layer expansion after this negative primary.
  Reasoning-process diagnosis now covers all 419 validation traces, with 24 fixed qualitative examples and
  96 final-FFN readout observations. Excluding reasoning <=256, validation hits are 100/336 (29.76%). Early
  alerts include both repeated action elaboration and later-revised solutions. At leads 256/64, the final FFN
  suppresses closing in all 24 cases; at lead 1 the residual-only readout already gives median P(close)=99.26%.
  P2.1 newline control completed: same earlier prose gives median P(close) 99.77% with one newline versus
  4.01e-15 with two; the terminal readout is strongly format-dependent in this model. Stop this format branch;
  reflection-state inspection is now conditional on its value for forecasting. Complete-history GRU plus a
  matched request-batch MLP comparison is finished: GRU lowers target MSE but loses first-alert accuracy
  (validation 31.50% vs incumbent 43.68%; development test 34.38% vs 44.83%). Original MLP reproduces exactly.
  Retain incumbent. A real Gemma+DFlash k16 boundary comparison is now complete on 32 fixed TPOT replay tasks
  (99.10 s, no truncations/errors); intentional replay instructions and full histories are retained. On 25 valid local
  tool requests, assumed 1.8 s restore gives target vs first-draft lateness 88% vs 80%, blocked 1.281 vs 1.084 s,
  idle 0.157 vs 0.218 s. It fails the frozen no-idle-increase gate. Only 6 of 14 advance alerts have the actual boundary
  within the next 15 tokens (30–104 ms lead); 15 requests close reasoning at their first output token.
  Next analyze sandbox need and time until tool readiness using these captured trajectories. Full OUTLETS and independent
  real-restore confirmation remain conditional; pooling and this DFlash experiment service are stopped.
  Unused 4B pretrained weights removed (5.19 GB); tokenizer/config and experiment artifacts retained. New Gemma outputs
  are observational only; no generated tool execution or restore-policy integration. Protocols, paired intervals and limitations:
  `analysis/development/output-length-prediction-handoff.md`; frozen outputs: `reasoning-explore-20260914/`.
  Literature synthesis, current mechanism evidence and ordered Phase 2 TODOs: [REASONING.md](../REASONING.md).
- **Sandbox interface** — three signals + prefill bound, tables and restore-policy scores in the handoff; restore p50 1.0 s / p90 1.8 s.
- **Baselines** — DualMap/CacheWise/ThunderAgent family starves under saturation (M4 §3.3 (4)); PD/PPD closed (M3).
- **Related work** — M4 §5 (hidden-state tool prediction; agent KV offloading: MORI, CacheWise, TokenCake, Continuum).

## 3. Queue

- Reasoning full-observation follow-up DONE: feature supplement, frozen comparisons, overlap sensitivity,
  full-validation recalibration and full-data current/history refits are complete. The dedicated pooling server
  is stopped; no further predictor, generation or TPOT run is queued. All TPOT lanes below had completed by 06:41 UTC.
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
- New host: `ssh -p 35803 root@connect.singapore-a.gpuhub.com`, 1× Pro 6000, driver 580.95, cgroup 110 GiB / 22 cores,
  disk 33 GB free; the whole old `/workspace` arrived by cloud transfer (models, venvs, outlen incl. the 50 GB OUTLETS
  caches, 69 launch logs, manifests). Python supervisord started by hand; source shipped at d038f86a.

## 4. Next

0. **Direction (user, 04:05 UTC 2026-09-14): the frontier is per-stream decode at 300–1000 tok/s.** Reproduced here
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

1. **RUNNING: where the TPOT service level binds** (`results/host-lanes/tpot5-sla-20260915.sh`, launched 17:27 UTC
   2026-09-15, user go "跑吧"; host log `/workspace/tpot5-sla-20260915.log`, results `/workspace/tpot-20260915/`).
   Gemma 4 + fp8 KV on TRITON_ATTN with `--max-num-seqs` raised to 128, concurrency 16/32/64/96/128 on real agent
   prefixes, with DFlash k=15 and without speculative decoding. The client now samples the engine per level
   (running/waiting queue, KV usage, preemptions) so the two candidate limits are separable. ≈ 35 min.
   **Pre-registered 17:30 UTC 2026-09-15, before any number exists.** Readout: for each service level in
   {25, 50, 100, 200} ms, the largest concurrency whose median TPOT stays under it, and whether the engine reached
   that concurrency (running ≈ requested, waiting ≈ 0, no preemptions) or KV capacity stopped it first. Decision:
   (a) if at 50 ms and above the binding constraint is capacity at every level — TPOT still under the service level
   where the engine runs out of KV — then the service-level formulation reduces to capacity, "minimise latency
   subject to the SLA" becomes "fit the most agents", and the mechanism question returns to residency and admission;
   (b) if TPOT crosses 50 ms while KV headroom remains, an SLA-aware admission controller that predicts TPOT from the
   running batch's KV bytes (the relation measured in lane 3: halving KV bytes halves TPOT at c ≥ 16) is worth
   building. Secondary: at the largest feasible concurrency, if the no-speculation aggregate is within 5% of
   DFlash's, speculative decoding is a single-stream latency tool, not a capacity tool, at this operating point.

2. **PD/PPD re-evaluated at the frontier operating point** (analysis 2026-09-15, no run; the pool-scale and
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

3. **Residency routing in the pool simulator — pre-registered 2026-09-15 17:43 UTC, before any comparison is read.**
   `pd_pool_simulation.py --pool residency:M,P` adds the rule neither the published PPD decision engine nor our
   two-sided cost router implements: a step is disaggregated iff its history is not resident on the engine that would
   decode it; the prefill tier is stateless and pushes KV to the task's home engine; warm steps prefill their delta
   locally. Plumbing smoke (not evidence): 2× L40S, mixed56, 20.5% of steps disaggregated, runs clean.
   Primary comparison: the H200 profile at 8 GPUs with 32 decode slots, the only point in this simulator where
   residency exists (97% of prompt tokens cached), `residency:6,2` against the recorded `mixed:8` 8.0 min,
   `pd:4,4` 9.2 min and `twosided:6,2` 8.7 min (mean JCT ready-to-terminal, same seed and workload).
   Criterion: ≤ 7.6 min (5% better than colocated) → routing cold steps to a prefill tier is worth building and the
   2-GPU test at the frontier operating point is justified; within ±5% of 8.0 → once residency exists there is
   nothing left for disaggregation to remove, which closes the design; worse than 8.4 → the transfer and the prefill
   tier's queue cost more than the cold prefill they take away. Secondary, reported either way: share of steps
   disaggregated, prefill-tier utilisation, token-weighted TPOT. Same criterion re-applied on the frontier profile
   (Gemma 4 + fp8 KV constants) once the SLA sweep supplies its decode curve.

4. **Confirmatory test of residency routing under memory pressure — pre-registered 2026-09-15 17:47 UTC.**
   The pre-registered primary in §4.3 was run and **failed**: on the H200 profile at 8 GPUs (97% of prompt tokens
   resident) `residency:6,2` gives 8.5 min against colocated 8.0, `residency:7,1` 8.2, `twosided:6,2` 8.7,
   `pd:4,4` 9.2; only 2.3–2.5% of steps are cold, the prefill tier idles at 3–7%, so dedicating even one GPU of
   eight to it costs more than the cold prefill it removes. That is the "nothing left to remove" branch. One
   implementation defect was found and fixed first (a task's first step is never resident, so the home engine's
   queue was empty when the home was chosen and every task was homed on the same engine; homes are now balanced on
   the count of tasks homed per engine).
   **Unregistered secondary observation, therefore exploratory:** on the memory-constrained L40S profile at 8 GPUs,
   `residency:7,1` gives 45.0 min against colocated 57.4 and the best fixed PD (5:3) 54.3, and the home engines'
   cached share rises from 0.067 to 0.50 — taking cold prefills off the decode engines stops them thrashing their
   own KV, which raises residency, which makes fewer steps cold. At 2 GPUs the same rule is far worse (95.3 against
   55.7): the tier must be a small, well-used share of the fleet (1 of 8 at 76% busy, not 1 of 2 at 31%).
   Confirmatory design, fixed now: L40S profile, pool sizes 8 and 32, residency at 1/8 of the fleet (7:1, 28:4) and
   at 1/4 (6:2, 30:2), seeds 0/1/2, against colocated (57.4 at 8, 57.9 at 32) and the best fixed PD ratio (54.3,
   55.7). Criterion: the best residency ratio beats colocated by ≥ 10% at both pool sizes on all three seeds → the
   effect is real and a 2-GPU measured test at the frontier operating point is justified; 0–10% → report as
   marginal and do not build; a win at one pool size only → size-dependent, report and stop. Mechanism check
   reported either way: the home engines' cached share must rise against colocated, otherwise the win is not the
   claimed mechanism. Limits carried from §4.5: the engine model is an eager-mode 4B engine, so this says nothing
   about the KV-read-bound frontier regime until the frontier profile exists.
   **Result, read 17:58 UTC: criterion met at both pool sizes.** 8 GPUs 46.3 against colocated 57.5 (−19.5%,
   worst seed −17.6%), 32 GPUs 39.6 against 58.0 (−31.7%), and 15–29% ahead of the best fixed-PD ratio; the
   mechanism check passes (home cached share 0.068 → 0.471 and 0.629, TPOT 129 → 94 and 82 ms). The tier must
   be 1 GPU in 8 to 16: 2 in 8 loses to colocated (59.5) and 1 in 2 is far worse (95.3). Table, readings and
   limits: `analysis/results/pd-pool-sim-20260915/residency-routing.md`. Next, in order: fit the frontier
   profile from the SLA sweep's no-speculation curve and re-run this comparison there; then the 2-GPU
   measured test the criterion justifies.

5. Parked, with the record in §3: exclusive tiering (built, `scripts/serving/exclusive_tier/sitecustomize.py`; chain
   29 stopped at 95 min — the gain is the in-flight tokens only, ≈ 140K at `--max-num-seqs` 8, not the 236K of a full
   HBM, so exclusive-80 ≈ inclusive-115 GiB); sizing rule as online task admission (Σ contexts ≤ DRAM/1.4).

6. Push branch `codex/cleanup-research-dead-code` (≈ 150 commits ahead of origin).

5. **PD at pool scale, by simulation (user go "先做1-3吧", 2026-09-15).** The boss's questions — PD at 8 or 32
   instances, part of the traffic disaggregated, a 141 GB GPU — cannot be run on one GPU.
   `scripts/evaluation/pd_pool_simulation.py` replays the mixed56 workload (56 tasks, 2,470 steps, prompt and
   completion tokens per step from the FCFS run's replay logs, inter-step gaps median 0.02 s, the harness's 32-slot
   closed-loop admission) through a per-iteration model of vLLM's scheduler (max_num_seqs 8, 2,048-token chunks,
   275,008-token KV with LRU prefix cache keyed by task, newest-request preemption). Facts corrected on the way: the
   2×L40S mixed56 runs used **Qwen3-4B-Instruct-2507-FP8**, eager mode, 8 sequences per engine (not 32B); FCFS
   sticky's prefix-cache hits were 8% of prompt tokens (system prompt only) against DualMap's 76%.
   Calibration (all from the 2026-09-07/09 2×L40S runs): decode iteration 34.6 ms + 0.38 ms per sequence + 0.054 ms
   per thousand tokens of batch context (D side of the fixed-PD run, 3,997 requests, r² 0.42); prefill 0.040 ms per
   token × (1 + position/20K) from P's aggregate throughput (101.7M tokens, saturated); mixed engines pay 1.4× that
   prefill (fitted on FCFS sticky's TPOT/JCT). Held out: the PPD run (3,902 of 4,015 requests local on one engine).
   **Gate, written 2026-09-15 16:03 UTC as the sweep launched, sweep outputs unread:** mean JCT (ready-to-terminal)
   within ±15% and token-weighted TPOT within ±20% on all three measured runs, cached share within 5 points.
   Result: FCFS 55.7 vs 55.3 min, TPOT 131 vs 142 ms, cached 5.2 vs 8.1%; fixed PD 68.6 vs 71.5 min, 42.1 vs 41.8 ms;
   PPD (held out) 114.4 vs 109.2 min, 134.6 vs 134 ms. Passed; amendment: the mixed-prefill factor was fitted on
   FCFS after seeing that the sum-of-parts model ran 20% fast, so FCFS is a calibration run, not a check.
   Predictions (`analysis/results/pd-pool-sim-20260915/sweep/`): mixed:N vs pd:P,D at N = 8 and 32, hybrid task
   splits, H200 constants (capacity 3.3×, bandwidth 5.6×, prefill compute 2.7×, NVLink transfer; assumptions, read
   as a band), capacity-only and bandwidth-only sensitivities. Limits: the decode model is an eager-mode 4B engine
   (fixed 35 ms per iteration dominates), so nothing here speaks to the KV-read-bound 32B regime or to spec decode;
   DualMap is not simulated; the workload has no tool time. Next step if any prediction matters: one rented
   multi-GPU day on the predicted best P:D ratio.

## 5. Decisions waiting on the user

- Exclusive tiering primary run (§4.1): go, and which T the new box allows.
- Whether the cloud transfer restored `/workspace/outlen` and the venvs (else rebuild via the bootstrap; outlen data
  from the 120 MB partial backup in `results/gpuhub-host-backup-20260914/` plus git).
- Push the branch.
