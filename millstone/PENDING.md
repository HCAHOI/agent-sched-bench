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
- **RUNNING since 17:27 UTC 2026-09-15** (user go "跑吧"): where the TPOT service level binds.
  `results/host-lanes/tpot5-sla-20260915.sh`, host log `/workspace/tpot5-sla-20260915.log`, results
  `/workspace/tpot-20260915/`. Gemma 4 26B-A4B FP8 + fp8 KV, TRITON_ATTN, `--max-num-seqs` 128, concurrency
  16/32/64/96/128 on real agent prefixes, with DFlash k=15 and without speculation. Decides: the largest
  concurrency under each service level in {25, 50, 100, 200} ms, whether latency or KV capacity binds first, and
  whether speculative decoding pays at the frontier concurrency. Pre-registration and readings: M4 §6.
- Nothing else queued.

## 4. Next (one line each; the records hold the reasoning)

1. Fit the frontier engine profile from the sweep's no-speculation curve and re-run the pool comparison there; the
   existing simulation is an eager-mode 4B engine and says nothing about this regime (M4 §6.2).
2. Make the pool simulator answer routing questions or stop quoting it for them: reproduce the measured two-sided
   run (75.0 min) with the router's real information, ±10%
   (`analysis/results/pd-pool-sim-20260915/residency-routing.md`).
3. Strip the simulator to mixed / PD / PPD and implement the published PPD rule from the pinned upstream
   (`ppd/optimizer/ppd_decision_engine.py`: 512-token short-input threshold, context class, nearest QPS point,
   offline lookup table) instead of our variants.
4. Parked: exclusive tiering (built; chain 29 stopped, M4 §6), sizing rule as online task admission.
5. Push branch `codex/cleanup-research-dead-code` (≈ 150 commits ahead of origin).

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
