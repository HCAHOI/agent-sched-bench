# Pending: experiment queue and open decisions

Current as of 2026-09-15 19:07 UTC. Rewritten, not appended: this file says
what is queued, why, and what each result decides. Records of finished work
live in the milestone files; this file only points at them.

## 1. Platform and standing constraints (2026-09-15)

- Host: `ssh -p 36715 root@connect.singapore-a.gpuhub.com`, one RTX Pro 6000 Blackwell (96 GB HBM), driver 595,
  container cgroup 120 GiB memory and 208 cores, ≈ 50 GB free on `/root/autodl-tmp`. The launcher takes one GPU with
  `--single-gpu 0`. The 120 GiB cgroup caps the DRAM tier: nothing at 144 GiB or above will start here.
  The image overwrites `/usr/bin/supervisord` with a Go binary at every restart; after a restart run
  `python3 -m supervisor.supervisord -c /etc/supervisor/supervisord.conf` or every launcher exits silently.
- Platform = one engine per GPU (multi-instance closed, M4 §3). Naming: HBM = GPU KV, DRAM = LMCache CPU tier.
- Two checkouts on the host: `/workspace/agent-sched-bench` (launchers) and `/workspace/agent-sched-bench-b`
  (`--remote-repo`, ship here while a launcher runs); models ≈ 97 GB under `/workspace/.hf_home`. The TPOT client
  reads its replay prefixes from `/workspace/outlen/replay-prefixes/prefixes.jsonl`, so that directory is in use by
  the running lane and is not free disk.
- Rules that cost us runs: no GPU run without the user's literal go naming it; `pkill -f` patterns never share a
  shell with text that matches them; results under `results/` are never deleted; runs > 30 min are pre-registered
  in this file before their numbers exist.

## 2. State of each line (the records hold the numbers)

- **Scheduling / KV pressure at 32B** — M4 §3.2 (store vs admission 2×2) and §3.3 (pressure axis, DRAM capacity
  curve, sizing rule, FIFO gate, DualMap starvation, tier simulation).
- **High per-stream decode regime** — M4 §6; raw curves, lane scripts and logs in
  `analysis/results/tpot-curve-20260914/`.
- **PD / PPD routing** — M3 §3, under withdrawal (audit: `analysis/results/ppd-run-audit-20260915.md`).
- **PD at pool scale, by simulation** — M3 §3; its residency-routing reading is retracted
  (`analysis/results/pd-pool-sim-20260915/residency-routing.md`).
- **Baselines** — DualMap/CacheWise/ThunderAgent family starves under saturation: M4 §3.3 (4).
- **Related work** — M4 §5 (hidden-state tool prediction; agent KV offloading: MORI, CacheWise, TokenCake, Continuum).

## 3. Queue

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
   run with the router's real information, ±10%
   (`analysis/results/pd-pool-sim-20260915/residency-routing.md`).
3. Strip the simulator to mixed / PD / PPD and implement the published PPD rule from the pinned upstream
   (`ppd/optimizer/ppd_decision_engine.py`: 512-token short-input threshold, context class, nearest QPS point,
   offline lookup table) instead of our variants.
4. Parked: exclusive tiering (built; chain 29 stopped, M4 §6), sizing rule as online task admission.
5. PD at pool scale by simulation is finished and recorded in M3 §3.
   **Gate, written 2026-09-15 16:03 UTC as the sweep launched, sweep outputs unread:** mean JCT (ready-to-terminal)
   within ±15% and token-weighted TPOT within ±20% on all three measured runs, cached share within 5 points.
   Amendment: the mixed-prefill factor was fitted on FCFS after seeing that the sum-of-parts model ran 20% fast, so
   FCFS is a calibration run, not a check.
6. Push branch `codex/cleanup-research-dead-code` (≈ 150 commits ahead of origin).

## 5. Decisions waiting on the user

- Whether to fund a rented multi-GPU PD/PPD test at all. M4 §6.2 pre-registers the trigger (f ≥ 20% computed from
  the running sweep, ≈ 4 h of rental); the audit adds a precondition, that any rented test must first show the
  decode side holding conversations at a cached share well above the level the audit measured
  (`analysis/results/ppd-run-audit-20260915.md`).
- Whether to spend the work validating the pool simulator for routing (§4.2) before its predictions are quoted
  again (§4.1), or to drop the simulator line instead.
- Push the branch (§4.6).
