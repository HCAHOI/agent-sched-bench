# Pending: experiment queue and open decisions

Current as of 2026-09-16 08:24 UTC. Rewritten, not appended: this file says
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
- **PD / PPD routing** — M3 §3, where the public-PPD number is withdrawn and the audit that withdrew it is recorded.
  The code now carries two paradigms only: `ROUTER_POLICY=pd` (classic, every step P then D) and `ROUTER_POLICY=ppd`
  (the upstream decision engine, unchanged). Our two-sided router, state-aware guard and extended lookup table were
  deleted on 2026-09-16; every PD-family run writes the decode engine's prefix-cache share to `mechanism-check.json`.
- **PD at pool scale, by simulation** — M3 §3; its residency-routing reading is retracted
  (`analysis/results/pd-pool-sim-20260915/residency-routing.md`). The simulator now offers `mixed`, `pd` and `ppd`
  layouts only.
- **Baselines** — DualMap/CacheWise/ThunderAgent family starves under saturation: M4 §3.3 (4).
- **Related work** — M4 §5 (hidden-state tool prediction; agent KV offloading: MORI, CacheWise, TokenCake, Continuum).

## 3. Queue

- Empty. The TPOT service-level sweep of 2026-09-15 finished; its pre-registration and readings are in M4 §6.
  The GPU host is idle.

## 4. Next (one line each; the records hold the reasoning)

1. Fit the frontier engine profile from the sweep's no-speculation curve and re-run the pool comparison there; the
   existing simulation is an eager-mode 4B engine and says nothing about this regime (M4 §6.2).
2. Make the pool simulator answer routing questions or stop quoting it for them: reproduce the measured two-sided
   run with the router's real information, ±10%
   (`analysis/results/pd-pool-sim-20260915/residency-routing.md`).
3. Done 2026-09-16: the simulator carries `mixed`, `pd` and `ppd` only. Its `ppd` layout reproduces the decisions
   the published engine actually made on this workload (turn one disaggregated, every later step local). Open: the
   engine's lookup-table clause is measured hardware data the simulator does not carry, so a workload where the
   table disagrees with the bypass would need the table itself.
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
 (recorded in M3 §3).
- Whether to spend the work validating the pool simulator for routing (§4.2) before its predictions are quoted
  again (§4.1), or to drop the simulator line instead.
- Push the branch (§4.6).
