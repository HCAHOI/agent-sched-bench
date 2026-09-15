# Next session: context-filling prompt

Paste this to a fresh agent session. It carries no goal and no budget; the
task follows separately. Update the host line whenever the machine changes.

---

You are joining an MLSys research project on scheduling LLM serving for
multi-step coding agents. Read these first, in order, before doing anything:

1. `CLAUDE.md` at the repo root (working rules; §3 integrity rules are
   non-negotiable).
2. `millstone/MILESTONE-2-Multi-Instance.md`: the current state of the
   research on 2× L40S: what was tried, what won, what failed and why. Every
   number you need is there; treat it as the record.
3. `millstone/MILESTONE-3-Frontiers.md`: the three frontiers opened after
   Milestone 2 (DualMap fairness, cross-instance load balance, PD/PPD
   routing). Read its §0 before comparing anything to a baseline. §3 is
   closed for fixed disaggregation and for our two-sided expected-cost
   router; the public-PPD number there is **withdrawn**, because that run
   made its routing decisions into an empty prefix cache (cached prompt
   share 0.013) and so never exercised the published mechanism. §3 carries
   the audit against the upstream design that establishes the withdrawal.
4. `millstone/MILESTONE-4-Pressure.md`. §3 closed multi-instance scheduling
   (one engine per GPU is the platform); §3.2 is the store-versus-admission
   decomposition at a sized DRAM tier; §3.3 the pressure axis, the DRAM
   capacity curve and the sizing rule (≈ 1.4 × concurrency × mean context),
   why eviction policy and dispatch order do not help on this pool, and why
   residency-first admission starves; §5 the related work checked. **§6, the
   high per-stream decode regime (300–1,000 tokens/s per stream, measured
   2026-09-14/15 on a single-GPU host with vLLM 0.28.0), is the current
   stage**: §6.1 why prefill stops being a rounding error there, §6.2 the
   live PD/PPD decision rule, pre-registered 2026-09-15 17:30 UTC. §7 lists
   what every run measures. Naming: HBM = GPU KV, DRAM = LMCache CPU tier.
5. `millstone/PENDING.md`: the live queue — platform constraints, the state
   of each line with pointers, what is running, and the next proposals.
   Check its "current as of" line; nothing in it is a result.
6. `millstone/MILESTONE-1-Single-Instance.md` §3 only, for metric definitions
   (JCT, TTFT, TPOT, cached-prompt share, cohort windows). The rest of M1 is
   the closed single-GPU stage. **Milestone 1 is written in Chinese**,
   §3 included.

The other research lane in this repository — tool-resource prediction, phase
leasing and compaction, on an A100 with PennyLane and SQLGlot traces — has its
own index at `analysis/README.md`; the two index systems do not cross-reference
each other.

Repository layout that matters:

- `results/<run>/` holds every replay run named in Milestones 2, 3 and 4
  (gitignored; never delete or rewrite). A recent launcher run contains
  `replay-command.json` (exact local replay argv and env), `launch.conf`
  (host supervisor program with every host-side flag),
  `output/throughput_summary.json` (per-task JCT and success), `server/`
  (host-side engine logs, telemetry, KV events, source tarballs, and the
  run's `manifest.yaml`), and `comparison.json` plus `comparison.txt` when a
  comparison was run against that directory. Older and partial runs carry
  less: of 120 directories, 92 have `server/manifest.yaml` (24 have one at
  the top level instead), 51 have `output/throughput_summary.json`, 37 have
  `comparison.json`, 15 have a `protocol.json`. Check what a directory
  actually holds before citing it. `results/vast-host-backup-20260910/`
  holds smokes, calibration runs, and launch logs pulled off an old host.
- `analysis/development/pool64-distinct-v4/`: the Milestone 4 workload (64
  distinct tasks, 1,951 original requests, peak context ≤ 60K tokens, no
  parallel tool calls, no zero-completion steps) with `manifest.yaml`
  (absolute trace paths) and `task-source.json`; the replacement stream keeps
  the concurrency constant. `mixed56-2l40s-concurrency32-v1/` is the
  Milestone 2/3 workload.
- `scripts/evaluation/run_two_instance_fcfs.sh` runs on the GPU host: the
  engines, the cross-instance proxy, collectors. `INSTANCE_POLICY`
  (fcfs | continuum), `ROUTER_POLICY` (least-requests | thunderagent | dualmap
  | pd | ppd | profile), `TASK_STICKY`. `SINGLE_GPU=<index>` runs a single
  engine on that GPU with one proxy backend, and needs `INSTANCES_PER_GPU=1`
  and `TENSOR_PARALLEL=1`; it is the only way the script accepts a host with
  one GPU. It refuses unreviewed combinations.
- `scripts/evaluation/vast_two_instance.py` runs here and drives one run end
  to end: supervisor program on the host, proxy tunnel, replay in Docker task
  containers, result pull into `results/<run>/server/`. `--single-gpu 0|1`
  sets `SINGLE_GPU`; `--smoke` and `--calibrate` are host-only; DualMap runs
  take `--calibration-run NAME`. Extra host env: `--env K=V`.
  `scripts/baselines/README.md`, section "Two-instance GPU host", has the
  full command set.
- `scripts/baselines/{thunderagent,dualmap,ppd}_official.sh` pin the public
  upstreams by commit; the ThunderAgent fix patches and four PPD patches sit
  beside them.

GPU host, as of 2026-09-15.

**Read this before you use any number below: this pipeline has never been run
to completion on the current machine.** The inventory that follows — address,
GPU, driver, cgroup limits, free disk, what is already on disk — was read off
the machine and is recorded in `PENDING.md` §1. Every *measurement* this block
used to carry — bootstrap VERIFY, FCFS smoke, KV cache capacity per GPU,
decode TPOT, first-request JIT time, and the whole DRAM-tier sizing envelope —
was taken on hosts that no longer exist. All of them are UNMEASURED here.
Run the bootstrap and a smoke on this host and measure them again before you
quote, compare or plan against any of them. Re-read `PENDING.md` §1 too; it
changes more often than this file.

- `ssh -p 36715 root@connect.singapore-a.gpuhub.com`, a gpuhub/AutoDL
  container with **one** RTX Pro 6000 Blackwell (96 GB HBM), driver 595,
  cgroup limits of 120 GiB memory and 208 cores, about 50 GB free on
  `/root/autodl-tmp`. `/workspace` is a symlink into `/root/autodl-tmp`, so
  the usual paths hold: repo snapshot at `/workspace/agent-sched-bench`,
  checkouts and venvs under `/workspace/.cache/agent-sched-bench/`
  (`XDG_CACHE_HOME`), uv and its Python under `/workspace/.cache/uv` and
  `/workspace/.uv-python`, models (≈ 97 GB, already present) under
  `/workspace/.hf_home`, patched ThunderAgent variants at
  `/workspace/ThunderAgent-{pending-release,capacity-consistent}-7ddc861`
  with venvs under `/workspace/venvs/`, CUDA JIT cache at
  `/workspace/.nv/ComputeCache`. `/workspace/outlen` is in use by the running
  TPOT lane (it holds the replay prefixes), so it is not free disk.
- One GPU means every launcher run is `--single-gpu 0`. The two-engine
  topology cannot be run here at all.
- The launcher controls runs through the Debian `supervisor` package (the
  program in `launch.conf`), installed by hand. The image overwrites `/usr/bin/supervisord` with a Go binary at
  every restart (and a rental lapse restarts the container): after a restart
  run `python3 -m supervisor.supervisord -c /etc/supervisor/supervisord.conf`
  before launching, or every launcher exits silently with an empty log.
- The 120 GiB cgroup cap bounds the pinned DRAM tier: nothing at 144 GiB or
  above can start here. The DRAM-tier sizing figures behind M4 §3.3 were
  measured under a 240 GB cgroup on the retired 2-GPU box, so a 144 GiB tier
  cannot even start on this machine and none of that envelope transfers.
  Re-measure it here before exercising the §3.3 sizing rule again.
- The bootstrap (`benchmark_server.sh --serving-host`, VERIFY OK), the FCFS
  `--smoke`, the per-GPU KV-cache token count, the smoke decode TPOT and the
  first-request JIT time have never been produced on this host. Milestone 4
  §6 records vLLM 0.28.0 as the current serving stack, but that was the
  TPOT-curve lane, not the two-instance launcher: the launcher's engine build
  here is unverified.
- Two concurrent runs sharing the host is moot on one GPU. The flags still
  exist and default to `--tunnel-port 19019` and `--port-base 0`; a second
  checkout `/workspace/agent-sched-bench-b` (launcher `--remote-repo`)
  receives shipped code while a launcher runs in the main one.
- Nothing from the L40S hosts or the 2-GPU Pro 6000 box carries over to a
  comparison. DualMap must be recalibrated (`--calibrate`) and every baseline
  rerun here before a candidate is compared. The L40S results under
  `results/` stay as the Milestone 2 and 3 record; the 2-GPU Pro 6000 results
  stay as the Milestone 4 §1–§3 record.
- The host source is a snapshot of the local HEAD. After committing code the
  host executes, re-ship it (README commands) before launching, and never
  while a launcher is executing there (the workers import from disk).
- PD-family runs (`ROUTER_POLICY=pd|ppd|profile`): the pinned vLLM 0.28.0
  wheel is a CUDA 13 build needing driver 580 or newer, and the launcher
  falls back to `--env PPD_CUDA=cu129 --env PPD_VENV=<venv>` on older
  drivers. This host reports driver 595, so the default CUDA 13 wheel is the
  branch that applies — but no PD-family run has launched here, so the venv
  state is unverified; re-check before launching one. The launcher defaults
  the UCX transport to `all/all` (GPU-direct over PCIe); TCP over loopback
  stalled KV pushes on the L40S host.
- The PPD upstream exists only on the host, so `tests/test_ppd_*.py` fail
  locally on import; that is expected.
- Commit 2ebe6f4 made replays survive a replacement-task failure (recorded
  in `throughput_summary.json` as `replacement_failures`) and gave the
  least-requests proxy one retry on a dropped engine connection. Runs before
  it that hit this defect: the Poisson Continuum run (aborted) and the
  two-sided PPD run (all originals finished; summary reconstructed, see M3 §3).

Caveats you must not lose:

- The public-PPD result is withdrawn, not merely caveated: that run's prefix
  cache always missed, so it measured the routing rule and not the mechanism
  (M3 §3). Do not quote its JCT as a PPD number anywhere.
- Fixed PD, the withdrawn PPD run and the profiling matrix ran on vLLM
  0.28.0 with NIXL; every other policy on vLLM 0.10.2. Original ThunderAgent and the 2026-09-06
  FCFS/Continuum repeats ran on a different host with CUPTI. JCT is comparable
  across these; TPOT carries a configuration offset.
- One physical run per policy. The paired bootstrap over 28 source
  trajectories in `comparison.json` is the uncertainty estimate.
- `scripts/evaluation/compare_two_instance_runs.py --candidate results/X
  --reference results/Y [--reference ...] --out results/X/comparison.json`
  produces the checklist below from run directories; it reproduces the
  2026-09-09 `comparison.json` values. `gpu_balance_summary.py` adds the
  per-GPU busy/idle split for load-balance questions.

Whenever a run is compared to a baseline, check and report all of these,
whichever direction each moved:

- completion: 64/64 tasks and 1,951 original requests (pool64-v4), before
  anything else counts
- task JCT mean, P95, max, and cohort makespan
- engine TPOT, token-weighted
- completed LLM steps per minute over a common window, original plus
  background
- cached-prompt share on original-task inputs
- paired mean-JCT difference with its bootstrap interval
- the worst tasks by JCT (the same six under every policy so far)

These are checks, not gates: the work moves along a Pareto frontier and a
change that wins everywhere is not expected.

Do not begin any experiment or code change from this prompt alone; wait for
the task that follows it.
