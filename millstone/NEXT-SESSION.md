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
3. `millstone/MILESTONE-1-Single-Instance.md` §3 only, for metric definitions
   (JCT, TTFT, TPOT, cached-prompt share, cohort windows). The rest of M1 is
   the closed single-GPU stage.

Repository layout that matters:

- `results/<run>/` holds every replay run named in Milestone 2 (gitignored;
  never delete or rewrite). Each has `manifest.yaml`, `replay-command.json`
  (exact local replay argv and env), `launch.conf` (host supervisor program
  with every host-side flag), `protocol.json` where one was written,
  `output/throughput_summary.json` (per-task JCT and success), `server/`
  (host-side engine logs, telemetry, KV events, source tarballs), and for the
  2026-09-09 comparisons `comparison.json`. `results/vast-host-backup-20260910/`
  holds smokes, calibration runs, and launch logs pulled off the old host.
- `analysis/development/mixed56-2l40s-concurrency32-v1/`: the workload.
  `manifest.yaml` carries absolute trace paths for this machine (the loader
  requires absolute paths), `task-source.json` the 28 task records, `traces/`
  the 56 replay traces (gitignored, 548 MB).
- `scripts/evaluation/run_two_instance_fcfs.sh` runs on the GPU host: two
  engines, the cross-instance proxy, collectors. `INSTANCE_POLICY`
  (fcfs | continuum), `ROUTER_POLICY` (least-requests | thunderagent | dualmap
  | pd | ppd | profile), `TASK_STICKY`. It refuses unreviewed combinations.
- `scripts/setup/vast_host.sh` is the host lifecycle from here: `use HOST PORT`
  remembers the host in `.vast-host`, `bootstrap` ships the local HEAD and
  builds/verifies the host (idempotent), `status`, `verify`, `ship`, `ssh`.
- `scripts/evaluation/vast_two_instance.py` runs here and drives one run end
  to end: supervisor program on the host, proxy tunnel, mixed56 replay in
  Docker task containers, result pull into `results/<run>/server/`. Host and
  port come from `.vast-host`. `--smoke` and `--calibrate` are host-only;
  DualMap runs take `--calibration-run NAME`. Extra host env: `--env K=V`.
  `scripts/baselines/README.md`, section "Two-instance GPU host", has the
  full command set.
- `scripts/baselines/{thunderagent,dualmap,ppd}_official.sh` pin the public
  upstreams by commit; the ThunderAgent fix patches and four PPD patches sit
  beside them.

GPU host, as of 2026-09-10:

- `ssh -p 28229 root@118.163.199.123`, Vast container C.50481401, 2× L40S,
  driver 570, Ubuntu 24.04, no volume: nothing on it survives a recycle, and
  the home directory is regenerated on every start. Everything lives under
  `/workspace`: repo snapshot at `/workspace/agent-sched-bench`, checkouts and
  venvs under `/workspace/.cache/agent-sched-bench/` (`XDG_CACHE_HOME`),
  model under `/workspace/.hf_home`, patched ThunderAgent variants at
  `/workspace/ThunderAgent-{pending-release,capacity-consistent}-7ddc861`
  with venvs under `/workspace/venvs/`.
- Verified on this host on 2026-09-10: FCFS least-requests `--smoke`
  (`results/fcfs-least-requests-smoke-20260910-r2`) and DualMap `--calibrate`
  (`results/dualmap-calibration-20260910-r1`), which measured
  `DUALMAP_PREFILL_TPOT=5.543863341017641e-05` (old host: 5.44e-05). Pass that
  value with `--env` for DualMap runs on this host; recalibrate on a new one.
- The host source is a snapshot of the local HEAD. After committing code the
  host executes, run `scripts/setup/vast_host.sh ship` before launching.
- The PPD upstream exists only on the host, so `tests/test_ppd_*.py` fail
  locally on import; that is expected.

Caveats you must not lose:

- Fixed PD, PPD and the profiling matrix ran on vLLM 0.28.0 with NIXL; every
  other policy on vLLM 0.10.2. Original ThunderAgent and the 2026-09-06
  FCFS/Continuum repeats ran on a different host with CUPTI. JCT is comparable
  across these; TPOT carries a configuration offset.
- One physical run per policy. The paired bootstrap over 28 source
  trajectories in `comparison.json` is the uncertainty estimate.
- The script that produced `comparison.json` was never committed; its
  outputs are in the run directories. Re-implement from those if needed.

Whenever a run is compared to a baseline, check and report all of these,
whichever direction each moved:

- completion: 56/56 tasks and 2,470 requests, before anything else counts
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
