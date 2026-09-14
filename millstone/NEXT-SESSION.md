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
   routing), all three closed on 2026-09-11 with the runs and mechanisms
   that closed them. Read its §0 before comparing anything to a baseline.
4. `millstone/MILESTONE-4-Pressure.md`: the current stage. §3 closed
   multi-instance scheduling (one engine per GPU is the platform); §3.2 is
   the store-versus-admission decomposition at a sized DRAM tier; §3.3 the
   pressure axis, the DRAM capacity curve and the sizing rule (≈ 1.4 ×
   concurrency × mean context), why eviction policy and dispatch order do
   not help on this pool, and why residency-first admission starves; §5 the
   related work checked. Naming: HBM = GPU KV, DRAM = LMCache CPU tier.
5. `millstone/PENDING.md`: the live queue — platform constraints, the state
   of each line with pointers, what is running, and the next proposals
   (exclusive tiering first). Rewritten 2026-09-14; nothing there is a result.
6. `analysis/development/output-length-prediction-handoff.md`: the
   output-length line (closed as a prompt-side hidden-state limit, with the
   sampling ceiling that proves the information exists), the hazard probes
   (P(remaining ≤ X) along the generation) and the three-signal sandbox
   interface with its measured coverage.
7. `millstone/MILESTONE-1-Single-Instance.md` §3 only, for metric definitions
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
- `analysis/development/pool64-distinct-v4/`: the Milestone 4 workload (64
  distinct tasks, 1,951 original requests, peak context ≤ 60K tokens, no
  parallel tool calls, no zero-completion steps) with `manifest.yaml`
  (absolute trace paths) and `task-source.json`; the replacement stream keeps
  the concurrency constant. `mixed56-2l40s-concurrency32-v1/` is the
  Milestone 2/3 workload.
- `scripts/evaluation/run_two_instance_fcfs.sh` runs on the GPU host: two
  engines, the cross-instance proxy, collectors. `INSTANCE_POLICY`
  (fcfs | continuum), `ROUTER_POLICY` (least-requests | thunderagent | dualmap
  | pd | ppd | profile), `TASK_STICKY`. It refuses unreviewed combinations.
- `scripts/evaluation/vast_two_instance.py` runs here and drives one run end
  to end: supervisor program on the host, proxy tunnel, mixed56 replay in
  Docker task containers, result pull into `results/<run>/server/`.
  `--smoke` and `--calibrate` are host-only;
  DualMap runs take `--calibration-run NAME`. Extra host env: `--env K=V`.
  `scripts/baselines/README.md`, section "Two-instance GPU host", has the
  full command set.
- `scripts/baselines/{thunderagent,dualmap,ppd}_official.sh` pin the public
  upstreams by commit; the ThunderAgent fix patches and four PPD patches sit
  beside them.

GPU host, as of 2026-09-14 (a single-GPU instance may replace it; then re-run
`benchmark_server.sh --serving-host`, re-download the models (≈ 97 GB) and
copy `/workspace/outlen` if the output-length lane continues):

- `ssh -p 41548 root@connect.singapore-a.gpuhub.com`, a gpuhub/AutoDL
  container with 2× RTX Pro 6000 Blackwell Server Edition (96 GB each),
  driver 595 (CUDA 13.2 native), PCIe 5 on one NUMA node, cgroup limits of
  50 cores and 240 GB. The root filesystem is a 30 GB ephemeral overlay; the
  250 GB persistent disk is `/root/autodl-tmp`, and `/workspace` is a symlink
  into it so every path below is unchanged: repo snapshot at
  `/workspace/agent-sched-bench`, checkouts and venvs under
  `/workspace/.cache/agent-sched-bench/` (`XDG_CACHE_HOME`), uv and its
  Python under `/workspace/.cache/uv` and `/workspace/.uv-python`, model under
  `/workspace/.hf_home`, patched ThunderAgent variants at
  `/workspace/ThunderAgent-{pending-release,capacity-consistent}-7ddc861`
  with venvs under `/workspace/venvs/`, CUDA JIT cache at
  `/workspace/.nv/ComputeCache`.
- The driver controls runs through the Debian `supervisor` package, installed
  by hand. The image overwrites `/usr/bin/supervisord` with a Go binary at
  every restart (and a rental lapse restarts the container): after a restart
  run `python3 -m supervisor.supervisord -c /etc/supervisor/supervisord.conf`
  before launching, or every launcher exits silently with an empty log.
- The cgroup memory cap (240 GB) bounds the pinned DRAM tier: 144 GiB starts,
  192 GiB does not. Two single-GPU runs share the host only if their tiers
  plus two engines fit (48 + 96 GiB did; 144 + 96 did not).
- A second checkout `/workspace/agent-sched-bench-b` (launcher `--remote-repo`)
  receives shipped code while a launcher runs in the main one; two concurrent
  single-GPU runs need `--port-base 100 --tunnel-port 19100`.
- Verified on this host on 2026-09-11: bootstrap (`benchmark_server.sh
  --serving-host`, VERIFY OK) and FCFS least-requests `--smoke`
  (`results/fcfs-least-requests-smoke-20260911-r2`): vLLM 0.10.2 with Flash
  Attention in eager mode as on L40S, KV cache 624,880 tokens per GPU for
  Qwen3-4B-FP8 (L40S: 275,008), smoke decode TPOT 31 ms (L40S: 39 ms). The
  first request on a fresh JIT cache took 72 s; the cache now persists.
- Nothing from the L40S hosts carries over: DualMap must be recalibrated
  (`--calibrate`) and every baseline rerun here before a candidate is
  compared. The L40S results under `results/` stay as the Milestone 2 and 3
  record.
- The host source is a snapshot of the local HEAD. After committing code the
  host executes, re-ship it (README commands) before launching, and never
  while a launcher is executing there (the workers import from disk).
- PD-family runs (`ROUTER_POLICY=pd|ppd|profile`) use the default CUDA 13
  wheel on this driver (`--env PPD_NATIVE_PUSH=1` only). On drivers older
  than 580 they need `--env PPD_CUDA=cu129 --env PPD_VENV=<venv>`. The
  launcher defaults the UCX transport to `all/all` (GPU-direct over PCIe);
  TCP over loopback stalled KV pushes on the L40S host.
- The PPD upstream exists only on the host, so `tests/test_ppd_*.py` fail
  locally on import; that is expected.
- Commit 2ebe6f4 made replays survive a replacement-task failure (recorded
  in `throughput_summary.json` as `replacement_failures`) and gave the
  least-requests proxy one retry on a dropped engine connection. Runs before
  it that hit this defect: the Poisson Continuum run (aborted) and the
  two-sided PPD run (all originals finished; summary reconstructed, see M3 §3).

Caveats you must not lose:

- Fixed PD, PPD and the profiling matrix ran on vLLM 0.28.0 with NIXL; every
  other policy on vLLM 0.10.2. Original ThunderAgent and the 2026-09-06
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
