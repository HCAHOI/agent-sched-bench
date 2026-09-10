# Paper baselines

These adapters pin either authors' code or the smallest paper-derived subset
needed by the physical replay. They do not make every method a full-system
reproduction.

| Cell name | Adapter | Fidelity boundary | Preparation |
|---|---|---|---|
| `fcfs` | stock repo vLLM | Physical control with prefix caching; no paper policy | `bash scripts/setup/benchmark_server.sh --gpu` |
| `thunderagent` | `thunderagent_official.sh` | Official program-aware proxy; it does not manage tool containers | `./scripts/baselines/thunderagent_official.sh install` |
| `agentix` | `agentix_reproduction.sh` | PLAS arrival-priority subset; no ATLAS, dynamic quantum, KV-swap kernel, or multi-engine routing | `./scripts/baselines/agentix_reproduction.sh install` |
| `continuum-public` | `continuum_public.sh` | Authors' public fixed-2-second TTL fork, not the paper estimator | `./scripts/baselines/continuum_public.sh install` |
| `continuum-reproduction` | `continuum_reproduction.sh` | Public fork plus the paper TTL equation; requires a hardware-measured prefill/reload profile | `./scripts/baselines/continuum_reproduction.sh install` |
| `native-priority` | stock repo vLLM priority scheduler | Priority 1 for each program's initial request, priority 0 for causal tool returns | `bash scripts/setup/benchmark_server.sh --gpu` |
| `native-priority-aging` | `native_priority_aging.sh` | Same return priority with promotion after one `max_num_seqs=8` batch of bypasses | `./scripts/baselines/native_priority_aging.sh install` |
| `cachewise-disabled` | `cachewise_reproduction.sh serve-disabled` | Exact CacheWise fork/config control without CacheWise scheduling or policy payloads | `./scripts/baselines/cachewise_reproduction.sh install` |
| `cachewise` | `cachewise_reproduction.sh` | Authors' vLLM fork plus reconstructed scheduler hooks; the exact paper split and engine attachment hook are unpublished | Run `cachewise_official.sh install`, `cachewise_official.sh train-published`, then `cachewise_reproduction.sh install` |
| `saga` | `saga_reproduction.sh` | Single-GPU KV/arrival-priority subset; no private multi-GPU routing, migration, or prefetch system | `./scripts/baselines/saga_reproduction.sh install` and build a causal profile |

`cachewise_official.sh` is only the authors' released duration predictor. It is
not a serving baseline. `continuum-public` and `continuum-reproduction` remain
separate because only the latter adds the estimator described in the paper.
Murakkab remains related work only: its declared-DAG/MILP input contract does
not match black-box trace replay, so it has no active adapter here.

## Physical runner

The single entry point is `scripts/evaluation/run_paper_baseline.sh`. It accepts
any compatible replay manifest and runs cells sequentially on one A100 80 GB.
Every output root must be new.

```bash
export MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
export MANIFEST=/absolute/path/to/replay-manifest.yaml
export RUN_ROOT=/absolute/path/to/new-baseline-run
export CELLS='fcfs-r1 thunderagent-r1'
export CONCURRENCY=4

./scripts/evaluation/run_paper_baseline.sh --preflight
./scripts/evaluation/run_paper_baseline.sh --run
```

Add `agentix-r1`, `continuum-public-r1`, `continuum-reproduction-r1`,
`native-priority-r1`, `native-priority-aging-r1`, `cachewise-disabled-r1`,
`cachewise-r1`, or `saga-r1` only after its preparation above. Continuum
reproduction additionally needs `CONTINUUM_REPRODUCTION_PROFILE`; SAGA needs
`SAGA_PROFILE`. Use `TRACE_TOOL_REPLAY=0` for real task-container tool execution
or `1` only for an explicitly declared trace-timed replay.

Result receipts:

- [PennyLane low-pressure suite](../../analysis/results/pennylane-paper-baseline-suite-physical-v1.md)
- [Unique-128 high-pressure suite](../../analysis/results/mixed128-poisson-unique-baselines-20260827/result.md)

The dated one-off PennyLane suite driver was removed after its final state and
result receipt were retained in Git (`3b2ea8d` and `454bd13`). Recover it from
those revisions only to inspect the historical protocol, not as a current
entry point.

## Two-instance GPU host (2× L40S on Vast)

The two-instance runs split work across two machines. The GPU host runs the
engines, proxy and collectors through `scripts/evaluation/run_two_instance_fcfs.sh`;
this machine replays the agent workload in Docker task containers against the
host proxy over an SSH tunnel. Vast regenerates the home directory on every
container start, so everything on the host lives under `/workspace`:

| Path on host | Content |
|---|---|
| `/workspace/agent-sched-bench` | repo snapshot of the local HEAD plus `uv.lock`; `SOURCE_REV` names the commit |
| `/workspace/.cache/agent-sched-bench/` | upstream checkouts and venvs (`XDG_CACHE_HOME=/workspace/.cache`) |
| `/workspace/.hf_home` | model cache (`HF_HOME`) |
| `/workspace/ThunderAgent-{pending-release,capacity-consistent}-7ddc861`, `/workspace/venvs/` | patched ThunderAgent variants and their venvs |
| `/workspace/manifests/<run>.yaml`, `/workspace/<run>-launch.log`, `/workspace/bootstrap.log` | per-run manifest copy, supervisor log, build log |
| `/workspace/agent-sched-bench/results/<run>` | host-side run directory, pulled back into `results/<run>/server/` |

New machine, three commands from this repo:

```bash
scripts/setup/vast_host.sh use HOST PORT     # remember it in .vast-host (gitignored)
scripts/setup/vast_host.sh bootstrap         # ship source, build and verify the host (~35 min fresh, idempotent)
.venv/bin/python scripts/evaluation/vast_two_instance.py --name fcfs-smoke-r1 --router-policy least-requests --smoke
```

`bootstrap` follows the host log and stops at `BOOTSTRAP COMPLETE` or
`BOOTSTRAP FAILED`; rerunning skips finished steps. `vast_host.sh status`,
`verify`, `ship` and `ssh` cover the rest. After committing code the host
executes, run `ship` again; the launcher stamps `SOURCE_BASE_REV` with the
local HEAD and refuses nothing, so an unshipped host silently runs old code.

Runs, all from this machine (`--smoke` and `--calibrate` are host-only;
anything else replays mixed56):

```bash
L=".venv/bin/python scripts/evaluation/vast_two_instance.py"
$L --name dualmap-calibration-r1 --router-policy dualmap --calibrate
$L --name mixed56-vast-dualmap-r1 --router-policy dualmap --calibration-run dualmap-calibration-r1 --env DUALMAP_CPU_CACHE_GIB=48
$L --name mixed56-vast-continuum-r1 --router-policy least-requests --instance-policy continuum --task-sticky
```

Every run records its supervisor conf, the exact replay argv and environment,
calibration source, start and end times, and exit codes under `results/<run>/`.
