# Operations Guide

Operator-facing reference for collecting, replaying, and viewing traces on the
`dev/cpu-only` branch. This complements `README.md` with exact flags, env vars,
resume semantics, and benchmark plugin rules.

## Invariants

This branch is cloud-provider-only. Do not reintroduce:

- local/private OpenAI-compatible endpoints
- local HF / vLLM / self-hosted model serving
- internal recording hooks
- GPU profiling

Provider API bases are validated; localhost, loopback, private, link-local, and
unspecified addresses are rejected.

## Trace Collect

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-pro \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --concurrency 2 \
    --sample 5
```

### Required

- `--provider`: one of `openrouter`, `dashscope`, `openai`, `siliconflow`,
  `deepseek`.
- `--model`: model slug for the provider.
- `--scaffold openclaw`.
- `--mcp-config` for OpenClaw. YAML path, or literal `none` for an explicit
  MCP-less run.
- `--container docker|podman` for container-mode benchmarks.

### API key

- Resolved from the provider env var by default:
  - `OPENROUTER_API_KEY`
  - `DASHSCOPE_API_KEY`
  - `OPENAI_API_KEY`
  - `SILICONFLOW_API_KEY`
  - `DEEPSEEK_API_KEY`
- Override with `--api-key` or `--api-base` for OpenAI-compatible gateways.

### Task selection

Selection order:

1. `--instance-ids a,b,c` filters/reorders by explicit IDs.
2. `--skip N` drops the first N remaining tasks.
3. `--sample N` keeps only the first N remaining tasks.

Both `--skip` and `--sample` reject negative values.

### Concurrency

`--concurrency N` runs up to N tasks at once. Default is 1 (sequential).

- Applies at benchmark task level for both:
  - SWE-style `task_container_agent` runs
  - Terminal-Bench `host_controller` runs
- `concurrency=1` keeps the original sequential path, including image prefetch.
- `concurrency>1` schedules non-terminal tasks under an asyncio semaphore.
- `results.jsonl` is always written in original task order, regardless of
  completion order.

### Resume

`--run-id <existing run dir>` resumes an interrupted run.

An instance is skipped on resume if any of its `attempt_*/run_manifest.json`
has:

- `status=completed`, or
- `status=exhausted` (max iterations reached).

`status=error` attempts are not terminal and will be rerun on resume.

### Local task cache

Opt-in via `AGENT_SCHED_BENCH_USE_LOCAL_TASK_CACHE=1`. When enabled, SWE-Bench
Verified and SWE-rebench load from `<data_root>/tasks.json` before hitting
HuggingFace. Locally cached rows are stamped with provenance:

- `task_source_kind=benchmark_local_json`
- `task_source_id=<instance_id>`
- `task_source_path=<cache path>`

Rows without `instance_id` are rejected.

### Task-container environment

Task-container Python runtime dependencies are bootstrapped into immutable shared cache generations under `~/.cache/task-container-bootstrap/<platform>/<config-hash>/`. A file lock serializes writes, and each cache marker records requirements, Python runtime, pip index, pip-resolution env fingerprint, architecture, image platform, Python ABI/OS/libc fingerprint, and installed package/version manifests. Stale or contaminated generations are not reused, and active generations are not deleted while another attempt may still be reading them.

Bootstrap and apt/pip behavior inside task containers can be tuned with:

- `TASK_CONTAINER_PIP_INDEX_URL`
- `TASK_CONTAINER_PIP_EXTRA_INDEX_URL`
- `TASK_CONTAINER_PIP_TRUSTED_HOST`
- `TASK_CONTAINER_PIP_CERT`
- `TASK_CONTAINER_SSL_CERT_FILE`
- `TASK_CONTAINER_HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY`
- `TASK_CONTAINER_APT_MIRROR`
- `TASK_CONTAINER_APT_SECURITY_MIRROR`

APT mirror setup supports Debian and Ubuntu task images. It runs as container
root so it can rewrite `/etc/apt/sources.list.d`.

During task-container agent runs, stdout is streamed live to the operator terminal and also written to the per-attempt raw stdout artifact. `resources.json` summaries include `monitoring.status` (`collected`, `enabled_no_samples`, or `disabled`) plus `monitoring_disabled` so empty sample lists are explicit.

## Trace Simulate

Cloud replay only. No LLM requests are issued during replay.

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --manifest /abs/path/to/simulate-manifest.yaml \
    --concurrency 1,2,4,8 \
    --workers 8 \
    --prep-concurrency 20 \
    --container docker \
    --replay-speed 50
```

### Modes

- `--mode cloud_model` (default and only supported mode on this branch).

### Concurrency

- `--concurrency 8`: one bounded replay with at most 8 active traces.
- `--concurrency 1,2,4,8`: throughput sweep, writes `throughput_sweep.jsonl`.
- `--workers N`: for high-concurrency replay, split active traces across N OS
  processes, each with its own asyncio event loop. Default `1` preserves the
  single-process path.
- `--prep-concurrency N`: system-wide container preparation throttle shared
  across workers. `0` preserves the default limit of 20.
- `--resource-monitoring auto|on|off`: built-in container resource sampling.
  `auto` enables it for container replay and leaves host replay unmonitored.
- `--pmu-monitoring auto|on|off`: PMU-backed cgroup memory-access telemetry.
  `auto` enables it only for non-concurrent container replay; explicit `on` is
  rejected when `--concurrency > 1` or `--workers > 1`.
- `--memory-bandwidth-monitoring auto|on|off`: host memory-bandwidth telemetry.
  `auto` enables it only for non-concurrent container replay; explicit `on` is
  recorded as requested but safely disabled when `--concurrency > 1` or
  `--workers > 1`.

For large closed-loop replay, use `--workers` near
`min(concurrency, os.cpu_count())` (or lower if memory/process overhead matters).
Worker mode intentionally differs from the legacy bounded queue: it prepares a
full wave under `--prep-concurrency`, waits for every session in that wave at a
global all-ready barrier, then releases replay from a shared time zero. This
excludes container warm-up from action timestamps and improves high-concurrency
timing comparability, but it means wave members wait for the slowest preparation
before replay starts. In worker mode the global `ContainerResourceRecorder` is
disabled to avoid cross-process Docker-stat interference; per-task
`resources.json` artifacts are still written.

### Timing

- `--replay-speed N`: wall-clock acceleration for source inter-action gaps and
  source-scaled action durations.
- `--llm-timing source-scaled` (default): sleep for source LLM duration divided
  by `--replay-speed`.
- `--llm-timing ttft-tpot`: sleep for `--llm-ttft-ms + (completion_tokens - 1) *
  --llm-tpot-ms`. Tool timing and inter-action gaps still use source timing
  scaled by `--replay-speed`.
- Replay records expected-vs-actual sleep drift for source gaps, LLM replay
  sleeps, and trace-replayed tool sleeps in all modes; worker-mode runs also
  record worker-start sleep drift. This is an intentional trace-schema addition:
  per-action details live under `data.sim_metrics.source_gap_sleep` /
  `data.sim_metrics.action_sleep`, and per-task summaries include aggregate
  `sleep_drift` statistics.

### Resource-integrated timeout

When a source `exec` tool interval carries `resource_timeline` for a single
`exec.command`, replay uses it for an online source-equivalent timeout. The
fixed v1 model:

- CPU-active at >=0.05 core
- network-active at >=1024 B/s
- samples replay every 0.5s
- 5-60s stall detector plus a 24h outer protocol guard

Host/no-op replay and multi-command exec preserve `resource_timeline` as source
metadata only.

### Checkpoint forced sync

Collect emits `checkpoint_after` archives for checkpointed `exec` actions.
Simulate can force-sync these checkpoints to verify replay reproduces source
state. A forced-sync smoke helper is available at
`scripts/smoke_checkpoint_forced_sync.py`.

### Manifest format

Simplest form, a list of absolute trace paths:

```yaml
- /abs/path/task-a/attempt_1/trace.jsonl
- /abs/path/task-b/attempt_1/trace.jsonl
```

Structured form:

```yaml
version: 1
defaults:
  task_source: /abs/path/data/swe-rebench/tasks.json
traces:
  - trace: /abs/path/task-a/attempt_1/trace.jsonl
    label: task-a
  - trace: /abs/path/task-b/attempt_1/trace.jsonl
    docker_image: custom/image:tag
```

## Gantt Viewer

### Serve

```bash
PYTHONPATH=src python -m trace_collect.cli gantt-serve
```

Interactive web viewer for trace inspection.

### Export

```bash
PYTHONPATH=src python -m trace_collect.cli gantt-export --help
```

Static exports for offline sharing.

## Benchmark Plugin Rules

All benchmark-specific behavior belongs in:

- `src/agents/benchmarks/<plugin>.py`
- `configs/benchmarks/<slug>.yaml`

Forbidden:

- hardcoding dataset names in `collector.py`, `cli.py`, or scaffold code
- per-benchmark collector CLI flags
- dataset-specific magic numbers disguised as general methods

### Registered benchmarks

| Slug | Runtime mode | Dataset | Scaffold |
|---|---|---|---|
| `swe-bench-verified` | `task_container_agent` | `princeton-nlp/SWE-bench_Verified` | openclaw |
| `swe-rebench` | `task_container_agent` | `nebius/SWE-rebench` | openclaw |
| `terminal-bench` | `host_controller` | Terminal-Bench tasks | openclaw |

### Adding a benchmark

1. Create `configs/benchmarks/<slug>.yaml` with dataset, image, selection, and
   prompt defaults.
2. Implement a `Benchmark` subclass in `src/agents/benchmarks/<plugin>.py`.
3. Register the plugin class in `src/agents/benchmarks/__init__.py`.
4. Add normalization and config tests.
5. Do not add CLI flags specific to the new benchmark.

## Explicitly Removed

- `vllm`, `torch`, `transformers`, `accelerate` dependencies
- `src/serving/` local backend code
- internal recording hooks (`recording_provider`)
- local-HF setup (`HF_TOKEN`, `MODEL_PATH`) in `scripts/setup/configure_env.sh`
- vLLM serving, metrics, startup parsing, and scheduler hooks
- GPU / `nvidia-smi` profiling and `profile-gpu`
- `local_model` simulation mode
