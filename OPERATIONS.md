# Operations Guide

Operator-facing reference for collecting, replaying, and viewing traces. This
complements `README.md` with exact flags, environment variables, resume
semantics, and benchmark plugin rules.

## Invariants

- Trace collection obtains model actions from the registered remote/Codex
  provider path. Provider API bases reject localhost, loopback, private,
  link-local, and unspecified addresses.
- Replay always follows the recorded action sequence and re-executes tool calls
  in the task runtime. It can use source LLM timing without inference or send
  fixed-trajectory shadow requests to a separate local vLLM server.
- `--tool-resource-telemetry clause` is observation-only: it neither queries
  nor updates the resource KB. Use an explicit tool-resource profile when the
  KB/predictor services are part of the experiment.
- Result-affecting configuration belongs in the run artifacts. A directory or
  config name alone is not evidence that an experiment completed successfully.

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
  `deepseek`, `pioneer`, or `codex`.
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
  - `PIONEER_API_KEY`
- Codex uses the ChatGPT subscription credentials written by `codex login`;
  `CODEX_ACCESS_TOKEN` and `CODEX_ACCOUNT_ID` can override the local login.
- Override with `--api-key` or `--api-base` for OpenAI-compatible gateways.
- `--service-tier fast` is supported only by Codex.

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

An instance is accepted and skipped only when one nested
`attempt_*/run_manifest.json` has `status=completed` or `status=exhausted` and
its resource evidence, when present, is valid:

- if `resource_observations.json` is absent, the terminal manifest is accepted;
- if it is present, it must be a JSON object with
  `telemetry_quality=ok`, `collection_validity=valid`, and `cleanup=ok`;
- malformed or non-object resource evidence, or any other value for those
  three fields, makes the attempt non-terminal for resume.

`status=error` attempts are never accepted, even when the error text or
`exit_status` mentions max-iteration exhaustion. Resume scans only the nested
attempt layout; it does not accept legacy flat manifests.

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

The only trace-replay mode is named `cloud_model`. Without
`--shadow-llm-api-base`, replay uses the recorded LLM timing and issues no new
model requests. With a shadow API and model, it submits fixed-trajectory
requests to the serving system while preserving the recorded tool sequence.

```bash
PYTHONPATH=src:. uv run python -m trace_collect.cli simulate \
    --manifest /abs/path/to/simulate-manifest.yaml \
    --concurrency 1,2,4,8 \
    --workers 8 \
    --prep-concurrency 20 \
    --container docker \
    --replay-speed 50
```

### Modes

- `--mode cloud_model` is the only trace format/replay mode.
- Source-timing replay is CPU-only apart from the real task workload.
- Shadow serving uses `--shadow-llm-api-base`, `--shadow-llm-model`, and one of
  the registered `--shadow-llm-mode` policies. It requires a separately
  launched serving process and does not replace the collection provider.

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

#### Full-corpus fresh-277 replay at --replay-speed 20 (8-core / 15 GB host)

```
PYTHONPATH=src:. uv run python -m trace_collect.cli simulate \
    --manifest /abs/path/to/fresh-277-manifest.yaml \
    --container docker \
    --workers 8 \
    --concurrency 8 \
    --prep-concurrency 8 \
    --replay-speed 20 \
    --cleanup-images \
    --output-dir traces/fresh-277-replay
```

- Concurrency: on an 8-core / 15 GB host keep `--workers 8 --concurrency 8` so
  at most 8 task containers run at once (~1 core, ~1.5 GB each). Drop to
  `--workers 4 --concurrency 4` if replayed builds (`make`, `pytest`) are
  memory-heavy. `--prep-concurrency` throttles concurrent image builds; keep it
  <= workers to avoid a build-time memory spike.
- Disk (`--cleanup-images` REQUIRED for fresh-277): the fresh-277 corpus has
  **277 unique ~2.8 GB source images** (one per task), so pulling them all is
  ~775 GB. `--cleanup-images` skips the up-front global prefetch and instead
  pulls each task's image on demand, then removes it once no pending session
  still references it. Resident image footprint is then ~`concurrency` x 3 GB
  (~24 GB at `--concurrency 8`) plus a few MB per trace, not the full corpus.
  - WARNING: an earlier version of this runbook claimed "each fixed image is
    built once and shared; budget ~2-3 GB for images." That estimate was WRONG
    for fresh-277 — it holds only for shared-image corpora (e.g. a single
    swe-bench base image). Without `--cleanup-images`, the fresh-277 prefetch
    stage pulls every unique image up front and fills the disk before a single
    task replays. Leave `--cleanup-images` OFF only for shared-image corpora.
- Wall clock: replay re-executes tool commands for real (unaffected by
  `--replay-speed`), so total time is dominated by real tool execution, not the
  20x-accelerated LLM gaps. Budget hours for the full 277, not minutes.

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

## Tool-resource telemetry replay

Tool-resource replay uses two long-lived services. `resource-telemetryd` is the
only root process; `resource-agentd`, the replay, and the SQLite database run as
the invoking user. Use separate socket directories so the root-owned telemetry
socket can be group-readable without making the resource socket root-owned.
Both services derive client ownership from Unix-socket peer credentials, so an
idle queued task remains live while its coordinator process exists; no
heartbeat or active-session TTL is required. Runs follow the coordinator peer;
each trace separately follows the worker peer that opened it, so a killed
worker aborts only its own collector.

The following setup reproduces the exact task cohort from a prior Terminal-Bench
baseline while reloading task metadata through the benchmark plugin. It does not
construct `tasks.json` rows by hand.

```bash
export BASELINE="$PWD/traces/terminal-bench/tb-dev20-c1-ladder-20260727-r4"
export RUN_INPUT="/tmp/tb-call-promotion-input"
export TRACE_SOURCE="$PWD/traces/terminal-bench/tb-all/canonical"
test ! -e "$RUN_INPUT" || {
  echo "refusing to reuse RUN_INPUT: $RUN_INPUT" >&2
  exit 1
}
mkdir -p "$RUN_INPUT"

PYTHONPATH=src:. uv run python - <<'PY'
import json
import os
from pathlib import Path

import yaml

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from trace_collect.simulate_manifest import SIMULATE_MANIFEST_SCHEMA_VERSION

baseline = Path(os.environ["BASELINE"])
output = Path(os.environ["RUN_INPUT"])
config = BenchmarkConfig.from_yaml(Path("configs/benchmarks/terminal-bench.yaml"))
benchmark = get_benchmark_class(config.slug)(config)
tasks = {task["instance_id"]: task for task in benchmark.load_tasks()}
rows = json.loads((baseline / "throughput_summary.json").read_text())["tasks"]
selected = [tasks[row["run_instance_id"]] for row in rows]
traces = [
    {
        "trace": str(
            (Path(os.environ["TRACE_SOURCE"]) / task["instance_id"] / "trace.jsonl")
            .resolve()
        ),
        "label": f"tb-dev20-{task['instance_id']}",
    }
    for task in selected
]
missing = [row["trace"] for row in traces if not Path(row["trace"]).is_file()]
if missing:
    raise SystemExit(f"missing canonical traces: {missing}")
(output / "tasks.json").write_text(
    json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(output / "manifest.yaml").write_text(
    yaml.safe_dump(
        {
            "version": SIMULATE_MANIFEST_SCHEMA_VERSION,
            "defaults": {"task_source": str((output / "tasks.json").resolve())},
            "traces": traces,
        },
        sort_keys=False,
    ),
    encoding="utf-8",
)
PY
```

Create fresh service state and start both daemons with logs redirected to durable
files. The example bucket edges reproduce the development diagnostic baseline;
they are not authoritative latency boundaries and must not be used for a
claim-bearing run.

```bash
export RUN_STATE="/tmp/tb-call-promotion-state"
export OUTPUT_DIR="$PWD/traces/terminal-bench/tb-call-promotion-c1-50x"
export USER_UID="$(id -u)"
export USER_GID="$(id -g)"
export TELEMETRY_RUNTIME="$RUN_STATE/telemetry-runtime"
export TELEMETRY_SOCKET="$TELEMETRY_RUNTIME/telemetry.sock"
export RESOURCE_SOCKET="$RUN_STATE/resource.sock"
for path in "$RUN_STATE" "$OUTPUT_DIR"; do
  test ! -e "$path" || {
    echo "refusing to reuse run state or output: $path" >&2
    exit 1
  }
done
mkdir -p "$RUN_STATE"
sudo -n install -d -m 0750 -o root -g "$USER_GID" "$TELEMETRY_RUNTIME"

cat >"$RUN_STATE/resource.yaml" <<YAML
tool_resource:
  endpoint: unix://$RESOURCE_SOCKET
  behavior: observe_predict_learn
  update_policy: causal
  snapshot: latest_at_run_start
  telemetry_requirement: required_for_valid_evidence
  latency_bucket_edges_ms: [10, 100, 1000]
YAML

sudo -n env PYTHONPATH="$PWD/src:/usr/lib/python3/dist-packages" \
  "$PWD/.venv/bin/python" \
  -m tool_resource.telemetryd \
  --socket "$TELEMETRY_SOCKET" \
  --allowed-uid "$USER_UID" \
  --socket-gid "$USER_GID" \
  --container-runtime docker \
  --state-dir "$TELEMETRY_RUNTIME/state" \
  >"$RUN_STATE/telemetryd.log" 2>&1 &
TELEMETRY_PID=$!

PYTHONPATH=src "$PWD/.venv/bin/python" -m tool_resource.resource_agentd \
  --socket "$RESOURCE_SOCKET" \
  --database "$RUN_STATE/observations.sqlite3" \
  --telemetry-socket "$TELEMETRY_SOCKET" \
  --allowed-uid "$USER_UID" \
  >"$RUN_STATE/resource-agentd.log" 2>&1 &
RESOURCE_PID=$!

for _ in $(seq 1 100); do
  test -S "$TELEMETRY_SOCKET" -a -S "$RESOURCE_SOCKET" && break
  sleep 0.1
done
test -S "$TELEMETRY_SOCKET"
test -S "$RESOURCE_SOCKET"
```

Run the serial 50x replay and read the evidence gates. Deliberately omit
`--cleanup-images`: the simulator globally prefetches and prebuilds the full
20-task image set before starting replay instead of pulling and deleting images
per task. Any call-level failure can now be joined across daemon logs by
`(run_id, trace_id, call_id)`, and each `resource_observations.json` call records
its resource-agentd `telemetry_status`. The runner joins asynchronous collector
attachment before replay time zero; OpenTrace and every online Begin/End call
remain non-blocking with respect to telemetry.

```bash
PYTHONPATH=src:. uv run python -m trace_collect.cli simulate \
  --manifest "$RUN_INPUT/manifest.yaml" \
  --container docker \
  --concurrency 1 \
  --workers 1 \
  --prep-concurrency 1 \
  --replay-speed 50 \
  --tool-resource-profile "$RUN_STATE/resource.yaml" \
  --output-dir "$OUTPUT_DIR"

PYTHONPATH=src:. uv run python scripts/evaluation/report_clause_coverage.py \
  "$OUTPUT_DIR" --baseline "$BASELINE"
```

Stop the services after the run. The trace output and SQLite database are
evidence; remove only sockets and temporary collector state.

```bash
kill "$RESOURCE_PID"
sudo -n kill "$TELEMETRY_PID"
wait "$RESOURCE_PID" || true
wait "$TELEMETRY_PID" || true
sudo -n rm -r "$TELEMETRY_RUNTIME"
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
