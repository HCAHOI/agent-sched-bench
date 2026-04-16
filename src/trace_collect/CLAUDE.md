# trace_collect — Agent Guide

Module for collecting, simulating, importing, and inspecting LLM agent traces
on SWE-style benchmarks.

## CLI Entry Point

```
python -m trace_collect.cli <subcommand> [OPTIONS]
```

Subcommands: (none) = **collect**, `simulate`, `import-claude-code`, `inspect`, `gantt-serve`.

---

## Part 1 — Collect

Run agents on benchmark tasks inside containers; record canonical JSONL traces.

### Minimal Example

```bash
OPENROUTER_API_KEY=sk-... python -m trace_collect.cli \
  --provider openrouter --model anthropic/claude-sonnet-4-20250514 \
  --scaffold openclaw --container docker \
  --benchmark swe-rebench \
  --mcp-config configs/mcp/context7.yaml \
  --max-iterations 80
```

### CLI Flags (collect)

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--provider` | yes | — | `openrouter`, `dashscope`, `openai`, `siliconflow` |
| `--model` | yes | — | Full model ID (e.g. `anthropic/claude-sonnet-4-20250514`) |
| `--container` | container benchmarks only | — | `docker` or `podman` |
| `--scaffold` | no | `openclaw` | `openclaw` (structured tools) |
| `--benchmark` | no | `swe-bench-verified` | Slug from `configs/benchmarks/<slug>.yaml` |
| `--mcp-config` | **yes for openclaw** | `None` | YAML path or literal `none` |
| `--max-iterations` | no | `100` | Max agent loop iterations per task |
| `--sample` | no | all | Run only first N tasks |
| `--instance-ids` | no | all | Comma-separated instance IDs |
| `--max-context-tokens` | no | `256000` | Sliding window token budget |
| `--prompt-template` | no | from YAML | Override under `configs/prompts/swe_rebench/` |
| `--min-free-disk-gb` | no | `30.0` | Disk space preflight |
| `--run-id` | no | auto | Resume an interrupted run (pass existing run dir) |
| `--api-base` | no | from provider | Override API base URL |
| `--api-key` | no | from env | Override API key |

### Provider System

Defined in `src/llm_call/providers.py`. Resolution: CLI `--api-key` > env var > error.

| Provider | Env Var | API Base |
|----------|---------|----------|
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `dashscope` | `DASHSCOPE_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `siliconflow` | `SILICONFLOW_API_KEY` | `https://api.siliconflow.com/v1` |

### Scaffold System

| Scaffold | Description | MCP | Tools |
|----------|-------------|-----|-------|
| `openclaw` | Structured tools agent (nanobot) | **mandatory** | filesystem, shell, web, MCP, memory, skills |
| `tongyi-deepresearch` | Vendored ReAct scaffold from Alibaba-NLP/DeepResearch (pinned SHA `f72f75d8`, Apache-2.0) | no | search (Serper), visit (Jina Reader), google_scholar (Serper), parse_file (DashScope docmind), PythonInterpreter (sandbox_fusion) |

For openclaw, `--mcp-config` is enforced: pass a YAML path or the literal `none`.
MCP YAML lives in `configs/mcp/` (e.g. `context7.yaml`).

### Benchmark Plugin Architecture

Benchmarks live in `src/agents/benchmarks/` with YAML config at `configs/benchmarks/<slug>.yaml`.

| Slug | Dataset | Default Split | Default Iterations |
|------|---------|---------------|-------------------|
| `swe-bench-verified` | `princeton-nlp/SWE-bench_Verified` | `test` | 100 |
| `swe-rebench` | `nebius/SWE-rebench` | `filtered` | 100 |
| `terminal-bench` | (custom) | — | 100 |
| `deep-research-bench` | configured in YAML | `test` | 100 |
| `browsecomp` | configured in YAML | `test` | 100 |

BenchmarkConfig fields: `slug`, `display_name`, `harness_dataset`, `harness_split`,
`data_root`, `repos_root`, `trace_root`, `default_max_iterations`,
`selection_n`, `selection_seed`, `default_prompt_template`, `exclude_lite`.

**FORBIDDEN**: hardcoding dataset names in collector/cli/scaffold code. All
benchmark specifics go through the plugin + YAML.

### Container & ARM/x86

Each task runs in a fresh Docker/Podman container. Image reference comes from
the benchmark plugin's `image_name_for(task)`.

- Architecture normalization: `amd64`/`x86_64` → `amd64`, `arm64`/`aarch64` → `arm64`
- SWE-rebench ships fully-qualified x86_64 image URIs; ARM Macs use QEMU emulation
- Image prep: source image → fixed image (permission corrections, cached)
- Environment passthrough: `HTTP_PROXY`, `HTTPS_PROXY`, `PIP_INDEX_URL`, etc.
- Network mode: defaults to host; override via `--network-mode`

### Checkpointing / Resume

```bash
python -m trace_collect.cli --run-id traces/swe-rebench/model/20260414T120000 ...
```

- `load_completed_ids(run_dir)` scans `attempt_*/run_manifest.json` for `"status": "completed"`
- Completed tasks are skipped; failed/missing tasks re-run
- Each re-run increments the attempt number (`attempt_1`, `attempt_2`, ...)
- `run_manifest.json` status: `"completed"` or `"error"`

### Execution Model

Tasks run **sequentially** (one container at a time). A `ThreadPoolExecutor`
prefetches the *next* task's Docker image while the current task executes.

### Retry / Error Handling

- LLM retries: `(2, 4, 8)` second exponential backoff on HTTP 429/500/502/503/504
- Exit statuses: `success`, `error`, `tool_error`, `empty_final_response`, `max_iterations`, `timeout`, `failed`
- Disk space preflight at `--min-free-disk-gb` (default 30 GB)
- Container start timeout: 180s; stop timeout: 30s

---

## Part 2 — Simulate

Replay collected traces to measure timing under different infrastructure.

### Two Modes

| Aspect | `cloud_model` | `local_model` |
|--------|---------------|---------------|
| **LLM calls** | Replayed from source timing (no API calls) | Sent to real local OpenAI-compatible endpoint |
| **Timing source** | Source trace `ts_start`/`ts_end` × `replay_speed` | Actual TTFT + TPOT from live model |
| **Tool execution** | MCP tools (`mcp_*`) replayed from trace; others re-executed in container | All tools re-executed in container |
| **Multi-trace** | Yes (via `--trace-manifest`) | Single trace only (`--source-trace`) |
| **LLM client** | None created (forbidden) | Requires `--api-base`, `--model`, `--api-key` |
| **Use case** | "What if N agents run concurrently?" | "What if we self-host on local vLLM?" |

### Minimal Examples

```bash
# Cloud model: replay 3 traces at 50x speed, Poisson arrival
python -m trace_collect.cli simulate \
  --trace-manifest manifest.json \
  --mode cloud_model \
  --container docker \
  --replay-speed 50 \
  --arrival-mode poisson --arrival-rate-per-s 0.5 --arrival-seed 42

# Local model: measure real TTFT/TPOT against local vLLM
python -m trace_collect.cli simulate \
  --source-trace traces/.../trace.jsonl \
  --mode local_model \
  --provider openai --api-base http://localhost:8000/v1 \
  --api-key dummy --model Qwen/Qwen3-32B \
  --container docker \
  --metrics-url http://localhost:8000/metrics
```

### CLI Flags (simulate)

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--source-trace` | one of | — | Mutually exclusive with `--trace-manifest` |
| `--trace-manifest` | one of | — | JSON array of `{source_trace, task_source?, docker_image?}` |
| `--mode` | no | `local_model` | `local_model` or `cloud_model` |
| `--task-source` | no | `data/swe-rebench/tasks.json` | Path to tasks JSON |
| `--output-dir` | no | `traces/simulate` | Output directory |
| `--container` | no | `docker` | `docker` or `podman` |
| `--network-mode` | no | `host` | Container network mode |
| `--command-timeout` | no | `600.0` | Seconds per command |
| `--replay-speed` | no | `1.0` | Wall-clock acceleration (cloud_model only) |
| `--warmup-skip-iterations` | no | `0` | Tag first N iterations as warmup |
| `--arrival-mode` | no | `closed_loop` | `closed_loop` or `poisson` |
| `--arrival-rate-per-s` | no | — | Required for poisson mode |
| `--arrival-seed` | no | — | RNG seed for reproducibility |
| `--metrics-url` | no | — | vLLM Prometheus endpoint (local_model only; forbidden for cloud_model) |

LLM flags (`--provider`, `--api-base`, `--api-key`, `--model`) required for `local_model` only.

### Arrival Modes

| Mode | Behavior |
|------|----------|
| `closed_loop` | All tasks arrive at t=0, compete for resources |
| `poisson` | Inter-arrival times ~ Exp(rate). Seeded RNG for reproducibility |

Implementation: `harness.runner.build_arrival_offsets()` generates offsets;
cloud_model sessions `asyncio.gather` with per-session delay.

### Trace Manifest Format

```json
[
  {"source_trace": "path/to/trace-a.jsonl"},
  {"source_trace": "path/to/trace-b.jsonl", "docker_image": "custom:latest"},
  {"source_trace": "path/to/trace-c.jsonl", "task_source": "other-tasks.json"}
]
```

Paths resolved relative to manifest directory.

### vLLM Metrics Integration

When `--metrics-url` is set, the simulator snapshots Prometheus metrics per
iteration and stores them in `TraceAction.data.sim_metrics`:

- `PreemptionSnapshot`: `num_preemptions_total`, `gpu_cache_usage_perc`,
  `cpu_cache_usage_perc`, `gpu_prefix_cache_hit_rate`, `cpu_prefix_cache_hit_rate`
- When unset: empty (all-None) snapshots recorded as explicit opt-out

### Container Resource Sampling

`ContainerStatsSampler` runs a background thread at 1s intervals:
- Metrics: CPU %, memory MB, disk I/O MB, network I/O MB, context switches
- Prefers cgroup v2 host-side reads; falls back to `docker exec`-based aggregation
- Output: `resources.json` with `{samples: [...], summary: {...}}`

### Key Dataclasses (simulator.py)

- `LoadedTraceSession` — parsed source trace: actions, iterations, metadata, task
- `PreparedContainer` — container_id + ContainerAgent handle
- `PreparedTraceSession` — loaded session + container + sampler

---

## Part 3 — import-claude-code

Convert Claude Code session JSONL to canonical trace format.

```bash
python -m trace_collect.cli import-claude-code \
  --session ~/.claude/projects/<slug>/<uuid>.jsonl \
  --output-dir traces
```

- `--no-sidechains` skips folding `subagents/agent-*.jsonl` into output
- Output: `<output-dir>/claude-code-import/<uuid>/<uuid>.jsonl`
- Handles the `toolUseResult` string-vs-dict gotcha in CC JSONL format

---

## Part 4 — inspect

Post-hoc CLI for querying traces.

```bash
python -m trace_collect.cli inspect traces/.../trace.jsonl overview
python -m trace_collect.cli inspect traces/.../trace.jsonl timeline --json
python -m trace_collect.cli inspect traces/.../trace.jsonl tools --agent django__django-12345
```

Subcommands: `overview`, `step`, `messages`, `response`, `events`, `tools`, `search`, `timeline`.
Filters: `--agent`, `--role`, `--category`, `--iteration`. Output: `--json`.

---

## Trace Schema (v5)

Format: JSONL, one JSON record per line. Four record types:

### trace_metadata (first line)

```json
{
  "type": "trace_metadata",
  "trace_format_version": 5,
  "scaffold": "openclaw",
  "benchmark": "swe-rebench",
  "model": "openrouter/anthropic/claude-sonnet-4-20250514",
  "api_base": "https://openrouter.ai/api/v1",
  "instance_id": "django__django-12345",
  "max_iterations": 100,
  "prompt_template": "cc_aligned",
  "agent_runtime_mode": "host_controller",
  "scaffold_capabilities": {"tools": ["bash"], "memory": false, "skills": false},
  "run_config": {"mcp_config": "context7.yaml"}
}
```

### action (LLM call or tool execution)

```json
{
  "type": "action",
  "action_type": "llm_call",
  "action_id": "llm_0",
  "agent_id": "django__django-12345",
  "iteration": 0,
  "ts_start": 1234567890.123,
  "ts_end": 1234567895.456,
  "data": {
    "llm_wall_latency_ms": 5000.0,
    "llm_call_time_ms": 4800.0,
    "prompt_tokens": 1200,
    "completion_tokens": 350
  }
}
```

`action_type`: `llm_call` | `tool_exec`.
Timing fields in `data`: `llm_wall_latency_ms` (local wall), `llm_call_time_ms`
(preferred), `openrouter_generation_time_ms` (provider-side).

### event

```json
{
  "type": "event",
  "category": "SCHEDULING",
  "ts": 1234567890.123,
  "iteration": 0,
  "description": "..."
}
```

Categories: `SCHEDULING`, `SESSION`, `CONTEXT`, `TOOL`, `LLM`, `MCP`, `MEMORY`, `SUBAGENT`.

### summary (per-agent, at end)

```json
{
  "type": "summary",
  "agent_id": "django__django-12345",
  "success": true,
  "n_iterations": 15,
  "elapsed_s": 120.5
}
```

### Simulation-specific fields in action.data

When produced by `simulate`:
- `sim_metrics.timing`: `{ttft_ms, tpot_ms, total_ms}` (local_model)
- `sim_metrics.warmup`: bool
- `sim_metrics.source`: `"replayed_from_trace"` or `"executed_in_container"`
- `replay_mode`, `replay_speed` (cloud_model)
- `source_llm_latency_ms`, `source_duration_ms`

---

## Output Directory Layout

### collect

```
traces/<benchmark>/<safe-model>/<timestamp>/        # run_dir
  results.jsonl                                     # per-task summary
  <instance_id>/
    attempt_1/                                      # (or attempt_2, ...)
      run_manifest.json
      trace.jsonl           # canonical trace
      results.json
      resources.json        # container stats
      tool_calls.json       # flattened tool calls
      container_stdout.txt
```

### simulate

```
traces/simulate/
  simulate_<model-or-mode>_<timestamp>.jsonl   # combined trace
  <agent_id>/
    attempt_1/
      trace.jsonl          # per-task trace
      resources.json
```

### Artifact Files

| File | Description |
|------|-------------|
| `run_manifest.json` | Status + metadata + artifact pointers (schema v1) |
| `trace.jsonl` | Canonical trace (v5 JSONL) |
| `results.json` | Task result summary |
| `resources.json` | `{samples: [...], summary: {...}}` |
| `tool_calls.json` | Flattened `{timestamp, tool, id, input, duration_ms, result_preview}` |
| `container_stdout.txt` | Raw container logs |
| `results.jsonl` | Run-level summary (one line per task) |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point (argparse, subcommand dispatch) |
| `collector.py` | Collection orchestration, image prefetch, sequential dispatch |
| `simulator.py` | Simulation: cloud/local model replay, container prep, arrival offsets |
| `attempt_pipeline.py` | Per-attempt lifecycle: container start → agent run → artifact write |
| `attempt_layout.py` | Canonical artifact filenames and writers |
| `claude_code_import.py` | CC session → canonical trace converter |
| `runtime/task_container.py` | In-container entrypoint and runtime bootstrap |
| `runtime/entrypoint.py` | Container-side JSON stdin/stdout protocol |
| `src/llm_call/providers.py` | Provider registry |
| `src/llm_call/config.py` | `resolve_llm_config()` |
| `src/agents/benchmarks/base.py` | `Benchmark` ABC + `BenchmarkConfig` |
| `src/agents/base.py` | `TraceAction`, `LLMCallResult`, retry logic |
| `src/harness/runner.py` | `build_arrival_offsets()` |
| `src/harness/container_stats_sampler.py` | `ContainerStatsSampler` |
| `src/harness/metrics_client.py` | `VLLMMetricsClient` |
| `src/harness/trace_logger.py` | `TraceLogger` (JSONL writer) |
