# agent-sched-bench (`dev/cpu-only`)

CPU/cloud-provider-only branch for collecting and replaying multi-step agent
traces. This branch intentionally removes local model backends, vLLM,
HuggingFace model execution, KV/sparse-attention experiments, and GPU profiling.

Top-level capabilities:

1. **Trace collect** — run agent scaffolds on benchmark tasks and record canonical
   JSONL traces with cloud/OpenAI-compatible model providers.
2. **Trace simulate** — replay collected traces under bounded concurrency using
   source-trace timing; no LLM requests are issued during replay.
3. **Gantt viewer demo** — inspect traces as multi-lane Gantt charts under
   `demo/gantt_viewer/`.

## Repository Layout

```text
agent-sched-bench/
├── configs/            # benchmark, prompt, MCP, and replay YAMLs
├── demo/gantt_viewer/  # FastAPI backend + Solid.js frontend
├── scripts/            # setup, download, smoke, and utility shells
├── src/
│   ├── agents/         # scaffolds + benchmark plugins
│   ├── harness/        # container/runtime samplers and trace logger
│   ├── llm_call/       # provider registry + OpenAI-compatible client
│   └── trace_collect/  # CLI: collect / simulate / gantt-serve / gantt-export
└── tests/
```

## Development Workflow

Use `uv` and the project `.venv` at the repo root:

```bash
uv sync --extra dev
source .venv/bin/activate
make help
make test
make lint
```

## Trace Collect

Run an agent scaffold on a benchmark and record a canonical v5 JSONL trace per
task. The CLI requires explicit `--provider` and `--model`; benchmark specifics
come from `configs/benchmarks/<slug>.yaml`.

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider dashscope \
    --model qwen-plus-latest \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 2
```

Key flags: `--benchmark <slug>`, `--scaffold openclaw`, `--mcp-config` (required
for OpenClaw; YAML path or literal `none`), `--skip N`, `--sample N`,
`--instance-ids a,b,c`, `--concurrency N`, `--run-id <path>`,
`--prompt-template <name>`, and provider sampling flags `--temperature`,
`--top-p`, `--top-k`, `--repetition-penalty`.

Supported providers live in `src/llm_call/providers.py`: `openrouter`,
`dashscope`, `openai`, `siliconflow`, and `deepseek`. Use `--api-base` and
`--api-key` for OpenAI-compatible gateways when the built-in provider URL or env
var is not enough.

## Trace Simulate

Replay collected traces with bounded concurrency using source action timing. This
branch supports cloud replay only; there is no local model/vLLM mode.

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --manifest /abs/path/to/simulate-manifest.yaml \
    --concurrency 1,2,4,8 \
    --container docker \
    --replay-speed 50
```

`--concurrency 8` runs one bounded-queue replay with at most 8 active traces.
`--concurrency 1,2,4,8` runs a sweep and writes `throughput_sweep.jsonl`.
By default, replay sleeps source inter-action gaps and action durations scaled by
`--replay-speed`. To replace source LLM durations with a fixed model, pass
`--llm-timing ttft-tpot --llm-ttft-ms <ms> --llm-tpot-ms <ms>`; tool timing and
inter-action gaps still use source timing scaled by `--replay-speed`.

Manifest input is YAML. The simplest form is a list of absolute trace paths:

```yaml
- /abs/path/task-a/attempt_1/trace.jsonl
- /abs/path/task-b/attempt_1/trace.jsonl
```

Structured entries can override task source or image:

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

## Registered Benchmarks

| Slug | Task shape | Dataset | Scaffolds |
|---|---|---|---|
| `swe-bench-verified` | `swe_patch` | `princeton-nlp/SWE-bench_Verified` | openclaw |
| `swe-rebench` | `swe_patch` | `nebius/SWE-rebench` | openclaw |
| `terminal-bench` | `terminal_task` | Terminal-Bench tasks | openclaw |

Dataset names, image namespaces, and CLI-visible defaults must live in YAML —
not in `collector.py`, `cli.py`, or scaffold code.

## Viewing Traces

Use the Gantt viewer demo for interactive trace inspection:

```bash
PYTHONPATH=src python -m trace_collect.cli gantt-serve
```

For static exports, use:

```bash
PYTHONPATH=src python -m trace_collect.cli gantt-export --help
```

See `OPERATIONS.md` for detailed operator reference: collect/simulate flags,
resume semantics, task-container env vars, Gantt viewer, and benchmark plugin
rules.

## Explicitly removed from this branch

- `vllm`, `torch`, `transformers`, `accelerate` dependencies
- `src/serving/` local backend code
- local-HF recording, KV eviction, sparse attention, and per-head artifacts
- vLLM serving, metrics, startup parsing, and scheduler hooks
- GPU / `nvidia-smi` profiling and `profile-gpu`
