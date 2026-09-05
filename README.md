# agent-sched-bench

Research environment for measuring and scheduling multi-step LLM agents. The
repository covers four connected workflows:

1. collect benchmark agent traces with raw model/tool outputs and timing;
2. replay the recorded tool trajectory under controlled concurrency;
3. observe and predict clause-level latency, CPU, RSS, and disk behavior; and
4. evaluate CPU/GPU serving and scheduling policies against paper baselines.

Trace collection uses the registered remote/Codex providers. GPU-backed vLLM
is optional and is used by serving and fixed-trajectory shadow-replay
experiments, not as an alternative collection provider.

Before changing tool-resource data, prediction, evaluation, or scheduler
integration, read
[`analysis/development/tool-resource-canonical-objective.md`](analysis/development/tool-resource-canonical-objective.md).
It is the authoritative objective and evidence boundary; older development
notes are not.

## Setup

The single environment entry point is:

```bash
bash scripts/setup/benchmark_server.sh
source .venv/bin/activate
```

The script installs Python 3.12 and the dependencies declared by the project,
verifies Docker, installs the eBPF prerequisite when needed, and creates
`.venv`. On a GPU serving node use:

```bash
bash scripts/setup/benchmark_server.sh --gpu
```

`--gpu` installs the pinned `serving-spike` extra and verifies CUDA, torch, and
vLLM. The repository does not use conda. Run `make help` for development and
download targets.

## Where Things Live

| Path | Purpose |
|---|---|
| [`configs/`](configs/README.md) | Benchmark definitions, frozen corpus definitions, prompts, replay, and serving configs. |
| [`data/`](data/README.md) | Materialized benchmark metadata and repositories on this machine. |
| [`traces/`](traces/README.md) | Raw collections and replay outputs; most contents are intentionally untracked. |
| `trace_archives/` | Compact retained trace archives checked into this checkout. |
| `outputs/traces/` | Shareable consolidated trace bundles and their checksums. |
| [`analysis/`](analysis/README.md) | Current research authority chain and result artifacts. |
| `scripts/baselines/` | Paper baseline adapters and fidelity boundaries. |
| `scripts/evaluation/` | Physical runners and result evaluators. |
| [`src/tool_resource/`](src/tool_resource/README.md) | Clause telemetry, causal resource KB, predictor, and service interfaces. |
| `src/trace_collect/` | Collection, resume, replay, and trace-artifact implementation. |
| `src/agents/benchmarks/` | Benchmark plugins paired with `configs/benchmarks/*.yaml`. |
| `tests/` | Focused regression and evaluation-semantics tests. |

The three locations are deliberately different: `configs/corpora/` defines
cohorts, `data/` holds task metadata, and `traces/` holds executions. A corpus
JSON is not proof that every referenced trace is present or valid. See the
linked data and trace maps before starting an evaluation.

## Collect Traces

Benchmark-specific dataset, image, selection, and prompt defaults live in
`configs/benchmarks/<slug>.yaml`.

```bash
PYTHONPATH=src python -m trace_collect.cli \
  --provider codex \
  --model gpt-5.6-sol \
  --benchmark swe-rebench \
  --scaffold openclaw \
  --container docker \
  --mcp-config none \
  --max-iterations 100 \
  --concurrency 2 \
  --sample 2
```

Registered providers are `openrouter`, `dashscope`, `openai`, `siliconflow`,
`deepseek`, `pioneer`, and `codex`. Use `--service-tier fast` only with Codex.
For observation-only clause telemetry add `--tool-resource-telemetry clause`;
this does not query, update, or persist the resource KB.

Resume an interrupted collection with `--run-id <existing-run-directory>`.
The exact acceptance rules are documented in [OPERATIONS.md](OPERATIONS.md#resume).

## Replay Traces

Replay executes the recorded tool calls in real task containers. By default it
uses source-trace LLM timing and issues no new model requests:

```bash
PYTHONPATH=src:. uv run python -m trace_collect.cli simulate \
  --manifest /abs/path/to/manifest.yaml \
  --container docker \
  --concurrency 4 \
  --workers 4 \
  --prep-concurrency 4 \
  --replay-speed 20 \
  --output-dir /abs/path/to/output
```

`--replay-speed` scales recorded gaps and synthetic LLM sleeps, never real tool
execution, timeouts, or telemetry clocks. Fixed-trajectory GPU evaluation adds
`--shadow-llm-api-base` and `--shadow-llm-model`; the policy is selected with
`--shadow-llm-mode`. This measures serving behavior without letting newly
generated text alter the recorded action sequence.

See [OPERATIONS.md](OPERATIONS.md) for manifest format, resource monitoring,
large-corpus disk constraints, and daemon-backed tool-resource replay.

## Registered Benchmarks

| Slug | Runtime | Dataset source |
|---|---|---|
| `swe-bench-verified` | task container | `princeton-nlp/SWE-bench_Verified` |
| `swe-rebench` | task container | `nebius/SWE-rebench` |
| `terminal-bench` | host controller | pinned local Terminal-Bench registry |

Add benchmark behavior through `src/agents/benchmarks/` and
`configs/benchmarks/<slug>.yaml`; do not add dataset-specific collector flags
or hardcode dataset names in the collection core.

## Inspect Traces

```bash
PYTHONPATH=src python -m trace_collect.cli gantt-serve
PYTHONPATH=src python -m trace_collect.cli gantt-export --help
```

The interactive viewer lives in `demo/gantt_viewer/`.
