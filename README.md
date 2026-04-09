# agent-sched-bench

Benchmark environment for studying agent scheduling and KV-cache management on
multi-step LLM workloads. The repository is organized around explicit
checkpoints so each infrastructure step can be reviewed, committed, and
reproduced independently.

## Scope

- `docs/agent_benchmark_spec.md` is the product and experiment source of truth.
- `AGENTS.md` and `CLAUDE.md` define research-integrity and process rules.
- Real downloads, serving, and experiment runs are scripted but gated for manual
  approval at checkpoint boundaries.

## Planned Checkpoints

1. `BOOTSTRAP-0`: repository scaffold, tooling, configs, and plan docs
2. `ENV-1..5`: environment, model, serving, sync, and preemption controls
3. `AGENT-1..4`: base interface plus code, data, and research agents
4. `HARNESS-1..3`: runner, metrics, and trace logging
5. `ANALYSIS-1` and `REPLAY-1`: trace processing and replay fallback

## Repository Layout

```text
agent-sched-bench/
├── configs/
├── docs/
├── scripts/
├── src/
│   ├── agents/
│   ├── harness/
│   └── serving/
└── tests/
```

## Benchmarks

Supported benchmarks are loaded via the plugin registry in
`src/agents/benchmarks/`. Each benchmark ships a Python plugin class
plus a YAML config in `configs/benchmarks/<slug>.yaml`.

Currently registered:

| Slug | `task_shape` | Dataset | Split | Docker | Scaffolds | Scoring |
|---|---|---|---|---|---|---|
| `swe-bench-verified` | `swe_patch` | `princeton-nlp/SWE-bench_Verified` | `test` | `swebench/sweb.eval.x86.*` (namespace-prefixed) | mini-swe-agent, openclaw | harness (pytest in container) |
| `swe-rebench` | `swe_patch` | `nebius/SWE-rebench` | `filtered` | `swerebench/sweb.eval.x86_64.*` (fully qualified) | mini-swe-agent, openclaw | harness (pytest in container) |

### Running a benchmark

    conda activate ML
    make download-swe-rebench          # or download-swebench-verified
    make setup-swe-rebench-repos       # or setup-swebench-repos
    PYTHONPATH=src python -m trace_collect.cli \
        --provider dashscope \
        --benchmark swe-rebench \
        --scaffold openclaw \
        --sample 2

Flags the collect CLI accepts: `--benchmark <slug>` (default
`swe-bench-verified`), `--scaffold mini-swe-agent|openclaw`,
`--sample N` (optional task cap), `--instance-ids a,b,c` (optional
explicit list), `--run-id <path>` (resume an interrupted run).

### Adding a new benchmark

See `docs/benchmark_plugin_spec.md` for the full plugin protocol. Short version:
1. Create `src/agents/benchmarks/<slug>.py` with a Benchmark subclass.
2. Create `configs/benchmarks/<slug>.yaml` with BenchmarkConfig fields.
3. Register the class in `src/agents/benchmarks/__init__.py::REGISTRY`.
4. Add `tests/test_<slug>_plugin.py` with unit tests for normalize_task
   and any benchmark-specific quirks.
5. Add `make download-<slug>` / `make setup-<slug>-repos` Makefile targets.

## Development Workflow

```bash
make help    # list all targets
make sync    # install dependencies
make test    # run full test suite
make lint    # run ruff
```

## Gantt Viewer

The old static `python -m trace_collect.cli gantt ...` flow has been replaced
by the dynamic viewer under `demo/gantt_viewer/`.

Useful entrypoints:

```bash
make gantt-viewer-install
make gantt-viewer-dev
make gantt-viewer-build
make gantt-viewer-test
```

The viewer discovers the shipped acceptance traces from
`demo/gantt_viewer/configs/example.yaml`, serves API routes from FastAPI, and
in production mounts the built frontend from `demo/gantt_viewer/frontend/dist`.
See `demo/gantt_viewer/README.md` for the exact workflow and acceptance checks.
For agent-oriented runtime trace management, use
`demo/gantt_viewer/AGENT_INTERFACE.md`.
The viewer only accepts canonical trace JSONL; convert raw Claude Code
sessions first via `python -m trace_collect.cli import-claude-code`.

Smoke-only subsets belong in dedicated `*_smoke.yaml` workload configs. Default
workload configs should describe the full benchmark dataset path.
