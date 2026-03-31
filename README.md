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
│   ├── analysis/
│   ├── harness/
│   └── serving/
└── tests/
```

## Development Workflow

```bash
make help
make sync
make verify-bootstrap
make test
```

All nontrivial checkpoints must pass targeted tests, undergo an independent
review, and then be committed with the checkpoint identifier in the commit
message.

Smoke-only subsets belong in dedicated `*_smoke.yaml` workload configs. Default
workload configs should describe the full benchmark dataset path.
