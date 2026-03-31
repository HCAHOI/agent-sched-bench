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
make verify-env1
make verify-env2
make test
```

All nontrivial checkpoints must pass targeted tests, undergo an independent
review, and then be committed with the checkpoint identifier in the commit
message.

Smoke-only subsets belong in dedicated `*_smoke.yaml` workload configs. Default
workload configs should describe the full benchmark dataset path.

## ENV-1

`ENV-1` provides the first real server-side asset:

- `scripts/setup_server.sh` installs Ubuntu base packages, `uv`, Python 3.11,
  a repo-local server venv, CUDA-enabled `torch`, and writes an environment
  report.
- `scripts/report_server_env.py` records GPU visibility, memory, free disk, and
  SSH-key presence to JSON for auditability, and `ENV-1` only succeeds when the
  report confirms the expected `A100-SXM-40GB`, at least `40 GiB` of GPU
  memory, `CUDA >= 12.1`, and `torch.cuda.is_available()`.

Run the setup script only on the target Ubuntu server after approval.

## ENV-2

`ENV-2` adds the model download and verification path:

- `scripts/download_model.sh` downloads `Llama-3.1-8B-Instruct` through either
  HuggingFace Hub or ModelScope into `/data/models/...`, writes `MODEL_PATH`
  into `.env`, and then runs the verification step.
- `scripts/report_model_artifact.py` audits the downloaded artifact, records the
  resolved package versions, inventories files, and by default performs a full
  `AutoModelForCausalLM.from_pretrained()` check before declaring success.

`ENV-2` acceptance requires the full load path. Config-only verification may be
used for debugging, but it is explicitly non-acceptance.

The real download and load check still must run on the approved server after
checkpoint approval.
