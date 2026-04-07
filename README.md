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

## Benchmarks

Supported benchmarks are loaded via the plugin registry in
`src/agents/benchmarks/`. Each benchmark ships a Python plugin class
plus a YAML config in `configs/benchmarks/<slug>.yaml`.

Currently registered:

| Slug | Dataset | Split | Docker | Scaffolds |
|---|---|---|---|---|
| `swe-bench-verified` | `princeton-nlp/SWE-bench_Verified` | `test` | `swebench/sweb.eval.x86.*` (namespace-prefixed) | mini-swe-agent, openclaw |
| `swe-rebench` | `nebius/SWE-rebench` | `filtered` | `swerebench/sweb.eval.x86_64.*` (fully qualified) | mini-swe-agent, openclaw |

### Running a benchmark

    conda activate ML
    make download-swe-rebench          # (or download-swebench-verified)
    make setup-swe-rebench-repos       # (or setup-swebench-repos)
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
make help
make sync
make verify-bootstrap
make verify-env1
make verify-env2
make verify-env3a
make verify-env3b
make verify-env3c
make verify-env4
make verify-env5
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
  report confirms the expected `A100-PCIE-40GB`, at least `40 GiB` of GPU
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

## ENV-3a

`ENV-3a` covers the raw vLLM baseline:

- `scripts/serve_vllm.sh` installs the configured vLLM package into
  `.venv-server`, launches the OpenAI-compatible server, and runs readiness
  checks against `/v1/models`, `/v1/chat/completions`, and `/metrics`.
- `src/serving/engine_launcher.py` owns the launch command contract.
- `src/serving/health_check.py` writes a JSON server report after the readiness
  checks pass.

As with earlier environment checkpoints, actual acceptance still requires
running the script on the approved server.

## ENV-3b

`ENV-3b` covers the Continuum fork path:

- `scripts/serve_continuum.sh` clones or updates the official Continuum repo,
  installs it into `.venv-continuum`, records the resolved repo commit and
  package versions, starts the Continuum server, and runs repeated
  `program_id`-aware health checks.
- `src/serving/continuum_launcher.py` owns the Continuum launch contract.
- `src/serving/health_check.py` is reused for repeated multi-turn validation.

`ENV-3b` only treats a run as acceptance-ready when `CONTINUUM_REF` is pinned to
an immutable commit or tag and the repeated checks observe prefix-cache hit
metrics above zero.

As with the other environment checkpoints, actual acceptance still requires
running the script on the approved server.

## ENV-4

`ENV-4` ties together the existing private GitHub sync workflow:

- `scripts/pull_repo.sh` fast-forwards the current branch only when tracked
  files are clean.
- `scripts/run_smoke.sh` runs the current infrastructure smoke suite.
- `scripts/run_sweep.sh` becomes the real harness entry once HARNESS lands, and
  still fails closed until `src/harness/runner.py` exists.
- `scripts/collect_results.sh` is a real `rsync` wrapper for pulling results
  back from a remote server.

## ENV-5

`ENV-5` adds explicit preemption control and observability around the raw vLLM
path:

- `configs/systems/vllm_baseline.yaml`, `vllm_preserve.yaml`, and
  `vllm_no_preempt.yaml` now encode the preemption-sensitive system variants.
- `src/serving/engine_launcher.py` and `scripts/serve_vllm.sh` now accept
  `--preemption-mode` and `--max-num-seqs`.
- `src/harness/scheduler_hooks.py` parses `/metrics`, parses scheduler-hook log
  lines, and emits a JSON report that distinguishes cumulative snapshots,
  baseline deltas, and run-specific scheduler-hook evidence.

## ENV-3c

`ENV-3c` covers the ThunderAgent middleware path:

- `scripts/serve_thunderagent.sh` clones or updates the official ThunderAgent
  repo, installs it into `.venv-thunderagent`, records the resolved repo commit
  and runtime package versions, starts the proxy, and verifies program/profile
  tracking through the proxy APIs.
- `src/serving/thunderagent_launcher.py` owns the proxy launch contract.
- `src/serving/thunderagent_check.py` validates `/v1/models`, repeated
  `program_id` requests, `/programs`, `/profiles/{program_id}`, and `/metrics`.

`ENV-3c` only treats a run as acceptance-ready when `THUNDERAGENT_REF` is
pinned to an immutable commit or tag and the proxy reports program/profile state
for the repeated `program_id`.

As with the other environment checkpoints, actual acceptance still requires
running the script on the approved server.
