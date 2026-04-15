# Plan: Multi-Benchmark Expansion (Ralplan R3 Consensus)

**Date**: 2026-04-15
**Branch**: `feat/multi-benchmark`
**Status**: Approved (Architect APPROVE + Critic APPROVE, 2 iterations)
**Estimated Duration**: 12-16 days

## Goal

Expand the benchmark system from SWE-only to multi-domain:
- **SWE/code**: swe-bench-verified, swe-rebench (existing)
- **Terminal**: terminal-bench (existing)
- **Deep research**: DeepResearchBench (new)
- **Browsing comprehension**: browsecomp (new)

Add **Qwen Deep Research** as a new scaffold for research-style benchmarks.
Remove **mini-swe-agent** scaffold (no longer maintained).
Ensure trace collection, simulation, and visualization work for all benchmark×scaffold combinations.

## RALPLAN-DR Summary

### Principles
1. **Plugin purity** — benchmark specifics only in plugin + YAML, never in collector/cli
2. **Scaffold-benchmark orthogonality** — independent axes, capability matrix is the only link
3. **Trace format universality** — v5 JSONL accommodates all benchmark types without branching
4. **Execution environment abstraction** — support both container and host modes
5. **Maximal code reuse** — new scaffolds reuse `src/llm_call/`, `TraceLogger`, resource samplers

### Decision
**Option A: Thin Abstraction Layer** chosen over Option B (Full Runner Protocol).

Option B invalidated: deep research benchmarks are fundamentally simpler (no containers, no test patches, no code edits). A full Runner protocol over-engineers the simple case. The existing `benchmark.build_runner()` + `runtime_mode_for()` pattern (already validated by terminal-bench) provides sufficient extension points.

### Scaffold × Benchmark Capability Matrix (target state)

| Benchmark | openclaw | qwen-deep-research |
|-----------|----------|---------------------|
| swe-bench-verified | ✅ container | ✗ |
| swe-rebench | ✅ container | ✗ |
| terminal-bench | ✅ host | ✗ |
| deep-research-bench | ✅ host | ✅ host |
| browsecomp | ✅ host | ✅ host |

---

## Phase 0: Infrastructure + Schema (0.5 day)

### 0.1 Create branch
```bash
git checkout -b feat/multi-benchmark
```

### 0.2 Trace metadata schema extension
**Edit**: `src/harness/trace_logger.py`
- Add optional `execution_environment` field to `log_metadata()` (default: `"container"`)
- **Backward compatibility**: all trace readers (simulator, inspector, Gantt viewer) use `.get("execution_environment", "container")` when parsing metadata

### 0.3 Define minimal Runner Protocol
**Edit**: `src/agents/benchmarks/base.py`
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Runner(Protocol):
    async def run_task(
        self, task: dict[str, Any], *, attempt_ctx: Any, prompt_template: str
    ) -> Any: ...
```
- Existing `SWEBenchRunner` already has compatible `run_task()` — natural conformance
- `TerminalBenchRunner` has `run_openclaw_task()` — adapt via thin wrapper in `build_runner()` or rename

### Acceptance Criteria
- [ ] `trace_logger.py` `log_metadata()` accepts `execution_environment` parameter
- [ ] `Runner` Protocol importable: `from agents.benchmarks.base import Runner`
- [ ] `make test` passes (pure additive, no breakage)

---

## Phase 1a: Generalize Collector Dispatch (1.5 days)

### 1a.1 `container_executable` becomes `str | None`

Downstream chain (every function touched):

| File | Function/Parameter | Change |
|------|-------------------|--------|
| `cli.py:66-69` | `--container` argparse | `required=False`, default=`None` |
| `cli.py:340` | `container_executable=args.container` | Passes `None` for host-mode |
| `collector.py:697` | `collect_traces(container_executable: str)` | → `str \| None` |
| `collector.py:343` | `_run_scaffold_tasks(container_executable: str)` | → `str \| None` |
| `collector.py:399-420` | `_ensure_task_source_ready()` / prefetch | Guard: `if source_image is None: skip` (existing per-task None check) |
| `collector.py:463-470` | `_cleanup_task_images()` | Guard: `if source_image is None and fixed_image is None: skip` |
| `collector.py:422-427` | `run_attempt()` | `container_executable: str \| None` |
| `attempt_pipeline.py` | `run_attempt()`, `start_task_container()` | Guard: host-mode tasks skip container functions |

When `source_image is None`:
- Skip `_ensure_task_source_ready()`
- Skip image prefetch (`executor.submit`)
- Skip `_cleanup_task_images()`
- `ThreadPoolExecutor` still created (cheap) but never receives work

### 1a.2 Unified `collect_traces()` entry point

Target signature:
```python
async def collect_traces(
    *,
    scaffold: str,
    provider_name: str | None = None,
    container_executable: str | None = None,
    benchmark_config_path: Path,
    ...
) -> Path:
```

Target implementation (replaces if/elif/else scaffold dispatch):
```python
async def collect_traces(...) -> Path:
    config = BenchmarkConfig.from_yaml(benchmark_config_path)
    benchmark_cls = get_benchmark_class(config.slug)
    benchmark = benchmark_cls(config)
    benchmark.validate_scaffold_support(scaffold)

    # Validate container requirement
    if benchmark.execution_environment == "container" and container_executable is None:
        raise ValueError("--container required for container-mode benchmarks")

    tasks = benchmark.load_tasks()
    tasks = _select_tasks(tasks, benchmark, ...)
    run_dir = build_run_dir(benchmark, model)

    # Unified dispatch via build_runner()
    runner = benchmark.build_runner(scaffold=scaffold, provider=..., ...)

    def inner_factory(task):
        async def _inner(attempt_ctx):
            return await runner.run_task(task, attempt_ctx=attempt_ctx, ...)
        return _inner

    return await _run_scaffold_tasks(
        benchmark=benchmark, tasks=tasks, run_dir=run_dir,
        inner_factory=inner_factory, container_executable=container_executable, ...
    )
```

**Delete**: `collect_miniswe_traces()` and `collect_openclaw_traces()` as separate functions — logic merged into unified `collect_traces()`.

### Acceptance Criteria
- [ ] `collect_traces()` accepts any scaffold, dispatches via `build_runner()`
- [ ] `--container` omittable for host-mode benchmarks
- [ ] Existing openclaw + swe-rebench collection unchanged (regression tests pass)
- [ ] `grep -n 'collect_miniswe_traces\|collect_openclaw_traces' src/` returns 0 results

---

## Phase 1b: Remove mini-swe-agent (1 day)

### Complete File List (31 files)

| File | Action | Details |
|------|--------|---------|
| `src/agents/miniswe/agent.py` | **DELETE** | Entire file |
| `src/agents/miniswe/__init__.py` | **DELETE** | Entire file |
| `src/llm_call/miniswe.py` | **DELETE** | Entire file |
| `src/llm_call/__init__.py` | **MODIFY** | Remove `build_miniswe_litellm_model_name` import + `__all__` entry |
| `src/llm_call/providers.py` | **MODIFY** | Remove `miniswe_litellm_prefix` field from `ProviderDefinition` and all PROVIDERS entries |
| `src/llm_call/config.py` | **MODIFY** | Remove miniswe references |
| `src/trace_collect/cli.py` | **MODIFY** | Remove `"miniswe"` from `--scaffold` choices, default → `"openclaw"` |
| `src/trace_collect/collector.py` | **MODIFY** | Delete `_run_miniswe_in_task_container()` (~120 lines) |
| `src/trace_collect/runtime/task_container.py` | **MODIFY** | Remove miniswe branches |
| `src/trace_collect/runtime/entrypoint.py` | **MODIFY** | Remove miniswe references |
| `src/agents/base.py` | **MODIFY** | Remove miniswe references |
| `src/agents/benchmarks/swe_bench_verified.py` | **MODIFY** | Remove `runtime_mode_for("miniswe")` branch |
| `src/agents/benchmarks/swe_rebench.py` | **MODIFY** | Remove miniswe SUPPORTED_SCAFFOLDS and runtime_mode branch |
| `src/harness/runner.py` | **MODIFY** | Remove miniswe references |
| `src/agents/openclaw/eval/prepare.py` | **MODIFY** | Remove miniswe reference |
| `Makefile` | **MODIFY** | Remove `smoke-swe-rebench-miniswe` and similar targets |
| `README.md` | **MODIFY** | Remove miniswe scaffold documentation |
| `pyproject.toml` | **MODIFY** | Remove `mini-swe-agent` dependency |
| `src/trace_collect/CLAUDE.md` | **MODIFY** | Update scaffold table, remove miniswe row |
| `docs/CURRENT_PLAN.md` | **REPLACE** | This plan |
| `tests/test_miniswe_container_runtime.py` | **DELETE** | miniswe-only test file |
| `tests/test_session_runner_actions.py` | **MODIFY** | Remove miniswe references |
| `tests/test_terminal_bench_plugin.py` | **MODIFY** | Remove miniswe references |
| `tests/test_task_container_runtime.py` | **MODIFY** | Remove miniswe branch tests |
| `tests/test_openclaw_runtime_selection.py` | **MODIFY** | Remove miniswe option tests |
| `tests/test_collector_task_container_runtime.py` | **MODIFY** | Remove miniswe collection tests |
| `tests/test_attempt_pipeline.py` | **MODIFY** | Remove miniswe references |
| `tests/test_collector_runtime_mode.py` | **MODIFY** | Remove miniswe dispatch tests |
| `tests/test_sweep.py` | **MODIFY** | Remove miniswe references |
| `tests/test_swe_rebench_plugin.py` | **MODIFY** | Remove miniswe scaffold tests |
| `tests/test_llm_call_config.py` | **MODIFY** | Remove `build_miniswe_litellm_model_name` tests |

> **Note**: Executor should run `grep -rn 'miniswe\|mini.swe\|MiniSWE\|mini_swe' src/ tests/` at execution time for authoritative file list.

### Acceptance Criteria
- [ ] `grep -rn 'miniswe\|mini.swe\|MiniSWE\|mini_swe' src/ tests/` returns 0 results
- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] `python -m trace_collect.cli --help` shows `--scaffold` without `miniswe`
- [ ] `python -c "from llm_call import build_miniswe_litellm_model_name"` raises ImportError

### 🔒 REVIEW GATE: Phase 1b complete — verify miniswe fully removed before proceeding

---

## Phase 2: Modular Benchmark Interface (1.5-2 days)

### 2.1 Benchmark base class: add `execution_environment`
**Edit**: `src/agents/benchmarks/base.py`
```python
@property
def execution_environment(self) -> str:
    """Return 'container' or 'host'. Default: 'container'."""
    return "container"
```
Only this property — `evaluation_method` and `requires_network` deferred until consumers exist.

### 2.2 Capability matrix (runtime probe)
**New**: `src/agents/capabilities.py`
```python
ALL_SCAFFOLDS = ("openclaw", "qwen-deep-research")

def scaffold_benchmark_matrix() -> dict[str, set[str]]:
    """Derive capability matrix by probing registered plugins."""
    from agents.benchmarks import REGISTRY
    matrix: dict[str, set[str]] = {}
    for slug, cls in REGISTRY.items():
        for scaffold in ALL_SCAFFOLDS:
            try:
                # Probe via runtime_mode_for (raises for unsupported)
                # Need lightweight probe without full config
                ...
            except (NotImplementedError, ValueError):
                pass
    return matrix

def validate_scaffold_benchmark(scaffold: str, benchmark_slug: str) -> None:
    """Early CLI validation. Raises ValueError for invalid combinations."""
```

### 2.3 DeepResearchBench plugin
**New**: `src/agents/benchmarks/deep_research_bench.py`, `configs/benchmarks/deep-research-bench.yaml`, `configs/prompts/deep_research_bench/default.txt`
- `slug = "deep-research-bench"`
- `load_tasks()` — from HuggingFace dataset (ID configured in YAML `harness_dataset`)
- `normalize_task()` → `{instance_id, problem_statement, reference_answer, topic, difficulty, domain}`
- `execution_environment` → `"host"`
- `runtime_mode_for()` → `"host_controller"` for all scaffolds
- `build_runner(scaffold="openclaw")` → openclaw runner (host-mode)
- `build_runner(scaffold="qwen-deep-research")` → `QwenDeepResearchRunner`
- `image_name_for()` → `None`

### 2.4 BrowseComp plugin
**New**: `src/agents/benchmarks/browsecomp.py`, `configs/benchmarks/browsecomp.yaml`, `configs/prompts/browsecomp/default.txt`
- `slug = "browsecomp"`
- Data source: HuggingFace (configured in YAML)
- Task schema: `{instance_id, problem_statement, reference_answer, source_urls}`
- Same host-mode pattern as DeepResearchBench

### 2.5 Register new benchmarks
**Edit**: `src/agents/benchmarks/__init__.py` — add to REGISTRY

### Acceptance Criteria
- [ ] `get_benchmark_class("deep-research-bench")` and `get_benchmark_class("browsecomp")` return correct classes
- [ ] Both plugins: `execution_environment` returns `"host"`
- [ ] `validate_scaffold_benchmark("openclaw", "deep-research-bench")` passes
- [ ] `validate_scaffold_benchmark("qwen-deep-research", "swe-rebench")` raises ValueError
- [ ] `scaffold_benchmark_matrix()` returns correct mapping

---

## Phase 3: Qwen Deep Research Scaffold (2 days)

### 3.1 Runner implementation
**New**: `src/agents/qwen_deep_research/__init__.py`, `src/agents/qwen_deep_research/runner.py`
```python
class QwenDeepResearchRunner:
    """Qwen Deep Research API wrapper, conforming to Runner Protocol."""

    def __init__(self, *, model: str, api_base: str, api_key: str,
                 max_iterations: int, benchmark_slug: str):
        self.client = create_async_openai_client(api_base, api_key)
        self.model = model

    async def run_task(self, task: dict, *, attempt_ctx, prompt_template: str) -> Any:
        # 1. Format research query from task['problem_statement']
        # 2. Call Qwen API via self.client (streaming for TTFT/TPOT)
        # 3. Log llm_call action via attempt_ctx.trace_logger
        # 4. Extract answer from response
        # 5. Return result with success/exit_status
```
- Reuses: `create_async_openai_client()`, `TraceLogger`, `summarize_llm_latencies()`
- dashscope provider already registered in `PROVIDERS`

### 3.2 Integration path
- DeepResearchBench/BrowseComp `build_runner(scaffold="qwen-deep-research")` → `QwenDeepResearchRunner`
- Collector dispatches via Phase 1a unified `collect_traces()` → `build_runner()` → `runner.run_task()`

### 3.3 CLI update
- `--scaffold` choices → `["openclaw", "qwen-deep-research"]`
- `--mcp-config` not required for qwen-deep-research

### Acceptance Criteria
- [ ] `isinstance(QwenDeepResearchRunner(...), Runner)` is True
- [ ] Mock API test: given mock Qwen response, produces valid v5 trace JSONL
- [ ] Trace contains `llm_call` action with `prompt_tokens`, `completion_tokens`, timing fields

---

## Phase 4: Simulator Restructuring (3-4 days) — HIGHEST RISK

### Functions requiring modification

| Function | Location | Current Behavior | Host-mode Behavior |
|----------|----------|------------------|-------------------|
| `PreparedContainer` | `simulator.py:47-53` | Required dataclass | Unchanged (only created for container-mode) |
| `PreparedTraceSession` | `simulator.py:57-63` | `container: PreparedContainer` required | `container: PreparedContainer \| None = None` |
| `_validate_loaded_sessions` | `simulator.py:295-341` | L308-313: no docker_image → raise | Skip docker_image check when `metadata.execution_environment == "host"` |
| `_prepare_container_session` | `simulator.py:343-385` | Always starts container | Unchanged (only called for container-mode) |
| **NEW** `_prepare_host_session` | — | — | Returns `PreparedTraceSession(loaded=loaded, container=None)` |
| `simulate()` main loop | `simulator.py:990-997` | All → `_prepare_container_session` | Branch: host → `_prepare_host_session()`, container → `_prepare_container_session()` |
| `simulate()` sampler init | `simulator.py:999-1009` | All → `ContainerStatsSampler` | Branch: host → `ProcessStatsSampler(pid)` or None, container → existing |
| `_run_local_model_simulation` | `simulator.py:476+` | Accesses `prepared.container.agent` for tools | Guard: `if prepared.container is not None: exec tools; else: skip` |
| `_replay_cloud_model_session` | `simulator.py:676/717` | Accesses `prepared.container` | Guard: host-mode sessions replay LLM timing only |
| `_run_cloud_model_replay` | `simulator.py:690/895` | Passes `prepared_session` | Unchanged (guards inside) |
| `simulate()` finally block | `simulator.py:1056-1080` | Unconditional `ctr.agent.stop()` + `stop_task_container()` | Guard: `if prepared.container is not None: stop; else: pass` |

### Key data structure change
```python
@dataclass(slots=True)
class PreparedTraceSession:
    loaded: LoadedTraceSession
    container: PreparedContainer | None = None   # ← NOW OPTIONAL
    sampler: ContainerStatsSampler | None = None  # or ProcessStatsSampler
    task_output_dir: Path | None = None
```

### `container_executable` in simulator
- `simulate()` signature: `container_executable: str = "docker"` → `container_executable: str | None = None`
- Only passed to `_prepare_container_session` when `execution_environment == "container"`

### Acceptance Criteria
- [ ] `simulate --source-trace <host-mode-trace.jsonl> --mode cloud_model` succeeds without Docker
- [ ] `simulate --source-trace <container-trace.jsonl> --mode cloud_model --container docker` unchanged behavior
- [ ] `simulate --source-trace <host-mode-trace.jsonl> --mode local_model --provider dashscope --model ... --api-key ...` completes LLM calls and produces trace
- [ ] Output trace JSONL contains valid `sim_metrics`
- [ ] All existing simulator tests pass

### 🔒 REVIEW GATE: Phase 4 complete — verify simulator works for both container and host modes

---

## Phase 5: Resource Tracking + Gantt (1 day)

### 5.1 ProcessStatsSampler
**New**: `src/harness/process_stats_sampler.py`
- Same interface as `ContainerStatsSampler`: `__init__(pid, interval_s=1.0)`, `start()`, `stop() -> list[dict]`
- Shares `summarize_samples()` from existing code
- Metrics: CPU %, memory MB, disk I/O (optional), context switches (optional)
- Linux: `/proc/<pid>/stat`, `/proc/<pid>/io`
- macOS: `psutil.Process(pid)` (psutil as optional dependency)
- Output: same `resources.json` schema

### 5.2 Collector integration
- Container benchmarks → `ContainerStatsSampler`
- Host benchmarks → `ProcessStatsSampler(runner_pid)`

### 5.3 Gantt viewer verification
- `build_gantt_payload()` in `demo/gantt_viewer/backend/payload.py` is already generic
- Verify research-style traces (mostly `llm_call`, few/no `tool_exec`) render correctly

### Acceptance Criteria
- [ ] `ProcessStatsSampler` instantiable on Linux and macOS
- [ ] Host-mode collection produces valid `resources.json`
- [ ] Gantt viewer loads and renders a deep research trace

---

## Phase 6: Testing (2 days)

### 6.1 New benchmark plugin tests
- `tests/test_deep_research_bench_plugin.py` — mock HuggingFace dataset, verify `load_tasks()`, `normalize_task()`, `execution_environment`
- `tests/test_browsecomp_plugin.py` — same pattern

### 6.2 Qwen runner tests
- `tests/test_qwen_deep_research_runner.py` — mock API calls, verify trace output format, token counting, timing

### 6.3 Capability matrix tests
- `tests/test_capabilities.py` — verify `scaffold_benchmark_matrix()`, `validate_scaffold_benchmark()` for valid/invalid combinations

### 6.4 Simulator integration tests
- Construct minimal host-mode trace fixture (hand-crafted JSONL)
- Test `simulate()` cloud_model + host-mode trace (no container started)
- Regression: existing container-mode trace still works

### 6.5 Regression
- `make test` — full suite, no regressions
- `make lint` — ruff clean

### Acceptance Criteria
- [ ] All new tests pass
- [ ] `make test` no regressions
- [ ] Coverage: every new benchmark plugin, runner, capability function, simulator host-mode path

### 🔒 FINAL REVIEW GATE: All phases complete, full test suite green

---

## File Summary

### New files (14)
| File | Phase |
|------|-------|
| `src/agents/capabilities.py` | 2 |
| `src/agents/benchmarks/deep_research_bench.py` | 2 |
| `src/agents/benchmarks/browsecomp.py` | 2 |
| `src/agents/qwen_deep_research/__init__.py` | 3 |
| `src/agents/qwen_deep_research/runner.py` | 3 |
| `src/harness/process_stats_sampler.py` | 5 |
| `configs/benchmarks/deep-research-bench.yaml` | 2 |
| `configs/benchmarks/browsecomp.yaml` | 2 |
| `configs/prompts/deep_research_bench/default.txt` | 2 |
| `configs/prompts/browsecomp/default.txt` | 2 |
| `tests/test_deep_research_bench_plugin.py` | 6 |
| `tests/test_browsecomp_plugin.py` | 6 |
| `tests/test_qwen_deep_research_runner.py` | 6 |
| `tests/test_capabilities.py` | 6 |

### Edited files (10+)
| File | Phase | Change |
|------|-------|--------|
| `src/agents/benchmarks/base.py` | 0, 2 | Runner Protocol + `execution_environment` |
| `src/agents/benchmarks/__init__.py` | 2 | Register new benchmarks |
| `src/agents/benchmarks/swe_bench_verified.py` | 1b | Remove miniswe |
| `src/agents/benchmarks/swe_rebench.py` | 1b | Remove miniswe |
| `src/trace_collect/cli.py` | 1a, 1b, 3 | container optional, scaffold choices |
| `src/trace_collect/collector.py` | 1a, 1b | Unified dispatch, remove miniswe |
| `src/trace_collect/simulator.py` | 4 | Host-mode support |
| `src/harness/trace_logger.py` | 0 | execution_environment field |
| `src/llm_call/__init__.py` | 1b | Remove miniswe exports |
| `src/llm_call/providers.py` | 1b | Remove miniswe_litellm_prefix |
| `Makefile` | 1b | Remove miniswe targets |
| + ~15 test files | 1b | Remove miniswe references |

### Deleted files (3)
| File | Phase |
|------|-------|
| `src/agents/miniswe/` (entire directory) | 1b |
| `src/llm_call/miniswe.py` | 1b |
| `tests/test_miniswe_container_runtime.py` | 1b |

---

## ADR: Multi-Benchmark Architecture Decision

**Decision**: Thin abstraction layer (Option A) — add `execution_environment` property to Benchmark base, unify collector via `build_runner()`, new scaffold as simple runner module.

**Drivers**: (1) Non-containerized benchmark support needed, (2) scaffold dispatch must be polymorphic not hardcoded, (3) miniswe removal creates opportunity for cleanup.

**Alternatives considered**: Full Runner Protocol abstraction (Option B) — rejected because deep research benchmarks are simpler than SWE-bench, making full protocol over-engineering.

**Why chosen**: Minimal disruption, follows existing patterns (terminal-bench already uses host_controller), ships in 12-16 days vs 20+ for Option B.

**Consequences**: Collector retains some scaffold awareness (container vs host branching). Future scaffolds must implement Runner Protocol. Simulator gains optional-container complexity.

**Follow-ups**: (1) Consider `evaluation_method` property when evaluation pipeline is built, (2) Consider `requires_network` when network isolation is needed, (3) Dynamic resource allocation (from NEXT_STEPS_PLAN TODO 6) should account for host-mode benchmarks.
