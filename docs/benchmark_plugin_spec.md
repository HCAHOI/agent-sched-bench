# Benchmark Plugin Spec

> Canonical reference for adding a new benchmark to agent-sched-bench.
> Contract locked during the SWE-rebench refactor (branch `dev/swe-rebench-plugin`, 2026-04-07).

---

## 1. Overview

Benchmarks are plugins, not special cases. Each benchmark ships a Python class
(subclassing `Benchmark`) plus a YAML config file; the collector and scaffolds
are benchmark-agnostic and interact only through the plugin interface. There is
no benchmark-specific branching in `src/trace_collect/collector.py` or any
scaffold module.

---

## 2. Directory Layout

```text
src/agents/benchmarks/
├── __init__.py          # REGISTRY dict + get_benchmark_class()
├── base.py              # BenchmarkConfig dataclass + Benchmark ABC
├── swe_bench_verified.py
├── swe_rebench.py
├── bfcl_v4.py           # task_shape='function_call' reference plugin
└── bfcl_runner.py       # In-process function-call runner + AST scoring

configs/benchmarks/
├── swe-bench-verified.yaml
├── swe-rebench.yaml
└── bfcl-v4.yaml
```

---

## 3. BenchmarkConfig Dataclass

Defined in `src/agents/benchmarks/base.py`. Load via
`BenchmarkConfig.from_yaml(path)`.

| Field | Type | Meaning |
|---|---|---|
| `slug` | `str` | Unique identifier used in CLI flags and REGISTRY lookup (e.g. `"swe-rebench"`) |
| `display_name` | `str` | Human-readable name for logging and reports |
| `harness_dataset` | `str` | HuggingFace dataset path passed to the SWE-bench harness |
| `harness_split` | `str` | Dataset split name (e.g. `"test"`, `"filtered"`) |
| `data_root` | `Path` | Local directory where dataset tasks are cached |
| `repos_root` | `Path \| None` | Root for pre-cloned repositories; `None` if not used |
| `trace_root` | `Path` | Output directory for trace JSONL files |
| `default_max_steps` | `int` | Default scaffold step budget when `--max-steps` is not specified |
| `selection_n` | `int` | Default number of tasks selected per run |
| `selection_seed` | `int` | Random seed for task selection (determinism) |
| `docker_namespace` | `str \| None` | Docker image namespace prefix passed to the harness. Set to `null` when images are fully qualified (e.g. SWE-rebench) |
| `exclude_lite` | `bool` | When `True`, drops single-file "lite" tasks from the selection pool. Default `False`; see research integrity note below |
| `extras` | `dict[str, Any]` | Arbitrary benchmark-specific knobs; not used by the base class |

**Research integrity note on `exclude_lite`:** Per CLAUDE.md §1, this knob
must not be flipped without a documented research rationale written directly in
the YAML. Undocumented selection filters compromise reproducibility.

---

## 4. Benchmark ABC Methods

Defined in `src/agents/benchmarks/base.py`.

### Abstract (must override)

**`load_tasks(self) -> list[dict]`**

Load and return all raw tasks for this benchmark. Each task dict must contain
at minimum an `instance_id` key. Implementations typically call
`datasets.load_dataset(...)` and return the rows as plain dicts.

**`normalize_task(self, raw: dict) -> dict`**

Normalize a single raw dataset row into the canonical task dict shape expected
by scaffolds (`instance_id`, `problem_statement`, `FAIL_TO_PASS`,
`image_name`, etc.). Called by `EvalTask.from_benchmark_instance()` — see
Section 5.

### Concrete defaults (may override)

**`derive_test_cmd(self, task: dict) -> str`**

Derives a `pytest` command from `task["FAIL_TO_PASS"]`. Handles both native
list form (SWE-rebench) and JSON-encoded string form (SWE-Bench Verified). No
dependency on legacy shims.

**`select_subset(self, tasks, n=None, seed=None) -> list[dict]`**

Returns the first `n` tasks sorted by `instance_id`. This benchmark-agnostic
default is intentionally simple. Subclasses with specific selection needs
(repo-stratified sampling, `exclude_lite` filtering) **must** override this.

**`build_harness_args(self, *, predictions_path, run_id, max_workers=1, timeout=1800, report_dir=None) -> dict`**

Returns a dict of keyword arguments suitable for invoking the SWE-bench
harness (`dataset_name`, `split`, `namespace`, `predictions_path`, `run_id`,
`max_workers`, `timeout`, `report_dir`). Delegates to `self.config` fields.

**`image_name_for(self, task: dict) -> str | None`**

Returns `task.get("image_name")`. Override if images require construction from
other task fields.

### Must override

**`build_runner(self, *, scaffold: str, **kwargs) -> Any`**

Builds and returns the scaffold runner for this benchmark. The base class
always raises `NotImplementedError` — this is intentional: a new benchmark
that forgets to override will fail loudly at first use rather than silently
dispatching to an incompatible runner.

```python
# base.py raises:
raise NotImplementedError(
    f"Benchmark {self.slug!r} (task_shape={self.task_shape!r}) does not implement "
    f"build_runner; subclasses must override this method for scaffold={scaffold!r}"
)
```

---

## 5. task_shape Discriminator

```python
task_shape: ClassVar[Literal["swe_patch", "function_call"]] = "swe_patch"
```

Set as a class variable on each plugin. Current values:

- `"swe_patch"` — SWE-bench-style patch tasks; `build_runner` returns an
  `SWEBenchRunner`. Requires `repos_root` to be set (collector enforces
  this at dispatch time). Scaffolds supported: `mini-swe-agent`, `openclaw`.
- `"function_call"` — BFCL-v4-style function-call tasks; `build_runner`
  returns a `BFCLRunner`. `repos_root` is typically `null` and the
  collector's `repos_root` gate is skipped based on this shape.
  Scaffolds supported: `openclaw` only. Plugins with this shape MUST
  refuse `mini-swe-agent` in `build_runner` with a descriptive
  `NotImplementedError` — mini-swe is bash-in-repo and cannot emit
  structured function calls against a JSON-Schema tool spec.

The collector branches on `task_shape` in two places only, both in
`src/trace_collect/collector.py`:

1. The `repos_root is None` ValueError gate (fires only for `swe_patch`).
2. The `CollectedTaskResult.success` derivation — `swe_patch` uses
   `bool(model_patch)`, `function_call` uses
   `bool(eval_result.official_resolved)`.

No other collector or scaffold code knows about `task_shape`. Adding a
new shape (e.g. `"ml_eval"`) would require extending these two branches.

This discriminator was introduced in Phase 4 of the SWE-rebench refactor
(PM-2 mitigation) to prevent a non-SWE benchmark from silently
inheriting SWE-specific runner logic. It was first exercised by the
BFCL v4 plugin.

---

## 6. Registry

```python
# src/agents/benchmarks/__init__.py
REGISTRY: dict[str, type[Benchmark]] = {
    "swe-bench-verified": SWEBenchVerified,
    "swe-rebench": SWERebenchBenchmark,
    "bfcl-v4": BFCLv4Benchmark,
}
```

**Lookup API:**

```python
from agents.benchmarks import get_benchmark_class
cls = get_benchmark_class("swe-rebench")   # KeyError with known-slugs hint if not found
plugin = cls(config)
```

To register a new benchmark, add an entry to `REGISTRY` in
`src/agents/benchmarks/__init__.py` at module import time (no factory
decorator, no metaclass magic — plain dict insertion).

---

## 7. YAML Schema

All fields map 1:1 to `BenchmarkConfig` fields (Section 3). Path fields are
relative to the repo root and wrapped in `pathlib.Path` by `from_yaml()`.
`repos_root: null` is valid and yields `None`.

Worked example — `configs/benchmarks/swe-rebench.yaml`:

```yaml
# SWE-rebench benchmark plugin config.
# Used by agents.benchmarks.get_benchmark_class("swe-rebench").
slug: swe-rebench
display_name: "SWE-rebench (filtered)"
harness_dataset: nebius/SWE-rebench
harness_split: filtered
data_root: data/swe-rebench
repos_root: data/swe-rebench/repos
trace_root: traces/swe-rebench
default_max_steps: 50
selection_n: 32
selection_seed: 42
# nebius ships fully qualified image URIs (swerebench/sweb.eval.x86_64.*)
# so the harness's namespace prefix must NOT be applied.
docker_namespace: null
# exclude_lite default: false per CLAUDE.md §1 "no benchmark-specific
# tuning". Flip to true (with a prose rationale here) only if you have
# a documented research reason to drop single-file lite tasks.
exclude_lite: false
```

---

## 8. Trace Metadata Contract

Every trace produced by the collector carries these fields in its
`trace_metadata` record:

| Field | Value |
|---|---|
| `trace_format_version` | `5` (integer, strict) |
| `benchmark` | benchmark slug, e.g. `"swe-rebench"` |
| `benchmark_split` | split name, e.g. `"filtered"` |
| `scaffold` | scaffold name, e.g. `"openclaw"` |

`TraceData.load()` raises on any `trace_format_version` other than `5`. There
is no backfill path and no tolerance for v4 traces — they must be re-collected.

Stamped by `src/harness/trace_logger.py::log_metadata()`.

---

## 9. Worked Example: Adding a New Benchmark

Assume you are adding a benchmark with slug `<slug>` backed by HuggingFace
dataset `<HF dataset>` and split `<split>`.

1. **Create the plugin class.**
   Add `src/agents/benchmarks/<slug>.py` with a class that subclasses
   `Benchmark`. Set `slug = "<slug>"` and `task_shape` as a `ClassVar`. At
   minimum implement `load_tasks`, `normalize_task`, and `build_runner`.

2. **Create the YAML config.**
   Copy `configs/benchmarks/swe-rebench.yaml` as a template. Fill in
   `slug`, `display_name`, `harness_dataset: <HF dataset>`,
   `harness_split: <split>`, and the path fields. If the benchmark ships
   fully-qualified Docker image names, set `docker_namespace: null`.

3. **Register the plugin.**
   In `src/agents/benchmarks/__init__.py`, import the new class and add it to
   `REGISTRY`:
   ```python
   from agents.benchmarks.<slug> import <ClassName>
   REGISTRY["<slug>"] = <ClassName>
   ```

4. **Add unit tests.**
   Create `tests/test_<slug>_plugin.py`. At minimum test `normalize_task` on a
   representative raw row and any benchmark-specific selection logic.

5. **Add Makefile targets.**
   Add `make download-<slug>` (downloads the dataset) and
   `make setup-<slug>-repos` (clones required repositories, if any), following
   the pattern of the existing `download-swe-rebench` / `setup-swe-rebench-repos`
   targets.

6. **Smoke-test end-to-end.**
   ```bash
   conda activate ML
   PYTHONPATH=src python -m trace_collect.cli \
       --provider dashscope \
       --benchmark <slug> \
       --scaffold openclaw \
       --sample 2
   ```
   Verify that produced traces carry `trace_format_version: 5`,
   `benchmark: "<slug>"`, and `benchmark_split: "<split>"` in their
   `trace_metadata`.

7. **Run the review gate.**
   Per CLAUDE.md "Mandatory Review Gate", spawn a fresh `code-reviewer` agent
   before running any experiments. Document the findings in
   `docs/reviews/<slug>-plugin-review.md`.

---

## 10. Function-call benchmarks (BFCL v4 reference)

BFCL v4 is the first plugin with `task_shape = "function_call"`. This
section documents the differences from the default `swe_patch` shape so
future function-call benchmarks (OpenFunctions, API-Bank, ToolBench, …)
can be added with minimal surgery.

### 10.1 EvalTask extensions

Function-call benchmarks carry per-task data that SWE-patch tasks don't
need. These live on `EvalTask` as optional fields and are populated by
`from_benchmark_instance` after `benchmark.normalize_task` runs:

| Field | Type | Populated for | Description |
|---|---|---|---|
| `tools` | `list[dict]` | BFCL | JSON-Schema function specs the model must call against |
| `question` | `list[list[dict]]` | BFCL | Turn list; `[0][0]` is the initial user message |
| `ground_truth` | `list[dict]` | BFCL | Expected function calls, shape `[{fn_name: {arg: [accepted_values]}}]` |
| `category` | `str \| None` | BFCL | BFCL category (e.g. `"simple_python"`, `"irrelevance"`) |

`EvalTask.needs_prepare` returns `False` when `repo is None`, so function-call
tasks naturally skip the git-clone phase.

### 10.2 Scaffold compatibility

| Scaffold | `swe_patch` | `function_call` |
|---|---|---|
| `mini-swe-agent` | supported | **refused** (bash-in-repo only) |
| `openclaw` | supported | supported |

The `mini-swe-agent` refusal is enforced in two places with explicit
precedence:

1. **Collector dispatch gate** (`src/trace_collect/collector.py` —
   the `scaffold == "mini-swe-agent" and task_shape != "swe_patch"`
   branch in `collect_traces`) raises `ValueError` **first** for any
   production run routed through the CLI. This is the primary guard.
2. **Plugin-level `build_runner` refusal** (`BFCLv4Benchmark.build_runner`
   at `src/agents/benchmarks/bfcl_v4.py`) raises `NotImplementedError`.
   This is defense-in-depth for direct callers that bypass the
   collector (unit tests, future third-party harnesses).

Both fire loudly; there is no silent-fallback path. A new function-call
benchmark that supports a different scaffold combination should override
its own `build_runner` to refuse the unsupported scaffolds explicitly
and rely on the collector gate for the primary production guard.

### 10.3 BFCLRunner — routes through SessionRunner via custom ToolRegistry

`BFCLRunner` (in `src/agents/benchmarks/bfcl_runner.py`) walks the same
`SessionRunner` + `AgentLoop` scheduling path as SWE-patch benchmarks.
Each task builds a per-task `ToolRegistry` populated with
`BFCLNoOpTool` wrappers (one per entry in `task.tools`) and passes it
to `SessionRunner.run(tools=<registry>)` via the Phase 0 extension
point (see §11). Openclaw's default bash/file/web tools are NOT
registered on this path — the LLM sees only the BFCL-shipped function
schemas.

Per-task flow:

1. `build_bfcl_tool_registry(task.tools)` returns `(registry, recorder)`
   where `recorder` is a list closure shared across the registry's
   tool instances.
2. `_flatten_single_turn_question(task)` collapses `task.question[0]`
   into a single prompt string.
3. `SessionRunner.run(tools=registry, ...)` constructs an `AgentLoop`
   with the custom registry and walks the full bus dispatch path.
4. Every `tool_call` the LLM emits is dispatched to
   `BFCLNoOpTool.execute(**kwargs)`, which appends
   `{"name", "arguments"}` to the recorder and returns `"OK"` — no
   openclaw invariants assume side effects or bash output.
5. After the session ends, the runner reads the recorder as the
   `predicted_calls` list and scores it via `_ast_match`.
6. `EvalResult.usage` is recovered by summing `llm_call` action tokens
   from the trace file via `_sum_usage_from_trace`.

`max_iterations=1`: single-turn BFCL categories make exactly one LLM
call per task. The recorder captures the dispatched tool calls in
iteration 0 (the only iteration) before the loop's ``for-else``
branch fires. ``BFCLRunner.__init__`` validates ``max_iterations >= 1``
and raises ``ValueError`` otherwise — a zero cap would leave the
recorder permanently empty and silently yield score 0.0.

Scheduling-data delta vs v1: v1's bypass path emitted exactly 3
records per trace (metadata + single llm_call action + summary). v2's
SessionRunner path emits 8-20+ records depending on tool-call count
(irrelevance floors at ~8 with no tool dispatch; tasks with 1+ tool
calls produce 11+). Records include llm_call_start / llm_call_end /
tool_exec_start / tool_exec_end events + scheduling events from the
TraceCollectorHook. BFCL is now first-class for scheduling research.

**Error absorption**: a `provider.chat()` exception is caught inside
`AgentLoop` and surfaces as an "LLM returned error" trace event; the
session still completes with an empty recorder, and the runner
produces `stop_reason="completed"` + `official_resolved=False`. This
is the research-honest semantic: a provider error is "the model
couldn't produce a correct prediction", which scores False.
`EvalResult.error` is populated from the trace's ``llm_error`` event
(via ``BFCLRunner._extract_absorbed_llm_error``) so downstream
analysis can distinguish "wrong answer" (``score=0, error=None``)
from "model crashed" (``score=0, error=<message>``) without
re-walking the trace file.

### 10.4 AST-match rules

Implemented in `BFCLRunner._ast_match`. Mirrors the documented BFCL rules:

1. Function name must match exactly.
2. Every required argument must be present with an exact value match.
   Ground-truth values are lists of acceptable alternatives — the
   prediction matches if it equals any alternative.
3. Optional arguments (not listed in ground truth) may be omitted
   without failing the match.
4. All-or-nothing per call — no partial credit.
5. Categories with multiple expected calls (`parallel`, `multiple`,
   `parallel_multiple`) compare as sets — order doesn't matter.
6. The `irrelevance` / `live_irrelevance` categories are correct iff
   the predicted call list is empty (the model must NOT invoke any tool).

The scoring is reimplemented in-process rather than shelling out to
`bfcl-eval` because the PyPI package is not a hard dependency of this
repo. If `bfcl-eval` becomes available it can be swapped in behind the
same `_ast_match` interface.

### 10.5 Collector success field

`CollectedTaskResult.success` for function-call benchmarks is derived
from `eval_result.official_resolved` (set by `BFCLRunner._ast_match`),
not from `model_patch`. The `success_basis` field reports
`"official_resolved"` accordingly, so downstream analysis can tell the
two shapes apart without guessing.

### 10.6 No docker, no SWE-bench harness

BFCL v4 scoring is pure Python AST comparison — no containers, no
sandboxed test execution. The plugin explicitly refuses the SWE-bench
harness path:

- `BFCLv4Benchmark.build_harness_args` raises `NotImplementedError`.
- `BFCLv4Benchmark.image_name_for` returns `None`.
- `BFCLv4Benchmark.derive_test_cmd` raises `NotImplementedError`
  (defensive; should never be called on this shape).

The `--evaluate` flag on `trace_collect.cli` does not apply to BFCL
tasks. Scoring happens in-process inside `BFCLRunner.run_task`.

### 10.7 Data layout

```text
data/bfcl-v4/
├── raw/                      # Copied from ShishirPatil/gorilla
│   ├── BFCL_v4_simple_python.json
│   ├── BFCL_v4_multiple.json
│   ├── possible_answer/
│   │   └── BFCL_v4_simple_python.json
│   └── ...
└── tasks.json                # Merged JSONL (one row per line)
```

The merge step (in `scripts/setup/bfcl_v4_data.sh`) joins each task row
with its matching `possible_answer/` entry by `id` and writes one
JSONL row per task with keys: `category`, `id`, `question`, `function`,
`ground_truth`. The plugin's `load_tasks` reads this merged file.

`src/trace_collect/collector.py::load_tasks` auto-detects between JSON
array (SWE convention) and JSONL (BFCL convention) — both formats work.

---

## 11. Custom tool registries for non-swe_patch benchmarks

Function-call benchmarks (BFCL v4+, future OpenFunctions / API-Bank /
ToolBench integrations) need the LLM to see a per-task set of
functions that are not part of openclaw's default tool set. The
extension point is a new keyword-only parameter on both `AgentLoop`
and `SessionRunner.run()`:

```python
async def run(
    self,
    prompt: str,
    workspace: Path,
    *,
    ...
    tools: ToolRegistry | None = None,
) -> SessionRunResult:
```

**Replace semantics**: when `tools` is provided, `AgentLoop` uses the
registry as-is and does NOT call `_register_default_tools()`. The
default bash/file/web/memory tool set is only registered on the
no-override path. This is deliberate: if the LLM could see bash in
addition to the benchmark-provided functions, it could compute
answers out-of-band and return plain text — which would score False
under AST-match even when the model "knew" the answer.

**trace_metadata auto-derive**: when a custom registry is supplied,
`SessionRunner.run()` derives `scaffold_capabilities.tools` from the
registry's `get_definitions()` output and stamps
`scaffold_capabilities.source = "custom_registry"` as a sentinel so
downstream analysis can distinguish function_call traces from swe_patch
traces without re-parsing the plugin registry.

**How to wire up a new function_call benchmark** (BFCL is the reference
implementation):

1. Subclass `Tool` once per task (or once per schema dialect) with
   `execute()` recording the call and returning a neutral
   acknowledgment. See `agents.benchmarks.bfcl_tools.BFCLNoOpTool`.
2. Override `validate_params` to return `[]` if your schema dialect
   doesn't match standard JSON Schema — the runner scores at a
   dedicated match layer, not at openclaw's strict per-call validator.
3. Write a registry builder that returns `(registry, recorder)` — the
   recorder must be a closure shared across all tool instances in one
   `run_task` call so the runner can read a single chronological call
   log. See `build_bfcl_tool_registry`.
4. In your runner's `run_task`, call
   `self._session_runner.run(prompt=..., tools=<registry>, ...)` and
   read the recorder after the session returns.
5. Score the recorder contents however your benchmark demands
   (AST-match, execution comparison, llm-as-judge, etc.) and populate
   `EvalResult.official_resolved` + `evaluation_report`.

The pattern is non-reentrant — a fresh `(registry, recorder)` pair
per `run_task` call keeps concurrent invocations from
cross-contaminating. For BFCL, the recorder is a local `list[dict]`
bound at registry construction time.

**Schema normalization**: benchmarks that ship non-standard JSON
Schema dialects (BFCL uses `"type": "dict"` instead of `"object"`,
`"tuple"` instead of `"array"`, `"any"` for polymorphic args) should
normalize parameters before wrapping them as `Tool` instances. BFCL's
`_normalize_bfcl_schema` is a pure function that recursively rewrites
these to standard JSON Schema; copy-and-adapt for other dialects.
