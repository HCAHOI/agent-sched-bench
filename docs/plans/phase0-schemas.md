# Phase 0 — Schema Audit

> Companion artifact for `.omc/plans/trace-sim-vastai-pipeline.md` (REVISION 3 APPROVED 2026-04-08).
> Verbatim source-of-truth schemas for everything Phases 1–6 will touch.
> No code changes in Phase 0.

**Generated:** 2026-04-08 · iteration 1 of Ralph for `dev/trace-sim-vastai`
**Branch:** `dev/trace-sim-vastai` (no new branch — fresh fork from main per user instruction)

---

## (a) `connect_mcp_servers` expected dict shape

**File:** `src/agents/openclaw/tools/mcp.py`
**Function:** `connect_mcp_servers(mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack) -> None` at line 138

The `mcp_servers` parameter is **NOT a raw dict of dicts**. It is a dict keyed by **server name** (str) → **`MCPServerConfig` instance** (Pydantic-style model with attribute access). Pydantic config object lives at `src/agents/openclaw/config/schema.py:51`:

```python
class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""              # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""                  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)
    tool_timeout: int = 30         # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])
    # ["*"] = all tools; [] = none; explicit list = only those (raw or wrapped names)
```

**Auto-detection logic** (`mcp.py:149-159`): if `type` is None, then `command` set → stdio; `url` ending in `/sse` → sse; otherwise → streamableHttp.

**Tool wrapping**: each MCP server's tools become `MCPToolWrapper` instances (`mcp.py:75-135`) with name `f"mcp_{server_name}_{tool_def.name}"`. These are registered into the `ToolRegistry` parameter.

**Implication for Phase 4 `configs/mcp/context7.yaml`:** the YAML must serialize to a dict whose VALUES are `MCPServerConfig`-shaped — i.e., the YAML loader for `--mcp-config` must instantiate `MCPServerConfig` from each entry, not pass raw dicts to `connect_mcp_servers`. Phase 4's `cli.py` MCP loader needs to call `MCPServerConfig.model_validate(yaml_entry)` for each server.

**Example YAML shape** (Phase 4 must produce this):

```yaml
context7:
  type: streamableHttp           # or omit and let auto-detect handle it
  url: "https://mcp.context7.com/mcp"
  headers:
    Authorization: "Bearer ${CONTEXT7_API_KEY}"
  tool_timeout: 30
  enabled_tools: ["*"]            # or explicit list of context7 tool names
```

---

## (b) `PreemptionSnapshot.__dict__` field types

**File:** `src/harness/scheduler_hooks.py`
**Dataclass:** `PreemptionSnapshot` at line 15

```python
@dataclass(slots=True)
class PreemptionSnapshot:
    """Subset of vLLM metrics relevant to ENV-5 preemption analysis."""

    num_preemptions_total: float | None        # COUNTER (monotonic)
    gpu_cache_usage_perc: float | None         # GAUGE  (instantaneous, 0..1)
    cpu_cache_usage_perc: float | None         # GAUGE  (instantaneous, 0..1)
    gpu_prefix_cache_hit_rate: float | None    # RATIO  (instantaneous, 0..1)
    cpu_prefix_cache_hit_rate: float | None    # RATIO  (instantaneous, 0..1)
```

### Field type tags (load-bearing for Phase 2 `sim_metrics_delta.py`)

| Field | Semantic type | Delta meaningful? | Aggregation policy |
|---|---|---|---|
| `num_preemptions_total` | **counter** (monotonic non-decreasing) | **YES** — `delta = current - previous` is the new preemptions in this interval | sum / max |
| `gpu_cache_usage_perc` | **gauge** (instantaneous fraction in [0, 1]) | **NO** — delta of a gauge is meaningless ("change in utilization" is not utilization) | mean / max / min |
| `cpu_cache_usage_perc` | **gauge** | **NO** | mean / max / min |
| `gpu_prefix_cache_hit_rate` | **ratio** (hits / lookups in [0, 1]) | **NO** — rate-of-change of a hit rate is not a hit rate; the only meaningful aggregation is to recompute over a longer window | mean (weighted by lookup count if available) |
| `cpu_prefix_cache_hit_rate` | **ratio** | **NO** | mean |

**Phase 2 implication:** `src/analysis/sim_metrics_delta.py` MUST whitelist `_DELTA_VALID_FIELDS = frozenset({"num_preemptions_total"})`. Any attempt to compute `compute_preemption_delta` against a gauge or ratio field MUST raise `TypeError("Cannot compute delta of gauge/ratio field 'X'; use mean/max aggregation instead")`. The whitelist source MUST cite this Phase 0 doc to guarantee the categorization is auditable.

**Source values** (`scheduler_hooks.py:52-66`): the snapshot is constructed by `parse_prometheus_metrics(metrics_payload: str) -> PreemptionSnapshot` which scrapes:
- `vllm:num_preemptions_total`
- `vllm:gpu_cache_usage_perc`
- `vllm:cpu_cache_usage_perc`
- `vllm:gpu_prefix_cache_hit_rate`
- `vllm:cpu_prefix_cache_hit_rate`

**No `get_snapshot()` accessor exists yet.** Phase 2 will add one. Until then, `parse_prometheus_metrics(httpx.get(metrics_url).text)` is the only path. Phase 2's new `src/harness/metrics_client.py` will wrap this with a stable `get_snapshot() -> PreemptionSnapshot` interface so simulator.py replay loop has a single function to call after each iteration.

---

## (c) `MiniSWECodeAgent.prepare()` signature and side-effects on `self`

**File:** `src/agents/miniswe/agent.py`
**Class:** `MiniSWECodeAgent(AgentBase)` at line 164
**Method:** `async def prepare(self, task: dict[str, Any]) -> None` at line 208

### Signature

```python
async def prepare(self, task: dict[str, Any]) -> None
```

- **Type:** instance method on `MiniSWECodeAgent` (not a free function).
- **Returns:** `None`. State is communicated via instance attributes set as side effects.
- **Async:** yes.
- **Required `task` keys:** `repo` (str, "owner/name" form), `base_commit` (str, git commit hash), `instance_id` (str, used in error message).

### Side-effects on `self`

Both attributes are declared with type annotations in `__init__` (`agent.py:193-194`):

```python
self._workdir: Path | None = None
self._prepared = False
```

`prepare()` then mutates them at the end of the method (`agent.py:255-256`):

```python
self._workdir = workdir       # Path object — tempfile.mkdtemp(prefix="miniswe_")
self._prepared = True
```

### Workdir layout after `prepare()` succeeds

```
self._workdir/                  # = tempfile.mkdtemp(prefix="miniswe_")
└── repo/                       # cloned from local mirror or GitHub
    └── ... (checked out at base_commit, best-effort pip install -e .)
```

### Failure semantics

- Clone failure → calls `shutil.rmtree(workdir)` and raises `RuntimeError(f"Repo setup failed for {task['instance_id']}: ...")` (`agent.py:238-243`)
- Checkout failure → same teardown + raise
- pip install failure → silently ignored (best-effort, `agent.py:246-253`)

### Cleanup ownership

`prepare()` does NOT clean up `_workdir` automatically — that happens in `run()`'s `finally` block (`agent.py:274-277`):

```python
finally:
    shutil.rmtree(self._workdir, ignore_errors=True)
    self._workdir = None
    self._prepared = False
```

**Phase 1 implication for `scaffold_registry.py`:** the mini-swe adapter callable cannot return `None` as a `PreparedWorkspace` (because the simulator needs to know where the repo lives for tool execution). Two options:
1. **Adapter wraps the agent instance** and returns a small `PreparedWorkspace` dataclass with `workdir: Path` and a `cleanup() -> None` callable that owns the eventual `rmtree`. The adapter is what gets registered in `SCAFFOLD_PREPARE_REGISTRY`, not the raw `MiniSWECodeAgent.prepare`.
2. **Adapter returns the agent instance itself** and downstream code reads `agent._workdir`. Tighter coupling, less wrapping.

**Decision (locked here for Phase 1):** Option 1. Reason: less coupling, explicit cleanup contract, and the adapter is the right place to translate between the registry's contract and each scaffold's natural shape (which is the whole point of a polymorphic registry per Principle 1).

---

## (d) `prepare_workspace()` free function and SessionRunner composition

**File:** `src/agents/openclaw/eval/prepare.py`
**Function:** `async def prepare_workspace(...) -> float` at line 20 (NOT a class method)

### Signature

```python
async def prepare_workspace(
    workspace_dir: Path,
    repo: str,
    base_commit: str,
    *,
    repos_root: Path | None = None,
    clone_timeout: float = 300.0,
    checkout_timeout: float = 60.0,
    install_timeout: float = 600.0,
) -> float                   # returns elapsed_ms
```

- **Type:** module-level free function (NOT a method on any class).
- **Returns:** `float` — elapsed time in milliseconds (NOT a workspace handle).
- **Side-effects:** mutates `workspace_dir` in place (clones into it).

### Behavior

1. If `workspace_dir` exists → `shutil.rmtree(workspace_dir, ignore_errors=True)` (clean slate per call).
2. Clone from `repos_root / f"{repo.replace('/', '__')}.git"` if local mirror exists; otherwise `git clone --depth=1 https://github.com/{repo}.git`.
3. `git fetch --depth=1 origin <base_commit>` (with `--unshallow` fallback if depth-1 fetch fails).
4. `git checkout <base_commit>`.
5. Best-effort `pip install -e .` if `setup.py` or `pyproject.toml` exists.
6. Returns elapsed time in ms.

### Failure semantics

- Clone failure → `shutil.rmtree(workspace_dir)` + raise `RuntimeError(f"Clone failed for {repo}: ...")`
- Checkout failure → same teardown + raise
- pip install failure → silently ignored (best-effort)

### How `SWEBenchRunner.run_task` composes it

**File:** `src/agents/openclaw/eval/runner.py:120-171`

```python
async def run_task(self, task: EvalTask) -> EvalResult:
    ws = task.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    trace_file = ws / "trace.jsonl"

    # Phase 1: Prepare workspace (git clone + checkout)
    prepare_ms: float | None = None
    if task.needs_prepare:
        try:
            prepare_ms = await prepare_workspace(    # ← FREE FUNCTION CALL
                ws,
                repo=task.repo,
                base_commit=task.base_commit,
                repos_root=self.repos_root,
            )
        except Exception as e:
            ...
            return EvalResult(stop_reason="prepare_error", ...)

    # Phase 2: Run via SessionRunner (full bus dispatch)
    session_key = f"eval:{task.instance_id}"
    result = await self._session_runner.run(
        prompt=self._build_swe_bench_prompt(task.problem_statement),
        workspace=ws,
        session_key=session_key,
        trace_file=trace_file,
        instance_id=task.instance_id,
        channel="cli",
        prepare_ms=prepare_ms,
    )

    # Phase 3: Build EvalResult from session history
    ...
```

**Critical observations for Phase 1.5.0 (openclaw simulate design audit):**

1. **`prepare_workspace` is a free function with NO instance state.** Unlike `MiniSWECodeAgent.prepare`, it does not stash anything on `self`. The workspace path is passed in by the caller and the caller owns it.
2. **The runner ALWAYS goes through `SessionRunner.run()`** for the actual execution. There is no "linear loop" alternative on the openclaw side. SessionRunner is the bus-based dispatcher; it owns the agent loop iteration, tool registry connection, MCP wiring, and result collection.
3. **`task.needs_prepare`** is the gate — if False, the prepare phase is skipped. BFCL v4 sets this to False (no repo to clone). Phase 6 BFCL refusal guard can leverage this same field as the discriminator.
4. **`prepare_ms`** flows through `SessionRunner.run()` as a kwarg; it ends up in the trace as timing context. This is an important hint for Phase 2 sim_metrics: the simulator should also pass `prepare_ms` through if it ever measures it (it currently doesn't because mini-swe simulator hardcodes its own prepare path).

**Phase 1.5.0 design decision frame:** the openclaw simulate adapter MUST decide whether to:
- **(a)** Replace `SessionRunner.run()` with a linear `execute_trace_tool` loop (matches the current mini-swe simulator pattern but loses fidelity to bus-based scheduling — this is the whole research target so losing it is bad)
- **(b)** Keep `SessionRunner.run()` and feed it a "replay-mode provider" that returns recorded LLM responses from the source trace instead of actually calling the cloud API. The provider must be a drop-in for `LLMProvider` and emit the same shape of responses.

Option (b) preserves the bus-based scheduling shape that the simulator's research claim depends on. Option (a) would let mini-swe simulate behave the same way openclaw simulate does (linear), but at the cost of measuring something other than what openclaw actually does in production. Phase 1.5.0 will recommend option (b) absent contrary evidence.

---

## (e) `build_gantt_payload_multi` registry shape and payload flow

**File:** `src/trace_collect/gantt_data.py`
**Function:** `build_gantt_payload_multi` at line 178

### Module-level constants (the source-of-truth registries)

```python
# Line 21
_MARKER_CATEGORIES = frozenset({"SCHEDULING", "SESSION", "CONTEXT"})

# Line 48
ACTION_TYPE_MAP: dict[str, str] = {
    "llm_call": "llm",
    "tool_exec": "tool",
}

# Line 55
DEFAULT_SPAN_REGISTRY: dict[str, dict[str, Any]] = {
    "llm":        {"color": "#00E5FF", "label": "LLM Call",   "order": 0},
    "tool":       {"color": "#FF6D00", "label": "Tool Exec",  "order": 1},
    "scheduling": {"color": "#76FF03", "label": "Scheduling", "order": 2},
}

# Line 63
DEFAULT_MARKER_REGISTRY: dict[str, dict[str, str]] = {
    "message_dispatch":     {"symbol": "diamond", "color": "#76FF03"},
    "session_lock_acquire": {"symbol": "diamond", "color": "#76FF03"},
    "session_load":         {"symbol": "dot",     "color": "#76FF03"},
    "message_list_build":   {"symbol": "dot",     "color": "#4FC3F7"},
    "session_turn_save":    {"symbol": "dot",     "color": "#76FF03"},
    "task_complete":        {"symbol": "flag",    "color": "#FF6D00"},
    "llm_error":            {"symbol": "cross",   "color": "#FF1744"},
    "max_iterations":       {"symbol": "cross",   "color": "#FF1744"},
    "_default":             {"symbol": "dot",     "color": "#6b7280"},
}
```

### `build_gantt_payload_multi` signature and payload shape

```python
def build_gantt_payload_multi(
    traces: list[tuple[str, TraceData]],
    *,
    span_registry: dict[str, dict[str, Any]] | None = None,
    marker_registry: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "registries": {
            "spans":   span_registry   or DEFAULT_SPAN_REGISTRY,
            "markers": marker_registry or DEFAULT_MARKER_REGISTRY,
        },
        "traces": [
            build_gantt_payload(td, label=lbl) for lbl, td in traces
        ],
    }
```

### The payload-driven invariant (CRITICAL for Phase 5)

The registries are **shipped inside the JSON payload at the top level under `"registries"`**. The HTML template at `src/trace_collect/gantt_template.html` and the JS payload builder at `src/trace_collect/gantt_builder.js` consume the registry from the payload, **NOT from hardcoded JS constants**.

Module docstring at `gantt_data.py:1-11` confirms this explicitly:

> Build Gantt chart JSON payloads from TraceData. Actions (llm_call, tool_exec) ARE the spans directly. Scheduling overhead is computed from time gaps between consecutive actions. Events become point markers for observability detail.
>
> The default span / marker / action-type registries live here as module-level constants. They are shipped *inside* every Gantt payload so the HTML template renders from data, not from hard-coded JS literals — letting downstream users register new action_types or event names without touching the template.

### Phase 5 classification (PAYLOAD-ONLY vs RENDERING-LOGIC)

For each proposed Phase 5 edit, classify based on whether it changes registry CONTENT (payload-only) or rendering BEHAVIOR (logic-level):

| Proposed edit | Classification | JS edit required? |
|---|---|---|
| Add `"MCP"` to `_MARKER_CATEGORIES` | **Payload-only** — `_MARKER_CATEGORIES` controls which event categories produce point markers in `_build_spans_and_markers`. The JS reads markers from the payload list. Adding a category just emits more markers to the payload; JS renders them via the same path. | NO (verify in P5 preflight) |
| Add `"mcp_call": "mcp"` to `ACTION_TYPE_MAP` | **Payload-only** — `ACTION_TYPE_MAP` controls the span type label. The mapping flows through `_build_spans_and_markers` into the payload's `spans[].type` field, and JS reads the registry's `spans` block from the payload to look up the color/label. | NO (verify in P5 preflight) |
| Add `"mcp": {"color": ..., "label": ..., "order": ...}` to `DEFAULT_SPAN_REGISTRY` | **Payload-only** — registry is shipped in payload under `registries.spans`. JS renders from this dict. | NO (verify in P5 preflight) |
| Add `"mcp_*"` entries to `DEFAULT_MARKER_REGISTRY` | **Payload-only** — same reasoning as above. | NO (verify in P5 preflight) |
| Add tooltip rows for `data.sim_metrics.vllm_scheduler_snapshot.*` in `_extract_detail_from_action` | **Rendering-logic (POTENTIALLY)** — depends on whether JS replicates `_extract_detail` or reads pre-computed `detail` dicts from the payload. **Phase 5 preflight MUST grep `gantt_builder.js` to determine which.** If JS reads `spans[i].detail` directly from the payload, Python-only edit suffices. If JS has its own `_extract_detail` mirror, paired edits are required. | TBD by preflight |
| Add tooltip rows for `data.sim_metrics.timing.*` (ttft_ms, tpot_ms) | **Rendering-logic (POTENTIALLY)** — same as above. | TBD by preflight |

### Where Python and JS payload builders meet

`src/trace_collect/gantt_serve.py::_render_template` (per memory file `reference_trace_tools.md`) splices `__GANTT_BUILDER_JS__` and `__TRACE_JSON__` into the HTML at build time. The `__GANTT_BUILDER_JS__` placeholder is replaced with the contents of `gantt_builder.js`, which contains the JS payload builder for browser-uploaded traces (drag-drop). For pre-rendered HTML (the Python-built path), the Python `build_gantt_payload_multi` output is splat directly into `__TRACE_JSON__`.

**Phase 5 preflight TODO** (cited in plan §4 Phase 5 sub-task 0): read both `gantt_builder.js` and `gantt_template.html` end-to-end. Confirm the data-driven claim above by locating the JS code that consumes `payload.registries.spans` and `payload.registries.markers`. Document the exact JS function names + line numbers in `.omc/plans/phase5-parity-classification.md` BEFORE editing any code.

---

## Cross-references: how this audit feeds the later phases

| Phase | Consumes section | What it learns |
|---|---|---|
| Phase 1 (mini-swe registry) | (c) | Adapter wraps `MiniSWECodeAgent.prepare` and returns a `PreparedWorkspace` dataclass; cleanup contract owned by adapter. |
| Phase 2 (vLLM metrics) | (b) | Field type tags drive `_DELTA_VALID_FIELDS = frozenset({"num_preemptions_total"})` whitelist in `sim_metrics_delta.py`. |
| Phase 1.5.0 (openclaw audit) | (c), (d) | Confirms openclaw goes through `SessionRunner.run()` not a linear loop; design recommends replay-mode provider (option b). |
| Phase 1.5.1 (openclaw simulate impl) | (a), (d) | MCP results from source trace are reused via the recorded `tool_exec` shape; `prepare_workspace` is called as a free function from the adapter. |
| Phase 4 (MCP enablement) | (a) | YAML loader for `--mcp-config` must instantiate `MCPServerConfig.model_validate(entry)` for each server, NOT pass raw dicts. |
| Phase 5 (Gantt extension) | (e) | Registry-content edits are payload-only and need NO JS edits. Tooltip-row edits (sim_metrics) MAY need paired Python+JS — preflight MUST classify. |
| Phase 6 (BFCL refusal) | (d) | `task.needs_prepare = False` is the discriminator. The simulator can check the trace's metadata.task_shape and refuse with the canonical NotImplementedError before scaffold lookup. |

---

## Acceptance for US-001

- [x] File `.omc/plans/phase0-schemas.md` exists (this file)
- [x] Section (a) connect_mcp_servers dict shape with file:line refs
- [x] Section (b) PreemptionSnapshot fields with counter/gauge/ratio tags
- [x] Section (c) MiniSWECodeAgent.prepare signature + state side-effects
- [x] Section (d) prepare_workspace free function signature + SessionRunner composition
- [x] Section (e) build_gantt_payload_multi shape + payload-driven invariant for Phase 5 classification
- [x] No code changes (only this `.omc/plans/phase0-schemas.md` doc) — verified by `git diff --stat` after commit

End of Phase 0.
