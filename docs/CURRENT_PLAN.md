# Plan: Replace `research-agent` with vendored Tongyi-DeepResearch ReAct scaffold

**Ralplan R3** | 2026-04-16 | Branch: `feat/multi-benchmark`
**Consensus**: Planner → Architect (APPROVE-with-polish) → Critic v2→v3 ITERATE(8) → v3' → v5 (user simplifications) → v6 ITERATE(3) → v6 Critic APPROVE
**Status**: ALL R3 phases A, B, C, D, E, G, I, H2, J, F **COMPLETE** (2026-04-16). Homegrown multi-phase scaffold hard-deleted per R3 Principle #3; zero residual references outside this `docs/CURRENT_PLAN.md` R2_DEPRECATED archive section.

## 2026-04-16 PR #13 latest review triage addendum

- Scope: classify the newest PR #13 comments after head commit `34dea38`.
- Source reviews: Codex `4120399181`, Gemini `4120407885`.
- Must-fix in repo-owned code:
  - Increment Tongyi iteration state per model turn so multi-turn traces do not collapse to `iteration=0`.
  - Emit canonical `success` on Tongyi `tool_exec` actions so downstream tooling does not treat successful tool calls as failures.
  - Preserve async tool behavior in the traced tool wrapper; current sync-only wrapper can mis-trace async vendored tools such as `parse_file`.
- Real vendor/runtime risks:
  - Guard empty `SANDBOX_FUSION_ENDPOINTS` before `random.choice(...)` in `vendor/tool_python.py`.
  - Prevent missing-key fallback in `vendor/file_tools/file_parser.py` when IDP parsing fails for formats without parser entries.
- Not recommended to fix for this round:
  - Module-level `@staticmethod` in `vendor/tool_visit.py` is not a bug on the supported Python floor (`>=3.11`).
  - Retry-loop attempt-count mismatch and per-URL timeout message wording in vendor files are minor polish only.
- Architect re-review verdict:
  - Upgrade the two vendor/runtime risks above to must-fix if the scaffold continues to expose `PythonInterpreter` and `parse_file` as callable tools in prompt / `TOOL_CLASS`.
- Post-`f8d7a45` latest review triage:
  - Must-fix: move `_patched_vendor()` baseline snapshot + traced-wrapper construction under `_VENDOR_PATCH_LOCK` to prevent cross-run module-state contamination.
  - Optional polish: align `vendor/tool_python.py` retry loop to a single explicit 5-attempt budget instead of mixed `range(8)` / `/5` / `attempt == 4`.
- Post-`d773f78` latest review triage:
  - Fixed: prime psutil CPU baselines for newly cached root/child process handles in `process_stats_sampler.py`.
  - Fixed: retry only retryable transport errors in `TracedStreamingOpenAI` (`429` and `5xx`, plus connection/timeout classes), not all `APIStatusError`s.
  - Fixed: add jitter to Tongyi exponential backoff to reduce synchronized retries in concurrent sweeps.
  - Fixed: scope `Visit` summarizer env override to each task run and restore previous values after the run completes.

### Phase log (most recent first)

| Phase | Commit | Notes |
|-------|--------|-------|
| H2 | `d85c1fc` | Simulator replay test + `tests/fixtures/tongyi_deepresearch_minimal_v5.jsonl`; 340 / 340 green |
| I  | `9b63e44` | Paid smoke DRB + browsecomp via DashScope qwen-plus-latest; ~$0.14 total; 3 trace fixes (shared action_id counter, circular-ref strip, env alias) |
| E  | `b9e3a87` | Backend smoke against DashScope; AC#4 ttft/tpot invariants verified on real streaming |
| G  | `10faefb` | Scaffold registrations wired into `_research.py`, `cli.py`, capability tests |
| C+D| `cf30367` | Vendor patches (Buckets C+D only; A+B kept adapter-side) + `TongyiDeepResearchRunner` + trace infrastructure |
| A+B| `b1ebc07` | Upstream pin (SHA `f72f75d8`) + vendor 4 files verbatim |

---

## Why this plan replaces R2

R2 ("introduce `research-agent` scaffold") was shipped as commit `9d2b9a2` but did not fulfill the original user intent. User's 2026-04-15 directive was to introduce the **real** `Alibaba-NLP/DeepResearch` (Tongyi-DeepResearch) scaffold, not a homegrown 5-phase simplification. R3 corrects that by hard-replacing the homegrown scaffold with a vendored ReAct scaffold from upstream.

R2 content retained below as historical record under **R2_DEPRECATED** marker.

---

## R3 Principles (5)

1. **Upstream fidelity (soft)**: Vendor at pinned SHA. Prefer minimal patches, prefer adapter-layer logic over in-vendor edits. VENDOR_NOTES.md records all patch diff line counts by category for audit. **No LOC hard gate** (per user directive 2026-04-16).
2. **Trace completeness for scheduling**: Every LLM/tool call emits v5 TraceAction with canonical keys. Model-layer retries get `data.retry_of: <orig_action_id>` + `data.logical_turn_id: <uuid>`. Transport-layer (429/503) retries get `data.transport_retry: true` + the same `logical_turn_id`. `logical_turn_id` is generated **in the runner adapter** (Phase D), NOT in vendor source. `_build_summary` dedups on `logical_turn_id` for turn count; sums all TraceActions for wall-time totals.
3. **Hard replace with ≤3-day interim shim**: `src/agents/research_agent/` deleted in Phase F (≤3 days post Phase I green). Interim env-flag `OMCBENCH_ALLOW_DEPRECATED_SCAFFOLD=1` required to invoke old scaffold (default: raise DeprecationError). Shim lifetime written into VENDOR_NOTES.md with explicit deadline timestamp.
4. **Backend-agnostic scaffold**: Runner uses existing `create_async_openai_client(api_base, api_key)` pattern; no backend assumption. Backend chosen at `trace_collect.cli` call time (local vLLM / OpenRouter / DashScope / OpenAI / any OpenAI-compatible proxy). Runner adapter wraps client with 429/503 exponential backoff (max 3 retries) to cover cloud-backend transport failures. Simulate mode (host replay + cloud sleep-with-speedup) does not go through runner.
5. **Vendor only 2 enabled tools**: `tool_search.py` (Serper) + `tool_visit.py` (Jina) + `react_agent.py` + `prompt.py`. Drop `tool_scholar.py` / `tool_python.py` / `tool_file.py` files entirely; patch `react_agent.py` imports and `TOOL_CLASS` to remove references. Follow-up #2 re-vendors these if future experiments require.

## Decision Drivers (3)

1. Honor 2026-04-15 user directive that R2 failed to fulfill.
2. MLSys scheduling research requires realistic agent workload traces; homegrown 5-phase pipeline is structurally too simple and not what user asked for.
3. Minimize blast radius: openclaw / swe-bench / swe-rebench stacks untouched; only deep-research-bench + browsecomp affected.

## Options

| Option | Status | Rationale |
|---|---|---|
| **A: vendor enabled-tools-only + trace-hook + backend-agnostic runner** | **CHOSEN** | Matches user directive; bounded scope; auditable |
| A'': vendor all 5 tools, enable 2 | Rejected | Carries 3 unused tool files → "no just-in-case code" (CLAUDE.md §3, user priority #1) |
| B: git submodule + monkey-patch | Rejected | User chose vendor over submodule |
| C: OpenRouter-only backend | Rejected | Obsoleted by Principle #4 backend-agnosticism |
| D: keep homegrown phased + add tongyi | Rejected | User directive "完全替换" explicit; Architect concern moved to Follow-up #1 |

## Pre-mortem (5 scenarios)

1. **`qwen-agent==0.0.26` dep conflict on conda ML env**: Phase B isolated-env `import` smoke before wiring in; if conflict, pin openai/pydantic versions or revisit qwen-agent minor.
2. **Non-Tongyi model produces wrong stop pattern**: Acceptable; retry logic fires; scheduling trace still accumulates valid span data. Phase I asserts trace structure only, not answer quality.
3. **Model-layer retry storm**: `exit_status="retry_exhausted"`; preserve all retry TraceActions.
4. **Upstream SHA drift during plan execution (A→F)**: Frozen pin; re-pin only for blocker bugs with full audit re-run.
5. **Cloud backend 429/503 burst**: Adapter exponential backoff; exhaustion → `exit_status="rate_limit_exhausted"`; do NOT silently swap backend (human decision). Trace preserves transport-retry TraceActions for scheduling analysis.

## Phases

```
A (pin SHA + license check + freeze protocol)
→ B (vendor 4 files: react_agent / prompt / tool_search / tool_visit; strip others)
→ C (trace-hook patch, 3 buckets: hook emit / streaming shim / TOOL_CLASS+import prune
     — record per-bucket line count in VENDOR_NOTES.md; no LOC gate)
→ D (runner adapter: streaming shim + logical_turn_id assignment + 429/503 backoff wrapper)
→ E (backend smoke: spin up any chosen backend, run minimal ReAct turn, verify TraceAction emit)
→ H1 (unit tests with mocked qwen-agent + mocked tools)
→ G (wire registrations: _research.py / cli.py / capabilities.py / YAML configs)
→ I (paid smoke 1+1: deep-research-bench task + browsecomp task on user-chosen backend)
→ [gate: I trace satisfies AC#2 ∧ AC#4, else halt + diagnose]
→ H2 (integration replay tests on I-produced trace)
→ J (docs + VENDOR_NOTES.md with F deletion_deadline timestamp)
→ F (hard-delete src/agents/research_agent/ ≤3 days post-I-green; ADR amendment if exceeded)
```

## Acceptance Criteria (6)

1. **AC#1**: `rg "research_agent|research-agent" src/ tests/ configs/` → **0 matches** (except this CURRENT_PLAN.md R2_DEPRECATED archive section).
2. **AC#2**: Phase I smoke trace contains **≥1 complete ReAct step triplet** (`thought → tool_call → tool_response`). Task-agnostic structural assertion.
3. **AC#3**: `conda run -n ML python -m pytest tests/ -v` → **0 failures**.
4. **AC#4**: Phase I paid-smoke trace: **every** `llm_call` TraceAction has:
   - `ttft_ms` = wall time from request dispatch to first non-empty content chunk
   - `tpot_ms` = `(total_completion_wall_ms − ttft_ms) / completion_tokens`; `completion_tokens` from stream terminal `usage` chunk preferentially, with tokenizer re-count fallback (handles DashScope / proxies that strip per-chunk usage)
   - Both non-None = pass.
5. **AC#5**: Tool TraceActions have canonical `tool_args` / `tool_result` / `duration_ms` keys. Model-layer retry TraceActions have `retry_of` + `logical_turn_id`. Transport-layer (429/503) retry TraceActions have `transport_retry: true` + `logical_turn_id`.
6. **AC#6**: `src/agents/tongyi_deepresearch/VENDOR_NOTES.md` records:
   - upstream URL
   - pinned commit SHA
   - Apache-2.0 NOTICE file preserved at `src/agents/tongyi_deepresearch/vendor/NOTICE`
   - patch diff line count by bucket (hook emit / streaming shim / TOOL_CLASS+import prune)
   - `research_agent` deletion_deadline timestamp (= Phase I green date + 3 days)

## ADR

### Decision

Replace homegrown `src/agents/research_agent/` with vendored `Alibaba-NLP/DeepResearch` inference/ ReAct scaffold at pinned commit `<TBD Phase-A>`, Apache-2.0. Backend: any OpenAI-compatible endpoint via existing `create_async_openai_client`. Tool surface: only `search` (Serper) + `visit` (Jina) vendored and enabled. Other 3 upstream tools not vendored.

### Drivers

1. Honor 2026-04-15 user directive.
2. MLSys scheduling research requires realistic agent workload traces.
3. Bounded blast radius.

### Alternatives considered

See Options table above.

### Why chosen

Vendor + trace-hook + backend-agnostic runner = auditable fidelity + scheduling trace integrity + user intent alignment, without coupling to any specific backend or hardware spec.

### Consequences

- (+) New pip dep `qwen-agent==0.0.26`
- (+) ~900 LOC vendored code under `src/agents/tongyi_deepresearch/vendor/` (no hard cap per user)
- (+) `exit_status` enum gains `rate_limit_exhausted`
- (−) Historical `research-agent` traces deprecated (still valid v5 JSONL)
- (−) Scaffold family narrowed to ReAct class only → Follow-up #1 if scheduling-diversity analysis demands otherwise
- (−) Heavy / IterResearch mode unsupported (upstream did not open-source it) → Follow-up if released

### Follow-ups

1. If scheduling analysis needs phased-pipeline structural contrast → introduce honestly-named `phased-research` scaffold (not a Tongyi impersonation).
2. If experiments demand additional tools → vendor `tool_scholar.py` / `tool_python.py` / `tool_file.py` on cherry-pick basis.
3. If Serper cost binds → DDG adapter as optional replacement.
4. Cross-backend scheduling comparison (local vLLM vs cloud API on same task) — runner's backend-agnostic design enables this without code changes.

---

# R2_DEPRECATED (archived)

The content below this marker is the R2 plan that shipped as `9d2b9a2` and is superseded by R3. Retained for historical record only. New `research_agent` references in R2 text are grandfathered under AC#1's "except CURRENT_PLAN.md R2_DEPRECATED archive" exception.

---

# Plan: Replace `qwen-deep-research` with `research-agent` Scaffold

**Ralplan R2** | 2026-04-15 | Branch: `feat/multi-benchmark`
**Revision**: Incorporates Architect R1 feedback (6 items)

---

## Executive Summary

The current `qwen-deep-research` scaffold is a single OpenAI-compatible
streaming chat call that produces exactly one `llm_call` span. It is not a
multi-step research agent and cannot produce the rich traces (planner, search,
fetch, evidence, synthesis) needed for scheduling/simulation analysis. This plan
replaces it with a repo-owned, open-source `research-agent` scaffold that
implements an explicit multi-phase research workflow, emitting canonical v5 trace
records at each step. The `qwen-deep-research` scaffold is deleted outright with
no migration alias.

---

## RALPLAN-DR Summary

### Principles
1. **No benchmark gaming**: scaffold logic must be general, not tuned to specific datasets
2. **Traceable at step level**: every LLM call and tool execution produces a canonical v5 action record
3. **Simulatable**: traces must be replayable through the existing `simulator.py` cloud_model and local_model paths
4. **Provider-agnostic naming**: scaffold identity must not be named after a model family
5. **Delete over deprecate**: remove misleading code rather than carrying compatibility shims

### Decision Drivers (top 3)
1. The current scaffold produces a single `llm_call` span — useless for scheduling analysis
2. Naming (`qwen-*`) conflates scaffold identity with provider/model choice
3. The existing OpenClaw `WebSearchTool`/`WebFetchTool` are mature and reusable

### Viable Options

**Option A: Hard-delete `qwen-deep-research`, add `research-agent`**
- Pros: Clean break, no compatibility debt, clear naming
- Cons: All existing `qwen-deep-research` traces become "legacy scaffold" artifacts

**Option B: Keep alias temporarily, block paid experiments under old name**
- Pros: Softer migration for existing trace references
- Cons: Extra code paths, risk of someone using the broken scaffold for real experiments

**Option C: Implement official DashScope DeepResearch API path separately**
- Pros: Could produce genuine multi-step traces from DashScope's internal workflow
- Cons: Opaque API (no step-level control), vendor lock-in, not locally simulatable, DashScope API may change

### Invalidation of Alternatives
- **Option B**: Violates principle #5 (delete over deprecate). The alias code is pure debt with no research value.
- **Option C**: Violates principles #2 and #3. A black-box API call cannot produce step-level traces we control, and we cannot simulate timing of opaque internal steps. May be added later as a separate `dashscope-deep-research` provider-specific scaffold if DashScope exposes step-level hooks.

### Architect Steelman Antithesis (addressed)

The strongest counterargument: a fixed 5-phase pipeline predetermines trace
structure, limiting scheduling analysis value. A truly general scaffold
should be an agentic loop (like OpenClaw) where the LLM decides tool order.

**Resolution**: v1 is explicitly framed as a **structured baseline** whose
traces serve as a controlled comparison point against OpenClaw's emergent
traces. Within-phase concurrency (`asyncio.gather` for N search calls, K
fetch calls) creates genuine scheduling decisions even with a fixed phase
DAG. OpenClaw remains the agentic-loop scaffold; research-agent is the
structured-pipeline scaffold. Both produce multi-span traces suitable for
scheduling analysis from complementary angles.

### Recommended Decision: **Option A**

---

## Requirements Summary

### Functional
- R1: Multi-phase research workflow (plan, search, fetch, extract, synthesize, answer)
- R2: Each phase emits canonical v5 trace records (action, event, summary)
- R3: Compatible with existing simulator (cloud_model and local_model replay)
- R4: Compatible with Gantt viewer (action_type -> span type mapping)
- R5: Provider-agnostic: works with any OpenAI-compatible endpoint
- R6: No reference-answer leakage at any phase
- R7: Host-mode execution (no container required)

### Non-Functional
- R8: Reuse existing `WebSearchTool` / `WebFetchTool` from OpenClaw where possible
- R9: All existing tests continue to pass after migration
- R10: Paid smoke test with cost cap ($5 USD per run)

---

## Current-State Findings

### qwen-deep-research scaffold
- **`src/agents/qwen_deep_research/runner.py:34-272`**: `QwenDeepResearchRunner` -- single streaming LLM call
- **Line 86-110**: One `TraceAction(action_type="llm_call", action_id="llm_0", iteration=0)` -- always exactly one span
- **Line 75-80**: `scaffold_capabilities={"tools": [], "memory": False, "skills": False, "file_ops": "none"}` -- no tool support declared
- **Line 152-173**: `_build_messages()` renders prompt via `render_research_prompt()`, correctly excludes `reference_answer`
- **Line 218-271**: `_call_streaming()` -- standard OpenAI streaming with TTFT/TPOT measurement

### Registration points (all must be updated)
1. **`src/agents/benchmarks/_research.py:316`**: `SUPPORTED_SCAFFOLDS = {"openclaw", "qwen-deep-research"}`
2. **`src/agents/benchmarks/_research.py:362-368`**: `build_runner()` dispatch branch for `qwen-deep-research`
3. **`src/trace_collect/cli.py:59`**: `choices=["openclaw", "qwen-deep-research"]`
4. **`src/agents/capabilities.py:13-20`**: `all_scaffolds()` dynamically reads from plugins (auto-updates)
5. **`tests/test_qwen_deep_research_runner.py`**: 4 tests directly testing QwenDeepResearchRunner
6. **`tests/test_deep_research_bench_plugin.py:109,113-124`**: tests referencing `qwen-deep-research` scaffold

### OpenClaw tools (reusable)
- **`src/agents/openclaw/tools/web.py:65-234`**: `WebSearchTool` -- Brave, DuckDuckGo, Tavily, Searxng, Jina backends
- **`src/agents/openclaw/tools/web.py:235-458`**: `WebFetchTool` -- Jina Reader + readability-lxml fallback, SSRF protection

### Trace infrastructure
- **`src/harness/trace_logger.py:35-90`**: `TraceLogger` -- emits `trace_metadata`, `action`, `event`, `summary` records
- **`src/agents/base.py`**: `TraceAction` dataclass with `to_dict()` for v5 serialization
- **`src/trace_collect/simulator.py`**: groups actions by iteration, replays `llm_call` and `tool_exec` types
- **`demo/gantt_viewer/backend/payload.py`**: maps `action_type` to span types (`llm_call`->`llm`, `tool_exec`->`tool`)

### Prompt templates
- **`configs/prompts/deep_research_bench/default.md`**: generic research prompt with `{{task}}` placeholder
- **`configs/prompts/browsecomp/default.md`**: browsing-comprehension prompt with `{{task}}` placeholder

---

## Architecture Plan

### Scaffold Interface

The new scaffold `research-agent` implements the `Runner` protocol from
`src/agents/benchmarks/base.py:20-35`:

```
async def run_task(task, *, attempt_ctx, prompt_template) -> AttemptResult
```

### Runner Structure

```
src/agents/research_agent/
    __init__.py          # exports ResearchAgentRunner
    runner.py            # main runner: orchestrates phases
    phases.py            # phase definitions (plan, search, fetch, extract, synthesize)
    tools.py             # thin wrappers around OpenClaw web tools for trace emission
    evidence.py          # evidence accumulation data model
```

### Phase Architecture

The runner executes a fixed sequence of phases per task. Each phase consists
of one or more LLM calls and/or tool executions, all individually traced.

```
Phase 0: INIT
  - Log trace_metadata record
  - Render task prompt from benchmark template
  - Record scaffold_capabilities

Phase 1: PLAN (iteration=0)
  - llm_call: Given the task, generate N search queries
  - event: PHASE_TRANSITION -> "plan"
  - Output: list[str] of search queries

Phase 2: SEARCH (iteration=1, concurrent tool_execs via asyncio.gather)
  - tool_exec per query: web_search(query) via WebSearchTool
  - All N queries execute concurrently (asyncio.gather)
  - Each gets unique action_id but shares iteration=1
  - event: PHASE_TRANSITION -> "search"
  - Output: list[SearchResult] with titles, URLs, snippets

Phase 3: FETCH (iteration=2, concurrent tool_execs via asyncio.gather)
  - tool_exec per top-K URL: web_fetch(url) via WebFetchTool
  - All K fetches execute concurrently (asyncio.gather)
  - Each gets unique action_id but shares iteration=2
  - event: PHASE_TRANSITION -> "fetch"
  - Output: list[FetchedPage] with URL, content, fetch timing

Phase 4: EXTRACT (iteration=3)
  - llm_call: Given fetched pages + task, extract evidence passages
  - event: PHASE_TRANSITION -> "extract"
  - Output: list[Evidence] with source_url, passage, relevance_note

Phase 5: SYNTHESIZE (iteration=4)
  - llm_call: Given evidence + task, produce final answer
  - event: PHASE_TRANSITION -> "synthesize"
  - Output: str final answer

Phase 6: EMIT
  - Log summary record with aggregated timing/tokens
  - Return AttemptResult
```

### Iteration Numbering

Each phase uses a distinct `iteration` value for its actions:
- Phase 1 (plan): iteration=0
- Phase 2 (search): iteration=1 (all search tool_execs share one iteration)
- Phase 3 (fetch): iteration=2 (all fetch tool_execs share one iteration)
- Phase 4 (extract): iteration=3
- Phase 5 (synthesize): iteration=4

This maps cleanly to the simulator's iteration-grouped replay and the Gantt
viewer's iteration lanes.

### Model/Provider Abstraction

The runner accepts the same `model`, `api_base`, `api_key` parameters as
`QwenDeepResearchRunner`. It uses `create_async_openai_client()` from
`src/llm_call/__init__.py` for the LLM client. Any OpenAI-compatible endpoint
works (OpenRouter, DashScope, OpenAI, SiliconFlow, local vLLM).

### Tool/Search/Fetch Abstraction

Rather than reimplementing web tools, the runner instantiates OpenClaw's
`WebSearchTool` and `WebFetchTool` directly:

```python
from agents.openclaw.tools.web import WebSearchTool, WebFetchTool
```

Tool execution results are wrapped in `TraceAction(action_type="tool_exec")`
with timing, input args, and output content -- matching the exact schema that
OpenClaw's `TraceCollectorHook` produces.

### WebSearchConfig Construction

`WebSearchTool` requires a `WebSearchConfig` (from `agents.openclaw.config.schema`).
The runner constructs it at init time:
- Default: `WebSearchConfig(provider="duckduckgo")` (free, no API key)
- Override via `run_config` extras: pass `search_provider`, `search_api_key`
  through benchmark YAML extras or CLI `--mcp-config`-style mechanism
- The `tools.py` wrapper catches all exceptions at the OpenClaw boundary,
  converting them to structured error results

### scaffold_capabilities Declaration

The new scaffold declares in `trace_metadata`:
```python
scaffold_capabilities={
    "tools": ["web_search", "web_fetch"],
    "memory": False,
    "skills": False,
    "file_ops": "none",
}
```
This is used by downstream analysis and Gantt viewer metadata display.

### Evidence Model

```python
@dataclass
class Evidence:
    source_url: str
    passage: str           # extracted text
    relevance_note: str    # LLM's note on why this is relevant
    fetch_timestamp: float # when the page was fetched
```

Evidence objects are accumulated across phases and serialized into the
synthesis prompt. They are also stored in the trace's final summary for
downstream analysis.

### Trace Schema Mapping

| Phase | action_type | action_id pattern | iteration |
|-------|-------------|-------------------|-----------|
| Plan | `llm_call` | `llm_0` | 0 |
| Search | `tool_exec` | `tool_search_0`, `tool_search_1`, ... | 1 |
| Fetch | `tool_exec` | `tool_fetch_0`, `tool_fetch_1`, ... | 2 |
| Extract | `llm_call` | `llm_1` | 3 |
| Synthesize | `llm_call` | `llm_2` | 4 |

Event records logged at phase transitions:
```json
{"type": "event", "category": "SESSION", "event": "phase_transition",
 "data": {"phase": "search", "prev_phase": "plan"}}
```

### Simulator Compatibility

The simulator (`src/trace_collect/simulator.py`) processes traces by:
1. Reading v5 JSONL records
2. Grouping actions by `iteration`
3. For each `llm_call`: replay timing (cloud_model) or send real request (local_model)
4. For each `tool_exec`: replay from trace or re-execute in container

**research-agent traces are compatible without any simulator changes:**
- `research-agent` traces have `execution_environment: "host"`
- `_is_host_mode()` at `simulator.py:307` returns `True`
- `_prepare_host_session()` at line 406 sets `container=None`
- All `tool_exec` actions hit the `ctr is None` branch at `simulator.py:861`
  (cloud_model) or `simulator.py:627` (local_model), producing
  `replay_source="skipped_host_mode"`
- This means all tool timings are replayed from the source trace data
- No `_should_replay_tool()` addition is needed -- the host-mode path
  already handles this correctly

**No simulator code changes required.** Phase 5 (simulator compatibility)
is reduced to integration testing only.

### Gantt Compatibility

The Gantt viewer (`demo/gantt_viewer/backend/payload.py`) maps:
- `action_type="llm_call"` -> `"llm"` span (blue)
- `action_type="tool_exec"` -> `"tool"` span (green)

Research-agent traces use exactly these action types. Phase transitions
appear as `event` records which the Gantt viewer already renders as markers.
**No changes needed in the Gantt viewer.**

### BrowseComp Source URLs

BrowseComp tasks include `source_urls` -- URLs that are part of the problem
context (they tell the agent where to look). These are **not** reference
answers. The scaffold:

1. Passes `source_urls` to the planning phase as available context
2. The planner may include these URLs in its fetch targets
3. The `reference_answer` field is **never** passed to any LLM call or tool
4. Evidence extraction operates only on fetched page content, not reference answers

This is identical to how a human would use BrowseComp: you're given URLs
and a question, you read the pages, you answer.

---

## Migration Plan

### Files to DELETE
| File | Reason |
|------|--------|
| `src/agents/qwen_deep_research/__init__.py` | Entire scaffold removed |
| `src/agents/qwen_deep_research/runner.py` | Entire scaffold removed |
| `tests/test_qwen_deep_research_runner.py` | Tests for removed scaffold |

### Files to CREATE
| File | Purpose |
|------|---------|
| `src/agents/research_agent/__init__.py` | Export `ResearchAgentRunner` |
| `src/agents/research_agent/runner.py` | Main runner with phase orchestration |
| `src/agents/research_agent/phases.py` | Phase definitions and execution logic |
| `src/agents/research_agent/tools.py` | Traced tool wrappers |
| `src/agents/research_agent/evidence.py` | Evidence data model |
| `tests/test_research_agent_runner.py` | Unit tests for new scaffold |
| `tests/test_research_agent_phases.py` | Unit tests for phase logic |

### Files to MODIFY
| File | Change |
|------|--------|
| `src/agents/benchmarks/_research.py:316` | `SUPPORTED_SCAFFOLDS`: replace `"qwen-deep-research"` with `"research-agent"` |
| `src/agents/benchmarks/_research.py:355-369` | `build_runner()`: replace qwen dispatch with research-agent dispatch |
| `src/trace_collect/cli.py:59` | `choices`: replace `"qwen-deep-research"` with `"research-agent"` |
| `tests/test_deep_research_bench_plugin.py:109,112-124` | Update assertions for new scaffold |
| `tests/test_browsecomp_plugin.py:123` | Update `runtime_mode_for("qwen-deep-research")` assertion |
| `tests/test_simulate_cloud_model.py:431,493,628,688,801,840` | Update fixture scaffold strings to `"research-agent"` |
| `tests/test_collector_prompt_resolution.py:50-62,143-165` | Update scaffold choice tests |
| `tests/test_capabilities.py:21,33,37` | Update expected scaffold list |
| `demo/gantt_viewer/tests/test_payload.py:131,164` | Update test fixture scaffold names |
| `README.md:50-51` | Update scaffold table |
| `src/trace_collect/CLAUDE.md` | Update scaffold table and docs |

### Docs to UPDATE
| File | Change |
|------|--------|
| `src/trace_collect/CLAUDE.md` | Update scaffold table, remove qwen reference |
| `README.md` | Update benchmark/scaffold table |
| `CLAUDE.md` | No change needed (scaffold-agnostic) |

---

## Implementation Phases

### Phase 1: Scaffold skeleton + migration wiring (no LLM calls)
1. Create `src/agents/research_agent/` directory structure
2. Implement `ResearchAgentRunner` with `run_task()` stub that raises `NotImplementedError`
3. Implement `Evidence` dataclass in `evidence.py`
4. Wire into `_research.py`: update `SUPPORTED_SCAFFOLDS`, `build_runner()`
5. Wire into `cli.py`: update scaffold choices
6. Delete `src/agents/qwen_deep_research/` directory
7. Delete `tests/test_qwen_deep_research_runner.py`
8. Update `tests/test_deep_research_bench_plugin.py`
9. Run: `conda run -n ML python -m pytest tests/test_deep_research_bench_plugin.py -v`

**Gate**: all existing tests pass with updated references

### Phase 2: Tool wrappers + trace emission
1. Implement `tools.py`: `TracedWebSearch` and `TracedWebFetch` that wrap OpenClaw tools and emit `TraceAction(action_type="tool_exec")` records
2. Unit test tool wrappers with mock HTTP responses
3. Run: `conda run -n ML python -m pytest tests/test_research_agent_phases.py -v`

**Gate**: tool wrappers produce correct v5 trace records

### Phase 3: Phase logic implementation
1. Implement `phases.py`: `PlanPhase`, `SearchPhase`, `FetchPhase`, `ExtractPhase`, `SynthesizePhase`
2. Each phase: accepts inputs, calls LLM/tools, returns structured output + trace actions
3. LLM calls use `create_async_openai_client()` streaming with TTFT/TPOT measurement (reuse pattern from `QwenDeepResearchRunner._call_streaming`)
4. Unit test each phase with mock LLM/tool responses
5. Run: `conda run -n ML python -m pytest tests/test_research_agent_phases.py -v`

**Gate**: all phases produce correct trace records in isolation

### Phase 4: Runner integration
1. Implement `runner.py`: orchestrate phases sequentially, accumulate evidence, emit summary
2. Integration test: run full pipeline with mocked LLM + mocked tools, verify complete trace
3. Verify trace is valid v5 JSONL with correct iteration numbering
4. Run: `conda run -n ML python -m pytest tests/test_research_agent_runner.py -v`

**Gate**: full pipeline produces a valid multi-span trace

### Phase 5: Simulator + Gantt integration testing (no code changes)
1. Construct a synthetic research-agent trace fixture (hand-crafted v5 JSONL with
   3 llm_call + N tool_exec actions across 5 iterations)
2. Test: load in cloud_model simulator, verify all actions replayed via
   `skipped_host_mode` path (no container started)
3. Test: load in Gantt viewer payload parser, verify correct span count and types
4. Run: `conda run -n ML python -m pytest tests/test_simulator_validation.py tests/test_simulate_cloud_model.py -v`

**Gate**: simulator replays research-agent traces without error; Gantt parses correctly
**Note**: No simulator code changes needed -- host-mode `ctr is None` path
handles all tool_exec replay automatically.

### Phase 6: Paid smoke test (REQUIRES REVIEW GATE before proceeding)
1. Run against 1 task from deep-research-bench with a real LLM endpoint
2. Run against 1 task from browsecomp with a real LLM endpoint
3. Verify: multi-span trace, search results returned, pages fetched, answer produced
4. Cost cap: $5 USD total (use `--sample 1` and a cost-efficient model)
5. Token accounting: log prompt_tokens + completion_tokens per phase
6. Run:
   ```bash
   conda run -n ML python -m trace_collect.cli \
     --scaffold research-agent \
     --benchmark deep-research-bench \
     --provider dashscope --model qwen-plus-latest \
     --sample 1 --mcp-config none
   ```

**Gate**: real trace has >= 3 `llm_call` actions and >= 1 `tool_exec` action

### Phase 7: Gantt verification
1. Load smoke test trace in Gantt viewer
2. Verify: multiple colored spans across iterations, phase transition markers visible
3. Run:
   ```bash
   conda run -n ML python -m trace_collect.cli gantt-serve --dev
   # Load the smoke test trace in browser
   ```

**Gate**: Gantt viewer renders all spans and markers correctly

---

## Test Plan

### Unit Tests (`tests/test_research_agent_runner.py`)
- `test_research_agent_runner_writes_v5_trace`: full mock pipeline produces valid trace
- `test_research_agent_no_reference_leak`: `reference_answer` never appears in any LLM messages
- `test_research_agent_empty_search_graceful`: runner handles zero search results
- `test_research_agent_fetch_failure_graceful`: runner handles fetch errors without crashing
- `test_research_agent_iteration_numbering`: each phase uses correct iteration values
- `test_research_agent_action_id_uniqueness`: no duplicate action_ids in trace
- `test_research_agent_summary_aggregation`: summary totals match action-level sums

### Unit Tests (`tests/test_research_agent_phases.py`)
- `test_plan_phase_generates_queries`: plan phase returns non-empty query list
- `test_search_phase_traces_tool_exec`: search produces tool_exec actions
- `test_fetch_phase_traces_tool_exec`: fetch produces tool_exec actions
- `test_extract_phase_produces_evidence`: extract returns Evidence objects
- `test_synthesize_phase_uses_evidence`: synthesis prompt includes evidence passages

### Integration Tests (`tests/test_deep_research_bench_plugin.py` -- updated)
- `test_deep_research_bench_builds_research_agent_runner`: plugin dispatches correctly
- `test_deep_research_bench_runtime_and_runner_gating`: `research-agent` in supported scaffolds
- `test_collect_traces_dispatches_research_agent_runner`: end-to-end with mock LLM

### Simulator Checks (no code changes -- testing only)
- `test_simulator_replays_research_agent_trace`: cloud_model replay of a research-agent trace
  succeeds via `ctr is None` host-mode path, all tool_execs get `skipped_host_mode`
- Existing simulator tests pass unchanged (regression)

### Gantt Checks
- Manual verification: load trace, confirm multi-lane rendering
- Automated: `test_gantt_payload_parses_research_agent_trace`: TraceData.load succeeds, correct span count

### Paid Smoke Tests (post-review-gate only)
- 1 task x deep-research-bench x dashscope qwen-plus-latest
- 1 task x browsecomp x dashscope qwen-plus-latest
- Verify: >= 3 llm_call spans, >= 1 tool_exec span, non-empty final answer
- Cost cap: $5 USD total

---

## Research Integrity Review Checklist

- [ ] `reference_answer` is never passed to any LLM call (plan, extract, synthesize)
- [ ] `reference_answer` is never passed to any tool call
- [ ] BrowseComp `source_urls` are treated as inference-time information (part of problem, not the answer)
- [ ] No benchmark-specific branching in scaffold code (no `if benchmark == "browsecomp"`)
- [ ] Search queries are generated from `problem_statement` only, not from `reference_answer`
- [ ] Evidence extraction operates on fetched content, not on reference data
- [ ] All scaffold parameters are configurable, no magic numbers
- [ ] Phase count and structure are general, not tuned to specific benchmarks
- [ ] Prompt templates are per-benchmark (already in `configs/prompts/`), scaffold code is generic

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Search APIs fail / rate-limit | No search results -> poor answers | DuckDuckGo fallback (no API key needed); graceful degradation in search phase |
| Jina Reader rate-limits fetch | No page content -> extraction fails | readability-lxml fallback already in WebFetchTool |
| LLM doesn't follow JSON output format for plan phase | Unparseable search queries | Robust parsing with fallback to raw text splitting; retry once |
| Too many search results -> context overflow | Synthesis LLM call exceeds context window | Cap at K=5 URLs to fetch, truncate page content to configurable max_chars |
| Simulator unexpectedly needs changes | Replay fails on research-agent traces | Verified: host-mode `ctr is None` path handles all tool replay; Phase 5 is testing-only |
| Existing traces reference `qwen-deep-research` scaffold name | Analysis scripts break on old data | Old traces remain valid v5 JSONL; scaffold field is metadata, not a runtime dependency |

### Rollback / Stop Conditions
- If Phase 1 fails (test breakage): revert deletion, fix wiring first
- If Phase 3 LLM integration produces degenerate traces: pause, inspect prompts, do not proceed to paid tests
- If Phase 6 smoke test exceeds $5 cost cap: abort immediately, investigate token usage
- If review gate identifies reference-answer leakage: **hard stop**, fix before any further work

---

## Acceptance Criteria

1. `conda run -n ML python -m pytest tests/ -v` -- all tests pass (0 failures)
2. `conda run -n ML python -m pytest tests/test_research_agent_runner.py -v` -- >= 7 passing tests
3. `grep -r "qwen-deep-research" src/ configs/ tests/` -- returns 0 matches (fully removed)
4. `grep -r "qwen_deep_research" src/ tests/` -- returns 0 matches (fully removed)
5. Smoke test trace for deep-research-bench contains:
   - 1 `trace_metadata` record with `scaffold: "research-agent"`
   - >= 3 `action` records with `action_type: "llm_call"`
   - >= 1 `action` records with `action_type: "tool_exec"`
   - 1 `summary` record with `n_iterations >= 3`
   - Phase transition `event` records
6. Smoke test trace for browsecomp: same criteria as #5
7. Gantt viewer renders smoke test trace with distinct spans per phase
8. Simulator cloud_model replay of smoke test trace completes without error
9. No occurrence of `reference_answer` in any `messages_in` field across all trace actions

---

## Open Questions

1. **Max search queries per task**: Should the plan phase generate 3, 5, or a configurable number? **Proposed**: 5, configurable via `run_config`.
2. **Max pages to fetch**: Should we fetch top-K URLs from search results? **Proposed**: K=5, configurable.
3. **Page content truncation**: Max chars per fetched page? **Proposed**: 30,000 chars (matches WebFetchTool default of 50K but with headroom for multiple pages).
4. **Search provider default**: Which search backend for smoke tests? **Proposed**: DuckDuckGo (free, no API key).
5. **Should the scaffold support iterative refinement loops?** E.g., if initial evidence is insufficient, loop back to search. **Proposed**: Not in v1 -- keep it simple, add iteration in v2 if analysis shows need.
6. **Should research-agent support MCP tools in addition to built-in web tools?** **Proposed**: Not in v1. OpenClaw already handles MCP for research benchmarks.

---

## ADR: Replace qwen-deep-research with research-agent

### Decision
Delete the `qwen-deep-research` scaffold entirely and replace it with a new
`research-agent` scaffold that implements a multi-phase research workflow
with step-level tracing.

### Drivers
1. Current scaffold produces 1 trace span -- useless for scheduling analysis
2. Name conflates scaffold identity with provider/model choice
3. No tool execution traces -- cannot study search/fetch scheduling effects
4. Existing OpenClaw web tools are mature and directly reusable

### Alternatives Considered
- **Option B (keep alias)**: rejected -- pure compatibility debt with no research value
- **Option C (DashScope API)**: rejected -- opaque black-box, not locally simulatable

### Why Chosen
Option A (hard delete + new scaffold) provides a clean break with no
migration debt. The new scaffold produces rich multi-span traces that are
the prerequisite for scheduling/simulation research on research-style
benchmarks.

### Consequences
- All existing `qwen-deep-research` traces in `traces/` become historical artifacts
  (still valid v5 JSONL, just from a removed scaffold)
- Analysis scripts that filter on `scaffold=="qwen-deep-research"` will match
  only legacy data -- this is correct behavior
- Need to update any external documentation that references the old scaffold name

### Follow-ups
- v2: Add iterative refinement loop (search -> assess -> re-search) if v1 analysis
  shows single-pass search is insufficient
- Consider DashScope DeepResearch API as a separate scaffold if they expose
  step-level hooks in future
- Add more search backends (Perplexity, Google Custom Search) as providers mature
