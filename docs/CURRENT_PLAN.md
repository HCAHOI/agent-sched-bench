# Plan: Openclaw LLM/Tool Event Emission + Step Deduplication

## Context

Openclaw traces 缺少 LLM 调用事件。Mini-swe 实时发射 `llm_start`/`llm_end`/`tool_start`/`tool_end`
(agent.py:399-535)，openclaw 只在 `after_iteration` 追溯发射 `tool_execute`/`tool_complete`，
且 `step` 记录中 `messages_in` 和 `raw_response` 与事件数据高度重复（占 trace 文件 ~90% 体积）。

**目标：**
1. 让 openclaw 发射与 mini-swe 对齐的实时 LLM/tool 事件
2. 将 `messages_in`/`raw_response` 从 `step` 移入事件，消除重复
3. 为 simulator 的 event 驱动架构和甘特图可视化提供统一的事件基础

---

## Architecture: Current vs Target

### Current (openclaw trace emission in `_session_runner.py:TraceCollectorHook`)

```
_runner.py:AgentRunner.run() iteration loop:
  hook.before_iteration(ctx)          → records _iter_start_ts (NO event emitted)
  response = _request_model(...)      → LLM call happens (NO event emitted)
  ctx.response = response
                                      → [NO hook point here]
  if has_tool_calls:
    hook.before_execute_tools(ctx)    → records _before_exec_ts (NO event emitted)
    _execute_tools(...)               → tool execution (NO event emitted)
  hook.after_iteration(ctx)           → POST-HOC emits:
                                         tool_execute events (line 186)
                                         tool_complete/tool_error events (line 216)
                                         llm_error event (only on error, line 258)
                                         step record with messages_in + raw_response (line 272)
```

### Target

```
_runner.py:AgentRunner.run() iteration loop:
  hook.before_iteration(ctx)          → emit llm_call_start (LLM, carries messages_in)
  response = _request_model(...)      → LLM call happens
  ctx.response = response
  hook.after_llm_response(ctx) [NEW]  → emit llm_call_end (LLM, carries raw_response + tokens)
  if has_tool_calls:
    hook.before_execute_tools(ctx)    → emit tool_exec_start per tool (TOOL, carries name + args)
    _execute_tools(...)               → tool execution
  hook.after_iteration(ctx)           → emit tool_exec_end per tool (TOOL, carries result + duration)
                                         emit SLIM step record (no messages_in, no raw_response)
```

---

## Phase 1: Add `after_llm_response` hook point

### `src/agents/openclaw/_hook.py`
- Add method to `AgentHook`:
  ```python
  async def after_llm_response(self, context: AgentHookContext) -> None:
      pass
  ```
- Add fan-out in `CompositeHook.after_llm_response()`

### `src/agents/openclaw/_runner.py` (~line 109-117)
Currently:
```python
response = await self._request_model(spec, messages_for_model, hook, context)
raw_usage = self._usage_dict(response.usage)
context.response = response
context.usage = dict(raw_usage)
context.tool_calls = list(response.tool_calls)
self._accumulate_usage(usage, raw_usage)

if response.has_tool_calls:
```

Change to:
```python
response = await self._request_model(spec, messages_for_model, hook, context)
raw_usage = self._usage_dict(response.usage)
context.response = response
context.usage = dict(raw_usage)
context.tool_calls = list(response.tool_calls)
self._accumulate_usage(usage, raw_usage)

await hook.after_llm_response(context)  # NEW — always called

if response.has_tool_calls:
```

One line insertion. No other changes to `_runner.py`.

---

## Phase 2: Emit LLM events in TraceCollectorHook

### `src/agents/openclaw/_session_runner.py`

**`before_iteration()` — emit `llm_call_start`:**
```python
async def before_iteration(self, context: AgentHookContext) -> None:
    self._iter_start_ts = time.monotonic()
    self._iter_start_wall = time.time()
    self.emit_event(
        LLM, "llm_call_start",
        {"messages_in": self._clone_messages(context.messages)},
        step_idx=context.iteration,
    )
```

**New `after_llm_response()` — emit `llm_call_end`:**
```python
async def after_llm_response(self, context: AgentHookContext) -> None:
    self._after_llm_ts = time.monotonic()
    usage = context.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    llm_latency_ms = (self._after_llm_ts - self._iter_start_ts) * 1000

    resp_dict = self._build_raw_response(
        context=context,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    self.emit_event(
        LLM, "llm_call_end",
        {
            "raw_response": resp_dict,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": round(llm_latency_ms, 2),
            "finish_reason": context.response.finish_reason if context.response else None,
        },
        step_idx=context.iteration,
    )
```

---

## Phase 3: Real-time tool events

### `src/agents/openclaw/_session_runner.py`

**`before_execute_tools()` — emit `tool_exec_start`:**
```python
async def before_execute_tools(self, context: AgentHookContext) -> None:
    self._before_exec_ts = time.monotonic()
    if context.tool_calls:
        for tc in context.tool_calls:
            self._tool_start_ts[tc.name] = time.monotonic()
            self.emit_event(
                MCP if tc.name.startswith("mcp_") else TOOL,
                "tool_exec_start",
                {
                    "tool_name": tc.name,
                    "args_preview": json.dumps(tc.arguments, ensure_ascii=False)[:200],
                },
                step_idx=context.iteration,
            )
```

**`after_iteration()` — emit `tool_exec_end` (replaces post-hoc `tool_execute` + `tool_complete`):**
- **Remove** the post-hoc `tool_execute` emission block (current lines 161-188)
- **Rename** `tool_complete`/`tool_error` to `tool_exec_end` with `success` field
- Keep `llm_error` and `max_iterations` events as-is (error paths)

---

## Phase 4: Slim down `step` record

### `src/agents/openclaw/eval/types.py` (EvalTraceStep)

Remove three fields:
- `messages_in` → now in `llm_call_start` event
- `raw_response` → now in `llm_call_end` event
- `llm_output` → derivable from `raw_response` in `llm_call_end` event

**Keep** all other fields: `step_idx`, `phase`, `prompt_tokens`, `completion_tokens`,
`llm_latency_ms`, `tool_name`, `tool_args`, `tool_result`, `tool_duration_ms`,
`tool_success`, `tool_timeout`, `tool_ts_start`, `tool_ts_end`, `ts_start`, `ts_end`, `extra`.

### `src/agents/openclaw/_session_runner.py` (after_iteration)

Stop populating removed fields. Step becomes a lightweight iteration index.

### `trace_metadata` header

Bump `trace_format_version` to `3` in openclaw's metadata emission (in `SessionRunner.run()`).

---

## Phase 5: Update consumers

### `src/trace_collect/simulator.py`

Currently reads step fields:
- `step["messages_in"]` → send to local model
- `step["completion_tokens"]` → force token count
- `step["tool_name"]`, `step["tool_args"]` → execute tool
- `step["raw_response"]` → fallback for `_unwrap_tool_args`

Change to event-based reading:
```python
def load_trace_events_and_steps(trace_path, agent_id):
    """Load events and steps, pairing llm_call_start/end with step records."""
    events_by_step: dict[int, dict] = {}  # step_idx → {llm_start, llm_end, ...}
    steps = []
    for record in read_jsonl(trace_path):
        if record.get("agent_id") != agent_id:
            continue
        if record["type"] == "event":
            idx = record.get("step_idx", 0)
            events_by_step.setdefault(idx, {})[record["event"]] = record
        elif record["type"] == "step":
            steps.append(record)
    return steps, events_by_step
```

For each step, get `messages_in` from `events_by_step[i]["llm_call_start"]["data"]["messages_in"]`,
fall back to `step["messages_in"]` for v2 traces.

### `src/trace_collect/trace_inspector.py`

- Update `_normalize_legacy_event()`:
  - Add mappings: `tool_execute` → `tool_exec_start`, `tool_complete` → `tool_exec_end`
  - Mini-swe mappings: `llm_start` → `llm_call_start`, `llm_end` → `llm_call_end`,
    `tool_start` → `tool_exec_start`, `tool_end` → `tool_exec_end`
- Timeline rendering: use `llm_call_start`/`llm_call_end` pairs as Gantt spans
- Any code reading `step.messages_in` or `step.raw_response` → check events first

### `tests/test_trace_inspector.py`

- Update existing normalization tests for renamed events
- Add test: slim step records (no messages_in/raw_response)
- Add test: v2 backward compat (old traces still work)

---

## Event Data Contracts

### `llm_call_start` (category: LLM)
```json
{
  "type": "event", "category": "LLM", "event": "llm_call_start",
  "agent_id": "...", "step_idx": 0, "ts": 1775489215.85,
  "data": {
    "messages_in": [{"role": "system", "content": "..."}, ...]
  }
}
```

### `llm_call_end` (category: LLM)
```json
{
  "type": "event", "category": "LLM", "event": "llm_call_end",
  "agent_id": "...", "step_idx": 0, "ts": 1775489218.62,
  "data": {
    "raw_response": {"choices": [...], "usage": {...}},
    "prompt_tokens": 8181,
    "completion_tokens": 60,
    "llm_latency_ms": 2774.0,
    "finish_reason": "tool_calls"
  }
}
```

### `tool_exec_start` (category: TOOL|MCP)
```json
{
  "type": "event", "category": "TOOL", "event": "tool_exec_start",
  "agent_id": "...", "step_idx": 0, "ts": 1775489218.63,
  "data": {
    "tool_name": "list_dir",
    "args_preview": "{\"path\": \".\"}"
  }
}
```

### `tool_exec_end` (category: TOOL|MCP)
```json
{
  "type": "event", "category": "TOOL", "event": "tool_exec_end",
  "agent_id": "...", "step_idx": 0, "ts": 1775489218.74,
  "data": {
    "tool_name": "list_dir",
    "success": true,
    "duration_ms": 2.0,
    "result_preview": "..."
  }
}
```

### Slim `step` record (v3, no messages_in/raw_response/llm_output)
```json
{
  "type": "step", "agent_id": "...", "step_idx": 0,
  "phase": "acting",
  "prompt_tokens": 8181, "completion_tokens": 60,
  "llm_latency_ms": 2774.0,
  "tool_name": "list_dir", "tool_args": "{...}",
  "tool_result": "...", "tool_duration_ms": 2.0,
  "tool_success": true, "tool_timeout": null,
  "ts_start": 1775489215.97, "ts_end": 1775489218.74,
  "tool_ts_start": 1775489218.63, "tool_ts_end": 1775489218.74,
  "extra": {}
}
```

---

## Canonical Event Name Mapping

| Semantic | mini-swe (write) | openclaw (write, new) | Consumer canonical |
|----------|------------------|-----------------------|-------------------|
| LLM call start | `llm_start` | `llm_call_start` | `llm_call_start` |
| LLM call end | `llm_end` | `llm_call_end` | `llm_call_end` |
| Tool exec start | `tool_start` | `tool_exec_start` | `tool_exec_start` |
| Tool exec end | `tool_end` | `tool_exec_end` | `tool_exec_end` |

Consumer-side normalization in `trace_inspector.py:_normalize_legacy_event()` maps
mini-swe names → canonical. Old openclaw events (`tool_execute`, `tool_complete`) also mapped.

---

## Migration (No Backward Compat)

- **不保留 v2 兼容**。删除旧 trace，保留两个示例 trace 并用临时脚本转换为 v3 格式。
- 临时转换脚本用完即删。
- Consumer 代码只需要处理 v3 格式。

---

## Verification Checklist

- [ ] Openclaw smoke test produces `llm_call_start`/`llm_call_end`/`tool_exec_start`/`tool_exec_end` events
- [ ] Events have correct real-time timestamps (not post-hoc)
- [ ] `step` records no longer contain `messages_in`, `raw_response`, or `llm_output`
- [ ] Trace file size significantly reduced (~60-80%)
- [ ] `trace_inspector inspect <trace> events` shows new event types
- [ ] `trace_inspector inspect <trace> timeline` renders LLM spans from event pairs
- [ ] Simulator works with both v2 (old) and v3 (new) traces
- [ ] `pytest tests/test_trace_inspector.py` passes
- [ ] New tests for event emission, slim step, backward compat

---

## File Change Summary

| File | Change | Lines |
|------|--------|-------|
| `src/agents/openclaw/_hook.py` | Add `after_llm_response()` to AgentHook + CompositeHook | ~10 |
| `src/agents/openclaw/_runner.py` | Insert `await hook.after_llm_response(context)` after LLM call | ~1 |
| `src/agents/openclaw/_session_runner.py` | Emit 4 real-time events, slim step, remove post-hoc tool_execute | ~60 |
| `src/agents/openclaw/eval/types.py` | Remove `messages_in`, `raw_response`, `llm_output` from EvalTraceStep | ~15 |
| `src/trace_collect/simulator.py` | Event-based data reading with v2 fallback | ~30 |
| `src/trace_collect/trace_inspector.py` | Update normalization map + timeline event pairs | ~20 |
| `tests/test_trace_inspector.py` | Update + add tests | ~30 |

## Implementation Order

1. Phase 1: Hook point (`_hook.py`, `_runner.py`) — foundation
2. Phase 2: LLM events (`_session_runner.py`) — immediate value
3. Phase 3: Tool events (`_session_runner.py`) — complete the picture
4. Phase 4: Slim step (`eval/types.py`, `_session_runner.py`) — deduplication
5. Phase 5: Consumers (`simulator.py`, `trace_inspector.py`, tests) — integration
