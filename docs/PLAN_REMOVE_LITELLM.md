# Plan: Remove litellm dependency

> **Motivation**: The `litellm` library has been compromised by a supply-chain vulnerability.
> All litellm usage must be replaced with a secure alternative.

---

## Findings

### Current litellm usage

Only **2 files** in this codebase import litellm, both indirectly via `mini-swe-agent`:

| File | Line | Usage |
|------|------|-------|
| `src/agents/mini_swe_code_agent.py` | 273–282 | `from minisweagent.models.litellm_model import LitellmModel` |
| `src/trace_collect/replayer.py` | 385, 428–436 | `from minisweagent.models.litellm_model import LitellmModel` |

**No direct `litellm` dependency** exists in `pyproject.toml` — it arrives transitively through `mini-swe-agent>=2.0`.

**How `LitellmModel` is invoked** (identical in both files):
```python
lm = LitellmModel(
    model_name=f"openai/{self.model}",
    model_kwargs={
        "api_base": self.api_base,   # e.g. http://localhost:8000/v1
        "api_key": self.api_key,     # e.g. "EMPTY" for vLLM
        "drop_params": True,
        "temperature": 0.0,
    },
    cost_tracking="ignore_errors",
)
```

All endpoints are **OpenAI-compatible APIs** (vLLM, ModelScope DashScope, OpenRouter). No litellm-specific features (provider routing, cost calculation, model registry) are used.

---

## Replacement Strategy

### Why `openai` (not another gateway library)

- The `openai` Python library is **already a direct dependency** in `pyproject.toml`
- It natively supports **any OpenAI-compatible endpoint**: vLLM, ModelScope (DashScope), OpenRouter, DashScope, etc.
- It is maintained by OpenAI with a stable v1 API — no supply-chain risk from a third-party aggregator
- mini-swe-agent v2 uses **duck typing with protocols** — no subclassing required, any class implementing the right methods works

### mini-swe-agent v2 Model Protocol

A custom model must implement (no inheritance needed):

| Method | Purpose |
|--------|---------|
| `query(messages, **kwargs) -> dict` | Send messages to API, return response with metadata in `extra` |
| `format_message(**kwargs) -> dict` | Format user/assistant messages |
| `format_observation_messages(message, outputs, template_vars) -> list[dict]` | Format tool results |
| `get_template_vars(**kwargs) -> dict` | Return config for Jinja2 templating |
| `serialize() -> dict` | Return JSON-serializable config for trajectory saving |

---

## Implementation Steps

### Step 1: Create `src/agents/openai_model.py`

A new `OpenAICompatibleModel` class that:

1. Uses `openai.OpenAI(base_url=..., api_key=...)` for all API calls
2. Calls `client.chat.completions.create(model=..., messages=..., tools=[BASH_TOOL], ...)` 
3. Reuses mini-swe-agent's existing utilities:
   - `minisweagent.models.utils.actions_toolcall.parse_toolcall_actions` — parse tool calls
   - `minisweagent.models.utils.openai_multimodal.expand_multimodal_content` — multimodal support
   - `minisweagent.models.utils.format_toolcall_observation_messages` — format tool results
4. Sets `cost` to `0.0` (matching current `cost_tracking="ignore_errors"` behavior)
5. Accepts the same constructor kwargs as `LitellmModel`:
   - `model_name` (string, e.g. `"qwen3.5-plus"`)
   - `model_kwargs` (dict with `api_base`, `api_key`, `drop_params`, `temperature`)

### Step 2: Update `src/agents/mini_swe_code_agent.py`

```diff
- from minisweagent.models.litellm_model import LitellmModel
+ from agents.openai_model import OpenAICompatibleModel

- lm = LitellmModel(
+ lm = OpenAICompatibleModel(
      model_name=f"openai/{self.model}",
      ...
  )
```

### Step 3: Update `src/trace_collect/replayer.py`

Same changes as Step 2 — identical import and constructor swap.

### Step 4: Remove litellm from dependency chain

In `pyproject.toml`:
- If mini-swe-agent declares litellm as a hard dependency, add an override/exclusion
- If mini-swe-agent declares it as optional, simply don't install it

Verify that `mini-swe-agent` loads without litellm (v2 is duck-typed, the litellm module is only imported when explicitly used).

### Step 5: Documentation cleanup

Search and remove litellm references in:
- `docs/*.md` — any mentions
- Inline comments referencing litellm
- Config file comments or examples

### Step 6: Verify

- Run `pytest tests/` — confirm no regressions
- Verify the new model works with the vLLM endpoint (OpenAI-compatible API)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| mini-swe-agent hard-imports litellm at package level | v2 is duck-typed; `LitellmModel` is only imported when explicitly used |
| Cost tracking is lost | Already disabled via `cost_tracking="ignore_errors"`; no functional change |
| Provider-specific features missing | Only OpenAI-compatible APIs are used; no Anthropic/Gemini paths |
| Tool call parsing differs | Reusing mini-swe-agent's own `parse_toolcall_actions` utility |

---

## Files to Change

| File | Change |
|------|--------|
| `src/agents/openai_model.py` | **NEW** — OpenAICompatibleModel class |
| `src/agents/mini_swe_code_agent.py` | Import + constructor swap |
| `src/trace_collect/replayer.py` | Import + constructor swap |
| `pyproject.toml` | Exclude/remove litellm |
| `docs/*.md` | Remove litellm mentions |
