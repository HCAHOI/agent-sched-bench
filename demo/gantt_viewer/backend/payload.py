"""Build Gantt payloads from canonical trace data.

The registries live in this module and are shipped inside each payload so
renderers can extend span and marker types without hard-coded frontend logic.
"""

from __future__ import annotations

import json
from typing import Any

from trace_collect.trace_inspector import TraceData

_MARKER_CATEGORIES = frozenset({"SCHEDULING", "SESSION", "CONTEXT", "MCP"})

_LLM_CONTENT_MAX = 1000

_TOOL_ARGS_MAX = 200

_TOOL_PRIMARY_FIELDS: tuple[str, ...] = (
    "path",
    "file_path",
    "filepath",
    "command",
    "cmd",
    "pattern",
    "query",
    "url",
)

ACTION_TYPE_MAP: dict[str, str] = {
    "llm_call": "llm",
    "tool_exec": "tool",
    "mcp_call": "mcp",
}

DEFAULT_SPAN_REGISTRY: dict[str, dict[str, Any]] = {
    "llm":        {"color": "#00E5FF", "label": "LLM Call",   "order": 0},
    "tool":       {"color": "#FF6D00", "label": "Tool Exec",  "order": 1},
    "scheduling": {"color": "#76FF03", "label": "Scheduling", "order": 2},
    "mcp":        {"color": "#AB47BC", "label": "MCP Call",   "order": 3},
}

DEFAULT_MARKER_REGISTRY: dict[str, dict[str, str]] = {
    "message_dispatch":     {"symbol": "diamond", "color": "#76FF03", "label": "Message Dispatch"},
    "session_lock_acquire": {"symbol": "diamond", "color": "#76FF03", "label": "Session Lock Acquire"},
    "session_load":         {"symbol": "dot",     "color": "#76FF03", "label": "Session Load"},
    "message_list_build":   {"symbol": "dot",     "color": "#4FC3F7", "label": "Message List Build"},
    "session_turn_save":    {"symbol": "dot",     "color": "#76FF03", "label": "Session Turn Save"},
    "task_complete":        {"symbol": "flag",    "color": "#FF6D00", "label": "Task Complete"},
    "llm_error":            {"symbol": "cross",   "color": "#FF1744", "label": "Llm Error"},
    "max_iterations":       {"symbol": "cross",   "color": "#FF1744", "label": "Max Iterations"},
    "_default":             {"symbol": "dot",     "color": "#6b7280", "label": "Default"},
}

def _raw_response_message(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if choices:
        return choices[0].get("message") or {}
    return raw_response.get("message") or {}

def _raw_response_text(raw_response: dict[str, Any]) -> str:
    message = _raw_response_message(raw_response)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part)

def _raw_response_tool_calls(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    message = _raw_response_message(raw_response)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return [tc for tc in tool_calls if isinstance(tc, dict)]

    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "id": block.get("id"),
                "name": block.get("name"),
                "arguments": block.get("input") or {},
            }
        )
    return calls

def _extract_detail(event: dict[str, Any]) -> dict[str, Any]:
    """Extract lightweight event detail for tooltips."""
    data = dict(event.get("data") or {})
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        content = _raw_response_text(raw_resp)
        if content:
            data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

    data.pop("messages_in", None)
    data.pop("tool_result", None)

    for key in ("args_preview", "result_preview", "tool_args"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."
    return data

def build_gantt_payload(
    data: TraceData,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Build the single-trace payload consumed by the Gantt viewer."""
    meta = data.metadata
    scaffold = meta.get("scaffold", "unknown")
    instance_id = meta.get("instance_id", "")
    trace_id = label or f"{scaffold}/{instance_id}" or str(data.path.stem)

    t0 = _compute_t0(data)
    agents = list(data.agents)
    if not agents:
        agents = ["default"]

    lanes: list[dict[str, Any]] = []
    for agent_id in agents:
        agent_events = [e for e in data.events if e.get("agent_id") == agent_id]
        agent_actions = [a for a in data.actions if a.get("agent_id") == agent_id]

        spans, markers = _build_spans_and_markers(agent_actions, agent_events, t0)

        lanes.append({
            "agent_id": agent_id,
            "spans": spans,
            "markers": markers,
        })

    distinct_iters = {a.get("iteration", 0) for a in data.actions}

    return {
        "id": trace_id,
        "metadata": {
            "scaffold": scaffold,
            "model": meta.get("model"),
            "instance_id": instance_id,
            "mode": meta.get("mode"),
            "max_iterations": meta.get("max_iterations") or meta.get("max_steps"),
            "n_actions": len(data.actions),
            "n_iterations": len(distinct_iters),
            "n_events": len(data.events),
            "elapsed_s": _get_elapsed(data),
        },
        "t0": t0,
        "lanes": lanes,
    }

def build_gantt_payload_multi(
    traces: list[tuple[str, TraceData]],
    *,
    span_registry: dict[str, dict[str, Any]] | None = None,
    marker_registry: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the multi-trace payload, including the active registries."""
    return {
        "registries": {
            "spans": span_registry or DEFAULT_SPAN_REGISTRY,
            "markers": marker_registry or DEFAULT_MARKER_REGISTRY,
        },
        "traces": [
            build_gantt_payload(td, label=lbl) for lbl, td in traces
        ],
    }

def _compute_t0(data: TraceData) -> float:
    t0 = float("inf")
    for ev in data.events:
        ts = ev.get("ts", 0)
        if ts and ts < t0:
            t0 = ts
    for step in data.actions:
        ts = step.get("ts_start", 0)
        if ts and ts < t0:
            t0 = ts
    return t0 if t0 != float("inf") else 0.0

def _get_elapsed(data: TraceData) -> float | None:
    for s in data.summaries:
        if "elapsed_s" in s:
            return s["elapsed_s"]
    return None

def _build_spans_and_markers(
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    t0: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build spans from actions and markers from framework events."""
    spans: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    for act in actions:
        action_type = act.get("action_type")
        if action_type not in ACTION_TYPE_MAP:
            continue
        span_type = ACTION_TYPE_MAP[action_type]
        spans.append({
            "type": span_type,
            "start": act.get("ts_start", 0) - t0,
            "end": act.get("ts_end", 0) - t0,
            "start_abs": act.get("ts_start", 0),
            "end_abs": act.get("ts_end", 0),
            "iteration": act.get("iteration", 0),
            "detail": _extract_detail_from_action(act),
        })

    if spans and events:
        sorted_spans = sorted(spans, key=lambda s: s["start_abs"])
        for i in range(len(sorted_spans) - 1):
            gap_start = sorted_spans[i]["end_abs"]
            gap_end = sorted_spans[i + 1]["start_abs"]
            if gap_end <= gap_start:
                continue
            events_in_gap = [
                e for e in events
                if e.get("category") in _MARKER_CATEGORIES
                and gap_start < (e.get("ts") or 0) < gap_end
            ]
            if not events_in_gap:
                continue
            spans.append({
                "type": "scheduling",
                "start": gap_start - t0,
                "end": gap_end - t0,
                "start_abs": gap_start,
                "end_abs": gap_end,
                "iteration": sorted_spans[i + 1].get("iteration", 0),
                "detail": {
                    "gap_ms": round((gap_end - gap_start) * 1000, 1),
                    "events": [e.get("event", "?") for e in events_in_gap],
                },
            })

    for ev in events:
        category = ev.get("category", "")
        if category in _MARKER_CATEGORIES:
            markers.append({
                "type": category.lower(),
                "event": ev.get("event", "unknown"),
                "t": ev.get("ts", 0) - t0,
                "t_abs": ev.get("ts", 0),
                "iteration": ev.get("iteration", 0),
                "detail": _extract_detail_from_event(ev),
            })

    spans.sort(key=lambda s: s["start"])
    markers.sort(key=lambda m: m["t"])
    return spans, markers

def _summarize_tool_call(tc: dict[str, Any]) -> str | None:
    """Summarize one raw tool call for tooltip display."""
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or {}
    name = fn.get("name") or tc.get("name") or "?"

    raw_args = fn.get("arguments") or tc.get("arguments") or ""
    parsed: dict[str, Any] | None = None
    if isinstance(raw_args, dict):
        parsed = raw_args
    elif isinstance(raw_args, str):
        try:
            candidate = json.loads(raw_args)
            if isinstance(candidate, dict):
                parsed = candidate
        except (json.JSONDecodeError, ValueError):
            parsed = None

    if parsed is not None:
        for field in _TOOL_PRIMARY_FIELDS:
            if field in parsed and parsed[field] is not None:
                value = str(parsed[field])
                if len(value) > _TOOL_ARGS_MAX:
                    value = value[: _TOOL_ARGS_MAX] + "..."
                return f'{name}({field}="{value}")'

    if isinstance(raw_args, (dict, list)):
        args_str = json.dumps(raw_args, ensure_ascii=False)
    else:
        args_str = str(raw_args)
    preview = args_str[:_TOOL_ARGS_MAX]
    if len(args_str) > _TOOL_ARGS_MAX:
        preview += "..."
    return f"{name}({preview})"

def _extract_detail_from_action(act: dict[str, Any]) -> dict[str, Any]:
    """Extract action detail while preserving silent tool-call decisions."""
    data = dict(act.get("data") or {})
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        content = _raw_response_text(raw_resp)
        if content:
            if len(content) > _LLM_CONTENT_MAX:
                data["llm_content"] = content[:_LLM_CONTENT_MAX] + "..."
            else:
                data["llm_content"] = content
        tool_calls = _raw_response_tool_calls(raw_resp)
        if tool_calls:
            summaries = [
                s for s in (_summarize_tool_call(tc) for tc in tool_calls)
                if s is not None
            ]
            if summaries:
                data["tool_calls_requested"] = summaries

    data.pop("messages_in", None)
    return data

def _extract_detail_from_event(ev: dict[str, Any]) -> dict[str, Any]:
    data = dict(ev.get("data") or {})
    data.pop("messages_in", None)

    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        content = _raw_response_text(raw_resp)
        if content:
            data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

    for key in ("args_preview", "result_preview", "tool_args", "tool_result"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."
    return data
