"""Build Gantt chart JSON payloads from TraceData.

Actions (llm_call, tool_exec) ARE the spans directly.
Scheduling overhead is computed from time gaps between consecutive actions.
Events become point markers for observability detail.

The default span / marker / action-type registries live here as module-level
constants. They are shipped *inside* every Gantt payload so the HTML template
renders from data, not from hard-coded JS literals — letting downstream users
register new action_types or event names without touching the template.
"""

from __future__ import annotations

import json
from typing import Any

from trace_collect.trace_inspector import TraceData

# Categories that produce point markers (not spans).
_MARKER_CATEGORIES = frozenset({"SCHEDULING", "SESSION", "CONTEXT"})

# Maximum preview length for LLM narrative content in a tooltip. Large
# enough to show a full paragraph of reasoning in the click-to-pin
# scrollable tooltip without making hover cards absurdly tall.
_LLM_CONTENT_MAX = 1000

# Maximum length for tool-call argument previews in a tooltip.
_TOOL_ARGS_MAX = 200

# When summarizing an LLM tool call, prefer these argument field names
# for a terse ``tool_name(field="value")`` rendering. Order matters —
# the first match wins. These cover the most common filesystem / shell
# / search tool signatures used by openclaw and mini-swe.
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

# Map v4 ``action_type`` -> Gantt span type. Public so downstream code can
# extend it (e.g., adding ``mcp_call`` -> ``mcp``).
ACTION_TYPE_MAP: dict[str, str] = {
    "llm_call": "llm",
    "tool_exec": "tool",
}

# Default span registry shipped inside the payload.
# ``order`` controls vertical stacking when multiple span types share an iteration.
DEFAULT_SPAN_REGISTRY: dict[str, dict[str, Any]] = {
    "llm":        {"color": "#00E5FF", "label": "LLM Call",   "order": 0},
    "tool":       {"color": "#FF6D00", "label": "Tool Exec",  "order": 1},
    "scheduling": {"color": "#76FF03", "label": "Scheduling", "order": 2},
}

# Default marker registry — point-in-time event symbols.
# ``_default`` is the fallback when an event name is not in the map.
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



def _extract_detail(event: dict[str, Any]) -> dict[str, Any]:
    """Extract detail fields from event data for tooltip display.

    Heavy fields (messages_in, raw_response, tool_result) are excluded,
    but a brief ``llm_content`` preview is extracted from raw_response.
    """
    data = dict(event.get("data") or {})

    # Extract LLM content preview from raw_response before dropping it
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        choices = raw_resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            if content:
                data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

    data.pop("messages_in", None)
    data.pop("tool_result", None)

    # Truncate long string fields
    for key in ("args_preview", "result_preview", "tool_args"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."
    return data


def build_gantt_payload(
    data: TraceData,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Build a Gantt chart payload from a single TraceData.

    Args:
        data: Loaded and normalized trace data.
        label: Display label for this trace (defaults to scaffold/instance_id).

    Returns:
        A dict with structure::

            {
                "id": "openclaw/pylint-dev__pylint-7080",
                "metadata": {...},
                "t0": <earliest_timestamp>,
                "lanes": [{
                    "agent_id": "...",
                    "spans": [{"type": "llm", "start": 0.0, "end": 2.77, ...}, ...],
                    "markers": [{"type": "scheduling", "t": 0.0, ...}, ...]
                }]
            }
    """
    meta = data.metadata
    scaffold = meta.get("scaffold", "unknown")
    instance_id = meta.get("instance_id", "")
    trace_id = label or f"{scaffold}/{instance_id}" or str(data.path.stem)

    # Compute t0 from earliest event or step timestamp
    t0 = _compute_t0(data)

    # Group events and steps by agent_id
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
            # v4 vocabulary: 'iterations' is the loop counter; 'actions' are
            # the executable units (multiple actions can share an iteration).
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
    """Build a multi-trace Gantt payload.

    The registries are shipped inside the payload so the HTML template can
    render arbitrary span types and event names without code changes — pass
    custom dicts here to register new visual encodings per call.

    Args:
        traces: List of (label, TraceData) pairs.
        span_registry: Optional override for the default span registry.
        marker_registry: Optional override for the default marker registry.

    Returns:
        ``{"registries": {...}, "traces": [payload, ...]}``
    """
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
    """Find the earliest timestamp across events and steps."""
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
    """Extract elapsed_s from summary if available."""
    for s in data.summaries:
        if "elapsed_s" in s:
            return s["elapsed_s"]
    return None


def _build_spans_and_markers(
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    t0: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Gantt spans from v4 actions and markers from events.

    Each action (llm_call, tool_exec) becomes a span directly.
    Scheduling overhead is computed from time gaps between consecutive actions.
    Events with SCHEDULING/SESSION/CONTEXT categories become point markers.
    """
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

    # ── Compute scheduling spans from inter-action gaps ──
    # A scheduling span is emitted ONLY when the gap window between two
    # consecutive actions actually contains a framework-level event
    # (category in _MARKER_CATEGORIES). The event is the sole trigger —
    # there is intentionally no absolute duration threshold, because any
    # such threshold is platform-dependent noise (asyncio wake-up, HTTP
    # socket reuse, GC pauses). Requiring an event makes every green bar
    # "trusted evidence": each one can be hover-linked to the underlying
    # SCHEDULING / SESSION / CONTEXT event that explains it.
    if spans and events:
        sorted_spans = sorted(spans, key=lambda s: s["start_abs"])
        for i in range(len(sorted_spans) - 1):
            gap_start = sorted_spans[i]["end_abs"]
            gap_end = sorted_spans[i + 1]["start_abs"]
            if gap_end <= gap_start:
                continue  # overlapping / parallel actions → no gap
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

    # ── Build markers from events ──
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
    """Turn a single raw tool_calls[i] dict into a short summary string
    suitable for a tooltip row.

    Preference order:
      1. ``tool_name(primary_field="value")`` — when arguments decode as
         a JSON dict and contain one of ``_TOOL_PRIMARY_FIELDS`` as a
         top-level key. This highlights the "what did the model operate
         on" bit, which is usually more useful than a 200-character
         slice of the raw arguments JSON (which often starts with a
         ``content`` field for ``write_file``).
      2. ``tool_name(<first 200 chars of args>)`` — generic fallback
         when no primary field matches or arguments don't decode.

    Returns ``None`` if ``tc`` doesn't look like a tool-call dict.
    """
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or {}
    name = fn.get("name") or tc.get("name") or "?"

    raw_args = fn.get("arguments") or tc.get("arguments") or ""
    # Arguments can arrive as a dict (already parsed) or as a JSON string
    # (OpenAI's default wire format). Normalize to a dict if possible so
    # we can look up primary fields without touching the raw string.
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

    # Fallback: raw-args preview.
    if isinstance(raw_args, (dict, list)):
        args_str = json.dumps(raw_args, ensure_ascii=False)
    else:
        args_str = str(raw_args)
    preview = args_str[:_TOOL_ARGS_MAX]
    if len(args_str) > _TOOL_ARGS_MAX:
        preview += "..."
    return f"{name}({preview})"


def _extract_detail_from_action(act: dict[str, Any]) -> dict[str, Any]:
    """Extract tooltip detail from a v4 TraceAction record.

    For ``llm_call`` actions we always try to surface *something* about
    the LLM's decision even when the assistant message has no textual
    content: the OpenAI chat-completions protocol lets the model return
    ``content: null`` + ``tool_calls: [...]`` when it calls a tool
    silently. Reporting only ``llm_content`` in that case would make the
    tooltip look empty, even though the model clearly DID something.
    """
    data = dict(act.get("data") or {})

    # Extract from raw_response (both content text AND tool_calls) before
    # dropping the heavy field.
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        choices = raw_resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            if content:
                if len(content) > _LLM_CONTENT_MAX:
                    data["llm_content"] = content[:_LLM_CONTENT_MAX] + "..."
                else:
                    data["llm_content"] = content
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                summaries = [
                    s for s in (_summarize_tool_call(tc) for tc in tool_calls)
                    if s is not None
                ]
                if summaries:
                    data["tool_calls_requested"] = summaries

    # Drop heavy fields
    data.pop("messages_in", None)
    for key in ("tool_result", "tool_args", "args_preview", "result_preview"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."

    return data


def _extract_detail_from_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Extract tooltip detail from an event record."""
    data = dict(ev.get("data") or {})
    data.pop("messages_in", None)

    # Extract LLM content preview before dropping raw_response
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        choices = raw_resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            if content:
                data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

    for key in ("args_preview", "result_preview", "tool_args", "tool_result"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."
    return data
