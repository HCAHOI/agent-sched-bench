"""Build Gantt chart JSON payloads from TraceData.

Actions (llm_call, tool_exec) ARE the spans directly.
Scheduling overhead is computed from time gaps between consecutive actions.
Events become point markers for observability detail.
"""

from __future__ import annotations

from typing import Any

from trace_collect.trace_inspector import TraceData

# Categories that produce point markers (not spans).
_MARKER_CATEGORIES = frozenset({"SCHEDULING", "SESSION", "CONTEXT"})

# Map action_type to span type for Gantt rendering.
_ACTION_TYPE_MAP: dict[str, str] = {
    "llm_call": "llm",
    "tool_exec": "tool",
}

# Minimum gap (seconds) between actions to render as a scheduling span.
_MIN_SCHEDULING_GAP_S = 0.01  # 10ms


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

    return {
        "id": trace_id,
        "metadata": {
            "scaffold": scaffold,
            "model": meta.get("model"),
            "instance_id": instance_id,
            "mode": meta.get("mode"),
            "max_steps": meta.get("max_steps") or meta.get("max_iterations"),
            "n_steps": len(data.actions),
            "n_events": len(data.events),
            "elapsed_s": _get_elapsed(data),
        },
        "t0": t0,
        "lanes": lanes,
    }


def build_gantt_payload_multi(
    traces: list[tuple[str, TraceData]],
) -> dict[str, Any]:
    """Build a multi-trace Gantt payload.

    Args:
        traces: List of (label, TraceData) pairs.

    Returns:
        ``{"traces": [payload, ...]}``
    """
    return {
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
        if action_type not in _ACTION_TYPE_MAP:
            continue
        span_type = _ACTION_TYPE_MAP[action_type]
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
    if spans:
        sorted_spans = sorted(spans, key=lambda s: s["start_abs"])
        for i in range(len(sorted_spans) - 1):
            gap_start = sorted_spans[i]["end_abs"]
            gap_end = sorted_spans[i + 1]["start_abs"]
            gap = gap_end - gap_start
            if gap > _MIN_SCHEDULING_GAP_S:
                spans.append({
                    "type": "scheduling",
                    "start": gap_start - t0,
                    "end": gap_end - t0,
                    "start_abs": gap_start,
                    "end_abs": gap_end,
                    "iteration": sorted_spans[i + 1].get("iteration", 0),
                    "detail": {"gap_ms": round(gap * 1000, 1)},
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


def _extract_detail_from_action(act: dict[str, Any]) -> dict[str, Any]:
    """Extract tooltip detail from a v4 TraceAction record."""
    data = dict(act.get("data") or {})

    # Extract LLM content preview from raw_response before dropping
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        choices = raw_resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            if content:
                data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

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
