"""Build Gantt chart JSON payloads from TraceData.

Pairs start/end events into spans, converts unpaired events to markers,
and produces a lightweight JSON structure for the Canvas renderer.

Events are the primary data source (precise real-time timestamps).
Step records are fallback when events are missing for a given iteration.
"""

from __future__ import annotations

from typing import Any

from trace_collect.trace_inspector import TraceData

# Categories that produce point markers (not spans).
_MARKER_CATEGORIES = frozenset({"SCHEDULING", "SESSION", "CONTEXT"})

# Events that form span pairs: (start_event, end_event) -> span_type.
_SPAN_PAIRS: dict[str, tuple[str, str]] = {
    "llm": ("llm_call_start", "llm_call_end"),
    "tool": ("tool_exec_start", "tool_exec_end"),
}

# Reverse lookup: event_name -> (span_type, "start"|"end").
_EVENT_ROLE: dict[str, tuple[str, str]] = {}
for _stype, (_s, _e) in _SPAN_PAIRS.items():
    _EVENT_ROLE[_s] = (_stype, "start")
    _EVENT_ROLE[_e] = (_stype, "end")


def _pair_key(event: dict[str, Any], span_type: str) -> tuple:
    """Build the pairing key for an event.

    LLM events pair by step_idx alone (one LLM call per step).
    Tool events pair by (step_idx, tool_name) to handle multi-tool steps.
    """
    step_idx = event.get("step_idx", 0)
    if span_type == "tool":
        tool_name = (event.get("data") or {}).get("tool_name", "")
        return (step_idx, tool_name)
    return (step_idx,)


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
        agent_steps = [s for s in data.steps if s.get("agent_id") == agent_id]

        spans, markers = _build_spans_and_markers(agent_events, agent_steps, t0)

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
            "n_steps": len(data.steps),
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
    for step in data.steps:
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
    events: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    t0: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair events into spans, convert remainder to markers.

    Falls back to step records for iterations without matching events.
    """
    spans: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    # Collect start events by (span_type, pair_key)
    starts: dict[tuple, dict[str, Any]] = {}
    matched_indices: set[int] = set()
    # Track already-paired keys to deduplicate (e.g., mini-swe v3 has both
    # llm_start->normalized and llm_call_start flat, producing duplicate pairs)
    paired_keys: set[tuple] = set()

    # Index events
    for i, ev in enumerate(events):
        event_name = ev.get("event", "")
        category = ev.get("category", "")

        if event_name in _EVENT_ROLE:
            span_type, role = _EVENT_ROLE[event_name]
            key = (span_type, *_pair_key(ev, span_type))

            if role == "start":
                if key not in paired_keys:
                    starts[key] = ev
                    starts[key]["_idx"] = i
                else:
                    # Duplicate start for already-paired key — skip
                    matched_indices.add(i)
            elif role == "end":
                start_ev = starts.pop(key, None)
                if start_ev is not None:
                    # Matched pair → span
                    paired_keys.add(key)
                    matched_indices.add(start_ev["_idx"])
                    matched_indices.add(i)
                    detail = _extract_detail(start_ev)
                    detail.update(_extract_detail(ev))
                    spans.append({
                        "type": span_type,
                        "start": start_ev.get("ts", 0) - t0,
                        "end": ev.get("ts", 0) - t0,
                        "start_abs": start_ev.get("ts", 0),
                        "end_abs": ev.get("ts", 0),
                        "step_idx": ev.get("step_idx", 0),
                        "detail": detail,
                    })
                else:
                    if key in paired_keys:
                        # Duplicate end for already-paired key — skip silently
                        matched_indices.add(i)
                        # But enrich the existing span with any richer data
                        _enrich_span_from_duplicate(spans, ev, span_type)
                    else:
                        # Truly unmatched end → marker
                        matched_indices.add(i)
                        markers.append(_make_marker(ev, t0))

        elif category in _MARKER_CATEGORIES:
            matched_indices.add(i)
            markers.append(_make_marker(ev, t0))

    # Unmatched start events → markers
    for key, start_ev in starts.items():
        markers.append(_make_marker(start_ev, t0))

    # Fallback: steps without any matching events get a coarse span
    step_indices_with_spans = {s["step_idx"] for s in spans}
    for step in steps:
        idx = step.get("step_idx", 0)
        if idx not in step_indices_with_spans:
            ts_start = step.get("ts_start", 0)
            ts_end = step.get("ts_end", 0)
            if ts_start and ts_end and ts_end > ts_start:
                spans.append({
                    "type": "llm",
                    "start": ts_start - t0,
                    "end": ts_end - t0,
                    "start_abs": ts_start,
                    "end_abs": ts_end,
                    "step_idx": idx,
                    "detail": {
                        "fallback": True,
                        "tool_name": step.get("tool_name"),
                        "prompt_tokens": step.get("prompt_tokens"),
                        "completion_tokens": step.get("completion_tokens"),
                        "llm_latency_ms": step.get("llm_latency_ms"),
                    },
                })

    # Sort spans by start time
    spans.sort(key=lambda s: s["start"])
    markers.sort(key=lambda m: m["t"])

    return spans, markers


def _make_marker(ev: dict[str, Any], t0: float) -> dict[str, Any]:
    """Convert a single event into a point marker."""
    return {
        "type": ev.get("category", "SCHEDULING").lower(),
        "event": ev.get("event", "unknown"),
        "t": ev.get("ts", 0) - t0,
        "t_abs": ev.get("ts", 0),
        "step_idx": ev.get("step_idx", 0),
        "detail": _extract_detail(ev),
    }
