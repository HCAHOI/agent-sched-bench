"""Trace Inspector: parse and query JSONL trace files produced by trace_collect.

Supported record types:
    trace_metadata  – scaffold, mode, model info
    step            – one LLM+tool turn (step_idx, tokens, latency, tool info, messages_in)
    event           – unified lifecycle events (event, category, data, step_idx, ts)
    summary         – end-of-run aggregates

Legacy mini-swe flat events (llm_start/end, tool_start/end, action) and
openclaw events using 'iteration' are normalized to the unified envelope
on load via _normalize_legacy_event().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    """Truncate text to `limit` chars. limit<=0 means no truncation."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text) - limit} chars truncated)"


def _to_str(value: Any) -> str:
    """Convert any value to a string for display/truncation."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# Map flat event types to (category, event_name) for normalization.
# Covers mini-swe v2 flat names and v3 flat names (llm_call_start etc.).
_LEGACY_EVENT_MAP: dict[str, tuple[str, str]] = {
    # mini-swe v2 flat events
    "llm_start": ("LLM", "llm_call_start"),
    "llm_end": ("LLM", "llm_call_end"),
    "tool_start": ("TOOL", "tool_exec_start"),
    "tool_end": ("TOOL", "tool_exec_end"),
    "action": ("LLM", "llm_action"),
    # v3 flat events (mini-swe uses these as top-level record types)
    "llm_call_start": ("LLM", "llm_call_start"),
    "llm_call_end": ("LLM", "llm_call_end"),
    "tool_exec_start": ("TOOL", "tool_exec_start"),
    "tool_exec_end": ("TOOL", "tool_exec_end"),
}

# Keys that are structural (not event-specific payload).
_STRUCTURAL_KEYS = frozenset({"type", "agent_id", "ts", "step_idx", "iteration"})


def _normalize_legacy_event(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy mini-swe flat event into the unified envelope.

    Legacy format:  {"type": "llm_start", "agent_id": "...", "step_idx": 0, "ts": ..., ...}
    Normalized:     {"type": "event", "category": "LLM", "event": "llm_call_start",
                     "step_idx": 0, "ts": ..., "data": {...}, "agent_id": "..."}
    """
    rec_type = record.get("type", "")
    category, event_name = _LEGACY_EVENT_MAP[rec_type]

    # Collect payload fields (everything not structural).
    data = {k: v for k, v in record.items() if k not in _STRUCTURAL_KEYS}

    return {
        "type": "event",
        "agent_id": record.get("agent_id"),
        "category": category,
        "event": event_name,
        "step_idx": record.get("step_idx", record.get("iteration", 0)),
        "ts": record.get("ts", 0.0),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TraceData:
    path: Path
    metadata: dict[str, Any]
    actions: list[dict[str, Any]]  # sorted by iteration, then ts_start
    events: list[dict[str, Any]]  # sorted by ts
    summaries: list[dict[str, Any]]
    agents: list[str]  # unique agent_ids in order seen

    @classmethod
    def load(cls, path: Path, agent_filter: str | None = None) -> "TraceData":
        """Parse a JSONL trace file into a TraceData object.

        Args:
            path: Path to the JSONL trace file.
            agent_filter: If provided, keep only records whose agent_id contains
                          this substring. trace_metadata records (no agent_id)
                          are always kept.
        """
        metadata: dict[str, Any] = {}
        actions: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        seen_agents: dict[str, None] = {}  # ordered-set via insertion-order dict

        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec_type = record.get("type", "")
                agent_id = record.get("agent_id")

                if agent_filter is not None and agent_id is not None:
                    if agent_filter not in agent_id:
                        continue

                if agent_id is not None:
                    seen_agents[agent_id] = None

                if rec_type == "trace_metadata":
                    metadata.update(record)
                elif rec_type == "action" and "action_type" in record:
                    # v4 TraceAction records (have action_type field)
                    actions.append(record)
                elif rec_type == "step":
                    # Legacy v3 step records — treat as actions for compat
                    actions.append(record)
                elif rec_type == "event":
                    if "step_idx" not in record and "iteration" in record:
                        record["step_idx"] = record["iteration"]
                    events.append(record)
                elif rec_type in _LEGACY_EVENT_MAP:
                    events.append(_normalize_legacy_event(record))
                elif rec_type == "summary":
                    summaries.append(record)

        # Sort actions: v4 by (iteration, ts_start), legacy steps by step_idx
        actions.sort(key=lambda r: (
            r.get("iteration", r.get("step_idx", 0)),
            r.get("ts_start", 0),
        ))
        events.sort(key=lambda r: r.get("ts", 0.0))

        # Enrich actions with data from events (for v3 slim steps and
        # for v4 actions that may need messages_in/raw_response from events)
        llm_starts: dict[int, dict[str, Any]] = {}
        llm_ends: dict[int, dict[str, Any]] = {}
        for ev in events:
            ename = ev.get("event")
            idx = ev.get("step_idx", 0)
            if ename == "llm_call_start":
                llm_starts[idx] = ev.get("data", {})
            elif ename == "llm_call_end":
                llm_ends[idx] = ev.get("data", {})
        for act in actions:
            idx = act.get("iteration", act.get("step_idx", 0))
            data = act.get("data", {})
            # For v4 actions, messages_in/raw_response are in data
            # For legacy steps, they may be missing (v3 slim) — enrich from events
            msgs = data.get("messages_in") or act.get("messages_in")
            raw = data.get("raw_response") or act.get("raw_response")
            if not msgs and idx in llm_starts:
                msgs = llm_starts[idx].get("messages_in")
            if not raw and idx in llm_ends:
                raw = llm_ends[idx].get("raw_response")
            if msgs:
                act["messages_in"] = msgs
            if raw:
                act["raw_response"] = raw
            # Derive llm_output for search/display
            if "llm_output" not in act:
                llm_content = data.get("llm_content")
                if llm_content:
                    act["llm_output"] = llm_content
                elif raw:
                    choices = raw.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        act["llm_output"] = msg.get("content", "")

        return cls(
            path=path,
            metadata=metadata,
            actions=actions,
            events=events,
            summaries=summaries,
            agents=list(seen_agents.keys()),
        )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_overview(data: TraceData, as_json: bool = False) -> None:
    """Print a high-level summary of the trace."""
    def _aget(act: dict, key: str, default: Any = 0) -> Any:
        """Get a field from action — check data dict first, then top-level."""
        val = (act.get("data") or {}).get(key)
        if val is not None:
            return val
        return act.get(key, default)

    total_prompt = sum(_aget(s, "prompt_tokens", 0) for s in data.actions)
    total_completion = sum(_aget(s, "completion_tokens", 0) for s in data.actions)
    total_tokens = total_prompt + total_completion
    total_llm_ms = sum(_aget(s, "llm_latency_ms", 0) for s in data.actions)
    total_tool_ms = sum(_aget(s, "tool_duration_ms", 0) or _aget(s, "duration_ms", 0) for s in data.actions)

    # Aggregate elapsed from summaries if available
    elapsed_s: float | None = None
    success: bool | None = None
    for summary in data.summaries:
        if "elapsed_s" in summary:
            elapsed_s = summary["elapsed_s"]
        if "success" in summary:
            success = summary["success"]

    # Tool usage counts
    tool_counts: dict[str, int] = {}
    for act in data.actions:
        tool = _aget(act, "tool_name", None)
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    info: dict[str, Any] = {
        "path": str(data.path),
        "agents": data.agents,
        "scaffold": data.metadata.get("scaffold"),
        "mode": data.metadata.get("mode"),
        "model": data.metadata.get("model"),
        "n_steps": len(data.actions),
        "n_events": len(data.events),
        "tool_counts": tool_counts,
        "total_tokens": total_tokens,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_llm_ms": total_llm_ms,
        "total_tool_ms": total_tool_ms,
        "elapsed_s": elapsed_s,
        "success": success,
    }

    if as_json:
        print(json.dumps(info, indent=2))
        return

    print(f"Trace: {data.path}")
    print(f"  Agents    : {', '.join(data.agents) if data.agents else '(none)'}")
    print(f"  Scaffold  : {info['scaffold']}")
    print(f"  Mode      : {info['mode']}")
    print(f"  Model     : {info['model']}")
    print(f"  Steps     : {info['n_steps']}")
    print(f"  Events    : {info['n_events']}")
    if tool_counts:
        counts_str = ", ".join(
            f"{k}={v}" for k, v in sorted(tool_counts.items(), key=lambda x: -x[1])
        )
        print(f"  Tools     : {counts_str}")
    print(
        f"  Tokens    : {total_tokens} (prompt={total_prompt}, completion={total_completion})"
    )
    print(f"  LLM time  : {total_llm_ms:.0f} ms")
    print(f"  Tool time : {total_tool_ms:.0f} ms")
    if elapsed_s is not None:
        print(f"  Elapsed   : {elapsed_s:.1f} s")
    if success is not None:
        print(f"  Success   : {success}")


def cmd_step(
    data: TraceData,
    step_idx: int,
    *,
    truncate: int = 2000,
    as_json: bool = False,
) -> None:
    """Print all fields of a single step record."""
    step = next((s for s in data.actions if s.get("iteration", s.get("step_idx")) == step_idx), None)
    if step is None:
        avail = [s.get("step_idx") for s in data.actions]
        msg = f"step {step_idx} not found (available: {avail})"
        if as_json:
            print(json.dumps({"error": msg}))
        else:
            print(f"ERROR: {msg}")
        return

    if as_json:
        # Truncate large fields before JSON output
        out = dict(step)
        for key in ("tool_args", "tool_result"):
            if key in out:
                out[key] = _truncate(_to_str(out[key]), truncate)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"--- Step {step_idx} ---")
    print(f"  agent_id        : {step.get('agent_id')}")
    print(f"  phase           : {step.get('phase')}")
    print(f"  prompt_tokens   : {step.get('prompt_tokens')}")
    print(f"  completion_tokens: {step.get('completion_tokens')}")
    print(f"  llm_latency_ms  : {step.get('llm_latency_ms')}")
    print(f"  ttft_ms         : {step.get('ttft_ms')}")
    print(f"  tpot_ms         : {step.get('tpot_ms')}")
    print(f"  ts_start        : {step.get('ts_start')}")
    print(f"  ts_end          : {step.get('ts_end')}")
    print(f"  tool_name       : {step.get('tool_name')}")
    print(f"  tool_duration_ms: {step.get('tool_duration_ms')}")
    print(f"  tool_success    : {step.get('tool_success')}")
    print(f"  tool_ts_start   : {step.get('tool_ts_start')}")
    print(f"  tool_ts_end     : {step.get('tool_ts_end')}")
    if "tool_args" in step:
        print(f"  tool_args       : {_truncate(_to_str(step['tool_args']), truncate)}")
    if "tool_result" in step:
        print(
            f"  tool_result     : {_truncate(_to_str(step['tool_result']), truncate)}"
        )
    if "llm_output" in step:
        print(f"  llm_output      : {_truncate(_to_str(step['llm_output']), truncate)}")


def cmd_messages(
    data: TraceData,
    step_idx: int,
    *,
    role_filter: str | None = None,
    truncate: int = 2000,
    as_json: bool = False,
) -> None:
    """Print messages_in from a step record, optionally filtered by role."""
    step = next((s for s in data.actions if s.get("iteration", s.get("step_idx")) == step_idx), None)
    if step is None:
        if as_json:
            print(json.dumps({"error": f"step {step_idx} not found"}))
        else:
            print(f"ERROR: step {step_idx} not found")
        return

    messages: list[dict[str, Any]] = step.get("messages_in", [])
    if role_filter:
        messages = [m for m in messages if m.get("role") == role_filter]

    if as_json:
        out = [
            {
                "role": m.get("role"),
                "content": _truncate(_to_str(m.get("content", "")), truncate),
            }
            for m in messages
        ]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(
        f"--- Messages for step {step_idx}"
        + (f" (role={role_filter})" if role_filter else "")
        + " ---"
    )
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = _truncate(_to_str(msg.get("content", "")), truncate)
        print(f"  [{i}] {role}: {content}")


def cmd_response(
    data: TraceData,
    step_idx: int,
    *,
    truncate: int = 2000,
    as_json: bool = False,
) -> None:
    """Print raw_response from a step record."""
    step = next((s for s in data.actions if s.get("iteration", s.get("step_idx")) == step_idx), None)
    if step is None:
        if as_json:
            print(json.dumps({"error": f"step {step_idx} not found"}))
        else:
            print(f"ERROR: step {step_idx} not found")
        return

    raw = step.get("raw_response")
    if raw is None:
        if as_json:
            print(json.dumps({"error": f"Step {step_idx} has no raw_response field."}))
        else:
            print(f"Step {step_idx} has no raw_response field.")
        return

    text = json.dumps(raw, indent=2, ensure_ascii=False)
    text = _truncate(text, truncate)

    if as_json:
        # Wrap in a container so output is always valid JSON
        print(
            json.dumps(
                {"step_idx": step_idx, "raw_response": raw},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print(f"--- raw_response for step {step_idx} ---")
    print(text)


def cmd_events(
    data: TraceData,
    *,
    category: str | None = None,
    iteration: int | None = None,
    as_json: bool = False,
) -> None:
    """List events, optionally filtered by category and/or iteration."""
    events = data.events

    if category is not None:
        cat_upper = category.upper()
        events = [
            e for e in events if e.get("category", "").upper() == cat_upper
        ]
    if iteration is not None:
        events = [
            e for e in events if e.get("step_idx") == iteration
        ]

    if as_json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
        return

    if not events:
        print("No events found.")
        return

    print(f"--- Events ({len(events)} total) ---")
    for ev in events:
        ts = ev.get("ts") or ev.get("ts_start") or "?"
        name = ev.get("event", "?")
        cat = ev.get("category", "?")
        itr = ev.get("step_idx", "?")
        data_fields = ev.get("data", {})

        data_str = ""
        if isinstance(data_fields, dict) and data_fields:
            items = []
            for k, v in list(data_fields.items())[:5]:
                if k == "tool_args":
                    v = str(v)[:80]
                items.append(f"{k}={v}")
            data_str = " | " + ", ".join(items)
        print(f"  ts={ts:<12} event={name:<20} cat={cat:<8} step={itr}{data_str}")


def cmd_tools(
    data: TraceData,
    *,
    step_idx: int | None = None,
    as_json: bool = False,
) -> None:
    """Aggregate tool usage statistics, sorted by count descending."""
    steps = data.actions
    if step_idx is not None:
        steps = [s for s in steps if s.get("iteration", s.get("step_idx")) == step_idx]

    # name -> {count, total_ms, successes}
    agg: dict[str, dict[str, Any]] = {}
    for step in steps:
        tool = step.get("tool_name")
        if not tool:
            continue
        if tool not in agg:
            agg[tool] = {"count": 0, "total_duration_ms": 0.0, "successes": 0}
        agg[tool]["count"] += 1
        agg[tool]["total_duration_ms"] += step.get("tool_duration_ms", 0.0) or 0.0
        if step.get("tool_success"):
            agg[tool]["successes"] += 1

    rows = []
    for name, stats in agg.items():
        count = stats["count"]
        total_ms = stats["total_duration_ms"]
        successes = stats["successes"]
        success_rate = successes / count if count > 0 else 0.0
        rows.append(
            {
                "tool_name": name,
                "count": count,
                "total_duration_ms": total_ms,
                "success_rate": success_rate,
            }
        )
    rows.sort(key=lambda r: -r["count"])

    if as_json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("No tool calls found.")
        return

    header = f"{'Tool':<20} {'Count':>6} {'Total ms':>10} {'Success%':>10}"
    print(
        f"--- Tool Usage{' (step=' + str(step_idx) + ')' if step_idx is not None else ''} ---"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"  {row['tool_name']:<18} {row['count']:>6} "
            f"{row['total_duration_ms']:>10.0f} {row['success_rate'] * 100:>9.1f}%"
        )


def cmd_search(
    data: TraceData,
    pattern: str,
    *,
    truncate: int = 200,
    as_json: bool = False,
) -> None:
    """Search llm_output fields of all steps using a regex pattern."""
    if not pattern:
        if as_json:
            print(json.dumps({"error": "search pattern is required."}))
        else:
            print("ERROR: search pattern is required.")
        return

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        if as_json:
            print(json.dumps({"error": f"invalid regex pattern: {exc}"}))
        else:
            print(f"ERROR: invalid regex pattern: {exc}")
        return

    results = []
    for step in data.actions:
        llm_output = step.get("llm_output", "") or ""
        if not isinstance(llm_output, str):
            llm_output = json.dumps(llm_output)
        match = regex.search(llm_output)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(llm_output), match.end() + 60)
            context = llm_output[start:end]
            if truncate > 0:
                context = _truncate(context, truncate)
            results.append(
                {
                    "step_idx": step.get("step_idx"),
                    "match_start": match.start(),
                    "context": context,
                }
            )

    if as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        print(f"No matches for pattern: {pattern!r}")
        return

    print(f"--- Search results for {pattern!r} ({len(results)} match(es)) ---")
    for r in results:
        print(f"  step {r['step_idx']}: ...{r['context']}...")


# ---------------------------------------------------------------------------
# Timeline rendering (ported from scripts/trace_timeline.py)
# ---------------------------------------------------------------------------

_TIMELINE_ICONS: dict[str, str] = {
    # CONTEXT
    "skill_load": "📦", "skill_load_failed": "❌📦",
    "skills_summary_build": "📋", "memory_context_load": "🧠",
    "system_prompt_build": "📐", "message_list_build": "📬",
    # MEMORY
    "consolidation_trigger": "🧹", "consolidation_llm_call": "🧠⚡",
    "memory_write": "💾", "history_append": "📝",
    "consolidation_failure": "⚠️🧠", "raw_archive": "📦⚠️",
    "consolidation_complete": "✅🧠",
    "background_consolidation_scheduled": "🔄🧠",
    # MCP
    "mcp_connect_start": "🔌", "mcp_server_connect": "🔗",
    "mcp_server_connected": "✅🔌", "mcp_server_failed": "❌🔌",
    "mcp_tool_register": "📝🔧", "mcp_tool_call": "⚡🔧",
    "mcp_tool_timeout": "⏰🔧", "mcp_disconnect": "🔌❌",
    # SESSION
    "session_create": "🆕", "session_load": "📂",
    "session_save": "💾", "session_turn_save": "💾↩️",
    "checkpoint_set": "📍", "checkpoint_restore": "🔄📍",
    "checkpoint_clear": "🗑️📍",
    # LLM
    "llm_request": "▶️🤖", "llm_response": "◀️🤖",
    "llm_retry": "🔄🤖", "llm_error": "❌🤖",
    "finalization_retry": "🔄📝", "max_iterations": "⏹️",
    # Normalized mini-swe events
    "llm_call_start": "▶️🤖", "llm_call_end": "◀️🤖",
    "llm_action": "▶️🤖",
    "tool_exec_start": "⚙️", "tool_exec_end": "✅",
    # TOOL
    "tool_prepare": "🔧", "tool_prepare_error": "❌🔧",
    "tool_execute": "⚙️", "tool_complete": "✅",
    "tool_error": "❌", "tool_timeout": "⏰", "tool_cancelled": "🚫",
    "external_lookup_blocked": "🚫🔍",
    # Tool-specific
    "file_read": "📖", "file_write": "📝", "file_edit": "✏️",
    "dir_list": "📁", "exec_command": "⚙️💻", "exec_safety_block": "🛡️",
    "web_search": "🔍", "web_fetch": "🌐", "send_message": "📨",
    # SUBAGENT
    "subagent_spawn": "🌱", "subagent_start": "🏃🌱",
    "subagent_tool_execute": "⚙️🌱", "subagent_complete": "✅🌱",
    "subagent_error": "❌🌱", "subagent_cancel": "🚫🌱",
    "subagent_announcement": "📢🌱",
    # SCHEDULING
    "message_dispatch": "📤", "session_lock_acquire": "🔒",
    "session_lock_release": "🔓", "concurrency_gate_acquire": "🚦",
    "task_complete": "🏁", "priority_command_bypass": "⚡",
}

_CATEGORY_SHORT: dict[str, str] = {
    "SCHEDULING": "sched", "SESSION": "session", "CONTEXT": "context",
    "LLM": "llm", "TOOL": "tool", "MCP": "mcp",
    "MEMORY": "memory", "SUBAGENT": "subagent",
}


def _fmt_tl_event(rec: dict[str, Any], t0: float = 0.0) -> str:
    """Format a single event record for timeline display."""
    event_name = rec.get("event", "unknown")
    category = rec.get("category", "")
    data = rec.get("data", {})
    step = rec.get("step_idx", "?")
    ts = rec.get("ts", 0.0)
    rel = ts - t0 if t0 > 0 and ts > 0 else 0.0

    icon = _TIMELINE_ICONS.get(event_name, "  ")
    cat = _CATEGORY_SHORT.get(category, category.lower()[:6])

    parts: list[str] = []
    for key in (
        "skill_name", "source", "server_name", "tool_name", "transport",
        "tools_registered", "session_key", "task_id", "label",
        "command_preview", "path", "query",
        "error_message", "error_type", "request_id",
    ):
        if key in data:
            parts.append(f"{key}={data[key]}")
    for key in (
        "http_status", "wait_ms", "dispatch_duration_ms",
        "history_messages", "total_messages", "memory_size_chars",
        "messages_count", "duration_ms", "consecutive_failures",
        "result_count",
    ):
        if key in data:
            parts.append(f"{key}={data[key]}")
    if "success" in data:
        parts.append("ok" if data["success"] else "FAIL")

    detail = "  ".join(parts)
    return f"  +{rel:7.1f}s {icon} [{cat:>7}] {event_name:<30} step={step:<3} {detail}"


def _fmt_tl_miniswe_event(
    rec: dict[str, Any], t0: float, tool_start_ts: dict[int | str, float]
) -> str | None:
    """Format a normalized mini-swe event for timeline. Returns None to skip."""
    event_name = rec.get("event", "")
    ts = rec.get("ts", 0.0)
    if ts <= 0:
        return None
    rel = ts - t0
    data = rec.get("data", {})
    step = rec.get("step_idx", "?")

    if event_name == "llm_call_start":
        return f"  +{rel:7.1f}s  ▶ LLM           step={step}"
    elif event_name == "llm_call_end":
        lat = (data.get("latency_ms") or 0) / 1000
        pt = data.get("prompt_tokens", 0)
        ct = data.get("completion_tokens", 0)
        return f"  +{rel:7.1f}s  ◀ LLM done      step={step}  lat={lat:.1f}s  {pt}+{ct}tok"
    elif event_name == "tool_exec_start":
        tool_start_ts[step] = ts
        tool_args = data.get("tool_args", "")
        try:
            args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            cmds = args.get("commands", [args.get("command", "")]) if isinstance(args, dict) else [str(tool_args)]
        except Exception:
            cmds = [str(tool_args)]
        first = cmds[0][:52].replace("\n", "↵") if cmds else ""
        return f"  +{rel:7.1f}s  ⚙  bash         step={step}  $ {first}"
    elif event_name == "tool_exec_end":
        start = tool_start_ts.get(step)
        dur = (ts - start) if start is not None else 0.0
        ok = "✓" if data.get("success") else "✗"
        return f"  +{rel:7.1f}s  {ok}  bash done    step={step}  dur={dur:.1f}s"
    return None


def _fmt_tl_step(rec: dict[str, Any], t0: float = 0.0) -> list[str]:
    """Format a step record for timeline. Returns lines to print."""
    lines: list[str] = []
    step_idx = rec.get("step_idx", "?")
    pt = rec.get("prompt_tokens", 0)
    ct = rec.get("completion_tokens", 0)
    llm_lat = rec.get("llm_latency_ms", 0) / 1000
    rel = rec.get("ts_start", 0) - t0 if t0 > 0 and rec.get("ts_start", 0) > 0 else 0.0

    if pt or ct:
        ttft = rec.get("ttft_ms")
        tpot = rec.get("tpot_ms")
        timing_extra = ""
        if ttft is not None and ttft > 0:
            timing_extra = f"  ttft={ttft:.0f}ms"
            if tpot is not None and tpot > 0:
                timing_extra += f" tpot={tpot:.1f}ms"
        lines.append(
            f"  +{rel:7.1f}s step={step_idx:<3}  ◀ LLM  {pt}+{ct}tok  "
            f"lat={llm_lat:.1f}s{timing_extra}"
        )

    tool_name = rec.get("tool_name")
    if tool_name:
        tool_dur = rec.get("tool_duration_ms")
        dur_str = f"  dur={tool_dur / 1000:.1f}s" if tool_dur else ""
        ok = "✓" if rec.get("tool_success") else "✗"
        result_preview = (rec.get("tool_result") or "")[:80].replace("\n", "↵")
        if result_preview:
            result_preview = f"  {result_preview}"
        tool_ts = rec.get("tool_ts_start") or 0
        tool_rel = tool_ts - t0 if t0 > 0 and tool_ts > 0 else rel
        lines.append(
            f"  +{tool_rel:7.1f}s step={step_idx:<3}  {ok}  {tool_name}{dur_str}{result_preview}"
        )
    return lines


def _print_tl_summary(summary: dict[str, Any]) -> None:
    """Print timeline summary footer."""
    llm_s = summary.get("total_llm_ms", 0) / 1000
    tool_s = summary.get("total_tool_ms", 0) / 1000
    elapsed = summary.get("elapsed_s", 0)
    n = summary.get("n_steps", 0)
    tokens = summary.get("total_tokens", 0)
    ok = "✓ success" if summary.get("success") else "✗ failed"
    prepare_ms = summary.get("prepare_ms")
    prepare_str = f"  prepare={prepare_ms:.0f}ms" if prepare_ms else ""

    print("─" * 80)
    print(
        f"  {ok}  {n} steps  "
        f"elapsed={elapsed:.0f}s  LLM={llm_s:.1f}s  tool={tool_s:.1f}s  "
        f"tokens={tokens}{prepare_str}"
    )
    tool_ms = summary.get("tool_ms_by_name", {})
    if tool_ms:
        print("  Tool time breakdown:")
        for name, ms in sorted(tool_ms.items(), key=lambda x: -x[1]):
            if ms > 0:
                print(f"    {name:20s}: {ms / 1000:.1f}s")
    timeouts = summary.get("tool_timeouts", {})
    if timeouts:
        print("  Tool timeouts:")
        for name, count in sorted(timeouts.items()):
            print(f"    {name:20s}: {count}")


def cmd_timeline(data: TraceData) -> None:
    """Print a concise per-step timeline with icons and relative timestamps."""

    scaffold = data.metadata.get("scaffold", "")
    mode = data.metadata.get("mode", "collect")
    model = data.metadata.get("model") or data.metadata.get("local_model", "")

    # Header
    print(f"Trace: {data.path.name}")
    print(f"  Scaffold: {scaffold}  Mode: {mode}")
    if model:
        print(f"  Model: {model}")
    if mode == "simulate":
        src = data.metadata.get("source_model", "?")
        local = data.metadata.get("local_model", "?")
        print(f"  Simulate: {src} → {local}")

    # Detect scaffold from events if metadata is missing.
    # Check mini-swe first (specific normalized names), then openclaw
    # (has openclaw-specific events like tool_execute, session_load).
    _OPENCLAW_ONLY_EVENTS = frozenset({
        "tool_execute", "tool_complete", "tool_error", "session_load",
        "message_dispatch", "session_lock_acquire", "skill_load",
    })
    if not scaffold:
        if any(e.get("event") in ("llm_call_start", "tool_exec_start") for e in data.events):
            scaffold = "mini-swe-agent"
        elif any(e.get("event") in _OPENCLAW_ONLY_EVENTS for e in data.events):
            scaffold = "openclaw"

    for agent_id in data.agents:
        agent_steps = [s for s in data.actions if s.get("agent_id") == agent_id]
        agent_events = [e for e in data.events if e.get("agent_id") == agent_id]
        agent_summaries = [s for s in data.summaries if s.get("agent_id") == agent_id]

        print(f"\nTimeline: {agent_id}")
        print("─" * 80)

        if scaffold == "openclaw":
            _tl_render_openclaw(agent_steps, agent_events)
        elif scaffold == "mini-swe-agent":
            _tl_render_miniswe(agent_steps, agent_events)
        else:
            _tl_render_steps_only(agent_steps, scaffold)

        summary = agent_summaries[0] if agent_summaries else None
        if summary:
            _print_tl_summary(summary)


def _tl_render_openclaw(
    steps: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    """Render openclaw timeline: interleave steps and events sorted by (step_idx, ts)."""
    entries: list[tuple[int, float, str, dict[str, Any]]] = []
    for s in steps:
        entries.append((s.get("step_idx", 0), s.get("ts_start", 0), "step", s))
    for e in events:
        entries.append((e.get("step_idx", -1), e.get("ts", 0), "event", e))
    entries.sort(key=lambda x: (x[0], x[1]))

    t0 = min((ts for _, ts, _, _ in entries if ts > 0), default=0.0)

    for _, _, entry_type, rec in entries:
        if entry_type == "step":
            for line in _fmt_tl_step(rec, t0):
                print(line)
        else:
            print(_fmt_tl_event(rec, t0))


def _tl_render_miniswe(
    steps: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    """Render mini-swe timeline using normalized events."""
    all_recs = steps + events
    all_recs.sort(key=lambda r: r.get("ts") or r.get("ts_start") or 0)

    t0: float | None = None
    tool_start_ts: dict[int | str, float] = {}

    for r in all_recs:
        ts = r.get("ts") or r.get("ts_start")
        if ts is None or ts <= 0:
            continue
        if t0 is None:
            t0 = ts

        rec_type = r.get("type", "")
        if rec_type == "event":
            line = _fmt_tl_miniswe_event(r, t0, tool_start_ts)
            if line:
                print(line)
        elif rec_type == "step":
            # Step records don't duplicate with events in mini-swe
            pass


def _tl_render_steps_only(
    steps: list[dict[str, Any]], scaffold: str
) -> None:
    """Fallback: render only step records."""
    steps_sorted = sorted(steps, key=lambda r: r.get("step_idx", 0))
    for step in steps_sorted:
        for line in _fmt_tl_step(step, 0.0):
            print(line)
