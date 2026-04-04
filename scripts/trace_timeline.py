#!/usr/bin/env python3
"""Print a concise per-step timeline from a JSONL trace file.

Supports multiple agent scaffolds (mini-swe-agent, OpenClaw/nanobot, simulate)
by reading the ``trace_metadata`` header record to dispatch to the correct
renderer.  Falls back to heuristic detection for legacy traces.

Usage:
    python scripts/trace_timeline.py traces/<run>.jsonl [agent_id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ── Scaffold detection ────────────────────────────────────────────────

def detect_scaffold(records: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """Return ``(scaffold, mode, metadata_record)`` from trace records.

    Checks for a ``trace_metadata`` header first, then falls back to
    heuristic detection for legacy traces.
    """
    for r in records:
        if r.get("type") == "trace_metadata":
            return (
                r.get("scaffold", "unknown"),
                r.get("mode", "collect"),
                r,
            )
        # Legacy: old replay_metadata / simulate_metadata formats
        if r.get("type") == "replay_metadata":
            return ("unknown", "replay", r)
        if r.get("type") == "simulate_metadata":
            return ("unknown", "simulate", r)

    # Heuristic: if any record has category field → OpenClaw events
    for r in records:
        if r.get("type") == "event" and r.get("category"):
            return ("openclaw", "collect", {})

    # Heuristic: if events use llm_start/tool_start → mini-swe-agent
    for r in records:
        if r.get("type") in ("llm_start", "tool_start"):
            return ("mini-swe-agent", "collect", {})

    return ("unknown", "collect", {})


# ── OpenClaw event rendering ─────────────────────────────────────────

_OPENCLAW_ICONS: dict[str, str] = {
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
    # TOOL
    "tool_prepare": "🔧", "tool_prepare_error": "❌🔧",
    "tool_execute": "⚙️", "tool_complete": "✅",
    "tool_error": "❌", "tool_timeout": "⏰",
    "tool_cancelled": "🚫", "external_lookup_blocked": "🚫🔍",
    # Tool-specific
    "file_read": "📖", "file_write": "📝", "file_edit": "✏️",
    "dir_list": "📁", "exec_command": "⚙️💻",
    "exec_safety_block": "🛡️",
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


def _format_openclaw_event(rec: dict[str, Any]) -> str:
    """Format an OpenClaw event record as a single line."""
    event_name = rec.get("event", "unknown")
    category = rec.get("category", "")
    data = rec.get("data", {})
    iteration = rec.get("iteration", "?")

    icon = _OPENCLAW_ICONS.get(event_name, "  ")
    cat = _CATEGORY_SHORT.get(category, category.lower()[:6])

    parts: list[str] = []
    for key in ("skill_name", "source", "server_name", "tool_name",
                "transport", "tools_registered", "session_key", "task_id",
                "label", "command_preview", "path", "query"):
        if key in data:
            parts.append(f"{key}={data[key]}")
    for key in ("memory_size_chars", "messages_count", "duration_ms",
                "consecutive_failures", "result_count"):
        if key in data:
            parts.append(f"{key}={data[key]}")
    if "success" in data:
        parts.append("ok" if data["success"] else "FAIL")

    detail = "  ".join(parts)
    return f"  {icon} [{cat:>7}] {event_name:<30} iter={iteration:<3} {detail}"


# ── Mini-swe-agent event rendering ───────────────────────────────────

def _format_miniswe_event(rec: dict[str, Any], t0: float,
                          tool_start_ts: dict[int | str, float]) -> str | None:
    """Format a mini-swe-agent event. Returns None to skip."""
    rtype = rec.get("type", "")
    ts = rec.get("ts") or rec.get("ts_start")
    if ts is None:
        return None
    rel = ts - t0

    if rtype == "llm_start":
        idx = rec.get("step_idx", "?")
        return f"  +{rel:7.1f}s  ▶ LLM           step={idx}"
    elif rtype == "llm_end":
        idx = rec.get("step_idx", "?")
        lat = (rec.get("latency_ms") or 0) / 1000
        pt = rec.get("prompt_tokens", 0)
        ct = rec.get("completion_tokens", 0)
        return f"  +{rel:7.1f}s  ◀ LLM done      step={idx}  lat={lat:.1f}s  {pt}+{ct}tok"
    elif rtype == "tool_start":
        idx = rec.get("step_idx", "?")
        tool_start_ts[idx] = ts
        try:
            args = json.loads(rec.get("tool_args") or "{}")
            cmds = args.get("commands", [args.get("command", "")])
        except Exception:
            cmds = [rec.get("tool_args", "")]
        first = cmds[0][:52].replace("\n", "↵") if cmds else ""
        return f"  +{rel:7.1f}s  ⚙  bash         step={idx}  $ {first}"
    elif rtype == "tool_end":
        idx = rec.get("step_idx", "?")
        start = tool_start_ts.get(idx)
        dur = (ts - start) if start is not None else 0.0
        ok = "✓" if rec.get("success") else "✗"
        return f"  +{rel:7.1f}s  {ok}  bash done    step={idx}  dur={dur:.1f}s"
    return None


# ── Step rendering (shared) ──────────────────────────────────────────

def _format_step(rec: dict[str, Any], scaffold: str) -> list[str]:
    """Format a step record. Returns lines to print."""
    lines: list[str] = []
    step_idx = rec.get("step_idx", "?")
    pt = rec.get("prompt_tokens", 0)
    ct = rec.get("completion_tokens", 0)
    llm_lat = rec.get("llm_latency_ms", 0) / 1000

    replay_tag = ""
    if rec.get("from_replay") is True:
        replay_tag = " [REPLAY]"

    # LLM info
    if pt or ct:
        ttft = rec.get("ttft_ms")
        tpot = rec.get("tpot_ms")
        timing_extra = ""
        if ttft is not None and ttft > 0:
            timing_extra = f"  ttft={ttft:.0f}ms"
            if tpot is not None and tpot > 0:
                timing_extra += f" tpot={tpot:.1f}ms"
        lines.append(
            f"  step={step_idx:<3}  ◀ LLM  {pt}+{ct}tok  "
            f"lat={llm_lat:.1f}s{timing_extra}{replay_tag}"
        )

    # Tool info
    tool_name = rec.get("tool_name")
    if tool_name:
        tool_dur = rec.get("tool_duration_ms")
        dur_str = f"  dur={tool_dur/1000:.1f}s" if tool_dur else ""
        ok = "✓" if rec.get("tool_success") else "✗"
        result_preview = (rec.get("tool_result") or "")[:80].replace("\n", "↵")
        if result_preview:
            result_preview = f"  {result_preview}"
        lines.append(f"  step={step_idx:<3}  {ok}  {tool_name}{dur_str}{result_preview}")

    return lines


# ── Summary rendering ────────────────────────────────────────────────

def _print_summary(summary: dict[str, Any]) -> None:
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

    # Tool time breakdown (OpenClaw traces include this)
    tool_ms = summary.get("tool_ms_by_name", {})
    if tool_ms:
        print("  Tool time breakdown:")
        for name, ms in sorted(tool_ms.items(), key=lambda x: -x[1]):
            if ms > 0:
                print(f"    {name:20s}: {ms/1000:.1f}s")

    timeouts = summary.get("tool_timeouts", {})
    if timeouts:
        print("  Tool timeouts:")
        for name, count in sorted(timeouts.items()):
            print(f"    {name:20s}: {count}")


# ── Main timeline dispatcher ─────────────────────────────────────────

def timeline(path: Path, filter_agent: str | None = None) -> None:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not records:
        print(f"Empty trace file: {path}", file=sys.stderr)
        return

    scaffold, mode, metadata = detect_scaffold(records)

    # Print header
    print(f"Trace: {path.name}")
    print(f"  Scaffold: {scaffold}  Mode: {mode}")
    if metadata.get("model") or metadata.get("local_model"):
        model_str = metadata.get("model") or metadata.get("local_model", "")
        print(f"  Model: {model_str}")
    if mode == "replay":
        fs = metadata.get("from_step", "?")
        orig = metadata.get("original_steps", "?")
        print(f"  Replay from step {fs} (original: {orig} steps)")
    if mode == "simulate":
        src_model = metadata.get("source_model", "?")
        local_model = metadata.get("local_model", "?")
        print(f"  Simulate: {src_model} → {local_model}")

    # Filter by agent_id if requested
    agents: list[str] = []
    seen: set[str] = set()
    for r in records:
        aid = r.get("agent_id", "")
        if aid and aid not in seen:
            agents.append(aid)
            seen.add(aid)
    if filter_agent:
        agents = [a for a in agents if filter_agent in a]

    for agent in agents:
        recs = [r for r in records if r.get("agent_id") == agent or
                r.get("type") == "trace_metadata"]
        _render_agent(agent, recs, scaffold, mode)
        if agent != agents[-1]:
            print()


def _render_agent(agent: str, recs: list[dict[str, Any]],
                  scaffold: str, mode: str) -> None:
    print(f"\nTimeline: {agent}")
    print("─" * 80)

    if scaffold == "openclaw" or _has_openclaw_events(recs):
        _render_openclaw_timeline(recs)
    elif scaffold == "mini-swe-agent" or _has_miniswe_events(recs):
        _render_miniswe_timeline(recs, mode)
    else:
        # Generic: render step records only
        _render_step_only_timeline(recs)

    # Summary
    summary = next((r for r in recs if r.get("type") == "summary"), None)
    if summary:
        _print_summary(summary)


def _has_openclaw_events(recs: list[dict[str, Any]]) -> bool:
    return any(r.get("type") == "event" and r.get("category") for r in recs)


def _has_miniswe_events(recs: list[dict[str, Any]]) -> bool:
    return any(r.get("type") in ("llm_start", "tool_start") for r in recs)


def _render_openclaw_timeline(recs: list[dict[str, Any]]) -> None:
    """Render OpenClaw traces: interleave steps and events sorted by (iteration, ts)."""
    entries: list[tuple[int, float, str, dict[str, Any]]] = []
    for r in recs:
        rtype = r.get("type", "")
        if rtype == "step":
            entries.append((r.get("step_idx", 0), r.get("ts_start", 0), "step", r))
        elif rtype == "event":
            entries.append((r.get("iteration", -1), r.get("ts", 0), "event", r))
    entries.sort(key=lambda x: (x[0], x[1]))

    for _, _, entry_type, rec in entries:
        if entry_type == "step":
            for line in _format_step(rec, "openclaw"):
                print(line)
        else:
            print(_format_openclaw_event(rec))


def _render_miniswe_timeline(recs: list[dict[str, Any]], mode: str) -> None:
    """Render mini-swe-agent traces with llm_start/end, tool_start/end events."""
    t0: float | None = None
    tool_start_ts: dict[int | str, float] = {}
    _in_replay = False
    _replay_steps = {r.get("step_idx", -1) for r in recs
                     if r.get("from_replay") is True and r.get("type") == "step"}

    for r in recs:
        rtype = r.get("type", "")
        if rtype in ("trace_metadata", "replay_metadata", "summary"):
            continue
        ts = r.get("ts") or r.get("ts_start")
        if ts is None:
            continue

        step_idx = r.get("step_idx")
        if not _in_replay and step_idx in _replay_steps:
            _in_replay = True
            t0 = ts
            print("─" * 80)
            print("  *** REPLAY STARTS ***")
            print("─" * 80)
        if t0 is None:
            t0 = ts

        line = _format_miniswe_event(r, t0, tool_start_ts)
        if line:
            print(line)


def _render_step_only_timeline(recs: list[dict[str, Any]]) -> None:
    """Fallback: render only step records (for simulate or unknown scaffolds)."""
    steps = sorted(
        (r for r in recs if r.get("type") == "step"),
        key=lambda r: r.get("step_idx", 0),
    )
    scaffold = "unknown"
    # Check metadata for scaffold
    for r in recs:
        if r.get("type") == "trace_metadata":
            scaffold = r.get("scaffold", "unknown")
            break

    for step in steps:
        for line in _format_step(step, scaffold):
            print(line)


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    filter_agent = sys.argv[2] if len(sys.argv) > 2 else None
    timeline(path, filter_agent)


if __name__ == "__main__":
    main()
