"""Trace Inspector: parse and query JSONL trace files produced by trace_collect.

Supported record types:
    trace_metadata  – scaffold, mode, model info
    step            – one LLM+tool turn (step_idx, tokens, latency, tool info, messages_in)
    event           – OpenClaw lifecycle events (event, category, data, iteration, ts)
    summary         – end-of-run aggregates
    llm_start/end   – low-level LLM timing markers
    tool_start/end  – low-level tool timing markers
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


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TraceData:
    path: Path
    metadata: dict[str, Any]
    steps: list[dict[str, Any]]  # sorted by step_idx
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
        steps: list[dict[str, Any]] = []
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
                    # Skip malformed lines silently
                    continue

                rec_type = record.get("type", "")
                agent_id = record.get("agent_id")

                # Apply agent filter (metadata records have no agent_id, keep them)
                if agent_filter is not None and agent_id is not None:
                    if agent_filter not in agent_id:
                        continue

                if agent_id is not None:
                    seen_agents[agent_id] = None

                if rec_type == "trace_metadata":
                    metadata.update(record)
                elif rec_type == "step":
                    steps.append(record)
                elif rec_type == "event":
                    events.append(record)
                elif rec_type == "summary":
                    summaries.append(record)
                # llm_start/end, tool_start/end — not stored separately

        steps.sort(key=lambda r: r.get("step_idx", 0))
        events.sort(key=lambda r: r.get("ts", 0.0))

        return cls(
            path=path,
            metadata=metadata,
            steps=steps,
            events=events,
            summaries=summaries,
            agents=list(seen_agents.keys()),
        )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_overview(data: TraceData, as_json: bool = False) -> None:
    """Print a high-level summary of the trace."""
    total_prompt = sum(s.get("prompt_tokens", 0) for s in data.steps)
    total_completion = sum(s.get("completion_tokens", 0) for s in data.steps)
    total_tokens = total_prompt + total_completion
    total_llm_ms = sum(s.get("llm_latency_ms", 0.0) for s in data.steps)
    total_tool_ms = sum(s.get("tool_duration_ms", 0.0) for s in data.steps)

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
    for step in data.steps:
        tool = step.get("tool_name")
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    info: dict[str, Any] = {
        "path": str(data.path),
        "agents": data.agents,
        "scaffold": data.metadata.get("scaffold"),
        "mode": data.metadata.get("mode"),
        "model": data.metadata.get("model"),
        "n_steps": len(data.steps),
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
    step = next((s for s in data.steps if s.get("step_idx") == step_idx), None)
    if step is None:
        print(
            f"ERROR: step {step_idx} not found (available: {[s.get('step_idx') for s in data.steps]})"
        )
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
    step = next((s for s in data.steps if s.get("step_idx") == step_idx), None)
    if step is None:
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
    step = next((s for s in data.steps if s.get("step_idx") == step_idx), None)
    if step is None:
        print(f"ERROR: step {step_idx} not found")
        return

    raw = step.get("raw_response")
    if raw is None:
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
        events = [e for e in events if e.get("category") == category]
    if iteration is not None:
        events = [e for e in events if e.get("iteration") == iteration]

    if as_json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
        return

    if not events:
        print("No events found.")
        return

    print(f"--- Events ({len(events)} total) ---")
    for ev in events:
        ts = ev.get("ts", "?")
        name = ev.get("event", "?")
        cat = ev.get("category", "?")
        itr = ev.get("iteration", "?")
        data_fields = ev.get("data", {})
        # Render key data fields compactly
        data_str = ""
        if isinstance(data_fields, dict) and data_fields:
            data_str = " | " + ", ".join(
                f"{k}={v}" for k, v in list(data_fields.items())[:5]
            )
        print(f"  ts={ts:<12} event={name:<20} cat={cat:<8} iter={itr}{data_str}")


def cmd_tools(
    data: TraceData,
    *,
    step_idx: int | None = None,
    as_json: bool = False,
) -> None:
    """Aggregate tool usage statistics, sorted by count descending."""
    steps = data.steps
    if step_idx is not None:
        steps = [s for s in steps if s.get("step_idx") == step_idx]

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
        print("ERROR: search pattern is required.")
        return

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        print(f"ERROR: invalid regex pattern: {exc}")
        return

    results = []
    for step in data.steps:
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
