#!/usr/bin/env python3
"""Print a concise per-step timeline from a JSONL trace file.

Usage:
    python scripts/trace_timeline.py traces/swebench/<run>.jsonl [agent_id]

If agent_id is omitted, prints all agents found in the file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def timeline(path: Path, filter_agent: str | None = None) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # Check for replay metadata and print header
    for r in records:
        if r.get("type") == "replay_metadata":
            print(
                f"[REPLAY] from_step={r.get('from_step')} "
                f"model={r.get('model')} max_steps={r.get('max_steps')} "
                f"original={r.get('original_trace')}"
            )
            print()
            break

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
        recs = [r for r in records if r.get("agent_id") == agent]
        _print_agent_timeline(agent, recs)
        if agent != agents[-1]:
            print()


def _print_agent_timeline(agent: str, recs: list[dict]) -> None:
    t0: float | None = None
    # Track tool_start absolute ts per step to compute real wall-clock dur.
    tool_start_ts: dict[int | str, float] = {}

    print(f"Timeline: {agent}")
    print("─" * 72)

    for r in recs:
        rtype = r.get("type", "")
        ts = r.get("ts") or r.get("ts_start")
        if ts is None:
            continue
        if t0 is None:
            t0 = ts
        rel = ts - t0

        if rtype == "llm_start":
            idx = r.get("step_idx", "?")
            print(f"  +{rel:7.1f}s  ▶ LLM           step={idx}")

        elif rtype == "llm_end":
            idx = r.get("step_idx", "?")
            lat = (r.get("latency_ms") or 0) / 1000
            pt = r.get("prompt_tokens", 0)
            ct = r.get("completion_tokens", 0)
            print(f"  +{rel:7.1f}s  ◀ LLM done      step={idx}  lat={lat:.1f}s  {pt}+{ct}tok")

        elif rtype == "tool_start":
            idx = r.get("step_idx", "?")
            tool_start_ts[idx] = ts
            try:
                args = json.loads(r.get("tool_args") or "{}")
                if "commands" in args:
                    cmds = args["commands"]
                else:
                    cmds = [args.get("command", "")]
            except Exception:
                cmds = [r.get("tool_args", "")]
            first = cmds[0][:52].replace("\n", "↵")
            print(f"  +{rel:7.1f}s  ⚙  bash         step={idx}  $ {first}")
            for extra_cmd in cmds[1:]:
                print(f"  {'':10}             {'':9}  $ {extra_cmd[:52].replace(chr(10), '↵')}")

        elif rtype == "tool_end":
            idx = r.get("step_idx", "?")
            # Compute duration from timestamps for accuracy; duration_ms in the
            # event is None when mini-swe-agent doesn't record tool_ts_end.
            start_ts = tool_start_ts.get(idx)
            dur = (ts - start_ts) if start_ts is not None else 0.0
            ok = "✓" if r.get("success") else "✗"
            print(f"  +{rel:7.1f}s  {ok}  bash done    step={idx}  dur={dur:.1f}s")

    summary = next((r for r in recs if r.get("type") == "summary"), None)
    if summary:
        llm_s = summary.get("total_llm_ms", 0) / 1000
        tool_s = summary.get("total_tool_ms", 0) / 1000
        elapsed = summary.get("elapsed_s", 0)
        n = summary.get("n_steps", 0)
        ok = "✓ success" if summary.get("success") else "✗ failed"
        print("─" * 72)
        print(
            f"  {ok}  {n} steps  "
            f"elapsed={elapsed:.0f}s  LLM={llm_s:.1f}s  tool={tool_s:.1f}s  "
            f"tokens={summary.get('total_tokens', 0)}"
        )


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
