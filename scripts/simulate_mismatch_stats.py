#!/usr/bin/env python3
"""Extract mismatch and forced-sync stats from a simulate trace JSONL.

Usage:
  .venv/bin/python scripts/simulate_mismatch_stats.py <simulate_trace.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <simulate_trace.jsonl>")
    trace_path = Path(sys.argv[1])
    if not trace_path.is_file():
        raise SystemExit(f"not a file: {trace_path}")

    tool_execs = []
    summaries = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec":
            tool_execs.append(record)
        elif record.get("type") == "summary":
            summaries.append(record)

    # --- Overall counts ---
    total = len(tool_execs)
    mismatches = [t for t in tool_execs if not (t.get("data") or {}).get("replay_outcome_match", True)]
    matched = total - len(mismatches)

    # --- Mismatch reasons ---
    reason_counts: Counter[str] = Counter()
    for t in mismatches:
        reason = (t.get("data") or {}).get("mismatch_reason", "unknown")
        reason_counts[reason] += 1

    # --- Forced sync stats ---
    fs_attempted = 0
    fs_success = 0
    fs_fallback = 0
    fs_missing = 0
    fs_errors = Counter()
    for t in tool_execs:
        data = t.get("data") or {}
        if not data.get("forced_sync_attempted"):
            continue
        fs_attempted += 1
        if data.get("forced_sync_success"):
            fs_success += 1
        if data.get("forced_sync_fallback"):
            fs_fallback += 1
        status = data.get("forced_sync_status", "")
        if status == "checkpoint_missing":
            fs_missing += 1
        elif "failed" in status or "error" in str(data.get("forced_sync_error", "")).lower():
            fs_errors[status] += 1

    # --- Checkpoint coverage (from source traces, not simulate output) ---
    # The simulate trace does not copy checkpoint_after from source actions.
    # Read source traces from simulate metadata to get real coverage.
    cp_present = 0
    cp_skipped = 0
    cp_types: Counter[str] = Counter()
    source_trace_paths = _load_source_trace_paths(trace_path)
    if source_trace_paths:
        cp_present, cp_skipped, cp_types = _scan_source_checkpoint_coverage(
            source_trace_paths
        )
    else:
        # Fallback: scan sibling directories
        run_dir = trace_path.parent.parent  # simulate_output/../
        cp_present, cp_skipped, cp_types = _scan_source_checkpoint_coverage(
            _discover_source_traces(run_dir)
        )

    # --- Per-agent summary ---
    agent_stats: dict[str, dict] = {}
    for t in tool_execs:
        agent_id = t.get("agent_id", "unknown")
        if agent_id not in agent_stats:
            agent_stats[agent_id] = {
                "total": 0, "mismatches": 0, "fs_attempted": 0, "fs_success": 0, "fs_fallback": 0
            }
        s = agent_stats[agent_id]
        s["total"] += 1
        data = t.get("data") or {}
        if not data.get("replay_outcome_match", True):
            s["mismatches"] += 1
        if data.get("forced_sync_attempted"):
            s["fs_attempted"] += 1
        if data.get("forced_sync_success"):
            s["fs_success"] += 1
        if data.get("forced_sync_fallback"):
            s["fs_fallback"] += 1

    # --- Print report ---
    print("=" * 60)
    print("SIMULATE MISMATCH & FORCED-SYNC REPORT")
    print("=" * 60)
    print(f"Trace: {trace_path}")
    print(f"Tool exec actions: {total}")
    print(f"  Matched:         {matched} ({100*matched/total:.1f}%)" if total else "")
    print(f"  Mismatched:      {len(mismatches)} ({100*len(mismatches)/total:.1f}%)" if total else "")
    print()

    if reason_counts:
        print("Mismatch reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
        print()

    print("Forced sync:")
    print(f"  Attempted:       {fs_attempted}")
    print(f"  Successful:      {fs_success}")
    print(f"  Fallback (C):    {fs_fallback}")
    print(f"  Checkpoint miss: {fs_missing}")
    if fs_errors:
        print("  Errors:")
        for status, count in fs_errors.most_common():
            print(f"    {status}: {count}")
    print()

    recovery_rate = (100 * fs_success / fs_attempted) if fs_attempted else 0
    fallback_rate = (100 * fs_fallback / fs_success) if fs_success else 0
    print(f"Recovery rate:     {fs_success}/{fs_attempted} = {recovery_rate:.1f}%")
    print(f"Fallback ratio:    {fs_fallback}/{fs_success} = {fallback_rate:.1f}% (of successful)")
    print()

    print("Checkpoint coverage (from source traces):")
    total_checkpointed = cp_present + cp_skipped
    if total_checkpointed:
        print(f"  Checkpoints created: {cp_present}")
        print(f"  Skipped (no Δ):      {cp_skipped}")
        print(f"  Coverage:            {cp_present}/{cp_present + cp_skipped} exec tools checkpointed ({100*cp_present/(cp_present+cp_skipped):.1f}%)")
    else:
        print(f"  (no source traces found)")
    for kind, count in cp_types.most_common():
        print(f"    {kind}: {count}")
    print()

    if len(agent_stats) > 1:
        print("Per-agent breakdown:")
        header = f"  {'Agent':<40} {'Total':>6} {'Mismatch':>9} {'FS_OK':>6} {'FS_FB':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for agent_id, s in sorted(agent_stats.items()):
            short_id = agent_id[:38] + ".." if len(agent_id) > 40 else agent_id
            print(
                f"  {short_id:<40} {s['total']:>6} {s['mismatches']:>9} "
                f"{s['fs_success']:>6} {s['fs_fallback']:>6}"
            )
        print()

    # --- Summary-level stats ---
    total_elapsed = 0.0
    total_unresolved = 0
    for summary in summaries:
        total_elapsed += summary.get("elapsed_s", 0)
        total_unresolved += summary.get("unresolved_mismatches", 0)
    print(f"Aggregate from summaries:")
    print(f"  Total wall time:  {total_elapsed:.1f}s")
    print(f"  Unresolved:       {total_unresolved}")


def _load_source_trace_paths(simulate_trace: Path) -> list[Path]:
    """Extract source trace paths from simulate trace metadata (first JSONL line)."""
    with simulate_trace.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "trace_metadata":
                continue
            entries = record.get("source_trace_entries", [])
            if entries:
                return [Path(e["source_trace"]) for e in entries if "source_trace" in e]
            # Older format: source_traces list of strings
            raw = record.get("source_traces", [])
            return [Path(p) for p in raw if isinstance(p, str)]
    return []


def _discover_source_traces(run_dir: Path) -> list[Path]:
    """Discover trace.jsonl files under run_dir/<instance>/attempt_*/."""
    traces = []
    for instance_dir in sorted(run_dir.iterdir()):
        if not instance_dir.is_dir():
            continue
        attempts = sorted(instance_dir.glob("attempt_*/trace.jsonl"))
        if attempts:
            traces.append(attempts[-1])  # latest attempt
    return traces


def _scan_source_checkpoint_coverage(
    source_paths: list[Path],
) -> tuple[int, int, Counter[str]]:
    """Scan source traces for checkpoint_after fields.

    Returns (created, skipped, kind_counts).
    """
    created = 0
    skipped = 0
    kind_counts: Counter[str] = Counter()
    for src in source_paths:
        if not src.is_file():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "action" or record.get("action_type") != "tool_exec":
                continue
            data = record.get("data") or {}
            cp = data.get("checkpoint_after")
            if isinstance(cp, dict):
                if cp.get("skipped"):
                    skipped += 1
                else:
                    kind = cp.get("kind", "unknown")
                    kind_counts[kind] += 1
                    created += 1
    return created, skipped, kind_counts


if __name__ == "__main__":
    main()
