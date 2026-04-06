#!/usr/bin/env python3
"""Migrate old trace JSONL files to the action/event format.

For each type="step" record, inserts a type="action" record immediately
before it.  Adds ttft_ms/tpot_ms=null to step records that lack them.
Idempotent: skips files that already contain type="action" records.

Usage:
    # Single file (in-place):
    python scripts/migrate_trace_format.py traces/swebench/run.jsonl

    # Directory (recursive, all .jsonl files):
    python scripts/migrate_trace_format.py traces/

    # Dry-run (preview without writing):
    python scripts/migrate_trace_format.py --dry-run traces/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_action_from_step(step: dict) -> dict:
    """Extract an action record from a step record."""
    ts_start = step.get("ts_start", 0.0)
    llm_latency_ms = step.get("llm_latency_ms", 0.0)
    return {
        "type": "action",
        "agent_id": step.get("agent_id", ""),
        "step_idx": step.get("step_idx", 0),
        "program_id": step.get("program_id", ""),
        "tool_name": step.get("tool_name"),
        "tool_args": step.get("tool_args"),
        "prompt_tokens": step.get("prompt_tokens", 0),
        "completion_tokens": step.get("completion_tokens", 0),
        "llm_latency_ms": llm_latency_ms,
        "ttft_ms": step.get("ttft_ms"),
        "ts": ts_start + llm_latency_ms / 1000,
        "extra": {},
    }


def migrate_file(path: Path, *, dry_run: bool = False) -> bool:
    """Migrate a single JSONL file.  Returns True if modified."""
    lines = path.read_text(encoding="utf-8").splitlines()

    # Check idempotency: if any action record exists, skip
    has_action = any(
        _safe_parse(line).get("type") == "action" for line in lines if line.strip()
    )
    if has_action:
        return False

    new_lines: list[str] = []
    modified = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue

        record = _safe_parse(stripped)
        if record is None:
            new_lines.append(line)
            continue

        if record.get("type") == "step":
            # Ensure new fields exist
            if "ttft_ms" not in record:
                record["ttft_ms"] = None
                modified = True
            if "tpot_ms" not in record:
                record["tpot_ms"] = None
                modified = True

            # Insert action record before step
            action = _build_action_from_step(record)
            new_lines.append(json.dumps(action, ensure_ascii=False))
            new_lines.append(json.dumps(record, ensure_ascii=False))
            modified = True
        else:
            new_lines.append(line)

    if modified and not dry_run:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return modified


def _safe_parse(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate trace JSONL to action/event format."
    )
    parser.add_argument("path", help="JSONL file or directory to migrate.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing."
    )
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.rglob("*.jsonl"))
    else:
        print(f"ERROR: {target} is not a file or directory.", file=sys.stderr)
        sys.exit(1)

    migrated = 0
    skipped = 0
    for f in files:
        if migrate_file(f, dry_run=args.dry_run):
            migrated += 1
            prefix = "[dry-run] " if args.dry_run else ""
            print(f"{prefix}Migrated: {f}")
        else:
            skipped += 1

    print(f"\nDone: {migrated} migrated, {skipped} skipped (already current).")


if __name__ == "__main__":
    main()
