#!/usr/bin/env python3
"""Ad-hoc smoke helper for checkpoint_after + forced-sync validation.

This is intentionally not part of the benchmark pipeline. It copies a collected
trace, flips one checkpointed exec action's source outcome to force a replay
mismatch, and writes a one-trace simulate manifest.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: scripts/smoke_checkpoint_forced_sync.py "
            "<attempt_dir> <output_dir>"
        )
    attempt_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    trace_path = attempt_dir / "trace.jsonl"
    if not trace_path.is_file():
        raise SystemExit(f"missing trace: {trace_path}")

    records = _read_jsonl(trace_path)
    checkpointed = [
        record
        for record in records
        if record.get("type") == "action"
        and record.get("action_type") == "tool_exec"
        and isinstance(record.get("data"), dict)
        and isinstance(record["data"].get("checkpoint_after"), dict)
    ]
    print(f"checkpointed_tool_actions={len(checkpointed)}")
    for record in checkpointed:
        data = record["data"]
        cp = data["checkpoint_after"]
        cp_path = Path(str(cp["path"]))
        if not cp_path.is_absolute():
            cp_path = trace_path.parent / cp_path
        print(
            "checkpoint",
            f"action={record.get('action_id')}",
            f"tool={data.get('tool_name')}",
            f"path={cp_path}",
            f"exists={cp_path.is_file()}",
            f"size={cp_path.stat().st_size if cp_path.is_file() else 'missing'}",
        )
    if not checkpointed:
        raise SystemExit("no checkpoint_after action found")

    target = checkpointed[0]
    target_data = target["data"]
    target_data["success"] = not bool(target_data.get("success", True))
    target_data["tool_result"] = (
        str(target_data.get("tool_result") or "")
        + "\n\n[smoke] source outcome flipped to force replay mismatch\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mutated_trace = output_dir / "trace_forced_mismatch.jsonl"
    manifest = output_dir / "manifest.yaml"
    _write_jsonl(mutated_trace, records)
    yaml.safe_dump([str(mutated_trace)], manifest.open("w", encoding="utf-8"))

    # Preserve relative checkpoint paths by copying checkpoint archives next to
    # the mutated trace with the same relative layout.
    for record in checkpointed:
        cp = record["data"]["checkpoint_after"]
        raw_path = Path(str(cp["path"]))
        if raw_path.is_absolute():
            continue
        src = trace_path.parent / raw_path
        dst = mutated_trace.parent / raw_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print(f"mutated_trace={mutated_trace}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
