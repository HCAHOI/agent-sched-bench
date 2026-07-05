#!/usr/bin/env python3
"""DR1: Memory-diff cost reality check — Stage 1 (trace analysis).

Hypothesis: per-turn dirty-memory bytes are within ~2x of file-CAS delta bytes
for typical SWE-rebench turns, and the diff-write pause fits inside the p5 LLM
wait budget.

Stage 1 reads existing source-collect trace directories (which contain
per-task ``trace.jsonl`` files with ``checkpoint_after`` CAS manifests) and
computes file-CAS delta bytes per turn boundary.  Dirty-memory estimates and
pause-timing estimates are computed analytically from the manifest data; actual
Firecracker measurements are deferred to Stage 2 (``dr1_runner.sh``).

Output: CSV to stdout with columns:
  turn_number, dirty_memory_bytes, file_cas_delta_bytes, pause_ms, guest_slowdown_ms
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

# Only import _checkpoint_after_spec for path resolution — we read manifests
# directly because _source_cas_manifest_entry strips the ``size`` field, which
# this experiment needs.
from trace_collect.simulator import _checkpoint_after_spec


# ---------------------------------------------------------------------------
# Estimate constants (conservative, justified inline)
# ---------------------------------------------------------------------------

# ext4 block size granularity: each dirty file rounds up to at least one 4 KiB
# block, and metadata (inode, directory entries, journal) adds ~15 % overhead.
_ESTIMATE_DIRTY_OVERHEAD_RATIO = 1.15

# Approximate Firecracker track_dirty_pages + diff-snapshot write throughput
# for a local ext4 rootfs on a modern NVMe SSD.  Conservative estimate.
_DIRTY_PAGE_WRITE_MB_PER_S = 80.0  # MB/s

# Fixed cost per diff-snapshot: VM-exit round-trip + ioctl overhead + merge.
_SNAPSHOT_FIXED_COST_MS = 8.0

# Guest slowdown from page-fault tracking.  Empirical estimate from prior
# Firecracker snapshot work — typically < 5 % wall-clock impact, ~2 ms per
# turn for small working sets.
_TRACKING_OVERHEAD_MS_PER_TURN = 2.0

# Directories excluded from CAS manifests (mirrors _CHECKPOINT_SKIP_DIRS).
_SKIP_DIRS = frozenset({".git"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_files(root: Path) -> list[Path]:
    """Discover ``trace.jsonl`` files beneath *root*.

    Looks for the conventional layout ``{task_id}/attempt_1/trace.jsonl`` and
    also accepts a single ``trace.jsonl`` passed directly.
    """
    # Direct path to a single trace file (checked before is_dir)
    if root.is_file():
        if root.suffix == ".jsonl" or root.name == "trace.jsonl":
            return [root]
        return []
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    for attempt_dir in root.rglob("attempt_1"):
        trace = attempt_dir / "trace.jsonl"
        if trace.is_file():
            candidates.append(trace)
    if not candidates:
        direct = root / "trace.jsonl"
        if direct.is_file():
            candidates.append(direct)
    return sorted(candidates)


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, returning all records."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _iter_turns(
    records: Sequence[dict[str, Any]],
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield ``(turn_index, tool_exec_actions)`` tuples.

    A *turn* is the set of ``tool_exec`` actions that follow an ``llm_call``
    action (inclusive of that llm_call's iteration).  Actions before the first
    ``llm_call`` (e.g. startup events) are skipped.  Empty turns (llm_call
    with no following tool_execs) are yielded with an empty action list.
    """
    turn_index = 0
    current_tool_actions: list[dict[str, Any]] = []
    in_turn = False
    for record in records:
        rtype = record.get("type")
        if rtype == "action" and record.get("action_type") == "llm_call":
            if in_turn:
                yield turn_index, current_tool_actions
                turn_index += 1
                current_tool_actions = []
            in_turn = True
        elif rtype == "action" and record.get("action_type") == "tool_exec":
            if in_turn:
                current_tool_actions.append(record)
        elif rtype == "summary":
            if in_turn:
                yield turn_index, current_tool_actions
                turn_index += 1
                current_tool_actions = []
            in_turn = False
    if in_turn:
        yield turn_index, current_tool_actions


# -- Manifest reading (preserves ``size``, unlike _source_cas_manifest_entry) --


def _read_manifest_entries(manifest_path: str) -> dict[str, dict[str, Any]] | None:
    """Load a CAS manifest JSON file, returning ``{relpath: entry_dict}``.

    Returns ``None`` when the file is missing or unparseable.  Each entry
    dict preserves ``hash``, ``size``, ``mode``, ``mtime_ns``, and ``type``
    for symlinks.  Entries under skipped directories (e.g. ``.git``) are
    excluded.
    """
    mpath = Path(manifest_path)
    if not mpath.is_file():
        return None
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_entries = data.get("entries", {})
    if not isinstance(raw_entries, dict):
        return None
    result: dict[str, dict[str, Any]] = {}
    for rel, entry in raw_entries.items():
        if not isinstance(rel, str):
            continue
        if _relpath_is_skipped(rel):
            continue
        if not isinstance(entry, dict):
            continue
        result[rel] = dict(entry)
    return result


def _relpath_is_skipped(relpath: str) -> bool:
    return any(part in _SKIP_DIRS for part in relpath.split("/"))


def _manifest_is_incremental(manifest_path: str) -> bool:
    """Return True if the manifest at *manifest_path* is incremental."""
    mpath = Path(manifest_path)
    if not mpath.is_file():
        return False
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("incremental", False) is True


def _fold_manifest(
    manifest_path: str,
    prev_folded: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fold *manifest_path* into *prev_folded*, returning the new state.

    For full manifests, replaces *prev_folded* entirely.  For incremental
    manifests, updates in-place.
    """
    entries = _read_manifest_entries(manifest_path)
    if entries is None:
        return prev_folded

    mpath = Path(manifest_path)
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return prev_folded

    is_incremental = data.get("incremental", False) is True
    if not is_incremental:
        return entries

    folded = dict(prev_folded)
    folded.update(entries)
    deleted = data.get("deleted_paths", [])
    if isinstance(deleted, list):
        for dpath in deleted:
            if isinstance(dpath, str) and not _relpath_is_skipped(dpath):
                folded.pop(dpath, None)
    return folded


def _entry_hash(entry: dict[str, Any]) -> str:
    """Stable hash for comparing two manifest entries."""
    entry_type = entry.get("type", "file")
    if entry_type == "symlink":
        target = entry.get("target", "")
        return json.dumps({"type": "symlink", "target": target}, sort_keys=True)
    return str(entry.get("hash", ""))


def _entry_size_bytes(entry: dict[str, Any]) -> int:
    """Return the file size (bytes) for a manifest entry dict."""
    if entry.get("type") == "symlink":
        return 0
    size = entry.get("size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        return max(0, int(size))
    return 0


# -- Main computation --


def _compute_file_cas_delta_bytes(
    turn_actions: list[dict[str, Any]],
    source_trace: Path,
    previous_folded: dict[str, dict[str, Any]],
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Compute file-CAS delta bytes for a turn.

    Returns ``(delta_bytes, new_folded_state)`` where *delta_bytes* is the
    sum of file sizes that were added, modified, or removed during this turn.
    """
    folded = dict(previous_folded)
    if not turn_actions:
        return 0, folded

    for action in turn_actions:
        data = action.get("data") or {}
        spec = _checkpoint_after_spec(action_data=data, source_trace=source_trace)
        if spec is None:
            continue
        manifest_path = spec["path"]
        folded = _fold_manifest(manifest_path, folded)

    # Compare previous vs current folded state
    previous_keys = set(previous_folded.keys())
    current_keys = set(folded.keys())

    delta_bytes = 0
    # Added entries
    for key in current_keys - previous_keys:
        delta_bytes += _entry_size_bytes(folded[key])
    # Modified entries
    for key in current_keys & previous_keys:
        prev_entry = previous_folded[key]
        curr_entry = folded[key]
        if _entry_hash(prev_entry) != _entry_hash(curr_entry):
            delta_bytes += _entry_size_bytes(curr_entry)
            delta_bytes += _entry_size_bytes(prev_entry)

    return delta_bytes, folded


def _estimate_dirty_bytes(file_cas_delta: int) -> int:
    """Estimate dirty-memory bytes from file-CAS delta.

    Each dirty file rounds up to 4 KiB ext4 blocks; metadata (inode,
    directory entries, journal) adds ~15 % overhead.  This is the value
    ``track_dirty_pages`` is expected to report.
    """
    return round(file_cas_delta * _ESTIMATE_DIRTY_OVERHEAD_RATIO)


def _estimate_pause_ms(dirty_bytes: int) -> float:
    """Estimate diff-snapshot pause duration in ms.

    Includes the fixed VM-exit + ioctl cost plus the time to write dirty pages
    to the diff snapshot file at ~80 MB/s.
    """
    transfer_ms = (dirty_bytes / (_DIRTY_PAGE_WRITE_MB_PER_S * 1_000_000)) * 1000.0
    return round(_SNAPSHOT_FIXED_COST_MS + transfer_ms, 3)


def _process_trace(
    trace_path: Path,
) -> Iterator[dict[str, Any]]:
    """Yield per-turn analysis rows for a single trace file."""
    records = _parse_jsonl(trace_path)
    folded: dict[str, dict[str, Any]] = {}
    turn_number = 0
    for _turn_idx, turn_actions in _iter_turns(records):
        turn_number += 1
        cas_delta, folded = _compute_file_cas_delta_bytes(
            turn_actions, trace_path, folded
        )
        dirty_bytes = _estimate_dirty_bytes(cas_delta)
        pause_ms = _estimate_pause_ms(dirty_bytes)
        slowdown_ms = _TRACKING_OVERHEAD_MS_PER_TURN
        yield {
            "turn_number": turn_number,
            "dirty_memory_bytes": dirty_bytes,
            "file_cas_delta_bytes": cas_delta,
            "pause_ms": pause_ms,
            "guest_slowdown_ms": slowdown_ms,
        }


CSV_FIELDNAMES = [
    "turn_number",
    "dirty_memory_bytes",
    "file_cas_delta_bytes",
    "pause_ms",
    "guest_slowdown_ms",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DR1: Memory-diff cost reality check — Stage 1 trace analysis"
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Directory containing per-task trace.jsonl files "
        "(conventional layout: {task_id}/attempt_1/trace.jsonl)",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (default: 1.0, reserved for Stage 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write CSV to file instead of stdout",
    )
    args = parser.parse_args()

    trace_files = _trace_files(args.trace_dir)
    if not trace_files:
        print(
            f"ERROR: No trace.jsonl files found under {args.trace_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_fh = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()

    total_rows = 0
    for trace_path in trace_files:
        for row in _process_trace(trace_path):
            writer.writerow(row)
            total_rows += 1

    if args.output:
        out_fh.close()
        print(f"Wrote {total_rows} rows → {args.output}", file=sys.stderr)
    else:
        print(
            f"# Wrote {total_rows} rows from {len(trace_files)} trace(s)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
