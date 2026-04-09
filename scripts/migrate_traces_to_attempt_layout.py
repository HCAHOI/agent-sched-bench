#!/usr/bin/env python3
"""One-shot migration: flat trace layout → nested attempt_1/ layout.

Walks ``traces/swe-rebench/<model>/<run_id>/*.jsonl`` (legacy flat format)
and rewrites each trace file as
``traces/swe-rebench/<model>/<run_id>/<instance_id>/attempt_<N>/trace.jsonl``
with a stub ``run_manifest.json`` + empty ``resources.json``. Idempotent:
re-running will not duplicate already-migrated files.

Usage::

    # Dry run — see what would move
    python scripts/migrate_traces_to_attempt_layout.py --dry-run traces/swe-rebench

    # Actually migrate
    python scripts/migrate_traces_to_attempt_layout.py --apply traces/swe-rebench
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
STUB_MODEL = "migrated"


def _iter_flat_traces(root: Path):
    """Yield (trace_file, parent_run_dir) for every legacy flat trace.

    A legacy trace is a JSONL file directly inside a run directory whose
    stem is an instance_id (not ``results`` or ``preds`` or similar).
    """
    for trace_file in sorted(root.rglob("*.jsonl")):
        if trace_file.name in ("results.jsonl", "preds.jsonl"):
            continue
        # Skip files that are already inside an attempt_* subdir.
        if any(p.startswith("attempt_") for p in trace_file.parts):
            continue
        run_dir = trace_file.parent
        yield trace_file, run_dir


def _next_attempt_number(instance_dir: Path) -> int:
    """Return the next sequential attempt number for *instance_dir*."""
    if not instance_dir.exists():
        return 1
    existing = sorted(instance_dir.glob("attempt_*"))
    if not existing:
        return 1
    nums: list[int] = []
    for path in existing:
        try:
            nums.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(nums) + 1) if nums else 1


def _read_trace_metadata(trace_file: Path) -> dict:
    """Extract the first trace_metadata and summary records if present."""
    metadata: dict = {}
    summary: dict = {}
    try:
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "trace_metadata" and not metadata:
                    metadata = entry
                elif entry.get("type") == "summary" and not summary:
                    summary = entry
                if metadata and summary:
                    break
    except OSError:
        pass
    return {"metadata": metadata, "summary": summary}


def _write_stub_manifest(
    attempt_dir: Path,
    instance_id: str,
    trace_info: dict,
    source_path: Path,
) -> None:
    metadata = trace_info.get("metadata", {}) or {}
    summary = trace_info.get("summary", {}) or {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "migrated",
        "characterization_only": False,
        "task": {
            "instance_id": instance_id,
            "repo": metadata.get("repo", ""),
            "docker_image": metadata.get("docker_image", ""),
        },
        "attempt": attempt_dir.name,
        "model": {
            "requested": metadata.get("model", STUB_MODEL),
            "claude_binary": None,
            "auth_mode": None,
            "auth_env_vars_present": [],
            "extra_env_keys": [],
        },
        "runtime": {
            "home": None,
            "wrapper_enabled": False,
            "memory_limit": None,
            "cpu_limit": None,
            "start_time": metadata.get("start_time"),
            "end_time": metadata.get("end_time"),
            "min_free_disk_gb": None,
        },
        "artifacts": {
            "results_json": "",
            "resources_json": "resources.json",
            "trace_jsonl": "trace.jsonl",
            "tool_calls_json": "",
            "claude_output_txt": "",
            "claude_stderr_txt": "",
            "resource_plot_png": "",
        },
        "replay": {
            "replay_ready": False,
            "source_image": metadata.get("docker_image", ""),
            "fixed_image_name": "",
            "tool_call_count": 0,
        },
        "result_summary": {
            "exit_code": 0 if summary.get("success") else 1,
            "error": summary.get("error"),
            "total_time": summary.get("elapsed_s"),
            "characterization_time": summary.get("elapsed_s"),
            "active_time": (summary.get("total_llm_ms") or 0.0) / 1000.0,
            "tool_time": (summary.get("total_tool_ms") or 0.0) / 1000.0,
            "tool_ratio_active": 0.0,
        },
        "migrated_from": str(source_path),
        "migrated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", ""),
    }
    (attempt_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_empty_resources(attempt_dir: Path) -> None:
    payload = {
        "samples": [],
        "summary": {
            "sample_count": 0,
            "duration_seconds": 0,
            "memory_mb": {"min": 0, "max": 0, "avg": 0},
            "cpu_percent": {"min": 0, "max": 0, "avg": 0},
            "note": "migrated_pre_pipeline (no resource samples available)",
        },
    }
    (attempt_dir / "resources.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def migrate_one(
    trace_file: Path, run_dir: Path, *, apply: bool
) -> tuple[Path, bool]:
    """Move a single flat trace into the attempt_1/ layout.

    Returns ``(attempt_dir, did_move)``. ``did_move`` is False in dry-run mode
    or when the target already exists.
    """
    instance_id = trace_file.stem
    instance_dir = run_dir / instance_id

    # Idempotency: if the instance_dir already has an attempt with a trace,
    # skip (unless the flat file is newer — then use next attempt number).
    attempt_num = _next_attempt_number(instance_dir)
    attempt_dir = instance_dir / f"attempt_{attempt_num}"

    if (attempt_dir / "trace.jsonl").exists():
        return attempt_dir, False

    if not apply:
        print(f"[dry-run] {trace_file} → {attempt_dir}/trace.jsonl")
        return attempt_dir, False

    attempt_dir.mkdir(parents=True, exist_ok=True)
    target = attempt_dir / "trace.jsonl"
    shutil.move(str(trace_file), str(target))
    trace_info = _read_trace_metadata(target)
    _write_stub_manifest(attempt_dir, instance_id, trace_info, trace_file)
    _write_empty_resources(attempt_dir)
    print(f"[apply] {trace_file.name} → {target}")
    return attempt_dir, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace tree root to scan")
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--dry-run", action="store_true", help="Print actions only")
    mx.add_argument("--apply", action="store_true", help="Move files")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    moved = 0
    seen = 0
    for trace_file, run_dir in _iter_flat_traces(root):
        seen += 1
        _, did_move = migrate_one(trace_file, run_dir, apply=args.apply)
        if did_move:
            moved += 1

    mode = "apply" if args.apply else "dry-run"
    print(
        f"[{mode}] scanned {seen} flat traces, "
        f"{'moved' if args.apply else 'would move'} {moved if args.apply else seen}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
