#!/usr/bin/env python3
"""Generate simulate inputs from a collect run directory.

Usage:
  uv run python scripts/gen_simulate_manifest.py <collect_run_dir>

Outputs:
  <collect_run_dir>/simulate_manifest.yaml   - manifest for simulate
  <collect_run_dir>/tasks.json               - task source for simulate

The script discovers trace.jsonl files under each instance/attempt_*/ dir
and extracts the task data from run_manifest.json (written by the collector).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <collect_run_dir>")
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    trace_paths: list[str] = []
    tasks: list[dict] = []

    for instance_dir in sorted(run_dir.iterdir()):
        if not instance_dir.is_dir():
            continue
        # Find the latest attempt with a trace.jsonl
        attempts = sorted(instance_dir.glob("attempt_*/trace.jsonl"))
        if not attempts:
            print(f"SKIP {instance_dir.name}: no trace.jsonl found")
            continue
        trace_path = attempts[-1]  # latest attempt
        trace_paths.append(str(trace_path))

        # Try to extract task from run_manifest.json
        attempt_dir = trace_path.parent
        manifest_path = attempt_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            task = manifest.get("task", {})
            if task:
                tasks.append(task)
                continue

        # Fallback: construct minimal task entry
        tasks.append({"instance_id": instance_dir.name})
        print(f"WARN {instance_dir.name}: no task data in run_manifest, using minimal entry")

    if not trace_paths:
        raise SystemExit(f"no trace.jsonl files found under {run_dir}")

    # Write tasks.json
    tasks_path = run_dir / "tasks.json"
    tasks_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks -> {tasks_path}")

    # Write manifest
    manifest_path = run_dir / "simulate_manifest.yaml"
    manifest = {
        "version": 1,
        "defaults": {"task_source": str(tasks_path)},
        "traces": trace_paths,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    print(f"Wrote manifest ({len(trace_paths)} traces) -> {manifest_path}")

    # Print next steps
    print()
    print("=== Next: run simulate ===")
    print(f"uv run python -m trace_collect.cli simulate \\")
    print(f"  --manifest {manifest_path} \\")
    print(f"  --task-source {tasks_path} \\")
    print(f"  --container docker \\")
    print(f"  --concurrency 1 \\")
    print(f"  --output-dir {run_dir}/simulate_output")


if __name__ == "__main__":
    main()
