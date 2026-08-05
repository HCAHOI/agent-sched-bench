#!/usr/bin/env python3
"""Discover one frozen target's pre-existing file footprint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.build_physical_state_conditions import (  # noqa: E402
    _sha256,
    read_prepared_artifact,
    read_prepared_artifact_with_digest,
)
from scripts.evaluation.build_physical_state_template import (  # noqa: E402
    RAW_SCHEMA,
    opened_paths,
)
from scripts.evaluation.prepare_physical_state_task import (  # noqa: E402
    _manifest_task,
)
from trace_collect.attempt_pipeline import (  # noqa: E402
    start_task_container,
    stop_task_container,
)

DISCOVERY_SCHEMA = "sqlglot-physical-state-discovery-v1"
STRACE_CONTAINER_PATH = "/tmp/physical-state.strace"
RAW_CONTAINER_PATH = "/tmp/physical-state-open-paths.json"
FILTER_CONTAINER_PATH = "/tmp/build-physical-state-template.py"
TEMPLATE_CONTAINER_PATH = "/tmp/physical-state-template.json"
PROBE_INPUT_CONTAINER_PATH = "/tmp/physical-state-template.tsv"
FILTER_SOURCE = _ROOT / "scripts/evaluation/build_physical_state_template.py"
CONTAINER_PYTHON = "/opt/conda/envs/testbed/bin/python"


def _run(
    argv: Sequence[str], *, timeout: float, check: bool = True
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    result = subprocess.run(
        list(argv), capture_output=True, text=True, check=False, timeout=timeout
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    return result, elapsed_ms


def _copy(
    executable: str, source: str | Path, destination: str | Path
) -> float:
    _result, elapsed_ms = _run(
        [executable, "cp", str(source), str(destination)], timeout=120
    )
    return elapsed_ms


def _start(
    image_id: str, task_id: str, purpose: str, executable: str, *, network: str
) -> tuple[str, float]:
    started = time.monotonic()
    container_id = start_task_container(
        image_id,
        executable=executable,
        run_as_host_user=False,
        mount_host_home=False,
        container_home="/root",
        network_mode=network,
        extra_args=[
            "--label",
            "agent-sched-bench.component=physical-state-discovery",
            "--label",
            f"agent-sched-bench.task_instance_id={task_id}",
            "--label",
            f"agent-sched-bench.discovery_purpose={purpose}",
        ],
    )
    return container_id, (time.monotonic() - started) * 1000


def _stop(container_id: str, executable: str) -> float:
    started = time.monotonic()
    stop_task_container(container_id, executable=executable)
    return (time.monotonic() - started) * 1000


def _ensure_strace(container_id: str, executable: str) -> dict[str, Any]:
    probe, probe_ms = _run(
        [executable, "exec", container_id, "strace", "--version"],
        timeout=30,
        check=False,
    )
    install_ms = 0.0
    installed = probe.returncode == 0
    if not installed:
        _update, update_ms = _run(
            [executable, "exec", container_id, "apt-get", "update"], timeout=900
        )
        _install, package_ms = _run(
            [
                executable,
                "exec",
                container_id,
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "strace",
            ],
            timeout=900,
        )
        install_ms = update_ms + package_ms
        probe, version_ms = _run(
            [executable, "exec", container_id, "strace", "--version"], timeout=30
        )
        probe_ms += version_ms
    return {
        "already_present": installed,
        "install_ms": install_ms,
        "version_probe_ms": probe_ms,
        "version": probe.stdout.strip(),
    }


def _discover(
    *,
    task: dict[str, Any],
    prepared: dict[str, Any],
    staging: Path,
    executable: str,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    container_id, start_ms = _start(
        str(prepared["prepared_image_id"]),
        task_id,
        "strace",
        executable,
        network="host",
    )
    stop_ms = 0.0
    try:
        strace = _ensure_strace(container_id, executable)
        args = task["target_tool_args"]
        command = [
            executable,
            "exec",
            "-w",
            str(args["working_dir"]),
            container_id,
            "strace",
            "-f",
            "-qq",
            "-yy",
            "-e",
            "trace=%file",
            "-s",
            "4096",
            "-o",
            STRACE_CONTAINER_PATH,
            "/bin/sh",
            "-c",
            str(args["command"]),
        ]
        target, target_ms = _run(
            command, timeout=float(args["timeout"]) + 60, check=False
        )
        if target.returncode != 0:
            raise RuntimeError(
                f"discovery target exit changed: source=0 discovery={target.returncode}"
            )
        strace_path = staging / "strace.log"
        copy_ms = _copy(
            executable, f"{container_id}:{STRACE_CONTAINER_PATH}", strace_path
        )
    finally:
        stop_ms = _stop(container_id, executable)

    with strace_path.open(encoding="utf-8") as source:
        paths = opened_paths(source)
    if not paths:
        raise ValueError("strace contained no resolved successful file accesses")
    raw_path = staging / "open-paths.json"
    raw_path.write_text(
        json.dumps(
            {"schema": RAW_SCHEMA, "source": "strace.log", "paths": paths},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "container_start_ms": start_ms,
        "container_stop_ms": stop_ms,
        "strace": strace,
        "target_ms": target_ms,
        "target_exit_code": target.returncode,
        "target_stdout_tail": target.stdout[-2000:],
        "target_stderr_tail": target.stderr[-2000:],
        "strace_copy_ms": copy_ms,
        "resolved_path_count": len(paths),
    }


def _filter_clean_image(
    *,
    task_id: str,
    prepared: dict[str, Any],
    staging: Path,
    executable: str,
) -> dict[str, Any]:
    container_id, start_ms = _start(
        str(prepared["prepared_image_id"]),
        task_id,
        "pre-target-filter",
        executable,
        network="none",
    )
    copy_in_ms = 0.0
    copy_out_ms = 0.0
    stop_ms = 0.0
    try:
        filter_source_bytes = FILTER_SOURCE.read_bytes()
        frozen_filter = staging / "build-physical-state-template.py"
        frozen_filter.write_bytes(filter_source_bytes)
        copy_in_ms += _copy(
            executable, staging / "open-paths.json", f"{container_id}:{RAW_CONTAINER_PATH}"
        )
        copy_in_ms += _copy(
            executable, frozen_filter, f"{container_id}:{FILTER_CONTAINER_PATH}"
        )
        result, filter_ms = _run(
            [
                executable,
                "exec",
                container_id,
                CONTAINER_PYTHON,
                FILTER_CONTAINER_PATH,
                "filter",
                "--raw",
                RAW_CONTAINER_PATH,
                "--out",
                TEMPLATE_CONTAINER_PATH,
                "--probe-input",
                PROBE_INPUT_CONTAINER_PATH,
            ],
            timeout=120,
        )
        copy_out_ms += _copy(
            executable,
            f"{container_id}:{TEMPLATE_CONTAINER_PATH}",
            staging / "template.json",
        )
        copy_out_ms += _copy(
            executable,
            f"{container_id}:{PROBE_INPUT_CONTAINER_PATH}",
            staging / "template.tsv",
        )
    finally:
        stop_ms = _stop(container_id, executable)
    template = json.loads((staging / "template.json").read_text(encoding="utf-8"))
    return {
        "container_start_ms": start_ms,
        "container_stop_ms": stop_ms,
        "network_mode": "none",
        "filter_source": str(FILTER_SOURCE.relative_to(_ROOT)),
        "filter_source_sha256": hashlib.sha256(filter_source_bytes).hexdigest(),
        "filter_ms": filter_ms,
        "copy_in_ms": copy_in_ms,
        "copy_out_ms": copy_out_ms,
        "stdout": result.stdout,
        "file_count": template["file_count"],
        "total_bytes": template["total_bytes"],
        "truncated_by": template["truncated_by"],
    }


def discover_task(
    task_id: str,
    prepared_path: Path,
    out_dir: Path,
    *,
    executable: str,
) -> dict[str, Any]:
    task, _actions, _input_digests = _manifest_task(task_id)
    prepared, prepared_digest = read_prepared_artifact_with_digest(
        task, prepared_path
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "strace": out_dir / f"{task_id}.strace",
        "raw": out_dir / f"{task_id}.open-paths.json",
        "template": out_dir / f"{task_id}.json",
        "probe_input": out_dir / f"{task_id}.tsv",
        "artifact": out_dir / f"{task_id}.discovery.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"discovery output already exists: {existing}")

    with TemporaryDirectory(prefix=f"physical-state-discovery-{task_id}-") as tmp:
        staging = Path(tmp)
        started = time.monotonic()
        discovery = _discover(
            task=task,
            prepared=prepared,
            staging=staging,
            executable=executable,
        )
        clean_filter = _filter_clean_image(
            task_id=task_id,
            prepared=prepared,
            staging=staging,
            executable=executable,
        )
        artifact = {
            "schema": DISCOVERY_SCHEMA,
            "task_id": task_id,
            "container_runtime": executable,
            "prepared_artifact": str(prepared_path.resolve()),
            "prepared_artifact_sha256": prepared_digest,
            "prepared_image": prepared["prepared_image"],
            "prepared_image_id": prepared["prepared_image_id"],
            "target_action_id": task["target_action_id"],
            "target_tool_args": task["target_tool_args"],
            "source_target_exit_code": 0,
            "strace_options": ["-f", "-qq", "-yy", "-e", "trace=%file", "-s", "4096"],
            "discovery": discovery,
            "pre_target_filter": clean_filter,
            "total_ms": (time.monotonic() - started) * 1000,
            "outputs": {
                "strace_sha256": _sha256(staging / "strace.log"),
                "raw_sha256": _sha256(staging / "open-paths.json"),
                "template_sha256": _sha256(staging / "template.json"),
                "probe_input_sha256": _sha256(staging / "template.tsv"),
            },
        }
        (staging / "discovery.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for source_name, output_name in (
            ("strace.log", "strace"),
            ("open-paths.json", "raw"),
            ("template.json", "template"),
            ("template.tsv", "probe_input"),
            ("discovery.json", "artifact"),
        ):
            shutil.move(str(staging / source_name), outputs[output_name])
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--prepared-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--container", choices=("docker", "podman"), default="docker")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    task, _actions, _digests = _manifest_task(args.task_id)
    prepared = read_prepared_artifact(task, args.prepared_artifact)
    if not args.execute:
        print(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "prepared_image_id": prepared["prepared_image_id"],
                    "target_action_id": task["target_action_id"],
                    "execution_requested": False,
                },
                sort_keys=True,
            )
        )
        return
    discover_task(
        args.task_id,
        args.prepared_artifact,
        args.out_dir,
        executable=args.container,
    )


if __name__ == "__main__":
    main()
