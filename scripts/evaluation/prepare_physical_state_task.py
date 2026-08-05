#!/usr/bin/env python3
"""Replay one frozen SQLGlot prefix and commit its filesystem state."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import posixpath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from harness.container_image_prep import (  # noqa: E402
    ensure_source_image,
    normalize_image_reference,
)
from trace_collect.attempt_pipeline import (  # noqa: E402
    configure_task_container_apt_mirror,
    start_task_container,
    stop_task_container,
)
from trace_collect.openclaw_tools import (  # noqa: E402
    ContainerAgent,
    execute_trace_tool_detailed,
)
from trace_collect.simulator import (  # noqa: E402
    _command_exit_code,
    _source_tool_success,
)

MANIFEST_SCHEMA = "sqlglot-physical-state-manifest-v1"
MANIFEST_PATH = _ROOT / "analysis/development/sqlglot-physical-state-manifest.json"
PREPARED_IMAGE_REPOSITORY = "agent-sched-bench/sqlglot-physical-state"
REPLAY_TOOLS = frozenset({"exec", "write_file", "edit_file"})
PROBE_SOURCE = _ROOT / "scripts/evaluation/physical_state_probe.c"
PROBE_CONTAINER_PATH = "/opt/agent-sched-bench/physical-state-probe"
PROBE_COMPILE_FLAGS = (
    "-O2",
    "-std=c11",
    "-D_POSIX_C_SOURCE=200809L",
    "-static",
    "-Wall",
    "-Wextra",
    "-Werror",
)
_TOOL_ARG_KEYS = {
    "exec": frozenset({"command", "timeout", "working_dir"}),
    "write_file": frozenset({"path", "content"}),
    "edit_file": frozenset({"path", "old_text", "new_text", "replace_all"}),
}


def _manifest_task(
    task_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    manifest_path = MANIFEST_PATH
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("physical-state manifest schema changed")
    matches = [
        row for row in manifest.get("tasks", ()) if row.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one manifest row for {task_id}, found {len(matches)}"
        )
    task = matches[0]
    wanted = list(task["prefix_replay_action_ids"])
    wanted_set = set(wanted)
    source_trace_value = Path(str(task["source_trace"]))
    if source_trace_value.is_absolute():
        raise ValueError("source trace must be repository-relative")
    source_trace = (_ROOT / source_trace_value).resolve()
    if not source_trace.is_relative_to(_ROOT):
        raise ValueError("source trace escapes the repository")
    actions = []
    source_trace_bytes = source_trace.read_bytes()
    for line in source_trace_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("action_id") in wanted_set:
            actions.append(row)
    found = [row.get("action_id") for row in actions]
    if found != wanted:
        raise ValueError(
            f"source prefix for {task_id} differs from the frozen manifest"
        )
    return task, actions, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_trace_sha256": hashlib.sha256(source_trace_bytes).hexdigest(),
    }


async def replay_prefix(
    agent: ContainerAgent,
    actions: Sequence[Mapping[str, Any]],
    *,
    command_timeout_s: float,
) -> list[dict[str, Any]]:
    """Replay frozen actions, rejecting any changed command or tool outcome."""
    rows = []
    for action in actions:
        data = action.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"action {action.get('action_id')} has no data")
        tool_name = str(data.get("tool_name") or "")
        if tool_name not in REPLAY_TOOLS:
            raise ValueError(
                f"prefix action {action.get('action_id')} uses unsupported {tool_name!r}"
            )
        tool_args = data.get("tool_args")
        if not isinstance(tool_args, str):
            raise ValueError(f"action {action.get('action_id')} has invalid tool args")
        try:
            params = json.loads(tool_args)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"prefix action {action.get('action_id')} has invalid JSON args"
            ) from exc
        if not isinstance(params, dict) or set(params) != _TOOL_ARG_KEYS[tool_name]:
            raise ValueError(
                f"prefix action {action.get('action_id')} args do not match "
                f"the frozen {tool_name} schema"
            )
        if tool_name == "exec":
            if (
                not isinstance(params["command"], str)
                or not params["command"]
                or params["working_dir"] != "/testbed"
                or isinstance(params["timeout"], bool)
                or not isinstance(params["timeout"], (int, float))
                or not 0 < float(params["timeout"]) <= 600
            ):
                raise ValueError(
                    f"prefix action {action.get('action_id')} has invalid exec args"
                )
        elif (
            not isinstance(params["path"], str)
            or not params["path"].startswith("/testbed/")
            or posixpath.normpath(params["path"]) != params["path"]
            or not all(
                isinstance(params[key], str)
                for key in (
                    ("content",)
                    if tool_name == "write_file"
                    else ("old_text", "new_text")
                )
            )
            or (
                tool_name == "edit_file" and not isinstance(params["replace_all"], bool)
            )
        ):
            raise ValueError(
                f"prefix action {action.get('action_id')} has invalid {tool_name} args"
            )
        started = time.monotonic()
        result, replay_success, inner_ms, metadata = await execute_trace_tool_detailed(
            agent=agent,
            tool_name=tool_name,
            tool_args_json=tool_args,
            command_timeout_s=command_timeout_s,
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        failure_kind = metadata.get("replay_failure_kind")
        if failure_kind:
            raise RuntimeError(
                f"prefix action {action.get('action_id')} failed: {failure_kind}"
            )

        source_result = str(data.get("tool_result", data.get("result", "")))
        source_success = _source_tool_success(dict(data))
        source_exit = _command_exit_code(source_result) if tool_name == "exec" else None
        replay_exit = _command_exit_code(result) if tool_name == "exec" else None
        if tool_name == "exec":
            if source_exit is None or replay_exit is None or replay_exit != source_exit:
                raise RuntimeError(
                    f"prefix action {action.get('action_id')} exit changed: "
                    f"source={source_exit}, replay={replay_exit}"
                )
        elif replay_success is not source_success:
            raise RuntimeError(
                f"prefix action {action.get('action_id')} success changed: "
                f"source={source_success}, replay={replay_success}"
            )
        rows.append(
            {
                "action_id": action.get("action_id"),
                "tool_name": tool_name,
                "source_success": source_success,
                "replay_transport_success": replay_success,
                "source_exit_code": source_exit,
                "replay_exit_code": replay_exit,
                "replay_duration_ms": inner_ms if inner_ms is not None else elapsed_ms,
            }
        )
    return rows


def _prepared_image(task_id: str) -> str:
    tag = re.sub(r"[^a-z0-9_.-]+", "-", task_id.lower()).strip("-.")
    return f"{PREPARED_IMAGE_REPOSITORY}:{tag}"


def _inspect_image_id(image: str, executable: str, *, required: bool) -> str | None:
    result = subprocess.run(
        [executable, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if required:
        raise RuntimeError(
            f"image inspect failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_probe(directory: Path) -> tuple[Path, dict[str, Any]]:
    compiler = Path("/usr/bin/cc")
    binary = directory / "physical-state-probe"
    version = subprocess.run(
        [str(compiler), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    command = [
        str(compiler),
        *PROBE_COMPILE_FLAGS,
        str(PROBE_SOURCE),
        "-o",
        str(binary),
    ]
    started = time.monotonic()
    subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    compile_ms = (time.monotonic() - started) * 1000
    return binary, {
        "source": str(PROBE_SOURCE.relative_to(_ROOT)),
        "source_sha256": _sha256(PROBE_SOURCE),
        "compiler": str(compiler),
        "compiler_version": version.strip(),
        "compile_flags": list(PROBE_COMPILE_FLAGS),
        "compile_ms": compile_ms,
        "binary_size_bytes": binary.stat().st_size,
        "binary_sha256": _sha256(binary),
        "container_path": PROBE_CONTAINER_PATH,
    }


def _install_probe(
    binary: Path, container_id: str, executable: str
) -> dict[str, float]:
    started = time.monotonic()
    commands = (
        [
            executable,
            "exec",
            container_id,
            "mkdir",
            "-p",
            str(Path(PROBE_CONTAINER_PATH).parent),
        ],
        [executable, "cp", str(binary), f"{container_id}:{PROBE_CONTAINER_PATH}"],
        [executable, "exec", container_id, "chmod", "0555", PROBE_CONTAINER_PATH],
    )
    for command in commands:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    return {"copy_ms": (time.monotonic() - started) * 1000}


async def _cleanup(
    agent: ContainerAgent | None, container_id: str, executable: str
) -> None:
    agent_error: BaseException | None = None
    if agent is not None:
        try:
            await agent.stop()
        except BaseException as exc:
            agent_error = exc
    try:
        await asyncio.to_thread(
            stop_task_container, container_id, executable=executable
        )
    except BaseException as exc:
        if agent_error is not None:
            exc.__context__ = agent_error
        raise
    if agent_error is not None:
        raise agent_error


async def prepare_task(
    task_id: str,
    out: Path,
    *,
    container_executable: str,
    command_timeout_s: float,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    task, actions, input_digests = _manifest_task(task_id)
    probe_temp = TemporaryDirectory(prefix="physical-state-probe-")
    try:
        probe_binary, probe_provenance = await asyncio.to_thread(
            _build_probe, Path(probe_temp.name)
        )
    except BaseException:
        probe_temp.cleanup()
        raise
    source_image = normalize_image_reference(str(task["image"]))
    prepared_image = _prepared_image(task_id)
    if _inspect_image_id(prepared_image, container_executable, required=False):
        raise FileExistsError(f"prepared image already exists: {prepared_image}")

    try:
        await asyncio.to_thread(
            ensure_source_image,
            source_image,
            container_executable=container_executable,
        )
    except BaseException:
        probe_temp.cleanup()
        raise
    source_image_id = _inspect_image_id(
        source_image, container_executable, required=True
    )
    container_id = await asyncio.to_thread(
        start_task_container,
        source_image,
        executable=container_executable,
        run_as_host_user=False,
        mount_host_home=False,
        container_home="/root",
        extra_args=[
            "--label",
            "agent-sched-bench.component=physical-state-prepare",
            "--label",
            f"agent-sched-bench.task_instance_id={task_id}",
        ],
    )
    agent: ContainerAgent | None = None
    artifact: dict[str, Any] | None = None
    try:
        apt_mirror = await asyncio.to_thread(
            configure_task_container_apt_mirror,
            container_id,
            executable=container_executable,
        )
        agent = ContainerAgent(container_id, container_executable)
        await agent.start()
        rows = await replay_prefix(
            agent,
            actions,
            command_timeout_s=command_timeout_s,
        )
        probe_provenance.update(
            await asyncio.to_thread(
                _install_probe,
                probe_binary,
                container_id,
                container_executable,
            )
        )
        await agent.stop()
        agent = None
        commit = await asyncio.to_thread(
            subprocess.run,
            [container_executable, "commit", container_id, prepared_image],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if commit.returncode != 0:
            raise RuntimeError(
                f"container commit failed: {commit.stderr.strip() or commit.stdout.strip()}"
            )
        image_id = commit.stdout.strip()
        artifact = {
            "schema": "sqlglot-physical-state-prepared-task-v1",
            "manifest": str(MANIFEST_PATH.relative_to(_ROOT)),
            "manifest_sha256": input_digests["manifest_sha256"],
            "manifest_schema": MANIFEST_SCHEMA,
            "task_id": task_id,
            "source_image": source_image,
            "source_image_id": source_image_id,
            "prepared_image": prepared_image,
            "prepared_image_id": image_id,
            "container_runtime": container_executable,
            "network_mode": "host",
            "command_timeout_s": command_timeout_s,
            "apt_mirror": apt_mirror,
            "replayed_tools": sorted(REPLAY_TOOLS),
            "probe": probe_provenance,
            "source_trace": task["source_trace"],
            "source_trace_sha256": input_digests["source_trace_sha256"],
            "target_action_id": task["target_action_id"],
            "target_command": task["target_command"],
            "prefix_actions": rows,
        }
    finally:
        try:
            await _cleanup(agent, container_id, container_executable)
        finally:
            probe_temp.cleanup()
    if artifact is None:
        raise AssertionError("preparation completed without an artifact")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--container", choices=("docker", "podman"), default="docker")
    parser.add_argument("--command-timeout", type=float, default=600.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    task, actions, _input_digests = _manifest_task(args.task_id)
    if not args.execute:
        print(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "source_image": task["image"],
                    "prepared_image": _prepared_image(args.task_id),
                    "prefix_actions": len(actions),
                    "execution_requested": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.out is None:
        parser.error("--out is required with --execute")
    asyncio.run(
        prepare_task(
            args.task_id,
            args.out,
            container_executable=args.container,
            command_timeout_s=args.command_timeout,
        )
    )


if __name__ == "__main__":
    main()
