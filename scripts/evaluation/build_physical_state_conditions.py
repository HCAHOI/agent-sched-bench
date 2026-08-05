#!/usr/bin/env python3
"""Build the frozen cold/warm replay inputs from prepared task artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.evaluation.build_physical_state_template import (  # noqa: E402
    MAX_BYTES,
    MAX_PATHS,
    TEMPLATE_SCHEMA,
    _normalized_path,
)
from scripts.evaluation.build_physical_state_manifest import (  # noqa: E402
    FROZEN_TASK_IDS,
)
from scripts.evaluation.prepare_physical_state_task import (  # noqa: E402
    MANIFEST_PATH,
    MANIFEST_SCHEMA,
    PROBE_COMPILE_FLAGS,
    PROBE_CONTAINER_PATH,
    PROBE_SOURCE,
    _prepared_image,
)
from harness.container_image_prep import normalize_image_reference  # noqa: E402

TASKS_PATH = _ROOT / "data/swe-rebench/tasks.json"
PREPARED_SCHEMA = "sqlglot-physical-state-prepared-task-v1"
DISCOVERY_SCHEMA = "sqlglot-physical-state-discovery-v1"
PROTOCOL_SCHEMA = "sqlglot-physical-state-conditions-v1"
ORDER_SEED = 42
TEMPLATE_CONTAINER_PATH = "/tmp/physical-state-template.tsv"
FILTER_SOURCE_RELATIVE = "scripts/evaluation/build_physical_state_template.py"


def condition_schedule(task_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Return the frozen two-repetition counterbalanced condition order."""
    rng = random.Random(ORDER_SEED)
    rows = []
    for task_id in task_ids:
        first = ("cold", "warm") if rng.getrandbits(1) else ("warm", "cold")
        for repeat, pair in ((1, first), (2, tuple(reversed(first)))):
            for condition in pair:
                rows.append(
                    {
                        "task_id": task_id,
                        "repeat": repeat,
                        "condition": condition,
                    }
                )
    return rows


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_target(task: Mapping[str, Any]) -> dict[str, Any]:
    trace = (_ROOT / str(task["source_trace"])).resolve()
    if not trace.is_relative_to(_ROOT):
        raise ValueError("source trace escapes the repository")
    matches = []
    for line_number, line in enumerate(trace.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("action_id") == task["target_action_id"]:
            matches.append((line_number, row))
    if len(matches) != 1 or matches[0][0] != task["target_trace_line"]:
        raise ValueError(f"frozen target moved for {task['task_id']}")
    action = matches[0][1]
    data = action.get("data")
    if not isinstance(data, Mapping) or data.get("tool_name") != "exec":
        raise ValueError(f"frozen target is not exec for {task['task_id']}")
    raw_args = data.get("tool_args")
    if not isinstance(raw_args, str):
        raise ValueError(f"frozen target args changed for {task['task_id']}")
    args = json.loads(raw_args)
    frozen_args = task.get("target_tool_args")
    source_duration_ms = data.get("duration_ms")
    if (
        not isinstance(args, Mapping)
        or dict(args) != frozen_args
        or set(args) != {"command", "timeout", "working_dir"}
        or args["command"] != task["target_command"]
        or args["working_dir"] != "/testbed"
        or isinstance(args["timeout"], bool)
        or not isinstance(args["timeout"], (int, float))
        or not 0 < float(args["timeout"]) <= 600
        or isinstance(source_duration_ms, bool)
        or not isinstance(source_duration_ms, (int, float))
        or source_duration_ms != task["target_source_duration_ms"]
        or data.get("success") is not True
        or "Exit code: 0" not in str(data.get("tool_result") or "")
    ):
        raise ValueError(f"frozen target contract changed for {task['task_id']}")
    return {
        "action_id": str(task["target_action_id"]),
        "tool_args": dict(args),
        "source_duration_ms": source_duration_ms,
    }


def _read_inputs(
    task: Mapping[str, Any], prepared_dir: Path, template_dir: Path
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    prepared_path = prepared_dir / f"{task_id}.json"
    template_path = template_dir / f"{task_id}.json"
    probe_input_path = template_dir / f"{task_id}.tsv"
    discovery_path = template_dir / f"{task_id}.discovery.json"
    raw_path = template_dir / f"{task_id}.open-paths.json"
    strace_path = template_dir / f"{task_id}.strace"
    prepared, prepared_digest = read_prepared_artifact_with_digest(
        task, prepared_path
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    probe_input = probe_input_path.read_text(encoding="utf-8")

    files = template.get("files")
    expected_probe_input = ""
    if isinstance(files, list):
        expected_probe_input = "".join(
            f"{row['size_bytes']}\t{row['path']}\n"
            for row in files
            if isinstance(row, Mapping)
        )
    paths = (
        [row.get("path") if isinstance(row, Mapping) else None for row in files]
        if isinstance(files, list)
        else []
    )
    if (
        template.get("schema") != TEMPLATE_SCHEMA
        or template.get("limits")
        != {"max_paths": MAX_PATHS, "max_bytes": MAX_BYTES}
        or not isinstance(files, list)
        or not files
        or len(files) > MAX_PATHS
        or template.get("file_count") != len(files)
        or template.get("total_bytes")
        != sum(int(row["size_bytes"]) for row in files)
        or int(template.get("total_bytes")) > MAX_BYTES
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not row["path"].startswith("/")
            or any(char in row["path"] for char in "\t\r\n")
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or row["size_bytes"] <= 0
            for row in files
        )
        or any(_normalized_path(path) != path for path in paths)
        or len(paths) != len(set(paths))
        or probe_input != expected_probe_input
    ):
        raise ValueError(f"template artifact contract changed for {task_id}")

    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery_outputs = discovery.get("outputs")
    clean_filter = discovery.get("pre_target_filter")
    if (
        discovery.get("schema") != DISCOVERY_SCHEMA
        or discovery.get("task_id") != task_id
        or discovery.get("prepared_artifact_sha256") != prepared_digest
        or discovery.get("prepared_image_id") != prepared["prepared_image_id"]
        or discovery.get("target_action_id") != task["target_action_id"]
        or discovery.get("target_tool_args") != task["target_tool_args"]
        or discovery.get("source_target_exit_code") != 0
        or not isinstance(discovery.get("discovery"), Mapping)
        or discovery["discovery"].get("target_exit_code") != 0
        or not isinstance(clean_filter, Mapping)
        or clean_filter.get("network_mode") != "none"
        or clean_filter.get("file_count") != template["file_count"]
        or clean_filter.get("total_bytes") != template["total_bytes"]
        or clean_filter.get("filter_source") != FILTER_SOURCE_RELATIVE
        or clean_filter.get("filter_source_sha256")
        != _sha256(_ROOT / FILTER_SOURCE_RELATIVE)
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get("strace_sha256") != _sha256(strace_path)
        or discovery_outputs.get("raw_sha256") != _sha256(raw_path)
        or discovery_outputs.get("template_sha256") != _sha256(template_path)
        or discovery_outputs.get("probe_input_sha256")
        != _sha256(probe_input_path)
    ):
        raise ValueError(f"discovery artifact contract changed for {task_id}")

    return {
        "prepared_path": prepared_path,
        "prepared": prepared,
        "prepared_digest": prepared_digest,
        "template_path": template_path,
        "probe_input_path": probe_input_path,
        "discovery_path": discovery_path,
        "probe_input": probe_input,
        "target": _read_target(task),
    }


def read_prepared_artifact(
    task: Mapping[str, Any], prepared_path: Path
) -> dict[str, Any]:
    """Read and validate one immutable prepared-image artifact."""
    return read_prepared_artifact_with_digest(task, prepared_path)[0]


def read_prepared_artifact_with_digest(
    task: Mapping[str, Any], prepared_path: Path
) -> tuple[dict[str, Any], str]:
    """Validate one prepared artifact and hash the exact parsed bytes."""
    task_id = str(task["task_id"])
    prepared_bytes = prepared_path.read_bytes()
    prepared = json.loads(prepared_bytes)

    probe = prepared.get("probe")
    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    image_id_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    prefix_rows = prepared.get("prefix_actions")
    if (
        prepared.get("schema") != PREPARED_SCHEMA
        or prepared.get("manifest") != str(MANIFEST_PATH.relative_to(_ROOT))
        or prepared.get("manifest_sha256") != _sha256(MANIFEST_PATH)
        or prepared.get("manifest_schema") != MANIFEST_SCHEMA
        or prepared.get("task_id") != task_id
        or prepared.get("source_image")
        != normalize_image_reference(str(task["image"]))
        or not image_id_pattern.fullmatch(str(prepared.get("source_image_id") or ""))
        or prepared.get("prepared_image") != _prepared_image(task_id)
        or not image_id_pattern.fullmatch(
            str(prepared.get("prepared_image_id") or "")
        )
        or prepared.get("source_trace") != task["source_trace"]
        or prepared.get("source_trace_sha256")
        != _sha256(_ROOT / str(task["source_trace"]))
        or prepared.get("target_action_id") != task["target_action_id"]
        or prepared.get("target_command") != task["target_command"]
        or not isinstance(prefix_rows, list)
        or [row.get("action_id") for row in prefix_rows]
        != task["prefix_replay_action_ids"]
        or not isinstance(probe, Mapping)
        or probe.get("source") != str(PROBE_SOURCE.relative_to(_ROOT))
        or probe.get("source_sha256") != _sha256(PROBE_SOURCE)
        or probe.get("compiler") != "/usr/bin/cc"
        or not isinstance(probe.get("compiler_version"), str)
        or not probe["compiler_version"]
        or probe.get("compile_flags") != list(PROBE_COMPILE_FLAGS)
        or not isinstance(probe.get("binary_size_bytes"), int)
        or probe["binary_size_bytes"] <= 0
        or not hash_pattern.fullmatch(str(probe.get("binary_sha256") or ""))
        or probe.get("container_path") != PROBE_CONTAINER_PATH
        or any(
            isinstance(probe.get(field), bool)
            or not isinstance(probe.get(field), (int, float))
            or probe[field] < 0
            for field in ("compile_ms", "copy_ms")
        )
    ):
        raise ValueError(f"prepared artifact contract changed for {task_id}")
    return prepared, hashlib.sha256(prepared_bytes).hexdigest()


def _action(
    *, task_id: str, action_id: str, iteration: int, tool_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    timestamp = iteration / 1000
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "agent_id": task_id,
        "instance_id": task_id,
        "iteration": iteration,
        "ts_start": timestamp,
        "ts_end": timestamp,
        "data": {
            "tool_name": tool_name,
            "tool_args": json.dumps(args, sort_keys=True),
            "tool_result": "Exit code: 0" if tool_name == "exec" else "File written",
            "duration_ms": 0.0,
            "success": True,
        },
    }


def _trace(
    *,
    task_id: str,
    condition: str,
    repeat: int,
    probe_input: str,
    target: Mapping[str, Any],
) -> str:
    metadata = {
        "type": "trace_metadata",
        "trace_format_version": 5,
        "scaffold": "openclaw",
        "instance_id": task_id,
        "model": "physical-state-controlled-replay",
        "mode": "simulate_input",
        "execution_environment": "container",
        "physical_state_condition": condition,
        "physical_state_repeat": repeat,
    }
    write = _action(
        task_id=task_id,
        action_id="physical_state_template",
        iteration=0,
        tool_name="write_file",
        args={"path": TEMPLATE_CONTAINER_PATH, "content": probe_input},
    )
    probe = _action(
        task_id=task_id,
        action_id=f"physical_state_probe_{condition}",
        iteration=1,
        tool_name="exec",
        args={
            "command": f"{PROBE_CONTAINER_PATH} {condition} {TEMPLATE_CONTAINER_PATH}",
            "timeout": 600,
            "working_dir": "/testbed",
        },
    )
    target_action = _action(
        task_id=task_id,
        action_id=str(target["action_id"]),
        iteration=2,
        tool_name="exec",
        args=dict(target["tool_args"]),
    )
    target_action["data"]["duration_ms"] = target["source_duration_ms"]
    return "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in (metadata, write, probe, target_action)
    )


def build_conditions(
    prepared_dir: Path,
    template_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Validate all frozen inputs and emit 48 deterministic replay traces."""
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("physical-state manifest schema changed")
    tasks = manifest.get("tasks")
    if (
        not isinstance(tasks, list)
        or tuple(task.get("task_id") for task in tasks) != FROZEN_TASK_IDS
    ):
        raise ValueError("physical-state task set changed")

    task_by_id = {str(task["task_id"]): task for task in tasks}
    inputs = {
        task_id: _read_inputs(task, prepared_dir, template_dir)
        for task_id, task in task_by_id.items()
    }
    schedule = condition_schedule(list(task_by_id))
    if len(schedule) != 48:
        raise AssertionError("physical-state schedule must contain 48 entries")
    traces: list[tuple[Path, str]] = []
    protocol_rows = []
    simulate_rows = []
    for index, row in enumerate(schedule):
        task_id = row["task_id"]
        repeat = row["repeat"]
        condition = row["condition"]
        name = f"{index:02d}-{task_id}-r{repeat}-{condition}.jsonl"
        trace_path = (out_dir / "traces" / name).resolve()
        task_inputs = inputs[task_id]
        traces.append(
            (
                trace_path,
                _trace(
                    task_id=task_id,
                    condition=condition,
                    repeat=repeat,
                    probe_input=task_inputs["probe_input"],
                    target=task_inputs["target"],
                ),
            )
        )
        label = f"{task_id}/r{repeat}/{condition}"
        simulate_rows.append(
            {
                "trace": str(trace_path),
                "docker_image": task_inputs["prepared"]["prepared_image_id"],
                "label": label,
            }
        )
        protocol_rows.append(
            {
                **row,
                "order_index": index,
                "label": label,
                "trace": str(trace_path),
                "prepared_artifact": str(task_inputs["prepared_path"].resolve()),
                "prepared_artifact_sha256": task_inputs["prepared_digest"],
                "prepared_image": task_inputs["prepared"]["prepared_image"],
                "prepared_image_id": task_inputs["prepared"]["prepared_image_id"],
                "template_artifact": str(task_inputs["template_path"].resolve()),
                "template_artifact_sha256": _sha256(task_inputs["template_path"]),
                "probe_input": str(task_inputs["probe_input_path"].resolve()),
                "probe_input_sha256": _sha256(task_inputs["probe_input_path"]),
                "discovery_artifact": str(task_inputs["discovery_path"].resolve()),
                "discovery_artifact_sha256": _sha256(
                    task_inputs["discovery_path"]
                ),
                "target_action_id": task_inputs["target"]["action_id"],
            }
        )

    out_dir.mkdir(parents=True)
    (out_dir / "traces").mkdir()
    for path, content in traces:
        path.write_text(content, encoding="utf-8")
    simulate_manifest = {
        "version": 1,
        "defaults": {"task_source": str(TASKS_PATH.resolve())},
        "traces": simulate_rows,
    }
    (out_dir / "simulate-manifest.json").write_text(
        json.dumps(simulate_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "manifest": str(MANIFEST_PATH.relative_to(_ROOT)),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "task_source": str(TASKS_PATH.relative_to(_ROOT)),
        "task_source_sha256": _sha256(TASKS_PATH),
        "order_seed": ORDER_SEED,
        "conditions": ["cold", "warm"],
        "repetitions": 2,
        "simulator": {
            "mode": "cloud_model",
            "concurrency": 1,
            "workers": 1,
            "prep_concurrency": 1,
            "resource_monitoring": "off",
            "pmu_monitoring": "off",
            "memory_bandwidth_monitoring": "off",
            "replay_speed": 1.0,
            "command_timeout_s": 600.0,
            "tool_resource_profile_required": True,
            "cleanup_images": False,
        },
        "entries": protocol_rows,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = build_conditions(args.prepared_dir, args.template_dir, args.out_dir)
    print(json.dumps({"entries": len(protocol["entries"]), "out": str(args.out_dir)}))


if __name__ == "__main__":
    main()
