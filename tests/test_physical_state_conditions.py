import hashlib
import json

import pytest

from scripts.evaluation import build_physical_state_conditions as conditions
from trace_collect.simulate_manifest import (
    _load_simulate_manifest,
    _load_trace_session,
)


def _inputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    manifest_path = repo / "manifest.json"
    probe_source = repo / "scripts/evaluation/physical_state_probe.c"
    probe_source.parent.mkdir(parents=True)
    probe_source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    probe_source_hash = hashlib.sha256(probe_source.read_bytes()).hexdigest()
    filter_source = repo / conditions.FILTER_SOURCE_RELATIVE
    filter_source.write_text("# frozen filter\n", encoding="utf-8")
    filter_source_hash = hashlib.sha256(filter_source.read_bytes()).hexdigest()
    prepared_dir = tmp_path / "prepared"
    template_dir = tmp_path / "templates"
    prepared_dir.mkdir()
    template_dir.mkdir()
    tasks = []
    manifest_tasks = []
    for index in range(12):
        task_id = f"task-{index:02d}"
        trace = repo / "traces" / task_id / "trace.jsonl"
        trace.parent.mkdir(parents=True)
        action_id = f"{task_id}-target"
        command = f"python -m pytest tests/test_{index}.py"
        tool_args = {
            "command": command,
            "timeout": 300,
            "working_dir": "/testbed",
        }
        trace.write_text(
            json.dumps(
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": action_id,
                    "agent_id": task_id,
                    "iteration": 1,
                    "ts_start": 1.0,
                    "ts_end": 2.0,
                    "data": {
                        "tool_name": "exec",
                        "tool_args": json.dumps(tool_args),
                        "tool_result": "1 passed\nExit code: 0",
                        "duration_ms": 1000.0,
                        "success": True,
                        "resource_observation": {"must_not": "survive"},
                        "resource_timeline": {"must_not": "survive"},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tasks.append({"instance_id": task_id, "docker_image": f"source-{index}"})
        manifest_tasks.append(
            {
                "task_id": task_id,
                "image": f"source-{index}",
                "source_trace": str(trace.relative_to(repo)),
                "target_action_id": action_id,
                "target_trace_line": 1,
                "target_command": command,
                "target_tool_args": tool_args,
                "target_source_duration_ms": 1000.0,
                "prefix_replay_action_ids": [],
            }
        )
        (prepared_dir / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "schema": conditions.PREPARED_SCHEMA,
                    "manifest": str(manifest_path.relative_to(repo)),
                    "manifest_schema": conditions.MANIFEST_SCHEMA,
                    "task_id": task_id,
                    "source_image": conditions.normalize_image_reference(
                        f"source-{index}"
                    ),
                    "source_image_id": f"sha256:{index + 100:064x}",
                    "prepared_image": conditions._prepared_image(task_id),
                    "prepared_image_id": f"sha256:{index:064x}",
                    "source_trace": str(trace.relative_to(repo)),
                    "target_action_id": action_id,
                    "target_command": command,
                    "prefix_actions": [],
                    "probe": {
                        "source": str(probe_source.relative_to(repo)),
                        "source_sha256": probe_source_hash,
                        "compiler": "/usr/bin/cc",
                        "compiler_version": "cc test",
                        "compile_flags": list(conditions.PROBE_COMPILE_FLAGS),
                        "compile_ms": 1.0,
                        "copy_ms": 2.0,
                        "binary_size_bytes": 100,
                        "container_path": conditions.PROBE_CONTAINER_PATH,
                        "binary_sha256": f"{index:064x}",
                    },
                }
            ),
            encoding="utf-8",
        )
        template = {
            "schema": conditions.TEMPLATE_SCHEMA,
            "limits": {
                "max_paths": conditions.MAX_PATHS,
                "max_bytes": conditions.MAX_BYTES,
            },
            "file_count": 1,
            "total_bytes": 3,
            "files": [{"path": "/testbed/file.py", "size_bytes": 3}],
        }
        (template_dir / f"{task_id}.json").write_text(
            json.dumps(template), encoding="utf-8"
        )
        (template_dir / f"{task_id}.tsv").write_text(
            "3\t/testbed/file.py\n", encoding="utf-8"
        )

    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema": conditions.MANIFEST_SCHEMA, "tasks": manifest_tasks}),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for task in manifest_tasks:
        task_id = task["task_id"]
        prepared_path = prepared_dir / f"{task['task_id']}.json"
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        prepared["manifest_sha256"] = manifest_hash
        prepared["source_trace_sha256"] = hashlib.sha256(
            (repo / task["source_trace"]).read_bytes()
        ).hexdigest()
        prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
        strace_path = template_dir / f"{task_id}.strace"
        raw_path = template_dir / f"{task_id}.open-paths.json"
        template_path = template_dir / f"{task_id}.json"
        probe_input_path = template_dir / f"{task_id}.tsv"
        strace_path.write_text("strace\n", encoding="utf-8")
        raw_path.write_text("{}\n", encoding="utf-8")
        discovery = {
            "schema": conditions.DISCOVERY_SCHEMA,
            "task_id": task_id,
            "prepared_artifact_sha256": hashlib.sha256(
                prepared_path.read_bytes()
            ).hexdigest(),
            "prepared_image_id": prepared["prepared_image_id"],
            "target_action_id": task["target_action_id"],
            "target_tool_args": task["target_tool_args"],
            "source_target_exit_code": 0,
            "discovery": {"target_exit_code": 0},
            "pre_target_filter": {
                "network_mode": "none",
                "file_count": 1,
                "total_bytes": 3,
                "filter_source": conditions.FILTER_SOURCE_RELATIVE,
                "filter_source_sha256": filter_source_hash,
            },
            "outputs": {
                "strace_sha256": hashlib.sha256(strace_path.read_bytes()).hexdigest(),
                "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "template_sha256": hashlib.sha256(
                    template_path.read_bytes()
                ).hexdigest(),
                "probe_input_sha256": hashlib.sha256(
                    probe_input_path.read_bytes()
                ).hexdigest(),
            },
        }
        (template_dir / f"{task_id}.discovery.json").write_text(
            json.dumps(discovery), encoding="utf-8"
        )
    tasks_path = repo / "tasks.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    monkeypatch.setattr(conditions, "_ROOT", repo)
    monkeypatch.setattr(conditions, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(conditions, "TASKS_PATH", tasks_path)
    monkeypatch.setattr(conditions, "PROBE_SOURCE", probe_source)
    monkeypatch.setattr(
        conditions, "FROZEN_TASK_IDS", tuple(f"task-{index:02d}" for index in range(12))
    )
    return prepared_dir, template_dir


def test_builds_counterbalanced_simulator_inputs(tmp_path, monkeypatch) -> None:
    prepared_dir, template_dir = _inputs(tmp_path, monkeypatch)
    out = tmp_path / "conditions"

    protocol = conditions.build_conditions(prepared_dir, template_dir, out)

    assert len(protocol["entries"]) == 48
    assert protocol["simulator"]["concurrency"] == 1
    assert protocol["simulator"]["resource_monitoring"] == "off"
    assert protocol["simulator"] == {
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
    }
    assert [
        (row["repeat"], row["condition"])
        for row in protocol["entries"][:4]
    ] == [(1, "cold"), (1, "warm"), (2, "warm"), (2, "cold")]
    manifest_entries = _load_simulate_manifest(
        out / "simulate-manifest.json", default_task_source=None
    )
    assert len(manifest_entries) == 48
    loaded = _load_trace_session(
        manifest_entries[0].trace,
        manifest_entries[0].task_source,
        0,
        manifest_entries[0].docker_image,
        manifest_entries[0].label,
    )
    assert [row["action_type"] for row in loaded.actions] == [
        "llm_call",
        "tool_exec",
        "llm_call",
        "tool_exec",
        "llm_call",
        "tool_exec",
        "llm_call",
    ]
    tool_actions = [
        row for row in loaded.actions if row["action_type"] == "tool_exec"
    ]
    assert [row["data"]["tool_name"] for row in tool_actions] == [
        "write_file",
        "exec",
        "exec",
    ]
    carrier_calls = [
        (row["data"]["raw_response"]["choices"][0]["message"].get("tool_calls") or [
            {}
        ])[0].get("id")
        for row in loaded.actions
        if row["action_type"] == "llm_call"
    ]
    assert carrier_calls == [
        *(row["data"]["tool_call_id"] for row in tool_actions),
        None,
    ]
    assert tool_actions[-1]["data"]["tool_args"].find("test_0.py") > 0
    assert "resource_observation" not in tool_actions[-1]["data"]
    assert "resource_timeline" not in tool_actions[-1]["data"]
    assert "1 passed" not in tool_actions[-1]["data"]["tool_result"]
    assert loaded.actions[-1]["data"]["raw_response"]["choices"][0][
        "finish_reason"
    ] == "stop"


def test_rejects_probe_input_that_differs_from_template(tmp_path, monkeypatch) -> None:
    prepared_dir, template_dir = _inputs(tmp_path, monkeypatch)
    (template_dir / "task-00.tsv").write_text(
        "4\t/testbed/file.py\n", encoding="utf-8"
    )

    try:
        conditions.build_conditions(prepared_dir, template_dir, tmp_path / "out")
    except ValueError as exc:
        assert "template artifact contract changed" in str(exc)
    else:
        raise AssertionError("mismatched probe input was accepted")


def test_rejects_duplicate_frozen_task_id(tmp_path, monkeypatch) -> None:
    prepared_dir, template_dir = _inputs(tmp_path, monkeypatch)
    manifest = json.loads(conditions.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["tasks"][-1]["task_id"] = manifest["tasks"][0]["task_id"]
    conditions.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="task set changed"):
        conditions.build_conditions(prepared_dir, template_dir, tmp_path / "out")


def test_rejects_incomplete_prepared_provenance(tmp_path, monkeypatch) -> None:
    prepared_dir, template_dir = _inputs(tmp_path, monkeypatch)
    path = prepared_dir / "task-00.json"
    prepared = json.loads(path.read_text(encoding="utf-8"))
    del prepared["probe"]["compile_flags"]
    path.write_text(json.dumps(prepared), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared artifact contract changed"):
        conditions.build_conditions(prepared_dir, template_dir, tmp_path / "out")


def test_rejects_target_timing_drift(tmp_path, monkeypatch) -> None:
    prepared_dir, template_dir = _inputs(tmp_path, monkeypatch)
    trace = conditions._ROOT / "traces/task-00/trace.jsonl"
    action = json.loads(trace.read_text(encoding="utf-8"))
    action["data"]["duration_ms"] = 999.0
    trace.write_text(json.dumps(action) + "\n", encoding="utf-8")
    prepared_path = prepared_dir / "task-00.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    prepared["source_trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
    discovery_path = template_dir / "task-00.discovery.json"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery["prepared_artifact_sha256"] = hashlib.sha256(
        prepared_path.read_bytes()
    ).hexdigest()
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")

    with pytest.raises(ValueError, match="target contract changed"):
        conditions.build_conditions(prepared_dir, template_dir, tmp_path / "out")


@pytest.mark.parametrize(
    "bad_path", ["/proc/cpuinfo", "/sys/kernel/uevent_seqnum", "/dev/null", "/testbed/../etc/passwd"]
)
def test_rejects_forbidden_or_noncanonical_template_path(
    tmp_path, monkeypatch, bad_path
) -> None:
    prepared_dir, template_dir = _inputs(tmp_path, monkeypatch)
    template_path = template_dir / "task-00.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["files"][0]["path"] = bad_path
    template_path.write_text(json.dumps(template), encoding="utf-8")
    (template_dir / "task-00.tsv").write_text(
        f"3\t{bad_path}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="template artifact contract changed"):
        conditions.build_conditions(prepared_dir, template_dir, tmp_path / "out")
