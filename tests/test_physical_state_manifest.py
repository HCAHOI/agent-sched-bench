import json

from scripts.evaluation.build_physical_state_manifest import build_manifest


def _write_trace(root, task_id: str, *, prefix_tool: str = "edit_file") -> None:
    attempt = root / task_id / "attempt_1"
    attempt.mkdir(parents=True)
    (attempt / "resource_observations.json").write_text(
        json.dumps({"collection_validity": "valid"}), encoding="utf-8"
    )
    (attempt / "results.json").write_text(
        json.dumps({"success": True}), encoding="utf-8"
    )
    actions = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": f"{task_id}-prefix",
            "data": {
                "tool_name": prefix_tool,
                "tool_args": json.dumps(
                    {"path": "/testbed/x", "old_text": "a", "new_text": "b"}
                ),
                "duration_ms": 2,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": f"{task_id}-target",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "python -m pytest tests/test_x.py"}),
                "tool_result": "1 passed\n\nExit code: 0",
                "duration_ms": 20,
            },
        },
    ]
    (attempt / "trace.jsonl").write_text(
        "".join(json.dumps(action) + "\n" for action in actions), encoding="utf-8"
    )


def test_manifest_freezes_seeded_label_free_prefix(tmp_path) -> None:
    trace_root = tmp_path / "traces"
    _write_trace(trace_root, "task-a")
    _write_trace(trace_root, "task-b")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            [
                {"instance_id": "task-a", "docker_image": "image-a"},
                {"instance_id": "task-b", "docker_image": "image-b"},
            ]
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(
        trace_root,
        tasks,
        seed=42,
        expected_eligible=2,
        expected_selected=("task-b", "task-a"),
        repo_root=tmp_path,
    )

    assert manifest["selection"]["resource_labels_read"] is False
    assert [row["task_id"] for row in manifest["tasks"]] == ["task-b", "task-a"]
    assert manifest["tasks"][0]["prefix_replay_action_ids"] == ["task-b-prefix"]
    assert manifest["tasks"][0]["target_action_id"] == "task-b-target"
