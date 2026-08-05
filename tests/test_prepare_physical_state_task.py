import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts.evaluation import prepare_physical_state_task as subject


def _action(
    action_id: str,
    tool: str,
    result: str,
    *,
    success: bool,
    args: dict | None = None,
) -> dict:
    if args is None:
        args = {
            "exec": {"command": "true", "timeout": 10, "working_dir": "/testbed"},
            "edit_file": {
                "path": "/testbed/x",
                "old_text": "a",
                "new_text": "b",
                "replace_all": False,
            },
        }.get(tool, {})
    return {
        "action_id": action_id,
        "data": {
            "tool_name": tool,
            "tool_args": json.dumps(args),
            "tool_result": result,
            "success": success,
        },
    }


def test_replay_prefix_requires_matching_exec_and_edit_outcomes(monkeypatch) -> None:
    replies = iter(
        [
            ("missing\n\nExit code: 1", False, 3.0, {}),
            ("Error: old_text not found", False, 4.0, {}),
        ]
    )

    async def execute(**_kwargs):
        return next(replies)

    monkeypatch.setattr(subject, "execute_trace_tool_detailed", execute)
    rows = asyncio.run(
        subject.replay_prefix(
            object(),
            [
                _action("exec", "exec", "missing\n\nExit code: 1", success=True),
                _action("edit", "edit_file", "not found", success=False),
            ],
            command_timeout_s=10,
        )
    )

    assert [row["replay_exit_code"] for row in rows] == [1, None]


def test_replay_prefix_rejects_changed_exec_exit(monkeypatch) -> None:
    async def execute(**_kwargs):
        return "ok\n\nExit code: 0", True, 1.0, {}

    monkeypatch.setattr(subject, "execute_trace_tool_detailed", execute)
    with pytest.raises(RuntimeError, match="exit changed"):
        asyncio.run(
            subject.replay_prefix(
                object(),
                [_action("exec", "exec", "missing\n\nExit code: 1", success=True)],
                command_timeout_s=10,
            )
        )


def test_replay_prefix_rejects_non_mutating_tool(monkeypatch) -> None:
    async def execute(**_kwargs):
        raise AssertionError("unsupported tool must not execute")

    monkeypatch.setattr(subject, "execute_trace_tool_detailed", execute)
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(
            subject.replay_prefix(
                object(),
                [_action("read", "read_file", "contents", success=True)],
                command_timeout_s=10,
            )
        )


def test_replay_prefix_rejects_tool_type_confusion(monkeypatch) -> None:
    async def execute(**_kwargs):
        raise AssertionError("mismatched args must not execute")

    monkeypatch.setattr(subject, "execute_trace_tool_detailed", execute)
    with pytest.raises(ValueError, match="frozen edit_file schema"):
        asyncio.run(
            subject.replay_prefix(
                object(),
                [
                    _action(
                        "edit",
                        "edit_file",
                        "ok",
                        success=True,
                        args={"command": "touch /testbed/escaped"},
                    )
                ],
                command_timeout_s=10,
            )
        )


def test_replay_prefix_rejects_path_traversal(monkeypatch) -> None:
    async def execute(**_kwargs):
        raise AssertionError("traversal must not execute")

    monkeypatch.setattr(subject, "execute_trace_tool_detailed", execute)
    with pytest.raises(ValueError, match="invalid edit_file args"):
        asyncio.run(
            subject.replay_prefix(
                object(),
                [
                    _action(
                        "edit",
                        "edit_file",
                        "ok",
                        success=True,
                        args={
                            "path": "/testbed/../root/file",
                            "old_text": "a",
                            "new_text": "b",
                            "replace_all": False,
                        },
                    )
                ],
                command_timeout_s=10,
            )
        )


def test_replay_prefix_rejects_runtime_failure_metadata(monkeypatch) -> None:
    async def execute(**_kwargs):
        return (
            "Exit code: <missing>",
            False,
            1.0,
            {"replay_failure_kind": "malformed_replay_exec_response"},
        )

    monkeypatch.setattr(subject, "execute_trace_tool_detailed", execute)
    with pytest.raises(RuntimeError, match="malformed_replay_exec_response"):
        asyncio.run(
            subject.replay_prefix(
                object(),
                [_action("exec", "exec", "ok\n\nExit code: 0", success=True)],
                command_timeout_s=10,
            )
        )


def test_cleanup_stops_container_after_agent_stop_failure(monkeypatch) -> None:
    stopped = []

    class Agent:
        async def stop(self):
            raise RuntimeError("agent stop failed")

    def stop_container(container_id, *, executable):
        stopped.append((container_id, executable))

    monkeypatch.setattr(subject, "stop_task_container", stop_container)
    with pytest.raises(RuntimeError, match="agent stop failed"):
        asyncio.run(subject._cleanup(Agent(), "cid", "docker"))
    assert stopped == [("cid", "docker")]


def test_prepare_records_result_affecting_provenance(tmp_path, monkeypatch) -> None:
    task = {
        "task_id": "task-a",
        "image": "source:tag",
        "source_trace": "trace.jsonl",
        "target_action_id": "target",
        "target_command": "pytest",
    }
    monkeypatch.setattr(subject, "_manifest_task", lambda _task_id: (task, []))
    probe = tmp_path / "probe"
    probe.write_bytes(b"probe")
    monkeypatch.setattr(
        subject,
        "_build_probe",
        lambda _directory: (
            probe,
            {
                "source_sha256": "source-hash",
                "binary_sha256": "binary-hash",
                "container_path": subject.PROBE_CONTAINER_PATH,
            },
        ),
    )
    monkeypatch.setattr(
        subject, "_install_probe", lambda *_args, **_kwargs: {"copy_ms": 7.0}
    )
    image_ids = iter((None, "sha256:source"))
    monkeypatch.setattr(
        subject,
        "_inspect_image_id",
        lambda *_args, **_kwargs: next(image_ids),
    )
    monkeypatch.setattr(subject, "ensure_source_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject, "start_task_container", lambda *_args, **_kwargs: "cid"
    )
    monkeypatch.setattr(
        subject,
        "configure_task_container_apt_mirror",
        lambda *_args, **_kwargs: {
            "configured": "true",
            "main_mirror": "http://mirror",
        },
    )

    class Agent:
        def __init__(self, *_args):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(subject, "ContainerAgent", Agent)

    async def replay(*_args, **_kwargs):
        return []

    monkeypatch.setattr(subject, "replay_prefix", replay)
    monkeypatch.setattr(subject, "stop_task_container", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="sha256:prepared\n", stderr=""
        ),
    )

    out = tmp_path / "result.json"
    artifact = asyncio.run(
        subject.prepare_task(
            "task-a", out, container_executable="docker", command_timeout_s=123
        )
    )

    assert artifact["manifest"] == str(subject.MANIFEST_PATH.relative_to(subject._ROOT))
    assert artifact["source_image_id"] == "sha256:source"
    assert artifact["prepared_image_id"] == "sha256:prepared"
    assert artifact["command_timeout_s"] == 123
    assert artifact["apt_mirror"]["configured"] == "true"
    assert artifact["probe"]["binary_sha256"] == "binary-hash"
    assert artifact["probe"]["copy_ms"] == 7.0
    assert out.exists()
