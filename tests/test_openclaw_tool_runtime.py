from __future__ import annotations

import asyncio
import json
import subprocess
from unittest.mock import patch

from trace_collect.openclaw_tools import execute_trace_tool


def _nested(tool_name: str, payload: dict) -> str:
    return json.dumps({tool_name: payload}, ensure_ascii=False)


def _make_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def test_exec_command_builds_correct_docker_exec() -> None:
    calls: list[list] = []

    def fake_run(*args, **kwargs):
        calls.append(list(args[0]) if args else [])
        return _make_result(stdout="hello\n")

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        result, success = asyncio.run(
            execute_trace_tool(
                container_id="cid-123",
                container_executable="docker",
                tool_name="exec",
                tool_args_json=_nested("exec", {"command": "echo hello"}),
                command_timeout_s=10.0,
            )
        )

    assert success is True
    assert "hello" in result
    cmd = calls[0]
    assert cmd[:2] == ["docker", "exec"]
    assert "-w" in cmd
    assert "/testbed" in cmd
    assert "cid-123" in cmd
    assert "bash" in cmd
    assert "echo hello" in cmd


def test_read_file_uses_cat() -> None:
    def fake_run(*args, **kwargs):
        return _make_result(stdout="file content\n")

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        result, success = asyncio.run(
            execute_trace_tool(
                container_id="cid-123",
                container_executable="docker",
                tool_name="read_file",
                tool_args_json=_nested("read_file", {"path": "/testbed/foo.py"}),
                command_timeout_s=10.0,
            )
        )

    assert success is True
    assert "file content" in result


def test_write_file_uses_stdin_pipe() -> None:
    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        calls.append({"cmd": list(args[0]) if args else [], "input": kwargs.get("input")})
        return _make_result()

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        result, success = asyncio.run(
            execute_trace_tool(
                container_id="cid-123",
                container_executable="docker",
                tool_name="write_file",
                tool_args_json=_nested("write_file", {"path": "/testbed/out.txt", "content": "payload"}),
                command_timeout_s=10.0,
            )
        )

    assert success is True
    assert "Successfully wrote" in result
    # The write call should have stdin_data
    write_call = [c for c in calls if c.get("input") is not None]
    assert len(write_call) >= 1
    assert "payload" in write_call[-1]["input"]


def test_edit_file_pipes_python_script() -> None:
    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        cmd = list(args[0]) if args else []
        calls.append({"cmd": cmd, "input": kwargs.get("input")})
        if "python3" in cmd:
            return _make_result(stdout=json.dumps({"ok": True, "msg": "Successfully edited /testbed/x.py"}))
        return _make_result()

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        result, success = asyncio.run(
            execute_trace_tool(
                container_id="cid-123",
                container_executable="docker",
                tool_name="edit_file",
                tool_args_json=_nested("edit_file", {
                    "path": "/testbed/x.py",
                    "old_text": "foo",
                    "new_text": "bar",
                }),
                command_timeout_s=10.0,
            )
        )

    assert success is True
    assert "Successfully edited" in result
    # Verify python3 was invoked with the edit script
    python_calls = [c for c in calls if "python3" in c["cmd"]]
    assert len(python_calls) == 1
    # The stdin should contain the edit request JSON
    stdin = python_calls[0]["input"]
    req = json.loads(stdin)
    assert req["path"] == "/testbed/x.py"
    assert req["old_text"] == "foo"
    assert req["new_text"] == "bar"


def test_list_dir_uses_ls() -> None:
    def fake_run(*args, **kwargs):
        cmd = list(args[0]) if args else []
        if "ls" in cmd:
            return _make_result(stdout=".\n..\nfoo.py\nbar.py\n")
        return _make_result()

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        result, success = asyncio.run(
            execute_trace_tool(
                container_id="cid-123",
                container_executable="docker",
                tool_name="list_dir",
                tool_args_json=_nested("list_dir", {"path": "/testbed"}),
                command_timeout_s=10.0,
            )
        )

    assert success is True
    assert "foo.py" in result


def test_timeout_returns_timeout_marker() -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        result, success = asyncio.run(
            execute_trace_tool(
                container_id="cid-123",
                container_executable="docker",
                tool_name="exec",
                tool_args_json=_nested("exec", {"command": "sleep 999"}),
                command_timeout_s=1.0,
            )
        )

    assert success is False
    assert "[timeout]" in result


def test_unsupported_tool_returns_error() -> None:
    result, success = asyncio.run(
        execute_trace_tool(
            container_id="cid-123",
            container_executable="docker",
            tool_name="nope_tool",
            tool_args_json="{}",
            command_timeout_s=5.0,
        )
    )

    assert success is False
    assert "Unsupported replay tool" in result


def test_podman_executable_used_in_commands() -> None:
    calls: list[list] = []

    def fake_run(*args, **kwargs):
        calls.append(list(args[0]) if args else [])
        return _make_result(stdout="ok\n")

    with patch("trace_collect.openclaw_tools.subprocess.run", side_effect=fake_run):
        asyncio.run(
            execute_trace_tool(
                container_id="cid-456",
                container_executable="podman",
                tool_name="exec",
                tool_args_json=_nested("exec", {"command": "whoami"}),
                command_timeout_s=10.0,
            )
        )

    assert calls[0][0] == "podman"
