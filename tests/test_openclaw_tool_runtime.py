from __future__ import annotations

import asyncio
import json
from pathlib import Path

from trace_collect.openclaw_tools import execute_trace_tool
from trace_collect.simulator import _exec_tool


def _nested(tool_name: str, payload: dict) -> str:
    return json.dumps({tool_name: payload}, ensure_ascii=False)


def test_execute_trace_tool_supports_openclaw_read_and_list(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / "pkg").mkdir(parents=True)
    (repo_dir / "pkg" / "sample.py").write_text("alpha\nbeta\n", encoding="utf-8")

    list_result, list_success = asyncio.run(
        execute_trace_tool(
            agent_id="task-1",
            tool_name="list_dir",
            tool_args_json=_nested("list_dir", {"path": "/tmp/source/task-1"}),
            repo_dir=repo_dir,
            command_timeout_s=5.0,
        )
    )
    assert list_success is True
    assert "📁 pkg" in list_result

    read_result, read_success = asyncio.run(
        execute_trace_tool(
            agent_id="task-1",
            tool_name="read_file",
            tool_args_json=_nested(
                "read_file",
                {"path": "/tmp/source/task-1/pkg/sample.py"},
            ),
            repo_dir=repo_dir,
            command_timeout_s=5.0,
        )
    )
    assert read_success is True
    assert "1| alpha" in read_result
    assert "2| beta" in read_result


def test_execute_trace_tool_supports_replay_observation_style_for_nested_exec(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "notes.txt").write_text("hello\n", encoding="utf-8")

    result, success = asyncio.run(
        execute_trace_tool(
            agent_id="task-1",
            tool_name="exec",
            tool_args_json=_nested(
                "exec",
                {"command": "cd /tmp/source/task-1 && cat notes.txt"},
            ),
            repo_dir=repo_dir,
            command_timeout_s=5.0,
            command_output_style="replay_observation",
        )
    )

    assert success is True
    assert "<returncode>0</returncode>" in result
    assert "hello" in result


def test_simulator_exec_tool_supports_openclaw_write_file(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)

    tool_result, duration_ms, tool_success = asyncio.run(
        _exec_tool(
            "task-1",
            repo_dir,
            "write_file",
            _nested(
                "write_file",
                {"path": "/tmp/source/task-1/out.txt", "content": "payload\n"},
            ),
            5.0,
        )
    )

    assert tool_success is True
    assert duration_ms >= 0.0
    assert "Successfully wrote" in tool_result
    assert (repo_dir / "out.txt").read_text(encoding="utf-8") == "payload\n"


def test_execute_trace_tool_rejects_unsupported_tool(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)

    result, success = asyncio.run(
        execute_trace_tool(
            agent_id="task-1",
            tool_name="nope_tool",
            tool_args_json="{}",
            repo_dir=repo_dir,
            command_timeout_s=5.0,
        )
    )

    assert success is False
    assert "Unsupported replay tool" in result
