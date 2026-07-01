from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from trace_collect.openclaw_tools import (
    _REPLAY_AGENT_SCRIPT,
    _RESOURCE_AWARE_AGENT_RESPONSE_TIMEOUT_S,
    ContainerAgent,
    execute_trace_tool,
    execute_trace_tool_detailed,
)


def _nested(tool_name: str, payload: dict) -> str:
    return json.dumps({tool_name: payload}, ensure_ascii=False)


class FakeAgent:
    """Minimal ContainerAgent stub that records requests and returns canned responses."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.requests: list[dict] = []
        self.timeouts: list[float] = []
        self._responses = responses or {}
        self._default = {"ok": True, "result": "ok", "returncode": 0}

    async def execute(self, request: dict, *, timeout_s: float = 600.0) -> dict:
        self.requests.append(request)
        self.timeouts.append(timeout_s)
        tool = request.get("tool", "")
        return self._responses.get(tool, self._default)


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, *, kill_error: BaseException | None = None) -> None:
        self.pid = 12345
        self.stdin = _FakeStdin()
        self.returncode: int | None = None
        self.kill_calls = 0
        self.wait_calls = 0
        self._kill_error = kill_error
        self._released = asyncio.Event()

    async def wait(self) -> int:
        self.wait_calls += 1
        await self._released.wait()
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        if self._kill_error is not None:
            self.returncode = 0
            self._released.set()
            raise self._kill_error
        self.returncode = -9
        self._released.set()

    def release(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._released.set()


def test_container_agent_stop_ignores_kill_race(monkeypatch) -> None:
    async def run_stop() -> tuple[_FakeProcess, ContainerAgent]:
        agent = ContainerAgent("cid", "docker")
        process = _FakeProcess(kill_error=ProcessLookupError())
        agent._process = process
        await agent.stop()
        return process, agent

    monkeypatch.setattr("trace_collect.openclaw_tools._AGENT_STOP_GRACE_S", 0.001)
    monkeypatch.setattr("trace_collect.openclaw_tools._AGENT_KILL_WAIT_S", 0.1)

    process, agent = asyncio.run(run_stop())

    assert agent._process is None
    assert process.stdin.closed is True
    assert process.kill_calls == 1
    assert process.returncode == 0


def test_container_agent_stop_kills_timed_out_process(monkeypatch) -> None:
    async def run_stop() -> tuple[_FakeProcess, ContainerAgent]:
        agent = ContainerAgent("cid", "docker")
        process = _FakeProcess()
        agent._process = process
        await agent.stop()
        return process, agent

    monkeypatch.setattr("trace_collect.openclaw_tools._AGENT_STOP_GRACE_S", 0.001)
    monkeypatch.setattr("trace_collect.openclaw_tools._AGENT_KILL_WAIT_S", 0.1)

    process, agent = asyncio.run(run_stop())

    assert agent._process is None
    assert process.stdin.closed is True
    assert process.kill_calls == 1
    assert process.returncode == -9


def test_container_agent_probe_python_picks_first_ge_311(monkeypatch) -> None:
    """ContainerAgent.start probes the container for a Python >=3.11.

    It must NOT hardcode ``python3``: it iterates the candidate list and
    selects the first one that satisfies the version check.  Here the first
    candidate fails (3.10) and the second succeeds (3.11+), so the agent
    must end up running the second candidate — not ``python3``.
    """
    calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.pid = 4242

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        async def wait(self) -> int:
            return self.returncode

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        # Probe invocations look like: <exe> exec -i -w /testbed <cid> <cand> -c <script>
        # The replay-agent start invocation appends _REPLAY_AGENT_SCRIPT.
        candidate = cmd[6] if len(cmd) > 6 else ""
        if cmd[-1] == _REPLAY_AGENT_SCRIPT:
            return _FakeProc(0)  # the actual agent start succeeds
        # First candidate ("/usr/bin/python3") reports 3.10 → fail,
        # second candidate ("/usr/bin/python") reports 3.11 → succeed.
        returncode = 0 if candidate == "/usr/bin/python" else 1
        return _FakeProc(returncode)

    monkeypatch.setattr(
        "trace_collect.openclaw_tools.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    agent = ContainerAgent("cid", "docker")
    asyncio.run(agent.start())

    probe_candidates = [c[6] for c in calls if c[-1] != _REPLAY_AGENT_SCRIPT]
    assert probe_candidates[0] == "/usr/bin/python3"
    assert probe_candidates[1] == "/usr/bin/python"
    assert agent._python_runtime == "/usr/bin/python"


def test_container_agent_probe_python_raises_when_no_ge_311(monkeypatch) -> None:
    """If no candidate satisfies >=3.11, start() raises a clear error."""

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 1  # version too old for every candidate

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        async def wait(self) -> int:
            return self.returncode

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(
        "trace_collect.openclaw_tools.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    agent = ContainerAgent("cid", "docker")
    with pytest.raises(RuntimeError, match="no Python >=3.11"):
        asyncio.run(agent.start())


def test_container_agent_probe_python_kills_timed_out_probe(monkeypatch) -> None:
    """Timed-out probe subprocesses are killed and drained before retrying."""
    calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self, *, hangs: bool) -> None:
            self.returncode: int | None = None
            self.pid = 4242
            self.kill_calls = 0
            self._hangs = hangs

        async def communicate(self) -> tuple[bytes, bytes]:
            if self._hangs and self.kill_calls == 0:
                await asyncio.sleep(60)
            self.returncode = -9 if self.kill_calls else 0
            return b"", b""

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

    procs: list[_FakeProc] = []

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        candidate = cmd[6] if len(cmd) > 6 else ""
        proc = _FakeProc(hangs=candidate == "/usr/bin/python3")
        procs.append(proc)
        return proc

    monkeypatch.setattr(
        "trace_collect.openclaw_tools.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr("trace_collect.openclaw_tools._PYTHON_PROBE_TIMEOUT_S", 0.001)
    monkeypatch.setattr(
        "trace_collect.openclaw_tools._PYTHON_PROBE_KILL_WAIT_S",
        0.1,
    )

    agent = ContainerAgent("cid", "docker")
    selected = asyncio.run(agent._probe_python())

    assert selected == "/usr/bin/python"
    assert calls[0][6] == "/usr/bin/python3"
    assert procs[0].kill_calls == 1


def test_container_agent_probe_python_wraps_kill_failure(monkeypatch) -> None:
    """Probe cleanup kill failures surface as explicit RuntimeError."""

    class _FakeProc:
        returncode: int | None = None
        pid = 4242

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(60)
            return b"", b""

        def kill(self) -> None:
            raise PermissionError("cannot kill")

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(
        "trace_collect.openclaw_tools.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr("trace_collect.openclaw_tools._PYTHON_PROBE_TIMEOUT_S", 0.001)

    agent = ContainerAgent("cid", "docker")
    with pytest.raises(RuntimeError, match="cleanup failed to kill"):
        asyncio.run(agent._probe_python())


def test_exec_command_uses_simulate_timeout_fallback() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["tool"] == "exec"
    assert agent.requests[0]["args"]["command"] == "echo hello"
    assert agent.requests[0]["args"]["timeout"] == 10.0
    assert agent.timeouts == [10.0]


def test_exec_command_simulate_timeout_fallback_is_capped() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello"}),
            command_timeout_s=999.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["args"]["timeout"] == 600.0
    assert agent.timeouts == [600.0]


def test_exec_command_source_timeout_fallback_preserves_source_timeout() -> None:
    agent = FakeAgent({"exec": {"ok": False, "result": "[timeout]", "returncode": 124}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "slow command"}),
            command_timeout_s=600.0,
            source_exec_timeout_s=300.0568,
        )
    )
    assert success is False
    assert "Exit code: 124" in result
    assert agent.requests[0]["args"]["timeout"] == 300.0568
    assert agent.timeouts == [300.0568]


def test_exec_command_passes_source_resource_timeline() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "ok", "returncode": 0}})
    source_resource_timeline = {
        "version": 1,
        "samples": [
            {
                "offset_s": 0.5,
                "dt_s": 0.5,
                "cpu_core_s": 1.0,
                "net_rx_bytes": 128,
            }
        ],
    }

    _result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "pytest", "timeout": 12}),
            command_timeout_s=600.0,
            source_resource_timeline=source_resource_timeline,
        )
    )

    assert success is True
    assert (
        agent.requests[0]["args"]["source_resource_timeline"]
        == source_resource_timeline
    )
    assert agent.timeouts == [_RESOURCE_AWARE_AGENT_RESPONSE_TIMEOUT_S]


def test_execute_trace_tool_detailed_preserves_resource_metadata() -> None:
    agent = FakeAgent(
        {
            "exec": {
                "ok": False,
                "result": "[resource_timeout]",
                "returncode": 124,
                "resource_timeout_policy": "resource_integrated",
                "resource_virtual_time_s": 12.5,
            }
        }
    )

    result, success, _duration, metadata = asyncio.run(
        execute_trace_tool_detailed(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "pytest", "timeout": 12}),
            command_timeout_s=600.0,
            source_resource_timeline={
                "version": 1,
                "samples": [{"offset_s": 0.5, "dt_s": 0.5, "cpu_core_s": 1.0}],
            },
        )
    )

    assert success is False
    assert "Exit code: 124" in result
    assert metadata == {
        "resource_timeout_policy": "resource_integrated",
        "resource_virtual_time_s": 12.5,
    }


def test_commands_do_not_claim_resource_timeline_support() -> None:
    agent = FakeAgent({"commands": {"ok": True, "result": "ok", "returncode": 0}})
    source_resource_timeline = {
        "version": 1,
        "samples": [{"offset_s": 0.5, "dt_s": 0.5, "cpu_core_s": 1.0}],
    }

    _result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"commands": ["pytest"], "timeout": 12}),
            command_timeout_s=600.0,
            source_resource_timeline=source_resource_timeline,
        )
    )

    assert success is True
    assert "source_resource_timeline" not in agent.requests[0]["args"]
    assert agent.timeouts == [12.0]


def test_resource_progress_uses_cpu_and_network_bottleneck() -> None:
    namespace: dict[str, object] = {}
    exec(_REPLAY_AGENT_SCRIPT.split("\nHANDLERS = ", 1)[0], namespace)
    timeline = {
        "version": 1,
        "samples": [
            {
                "dt_s": 1.0,
                "cpu_core_s": 4.0,
                "net_rx_bytes": 2000,
                "net_tx_bytes": 0,
            }
        ],
    }
    samples = namespace["_resource_source_samples"](timeline)

    progress = namespace["_resource_progress_increment"](
        samples,
        0.0,
        1.0,
        {"cpu_core_s": 2.0, "rx_bytes": 500.0, "tx_bytes": 0.0},
    )

    assert progress == 0.25


def test_exec_command_source_timeout_does_not_override_trace_timeout() -> None:
    agent = FakeAgent({"exec": {"ok": False, "result": "[timeout]", "returncode": 124}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "slow command", "timeout": 123}),
            command_timeout_s=600.0,
            source_exec_timeout_s=300.0568,
        )
    )
    assert success is False
    assert "Exit code: 124" in result
    assert agent.requests[0]["args"]["timeout"] == 123.0
    assert agent.timeouts == [123.0]


def test_exec_timeout_from_trace_overrides_simulate_default() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello", "timeout": 123}),
            command_timeout_s=600.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["args"]["timeout"] == 123.0
    assert agent.timeouts == [123.0]


def test_exec_timeout_from_trace_is_capped_like_openclaw() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello", "timeout": 999}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["args"]["timeout"] == 600.0
    assert agent.timeouts == [600.0]


def test_read_file_sends_correct_request() -> None:
    agent = FakeAgent({"read_file": {"ok": True, "result": "file content\n"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="read_file",
            tool_args_json=_nested("read_file", {"path": "/testbed/foo.py"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "file content" in result
    assert agent.requests[0]["tool"] == "read_file"
    assert agent.requests[0]["args"]["path"] == "/testbed/foo.py"


def test_write_file_sends_correct_request() -> None:
    agent = FakeAgent(
        {"write_file": {"ok": True, "result": "Successfully wrote /testbed/out.txt"}}
    )
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="write_file",
            tool_args_json=_nested(
                "write_file", {"path": "/testbed/out.txt", "content": "payload"}
            ),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "Successfully wrote" in result
    assert agent.requests[0]["args"]["content"] == "payload"


def test_edit_file_sends_correct_request() -> None:
    agent = FakeAgent(
        {"edit_file": {"ok": True, "result": "Successfully edited /testbed/x.py"}}
    )
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="edit_file",
            tool_args_json=_nested(
                "edit_file",
                {
                    "path": "/testbed/x.py",
                    "old_text": "foo",
                    "new_text": "bar",
                },
            ),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "Successfully edited" in result
    req = agent.requests[0]
    assert req["tool"] == "edit_file"
    assert req["args"]["old_text"] == "foo"
    assert req["args"]["new_text"] == "bar"
    assert req["args"]["replace_all"] is False


def test_list_dir_sends_correct_request() -> None:
    agent = FakeAgent({"list_dir": {"ok": True, "result": "foo.py\nbar.py"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="list_dir",
            tool_args_json=_nested("list_dir", {"path": "/testbed"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "foo.py" in result


def test_message_tool_replays_as_noop_without_container_request() -> None:
    agent = FakeAgent()
    result, success, duration_ms = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="message",
            tool_args_json=_nested("message", {"content": "done"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert result == "Message replayed as no-op"
    assert duration_ms == 0.0
    assert agent.requests == []


def test_source_runtime_artifact_read_file_fails_without_container_request() -> None:
    agent = FakeAgent()
    result, success, duration_ms = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="read_file",
            tool_args_json=_nested(
                "read_file",
                {
                    "path": (
                        "/root/agent-sched-bench/traces/x/attempt_1/"
                        "openclaw-runtime/tool-results/tool-results/cli_task/out.txt"
                    )
                },
            ),
            command_timeout_s=10.0,
        )
    )
    assert success is False
    assert duration_ms == 0.0
    assert "OpenClaw runtime artifact" in result
    assert agent.requests == []


def test_regular_runtime_tool_results_path_is_not_treated_as_source_artifact() -> None:
    agent = FakeAgent({"read_file": {"ok": True, "result": "normal file"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="read_file",
            tool_args_json=_nested(
                "read_file",
                {"path": "/testbed/pkg/runtime/tool-results/output.txt"},
            ),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert result == "normal file"
    assert (
        agent.requests[0]["args"]["path"]
        == "/testbed/pkg/runtime/tool-results/output.txt"
    )


def test_exec_nonzero_returncode_is_transport_success() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "error msg", "returncode": 1}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "false"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "Exit code: 1" in result


def test_exec_agent_failure_with_non_timeout_returncode_remains_failed() -> None:
    agent = FakeAgent(
        {"exec": {"ok": False, "result": "transport failed", "returncode": 1}}
    )
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "false"}),
            command_timeout_s=10.0,
        )
    )
    assert success is False
    assert "Exit code: 1" in result


def test_exec_zero_returncode_requires_agent_transport_ok() -> None:
    agent = FakeAgent({"exec": {"ok": False, "result": "tool failed", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "true"}),
            command_timeout_s=10.0,
        )
    )
    assert success is False
    assert "Exit code: 0" in result


def test_exec_timeout_remains_failed() -> None:
    agent = FakeAgent({"exec": {"ok": False, "result": "[timeout]", "returncode": 124}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "sleep 999"}),
            command_timeout_s=10.0,
        )
    )
    assert success is False
    assert "Exit code: 124" in result


def test_exec_shell_timeout_exit_code_is_success_when_agent_completed() -> None:
    agent = FakeAgent(
        {"exec": {"ok": True, "result": "command timed out", "returncode": 124}}
    )
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "timeout 5 pytest"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "Exit code: 124" in result


def test_exec_missing_returncode_fails_closed() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "missing rc"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo ok"}),
            command_timeout_s=10.0,
        )
    )
    assert success is False
    assert "Exit code: <missing>" in result


def test_commands_timeout_is_preserved_across_later_success(monkeypatch) -> None:
    namespace: dict[str, object] = {}
    exec(_REPLAY_AGENT_SCRIPT.split("\nHANDLERS = ", 1)[0], namespace)
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(cmd="slow", timeout=1)
        return subprocess.CompletedProcess("fast", 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)

    response = namespace["handle_commands"](
        {"commands": ["slow", "fast"], "timeout": 1}
    )

    assert calls == 2
    assert response["ok"] is False
    assert response["returncode"] == 124
    assert "[timeout]" in response["result"]
    assert "ok" in response["result"]


def test_commands_nonzero_returncode_is_preserved_across_later_success(
    monkeypatch,
) -> None:
    namespace: dict[str, object] = {}
    exec(_REPLAY_AGENT_SCRIPT.split("\nHANDLERS = ", 1)[0], namespace)
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                "fail", 100, stdout="", stderr="apt failed\n"
            )
        return subprocess.CompletedProcess("fast", 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)

    response = namespace["handle_commands"](
        {"commands": ["apt-get update", "echo ok"], "timeout": 1}
    )

    assert calls == 2
    assert response["ok"] is True
    assert response["returncode"] == 100
    assert "apt failed" in response["result"]
    assert "ok" in response["result"]


def test_unsupported_tool_returns_error() -> None:
    agent = FakeAgent()
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="nope_tool",
            tool_args_json="{}",
            command_timeout_s=5.0,
        )
    )
    assert success is False
    assert "Unsupported replay tool" in result


def test_commands_sends_list() -> None:
    agent = FakeAgent({"commands": {"ok": True, "result": "done", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"commands": ["echo a", "echo b"]}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert agent.requests[0]["tool"] == "commands"
    assert agent.requests[0]["args"]["commands"] == ["echo a", "echo b"]
    assert agent.requests[0]["args"]["timeout"] == 10.0
    assert agent.timeouts == [10.0]
