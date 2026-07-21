from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent
from terminal_bench.agents.failure_mode import FailureMode


def make_agent(**overrides: Any) -> TerminalBenchOpenClawAgent:
    config: dict[str, Any] = {
        "model_name": "z-ai/glm-5.1",
        "provider_name": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "env_key": "OPENROUTER_API_KEY",
        "max_iterations": 25,
    }
    config.update(overrides)
    return TerminalBenchOpenClawAgent(**config)


def test_install_script_is_noop_for_host_controller() -> None:
    agent = make_agent()

    script = agent._install_agent_script_path.read_text(encoding="utf-8")

    assert "OpenClaw runs on the host" in script
    assert "pip install" not in script
    assert "agent_sched_bench" not in script
    assert "OPENROUTER_API_KEY" not in script


def test_run_agent_commands_are_empty_for_host_controller() -> None:
    agent = make_agent()

    assert agent._run_agent_commands("solve sqlite query") == []
    assert agent._env == {}


def test_agent_rejects_host_local_api_base() -> None:
    with pytest.raises(ValueError, match="local/private OpenAI-compatible"):
        make_agent(
            model_name="local-model",
            provider_name="openai",
            api_base="http://172.17.0.1:33895/v1",
            api_key="test-key",
            env_key="OPENAI_API_KEY",
        )


def test_bootstrap_checks_real_modern_venv_creation() -> None:
    command = TerminalBenchOpenClawAgent._bootstrap_dependencies_command()

    assert "python3 -m venv --help" not in command
    assert '"$1" -m venv "$probe_root/venv"' in command
    assert '"$probe_root/venv/bin/python" -m pip --version' in command
    assert "python3 python3-pip python3-venv curl ca-certificates" in command
    assert "sys.version_info >= (3, 11)" in command
    assert "for candidate in python3 python3.13 python3.12 python3.11" in command
    assert "supported_python=$(find_supported_python)" in command
    assert "if install_python_deps; then" in command
    assert "/installed-agent/uv/uv python install 3.12" in command
    assert "/installed-agent/python/bin/python3" in command


def test_bridge_bootstrap_selects_stdlib_python() -> None:
    command = TerminalBenchOpenClawAgent._bootstrap_bridge_python_command()

    assert "venv_ready" not in command
    assert "pip --version" not in command
    assert "python3 curl ca-certificates" in command
    assert "sys.version_info >= (3, 6)" in command
    assert "sys.version_info >= (3, 11)" not in command
    assert "/installed-agent/uv/uv python install 3.12" not in command
    assert "/installed-agent/python/bin/python3" in command


def test_bridge_bootstrap_timeout_is_configurable() -> None:
    agent = make_agent(bridge_bootstrap_timeout_sec="12.5")

    assert agent._bridge_bootstrap_timeout_sec == 12.5
    with pytest.raises(ValueError, match="bridge_bootstrap_timeout_sec"):
        make_agent(bridge_bootstrap_timeout_sec=0)


def test_agent_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    agent = make_agent(api_key=None)

    assert agent._api_key == "env-key"
    assert agent._env == {}
    assert "env-key" not in agent._install_agent_script_path.read_text(encoding="utf-8")


def test_perform_task_runs_host_session_with_container_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeContainer:
        id = "abcdef1234567890"

        def exec_run(self, cmd: list[str], user: str | None = None) -> SimpleNamespace:
            calls.append(("exec_run", {"cmd": cmd, "user": user}))
            if cmd[:5] == ["tmux", "display-message", "-p", "-t", "agent"]:
                return SimpleNamespace(exit_code=0, output=b"/workdir\n")
            if cmd[:2] == ["bash", "-lc"]:
                return SimpleNamespace(exit_code=0, output=b"bootstrapped")
            raise AssertionError(f"unexpected container exec: {cmd!r}")

    class FakeSession:
        container = FakeContainer()
        _session_name = "agent"

    class FakeContainerAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            *,
            workdir: str,
        ) -> None:
            calls.append(
                (
                    "container_agent_init",
                    {
                        "container_id": container_id,
                        "container_executable": container_executable,
                        "workdir": workdir,
                    },
                )
            )

        async def start(self) -> None:
            calls.append(("container_agent_start", None))

        async def stop(self) -> None:
            calls.append(("container_agent_stop", None))

    class FakeProvider:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("provider_init", kwargs))

    class FakeSessionRunner:
        def __init__(self, provider: FakeProvider, **kwargs: Any) -> None:
            calls.append(("runner_init", kwargs))

        async def run(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(("runner_run", kwargs))
            trace_file = Path(kwargs["trace_file"])
            trace_file.write_text(
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "status": "started",
                        "prompt_runtime_label": kwargs["runtime_label"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(stop_reason="completed", error=None)

    async def fake_runtime_proof(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("runtime_proof", kwargs))
        return {
            "agent_execution_environment": "host",
            "tool_execution_environment": "task_container",
            "tool_container_id": "abcdef1234567890",
            "tool_container_user": "root",
            "tool_container_user_id": 0,
            "tool_container_workdir": "/workdir",
            "tool_container_os": "Linux",
            "tool_container_arch": "x86_64",
            "tool_container_python": "Python 3.6.9",
            "tool_runtime": "ContainerAgent",
        }

    def fake_build_tools(*args: Any, **kwargs: Any) -> list[str]:
        calls.append(("build_tools", kwargs))
        return ["container-tool"]

    monkeypatch.setattr("agents.terminal_bench.openclaw_agent.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.ContainerAgent",
        FakeContainerAgent,
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.create_provider",
        lambda **kwargs: FakeProvider(**kwargs),
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.SessionRunner",
        FakeSessionRunner,
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.container_runtime_proof",
        fake_runtime_proof,
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.build_container_tools_for_agent",
        fake_build_tools,
    )

    result = make_agent(api_key="secret-value").perform_task(
        "solve sqlite query",
        FakeSession(),  # type: ignore[arg-type]
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.NONE
    assert (tmp_path / "openclaw-complete.marker").read_text(encoding="utf-8") == (
        "completed\n"
    )
    trace_records = [
        json.loads(line)
        for line in (tmp_path / "openclaw-trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert trace_records[0]["openclaw_runtime"] == "host_session_runner"
    assert trace_records[0]["terminal_bench_container_id"] == "abcdef1234567890"
    assert trace_records[0]["terminal_bench_workdir"] == "/workdir"
    assert trace_records[0]["tool_container_python"] == "Python 3.6.9"
    assert "Shell/file tools `python3`: Python 3.6.9" in trace_records[0][
        "prompt_runtime_label"
    ]

    provider_kwargs = next(value for name, value in calls if name == "provider_init")
    assert provider_kwargs["api_key"] == "secret-value"
    assert provider_kwargs["api_base"] == "https://openrouter.ai/api/v1"
    assert provider_kwargs["default_model"] == "z-ai/glm-5.1"

    runner_kwargs = next(value for name, value in calls if name == "runner_init")
    assert runner_kwargs["model"] == "z-ai/glm-5.1"
    assert runner_kwargs["max_iterations"] == 25
    assert runner_kwargs["tool_overrides"] == ["container-tool"]

    run_kwargs = next(value for name, value in calls if name == "runner_run")
    assert run_kwargs["prompt"] == "solve sqlite query"
    assert run_kwargs["tool_workspace"] == Path("/workdir")
    assert run_kwargs["project_workspace"] == Path("/workdir")
    assert run_kwargs["session_key"] == "terminal-bench:abcdef123456"
    assert run_kwargs["channel"] == "terminal-bench"
    assert "Shell/file tools runtime: Linux x86_64" in run_kwargs["runtime_label"]
    assert "Shell/file tools `python3`: Python 3.6.9" in run_kwargs[
        "runtime_label"
    ]

    assert ("container_agent_start", None) in calls
    assert ("container_agent_stop", None) in calls
    rendered_calls = repr(calls)
    assert "secret-value" in repr(provider_kwargs)
    assert "secret-value" not in rendered_calls.replace(repr(provider_kwargs), "")

    container_execs = [value for name, value in calls if name == "exec_run"]
    assert not any(
        exec_call["cmd"][:2] == ["bash", "-lc"] for exec_call in container_execs
    )


def test_bridge_python_bootstrap_retries_after_container_agent_probe_failure() -> None:
    calls: list[list[str]] = []

    class FakeContainer:
        def exec_run(self, cmd: list[str], user: str | None = None) -> SimpleNamespace:
            assert user == "root"
            calls.append(cmd)
            if cmd[:4] == ["timeout", "--kill-after=5s", "3600s", "bash"]:
                return SimpleNamespace(exit_code=0, output=b"bootstrapped")
            raise AssertionError(f"unexpected container exec: {cmd!r}")

    class FakeSession:
        container = FakeContainer()

    class FakeAgent:
        def __init__(self) -> None:
            self.starts = 0

        async def start(self) -> None:
            self.starts += 1
            if self.starts == 1:
                raise RuntimeError("ContainerAgent: no Python >=3.6 found")

    fake_agent = FakeAgent()

    asyncio.run(
        make_agent()._start_container_agent(
            fake_agent,  # type: ignore[arg-type]
            session=FakeSession(),  # type: ignore[arg-type]
            deadline=None,
        )
    )

    assert fake_agent.starts == 2
    assert calls[0][:4] == ["timeout", "--kill-after=5s", "3600s", "bash"]
    assert "sys.version_info >= (3, 6)" in calls[0][-1]
    assert "sys.version_info >= (3, 11)" not in calls[0][-1]


def test_bridge_python_bootstrap_container_timeout_is_agent_timeout() -> None:
    calls: list[list[str]] = []

    class FakeContainer:
        def exec_run(self, cmd: list[str], user: str | None = None) -> SimpleNamespace:
            assert user == "root"
            calls.append(cmd)
            return SimpleNamespace(exit_code=124, output=b"timeout")

    class FakeSession:
        container = FakeContainer()

    with pytest.raises(TimeoutError, match="host-controller deadline exceeded"):
        make_agent()._bootstrap_container_bridge_python(
            FakeSession(),  # type: ignore[arg-type]
            timeout_s=3.2,
        )

    assert calls[0][:4] == ["timeout", "--kill-after=5s", "3.2s", "bash"]


def test_bridge_bootstrap_timeout_caps_agent_deadline() -> None:
    configured_timeout_s = 12.5
    remaining_agent_deadline_s = 7200.0
    calls: list[list[str]] = []

    class FakeContainer:
        def exec_run(self, cmd: list[str], user: str | None = None) -> SimpleNamespace:
            assert user == "root"
            calls.append(cmd)
            return SimpleNamespace(exit_code=0, output=b"bootstrapped")

    class FakeSession:
        container = FakeContainer()

    make_agent(
        bridge_bootstrap_timeout_sec=configured_timeout_s
    )._bootstrap_container_bridge_python(
        FakeSession(),  # type: ignore[arg-type]
        timeout_s=remaining_agent_deadline_s,
    )

    assert calls[0][:4] == [
        "timeout",
        "--kill-after=5s",
        f"{configured_timeout_s:g}s",
        "bash",
    ]


def test_bridge_bootstrap_timeout_format_never_rounds_up() -> None:
    assert TerminalBenchOpenClawAgent._format_timeout_s(1.0000006) == "1s"


def test_perform_task_reports_failed_host_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeContainer:
        id = "container-id"

        def exec_run(self, cmd: list[str], user: str | None = None) -> SimpleNamespace:
            if cmd[:5] == ["tmux", "display-message", "-p", "-t", "agent"]:
                return SimpleNamespace(exit_code=0, output=b"/workdir\n")
            if cmd[:2] == ["bash", "-lc"]:
                return SimpleNamespace(exit_code=0, output=b"bootstrapped")
            raise AssertionError(f"unexpected container exec: {cmd!r}")

    class FakeSession:
        container = FakeContainer()
        _session_name = "agent"

    class FakeContainerAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class FakeSessionRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(stop_reason="max_iterations", error=None)

    async def fake_runtime_proof(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr("agents.terminal_bench.openclaw_agent.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.ContainerAgent",
        FakeContainerAgent,
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.SessionRunner",
        FakeSessionRunner,
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.container_runtime_proof",
        fake_runtime_proof,
    )
    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.build_container_tools_for_agent",
        lambda *args, **kwargs: [],
    )

    result = make_agent().perform_task(
        "solve it",
        FakeSession(),  # type: ignore[arg-type]
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.UNKNOWN_AGENT_ERROR
    assert "stop_reason='max_iterations'" in (tmp_path / "openclaw-error.txt").read_text(
        encoding="utf-8"
    )
