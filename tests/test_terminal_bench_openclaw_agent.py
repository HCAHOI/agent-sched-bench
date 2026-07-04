from __future__ import annotations

import pytest
from pathlib import Path
from types import SimpleNamespace

from agents.openclaw.runtime_deps import (
    OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS,
    OPENCLAW_MCP_RUNTIME_REQUIREMENTS,
)
from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent
from terminal_bench.agents.failure_mode import FailureMode


class StubAgent(TerminalBenchOpenClawAgent):
    @classmethod
    def _build_wheel(cls) -> Path:
        return Path("/tmp/agent_sched_bench-0.1.0-py3-none-any.whl")


def test_install_script_uses_virtualenv() -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    script_path = agent._install_agent_script_path
    content = script_path.read_text(encoding="utf-8")
    assert "python3 -m venv /installed-agent/venv" in content
    for requirement in OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS:
        assert requirement in content
    assert (
        "/installed-agent/venv/bin/python -m pip install --no-deps /installed-agent/agent_sched_bench-0.1.0-py3-none-any.whl"
        in content
    )
    for heavy_dep in (
        "datasets",
        "terminal-bench",
        "trafilatura",
    ):
        assert heavy_dep not in content


def test_install_script_adds_mcp_only_when_configured() -> None:
    plain_agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    mcp_agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        mcp_config_path="/tmp/context7.yaml",
    )

    plain_content = plain_agent._install_agent_script_path.read_text(encoding="utf-8")
    mcp_content = mcp_agent._install_agent_script_path.read_text(encoding="utf-8")

    for requirement in OPENCLAW_MCP_RUNTIME_REQUIREMENTS:
        assert requirement not in plain_content
        assert requirement in mcp_content


def test_run_command_uses_venv_openclaw_and_iteration_limit() -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    commands = agent._run_agent_commands()
    assert len(commands) == 1
    command = commands[0].command
    assert command.startswith(
        'OPENROUTER_API_KEY="$(cat /installed-agent/.openclaw-api-key.fifo)" '
        "/installed-agent/venv/bin/openclaw "
    )
    assert "--max-iterations 25" in command
    assert "--prompt-file /installed-agent/openclaw-prompt.txt" in command
    assert "--prompt " not in command
    assert commands[0].max_timeout_sec == float("inf")


def test_run_command_does_not_embed_task_prompt() -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )

    command = agent._run_agent_commands()[0].command

    assert "sqlite" not in command
    assert "hello task" not in command


def test_run_command_uses_configured_agent_timeout() -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    commands = agent._run_agent_commands()
    assert commands[0].max_timeout_sec == 120.0


def test_agent_rejects_host_local_api_base() -> None:
    with pytest.raises(ValueError, match="local/private OpenAI-compatible"):
        StubAgent(
            model_name="local-model",
            provider_name="openai",
            api_base="http://172.17.0.1:33895/v1",
            api_key="test-key",
            env_key="OPENAI_API_KEY",
            max_iterations=25,
        )


def test_run_command_forwards_mcp_config_to_container() -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        mcp_config_path="/tmp/context7.yaml",
    )
    command = agent._run_agent_commands()[0].command
    assert "--mcp-config /installed-agent/context7.yaml --workspace ." in command


def test_bootstrap_checks_real_venv_creation() -> None:
    command = StubAgent._bootstrap_dependencies_command()
    assert "python3 -m venv --help" not in command
    assert 'python3 -m venv "$probe_root/venv"' in command
    assert '"$probe_root/venv/bin/python" -m pip --version' in command
    assert "python3 python3-pip python3-venv" in command


def test_agent_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in TerminalBenchOpenClawAgent._ENV_PASSTHROUGH:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key=None,
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    assert agent._api_key == "env-key"
    assert agent._secret_exec_environment() == {"OPENCLAW_SECRET_VALUE": "env-key"}
    assert agent._env == {"OPENCLAW_API_BASE": "https://openrouter.ai/api/v1"}
    assert "env-key" not in agent._create_env_setup_file()


def test_perform_task_does_not_embed_api_key_in_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "fake-openrouter-secret"
    agent = StubAgent(
        model_name="qwen/qwen3.7-max",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key=secret,
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    calls: list[object] = []

    class FakeContainer:
        id = "container-id"

        def exec_run(self, cmd, user=None, environment=None, detach=False):
            calls.append(("exec_run", cmd, user, environment, detach))
            return SimpleNamespace(exit_code=0, output=b"")

    class FakeSession:
        def __init__(self) -> None:
            self.container = FakeContainer()
            self._session_name = "agent"

        def copy_to_container(self, paths, container_dir=None):
            calls.append(("copy_to_container", paths, container_dir))

        def send_keys(self, keys, **kwargs):
            calls.append(("send_keys", keys, kwargs))

        def send_command(self, command):
            calls.append(("send_command", command))

        def capture_pane(self, capture_entire=False):
            calls.append(("capture_pane", capture_entire))
            return "installation ok"

    def fake_run(cmd, **kwargs):
        calls.append(("subprocess.run", cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.subprocess.run",
        fake_run,
    )

    result = agent.perform_task(
        "solve sqlite query",
        FakeSession(),
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.NONE
    assert secret not in agent._create_env_setup_file()
    command_texts = [
        repr(call[1])
        for call in calls
        if call[0] in {"exec_run", "send_keys", "send_command", "subprocess.run"}
    ]
    assert command_texts
    assert not any(secret in text for text in command_texts)
    secret_exec_envs = [
        call[3]
        for call in calls
        if call[0] == "exec_run"
        and call[3] == {"OPENCLAW_SECRET_VALUE": secret}
    ]
    assert len(secret_exec_envs) == 1
    send_commands = [call[1].command for call in calls if call[0] == "send_command"]
    assert len(send_commands) == 1
    assert 'OPENROUTER_API_KEY="$(cat /installed-agent/.openclaw-api-key.fifo)"' in (
        send_commands[0]
    )
    assert "--api-base https://openrouter.ai/api/v1" in send_commands[0]


def test_perform_task_cleans_tmux_session_on_agent_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="dummy",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    calls: list[object] = []

    class FakeContainer:
        id = "container-id"

        def exec_run(self, cmd, user=None, environment=None, detach=False):
            calls.append(("exec_run", cmd, user, environment, detach))
            return SimpleNamespace(exit_code=0, output=b"")

    class FakeSession:
        def __init__(self) -> None:
            self.container = FakeContainer()
            self._session_name = "agent"

        def copy_to_container(self, paths, container_dir=None):
            calls.append(("copy_to_container", paths, container_dir))

        def send_keys(self, keys, **kwargs):
            calls.append(("send_keys", keys, kwargs))

        def send_command(self, command):
            calls.append(("send_command", command))
            raise TimeoutError("agent command timed out")

        def capture_pane(self, capture_entire=False):
            calls.append(("capture_pane", capture_entire))
            return "pane output"

    def fake_run(cmd, **kwargs):
        calls.append(("subprocess.run", cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.subprocess.run",
        fake_run,
    )

    result = agent.perform_task(
        "solve sqlite query",
        FakeSession(),
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.AGENT_TIMEOUT
    assert (tmp_path / "openclaw-timeout.marker").read_text(encoding="utf-8") == (
        "timeout\n"
    )
    assert (tmp_path / "openclaw-timeout-pane.txt").read_text(
        encoding="utf-8"
    ) == "pane output"
    assert any(call[0] == "send_keys" and call[1] == ["C-c"] for call in calls)
    cleanup_commands = [
        call[1]
        for call in calls
        if call[0] == "exec_run" and call[1][:2] == ["sh", "-lc"]
    ]
    assert cleanup_commands
    assert "tmux kill-session -t agent" in cleanup_commands[-1][2]
    assert (
        "pkill -TERM -f '/installed-agent/venv/bin/openclaw'"
        in (cleanup_commands[-1][2])
    )
    copied_paths = [
        Path(call[1][2])
        for call in calls
        if call[0] == "subprocess.run" and call[1][:2] == ["docker", "cp"]
    ]
    prompt_paths = [
        path for path in copied_paths if path.name == agent.PROMPT_FILENAME
    ]
    assert len(prompt_paths) == 1
    assert prompt_paths[0].read_text(encoding="utf-8") == "solve sqlite query"
    send_commands = [call[1].command for call in calls if call[0] == "send_command"]
    assert send_commands
    assert "--prompt-file /installed-agent/openclaw-prompt.txt" in send_commands[-1]
    assert "sqlite" not in send_commands[-1]


def test_perform_task_cleans_tmux_session_on_bootstrap_timeout(
    tmp_path: Path,
) -> None:
    agent = StubAgent(
        model_name="z-ai/glm-5.1",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="dummy",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    calls: list[object] = []

    class FakeContainer:
        def exec_run(self, cmd, user=None):
            calls.append(("exec_run", cmd, user))
            if cmd[:1] == ["timeout"] and cmd[2:4] == ["bash", "-lc"]:
                return SimpleNamespace(exit_code=124, output=b"")
            return SimpleNamespace(exit_code=0, output=b"")

    class FakeSession:
        def __init__(self) -> None:
            self.container = FakeContainer()
            self._session_name = "agent"

        def copy_to_container(self, paths, container_dir=None):
            calls.append(("copy_to_container", paths, container_dir))

        def send_keys(self, keys, **kwargs):
            calls.append(("send_keys", keys, kwargs))

        def send_command(self, command):
            calls.append(("send_command", command))

        def capture_pane(self, capture_entire=False):
            calls.append(("capture_pane", capture_entire))
            return "bootstrap pane"

    result = agent.perform_task(
        "solve it",
        FakeSession(),
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.AGENT_TIMEOUT
    assert (tmp_path / "openclaw-timeout.marker").read_text(encoding="utf-8") == (
        "timeout\n"
    )
    assert (tmp_path / "openclaw-timeout-pane.txt").read_text(
        encoding="utf-8"
    ) == "bootstrap pane"
    assert not any(call[0] == "copy_to_container" for call in calls)
    assert not any(
        call[0] == "send_keys"
        and call[1] == ["source /installed-agent/setup-env.sh", "Enter"]
        for call in calls
    )
    cleanup_commands = [
        call[1]
        for call in calls
        if call[0] == "exec_run" and call[1][:2] == ["sh", "-lc"]
    ]
    assert cleanup_commands
    assert "tmux kill-session -t agent" in cleanup_commands[-1][2]
