from __future__ import annotations

from pathlib import Path

from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent


class StubAgent(TerminalBenchOpenClawAgent):
    @classmethod
    def _build_wheel(cls) -> Path:
        return Path('/tmp/agent_sched_bench-0.1.0-py3-none-any.whl')


def test_install_script_uses_virtualenv() -> None:
    agent = StubAgent(
        model_name='nvidia/nemotron-3-super-120b-a12b:free',
        provider_name='openrouter',
        api_base='https://openrouter.ai/api/v1',
        api_key='test-key',
        env_key='OPENROUTER_API_KEY',
        max_iterations=25,
    )
    script_path = agent._install_agent_script_path
    content = script_path.read_text(encoding='utf-8')
    assert 'python3 -m venv /installed-agent/venv' in content
    assert '/installed-agent/venv/bin/python -m pip install /installed-agent/agent_sched_bench-0.1.0-py3-none-any.whl' in content


def test_run_command_uses_venv_openclaw_and_iteration_limit() -> None:
    agent = StubAgent(
        model_name='nvidia/nemotron-3-super-120b-a12b:free',
        provider_name='openrouter',
        api_base='https://openrouter.ai/api/v1',
        api_key='test-key',
        env_key='OPENROUTER_API_KEY',
        max_iterations=25,
    )
    commands = agent._run_agent_commands('hello task')
    assert len(commands) == 1
    command = commands[0].command
    assert command.startswith('/installed-agent/venv/bin/openclaw ')
    assert '--max-iterations 25' in command
