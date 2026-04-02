from __future__ import annotations

from agents.code_agent import CodeAgent


def test_code_agent_constructor() -> None:
    """LocalSandbox is always created; no container image needed."""
    agent = CodeAgent(
        agent_id="code-1",
        api_base="http://localhost:8000/v1",
        model="mock",
        repos_root="data/swebench_repos",
    )
    assert agent._container_mgr is not None
    assert agent._container_id is None
    assert agent._prepared is False


def test_code_agent_no_latency_simulator() -> None:
    """ToolLatencySimulator is no longer used in CodeAgent."""
    agent = CodeAgent(
        agent_id="code-2",
        api_base="http://localhost:8000/v1",
        model="mock",
    )
    assert not hasattr(agent, "_latency_sim")


def test_code_agent_default_timeouts() -> None:
    """Verify increased default timeouts for real tool execution."""
    agent = CodeAgent(
        agent_id="code-3",
        api_base="http://localhost:8000/v1",
        model="mock",
    )
    assert agent.command_timeout_s == 120.0
    assert agent.task_timeout_s == 1200.0
