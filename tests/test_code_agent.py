from __future__ import annotations

from agents.mini_swe_code_agent import MiniSWECodeAgent


def test_constructor_defaults() -> None:
    agent = MiniSWECodeAgent(
        agent_id="code-1",
        api_base="http://localhost:8000/v1",
        model="mock",
    )
    assert agent._workdir is None
    assert agent._prepared is False
    assert not hasattr(agent, "_container_mgr")
    assert not hasattr(agent, "_latency_sim")


def test_constructor_with_repos_root() -> None:
    from pathlib import Path
    agent = MiniSWECodeAgent(
        agent_id="code-2",
        api_base="http://localhost:8000/v1",
        model="mock",
        repos_root="data/swebench_repos",
    )
    assert agent.repos_root == Path("data/swebench_repos")


def test_default_timeouts() -> None:
    agent = MiniSWECodeAgent(
        agent_id="code-3",
        api_base="http://localhost:8000/v1",
        model="mock",
    )
    assert agent.command_timeout_s == 120.0
    assert agent.task_timeout_s == 1200.0
    assert agent.max_steps == 60
