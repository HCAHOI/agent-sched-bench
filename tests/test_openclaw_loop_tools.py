"""Targeted tests for OpenClaw tool registration across runtimes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.openclaw._loop import AgentLoop
from agents.openclaw.bus.queue import MessageBus


def _fake_provider() -> SimpleNamespace:
    return SimpleNamespace(
        get_default_model=lambda: "qwen-plus-latest",
        generation=SimpleNamespace(max_tokens=1024),
    )


def test_agent_loop_disables_spawn_for_host_container_backend(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=tmp_path / "tool",
        model="qwen-plus-latest",
        container_workspace=SimpleNamespace(cwd="/testbed"),
    )

    assert loop.tools.has("spawn") is False


def test_agent_loop_keeps_spawn_for_local_tools(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=Path("/testbed"),
        model="qwen-plus-latest",
        container_workspace=None,
    )

    assert loop.tools.has("spawn") is True
