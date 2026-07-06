"""Targeted tests for OpenClaw tool registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.openclaw._loop import AgentLoop
from agents.openclaw._subagent import SubagentManager
from agents.openclaw.bus.events import InboundMessage
from agents.openclaw.bus.queue import MessageBus


def _fake_provider() -> SimpleNamespace:
    return SimpleNamespace(
        get_default_model=lambda: "qwen-plus-latest",
        generation=SimpleNamespace(max_tokens=1024),
    )


def test_agent_loop_keeps_spawn_for_local_tools(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=Path("/testbed"),
        model="qwen-plus-latest",
    )

    assert loop.tools.has("spawn") is True


def test_agent_loop_registers_sessions_yield_for_local_tools(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=Path("/testbed"),
        model="qwen-plus-latest",
    )

    assert loop.tools.has("sessions_yield") is True


def test_subagent_announcement_preserves_parent_session_key(tmp_path: Path) -> None:
    bus = MessageBus()
    manager = SubagentManager(
        provider=_fake_provider(),
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=1024,
    )

    async def run_test() -> None:
        await manager._announce_result(
            task_id="task-1",
            label="review",
            task="inspect code",
            result="done",
            origin={
                "channel": "eval",
                "chat_id": "case-1",
                "session_key": "eval:case-1",
            },
            status="ok",
        )
        msg = await bus.consume_inbound()
        assert msg.channel == "system"
        assert msg.chat_id == "eval:case-1"
        assert msg.session_key == "eval:case-1"

    import asyncio

    asyncio.run(run_test())


def test_agent_loop_records_error_outcome_when_dispatch_crashes(tmp_path: Path, monkeypatch) -> None:
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=Path("/testbed"),
        model="qwen-plus-latest",
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loop, "_process_message", boom)

    async def run_test() -> None:
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test-chat",
            content="fix bug",
        )
        await loop._dispatch(msg)
        outbound = await bus.consume_outbound()
        assert outbound.content == "Sorry, I encountered an error."
        assert loop._last_run_outcomes[msg.session_key] == {
            "stop_reason": "error",
            "error": "Sorry, I encountered an error.",
        }

    import asyncio

    asyncio.run(run_test())
