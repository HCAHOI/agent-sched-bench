"""Phase 0 regression tests for the openclaw tool-injection extension point.

Pins the contract that ``AgentLoop(tools=<custom>)`` and
``SessionRunner.run(tools=<custom>)`` have REPLACE semantics: when a
caller supplies a pre-built :class:`ToolRegistry`, the default
bash/file/web tool set is NOT registered, and ``trace_metadata``
reflects the actual custom tools rather than the hardcoded openclaw
list. This is the load-bearing invariant for BFCL v4 plugin v2 and any
future ``task_shape='function_call'`` benchmark.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.openclaw._loop import AgentLoop
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.registry import ToolRegistry


# ── Test fixtures ──────────────────────────────────────────────────────


class _FakeBFCLTool(Tool):
    """Minimal Tool subclass that records each call for inspection."""

    def __init__(self, name: str = "fake_bfcl_add") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Adds two numbers (fake BFCL tool)"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return "OK"


class _StubProvider(LLMProvider):
    """Deterministic provider used for constructing AgentLoop in tests."""

    def __init__(self) -> None:
        super().__init__(api_key="test", api_base="http://test")

    def get_default_model(self) -> str:
        return "stub-model"

    async def chat(  # type: ignore[override]
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="c0", name="fake_bfcl_add", arguments={"a": 2, "b": 3}
                )
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


# ── AgentLoop replace semantics (Phase 0A) ────────────────────────────


def test_agent_loop_custom_tool_registry_replaces_defaults(tmp_path: Path) -> None:
    """Passing tools=<custom> to AgentLoop must skip _register_default_tools().

    The resulting agent.tools is the caller's registry verbatim — no
    default openclaw tools (bash/read_file/write_file/etc.) leak in.
    """
    custom = ToolRegistry()
    fake = _FakeBFCLTool()
    custom.register(fake)

    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        tools=custom,
    )

    # Identity: the registry is not copied or replaced.
    assert agent.tools is custom

    # Only the custom tool is present.
    assert agent.tools.tool_names == ["fake_bfcl_add"]

    # None of the default openclaw tools leaked in.
    for default_name in (
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "web_search",
        "web_fetch",
        "send_message",
        "spawn",
    ):
        assert default_name not in agent.tools, (
            f"Default tool {default_name!r} leaked into custom registry"
        )


def test_agent_loop_default_tools_unchanged_when_no_override(
    tmp_path: Path,
) -> None:
    """Regression guard: when tools= is NOT provided, _register_default_tools()
    still runs and the default openclaw tool set is present. This pins the
    zero-behavior-change contract for Phase 0A.
    """
    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
    )
    # A sample of the default tools must be present (exact set depends on
    # which optional subsystems are enabled, so we check the core ones).
    for required_default in ("read_file", "write_file", "edit_file", "list_dir"):
        assert required_default in agent.tools, (
            f"Default tool {required_default!r} missing from default registry"
        )


def test_agent_loop_empty_custom_registry_is_still_replace(tmp_path: Path) -> None:
    """An empty custom ToolRegistry still counts as a custom registry —
    replace semantics are triggered by the presence of the kwarg, not by
    its content. This matters when a plugin wants to forbid all tools
    (e.g., a pure-text benchmark where the LLM must answer without any
    tool calls)."""
    empty = ToolRegistry()
    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        tools=empty,
    )
    assert agent.tools is empty
    assert len(agent.tools) == 0


# ── SessionRunner scaffold_capabilities auto-derivation (Phase 0B) ────


def test_session_runner_trace_metadata_reflects_custom_tools(
    tmp_path: Path,
) -> None:
    """When SessionRunner.run(tools=<custom>) is called, the trace_metadata
    scaffold_capabilities.tools field must list the CUSTOM tool names,
    not the hardcoded openclaw bash/file/web list.
    """
    from agents.openclaw._session_runner import SessionRunner

    custom = ToolRegistry()
    custom.register(_FakeBFCLTool("fake_bfcl_add"))
    custom.register(_FakeBFCLTool("fake_bfcl_sub"))  # second tool, same shape

    provider = _StubProvider()
    runner = SessionRunner(provider, model="stub-model", max_iterations=1)

    trace_file = tmp_path / "trace.jsonl"
    workspace = tmp_path / "ws"
    try:
        asyncio.run(
            runner.run(
                prompt="fake prompt",
                workspace=workspace,
                session_key="eval:test",
                trace_file=trace_file,
                instance_id="test_task",
                tools=custom,
            )
        )
    except Exception:
        # The real session may not complete cleanly under a stubbed
        # provider without MessageTool registered — that's fine. What we
        # care about is that the trace_metadata header was written BEFORE
        # any such failure, with the custom capability list.
        pass

    assert trace_file.exists(), "trace_file must be written even on partial run"
    first_line = trace_file.read_text(encoding="utf-8").splitlines()[0]
    metadata = json.loads(first_line)
    assert metadata["type"] == "trace_metadata"
    assert metadata["trace_format_version"] == 5
    caps = metadata["scaffold_capabilities"]
    # Auto-derived: both custom tools listed, no bash/file_read/etc.
    assert set(caps["tools"]) == {"fake_bfcl_add", "fake_bfcl_sub"}
    assert caps["source"] == "custom_registry"
    # And critically: the hardcoded openclaw bash/file list did NOT leak.
    for default in ("bash", "file_read", "web_search"):
        assert default not in caps["tools"]


def test_session_runner_default_capabilities_unchanged_when_no_tools(
    tmp_path: Path,
) -> None:
    """When SessionRunner.run() is called WITHOUT tools=, the trace_metadata
    scaffold_capabilities block matches the legacy openclaw default
    (bash + file_read/write/edit + web_search + web_fetch + send_message,
    memory=True, skills=True, file_ops='structured'). Zero behavior
    change for the swe_patch path.
    """
    from agents.openclaw._session_runner import SessionRunner

    provider = _StubProvider()
    runner = SessionRunner(provider, model="stub-model", max_iterations=1)

    trace_file = tmp_path / "trace.jsonl"
    workspace = tmp_path / "ws"
    try:
        asyncio.run(
            runner.run(
                prompt="fake prompt",
                workspace=workspace,
                session_key="eval:test",
                trace_file=trace_file,
                instance_id="test_task",
            )
        )
    except Exception:
        pass

    assert trace_file.exists()
    first_line = trace_file.read_text(encoding="utf-8").splitlines()[0]
    metadata = json.loads(first_line)
    caps = metadata["scaffold_capabilities"]
    assert "bash" in caps["tools"]
    assert "file_read" in caps["tools"]
    assert caps["memory"] is True
    assert caps["file_ops"] == "structured"
    # The custom-registry sentinel is NOT present on the default path.
    assert "source" not in caps
