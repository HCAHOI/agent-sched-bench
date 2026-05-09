from __future__ import annotations

import asyncio
from typing import Any

from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.registry import ToolRegistry


class _FakeProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(api_key="test", api_base="http://test")
        self.responses = responses
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        self.seen_messages.append([dict(message) for message in messages])
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return "fake-model"


class _ListDirTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return "ok"


def test_runner_reprompts_after_malformed_tool_call_text() -> None:
    asyncio.run(_run_malformed_tool_call_case())


def test_runner_preserves_inline_tool_markup_in_final_answer() -> None:
    asyncio.run(_run_inline_tool_markup_final_case())


async def _run_malformed_tool_call_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content=(
                    "I'll inspect the workspace.\n"
                    "<function=list_dir>\n"
                    "<parameter=path>\n"
                    "/app\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 8},
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_list",
                        name="list_dir",
                        arguments={"path": "/app"},
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 20, "completion_tokens": 4},
            ),
            LLMResponse(
                content="done",
                usage={"prompt_tokens": 30, "completion_tokens": 1},
            ),
        ]
    )
    tool = _ListDirTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Recover the secret."}],
            tools=registry,
            model="fake-model",
            max_iterations=4,
            max_tool_result_chars=1000,
        )
    )

    assert result.final_content == "done"
    assert tool.calls == [{"path": "/app"}]
    assert len(provider.seen_messages) == 3
    retry_messages = provider.seen_messages[1]
    assert retry_messages[-2]["role"] == "assistant"
    assert "<function=list_dir>" in retry_messages[-2]["content"]
    synth_calls = retry_messages[-2].get("tool_calls") or []
    assert len(synth_calls) == 1, "synthetic assistant_call missing"
    synth_id = synth_calls[0]["id"]
    assert synth_id.startswith("malformed_retry_")
    assert synth_calls[0]["function"]["name"] == "_invalid_tool_call"
    assert retry_messages[-1]["role"] == "tool"
    assert retry_messages[-1]["tool_call_id"] == synth_id
    assert retry_messages[-1]["name"] == "_invalid_tool_call"
    assert "not valid and no tool was executed" in retry_messages[-1]["content"]
    assert "<tool_call>" in retry_messages[-1]["content"]
    assert "list_dir" in retry_messages[-1]["content"]


async def _run_inline_tool_markup_final_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content=(
                    "The literal text `<function=list_dir>` is malformed tool-call "
                    "markup, not an action I need to execute."
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 8},
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(_ListDirTool())

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Explain this markup."}],
            tools=registry,
            model="fake-model",
            max_iterations=4,
            max_tool_result_chars=1000,
        )
    )

    assert result.final_content == (
        "The literal text `<function=list_dir>` is malformed tool-call "
        "markup, not an action I need to execute."
    )
    assert len(provider.seen_messages) == 1
