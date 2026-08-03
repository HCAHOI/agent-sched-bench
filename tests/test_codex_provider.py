"""Codex subscription provider contract checks."""

from __future__ import annotations
import asyncio

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import llm_call.codex as codex
from llm_call.codex import CodexProvider, _responses_input, load_codex_credentials
from llm_call.codex import _consume_stream
from llm_call.providers import create_provider


def test_codex_provider_loads_login_and_converts_tool_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "header.payload.signature",
                    "account_id": "account-123",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CODEX_ACCOUNT_ID", raising=False)

    credentials = load_codex_credentials()
    provider = create_provider(
        provider_name="codex",
        api_key=None,
        api_base="https://chatgpt.com/backend-api/codex",
        default_model="gpt-5.5",
    )
    instructions, items = _responses_input(
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Read it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
        ]
    )

    assert credentials.access_token == "header.payload.signature"
    assert credentials.account_id == "account-123"
    assert isinstance(provider, CodexProvider)
    assert instructions == "Use tools."
    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Read it."}],
        },
        {
            "type": "function_call",
            "name": "read",
            "arguments": '{"path":"a"}',
            "call_id": "call-1",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "contents"},
    ]


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_tokens", 1024),
        ("temperature", 0.1),
        ("top_p", 0.9),
        ("top_k", 20),
        ("repetition_penalty", 1.1),
    ],
)
def test_codex_provider_rejects_unsupported_generation_controls(
    setting: str, value: int | float
) -> None:
    with pytest.raises(ValueError, match=setting):
        CodexProvider(
            api_key="token",
            api_base=None,
            default_model="gpt-5.6-sol",
            **{setting: value},
        )


def test_codex_stream_requires_completed_event() -> None:
    async def stream(*events: object):
        for event in events:
            yield event

    delta = SimpleNamespace(type="response.output_text.delta", delta="partial")
    refusal = SimpleNamespace(type="response.refusal.delta", delta="refused")
    completed = SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(usage=None, id="response-1", status="completed"),
    )

    truncated = asyncio.run(_consume_stream(stream(delta), None))
    success = asyncio.run(_consume_stream(stream(delta, completed), None))
    refused = asyncio.run(_consume_stream(stream(refusal, completed), None))

    assert truncated.finish_reason == "error"
    assert success.finish_reason == "stop"
    assert success.content == "partial"
    assert refused.finish_reason == "stop"
    assert refused.content == "refused"


def test_codex_fast_tier_maps_to_priority_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, object] = {}

    async def stream():
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=None,
                id="response-1",
                model="gpt-5.6-sol",
                service_tier="priority",
                status="completed",
            ),
        )

    class FakeResponses:
        async def create(self, **kwargs):
            request.update(kwargs)
            return stream()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

        async def close(self):
            pass

    monkeypatch.setattr(codex, "AsyncOpenAI", FakeClient)
    provider = CodexProvider(
        api_key="test-token",
        api_base=None,
        default_model="gpt-5.6-sol",
        service_tier="fast",
    )

    response = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

    assert request["service_tier"] == "priority"
    assert response.extra["codex_metadata"]["service_tier"] == "priority"


def test_codex_backend_retry_message_is_transient() -> None:
    assert CodexProvider._is_transient_error(
        "An error occurred while processing your request. "
        "You can retry your request."
    )
