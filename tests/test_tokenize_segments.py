from __future__ import annotations

import asyncio
import json
import logging
import socket
import struct
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

import torch

from llm_call.provider_base import LLMResponse
from serving.recording.backend_hf import (
    HFRecordingProvider,
    HFRecordingServer,
    parse_text_tool_calls,
    tokenize_chat_with_segments,
)


class _FakeEncoding(dict):
    @property
    def input_ids(self) -> torch.Tensor:
        return self["input_ids"]


class _PrefixTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs,
    ) -> str:
        if tokenize:
            raise ValueError("test tokenizer only supports tokenize=False")
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            parts.append(f"<{role}>")
            content = message.get("content") or ""
            if content:
                parts.append(str(content))
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {})
                parts.append(f"<tool_call>{function.get('name')}()</tool_call>")
            parts.append(f"</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> _FakeEncoding:
        del return_tensors
        if add_special_tokens:
            raise ValueError("test expects add_special_tokens=False")
        encoded = _FakeEncoding(
            {"input_ids": torch.arange(len(text), dtype=torch.long).reshape(1, -1)}
        )
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor(
                [[(idx, idx + 1) for idx in range(len(text))]],
                dtype=torch.long,
            )
        return encoded


def test_tokenize_chat_with_segments_labels_tool_call_message() -> None:
    tokenizer = _PrefixTokenizer()
    encoded, segments, prompt_text = tokenize_chat_with_segments(
        tokenizer,
        [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "Find x."},
            {
                "role": "assistant",
                "content": "I will search.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "x"}',
                        },
                    }
                ],
            },
        ],
    )

    assert int(encoded.input_ids.shape[-1]) == len(prompt_text)
    roles = [segment["role"] for segment in segments]
    assert roles == ["system", "user", "assistant_call", "gen_prompt"]
    assistant_segment = segments[2]
    assert assistant_segment["has_content"] is True
    assert assistant_segment["has_tool_calls"] is True
    assert all(segment["token_start"] < segment["token_end"] for segment in segments)


def test_tokenize_chat_with_segments_records_first_seen_call() -> None:
    tokenizer = _PrefixTokenizer()
    _encoded, segments, _prompt_text = tokenize_chat_with_segments(
        tokenizer,
        [
            {"role": "user", "content": "Find x."},
            {
                "role": "tool",
                "content": "x=1",
                "tool_call_id": "call_1",
                "name": "search",
            },
        ],
        first_seen_call_by_message_index={0: 0, 1: 3},
        default_first_seen_call=4,
    )

    assert [segment["first_seen_call"] for segment in segments] == [0, 3, 4]
    assert [segment["first_seen_call_inferred"] for segment in segments] == [
        False,
        False,
        False,
    ]
    assert segments[1]["role"] == "tool_result"
    assert segments[1]["tool_call_id"] == "call_1"
    assert segments[1]["name"] == "search"


def test_hf_recording_provider_rejects_forced_tool_choice_before_generation() -> None:
    provider = HFRecordingProvider.__new__(HFRecordingProvider)

    response = asyncio.run(
        provider.chat(
            messages=[],
            tool_choice={"type": "function", "function": {"name": "save_memory"}},
        )
    )

    assert response.finish_reason == "error"
    assert "tool_choice" in (response.content or "")


def test_parse_text_tool_calls_normalizes_function_wrapped_names() -> None:
    _content, calls = parse_text_tool_calls(
        "<tool_call>\n"
        "<function=<function=exec>>\n"
        "<parameter=command>\n"
        "ls -la /app\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert calls[0].name == "exec"
    assert calls[0].arguments == {"command": "ls -la /app"}


def test_parse_text_tool_calls_preserves_plain_text_command_starting_with_t() -> None:
    _content, calls = parse_text_tool_calls(
        "<tool_call>\n"
        "<function=exec>\n"
        "<parameter=command>\n"
        "tesseract /app/code.png stdout\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert calls[0].arguments == {"command": "tesseract /app/code.png stdout"}


class _ReasoningAwareProvider:
    default_model = "toy"

    async def chat(self, **kwargs):
        if kwargs.get("reasoning_effort") is not None:
            return LLMResponse(
                content="Error: unsupported reasoning_effort",
                finish_reason="error",
                usage={},
            )
        return LLMResponse(content="ok", finish_reason="stop", usage={})


class _DelayedProvider:
    default_model = "toy"

    def __init__(self, *, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    async def chat(self, **_kwargs: Any) -> LLMResponse:
        self.started.set()
        await asyncio.to_thread(self.release.wait, 5.0)
        if self.fail:
            raise RuntimeError("provider boom")
        return LLMResponse(content="ok", finish_reason="stop", usage={})


def _send_chat_request_and_reset_client(
    api_base: str,
    provider: _DelayedProvider,
) -> None:
    parsed = urllib.parse.urlparse(api_base)
    assert parsed.hostname is not None
    assert parsed.port is not None
    payload = json.dumps(
        {"model": "toy", "messages": [{"role": "user", "content": "hi"}]}
    ).encode("utf-8")
    request = (
        f"POST {parsed.path}/chat/completions HTTP/1.1\r\n"
        f"Host: {parsed.hostname}:{parsed.port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii") + payload

    sock = socket.create_connection((parsed.hostname, parsed.port), timeout=5)
    try:
        sock.sendall(request)
        assert provider.started.wait(5)
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
    finally:
        sock.close()


def _wait_for_log(caplog: Any, needle: str) -> bool:
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if needle in caplog.text:
            return True
        time.sleep(0.05)
    return needle in caplog.text


def test_hf_recording_server_forwards_reasoning_effort() -> None:
    with HFRecordingServer(_ReasoningAwareProvider()) as server:
        payload = json.dumps(
            {
                "model": "toy",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{server.api_base}/chat/completions",
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))

    choice = body["choices"][0]
    assert choice["finish_reason"] == "error"
    assert "reasoning_effort" in choice["message"]["content"]


def test_hf_recording_server_ignores_client_disconnect_during_response(
    capsys: Any, caplog: Any
) -> None:
    provider = _DelayedProvider()
    caplog.set_level(logging.DEBUG, logger="serving.recording.backend_hf")

    with HFRecordingServer(provider) as server:
        _send_chat_request_and_reset_client(server.api_base, provider)
        provider.release.set()
        assert _wait_for_log(caplog, "client disconnected while writing HTTP response")

        healthy_payload = json.dumps(
            {"model": "toy", "messages": [{"role": "user", "content": "again"}]}
        ).encode("utf-8")
        healthy_request = urllib.request.Request(
            f"{server.api_base}/chat/completions",
            data=healthy_payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(healthy_request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))

    captured = capsys.readouterr()
    assert body["choices"][0]["message"]["content"] == "ok"
    assert "chat handler raised" not in captured.err
    assert "BrokenPipeError" not in captured.err


def test_hf_recording_server_ignores_client_disconnect_during_error_response(
    capsys: Any, caplog: Any
) -> None:
    provider = _DelayedProvider(fail=True)
    caplog.set_level(logging.DEBUG, logger="serving.recording.backend_hf")

    with HFRecordingServer(provider) as server:
        _send_chat_request_and_reset_client(server.api_base, provider)
        provider.release.set()
        assert _wait_for_log(
            caplog, "client disconnected while writing HTTP error response"
        )

        provider.fail = False
        healthy_payload = json.dumps(
            {"model": "toy", "messages": [{"role": "user", "content": "again"}]}
        ).encode("utf-8")
        healthy_request = urllib.request.Request(
            f"{server.api_base}/chat/completions",
            data=healthy_payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(healthy_request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))

    captured = capsys.readouterr()
    assert body["choices"][0]["message"]["content"] == "ok"
    assert "provider boom" in captured.err
    assert "BrokenPipeError" not in captured.err
