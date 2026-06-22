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


class _GroupingTokenizer:
    """Stub tokenizer whose chat template groups consecutive tool messages.

    Mimics Qwen3-Coder: a run of ``role:tool`` messages is rendered inside ONE
    ``<user>…</user>`` block, each wrapped in its own ``<tool>…</tool>`` marker.
    Rendering an incremental prefix that ends at a non-last tool closes the
    block early, so only the LAST tool of the batch is a text-prefix of the
    full render — reproducing the batched-tool misattribution bug. Tokenizer
    side is byte-identity (one token per char) so segment char spans map 1:1.
    """

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
        i = 0
        while i < len(messages):
            message = messages[i]
            role = message.get("role", "user")
            if role == "tool":
                parts.append("<user>")
                while i < len(messages) and messages[i].get("role") == "tool":
                    content = messages[i].get("content") or ""
                    parts.append(f"<tool>{content}</tool>")
                    i += 1
                parts.append("</user>")
                continue
            parts.append(f"<{role}>")
            content = message.get("content") or ""
            if content:
                parts.append(str(content))
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {})
                parts.append(f"<tool_call>{function.get('name')}()</tool_call>")
            parts.append(f"</{role}>")
            i += 1
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


def _two_tool_messages(*, first_exit: int, second_exit: int) -> list[dict]:
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {
            "role": "assistant",
            "content": "w",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "exec", "arguments": {"command": "a"}},
                },
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": {"command": "b"}},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c0",
            "content": f"FAILMARK out\nExit code: {first_exit}",
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": f"OKMARK out\nExit code: {second_exit}",
        },
    ]


def _segment_for_content(segments: list[dict], prompt_text: str, needle: str) -> dict:
    for segment in segments:
        chunk = prompt_text[int(segment["token_start"]) : int(segment["token_end"])]
        if needle in chunk:
            return segment
    raise AssertionError(f"no segment contains {needle!r}; segments={segments}")


def test_tokenize_chat_with_segments_splits_batched_tool_messages_two_way() -> None:
    tokenizer = _GroupingTokenizer()
    messages = _two_tool_messages(first_exit=1, second_exit=0)
    encoded, segments, prompt_text = tokenize_chat_with_segments(tokenizer, messages)

    assert int(encoded.input_ids.shape[-1]) == len(prompt_text)
    roles = [segment["role"] for segment in segments]
    assert roles.count("tool_result") == 2
    assert "unmatched" not in roles

    fail_seg = _segment_for_content(segments, prompt_text, "FAILMARK")
    ok_seg = _segment_for_content(segments, prompt_text, "OKMARK")
    assert fail_seg["tool_call_id"] == "c0"
    assert ok_seg["tool_call_id"] == "c1"
    assert fail_seg["exit_code"] == 1
    assert fail_seg["tool_error"] is True
    assert ok_seg["exit_code"] == 0
    assert ok_seg["tool_error"] is False
    assert fail_seg["message_index"] == 3
    assert ok_seg["message_index"] == 4

    # system/user/assistant segment boundaries are byte-identical to the
    # pre-fix output (the fix only subdivides the grouped tool region).
    sys_end = len("<system>SYS</system>")
    user_end = sys_end + len("<user>U</user>")
    asst_end = user_end + len("<assistant>w<tool_call>exec()</tool_call><tool_call>exec()</tool_call></assistant>")
    sys_seg = next(s for s in segments if s["role"] == "system")
    user_seg = next(s for s in segments if s["role"] == "user")
    asst_seg = next(s for s in segments if s["role"] == "assistant_call")
    assert (sys_seg["token_start"], sys_seg["token_end"]) == (0, sys_end)
    assert (user_seg["token_start"], user_seg["token_end"]) == (sys_end, user_end)
    assert (asst_seg["token_start"], asst_seg["token_end"]) == (user_end, asst_end)
    # Tool region is contiguous and its outer boundary is unchanged.
    tool_segs = sorted(
        (s for s in segments if s["role"] == "tool_result"),
        key=lambda s: int(s["token_start"]),
    )
    assert tool_segs[0]["token_start"] == asst_end
    assert tool_segs[-1]["token_end"] == len(prompt_text) - len("<assistant>")
    assert tool_segs[0]["token_end"] == tool_segs[1]["token_start"]

    from serving.kv_policies.metadata import build_token_metadata_from_segments

    ids = encoded.input_ids[0].tolist()
    table = build_token_metadata_from_segments(
        segments, input_token_count=len(ids), call_idx=10
    )
    fail_idx = prompt_text.index("FAILMARK")
    ok_idx = prompt_text.index("OKMARK")
    assert table[fail_idx].role == "tool_result"
    assert table[fail_idx].exit_code == 1
    assert table[fail_idx].tool_error is True
    assert table[ok_idx].role == "tool_result"
    assert table[ok_idx].exit_code == 0
    assert table[ok_idx].tool_error is False


def test_tokenize_chat_with_segments_splits_batched_tool_messages_three_way() -> None:
    tokenizer = _GroupingTokenizer()
    messages = _two_tool_messages(first_exit=1, second_exit=0)
    messages[3]["content"] = "FAILMARK out\nExit code: 1"
    messages[4]["content"] = "MIDMARK out\nExit code: 2"
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": "OKMARK out\nExit code: 0",
        }
    )
    messages[2]["tool_calls"].append(
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "exec", "arguments": {"command": "c"}},
        }
    )

    encoded, segments, prompt_text = tokenize_chat_with_segments(tokenizer, messages)
    roles = [segment["role"] for segment in segments]
    assert roles.count("tool_result") == 3
    assert "unmatched" not in roles

    fail_seg = _segment_for_content(segments, prompt_text, "FAILMARK")
    mid_seg = _segment_for_content(segments, prompt_text, "MIDMARK")
    ok_seg = _segment_for_content(segments, prompt_text, "OKMARK")
    assert fail_seg["exit_code"] == 1 and fail_seg["tool_error"] is True
    assert mid_seg["exit_code"] == 2 and mid_seg["tool_error"] is True
    assert ok_seg["exit_code"] == 0 and ok_seg["tool_error"] is False
    assert int(encoded.input_ids.shape[-1]) == len(prompt_text)


def test_tokenize_chat_with_segments_single_tool_unchanged() -> None:
    tokenizer = _GroupingTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {
            "role": "assistant",
            "content": "w",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "exec", "arguments": {"command": "a"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c0",
            "content": "OKMARK out\nExit code: 0",
        },
    ]
    encoded, segments, prompt_text = tokenize_chat_with_segments(tokenizer, messages)
    roles = [segment["role"] for segment in segments]
    assert roles == ["system", "user", "assistant_call", "tool_result", "gen_prompt"]
    tool_seg = _segment_for_content(segments, prompt_text, "OKMARK")
    assert tool_seg["exit_code"] == 0
    assert tool_seg["tool_error"] is False
    assert int(encoded.input_ids.shape[-1]) == len(prompt_text)


class _TrailingAssistantTokenizer(_GroupingTokenizer):
    """Stub whose template leaves a trailing assistant turn OPEN when generating.

    With ``add_generation_prompt=True`` the final assistant turn is not closed,
    so ``render(messages, gen=False)`` (closed) is NOT a text-prefix of the full
    render — the last message misaligns with no aligned closer, exercising the
    trailing-pending branch of :func:`tokenize_chat_with_segments`.
    """

    def apply_chat_template(
        self, messages, tokenize: bool, add_generation_prompt: bool, **_kwargs
    ) -> str:
        if tokenize:
            raise ValueError("test tokenizer only supports tokenize=False")
        parts: list[str] = []
        for i, message in enumerate(messages):
            role = message.get("role", "user")
            parts.append(f"<{role}>")
            content = message.get("content") or ""
            if content:
                parts.append(str(content))
            is_last = i == len(messages) - 1
            if not (is_last and role == "assistant" and add_generation_prompt):
                parts.append(f"</{role}>")
        return "".join(parts)


def test_tokenize_chat_with_segments_trailing_misaligned_keeps_true_role() -> None:
    tokenizer = _TrailingAssistantTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "TRAILMARK"},
    ]
    _encoded, segments, prompt_text = tokenize_chat_with_segments(tokenizer, messages)
    roles = [segment["role"] for segment in segments]
    assert "unmatched" not in roles
    trail_seg = _segment_for_content(segments, prompt_text, "TRAILMARK")
    # The trailing misaligned assistant keeps its true role (not priority-0
    # 'unmatched') even though it has no aligned closer.
    assert trail_seg["role"] == "assistant_message"
    assert trail_seg["message_index"] == 2


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
