"""Unit tests for the de-tokenization helpers that are prone to silent
mis-attribution: the message normalization copied from backend_hf, the
recording<->trace matcher, and the bit-exact alignment gate.

All helpers under test are tokenizer/torch-free (they take plain dicts), so this
runs in the local torch-less .venv. The one json_repair-dependent path is guarded
with importorskip.
"""
from __future__ import annotations

import pytest

from scripts.recoding_figures.detokenize_selected_blocks import (
    _alignment_ok,
    _match_messages,
    _normalize_messages,
    _normalize_tool_arguments,
    _sanitize_empty_content,
    _token_index_for_char,
)


def test_normalize_tool_arguments_dict_passthrough_and_nondict():
    assert _normalize_tool_arguments({"a": 1}) == {"a": 1}
    assert _normalize_tool_arguments(123) == {}
    assert _normalize_tool_arguments(None) == {}


def test_normalize_tool_arguments_string_is_parsed():
    pytest.importorskip("json_repair")
    assert _normalize_tool_arguments('{"path": "x.py"}') == {"path": "x.py"}


def test_sanitize_empty_content_rules():
    out = _sanitize_empty_content([{"role": "user", "content": ""}])
    assert out[0]["content"] == "(empty)"
    # assistant with tool_calls and empty content -> None (not "(empty)")
    out = _sanitize_empty_content(
        [{"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]}]
    )
    assert out[0]["content"] is None
    # empty text blocks stripped from a list content
    out = _sanitize_empty_content(
        [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}]}]
    )
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]
    # dict content wrapped in a list
    out = _sanitize_empty_content([{"role": "user", "content": {"type": "text", "text": "x"}}])
    assert out[0]["content"] == [{"type": "text", "text": "x"}]


def test_normalize_messages_restricts_keys_and_assistant_content_default():
    out = _normalize_messages(
        [{"role": "user", "content": "hi", "extra": "drop", "_openclaw_message_id": 7}]
    )
    assert out == [{"role": "user", "content": "hi"}]
    # assistant turn with only tool_calls gets an explicit content=None
    out = _normalize_messages([{"role": "assistant", "tool_calls": []}])
    assert out[0]["role"] == "assistant" and out[0]["content"] is None


def test_normalize_messages_coerces_tool_call_arguments():
    pytest.importorskip("json_repair")
    out = _normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
                ],
            }
        ]
    )
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}


def test_token_index_for_char_bisect():
    starts = [0, 2, 5, 6]
    assert _token_index_for_char(starts, 0) == 0
    assert _token_index_for_char(starts, 5) == 2
    assert _token_index_for_char(starts, 9) == 4  # past end


def _grid():
    # 4 prompt tokens with char spans; starts = [0,2,5,6]
    offsets = [(0, 2), (2, 5), (5, 6), (6, 9)]
    segments = [
        {"role": "system", "char_start": 0, "char_end": 5, "token_start": 0, "token_end": 2},
        {"role": "user", "char_start": 5, "char_end": 9, "token_start": 2, "token_end": 4},
        {"role": "generation", "token_start": 4, "token_end": 5},  # no char span -> skipped
    ]
    return offsets, segments


def test_alignment_ok_matches_and_skips_generation_segment():
    offsets, segments = _grid()
    ok, mism = _alignment_ok(offsets, segments, input_tokens=4)
    assert ok is True and mism == 0


def test_alignment_ok_length_mismatch():
    offsets, segments = _grid()
    ok, mism = _alignment_ok(offsets, segments, input_tokens=5)
    assert ok is False and mism == -1


def test_alignment_ok_boundary_mismatch():
    offsets, segments = _grid()
    segments[0]["token_end"] = 3  # wrong: char_end=5 -> token 2, not 3
    ok, mism = _alignment_ok(offsets, segments, input_tokens=4)
    assert ok is False and mism >= 1


def test_match_messages_prefers_trace_action_id():
    calls = [
        {"action_id": "llm_0", "messages_in": [{"role": "user", "content": "a"}], "prompt_tokens": 10},
        {"action_id": "llm_1", "messages_in": [{"role": "user", "content": "b"}], "prompt_tokens": 10},
    ]
    msgs, kind = _match_messages(calls, {1: "llm_1"}, call_idx=1, input_tokens=10)
    assert kind == "trace_action_id" and msgs[0]["content"] == "b"


def test_match_messages_refuses_ambiguous_prompt_tokens():
    # no action map; two calls share prompt_tokens; positional out of range
    calls = [
        {"action_id": None, "messages_in": [{"role": "user", "content": "a"}], "prompt_tokens": 10},
        {"action_id": None, "messages_in": [{"role": "user", "content": "b"}], "prompt_tokens": 10},
    ]
    msgs, kind = _match_messages(calls, {}, call_idx=9, input_tokens=10)
    assert msgs is None and kind == "ambiguous_prompt_tokens"


def test_match_messages_positional_when_prompt_tokens_agree():
    calls = [
        {"action_id": None, "messages_in": [{"role": "user", "content": "a"}], "prompt_tokens": 10},
        {"action_id": None, "messages_in": [{"role": "user", "content": "b"}], "prompt_tokens": 20},
    ]
    msgs, kind = _match_messages(calls, {}, call_idx=1, input_tokens=20)
    assert kind == "positional" and msgs[0]["content"] == "b"
