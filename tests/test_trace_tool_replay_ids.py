"""Trace tool replay must accept the model's tool call ids when the recorded
executor used its own ids (qwen3.7-max collections), and stay strict otherwise."""
import asyncio

import pytest

from trace_collect.openclaw_host_runtime import _TraceToolReplayState


def _llm(*ids: str) -> dict:
    return {"action_type": "llm_call",
            "data": {"raw_response": {"choices": [{"message": {"tool_calls": [{"id": i} for i in ids]}}]}}}


def _tool(call_id: str, name: str = "exec") -> dict:
    return {"action_type": "tool_exec", "data": {"tool_call_id": call_id, "tool_name": name,
                                                   "tool_args": {"command": "ls"}, "tool_result": "ok", "duration_ms": 0}}


def _run(state: _TraceToolReplayState, call_id: str, name: str = "exec") -> str:
    return asyncio.run(state.execute(call_id=call_id, tool_name=name, arguments={"command": "ls"}))


def test_matching_ids_unchanged() -> None:
    state = _TraceToolReplayState([_llm("call_a"), _tool("call_a")], replay_speed=1e6)
    assert state.alias == {}
    assert _run(state, "call_a") == "ok" and state.complete


def test_model_ids_alias_execution_records_by_order() -> None:
    actions = [_llm("call_0_0"), _tool("uy9W4K0eg"), _llm("call_1_0"), _tool("sjGlemy7B", "read_file")]
    state = _TraceToolReplayState(actions, replay_speed=1e6)
    assert state.alias == {"call_0_0": "uy9W4K0eg", "call_1_0": "sjGlemy7B"}
    assert _run(state, "call_0_0") == "ok"
    assert not state.complete
    assert _run(state, "call_1_0", "read_file") == "ok"
    assert state.complete
    with pytest.raises(RuntimeError, match="executed twice"):
        _run(state, "call_0_0")


def test_unknown_id_still_rejected() -> None:
    state = _TraceToolReplayState([_llm("call_0_0"), _tool("uy9W4K0eg")], replay_speed=1e6)
    with pytest.raises(RuntimeError, match="no tool call"):
        _run(state, "call_9_9")


def test_duplicate_model_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _TraceToolReplayState([_llm("call_x"), _tool("a"), _llm("call_x"), _tool("b")], replay_speed=1e6)


def test_failure_counts_align_model_ids_with_source_records() -> None:
    from trace_collect.openclaw_host_runtime import replay_action_failure_counts

    source = [
        {"action_type": "llm_call", "action_id": "llm_0",
         "data": {"raw_response": {"choices": [{"message": {"tool_calls": [{"id": "call_0_0"}]}}]}}},
        {"action_type": "tool_exec", "action_id": "tool_0_uy9W4K0eg",
         "data": {"tool_call_id": "uy9W4K0eg", "tool_name": "exec", "tool_args": {"command": "ls"}, "success": False}},
    ]
    replay = [
        {"type": "action", "action_type": "llm_call", "action_id": "llm_0", "data": {}},
        {"type": "action", "action_type": "tool_exec", "action_id": "tool_0_call_0_0",
         "data": {"tool_call_id": "call_0_0", "tool_name": "exec", "tool_args": {"command": "ls"}, "success": False}},
    ]
    counts = replay_action_failure_counts(source, replay, require_exact_tool_calls=True)
    assert counts.action_sequence_matches and counts.unexpected_replay_failed_actions == 0
    assert counts.replay_failed_actions == counts.source_failed_actions == 1
