from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.tool_latency_context import context_lengths_from_traces
from trace_collect.tool_latency_dataset import extract_tool_latency_samples


def _write_trace(path: Path, actions: list[dict[str, Any]]) -> None:
    records: list[dict[str, Any]] = [
        {"type": "trace_metadata", "trace_format_version": 5, "instance_id": "task-x"}
    ]
    records.extend({"type": "action", **action} for action in actions)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _llm(iteration: int, prompt_tokens: int) -> dict[str, Any]:
    return {
        "action_type": "llm_call",
        "agent_id": "task-x",
        "iteration": iteration,
        "action_id": f"llm_{iteration}",
        "ts_start": float(iteration),
        "ts_end": float(iteration) + 0.5,
        "data": {"prompt_tokens": prompt_tokens},
    }


def _tool(iteration: int, action_id: str) -> dict[str, Any]:
    return {
        "action_type": "tool_exec",
        "agent_id": "task-x",
        "iteration": iteration,
        "action_id": action_id,
        "ts_start": float(iteration) + 0.5,
        "ts_end": float(iteration) + 0.6,
        "data": {"tool_name": "exec", "tool_call_id": action_id},
    }


def test_context_length_matches_same_iteration_prompt_tokens(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    # Two iterations; iteration 0 requests two tools that share its context.
    _write_trace(
        trace,
        [
            _llm(0, 1000),
            _tool(0, "tool_0"),
            _tool(0, "tool_1"),
            _llm(1, 2500),
            _tool(1, "tool_2"),
        ],
    )

    context = context_lengths_from_traces([trace])

    # Sample ids align exactly with the latency dataset extractor.
    sample_ids = {s.sample_id for s in extract_tool_latency_samples(trace)}
    assert set(context) == sample_ids
    by_action = {
        sid.rsplit(":", 1)[1]: length for sid, length in context.items()
    }
    assert by_action == {"tool_0": 1000, "tool_1": 1000, "tool_2": 2500}


def test_missing_llm_call_fails_fast(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    # Tool call in iteration 3 has no llm_call in that iteration.
    _write_trace(trace, [_llm(0, 1000), _tool(3, "tool_9")])

    with pytest.raises(ValueError, match="no same-iteration llm_call"):
        context_lengths_from_traces([trace])


def test_non_integer_prompt_tokens_fails_fast(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    bad_llm = _llm(0, 1000)
    bad_llm["data"]["prompt_tokens"] = "1000"
    _write_trace(trace, [bad_llm, _tool(0, "tool_0")])

    with pytest.raises(ValueError, match="integer prompt_tokens"):
        context_lengths_from_traces([trace])
