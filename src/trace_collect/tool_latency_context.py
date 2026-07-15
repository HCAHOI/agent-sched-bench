"""Recover per-call resident context length from canonical trace JSONL.

The recompute restore cost (see :mod:`trace_collect.tool_latency_recompute`)
scales with the number of context tokens resident in the KV cache when a tool
call starts. In the agent loop each iteration is exactly one ``llm_call``
followed by the ``tool_exec`` actions it requested, so the ``prompt_tokens``
of that iteration's ``llm_call`` is the resident context length for every tool
call in the iteration. That value is known when the model dispatches the call,
so it is an inference-time signal, not oracle information.

This module joins each ``tool_exec`` sample to its same-iteration ``llm_call``
``prompt_tokens``. Sample ids are produced by
:func:`trace_collect.tool_latency_dataset.extract_tool_latency_samples`, so the
map keys align exactly with the latency dataset and with fitted decision rows.
Every tool call must have a same-iteration ``llm_call`` carrying an integer
``prompt_tokens``; a missing join fails fast rather than fabricating a length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from trace_collect.tool_gap_extractor import _int_field
from trace_collect.tool_latency_dataset import extract_tool_latency_samples
from trace_collect.trace_data import TraceData

__all__ = ["context_lengths_from_traces"]


def _prompt_tokens_by_iteration(trace: TraceData) -> dict[tuple[str, int], int]:
    """Map each (agent_id, iteration) to its llm_call prompt_tokens."""

    by_iteration: dict[tuple[str, int], int] = {}
    for action in trace.actions:
        if action.get("action_type") != "llm_call":
            continue
        agent_id = str(action.get("agent_id") or "")
        iteration = _int_field(action, "iteration")
        prompt_tokens = (action.get("data") or {}).get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
            raise ValueError(
                f"{trace.path}: llm_call at iteration {iteration} for agent "
                f"{agent_id!r} lacks an integer prompt_tokens"
            )
        key = (agent_id, iteration)
        existing = by_iteration.get(key)
        if existing is not None and existing != prompt_tokens:
            raise ValueError(
                f"{trace.path}: agent {agent_id!r} iteration {iteration} has "
                f"conflicting prompt_tokens {existing} and {prompt_tokens}"
            )
        by_iteration[key] = prompt_tokens
    return by_iteration


def context_lengths_from_traces(
    trace_paths: Iterable[Path],
    *,
    agent_filter: str | None = None,
) -> dict[str, int]:
    """Return ``{sample_id: context_length_tokens}`` over the given traces.

    ``context_length_tokens`` is the ``prompt_tokens`` of the tool call's
    same-iteration ``llm_call``. Raises if any tool call has no such llm_call.
    """

    context_by_sample: dict[str, int] = {}
    for trace_path in trace_paths:
        trace = TraceData.load(trace_path, agent_filter=agent_filter)
        prompt_tokens_by_iteration = _prompt_tokens_by_iteration(trace)
        for sample in extract_tool_latency_samples(
            trace_path, agent_filter=agent_filter
        ):
            key = (sample.agent_id, sample.iteration)
            prompt_tokens = prompt_tokens_by_iteration.get(key)
            if prompt_tokens is None:
                raise ValueError(
                    f"{trace_path}: tool call {sample.sample_id!r} has no "
                    f"same-iteration llm_call with prompt_tokens"
                )
            context_by_sample[sample.sample_id] = prompt_tokens
    if not context_by_sample:
        raise ValueError("no tool calls with context length found")
    return context_by_sample
