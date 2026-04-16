"""Unit tests for TongyiDeepResearchRunner (Ralplan R3 Phase D).

Covers US-D2 ACs: Runner protocol satisfaction, run_task success/empty paths,
scaffold_capabilities metadata, vendor monkey-patch cleanup.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.tongyi_deepresearch.runner import TongyiDeepResearchRunner
from trace_collect.attempt_pipeline import AttemptContext


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _delta_chunk(content: str, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content),
            finish_reason=finish_reason,
            index=0,
        )],
        usage=None,
    )


def _usage_chunk(p: int, c: int):
    class _U:
        def __init__(self):
            self.prompt_tokens = p
            self.completion_tokens = c

        def model_dump(self):
            return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}

    return SimpleNamespace(choices=[], usage=_U())


def _make_attempt_ctx(tmp_path: Path, instance_id: str = "inst-001") -> AttemptContext:
    task = {"instance_id": instance_id, "problem_statement": "What is 2+2?"}
    return AttemptContext(
        run_dir=tmp_path,
        instance_id=instance_id,
        attempt=1,
        task=task,
        model="fake-model",
        scaffold="tongyi-deepresearch",
        source_image=None,
        prompt_template="default",
        agent_runtime_mode="host_controller",
        execution_environment="host",
    )


def _install_fake_openai(script_factory):
    """Patch openai.OpenAI so TracedStreamingOpenAI's underlying client is fake."""

    class _FakeClient:
        def __init__(self, *_a, **_k):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *, model, messages, stream, stream_options, **kwargs):
            return iter(script_factory())

    return patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_FakeClient)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_runner_satisfies_runner_protocol():
    """AC-a: TongyiDeepResearchRunner duck-types the Runner protocol."""
    r = TongyiDeepResearchRunner(
        model="m", api_base="http://fake", api_key="k",
        max_iterations=10, benchmark_slug="deep-research-bench",
    )
    # Runner is a Protocol without @runtime_checkable, so isinstance would fail
    # even when duck-typing is satisfied. Check the contract directly.
    assert hasattr(r, "run_task")
    assert asyncio.iscoroutinefunction(r.run_task)


def test_runner_construction_accepts_optional_client():
    """AC-a: constructor accepts client kwarg for Runner-protocol parity."""
    client_sentinel = object()
    r = TongyiDeepResearchRunner(
        model="m", api_base="http://fake", api_key="k",
        max_iterations=10, benchmark_slug="deep-research-bench",
        client=client_sentinel,
    )
    # client is accepted but not required to be stored
    assert r.model == "m"
    assert r.api_base == "http://fake"


@pytest.mark.asyncio
async def test_run_task_completes_with_valid_answer(tmp_path):
    """AC-b: vendor returns <answer>...</answer> → exit_status='completed'."""
    ctx = _make_attempt_ctx(tmp_path)

    # Script: vendor will call chat.completions.create once; response includes <answer>
    def script():
        # The response content must include both <think> and <answer> tags
        # for vendor's sanity_check_output; and <answer>X</answer> for termination.
        yield _delta_chunk("<think>thinking</think>")
        yield _delta_chunk("<answer>four</answer>", finish_reason="stop")
        yield _usage_chunk(20, 5)

    r = TongyiDeepResearchRunner(
        model="fake-model", api_base="http://fake", api_key="k",
        max_iterations=5, benchmark_slug="deep-research-bench",
    )
    with _install_fake_openai(script):
        result = await r.run_task(
            ctx.task,
            attempt_ctx=ctx,
            prompt_template="default",
        )

    assert result.exit_status == "completed", (result.exit_status, result.error)
    assert result.success is True
    assert result.summary["final_answer"] == "four"
    assert result.summary["n_turns"] == 1
    assert result.summary["total_llm_ms"] > 0
    assert result.summary["model"] == "fake-model"
    # Vendor termination surfaced in summary for analysis
    assert result.summary["vendor_termination"] == "answer"


@pytest.mark.asyncio
async def test_run_task_empty_answer_yields_empty_final_response(tmp_path):
    """AC-c: vendor never emits <answer> → exit_status='empty_final_response'."""
    ctx = _make_attempt_ctx(tmp_path)

    # Vendor will retry until MAX_LLM_CALL_PER_RUN is exhausted without finding <answer>.
    # Keep output non-empty (so vendor doesn't hit its own "empty response" retry loop)
    # but without <answer> tags. Each call returns the same content.
    def script():
        yield _delta_chunk("<think>still thinking</think> no answer yet", finish_reason="stop")
        yield _usage_chunk(5, 5)

    r = TongyiDeepResearchRunner(
        model="fake-model", api_base="http://fake", api_key="k",
        max_iterations=2, benchmark_slug="deep-research-bench",
    )
    with _install_fake_openai(script):
        result = await r.run_task(
            ctx.task,
            attempt_ctx=ctx,
            prompt_template="default",
        )

    assert result.exit_status == "empty_final_response"
    assert result.success is False
    assert result.summary["final_answer"] == ""
    # We still collected trace actions — n_turns >= 1
    assert result.summary["n_turns"] >= 1


@pytest.mark.asyncio
async def test_run_task_logs_scaffold_capabilities(tmp_path):
    """AC-d: metadata record has scaffold='tongyi-deepresearch' + expected capabilities."""
    ctx = _make_attempt_ctx(tmp_path, instance_id="inst-meta")

    def script():
        yield _delta_chunk("<think>x</think><answer>y</answer>", finish_reason="stop")
        yield _usage_chunk(3, 3)

    r = TongyiDeepResearchRunner(
        model="fake-model", api_base="http://fake", api_key="k",
        max_iterations=5, benchmark_slug="deep-research-bench",
    )
    with _install_fake_openai(script):
        result = await r.run_task(ctx.task, attempt_ctx=ctx, prompt_template="default")

    trace_path: Path = result.trace_path
    assert trace_path.exists()
    lines = [json.loads(ln) for ln in trace_path.read_text().splitlines() if ln.strip()]
    metadata = next(ln for ln in lines if ln.get("type") == "trace_metadata")
    assert metadata["scaffold"] == "tongyi-deepresearch"
    assert metadata["execution_environment"] == "host"
    assert metadata["benchmark"] == "deep-research-bench"
    assert metadata["scaffold_capabilities"] == {
        "tools": ["search", "visit"],
        "memory": False,
        "skills": False,
        "file_ops": "none",
    }


@pytest.mark.asyncio
async def test_vendor_monkey_patch_restored_after_run(tmp_path):
    """AC-e: vendor.OpenAI and vendor.TOOL_CLASS are restored after run_task."""
    from agents.tongyi_deepresearch.vendor import react_agent as vendor

    orig_openai = vendor.OpenAI
    orig_tool_class_ids = [id(t) for t in vendor.TOOL_CLASS]
    orig_count_tokens = vendor.MultiTurnReactAgent.count_tokens

    ctx = _make_attempt_ctx(tmp_path, instance_id="inst-patch")

    def script():
        yield _delta_chunk("<think>t</think><answer>a</answer>", finish_reason="stop")
        yield _usage_chunk(3, 3)

    r = TongyiDeepResearchRunner(
        model="fake-model", api_base="http://fake", api_key="k",
        max_iterations=3, benchmark_slug="deep-research-bench",
    )
    with _install_fake_openai(script):
        await r.run_task(ctx.task, attempt_ctx=ctx, prompt_template="default")

    # After run, vendor module-level state must be restored
    assert vendor.OpenAI is orig_openai
    assert [id(t) for t in vendor.TOOL_CLASS] == orig_tool_class_ids
    assert vendor.MultiTurnReactAgent.count_tokens is orig_count_tokens
