from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.qwen_deep_research import QwenDeepResearchRunner
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.deep_research_bench import DeepResearchBenchBenchmark
from trace_collect.collector import collect_traces
from trace_collect.attempt_pipeline import AttemptContext


class _AsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _MockCompletions:
    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _AsyncStream:
        self.seen = kwargs
        return _AsyncStream(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="final "),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                    model_dump=lambda: {
                        "choices": [{"delta": {"content": "final "}}],
                    },
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="answer"),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                    model_dump=lambda: {
                        "choices": [
                            {
                                "delta": {"content": "answer"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=11,
                        completion_tokens=2,
                        model_dump=lambda: {
                            "prompt_tokens": 11,
                            "completion_tokens": 2,
                        },
                    ),
                    model_dump=lambda: {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 2,
                        },
                    },
                ),
            ]
        )


class _MockClient:
    def __init__(self) -> None:
        self.completions = _MockCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _make_attempt_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="research-1",
        attempt=1,
        task={"instance_id": "research-1", "problem_statement": "Question"},
        model="qwen-plus-latest",
        scaffold="qwen-deep-research",
        source_image=None,
        prompt_template="default",
        agent_runtime_mode="host_controller",
    )


def test_qwen_deep_research_runner_writes_v5_trace(tmp_path: Path) -> None:
    client = _MockClient()
    runner = QwenDeepResearchRunner(
        model="qwen-plus-latest",
        api_base="https://dashscope.example/v1",
        api_key="test-key",
        max_iterations=100,
        benchmark_slug="deep-research-bench",
        client=client,
    )
    ctx = _make_attempt_ctx(tmp_path)

    result = asyncio.run(
        runner.run_task(
            {
                "instance_id": "research-1",
                "problem_statement": "Question",
                "reference_answer": "Do not leak",
                "topic": "science",
            },
            attempt_ctx=ctx,
            prompt_template="default",
        )
    )

    assert result.success is True
    assert result.exit_status == "completed"
    assert result.total_tokens == 13
    assert client.completions.seen["stream"] is True
    assert client.completions.seen["stream_options"] == {"include_usage": True}
    messages = client.completions.seen["messages"]
    assert "Question" in messages[1]["content"]
    assert "science" in messages[1]["content"]
    assert "Do not leak" not in messages[1]["content"]

    records = [
        json.loads(line)
        for line in result.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["type"] == "trace_metadata"
    assert records[0]["trace_format_version"] == 5
    assert records[0]["scaffold"] == "qwen-deep-research"
    assert records[0]["execution_environment"] == "host"

    action = next(record for record in records if record.get("type") == "action")
    assert action["action_type"] == "llm_call"
    assert action["data"]["prompt_tokens"] == 11
    assert action["data"]["completion_tokens"] == 2
    assert action["data"]["llm_latency_ms"] >= 0
    assert action["data"]["llm_call_time_ms"] >= 0
    assert action["data"]["llm_wall_latency_ms"] >= 0
    assert action["data"]["ttft_ms"] is not None
    assert action["data"]["tpot_ms"] is not None
    assert action["data"]["llm_output"] == "final answer"


def test_qwen_prompt_template_controls_messages_without_reference_leak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = QwenDeepResearchRunner(
        model="qwen-plus-latest",
        api_base="https://dashscope.example/v1",
        api_key="test-key",
        max_iterations=100,
        benchmark_slug="deep-research-bench",
        client=_MockClient(),
    )
    monkeypatch.setattr(
        runner,
        "_load_prompt_template",
        lambda name: f"template={name} :: {{{{task}}}}",
    )

    messages = runner._build_messages(
        {
            "problem_statement": "Question",
            "reference_answer": "Do not leak",
            "domain": "science",
        },
        prompt_template="custom",
    )

    assert "template=custom :: Question" in messages[1]["content"]
    assert "science" in messages[1]["content"]
    assert "Do not leak" not in messages[1]["content"]


def test_collect_traces_dispatches_qwen_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agents.qwen_deep_research.runner.create_async_openai_client",
        lambda *, api_base, api_key: _MockClient(),
    )
    config = BenchmarkConfig(
        slug="deep-research-bench",
        display_name="DeepResearchBench",
        harness_dataset="example/deepresearchbench",
        harness_split="test",
        trace_root=tmp_path / "traces",
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
        extras={
            "id_field": "id",
            "question_field": "prompt",
            "answer_field": "article",
        },
    )
    benchmark = DeepResearchBenchBenchmark(config)
    benchmark.load_tasks = lambda: [  # type: ignore[method-assign]
        {
            "instance_id": "research-1",
            "problem_statement": "Question",
            "reference_answer": "Do not leak",
        }
    ]

    run_dir = asyncio.run(
        collect_traces(
            scaffold="qwen-deep-research",
            provider_name="dashscope",
            api_base="https://dashscope.example/v1",
            api_key="test-key",
            model="qwen-plus-latest",
            benchmark=benchmark,
            run_id=str(tmp_path / "run"),
            min_free_disk_gb=0.001,
        )
    )

    trace_path = run_dir / "research-1" / "attempt_1" / "trace.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["scaffold"] == "qwen-deep-research"
    assert any(record.get("action_type") == "llm_call" for record in records)
