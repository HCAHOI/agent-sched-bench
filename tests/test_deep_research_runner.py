from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from agents.deep_research.runner import DeepResearchRunner
from agents.deep_research.web_tools import (
    DeepResearchWebFetchTool,
    DeepResearchWebSearchTool,
)
from agents.openclaw._runner import AgentRunner
from llm_call.provider_base import LLMProvider, LLMResponse, ToolCallRequest
from trace_collect.attempt_pipeline import AttemptContext


class _FakeProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse], *, label: str) -> None:
        super().__init__(api_key=f"{label}-key", api_base=f"https://{label}.example/v1")
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.label = label

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
        del max_tokens, reasoning_effort, tool_choice
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "model": model,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError(f"{self.label} provider received unexpected chat call")
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return f"{self.label}-model"


def _extras(**overrides) -> dict[str, Any]:
    return {
        "scorer_template": "grader",
        "scorer_provider": "fake-scorer-provider",
        "scorer_model": "fake-scorer-model",
        "scorer_temperature": 0.0,
        "web_search_provider": "searxng",
        "web_search_base_url": "https://search.example",
        "web_fetch_provider": "jina",
        "max_search_calls": 4,
        "max_fetch_calls": 4,
        "tool_result_preview_chars": 1200,
        **overrides,
    }


def _task() -> dict[str, Any]:
    return {
        "instance_id": "browsecomp-0",
        "problem_statement": "PUBLIC_QUESTION_SENTINEL: find the public answer",
        "reference_answer": "SECRET_REFERENCE_SENTINEL",
        "problem_topic": "SECRET_TOPIC_SENTINEL",
        "task_source_kind": "browsecomp_official_csv",
        "task_source_id": "0",
        "task_source_path": "https://example.invalid/browsecomp.csv",
    }


def _attempt_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="browsecomp-0",
        attempt=1,
        task=_task(),
        model="fake-eval-model",
        scaffold="deep-research",
        source_image=None,
        prompt_template="default",
        agent_runtime_mode="host_controller",
        execution_environment="host",
    )


def _runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: _FakeProvider,
    scorer: _FakeProvider,
    extras: dict[str, Any] | None = None,
    max_iterations: int = 5,
    context_window_tokens: int = 2048,
) -> DeepResearchRunner:
    monkeypatch.setattr(
        DeepResearchRunner,
        "_build_scorer_provider",
        lambda self: scorer,
    )
    return DeepResearchRunner(
        provider=provider,
        workspace_base=tmp_path / "workspaces",
        benchmark_slug="browsecomp",
        benchmark_extras=extras or _extras(),
        max_iterations=max_iterations,
        context_window_tokens=context_window_tokens,
        model="fake-eval-model",
        provider_name="fake-eval-provider",
        env_key="FAKE_EVAL_KEY",
        api_base="https://eval.example/v1",
        api_key="eval-key",
        generation_config={},
        environ={},
    )


def _prompt_section(prompt: str, start: str, end: str) -> str:
    return prompt.split(start, 1)[1].split(end, 1)[0]


def test_runner_exposes_exact_web_only_tools_and_keeps_reference_out_of_model_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content=(
                    "Explanation: I found a plausible answer.\n"
                    "Exact Answer: wrong answer\n"
                    "Confidence: 52%"
                ),
                usage={"prompt_tokens": 11, "completion_tokens": 7},
            )
        ],
        label="eval",
    )
    scorer = _FakeProvider(
        [LLMResponse(content="reasoning: mismatch\ncorrect: no")],
        label="scorer",
    )
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    tool_schemas = provider.calls[0]["tools"]
    assert [tool["function"]["name"] for tool in tool_schemas] == [
        "web_search",
        "web_fetch",
    ]
    assert tool_schemas[0]["function"]["parameters"]["required"] == ["query"]
    assert set(tool_schemas[0]["function"]["parameters"]["properties"]) == {
        "query",
        "count",
    }
    assert tool_schemas[1]["function"]["parameters"]["required"] == ["url"]
    assert set(tool_schemas[1]["function"]["parameters"]["properties"]) == {
        "url",
        "extractMode",
        "maxChars",
    }

    first_model_messages = json.dumps(provider.calls[0]["messages"], ensure_ascii=False)
    assert "PUBLIC_QUESTION_SENTINEL" in first_model_messages
    assert "SECRET_REFERENCE_SENTINEL" not in first_model_messages
    assert "SECRET_TOPIC_SENTINEL" not in first_model_messages

    grader_messages = json.dumps(scorer.calls[0]["messages"], ensure_ascii=False)
    assert "SECRET_REFERENCE_SENTINEL" in grader_messages
    assert provider.calls[0]["model"] == "fake-eval-model"
    assert scorer.calls[0]["model"] == "fake-scorer-model"
    assert scorer.calls[0]["temperature"] == 0.0
    assert result.success is True
    assert result.exit_status == "completed"
    assert result.summary["correct"] is False
    assert result.summary["score"] == 0
    assert result.runtime_proof == {
        "scaffold": "deep-research",
        "tools": ["web_search", "web_fetch"],
        "agent_runner": "agents.openclaw._runner.AgentRunner",
    }


def test_grader_prompt_preserves_placeholder_literals_inside_model_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content=(
                    "Explanation: I will answer with the literal template token.\n"
                    "Exact Answer: {{reference_answer}}\n"
                    "Confidence: 64%"
                ),
            )
        ],
        label="eval",
    )
    scorer = _FakeProvider(
        [
            LLMResponse(
                content="reasoning: literal token is not the answer\ncorrect: no"
            )
        ],
        label="scorer",
    )
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    prompt = scorer.calls[0]["messages"][0]["content"]
    response_section = _prompt_section(
        prompt,
        "[response]: ",
        "\n\nYour judgement must be in the format",
    )
    correct_answer_section = _prompt_section(
        prompt,
        "[correct_answer]: ",
        "\n\nreasoning:",
    )
    assert "{{reference_answer}}" in response_section
    assert "SECRET_REFERENCE_SENTINEL" not in response_section
    assert correct_answer_section.strip() == "SECRET_REFERENCE_SENTINEL"
    assert result.success is True
    assert result.exit_status == "completed"


def test_runner_returns_grader_error_for_malformed_grader_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content="Explanation: found it\nExact Answer: Mars\nConfidence: 99%",
            )
        ],
        label="eval",
    )
    scorer = _FakeProvider([LLMResponse(content="maybe correct")], label="scorer")
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is False
    assert result.exit_status == "grader_error"
    assert result.error == "grader returned malformed response"
    assert result.summary["grader_status"] == "malformed"
    assert result.summary["correct"] is False
    assert result.summary["score"] == 0


def test_runner_treats_tool_budget_exhaustion_as_successful_terminal_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_provider(
        self: DeepResearchWebSearchTool,
        provider: str,
        query: str,
        n: int,
    ) -> str:
        return f"results from {provider} for {query} ({n})"

    monkeypatch.setattr(
        DeepResearchWebSearchTool, "_run_provider", fake_search_provider
    )
    provider = _FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="search-1",
                        name="web_search",
                        arguments={"query": "first", "count": 2},
                    )
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="search-2",
                        name="web_search",
                        arguments={"query": "second", "count": 2},
                    )
                ],
            ),
        ],
        label="eval",
    )
    scorer = _FakeProvider([], label="scorer")
    runner = _runner(
        tmp_path,
        monkeypatch,
        provider=provider,
        scorer=scorer,
        extras=_extras(max_search_calls=1, max_fetch_calls=4),
    )

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is True
    assert result.exit_status == "tool_budget_exhausted"
    assert result.error is None
    assert result.summary["tool_budget_exhausted"] is True
    assert result.summary["budget_exhausted_tool"] == "web_search"
    assert result.summary["search_calls_used"] == 1
    assert result.summary["grader_status"] == "not_run_budget_exhausted"
    assert result.summary["answer_parse_error"] == "missing_exact_answer"
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("tool_name", "tool_arguments", "expected_backend_tool", "expected_error"),
    [
        (
            "web_search",
            {"query": "outage", "count": 2},
            "web_search",
            "Error: searxng backend timed out",
        ),
        (
            "web_fetch",
            {"url": "https://example.com/outage"},
            "web_fetch",
            "Jina Reader failed: network connection timed out",
        ),
    ],
)
def test_runner_fails_closed_without_grading_when_web_backend_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    tool_arguments: dict[str, Any],
    expected_backend_tool: str,
    expected_error: str,
) -> None:
    if tool_name == "web_search":

        async def fake_search_provider(
            self: DeepResearchWebSearchTool,
            provider: str,
            query: str,
            n: int,
        ) -> str:
            del self, provider, query, n
            return expected_error

        monkeypatch.setattr(
            DeepResearchWebSearchTool, "_run_provider", fake_search_provider
        )
    else:

        async def fake_fetch_provider(
            self: DeepResearchWebFetchTool,
            provider: str,
            url: str,
            extract_mode: str,
            max_chars: int,
        ) -> str:
            del self, provider, extract_mode, max_chars
            return json.dumps({"error": expected_error, "url": url}, ensure_ascii=False)

        monkeypatch.setattr(
            DeepResearchWebFetchTool, "_run_provider", fake_fetch_provider
        )

    provider = _FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="tool-1",
                        name=tool_name,
                        arguments=tool_arguments,
                    )
                ],
            )
        ],
        label=f"eval-{tool_name}",
    )
    scorer = _FakeProvider([], label=f"scorer-{tool_name}")
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is False
    assert result.exit_status == "tool_backend_failed"
    assert result.error == expected_error
    assert result.summary["grader_status"] == "not_run_tool_backend_failed"
    assert result.summary["tool_backend_failed"] is True
    assert result.summary["backend_failed_tool"] == expected_backend_tool
    assert result.summary["backend_failure_error"] == expected_error
    assert result.summary["correct"] is False
    assert result.summary["score"] == 0
    assert scorer.calls == []
    [tool_call] = result.tool_calls
    assert tool_call["tool_name"] == expected_backend_tool
    assert tool_call["error"] == expected_error


def test_runner_preserves_max_iterations_stop_status_over_parse_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_provider(
        self: DeepResearchWebSearchTool,
        provider: str,
        query: str,
        n: int,
    ) -> str:
        return f"results from {provider} for {query} ({n})"

    monkeypatch.setattr(
        DeepResearchWebSearchTool, "_run_provider", fake_search_provider
    )
    provider = _FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="search-1",
                        name="web_search",
                        arguments={"query": "keep researching", "count": 1},
                    )
                ],
                usage={"prompt_tokens": 3, "completion_tokens": 2},
            )
        ],
        label="eval",
    )
    scorer = _FakeProvider([], label="scorer")
    runner = _runner(
        tmp_path,
        monkeypatch,
        provider=provider,
        scorer=scorer,
        max_iterations=1,
    )

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is True
    assert result.exit_status == "max_iterations"
    assert result.error is None
    assert result.summary["grader_status"] == "not_run"
    assert result.summary["answer_parse_error"] == "missing_exact_answer"
    assert result.summary["correct"] is False
    assert result.summary["score"] == 0
    assert scorer.calls == []


@pytest.mark.parametrize(
    (
        "responses",
        "expected_exit_status",
        "expected_error_fragment",
        "expected_eval_calls",
    ),
    [
        (
            [
                LLMResponse(
                    content="provider rejected request",
                    finish_reason="error",
                )
            ],
            "error",
            "provider rejected request",
            1,
        ),
        (
            [LLMResponse(content=""), LLMResponse(content="")],
            "empty_final_response",
            "couldn't produce a final answer",
            2,
        ),
    ],
    ids=["provider-error", "empty-final-response"],
)
def test_runner_preserves_agent_failure_stop_statuses_over_parse_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[LLMResponse],
    expected_exit_status: str,
    expected_error_fragment: str,
    expected_eval_calls: int,
) -> None:
    provider = _FakeProvider(responses, label="eval")
    scorer = _FakeProvider([], label="scorer")
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is False
    assert result.exit_status == expected_exit_status
    assert result.exit_status != "completed"
    assert result.error is not None
    assert expected_error_fragment in result.error
    assert result.summary["grader_status"] == "not_run"
    assert result.summary["answer_parse_error"] == "missing_exact_answer"
    assert result.summary["correct"] is False
    assert result.summary["score"] == 0
    assert len(provider.calls) == expected_eval_calls
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("case_name", "usage", "expected_total_tokens"),
    [
        (
            "reported-total",
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 100},
            100,
        ),
        (
            "total-absent",
            {"prompt_tokens": 11, "completion_tokens": 7},
            18,
        ),
        (
            "total-zero",
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 0},
            18,
        ),
    ],
)
def test_runner_prefers_reported_total_tokens_and_falls_back_when_absent_or_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    usage: dict[str, int],
    expected_total_tokens: int,
) -> None:
    case_dir = tmp_path / case_name
    provider = _FakeProvider(
        [
            LLMResponse(
                content="Explanation: found it\nExact Answer: Mars\nConfidence: 99%",
                usage=usage,
            )
        ],
        label=f"eval-{case_name}",
    )
    scorer = _FakeProvider(
        [LLMResponse(content="reasoning: mismatch\ncorrect: no")],
        label=f"scorer-{case_name}",
    )
    runner = _runner(case_dir, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(),
            attempt_ctx=_attempt_ctx(case_dir),
            prompt_template="default",
        )
    )

    assert result.exit_status == "completed"
    assert result.total_tokens == expected_total_tokens
    assert result.summary["total_tokens"] == expected_total_tokens


def test_runner_spills_oversized_tool_output_to_required_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_tool_output = "0123456789ABBEYOND_PREVIEW_SENTINEL-" + ("payload-" * 12)

    async def fake_fetch_provider(
        self: DeepResearchWebFetchTool,
        provider: str,
        url: str,
        extract_mode: str,
        max_chars: int,
    ) -> str:
        return long_tool_output

    monkeypatch.setattr(DeepResearchWebFetchTool, "_run_provider", fake_fetch_provider)
    provider = _FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="fetch-1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/source"},
                    )
                ],
            ),
            LLMResponse(
                content="Explanation: sourced\nExact Answer: Mars\nConfidence: 88%",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
        ],
        label="eval",
    )
    scorer = _FakeProvider(
        [LLMResponse(content="reasoning: match\ncorrect: yes")], label="scorer"
    )
    runner = _runner(
        tmp_path,
        monkeypatch,
        provider=provider,
        scorer=scorer,
        extras=_extras(tool_result_preview_chars=12),
    )

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )
    assert len(provider.calls) == 2
    second_model_messages = json.dumps(
        provider.calls[1]["messages"], ensure_ascii=False
    )
    assert "BEYOND_PREVIEW_SENTINEL" in second_model_messages
    assert "... (truncated)" not in second_model_messages

    [tool_call] = result.tool_calls
    spilled = tool_call["tool_result"]
    assert (
        spilled["artifact_path"]
        == "artifacts/deep_research_tool_results/tool_0_web_fetch.txt"
    )
    assert spilled["original_size"] == len(long_tool_output)
    assert spilled["preview"] == long_tool_output[:12]
    assert spilled["truncated_preview"] is True
    artifact_path = result.trace_path.parent / spilled["artifact_path"]
    assert artifact_path.read_text(encoding="utf-8") == long_tool_output
    assert tool_call["artifact_path"] == spilled["artifact_path"]
    assert tool_call["artifact_sha256"] == spilled["sha256"]
    assert result.artifacts[0].name == "deep_research_tool_results"
    assert result.artifacts[0].path == artifact_path.parent
    assert result.artifacts[0].required is True
    assert result.success is True
    assert result.summary["correct"] is True
    assert result.summary["score"] == 1


def test_trace_records_empty_finalization_retry_as_separate_llm_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content="", usage={"prompt_tokens": 10, "completion_tokens": 0}
            ),
            LLMResponse(
                content="Explanation: retry succeeded\nExact Answer: Mars\nConfidence: 81%",
                usage={"prompt_tokens": 12, "completion_tokens": 5},
            ),
        ],
        label="eval",
    )
    scorer = _FakeProvider(
        [LLMResponse(content="reasoning: retry answer\ncorrect: yes")],
        label="scorer",
    )
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is True
    assert result.exit_status == "completed"
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"] is not None
    assert provider.calls[1]["tools"] is None

    trace_records = [
        json.loads(line)
        for line in result.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    llm_calls = [
        record
        for record in trace_records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]
    assert [record["action_id"] for record in llm_calls] == ["llm_0", "llm_1"]
    assert [record["iteration"] for record in llm_calls] == [0, 0]
    assert llm_calls[0]["data"]["messages_in"] == provider.calls[0]["messages"]
    assert llm_calls[1]["data"]["messages_in"] == provider.calls[1]["messages"]
    assert llm_calls[0]["data"]["raw_response"]["content"] == ""
    assert (
        llm_calls[1]["data"]["raw_response"]["content"]
        == "Explanation: retry succeeded\nExact Answer: Mars\nConfidence: 81%"
    )


def test_trace_preserves_iteration_and_call_ids_for_same_turn_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_provider(
        self: DeepResearchWebSearchTool,
        provider: str,
        query: str,
        n: int,
    ) -> str:
        del self, provider
        return f"search results for {query} ({n})"

    async def fake_fetch_provider(
        self: DeepResearchWebFetchTool,
        provider: str,
        url: str,
        extract_mode: str,
        max_chars: int,
    ) -> str:
        del self, provider, extract_mode, max_chars
        return f"fetched page for {url}"

    monkeypatch.setattr(
        DeepResearchWebSearchTool, "_run_provider", fake_search_provider
    )
    monkeypatch.setattr(DeepResearchWebFetchTool, "_run_provider", fake_fetch_provider)
    provider = _FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="bad-fetch-missing-url",
                        name="web_fetch",
                        arguments={"extractMode": "markdown"},
                    ),
                    ToolCallRequest(
                        id="actual-search-call",
                        name="web_search",
                        arguments={"query": "same turn", "count": 2},
                    ),
                    ToolCallRequest(
                        id="actual-fetch-call",
                        name="web_fetch",
                        arguments={"url": "https://example.com/source"},
                    ),
                ],
            ),
            LLMResponse(
                content="Explanation: used both tools\nExact Answer: Mars\nConfidence: 90%",
            ),
        ],
        label="eval",
    )
    scorer = _FakeProvider(
        [LLMResponse(content="reasoning: answer accepted\ncorrect: yes")],
        label="scorer",
    )
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is True
    assert len(provider.calls) == 2
    tool_messages = [
        message
        for message in provider.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "bad-fetch-missing-url",
        "actual-search-call",
        "actual-fetch-call",
    ]
    assert "Invalid parameters" in tool_messages[0]["content"]

    trace_records = [
        json.loads(line)
        for line in result.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    tool_execs = [
        record
        for record in trace_records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    assert [record["iteration"] for record in tool_execs] == [0, 0]
    assert [record["data"]["tool_call_id"] for record in tool_execs] == [
        "actual-search-call",
        "actual-fetch-call",
    ]
    assert {record["data"]["tool_call_id"] for record in tool_execs} == {
        call["tool_call_id"] for call in result.tool_calls
    }


def test_trace_llm_messages_match_snipped_provider_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_provider(
        self: DeepResearchWebSearchTool,
        provider: str,
        query: str,
        n: int,
    ) -> str:
        del self, provider, query, n
        return "SEARCH_RESULT_SENTINEL " + ("x" * 4000)

    monkeypatch.setattr(
        DeepResearchWebSearchTool, "_run_provider", fake_search_provider
    )
    snipped_prompt = [{"role": "user", "content": "SNIPPED_MODEL_VISIBLE_PROMPT"}]

    def fake_snip_history(self, spec, messages):
        del self, spec
        if len(messages) > 1:
            return deepcopy(snipped_prompt)
        return messages

    monkeypatch.setattr(AgentRunner, "_snip_history", fake_snip_history)
    provider = _FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="search-1",
                        name="web_search",
                        arguments={"query": "force snip", "count": 1},
                    )
                ],
            ),
            LLMResponse(
                content="Explanation: after snip\nExact Answer: Mars\nConfidence: 77%",
            ),
        ],
        label="eval",
    )
    scorer = _FakeProvider(
        [LLMResponse(content="reasoning: enough\ncorrect: yes")],
        label="scorer",
    )
    runner = _runner(tmp_path, monkeypatch, provider=provider, scorer=scorer)

    result = asyncio.run(
        runner.run_task(
            _task(), attempt_ctx=_attempt_ctx(tmp_path), prompt_template="default"
        )
    )

    assert result.success is True
    assert len(provider.calls) == 2
    second_prompt_sent = provider.calls[1]["messages"]
    assert second_prompt_sent == snipped_prompt

    trace_records = [
        json.loads(line)
        for line in result.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    llm_calls = [
        record
        for record in trace_records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]
    assert len(llm_calls) == 2
    assert llm_calls[0]["data"]["messages_in"] == provider.calls[0]["messages"]
    assert llm_calls[1]["data"]["messages_in"] == second_prompt_sent
