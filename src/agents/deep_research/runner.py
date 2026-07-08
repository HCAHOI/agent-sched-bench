"""Host-controller deep-research runner for BrowseComp-style tasks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from agents.base import TraceAction
from agents.deep_research.web_tools import (
    DeepResearchWebConfig,
    DeepResearchWebFetchTool,
    DeepResearchWebSearchTool,
    ToolExecutionRecord,
    WebToolRuntimeState,
)
from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw.tools.registry import ToolRegistry
from harness.trace_logger import TraceLogger
from llm_call import UnifiedProvider
from llm_call.provider_base import LLMProvider, LLMResponse
from llm_call.providers import PROVIDERS
from trace_collect.attempt_pipeline import (
    AttemptArtifact,
    AttemptContext,
    AttemptResult,
)
from trace_collect.prompt_loader import load_prompt_template, render_prompt

_GRADER_PLACEHOLDERS = ("{{question}}", "{{model_response}}", "{{reference_answer}}")
_EXACT_ANSWER_RE = re.compile(r"(?im)^\s*Exact Answer\s*:\s*(.+?)\s*$")
_CONFIDENCE_RE = re.compile(r"(?im)^\s*Confidence\s*:\s*([0-9]{1,3})\s*%?\s*$")
_GRADER_CORRECT_RE = re.compile(r"(?im)^\s*correct\s*:\s*(yes|no)\s*$")
_TOOL_RESULT_PREVIEW_CHARS = 1200
# Fetched pages are JSON-wrapped before reaching the model; keep that wrapper
# separate from the trace/artifact preview budget.
_TOOL_RESULT_JSON_OVERHEAD_CHARS = 4096
_GRADER_PLACEHOLDER_RE = re.compile(
    r"\{\{(question|model_response|reference_answer)\}\}"
)


@dataclass(slots=True)
class _LLMTraceRecord:
    iteration: int
    messages_in: list[dict[str, Any]]
    response: dict[str, Any]
    usage: dict[str, int]
    ts_start: float
    ts_end: float


@dataclass(slots=True)
class _ToolMessageTraceRecord:
    iteration: int
    tool_name: str
    tool_call_id: str
    tool_args: str
    tool_result: str
    success: bool
    ts_start: float
    ts_end: float

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "iteration": self.iteration,
            "tool_call_id": self.tool_call_id,
            "tool_result": self.tool_result,
            "duration_ms": max(0.0, self.ts_end - self.ts_start) * 1000.0,
            "success": self.success,
            "error": None if self.success else self.tool_result,
        }


@dataclass(slots=True)
class _TraceCaptureHook(AgentHook):
    """Capture model calls emitted by ``AgentRunner``."""

    llm_records: list[_LLMTraceRecord] = field(default_factory=list)
    tool_message_records: list[_ToolMessageTraceRecord] = field(default_factory=list)
    _iteration_start: dict[int, float] = field(default_factory=dict)

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._iteration_start[context.iteration] = time.time()

    async def after_llm_response(self, context: AgentHookContext) -> None:
        if context.response is None:
            return
        ts_end = time.time()
        messages_in = context.model_messages or context.messages
        self.llm_records.append(
            _LLMTraceRecord(
                iteration=context.iteration,
                messages_in=_jsonable(messages_in),
                response=_response_to_dict(context.response),
                usage=dict(context.usage),
                ts_start=context.llm_call_start_ts
                if context.llm_call_start_ts is not None
                else self._iteration_start.get(context.iteration, ts_end),
                ts_end=ts_end,
            )
        )

    async def after_iteration(self, context: AgentHookContext) -> None:
        if not context.tool_calls:
            return
        calls_by_id = {call.id: call for call in context.tool_calls}
        events_by_call_id = {
            call.id: event for call, event in zip(context.tool_calls, context.tool_events)
        }
        for message in _trailing_tool_messages(context.messages):
            tool_call_id = str(message.get("tool_call_id") or "")
            if tool_call_id not in calls_by_id:
                continue
            call = calls_by_id[tool_call_id]
            timing = context.tool_timings.get(tool_call_id, {})
            ts_end = _float_or_now(timing.get("ts_end"))
            ts_start = _float_or_default(timing.get("ts_start"), ts_end)
            content = str(message.get("content") or "")
            event = events_by_call_id.get(tool_call_id)
            success = (
                event.get("status") == "ok"
                if event is not None
                else not content.startswith("Error")
            )
            tool_name = str(message.get("name") or call.name)
            self.tool_message_records.append(
                _ToolMessageTraceRecord(
                    iteration=context.iteration,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_args=json.dumps(call.arguments, ensure_ascii=False),
                    tool_result=content,
                    success=success,
                    ts_start=ts_start,
                    ts_end=ts_end,
                )
            )


class DeepResearchRunner:
    """Run BrowseComp tasks through a web-only host-controller scaffold."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        workspace_base: Path,
        benchmark_slug: str,
        benchmark_extras: Mapping[str, Any],
        max_iterations: int,
        context_window_tokens: int,
        model: str,
        provider_name: str | None,
        env_key: str | None,
        api_base: str,
        api_key: str,
        generation_config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.workspace_base = workspace_base
        self.benchmark_slug = benchmark_slug
        self.benchmark_extras = dict(benchmark_extras)
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.model = model
        self.provider_name = provider_name
        self.env_key = env_key
        self.api_base = api_base
        self.api_key = api_key
        self.generation_config = dict(generation_config or {})
        self.environ = environ if environ is not None else os.environ
        self.web_config = DeepResearchWebConfig.from_extras(self.benchmark_extras)
        self.web_config.validate(self.environ)
        self.scorer_provider_name = str(self.benchmark_extras["scorer_provider"])
        self.scorer_model = str(self.benchmark_extras["scorer_model"])
        self.scorer_temperature = float(
            self.benchmark_extras.get("scorer_temperature", 0.0)
        )
        self.scorer = self._build_scorer_provider()

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        problem = str(task["problem_statement"])
        reference_answer = str(task["reference_answer"])
        template = load_prompt_template(prompt_template, self.benchmark_slug)
        user_prompt = render_prompt(template, problem)

        attempt_ctx.attempt_dir.mkdir(parents=True, exist_ok=True)
        trace_logger = TraceLogger(attempt_ctx.attempt_dir, "trace")
        trace_path = trace_logger.path
        artifacts: list[AttemptArtifact] = []
        tool_state = WebToolRuntimeState(
            max_search_calls=self.web_config.max_search_calls,
            max_fetch_calls=self.web_config.max_fetch_calls,
        )
        registry = ToolRegistry()
        registry.register(
            DeepResearchWebSearchTool(
                config=self.web_config,
                state=tool_state,
                environ=self.environ,
            )
        )
        registry.register(
            DeepResearchWebFetchTool(
                config=self.web_config,
                state=tool_state,
                environ=self.environ,
            )
        )
        hook = _TraceCaptureHook()
        preview_chars = int(
            self.benchmark_extras.get(
                "tool_result_preview_chars", _TOOL_RESULT_PREVIEW_CHARS
            )
        )
        model_tool_result_chars = _model_tool_result_chars(
            self.benchmark_extras,
            fetch_max_chars=self.web_config.fetch_max_chars,
            preview_chars=preview_chars,
        )
        run_config = {
            "provider": self.provider_name,
            "api_base": self.api_base,
            "env_key": self.env_key,
            "generation_config": dict(self.generation_config),
            "scorer_provider": self.scorer_provider_name,
            "scorer_model": self.scorer_model,
            "scorer_temperature": self.scorer_temperature,
            "web_search_provider": self.web_config.search_provider,
            "web_search_api_key_env": self.web_config.search_api_key_env,
            "web_search_fallback_provider": self.web_config.search_fallback_provider,
            "web_search_base_url": self.web_config.search_base_url,
            "web_fetch_provider": self.web_config.fetch_provider,
            "web_fetch_api_key_env": self.web_config.fetch_api_key_env,
            "web_fetch_fallback_provider": self.web_config.fetch_fallback_provider,
            "web_fetch_max_chars": self.web_config.fetch_max_chars,
            "max_search_calls": self.web_config.max_search_calls,
            "max_fetch_calls": self.web_config.max_fetch_calls,
            "tool_result_preview_chars": preview_chars,
            "model_tool_result_chars": model_tool_result_chars,
        }
        trace_logger.log_metadata(
            scaffold="deep-research",
            execution_environment="host",
            benchmark=self.benchmark_slug,
            model=self.model,
            instance_id=attempt_ctx.instance_id,
            prompt_template=prompt_template,
            sensitive_artifacts=bool(
                self.benchmark_extras.get("sensitive_artifacts", True)
            ),
            run_config=run_config,
            task_source_kind=task.get("task_source_kind"),
            task_source_id=task.get("task_source_id"),
            task_source_path=task.get("task_source_path"),
            task_source_sha256=task.get("task_source_sha256"),
            task_source_schema=task.get("task_source_schema"),
            task_source_row_count=task.get("task_source_row_count"),
            scaffold_capabilities={
                "tools": registry.tool_names,
                "memory": False,
                "skills": False,
                "file_ops": "none",
            },
        )

        runner = AgentRunner(self.provider)
        result = await runner.run(
            AgentRunSpec(
                initial_messages=[{"role": "user", "content": user_prompt}],
                tools=registry,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=model_tool_result_chars,
                hook=hook,
                concurrent_tools=False,
                fail_on_tool_error=False,
                workspace=None,
                tool_results_dir=None,
                session_key=attempt_ctx.instance_id,
                context_window_tokens=self.context_window_tokens,
            )
        )

        _spill_large_tool_results(
            tool_state.records,
            attempt_ctx.attempt_dir,
            preview_chars=preview_chars,
        )
        for artifact_path in _tool_spill_artifact_paths(attempt_ctx.attempt_dir):
            artifacts.append(
                AttemptArtifact(
                    name="deep_research_tool_results",
                    path=artifact_path.parent,
                    required=True,
                )
            )
            break
        tool_actions = _combined_tool_actions(tool_state.records, hook.tool_message_records)


        for call_index, record in enumerate(hook.llm_records):
            trace_logger.log_trace_action(
                "deep_research",
                TraceAction(
                    action_type="llm_call",
                    action_id=f"llm_{call_index}",
                    agent_id="deep_research",
                    program_id=self.benchmark_slug,
                    instance_id=attempt_ctx.instance_id,
                    iteration=record.iteration,
                    ts_start=record.ts_start,
                    ts_end=record.ts_end,
                    data={
                        "messages_in": record.messages_in,
                        "raw_response": record.response,
                        "usage": record.usage,
                        "prompt_tokens": _int_usage_value(
                            record.usage.get("prompt_tokens")
                        ),
                        "completion_tokens": _int_usage_value(
                            record.usage.get("completion_tokens")
                        ),
                        "total_tokens": _total_tokens(record.usage),
                        "llm_latency_ms": max(
                            0.0, record.ts_end - record.ts_start
                        )
                        * 1000.0,
                        "llm_wall_latency_ms": max(
                            0.0, record.ts_end - record.ts_start
                        )
                        * 1000.0,
                        "model": self.model,
                        "provider": self.provider_name,
                    },
                ),
            )
        for tool_action in tool_actions:
            trace_logger.log_trace_action(
                "deep_research",
                TraceAction(
                    action_type="tool_exec",
                    action_id=(
                        f"tool_{tool_action['iteration']}_{tool_action['action_suffix']}"
                    ),
                    agent_id="deep_research",
                    program_id=self.benchmark_slug,
                    instance_id=attempt_ctx.instance_id,
                    iteration=tool_action["iteration"],
                    ts_start=tool_action["ts_start"],
                    ts_end=tool_action["ts_end"],
                    data=tool_action["data"],
                ),
            )

        summary, exit_status, success, error = await self._summarize_attempt(
            task=task,
            model_response=result.final_content or "",
            stop_reason=result.stop_reason,
            tool_state=tool_state,
            artifacts=artifacts,
            attempt_dir=attempt_ctx.attempt_dir,
            reference_answer=reference_answer,
        )
        summary.update(
            {
                "n_iterations": len(hook.llm_records),
                "total_llm_ms": _total_llm_ms(hook.llm_records),
                "total_tool_ms": _total_tool_ms_from_actions(tool_actions),
                "total_tokens": _total_tokens(result.usage),
                "tool_ms_by_name": _tool_ms_by_name_from_actions(tool_actions),
                "search_calls_used": tool_state.search_calls_used,
                "fetch_calls_used": tool_state.fetch_calls_used,
                "tool_budget_exhausted": tool_state.tool_budget_exhausted,
                "budget_exhausted_tool": tool_state.budget_exhausted_tool,
                "tool_backend_failed": tool_state.tool_backend_failed,
                "backend_failed_tool": tool_state.backend_failed_tool,
                "backend_failure_error": tool_state.backend_failure_error,
            }
        )
        trace_logger.log_summary("deep_research", summary)
        trace_logger.close()

        return AttemptResult(
            success=success,
            exit_status=exit_status,
            trace_path=trace_path,
            model_patch=result.final_content or "",
            tool_calls=[tool_action["data"] for tool_action in tool_actions],
            summary=summary,
            error=error,
            n_iterations=len(hook.llm_records),
            total_llm_ms=summary["total_llm_ms"],
            total_tool_ms=summary["total_tool_ms"],
            total_tokens=summary["total_tokens"],
            runtime_proof={
                "scaffold": "deep-research",
                "tools": registry.tool_names,
                "agent_runner": "agents.openclaw._runner.AgentRunner",
            },
            artifacts=artifacts,
        )

    def _build_scorer_provider(self) -> LLMProvider:
        if self.scorer_provider_name not in PROVIDERS:
            raise ValueError(
                f"unsupported scorer_provider: {self.scorer_provider_name}"
            )
        definition = PROVIDERS[self.scorer_provider_name]
        api_key = self.environ.get(definition.env_key, "")
        if not api_key:
            raise ValueError(
                "BrowseComp scorer requires configured scorer API key env "
                f"{definition.env_key}; evaluated-model provider is not reused"
            )
        return UnifiedProvider(
            api_key=api_key,
            api_base=definition.api_base,
            default_model=self.scorer_model,
            temperature=self.scorer_temperature,
        )

    async def _summarize_attempt(
        self,
        *,
        task: dict[str, Any],
        model_response: str,
        stop_reason: str,
        tool_state: WebToolRuntimeState,
        artifacts: list[AttemptArtifact],
        attempt_dir: Path,
        reference_answer: str,
    ) -> tuple[dict[str, Any], str | None, bool, str | None]:
        final_answer, confidence, answer_parse_error = parse_final_answer(
            model_response
        )
        summary: dict[str, Any] = {
            "final_answer": final_answer,
            "confidence": confidence,
            "answer_parse_error": answer_parse_error,
            "correct": False,
            "score": 0,
            "grader_status": "not_run",
            "grader_model": self.scorer_model,
            "grader_provider": self.scorer_provider_name,
            "grader_response": None,
        }
        if tool_state.tool_backend_failed:
            summary["grader_status"] = "not_run_tool_backend_failed"
            summary["answer_parse_error"] = (
                summary["answer_parse_error"] or "tool_backend_failed"
            )
            return (
                summary,
                "tool_backend_failed",
                False,
                tool_state.backend_failure_error or "web tool backend failed",
            )
        if stop_reason == "max_iterations":
            summary["answer_parse_error"] = (
                summary["answer_parse_error"] or "max_iterations"
            )
            return summary, "max_iterations", True, None
        if stop_reason in {
            "error",
            "empty_final_response",
            "malformed_tool_call_budget_exhausted",
            "tool_error",
        }:
            summary["answer_parse_error"] = summary["answer_parse_error"] or stop_reason
            return summary, stop_reason, False, model_response or stop_reason
        if tool_state.tool_budget_exhausted and answer_parse_error is not None:
            summary["grader_status"] = "not_run_budget_exhausted"
            return summary, "tool_budget_exhausted", True, None
        if answer_parse_error is not None:
            return summary, "completed", True, None

        try:
            grader = await self._grade(
                question=str(task["problem_statement"]),
                model_response=model_response,
                reference_answer=reference_answer,
                attempt_dir=attempt_dir,
                artifacts=artifacts,
            )
        except Exception as exc:
            summary["grader_status"] = "error"
            return summary, "grader_error", False, f"{type(exc).__name__}: {exc}"
        summary.update(grader)
        if summary.get("grader_status") != "completed":
            return summary, "grader_error", False, "grader returned malformed response"
        if tool_state.tool_budget_exhausted:
            return summary, "tool_budget_exhausted", True, None
        return summary, "completed", True, None

    async def _grade(
        self,
        *,
        question: str,
        model_response: str,
        reference_answer: str,
        attempt_dir: Path,
        artifacts: list[AttemptArtifact],
    ) -> dict[str, Any]:
        template = load_prompt_template(
            str(self.benchmark_extras["scorer_template"]),
            self.benchmark_slug,
            required_placeholders=_GRADER_PLACEHOLDERS,
        )
        prompt = _render_grader_prompt(
            template,
            question=question,
            model_response=model_response,
            reference_answer=reference_answer,
        )
        response = await self.scorer.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            model=self.scorer_model,
            temperature=self.scorer_temperature,
        )
        content = response.content or ""
        artifact_path = _write_grader_artifact(
            attempt_dir,
            {
                "prompt": prompt,
                "response": _response_to_dict(response),
                "scorer_provider": self.scorer_provider_name,
                "scorer_model": self.scorer_model,
            },
        )
        artifacts.append(
            AttemptArtifact(
                name="deep_research_grader", path=artifact_path, required=True
            )
        )
        match = _GRADER_CORRECT_RE.search(content)
        if not match:
            return {
                "grader_status": "malformed",
                "grader_response": content,
                "correct": False,
                "score": 0,
            }
        correct = match.group(1).lower() == "yes"
        return {
            "grader_status": "completed",
            "grader_response": content,
            "correct": correct,
            "score": 1 if correct else 0,
        }


def _render_grader_prompt(
    template: str,
    *,
    question: str,
    model_response: str,
    reference_answer: str,
) -> str:
    replacements = {
        "question": question,
        "model_response": model_response,
        "reference_answer": reference_answer,
    }
    return _GRADER_PLACEHOLDER_RE.sub(
        lambda match: replacements[match.group(1)],
        template,
    )


def _model_tool_result_chars(
    extras: Mapping[str, Any],
    *,
    fetch_max_chars: int,
    preview_chars: int,
) -> int:
    configured = extras.get("model_tool_result_chars")
    value = (
        int(configured)
        if configured is not None
        else int(fetch_max_chars) + _TOOL_RESULT_JSON_OVERHEAD_CHARS
    )
    if value <= 0:
        raise ValueError("model_tool_result_chars must be positive")
    return max(value, int(preview_chars))


def parse_final_answer(
    model_response: str,
) -> tuple[str | None, int | None, str | None]:
    exact_match = _EXACT_ANSWER_RE.search(model_response or "")
    if not exact_match:
        return None, None, "missing_exact_answer"
    final_answer = exact_match.group(1).strip()
    if not final_answer:
        return None, None, "empty_exact_answer"
    confidence: int | None = None
    confidence_match = _CONFIDENCE_RE.search(model_response or "")
    if confidence_match:
        confidence = int(confidence_match.group(1))
        if confidence < 0 or confidence > 100:
            return final_answer, confidence, "invalid_confidence"
    else:
        return final_answer, None, "missing_confidence"
    return final_answer, confidence, None


def _response_to_dict(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "tool_calls": [call.to_openai_tool_call() for call in response.tool_calls],
        "finish_reason": response.finish_reason,
        "usage": dict(response.usage or {}),
        "reasoning_content": response.reasoning_content,
        "thinking_blocks": response.thinking_blocks,
        "extra": _jsonable(response.extra),
    }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return repr(value)
    return value


def _spill_large_tool_results(
    records: list[ToolExecutionRecord],
    attempt_dir: Path,
    *,
    preview_chars: int,
) -> None:
    spill_dir = attempt_dir / "artifacts" / "deep_research_tool_results"
    for index, record in enumerate(records):
        text = _tool_result_text(record.tool_result)
        if len(text) <= preview_chars:
            continue
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / f"tool_{index}_{record.tool_name}.txt"
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record.artifact_path = path.relative_to(attempt_dir).as_posix()
        record.artifact_sha256 = digest
        record.original_size = len(text)
        record.preview = text[:preview_chars]
        record.truncated_preview = len(text) > preview_chars
        record.tool_result = {
            "artifact_path": record.artifact_path,
            "sha256": digest,
            "original_size": len(text),
            "preview": record.preview,
            "truncated_preview": record.truncated_preview,
        }


def _tool_spill_artifact_paths(attempt_dir: Path) -> list[Path]:
    spill_dir = attempt_dir / "artifacts" / "deep_research_tool_results"
    if not spill_dir.exists():
        return []
    return sorted(path for path in spill_dir.iterdir() if path.is_file())


def _tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)


def _write_grader_artifact(attempt_dir: Path, payload: dict[str, Any]) -> Path:
    path = attempt_dir / "artifacts" / "deep_research_grader" / "grader.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _total_llm_ms(records: list[_LLMTraceRecord]) -> float:
    return sum(max(0.0, record.ts_end - record.ts_start) * 1000.0 for record in records)


def _trailing_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    index = len(messages) - 1
    while index >= 0 and messages[index].get("role") == "tool":
        results.append(messages[index])
        index -= 1
    results.reverse()
    return results


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_now(value: Any) -> float:
    return _float_or_default(value, time.time())


def _combined_tool_actions(
    web_records: list[ToolExecutionRecord],
    message_records: list[_ToolMessageTraceRecord],
) -> list[dict[str, Any]]:
    web_by_call_id = {
        record.tool_call_id: record
        for record in web_records
        if record.tool_call_id is not None
    }
    seen_call_ids: set[str] = set()
    actions: list[dict[str, Any]] = []
    for index, message_record in enumerate(message_records):
        web_record = web_by_call_id.get(message_record.tool_call_id)
        if web_record is not None:
            seen_call_ids.add(message_record.tool_call_id)
            iteration = (
                web_record.iteration
                if web_record.iteration is not None
                else message_record.iteration
            )
            payload = web_record.to_trace_payload()
            actions.append(
                {
                    "iteration": iteration,
                    "action_suffix": web_record.tool_call_id
                    or f"{index}_{web_record.tool_name}",
                    "ts_start": web_record.ts_start,
                    "ts_end": web_record.ts_end,
                    "data": payload,
                }
            )
            continue
        actions.append(
            {
                "iteration": message_record.iteration,
                "action_suffix": message_record.tool_call_id
                or f"{index}_{message_record.tool_name}",
                "ts_start": message_record.ts_start,
                "ts_end": message_record.ts_end,
                "data": message_record.to_trace_payload(),
            }
        )

    for index, web_record in enumerate(web_records):
        if web_record.tool_call_id is not None and web_record.tool_call_id in seen_call_ids:
            continue
        iteration = web_record.iteration if web_record.iteration is not None else index
        actions.append(
            {
                "iteration": iteration,
                "action_suffix": web_record.tool_call_id
                or f"{index}_{web_record.tool_name}",
                "ts_start": web_record.ts_start,
                "ts_end": web_record.ts_end,
                "data": web_record.to_trace_payload(),
            }
        )
    return actions


def _total_tool_ms_from_actions(actions: list[dict[str, Any]]) -> float:
    return sum(
        max(0.0, float(action["ts_end"]) - float(action["ts_start"])) * 1000.0
        for action in actions
    )


def _tool_ms_by_name_from_actions(actions: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for action in actions:
        data = action.get("data") or {}
        name = str(data.get("tool_name") or "unknown")
        totals[name] = (
            totals.get(name, 0.0)
            + max(0.0, float(action["ts_end"]) - float(action["ts_start"])) * 1000.0
        )
    return totals


def _total_tokens(usage: Mapping[str, Any]) -> int:
    total_tokens = _int_usage_value(usage.get("total_tokens"))
    if total_tokens:
        return total_tokens
    return _int_usage_value(usage.get("prompt_tokens")) + _int_usage_value(
        usage.get("completion_tokens")
    )


def _int_usage_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
