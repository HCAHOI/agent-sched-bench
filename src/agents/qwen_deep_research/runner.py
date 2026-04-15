"""Qwen Deep Research runner backed by an OpenAI-compatible chat API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agents.base import TraceAction
from harness.trace_logger import TraceLogger
from llm_call import create_async_openai_client
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult
from trace_collect.latency_metrics import summarize_llm_latencies


def _dump_obj(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    return {"value": repr(obj)}


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


class QwenDeepResearchRunner:
    """Run research-style benchmark tasks through a Qwen-compatible endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        max_iterations: int,
        benchmark_slug: str,
        client: Any | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.max_iterations = max_iterations
        self.benchmark_slug = benchmark_slug
        self.client = client or create_async_openai_client(
            api_base=api_base,
            api_key=api_key,
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        trace_logger = TraceLogger(attempt_ctx.attempt_dir, "trace")
        trace_logger.log_metadata(
            scaffold="qwen-deep-research",
            execution_environment="host",
            benchmark=self.benchmark_slug,
            model=self.model,
            api_base=self.api_base,
            max_iterations=self.max_iterations,
            instance_id=attempt_ctx.instance_id,
            prompt_template=prompt_template,
            agent_runtime_mode=attempt_ctx.agent_runtime_mode,
            scaffold_capabilities={
                "tools": [],
                "memory": False,
                "skills": False,
                "file_ops": "none",
            },
        )

        messages = self._build_messages(task, prompt_template=prompt_template)
        try:
            call = await self._call_streaming(messages)
            action = TraceAction(
                action_type="llm_call",
                action_id="llm_0",
                agent_id=attempt_ctx.instance_id,
                program_id=attempt_ctx.instance_id,
                instance_id=attempt_ctx.instance_id,
                iteration=0,
                ts_start=call["ts_start"],
                ts_end=call["ts_end"],
                data={
                    "messages_in": messages,
                    "llm_output": call["content"],
                    "raw_response": call["raw_response"],
                    "prompt_tokens": call["prompt_tokens"],
                    "completion_tokens": call["completion_tokens"],
                    "llm_latency_ms": call["llm_latency_ms"],
                    "llm_call_time_ms": call["llm_latency_ms"],
                    "llm_wall_latency_ms": call["llm_latency_ms"],
                    "llm_timing_source": "stream_wall_clock_ms",
                    "ttft_ms": call["ttft_ms"],
                    "tpot_ms": call["tpot_ms"],
                    "finish_reason": call["finish_reason"],
                    "model": self.model,
                },
            )
            trace_logger.log_trace_action(attempt_ctx.instance_id, action)
            llm_summary = summarize_llm_latencies([action.data])
            total_tokens = call["prompt_tokens"] + call["completion_tokens"]
            success = bool(call["content"].strip())
            summary = self._summary(
                attempt_ctx.instance_id,
                success=success,
                llm_summary=llm_summary,
                total_tokens=total_tokens,
            )
            trace_logger.log_summary(attempt_ctx.instance_id, summary)
            return AttemptResult(
                success=success,
                exit_status="completed" if success else "empty_final_response",
                trace_path=trace_logger.path,
                model_patch="",
                error=None
                if success
                else "Qwen Deep Research returned an empty response",
                n_iterations=1,
                total_llm_ms=float(llm_summary["total_llm_ms"]),
                total_tool_ms=0.0,
                total_tokens=total_tokens,
                runtime_proof=self._runtime_proof(),
            )
        except Exception as exc:
            trace_logger.log_summary(
                attempt_ctx.instance_id,
                self._summary(
                    attempt_ctx.instance_id,
                    success=False,
                    llm_summary=summarize_llm_latencies([]),
                    total_tokens=0,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise
        finally:
            trace_logger.close()

    def _build_messages(
        self,
        task: dict[str, Any],
        *,
        prompt_template: str,
    ) -> list[dict[str, str]]:
        metadata = {
            key: task[key]
            for key in ("topic", "difficulty", "domain", "source_urls")
            if task.get(key)
        }
        prompt = self._load_prompt_template(prompt_template).replace(
            "{{task}}",
            str(task["problem_statement"]),
        )
        if metadata:
            prompt += (
                "\n\nInference-time metadata:\n"
                + json.dumps(metadata, ensure_ascii=False, indent=2)
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are Qwen Deep Research. Answer the research task using "
                    "only inference-time information. Do not assume access to "
                    "reference answers."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _load_prompt_template(self, name: str) -> str:
        prompt_dir = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "prompts"
            / self.benchmark_slug.replace("-", "_")
        )
        path = prompt_dir / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template {name!r} not found at {path}"
            )
        text = path.read_text(encoding="utf-8")
        if "{{task}}" not in text:
            raise ValueError(f"Prompt template {path} is missing '{{{{task}}}}'")
        return text

    def _runtime_proof(self) -> dict[str, str]:
        return {
            "agent_runtime_mode": "host_controller",
            "benchmark": self.benchmark_slug,
            "scaffold": "qwen-deep-research",
        }

    def _summary(
        self,
        instance_id: str,
        *,
        success: bool,
        llm_summary: dict[str, float | int | str],
        total_tokens: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "success": success,
            "n_iterations": 1 if success else 0,
            "total_llm_ms": llm_summary["total_llm_ms"],
            "total_llm_wall_ms": llm_summary["total_llm_wall_ms"],
            "total_llm_call_time_ms": llm_summary["total_llm_call_time_ms"],
            "llm_call_time_count": llm_summary["llm_call_time_count"],
            "llm_timing_source": llm_summary["llm_timing_source"],
            "total_tool_ms": 0.0,
            "total_tokens": total_tokens,
        }
        if error is not None:
            summary["error"] = error
        return summary

    async def _call_streaming(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        ts_start = time.time()
        mono_start = time.monotonic()
        first_token_mono: float | None = None
        content_parts: list[str] = []
        raw_chunks: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            chunk_dict = _dump_obj(chunk)
            raw_chunks.append(chunk_dict)
            chunk_usage = _get_field(chunk, "usage") or chunk_dict.get("usage")
            if chunk_usage:
                usage = _dump_obj(chunk_usage)
            choices = _get_field(chunk, "choices") or chunk_dict.get("choices") or []
            for choice in choices:
                delta = _get_field(choice, "delta") or {}
                delta_content = _message_content_to_text(_get_field(delta, "content"))
                if delta_content:
                    if first_token_mono is None:
                        first_token_mono = time.monotonic()
                    content_parts.append(delta_content)
                choice_finish = _get_field(choice, "finish_reason")
                if choice_finish:
                    finish_reason = str(choice_finish)

        mono_end = time.monotonic()
        ts_end = time.time()
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        elapsed_ms = (mono_end - mono_start) * 1000
        ttft_ms = (
            (first_token_mono - mono_start) * 1000
            if first_token_mono is not None
            else None
        )
        tpot_ms = None
        if ttft_ms is not None and completion_tokens > 1:
            tpot_ms = max(0.0, (elapsed_ms - ttft_ms) / (completion_tokens - 1))

        return {
            "content": "".join(content_parts),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": elapsed_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "finish_reason": finish_reason,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "raw_response": {
                "stream": raw_chunks,
                "usage": usage,
                "model": self.model,
            },
        }
