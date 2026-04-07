"""BFCL v4 runner — in-process function-call task execution + AST scoring.

This runner intentionally does NOT route through
:class:`agents.openclaw._session_runner.SessionRunner`. Openclaw's
``AgentLoop`` hardcodes its own tool set (bash/file/web/memory) which is
incompatible with BFCL's JSON-Schema-provided tool specs: each BFCL task
provides its OWN list of functions that the model must call. We call
``provider.chat()`` directly with the BFCL tool spec, record the model's
structured tool_calls, and score them against the ground truth via an
AST-style comparison that mirrors the documented BFCL rules.

Trace emission still conforms to the v5 contract
(:mod:`harness.trace_logger`): one ``trace_metadata`` header, one
``llm_call`` :class:`agents.base.TraceAction` per task, and a final
``summary`` record. This keeps Gantt / inspector parsers from treating
BFCL traces differently from SWE-patch traces.

v1 scope: single-turn categories only (simple, multiple, parallel,
parallel_multiple, live_*, irrelevance, java, javascript, rest). Each of
these asks the model for ONE set of function calls in response to a
single user message; we therefore make exactly one LLM call per task.
Multi-turn / memory / web-search / format-sensitivity categories require
a stateful tool simulator and are out of scope for this runner — the
plugin's :meth:`~agents.benchmarks.bfcl_v4.BFCLv4Benchmark.load_tasks`
filters them out at load time.

AST match rules (mirrored from the BFCL blog posts and the
``bfcl-eval`` package, reimplemented because ``bfcl-eval`` is not a
hard dependency of this repo):

1. Function name must match exactly.
2. Every required argument must be present with an exact value match.
   Ground-truth values are lists of acceptable alternatives — the
   predicted value matches if it equals any alternative.
3. Optional arguments (not listed in ground truth) may be omitted.
4. All-or-nothing per call — no partial credit.
5. Categories with multiple expected calls ("parallel", "multiple",
   "parallel_multiple") compare as sets of calls.
6. The ``irrelevance`` category is correct iff the predicted call list
   is empty (the model must NOT invoke any tool).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.base import TraceAction
from agents.openclaw.eval.types import EvalResult, EvalTask
from harness.trace_logger import TraceLogger

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark
    from agents.openclaw.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class BFCLRunner:
    """Runs a single BFCL v4 task through one LLM call + AST scoring."""

    def __init__(
        self,
        provider: "LLMProvider",
        workspace_base: Path,
        *,
        benchmark: "Benchmark | None" = None,
        mcp_servers: dict | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.workspace_base = Path(workspace_base).resolve()
        self.benchmark = benchmark
        # Unused by BFCL (single-turn, no tool dispatch loop) but accepted
        # so the plugin can forward arbitrary kwargs from build_runner.
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.model = model or (
            provider.get_default_model() if provider is not None else "unknown"
        )

    # ------------------------------------------------------------------
    # AST match — pure function, heavily unit-tested
    # ------------------------------------------------------------------

    @staticmethod
    def _args_match_ground_truth(
        predicted_args: dict[str, Any], gt_args: dict[str, list[Any]]
    ) -> bool:
        """Check if predicted args satisfy the ground-truth spec for ONE call.

        ``gt_args`` maps each required parameter name to a list of
        acceptable values. The predicted args must contain every key in
        ``gt_args`` with a value that equals one of the listed
        alternatives. Predicted args may include extra keys (treated as
        optional parameters) without failing the match.
        """
        for name, alternatives in gt_args.items():
            if name not in predicted_args:
                return False
            pred_value = predicted_args[name]
            if not isinstance(alternatives, list):
                # Defensive: some ground-truth rows may wrap the single
                # acceptable value without a list. Treat as [alternative].
                alternatives = [alternatives]
            if pred_value not in alternatives:
                return False
        return True

    @classmethod
    def _single_call_matches(
        cls,
        predicted: dict[str, Any],
        gt_entry: dict[str, dict[str, list[Any]]],
    ) -> bool:
        """Check if a single predicted call matches a single ground-truth entry.

        A ground-truth entry is ``{function_name: {arg: [values]}}``.
        Matches iff the function name is identical and the predicted
        args satisfy :meth:`_args_match_ground_truth`.
        """
        if len(gt_entry) != 1:
            return False
        gt_name, gt_args = next(iter(gt_entry.items()))
        if predicted.get("name") != gt_name:
            return False
        predicted_args = predicted.get("arguments", {})
        if not isinstance(predicted_args, dict):
            return False
        return cls._args_match_ground_truth(predicted_args, gt_args)

    @classmethod
    def _ast_match(
        cls,
        predicted: list[dict[str, Any]],
        ground_truth: list[dict[str, dict[str, list[Any]]]],
        *,
        category: str = "",
    ) -> bool:
        """AST-level comparison of predicted vs ground-truth function calls.

        Args:
            predicted: List of ``{"name": str, "arguments": dict}`` calls
                that the model emitted.
            ground_truth: List of ground-truth entries, each of shape
                ``{function_name: {arg: [acceptable_values]}}``. For
                single-call categories the list has length 1; for
                parallel/multiple categories each list element is an
                independent call that must be matched.
            category: Task category, used for irrelevance handling.

        Returns:
            True iff the prediction is correct under BFCL's rules.
        """
        if category == "irrelevance":
            return len(predicted) == 0

        if len(predicted) != len(ground_truth):
            return False

        # For parallel/multiple categories, match as sets: every
        # ground-truth entry must be matched by exactly one predicted
        # call (and vice versa). We greedy-assign because entries may
        # differ by function name and by arg values.
        remaining_predicted = list(predicted)
        for gt_entry in ground_truth:
            matched_idx: int | None = None
            for idx, pred in enumerate(remaining_predicted):
                if cls._single_call_matches(pred, gt_entry):
                    matched_idx = idx
                    break
            if matched_idx is None:
                return False
            remaining_predicted.pop(matched_idx)
        return not remaining_predicted

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(task: EvalTask) -> list[dict[str, Any]]:
        """Flatten BFCL's nested question-turn list into a message list.

        Single-turn categories have ``task.question = [[user_msg]]`` or
        ``[[system_msg, user_msg]]`` — we unwrap the outer list. If the
        first turn contains no system message, prepend a minimal one
        instructing the model to emit function calls.
        """
        if not task.question:
            return [
                {
                    "role": "system",
                    "content": "You are a function-calling assistant. "
                    "Invoke the provided tools to answer the user's "
                    "request.",
                },
                {"role": "user", "content": task.problem_statement or ""},
            ]

        first_turn = task.question[0] if isinstance(task.question[0], list) else []
        messages = list(first_turn)
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a function-calling assistant. "
                    "Invoke the provided tools to answer the user's "
                    "request.",
                },
            )
        return messages

    @staticmethod
    def _to_openai_tools_schema(
        bfcl_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Wrap BFCL function definitions in OpenAI's ``tools`` envelope.

        BFCL rows ship bare ``{name, description, parameters}`` dicts; the
        OpenAI-compatible providers expect
        ``{"type": "function", "function": {...}}``.
        """
        wrapped: list[dict[str, Any]] = []
        for fn in bfcl_tools:
            if not isinstance(fn, dict):
                continue
            if "type" in fn and "function" in fn:
                wrapped.append(fn)  # Already wrapped
            else:
                wrapped.append({"type": "function", "function": fn})
        return wrapped

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run_task(self, task: EvalTask) -> EvalResult:
        """Run one BFCL task: single LLM call + AST scoring + trace emit.

        Does NOT call :func:`prepare_workspace` — BFCL tasks have no git
        repo, so ``task.needs_prepare`` is False by construction (the
        plugin's :meth:`normalize_task` leaves ``repo`` / ``base_commit``
        unset on purpose).
        """
        ws = task.workspace_dir
        ws.mkdir(parents=True, exist_ok=True)

        trace_file = ws / "trace.jsonl"
        # TraceLogger takes (output_dir, run_id) and builds
        # ``<output_dir>/<run_id>.jsonl`` — use the workspace as the
        # output dir and "trace" as the run_id so the resulting path is
        # ``ws/trace.jsonl``.
        trace_logger = TraceLogger(ws, "trace")
        benchmark_slug = (
            self.benchmark.config.slug
            if self.benchmark is not None
            else "bfcl-v4"
        )
        benchmark_split = (
            self.benchmark.config.harness_split
            if self.benchmark is not None
            else "v4"
        )
        trace_logger.log_metadata(
            scaffold="openclaw",
            mode="collect",
            model=self.model,
            benchmark=benchmark_slug,
            benchmark_split=benchmark_split,
            instance_id=task.instance_id,
            category=task.category,
            max_iterations=self.max_iterations,
            scaffold_capabilities={
                "tools": "benchmark_provided",
                "memory": False,
                "skills": False,
                "file_ops": "none",
            },
        )

        messages = self._build_messages(task)
        openai_tools = self._to_openai_tools_schema(task.tools)

        t_llm_start = time.monotonic()
        try:
            response = await self.provider.chat(
                messages=messages,
                tools=openai_tools if openai_tools else None,
                model=self.model,
                tool_choice="auto" if openai_tools else None,
            )
        except Exception as exc:
            logger.exception("BFCL LLM call failed for %s", task.instance_id)
            trace_logger.log_summary(
                task.instance_id,
                {
                    "instance_id": task.instance_id,
                    "elapsed_s": time.monotonic() - t_llm_start,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            trace_logger.close()
            return EvalResult(
                instance_id=task.instance_id,
                content=None,
                stop_reason="llm_error",
                error=f"{type(exc).__name__}: {exc}",
                trace_file=trace_file,
                workspace_dir=ws,
                official_resolved=False,
                evaluation_report={
                    "category": task.category,
                    "score": 0.0,
                    "error": str(exc),
                },
            )
        t_llm_end = time.monotonic()
        llm_latency_ms = (t_llm_end - t_llm_start) * 1000

        predicted_calls: list[dict[str, Any]] = []
        tools_used: list[str] = []
        for tc in response.tool_calls:
            predicted_calls.append({"name": tc.name, "arguments": tc.arguments})
            tools_used.append(tc.name)

        usage = response.usage or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        action = TraceAction(
            action_type="llm_call",
            action_id=f"llm_0_{task.instance_id}",
            agent_id=task.instance_id,
            program_id=task.instance_id,
            iteration=0,
            ts_start=t_llm_start,
            ts_end=t_llm_end,
            data={
                "messages_in": messages,
                "tools_in": openai_tools,
                "raw_response": {
                    "content": response.content,
                    "tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    "finish_reason": response.finish_reason,
                },
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "llm_latency_ms": llm_latency_ms,
            },
        )
        trace_logger.log_trace_action(task.instance_id, action)

        resolved = self._ast_match(
            predicted_calls,
            task.ground_truth,
            category=task.category or "",
        )

        trace_logger.log_summary(
            task.instance_id,
            {
                "instance_id": task.instance_id,
                "category": task.category,
                "elapsed_s": t_llm_end - t_llm_start,
                "total_llm_ms": llm_latency_ms,
                "total_tool_ms": 0,
                "total_tokens": prompt_tokens + completion_tokens,
                "success": resolved,
            },
        )
        trace_logger.close()

        return EvalResult(
            instance_id=task.instance_id,
            content=json.dumps(predicted_calls, ensure_ascii=False),
            tools_used=tools_used,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            stop_reason="completed",
            trace_file=trace_file,
            run_ms=llm_latency_ms,
            workspace_dir=ws,
            official_resolved=resolved,
            evaluation_report={
                "category": task.category,
                "score": 1.0 if resolved else 0.0,
                "predicted_calls": predicted_calls,
                "ground_truth": task.ground_truth,
            },
        )
