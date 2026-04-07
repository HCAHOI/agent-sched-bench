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

v1 scope: single-turn categories only (simple_python/java/javascript,
multiple, parallel, parallel_multiple, live_simple/multiple/parallel/
parallel_multiple, irrelevance, live_relevance, live_irrelevance). Each
of these asks the model for ONE set of function calls in response to a
single user message; we therefore make exactly one LLM call per task.
Multi-turn / memory / web-search / format-sensitivity categories require
a stateful tool simulator and are out of scope for this runner — the
plugin's :meth:`~agents.benchmarks.bfcl_v4.BFCLv4Benchmark.load_tasks`
filters them out at load time.

AST match rules — **v1 subset** of the documented BFCL rules (see
``bfcl-eval`` PyPI package for the full reference implementation; not a
hard dependency of this repo):

1. Function name must match exactly.
2. Every required argument must be present with an exact value match.
   Ground-truth values are lists of acceptable alternatives — the
   predicted value matches if it equals any alternative.
3. Optional arguments (not listed in ground truth) may be omitted.
4. All-or-nothing per call — no partial credit.
5. Categories with multiple expected calls ("parallel", "multiple",
   "parallel_multiple") compare as sets of calls.
6. The ``irrelevance`` and ``live_irrelevance`` categories are correct
   iff the predicted call list is empty (the model must NOT invoke any
   tool).

**Known limitations (v1)** that diverge from the full BFCL spec:

- **No ``dont_care`` wildcard**. Some BFCL ground-truth rows use an
  empty list ``[]`` or the sentinel string ``"[don't care]"`` to
  signal "any value is acceptable". This runner treats those literally:
  empty-list alternatives never match anything, and the sentinel string
  matches only itself. This affects multi-language (simple_java,
  simple_javascript) tasks where syntactic values can't be enumerated.
- **No recursive nested-dict matching**. If a ground-truth argument is
  a dict whose values are themselves lists of alternatives, this
  runner compares the top-level dict by equality instead of recursing.
  In practice this bites only nested-parameter tasks.

When the ``bfcl-eval`` package is available on the Python path, prefer
its ``ast_checker`` over this in-process implementation by setting
``use_bfcl_eval=True`` on :class:`BFCLRunner` (future work). This is
tracked in the Phase 4 follow-ups.
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

#: Categories where the model must correctly abstain — scored as
#: "correct iff predicted call list is empty". Centralized as a module
#: constant so the shortcut applies uniformly to every irrelevance
#: variant listed in :attr:`agents.benchmarks.bfcl_v4.BFCLv4Benchmark._SUPPORTED_CATEGORIES`.
_IRRELEVANCE_CATEGORIES: frozenset[str] = frozenset({"irrelevance", "live_irrelevance"})

#: Default system prompt injected when the task's first turn has no
#: system message. Extracted as a constant so the prompt isn't silently
#: duplicated across code paths.
_DEFAULT_BFCL_SYSTEM_PROMPT: str = (
    "You are a function-calling assistant. Invoke the provided tools to "
    "answer the user's request."
)


class BFCLRunner:
    """Runs a single BFCL v4 task through one LLM call + AST scoring."""

    def __init__(
        self,
        provider: "LLMProvider",
        workspace_base: Path,
        *,
        benchmark: "Benchmark | None" = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.workspace_base = Path(workspace_base).resolve()
        self.benchmark = benchmark
        # max_iterations + context_window_tokens are accepted for trace
        # metadata stamping (so Gantt / inspector render BFCL traces
        # consistently with SWE-patch traces) but are not enforced as
        # hard limits: single-turn BFCL makes exactly one LLM call per
        # task, so there is no loop to bound.
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
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
            category: Task category. Triggers the irrelevance shortcut
                for :data:`_IRRELEVANCE_CATEGORIES`.

        Returns:
            True iff the prediction is correct under BFCL's rules.
        """
        if category in _IRRELEVANCE_CATEGORIES:
            # Irrelevance / live_irrelevance: the model must correctly
            # abstain. Ground truth is empty for these categories per
            # BFCL schema; we assert both to catch any upstream schema
            # drift (e.g., a hypothetical "negative example" ground
            # truth) loudly rather than silently miscategorizing.
            return len(predicted) == 0 and len(ground_truth) == 0

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
        first turn contains no system message, prepend the default
        function-calling system prompt (see :data:`_DEFAULT_BFCL_SYSTEM_PROMPT`).
        """
        if not task.question:
            return [
                {"role": "system", "content": _DEFAULT_BFCL_SYSTEM_PROMPT},
                {"role": "user", "content": task.problem_statement or ""},
            ]

        first_turn = task.question[0] if isinstance(task.question[0], list) else []
        messages = list(first_turn)
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(
                0,
                {"role": "system", "content": _DEFAULT_BFCL_SYSTEM_PROMPT},
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
                # Loud warning rather than silent drop — CLAUDE.md §4
                # research integrity "no silent failures".
                logger.warning(
                    "BFCLRunner: dropping non-dict tool entry (got %s): %r",
                    type(fn).__name__,
                    fn,
                )
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
        # In production ``benchmark`` is always non-None (the plugin's
        # ``build_runner`` passes ``benchmark=self``). The ``"unknown"``
        # fallback exists solely for unit tests that stub the provider
        # without attaching a benchmark — never silently invent a
        # plausible-looking slug/split pair.
        benchmark_slug = (
            self.benchmark.config.slug
            if self.benchmark is not None
            else "unknown"
        )
        benchmark_split = (
            self.benchmark.config.harness_split
            if self.benchmark is not None
            else "unknown"
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

        # For single-turn BFCL, elapsed_s == llm_latency (one LLM call,
        # no tool dispatch loop). n_steps=1 is the honest count: the
        # runner performed exactly one llm_call action — reporting 0
        # would lie to downstream steps-per-task aggregates.
        trace_logger.log_summary(
            task.instance_id,
            {
                "instance_id": task.instance_id,
                "category": task.category,
                "n_steps": 1,
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
