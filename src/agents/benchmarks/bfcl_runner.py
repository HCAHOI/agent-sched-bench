"""BFCL v4 runner — routes through openclaw ``SessionRunner`` with a
per-task custom :class:`ToolRegistry`.

Each BFCL task ships its own JSON-Schema tool spec (``task.tools``) that
the model must call against. In v2 the runner builds a
``BFCLNoOpTool``-backed registry via
:func:`agents.benchmarks.bfcl_tools.build_bfcl_tool_registry`, hands it
to ``SessionRunner.run(tools=...)`` via the Phase 0 extension point,
and lets the full openclaw scheduling path (bus dispatch, concurrency
gate, context window manager, hook events) handle the LLM turn and
tool dispatch. After the session completes the runner reads the
recorder list, scores the collected tool calls against
``task.ground_truth`` via :meth:`_ast_match`, and returns an
:class:`EvalResult`.

v2 routes single-turn BFCL through the SAME openclaw path as SWE-patch
benchmarks, so BFCL traces now emit ``llm_call_start/end``,
``tool_exec_start/end``, and session-level scheduling events instead
of the v1 bypass's single ``llm_call`` action. This gives BFCL tasks
parity with SWE-bench for scheduling research.

AST-match rules and known v1 limitations (no ``dont_care`` wildcard,
no recursive nested-dict matching) are documented in
``docs/benchmark_plugin_spec.md §10.4``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.benchmarks.bfcl_tools import build_bfcl_tool_registry
from agents.openclaw._session_runner import SessionRunner
from agents.openclaw.eval.types import EvalResult, EvalTask

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark
    from agents.openclaw.providers.base import LLMProvider

logger = logging.getLogger(__name__)

#: Categories scored as "correct iff predicted call list is empty".
_IRRELEVANCE_CATEGORIES: frozenset[str] = frozenset({"irrelevance", "live_irrelevance"})

#: Injected when the task's first turn has no system message.
_DEFAULT_BFCL_SYSTEM_PROMPT: str = (
    "You are a function-calling assistant. Invoke the provided tools to "
    "answer the user's request."
)


class BFCLRunner:
    """Runs a single BFCL v4 task through openclaw SessionRunner + AST scoring.

    Non-reentrant: each ``run_task`` call builds its own per-task
    ``ToolRegistry`` + recorder list and drives a fresh session, so
    concurrent calls on the same BFCLRunner instance are safe.
    """

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
        self.model = model or (
            provider.get_default_model() if provider is not None else "unknown"
        )
        # Single-turn BFCL: the model emits tool_calls on its first
        # response and we stop. max_iterations=1 is the honest bound.
        # If a BFCLv3-style multi-turn category is later re-enabled it
        # will need a higher limit, but v2's scope is single-turn only.
        effective_max_iterations = (
            max_iterations if max_iterations is not None else 1
        )
        if effective_max_iterations < 1:
            raise ValueError(
                f"BFCLRunner requires max_iterations >= 1, got "
                f"{effective_max_iterations}; the LLM must be allowed at "
                f"least one turn to emit tool calls."
            )
        self._session_runner = SessionRunner(
            provider,
            model=self.model,
            max_iterations=effective_max_iterations,
            context_window_tokens=context_window_tokens,
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
        """AST-level comparison of predicted vs ground-truth function calls."""
        if category in _IRRELEVANCE_CATEGORIES:
            # Model must correctly abstain. Both sides empty by BFCL schema;
            # the pair-check catches upstream schema drift loudly.
            return len(predicted) == 0 and len(ground_truth) == 0

        if len(predicted) != len(ground_truth):
            return False

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
    # Trace aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sum_usage_from_trace(trace_file: Path) -> dict[str, int]:
        """Sum prompt/completion tokens across all llm_call actions in a trace.

        BFCL single-turn tasks emit exactly one llm_call action so this is
        effectively a read of that one record, but the implementation
        handles multi-action traces correctly for forward compatibility.
        """
        if not trace_file.exists():
            return {}
        prompt_tokens = 0
        completion_tokens = 0
        for lineno, line in enumerate(
            trace_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "BFCL: skipping malformed trace line %s:%d (%s)",
                    trace_file,
                    lineno,
                    exc,
                )
                continue
            if rec.get("type") == "action" and rec.get("action_type") == "llm_call":
                data = rec.get("data", {})
                prompt_tokens += int(data.get("prompt_tokens", 0) or 0)
                completion_tokens += int(data.get("completion_tokens", 0) or 0)
        usage: dict[str, int] = {}
        if prompt_tokens or completion_tokens:
            usage["prompt_tokens"] = prompt_tokens
            usage["completion_tokens"] = completion_tokens
        return usage

    @staticmethod
    def _extract_absorbed_llm_error(trace_file: Path) -> str | None:
        """Return the error message of the first absorbed LLM error, if any.

        When ``provider.chat()`` raises inside ``SessionRunner``, openclaw's
        agent runner catches it, wraps the response as
        ``finish_reason='error'``, and emits an ``llm_error`` trace event.
        The session completes normally with an empty recorder, so the
        runner cannot tell "model produced wrong answer" from "model
        crashed" by inspecting the return value alone. Walk the trace
        once looking for the error event and lift its message into
        ``EvalResult.error`` so ``results.jsonl`` can distinguish the
        two failure modes without re-walking the trace downstream.
        """
        if not trace_file.exists():
            return None
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "event" and rec.get("event") == "llm_error":
                # TraceCollectorHook emits llm_error with shape
                # {"error_message": str, "finish_reason": str, ...extras}
                # See agents.openclaw._session_runner.py:285 for the canonical
                # field layout.
                data = rec.get("data", {})
                msg = (
                    data.get("error_message")
                    or data.get("error")
                    or data.get("content")
                    or data.get("message")
                )
                if msg:
                    return str(msg)
                return "llm_error event present but no message field"
        return None

    # ------------------------------------------------------------------
    # Prompt flattening
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_single_turn_question(task: EvalTask) -> str:
        """Collapse BFCL's nested question-turn list into one prompt string.

        Single-turn categories have ``task.question = [[user_msg]]`` or
        ``[[system_msg, user_msg]]``. We concatenate the user-role
        messages (the system prompt, if any, is folded into the
        session's own system setup — openclaw's SessionRunner injects
        its own system context via ``ContextBuilder``). If the task has
        no question turns, fall back to ``task.problem_statement``.
        """
        if not task.question:
            return task.problem_statement or ""

        first_turn = task.question[0] if isinstance(task.question[0], list) else []
        user_parts: list[str] = []
        for msg in first_turn:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and content:
                user_parts.append(str(content))
        if user_parts:
            return "\n\n".join(user_parts)
        return task.problem_statement or ""

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run_task(self, task: EvalTask) -> EvalResult:
        """Run one BFCL task through SessionRunner + score via AST match."""
        ws = task.workspace_dir
        ws.mkdir(parents=True, exist_ok=True)
        trace_file = ws / "trace.jsonl"

        # Build per-task BFCL registry + recorder (non-reentrant contract —
        # local to this run_task call).
        registry, recorder = build_bfcl_tool_registry(task.tools)

        prompt = self._flatten_single_turn_question(task)
        session_key = f"eval:{task.instance_id}"

        try:
            session_result = await self._session_runner.run(
                prompt=prompt,
                workspace=ws,
                session_key=session_key,
                trace_file=trace_file,
                instance_id=task.instance_id,
                channel="cli",
                tools=registry,
            )
        except Exception as exc:
            logger.exception(
                "BFCL SessionRunner.run failed for %s", task.instance_id
            )
            return EvalResult(
                instance_id=task.instance_id,
                content=None,
                stop_reason="session_error",
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

        # The recorder captured each tool call the LLM emitted during
        # the session (via BFCLNoOpTool.execute). Each entry is already
        # shaped as {"name": str, "arguments": dict} — the exact input
        # format _ast_match expects.
        predicted_calls = list(recorder)
        tools_used = [c["name"] for c in predicted_calls]

        resolved = self._ast_match(
            predicted_calls,
            task.ground_truth,
            category=task.category or "",
        )

        run_ms = session_result.elapsed_s * 1000 if session_result else 0.0
        usage = self._sum_usage_from_trace(trace_file)
        absorbed_error = self._extract_absorbed_llm_error(trace_file)

        return EvalResult(
            instance_id=task.instance_id,
            content=json.dumps(predicted_calls, ensure_ascii=False),
            tools_used=tools_used,
            usage=usage,
            stop_reason="completed",
            error=absorbed_error,
            trace_file=trace_file,
            run_ms=run_ms,
            workspace_dir=ws,
            official_resolved=resolved,
            evaluation_report={
                "category": task.category,
                "score": 1.0 if resolved else 0.0,
                "predicted_calls": predicted_calls,
                "ground_truth": task.ground_truth,
            },
        )
