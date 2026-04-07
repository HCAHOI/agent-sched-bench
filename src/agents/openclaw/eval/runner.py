"""SWE-bench evaluation runner — composes SessionRunner for scheduling.

Adds SWE-bench specific logic on top of the shared bus-based dispatch:
- prepare_workspace() for git clone + checkout
- EvalResult assembly with model_patch extraction
- run_batch() for concurrent multi-task evaluation

The scheduling core (MessageBus + AgentLoop + ResultCollector + trace
hooks) lives in ``agents.openclaw._session_runner``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from agents.openclaw._session_runner import (
    SessionRunner,
    TraceCollectorHook,
    inject_event_callbacks,
)
from agents.openclaw.eval.prepare import prepare_workspace
from agents.openclaw.eval.types import EvalResult, EvalTask
from agents.openclaw.providers.base import LLMProvider

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark

__all__ = [
    "SWEBenchRunner",
    "TraceCollectorHook",
    "inject_event_callbacks",
]


class SWEBenchRunner:
    """Runs SWE-bench tasks through OpenClaw's full bus-based scheduling.

    Composes ``SessionRunner`` for the dispatch phase, adding:
    - ``prepare_workspace()`` (git clone + checkout)
    - ``EvalResult`` assembly (tools_used, model_patch, etc.)
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace_base: Path,
        mcp_servers: dict | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
        repos_root: Path | None = None,
        benchmark: "Benchmark | None" = None,
    ) -> None:
        self.provider = provider
        self.workspace_base = Path(workspace_base).resolve()
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.model = model or provider.get_default_model()
        self.repos_root = repos_root
        self.benchmark = benchmark

        self._session_runner = SessionRunner(
            provider,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            mcp_servers=self.mcp_servers,
        )

    @staticmethod
    def _build_swe_bench_prompt(problem_statement: str) -> str:
        """Wrap problem_statement with SWE-bench task template.

        Aligned with mini-swe-agent's _INSTANCE_TEMPLATE for fair comparison,
        adapted for openclaw's structured tool set.
        """
        return f"""\
<pr_description>
Consider the following PR description:
{problem_statement}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer fixing a bug or implementing a feature.
Your task is to make changes to non-test files in the current working
directory to resolve the issue described in the PR description in a
way that is general and consistent with the codebase.

## Recommended Workflow

1. Analyse the codebase by finding and reading relevant files.
2. Create a script to reproduce the issue (use the exec tool).
3. Edit the source code to resolve the issue.
4. Verify your fix works by running your reproduction script again.
5. Test edge cases to ensure your fix is robust.

## Constraints

- MODIFY: Regular source code files in the current working directory.
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.).

## When Done

Stop after verifying your fix. The system will automatically extract
your changes as a git patch from the workspace.
</instructions>"""

    async def run_task(self, task: EvalTask) -> EvalResult:
        """Run a single evaluation task.

        Lifecycle:
        1. prepare_workspace() — git clone + checkout + pip install (if needed)
        2. SessionRunner.run() — full bus-based scheduling
        3. Extract result + model_patch from agent output
        """
        ws = task.workspace_dir
        ws.mkdir(parents=True, exist_ok=True)

        trace_file = ws / "trace.jsonl"

        # Phase 1: Prepare workspace (git clone + checkout)
        prepare_ms: float | None = None
        if task.needs_prepare:
            try:
                prepare_ms = await prepare_workspace(
                    ws,
                    repo=task.repo,
                    base_commit=task.base_commit,
                    repos_root=self.repos_root,
                )
            except Exception as e:
                logger.error("Prepare failed for {id}: {e}", id=task.instance_id, e=e)
                return EvalResult(
                    instance_id=task.instance_id,
                    content=None,
                    stop_reason="prepare_error",
                    error=str(e),
                    prepare_ms=prepare_ms,
                    trace_file=trace_file,
                    workspace_dir=ws,
                    base_commit=task.base_commit,
                )
        else:
            logger.info(
                "Skipping prepare for {id} (no repo/commit)",
                id=task.instance_id,
            )

        # Phase 2: Run via SessionRunner (full bus dispatch)
        session_key = f"eval:{task.instance_id}"
        result = await self._session_runner.run(
            prompt=self._build_swe_bench_prompt(task.problem_statement),
            workspace=ws,
            session_key=session_key,
            trace_file=trace_file,
            instance_id=task.instance_id,
            channel="cli",
            prepare_ms=prepare_ms,
        )

        # Phase 3: Build EvalResult from session history
        content = result.content
        tools_used: list[str] = []
        tool_events: list[dict[str, Any]] = []
        usage: dict[str, int] = {}

        if result.session_manager is not None:
            session = result.session_manager.get_or_create(session_key)
            for m in session.messages:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        tools_used.append(tc.get("name", ""))
                if m.get("role") == "tool":
                    tool_events.append(
                        {
                            "name": m.get("name", ""),
                            "status": "ok"
                            if not str(m.get("content", "")).startswith("Error")
                            else "error",
                            "detail": str(m.get("content", ""))[:200],
                        }
                    )

            if content is None:
                for m in reversed(session.messages):
                    if m.get("role") == "assistant" and not m.get("tool_calls"):
                        content = m.get("content")
                        break

        return EvalResult(
            instance_id=task.instance_id,
            content=content,
            tools_used=tools_used,
            usage=usage,
            stop_reason="completed",
            error=None,
            tool_events=tool_events,
            trace_file=trace_file,
            prepare_ms=prepare_ms,
            run_ms=result.elapsed_s * 1000,
            workspace_dir=ws,
            base_commit=task.base_commit,
        )

    async def run_batch(
        self,
        tasks: list[EvalTask],
        max_concurrent: int = 1,
        on_progress: Callable[[EvalResult], None] | None = None,
    ) -> list[EvalResult]:
        """Run multiple tasks concurrently using a semaphore."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[EvalResult | None] = [None] * len(tasks)

        async def _run_one(idx: int, task: EvalTask) -> None:
            async with semaphore:
                try:
                    result = await self.run_task(task)
                    results[idx] = result
                    if on_progress:
                        on_progress(result)
                except Exception as e:
                    logger.error("Task {} failed: {}", task.instance_id, e)
                    results[idx] = EvalResult(
                        instance_id=task.instance_id,
                        content=None,
                        stop_reason="error",
                        error=str(e),
                    )
                    if on_progress:
                        on_progress(results[idx])

        await asyncio.gather(*[_run_one(i, t) for i, t in enumerate(tasks)])
        return [r for r in results if r is not None]
