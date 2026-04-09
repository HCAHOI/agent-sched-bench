"""SWE-bench evaluation runner backed by the OpenClaw container session loop."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from agents.openclaw._session_runner import (
    SessionRunner,
    TraceCollectorHook,
    inject_event_callbacks,
)
from agents.openclaw.eval.types import EvalResult, EvalTask
from agents.openclaw.providers.base import LLMProvider

__all__ = [
    "SWEBenchRunner",
    "TraceCollectorHook",
    "inject_event_callbacks",
]

def _count_trace_iterations(trace_file: Path) -> int:
    if not trace_file.exists():
        return 0
    iterations: set[int] = set()
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "action" and rec.get("action_type") == "llm_call":
            it = rec.get("iteration")
            if isinstance(it, int):
                iterations.add(it)
    return len(iterations)

class SWEBenchRunner:
    """Run one repo-backed SWE task through the containerized OpenClaw loop."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace_base: Path,
        mcp_servers: dict | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.workspace_base = Path(workspace_base).resolve()
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.model = model or provider.get_default_model()

        self._session_runner = SessionRunner(
            provider,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            mcp_servers=self.mcp_servers,
        )

    @staticmethod
    def _build_swe_bench_prompt(
        problem_statement: str,
        *,
        prompt_template: str,
    ) -> str:
        """Render the SWE prompt from the configured external template."""
        from trace_collect.prompt_loader import load_prompt_template, render_prompt

        return render_prompt(
            load_prompt_template(prompt_template),
            problem_statement,
        )

    async def run_task(
        self,
        task: EvalTask,
        *,
        container_workspace: Any,
        prompt_template: str,
        tool_workspace: Path | None = None,
        exec_working_dir: str | None = None,
        trace_file: Path | None = None,
    ) -> EvalResult:
        """Run a single evaluation task inside the prepared task container."""
        ws = task.workspace_dir
        ws.mkdir(parents=True, exist_ok=True)

        effective_trace_file = trace_file or (ws / "trace.jsonl")
        effective_tool_workspace = tool_workspace or ws

        prompt_text = self._build_swe_bench_prompt(
            task.problem_statement, prompt_template=prompt_template
        )
        session_key = f"eval:{task.instance_id}"
        result = await self._session_runner.run(
            prompt=prompt_text,
            workspace=ws,
            tool_workspace=effective_tool_workspace,
            session_key=session_key,
            trace_file=effective_trace_file,
            instance_id=task.instance_id,
            channel="cli",
            prepare_ms=None,
            container_workspace=container_workspace,
        )

        content = result.content
        tools_used: list[str] = []
        tool_events: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        n_iterations = _count_trace_iterations(effective_trace_file)

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

        # Container-backed runs: extract the patch from inside the container
        # BEFORE the caller tears it down. The host workspace is empty in
        # this mode, so _extract_patch_from_workspace would return "".
        # Note: we do NOT pass ``:(exclude)...`` pathspecs via bash because
        # the parentheses are shell metacharacters; openclaw's runtime
        # artifacts (trace.jsonl, memory, etc.) are all written to the host
        # attempt directory, not to /testbed, so no excludes are required.
        container_patch: str | None = None
        diff_cwd = exec_working_dir or "/testbed"
        if container_workspace is not None:
            try:
                diff_cmd = (
                    f"cd {diff_cwd} && "
                    f"git config --add safe.directory {diff_cwd} 2>/dev/null; "
                    "git add -A 2>/dev/null; "
                    "git diff HEAD"
                )
                rc, stdout, stderr = await container_workspace.exec(
                    diff_cmd, timeout=60
                )
                if rc == 0:
                    container_patch = stdout.strip() or None
                else:
                    logger.warning(
                        "Container git diff failed for {id}: rc={rc} stderr={stderr}",
                        id=task.instance_id,
                        rc=rc,
                        stderr=stderr[:200],
                    )
            except Exception as exc:
                logger.warning(
                    "Container patch extraction raised for {id}: {exc}",
                    id=task.instance_id,
                    exc=exc,
                )
        elif exec_working_dir is not None:
            try:
                subprocess.run(
                    ["git", "config", "--add", "safe.directory", diff_cwd],
                    cwd=diff_cwd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=diff_cwd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                diff_result = subprocess.run(
                    ["git", "diff", "HEAD", "--", "."],
                    cwd=diff_cwd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if diff_result.returncode == 0:
                    container_patch = diff_result.stdout.strip() or None
                else:
                    logger.warning(
                        "Local patch extraction failed for {id}: rc={rc} stderr={stderr}",
                        id=task.instance_id,
                        rc=diff_result.returncode,
                        stderr=diff_result.stderr[:200],
                    )
            except Exception as exc:
                logger.warning(
                    "Local patch extraction raised for {id}: {exc}",
                    id=task.instance_id,
                    exc=exc,
                )

        return EvalResult(
            instance_id=task.instance_id,
            content=content,
            tools_used=tools_used,
            usage=usage,
            stop_reason="completed",
            error=None,
            tool_events=tool_events,
            trace_file=effective_trace_file,
            prepare_ms=None,
            run_ms=result.elapsed_s * 1000,
            workspace_dir=ws,
            base_commit=task.base_commit,
            container_model_patch=container_patch,
            n_iterations=n_iterations,
        )
