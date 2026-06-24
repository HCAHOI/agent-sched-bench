"""SWE-bench evaluation runner backed by the OpenClaw container session loop."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from agents.openclaw.config.schema import ExecToolConfig
from agents.openclaw._session_runner import (
    SessionRunner,
    TraceCollectorHook,
    inject_event_callbacks,
)
from agents.openclaw.eval.types import EvalResult, EvalTask, git_diff_excluding
from llm_call.provider_base import LLMProvider

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
        benchmark_slug: str,
        mcp_servers: dict | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
        exec_path_append: str = "",
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        del generation_config
        self.provider = provider
        self.workspace_base = Path(workspace_base).resolve()
        self.benchmark_slug = benchmark_slug
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.model = model or provider.get_default_model()
        self.exec_path_append = exec_path_append

        self._session_runner = SessionRunner(
            provider,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            mcp_servers=self.mcp_servers,
            exec_config=ExecToolConfig(path_append=self.exec_path_append),
        )

    @staticmethod
    def _read_submitted_patch(diff_cwd: str) -> str:
        patch_path = Path(diff_cwd) / "patch.txt"
        if not patch_path.exists():
            return ""
        content = patch_path.read_text(encoding="utf-8")
        return content.strip() if content.lstrip().startswith("diff --git") else ""

    @staticmethod
    def _extract_container_patch(
        diff_cwd: str,
        *,
        base_commit: str | None,
    ) -> str | None:
        submitted_patch = SWEBenchRunner._read_submitted_patch(diff_cwd)
        if submitted_patch:
            return submitted_patch

        subprocess.run(
            ["git", "config", "--add", "safe.directory", diff_cwd],
            cwd=diff_cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        # git_diff_excluding swallows TimeoutExpired internally (→ rc=124); any
        # other exception is a real fault and must fail fast, not be masked as
        # an empty patch (which SWE-bench would score as a spurious unresolved).
        diff_result = git_diff_excluding(
            diff_cwd,
            base_commit,
            EvalResult.exclude_pathspecs(),
            add_excludes=True,
            # QEMU-emulated x86 on arm64 makes git pathstat ~5-10x slower; the
            # agent's venv/egg-info can add thousands of files. 180s so a
            # slow-but-completing diff is not turned into a hard task failure.
            add_timeout=180,
            diff_timeout=180,
        )
        if diff_result.returncode == 0:
            return diff_result.stdout.strip() or None
        logger.warning(
            "Local patch extraction failed for cwd={cwd}: rc={rc} stderr={stderr}",
            cwd=diff_cwd,
            rc=diff_result.returncode,
            stderr=diff_result.stderr[:200],
        )
        return None

    def _build_swe_bench_prompt(
        self,
        problem_statement: str,
        *,
        prompt_template: str,
    ) -> str:
        """Render the SWE prompt from the configured external template."""
        from trace_collect.prompt_loader import load_prompt_template, render_prompt

        return render_prompt(
            load_prompt_template(prompt_template, self.benchmark_slug),
            problem_statement,
        )

    async def run_task(
        self,
        task: EvalTask,
        *,
        prompt_template: str,
        tool_workspace: Path | None = None,
        exec_working_dir: str | None = None,
        trace_file: Path | None = None,
    ) -> EvalResult:
        """Run a single evaluation task inside the prepared task container."""
        ws = task.workspace_dir
        ws.mkdir(parents=True, exist_ok=True)

        # Trace output must land outside the git checkout, not inside it.
        if trace_file is not None:
            effective_trace_file = Path(trace_file)
        else:
            logger.warning("run_task called without trace_file; using workspace sibling")
            effective_trace_file = ws.parent / f"{ws.name}-openclaw-trace.jsonl"
        effective_tool_workspace = tool_workspace or ws
        effective_project_workspace = effective_tool_workspace
        # Runtime dir sits next to the trace file, outside the target checkout.
        effective_runtime_dir = effective_trace_file.parent / "openclaw-runtime"

        prompt_text = self._build_swe_bench_prompt(
            task.problem_statement, prompt_template=prompt_template
        )
        session_key = f"eval:{task.instance_id}"
        result = await self._session_runner.run(
            prompt=prompt_text,
            workspace=ws,
            tool_workspace=effective_tool_workspace,
            project_workspace=effective_project_workspace,
            session_key=session_key,
            trace_file=effective_trace_file,
            runtime_dir=effective_runtime_dir,
            instance_id=task.instance_id,
            channel="cli",
            prepare_ms=None,
        )

        content = result.content
        tools_used: list[str] = []
        tool_events: list[dict[str, Any]] = []
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

        container_patch: str | None = None
        diff_cwd = exec_working_dir or "/testbed"
        if exec_working_dir is not None:
            container_patch = self._extract_container_patch(
                diff_cwd,
                base_commit=task.base_commit,
            )

        return EvalResult(
            instance_id=task.instance_id,
            content=content,
            tools_used=tools_used,
            usage={},
            stop_reason=result.stop_reason,
            error=result.error,
            tool_events=tool_events,
            trace_file=effective_trace_file,
            prepare_ms=None,
            run_ms=result.elapsed_s * 1000,
            workspace_dir=ws,
            base_commit=task.base_commit,
            container_model_patch=container_patch,
            n_iterations=n_iterations,
        )
