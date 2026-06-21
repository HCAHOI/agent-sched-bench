from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def git_diff_excluding(
    cwd: str | Path,
    base_commit: str | None,
    exclude_patterns: list[str],
    *,
    add_excludes: bool = False,
    add_timeout: float = 30.0,
    diff_timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Stage all changes and diff against *base_commit*, excluding runtime files.

    Returns the ``git diff`` ``CompletedProcess`` so callers keep their own
    success/empty handling. ``add_excludes`` mirrors the staging form used by
    the container runner (``git add -A -- . :(exclude)…``).
    """
    add_cmd = ["git", "add", "-A"]
    if add_excludes:
        add_cmd.append("--")
        add_cmd.append(".")
        for pat in exclude_patterns:
            add_cmd.append(f":(exclude){pat}")
    subprocess.run(
        add_cmd, cwd=cwd, capture_output=True, text=True, timeout=add_timeout,
        check=False,
    )
    diff_cmd = ["git", "diff", base_commit or "HEAD", "--", "."]
    for pat in exclude_patterns:
        diff_cmd.append(f":(exclude){pat}")
    return subprocess.run(
        diff_cmd, cwd=cwd, capture_output=True, text=True, timeout=diff_timeout,
        check=False,
    )


@dataclass
class EvalTask:
    instance_id: str
    problem_statement: str
    workspace_dir: Path

    repo: str | None = None
    base_commit: str | None = None
    image_name: str | None = None


@dataclass
class EvalResult:
    instance_id: str
    content: str | None
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    trace_file: Path | None = None
    prepare_ms: float | None = None
    run_ms: float | None = None
    workspace_dir: Path | None = None
    base_commit: str | None = None
    official_resolved: bool | None = None
    evaluation_report: dict[str, Any] | None = None
    n_iterations: int | None = None
    container_model_patch: str | None = None

    @property
    def model_patch(self) -> str:
        # Prefer the runner-captured diff when available. Otherwise fall back
        # to the workspace diff, then to content scraping.
        if self.container_model_patch:
            return self.container_model_patch
        patch = self._extract_patch_from_workspace()
        if patch:
            return patch
        return self._extract_patch_from_content()

    # Exclude runtime-owned files so the extracted patch stays source-only.
    _EXCLUDE_PATTERNS = [
        ".nanobot",
        ".openclaw",
        "openclaw-runtime",
        "memory",
        "sessions",
        "skills",
        "trace.jsonl",
        "MEMORY.md",
        "HISTORY.md",
        ".omc",
        "patch.txt",
        "local.bin",
    ]

    def _extract_patch_from_workspace(self) -> str:
        if not self.workspace_dir or not (self.workspace_dir / ".git").exists():
            return ""
        result = git_diff_excluding(
            self.workspace_dir, self.base_commit, self._EXCLUDE_PATTERNS
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def _extract_patch_from_content(self) -> str:
        if not self.content:
            return ""
        import re

        text = self.content.strip()
        if text.startswith("diff --git"):
            return text
        match = re.search(r"```diff\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(diff --git.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("diff --git"):
                return "\n".join(lines[i:]).strip()
        return ""


@dataclass
class EvalTraceSummary:
    type: str = "summary"
    agent_id: str = ""
    program_id: str = ""
    task_id: str = ""
    instance_id: str = ""
    n_iterations: int = 0
    total_llm_ms: float = 0.0
    total_llm_wall_ms: float = 0.0
    total_llm_call_time_ms: float = 0.0
    llm_call_time_count: int = 0
    llm_timing_source: str = "wall_clock_ms"
    total_tool_ms: float = 0.0
    total_tokens: int = 0
    tool_ms_by_name: dict[str, float] = field(default_factory=dict)
    tool_timeouts: dict[str, int] = field(default_factory=dict)
    success: bool | None = None
    elapsed_s: float = 0.0
    prepare_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "n_iterations": self.n_iterations,
            "total_llm_ms": self.total_llm_ms,
            "total_llm_wall_ms": self.total_llm_wall_ms,
            "total_llm_call_time_ms": self.total_llm_call_time_ms,
            "llm_call_time_count": self.llm_call_time_count,
            "llm_timing_source": self.llm_timing_source,
            "total_tool_ms": self.total_tool_ms,
            "total_tokens": self.total_tokens,
            "tool_ms_by_name": self.tool_ms_by_name,
            "tool_timeouts": self.tool_timeouts,
            "success": self.success,
            "elapsed_s": self.elapsed_s,
            "prepare_ms": self.prepare_ms,
        }


SCHEDULING = "SCHEDULING"
SESSION = "SESSION"
CONTEXT = "CONTEXT"
LLM = "LLM"
TOOL = "TOOL"
MCP = "MCP"
MEMORY = "MEMORY"
SUBAGENT = "SUBAGENT"

ALL_CATEGORIES = frozenset(
    [
        SCHEDULING,
        SESSION,
        CONTEXT,
        LLM,
        TOOL,
        MCP,
        MEMORY,
        SUBAGENT,
    ]
)


@dataclass
class EvalTraceEvent:
    agent_id: str
    program_id: str
    instance_id: str
    event: str
    category: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0
    iteration: int = 0
    type: str = "event"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "instance_id": self.instance_id,
            "event": self.event,
            "category": self.category,
            "data": self.data,
            "ts": self.ts,
            "iteration": self.iteration,
        }
