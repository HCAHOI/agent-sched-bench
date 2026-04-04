"""Data contracts for SWE-bench evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalTask:
    """A single SWE-bench instance to evaluate.

    Mirrors the official SWE-bench instance schema plus derived fields.
    """

    instance_id: str
    problem_statement: str
    workspace_dir: Path

    # SWE-bench instance fields (required for prepare phase)
    repo: str | None = None           # e.g. "django/django"
    base_commit: str | None = None    # e.g. "abc1234def"

    # Optional SWE-bench fields (for downstream harness evaluation)
    test_patch: str | None = None
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    image_name: str | None = None

    @classmethod
    def from_swebench_instance(cls, row: dict[str, Any], workspace_base: Path) -> "EvalTask":
        """Construct from a raw SWE-bench HuggingFace dataset row."""
        fail_to_pass = row.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            import json
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = [fail_to_pass] if fail_to_pass else []

        pass_to_pass = row.get("PASS_TO_PASS", [])
        if isinstance(pass_to_pass, str):
            import json
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = [pass_to_pass] if pass_to_pass else []

        return cls(
            instance_id=row["instance_id"],
            problem_statement=row["problem_statement"],
            workspace_dir=workspace_base / row["instance_id"],
            repo=row.get("repo"),
            base_commit=row.get("base_commit"),
            test_patch=row.get("test_patch"),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            image_name=row.get("image_name"),
        )

    @property
    def needs_prepare(self) -> bool:
        """Whether this task requires git clone + checkout."""
        return bool(self.repo) and bool(self.base_commit)


@dataclass
class EvalResult:
    """Outcome of evaluating a single task."""

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
    official_resolved: bool | None = None
    evaluation_run_id: str | None = None
    evaluation_report_path: str | None = None
    evaluation_report: dict[str, Any] | None = None

    @property
    def model_patch(self) -> str:
        """Extract the git patch from agent output content.

        Looks for content between git diff markers or the entire
        content if it looks like a patch (starts with 'diff --git').
        Returns empty string if no patch is found.
        """
        if not self.content:
            return ""
        text = self.content.strip()
        # Direct patch
        if text.startswith("diff --git"):
            return text
        # Extract from ```diff ... ``` block
        import re
        match = re.search(r"```diff\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Extract from ``` ... ``` block containing diff
        match = re.search(r"```\s*\n(diff --git.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: try to find any diff header
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("diff --git"):
                return "\n".join(lines[i:]).strip()
        return ""

    @property
    def patch_generated(self) -> bool:
        """Whether the agent completed without error and produced a patch."""
        return self.stop_reason == "completed" and bool(self.model_patch)

    @property
    def resolved(self) -> bool:
        """Whether the official SWE-bench harness marked the instance resolved."""
        return bool(self.official_resolved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "content": self.content,
            "model_patch": self.model_patch,
            "tools_used": self.tools_used,
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "tool_events": self.tool_events,
            "trace_file": str(self.trace_file) if self.trace_file else None,
            "prepare_ms": self.prepare_ms,
            "run_ms": self.run_ms,
            "patch_generated": self.patch_generated,
            "official_resolved": self.official_resolved,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluation_report_path": self.evaluation_report_path,
            "evaluation_report": self.evaluation_report,
            "resolved": self.resolved,
        }

    def to_swebench_prediction(self, model_name: str) -> dict[str, Any]:
        """Format as a SWE-bench harness prediction entry."""
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": model_name,
            "model_patch": self.model_patch,
        }


@dataclass
class EvalTraceStep:
    """A single step record, aligned with agent-sched-bench StepRecord.

    Emitted as one JSONL line per agent iteration checkpoint.
    Each step corresponds to one LLM call + optional tool execution.
    """

    step_idx: int
    type: str = "step"
    agent_id: str = ""
    program_id: str = ""
    phase: str = "acting"          # "acting" | "reasoning"
    instance_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    llm_output: str = ""
    tool_name: str | None = None
    tool_args: str | None = None
    tool_result: str | None = None
    tool_duration_ms: float | None = None
    tool_success: bool | None = None
    tool_timeout: bool | None = None
    tool_ts_start: float | None = None
    tool_ts_end: float | None = None
    ts_start: float = 0.0
    ts_end: float = 0.0
    messages_in: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "instance_id": self.instance_id,
            "step_idx": self.step_idx,
            "phase": self.phase,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_latency_ms": self.llm_latency_ms,
            "llm_output": self.llm_output,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "tool_duration_ms": self.tool_duration_ms,
            "tool_success": self.tool_success,
            "tool_timeout": self.tool_timeout,
            "tool_ts_start": self.tool_ts_start,
            "tool_ts_end": self.tool_ts_end,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "messages_in": self.messages_in,
            "raw_response": self.raw_response,
            "extra": self.extra,
        }


@dataclass
class EvalTraceSummary:
    """Summary record emitted at the end of a task run.

    Aligned with agent-sched-bench summary schema.
    """

    type: str = "summary"
    agent_id: str = ""
    program_id: str = ""
    task_id: str = ""
    instance_id: str = ""
    n_steps: int = 0
    total_llm_ms: float = 0.0
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
            "n_steps": self.n_steps,
            "total_llm_ms": self.total_llm_ms,
            "total_tool_ms": self.total_tool_ms,
            "total_tokens": self.total_tokens,
            "tool_ms_by_name": self.tool_ms_by_name,
            "tool_timeouts": self.tool_timeouts,
            "success": self.success,
            "elapsed_s": self.elapsed_s,
            "prepare_ms": self.prepare_ms,
        }


# ── Event Categories ──────────────────────────────────────────────────

SCHEDULING = "SCHEDULING"
SESSION = "SESSION"
CONTEXT = "CONTEXT"
LLM = "LLM"
TOOL = "TOOL"
MCP = "MCP"
MEMORY = "MEMORY"
SUBAGENT = "SUBAGENT"

ALL_CATEGORIES = frozenset([
    SCHEDULING, SESSION, CONTEXT, LLM, TOOL, MCP, MEMORY, SUBAGENT,
])


@dataclass
class EvalTraceEvent:
    """A fine-grained event from any subsystem in the agent lifecycle.

    Emitted as one JSONL line alongside EvalTraceStep records.
    See docs/eval-events.md for the complete catalog.
    """

    agent_id: str
    program_id: str
    instance_id: str
    event: str            # e.g. "skill_load", "mcp_tool_call"
    category: str         # one of ALL_CATEGORIES
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0       # wall-clock timestamp
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
