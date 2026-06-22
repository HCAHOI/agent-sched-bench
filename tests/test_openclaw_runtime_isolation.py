"""Regression tests for OpenClaw runtime workspace isolation.

These pin the design in ``docs/analysis/openclaw_runtime_isolation_plan.md``:
OpenClaw runtime state (sessions, memory, skills, tool-result spill, async
daemon prompts/pids/logs) must never contaminate a git-tracked task workspace.

Suites 1-8 mirror the plan's regression list.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("agents.openclaw._session_runner", reason="requires openclaw")

from agents.openclaw._context import ContextBuilder
from agents.openclaw._memory import MemoryStore
from agents.openclaw._skills import SkillsLoader
from agents.openclaw._session_runner import SessionRunner
from llm_call.provider_base import LLMProvider, LLMResponse
from agents.openclaw.session.manager import SessionManager
from agents.openclaw.utils.helpers import maybe_persist_tool_result


# ── Suite 1: SessionManager isolation ───────────────────────────────


def test_session_manager_does_not_create_workspace_sessions(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    runtime_sessions = tmp_path / "rt" / "sessions"

    manager = SessionManager(workspace, storage_dir=runtime_sessions)
    session = manager.get_or_create("cli:oc-test")
    session.add_message("user", "hello")
    manager.save(session)

    assert runtime_sessions.exists()
    assert any(runtime_sessions.glob("*.jsonl"))
    assert not (workspace / "sessions").exists()


# ── Suite 2: Memory/context read-write consistency ──────────────────


def test_memory_store_and_context_share_runtime_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    memory_dir = tmp_path / "rt" / "memory"

    store = MemoryStore(workspace, storage_dir=memory_dir)
    store.write_long_term("# Important fact\nThe answer is 42.")

    context = ContextBuilder(
        workspace, memory_dir=memory_dir, skills_dir=tmp_path / "rt" / "skills"
    )
    system_prompt = context.build_system_prompt()

    assert "The answer is 42" in system_prompt
    assert (memory_dir / "MEMORY.md").exists()
    assert not (workspace / "memory").exists()


# ── Suite 3: Skills isolation ───────────────────────────────────────


def test_skills_loader_uses_runtime_dir_not_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skills_dir = tmp_path / "rt" / "skills"
    (skills_dir / "demo-skill").mkdir(parents=True)
    (skills_dir / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody", encoding="utf-8"
    )

    loader = SkillsLoader(workspace, skills_dir=skills_dir)
    skills = loader.list_skills(filter_unavailable=False)
    names = [s["name"] for s in skills]
    assert "demo-skill" in names
    # No workspace skills dir is created or read.
    assert not (workspace / "skills").exists()


def test_skills_loader_without_skills_dir_does_not_touch_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loader = SkillsLoader(workspace, skills_dir=None)
    # Building the summary must not create workspace/skills.
    loader.build_skills_summary()
    assert not (workspace / "skills").exists()


def test_subagent_prompt_uses_runtime_skills_dir(tmp_path: Path) -> None:
    from agents.openclaw._subagent import SubagentManager
    from agents.openclaw.bus.queue import MessageBus

    workspace = tmp_path / "ws"
    workspace.mkdir()
    skills_dir = tmp_path / "rt" / "skills"

    provider = SimpleNamespace(get_default_model=lambda: "fake")
    manager = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=MessageBus(),
        max_tool_result_chars=1000,
        skills_dir=skills_dir,
    )
    # The subagent prompt builder must not instantiate workspace-local skills.
    prompt = manager._build_subagent_prompt()
    assert str(workspace / "skills") not in prompt
    assert not (workspace / "skills").exists()


# ── Suite 4: SessionRunner workspace cleanliness (git-status) ───────


class _ImmediateFinalProvider(LLMProvider):
    """Returns a final stop response on the first call; no tool calls."""

    def __init__(self) -> None:
        super().__init__(api_key="fake", api_base="fake")
        self.calls = 0

    def get_default_model(self) -> str:
        return "fake-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del (
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        self.calls += 1
        return LLMResponse(
            content="done",
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True
    )


def test_session_runner_keeps_git_repo_clean(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    runtime_dir = tmp_path / "rt"
    trace_file = tmp_path / "trace.jsonl"

    runner = SessionRunner(
        provider=_ImmediateFinalProvider(),
        model="fake-model",
        max_iterations=1,
        context_window_tokens=4096,
    )
    result = asyncio.run(
        runner.run(
            prompt="do nothing, just reply done",
            workspace=repo,
            session_key="cli:clean-test",
            trace_file=trace_file,
            runtime_dir=runtime_dir,
        )
    )
    assert result.content == "done"

    # Runtime state lives under runtime_dir, not the repo.
    assert (runtime_dir / "sessions").exists()
    assert (runtime_dir / "memory").exists()
    # The repo must contain no OpenClaw runtime artifacts.
    for forbidden in ("sessions", "memory", "skills", ".openclaw", ".nanobot"):
        assert not (repo / forbidden).exists(), f"{forbidden}/ leaked into workspace"
    assert not (repo / "trace.jsonl").exists()
    # git status must be clean (no untracked runtime files).
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.strip() == "", f"workspace dirty: {status.stdout!r}"


# ── Suite 5: Tool-result spill isolation ────────────────────────────


def test_tool_result_spill_goes_to_runtime_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool_results_dir = tmp_path / "rt" / "tool-results"
    big = "x" * 5000

    out = maybe_persist_tool_result(
        tool_results_dir=tool_results_dir,
        session_key="cli:spill",
        tool_call_id="call_1",
        content=big,
        max_chars=1000,
    )

    assert isinstance(out, str)
    assert "tool output persisted" in out
    assert (tool_results_dir / "tool-results").exists()
    assert not (workspace / ".nanobot").exists()
    assert not (workspace / "tool-results").exists()


def test_tool_result_spill_none_dir_returns_raw(tmp_path: Path) -> None:
    big = "x" * 5000
    out = maybe_persist_tool_result(
        tool_results_dir=None,
        session_key="cli:spill",
        tool_call_id="call_1",
        content=big,
        max_chars=1000,
    )
    assert out is big


# ── Suite 6: Terminal-Bench command construction ───────────────────


def test_terminal_bench_command_includes_runtime_dir() -> None:
    from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent

    class StubAgent(TerminalBenchOpenClawAgent):
        @classmethod
        def _build_wheel(cls) -> Path:
            return Path("/tmp/agent_sched_bench-0.1.0-py3-none-any.whl")

    agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    command = agent._run_agent_commands()[0].command

    assert "--workspace ." in command
    assert "--trace-output /agent-logs/openclaw-trace.jsonl" in command
    assert "--runtime-dir /agent-logs/openclaw-runtime" in command


# ── Suite 7: SWE/OpenClaw eval wiring ───────────────────────────────


def test_swe_eval_passes_runtime_dir_outside_tool_workspace(tmp_path: Path) -> None:
    from agents.openclaw.eval.runner import SWEBenchRunner
    from agents.openclaw.eval.types import EvalTask

    captured: dict[str, object] = {}
    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(
        provider=provider,
        workspace_base=tmp_path / "ws",
        benchmark_slug="swe-rebench",
    )

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="completed",
            error=None,
        )

    runner._session_runner.run = fake_run  # type: ignore[method-assign]

    task = EvalTask(
        instance_id="encode__httpx-2701",
        problem_statement="fix bug",
        workspace_dir=tmp_path / "repo",
        repo="encode/httpx",
        base_commit="HEAD",
        image_name="swerebench/example",
    )
    trace_file = tmp_path / "attempts" / "attempt_1" / "trace.jsonl"

    asyncio.run(
        runner.run_task(
            task,
            prompt_template="cc_aligned",
            trace_file=trace_file,
        )
    )

    assert "runtime_dir" in captured
    runtime_dir = Path(captured["runtime_dir"])
    tool_workspace = Path(captured["workspace"])
    # runtime_dir must be strictly outside the task workspace.
    assert runtime_dir != tool_workspace
    assert tool_workspace not in runtime_dir.parents
    # runtime_dir is sibling to the trace file, outside the checkout.
    assert runtime_dir.parent == trace_file.parent


def test_swe_eval_default_trace_not_inside_workspace(tmp_path: Path) -> None:
    """Omitting trace_file must NOT default into the target checkout."""
    from agents.openclaw.eval.runner import SWEBenchRunner
    from agents.openclaw.eval.types import EvalTask

    captured: dict[str, object] = {}
    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(
        provider=provider,
        workspace_base=tmp_path / "ws",
        benchmark_slug="swe-rebench",
    )

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="completed",
            error=None,
        )

    runner._session_runner.run = fake_run  # type: ignore[method-assign]

    task = EvalTask(
        instance_id="encode__httpx-2701",
        problem_statement="fix bug",
        workspace_dir=tmp_path / "repo",
        repo="encode/httpx",
        base_commit="HEAD",
        image_name="swerebench/example",
    )

    asyncio.run(runner.run_task(task, prompt_template="cc_aligned"))

    trace_file = Path(captured["trace_file"])
    ws = task.workspace_dir
    # The default trace must not be inside the workspace.
    assert ws not in trace_file.parents
    assert trace_file.parent != ws


# ── Suite 8: Async CLI state ────────────────────────────────────────


def test_run_async_does_not_create_workspace_openclaw_dir(tmp_path: Path) -> None:
    from unittest.mock import patch

    from agents.openclaw._cli import _run_async, build_parser

    workspace = tmp_path / "ws"
    workspace.mkdir()
    args = build_parser().parse_args(
        [
            "--prompt",
            "do something",
            "--workspace",
            str(workspace),
            "--async",
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--runtime-dir",
            str(tmp_path / "rt"),
        ]
    )

    captured: dict[str, object] = {}

    def _fake_spawn(
        cmd, pid_file, session_id, *, extra_env=None, trace_file=None, runtime_dir=None
    ):
        captured["cmd"] = list(cmd)
        captured["runtime_dir"] = runtime_dir
        captured["pid_file"] = pid_file
        return 12345

    with (
        patch("agents.openclaw._daemon.spawn_daemon", side_effect=_fake_spawn),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-key"}),
    ):
        rc = _run_async(args)

    assert rc == 0
    cmd = captured["cmd"]
    assert "--runtime-dir" in cmd
    # PID + prompt files live under the runtime dir, not the workspace.
    pid_file = Path(captured["pid_file"])
    assert pid_file.parent.parent == Path(captured["runtime_dir"])
    prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
    assert prompt_path.parent.parent == Path(captured["runtime_dir"])
    assert not (workspace / ".openclaw").exists()


# ── Trace metadata records runtime dirs ─────────────────────────────


def test_trace_metadata_records_runtime_dirs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    runtime_dir = tmp_path / "rt"
    trace_file = tmp_path / "trace.jsonl"

    runner = SessionRunner(
        provider=_ImmediateFinalProvider(),
        model="fake-model",
        max_iterations=1,
        context_window_tokens=4096,
    )
    asyncio.run(
        runner.run(
            prompt="done",
            workspace=repo,
            session_key="cli:meta-test",
            trace_file=trace_file,
            runtime_dir=runtime_dir,
        )
    )

    first = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[0])
    assert first["type"] == "trace_metadata"
    assert first["runtime_dir"] == str(runtime_dir)
    assert first["session_dir"] == str(runtime_dir / "sessions")
    assert first["memory_dir"] == str(runtime_dir / "memory")
    assert first["skills_dir"] == str(runtime_dir / "skills")
    assert first["tool_results_dir"] == str(runtime_dir / "tool-results")
