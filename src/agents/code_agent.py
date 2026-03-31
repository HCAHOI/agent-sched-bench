from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from agents.base import AgentBase, LLMCallResult, StepRecord
from agents.tool_calling import ToolCall, parse_tool_call, strip_code_fences


SYSTEM_PROMPT = """You are a software engineer. You will be given a GitHub issue
and a repository snapshot. Fix the bug by modifying the code.

Available tools:
- bash(command): Execute a bash command in the repository
- submit(patch): Submit your fix as a unified diff

Always think step by step. Use grep/find to locate relevant files first,
then read the code, then make targeted edits, then run tests."""


class CodeAgent(AgentBase):
    """SWE-bench-style coding agent with a lightweight temp-dir sandbox."""

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        max_steps: int = 40,
        command_timeout_s: float = 30.0,
        task_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(agent_id=agent_id, api_base=api_base, model=model)
        self.max_steps = max_steps
        self.command_timeout_s = command_timeout_s
        self.task_timeout_s = task_timeout_s
        self._workspace_path: Path | None = None

    def _format_issue(self, task: dict[str, Any]) -> str:
        return textwrap.dedent(
            f"""
            Instance ID: {task['instance_id']}
            Repository: {task['repo_path']}
            Test command: {task['test_cmd']}

            Problem statement:
            {task['problem_statement']}
            """
        ).strip()

    def _prepare_workspace(self, task: dict[str, Any]) -> Path:
        source_repo = Path(task["repo_path"]).resolve()
        if not source_repo.exists():
            raise FileNotFoundError(f"Repository path does not exist: {source_repo}")
        temp_root = Path(tempfile.mkdtemp(prefix=f"{task['instance_id']}-"))
        workspace = temp_root / source_repo.name
        shutil.copytree(source_repo, workspace)
        self._workspace_path = workspace
        return workspace

    def _cleanup_workspace(self) -> None:
        if self._workspace_path is None:
            return
        temp_root = self._workspace_path.parent
        shutil.rmtree(temp_root, ignore_errors=True)
        self._workspace_path = None

    def _parse_tool_call(self, response: str) -> ToolCall | None:
        return parse_tool_call(response, {"bash", "submit"})

    async def _run_subprocess(self, command: str, cwd: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
        return await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    async def _execute_tool(self, tool_call: ToolCall, task: dict[str, Any]) -> str:
        if self._workspace_path is None:
            raise RuntimeError("Workspace is not prepared")
        if tool_call.name != "bash":
            raise ValueError(f"Unsupported tool: {tool_call.name}")
        try:
            completed = await self._run_subprocess(
                tool_call.args,
                cwd=self._workspace_path,
                timeout_s=self.command_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out"

        output = ""
        if completed.stdout:
            output += completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        if completed.returncode != 0:
            return f"ERROR: command exited with {completed.returncode}\n{output}".strip()
        return output.strip() or "(no output)"

    async def _apply_and_test(self, patch_text: str, task: dict[str, Any]) -> tuple[bool, str]:
        if self._workspace_path is None:
            raise RuntimeError("Workspace is not prepared")
        cleaned_patch = strip_code_fences(patch_text)
        patch_path = self._workspace_path / ".agent_patch.diff"
        patch_path.write_text(cleaned_patch + "\n", encoding="utf-8")

        try:
            apply_result = await self._run_subprocess(
                f"git apply --whitespace=nowarn {patch_path.name}",
                cwd=self._workspace_path,
                timeout_s=self.command_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "ERROR: patch apply timed out"
        if apply_result.returncode != 0:
            return False, (apply_result.stderr or apply_result.stdout or "git apply failed").strip()

        try:
            test_result = await self._run_subprocess(
                task["test_cmd"],
                cwd=self._workspace_path,
                timeout_s=self.command_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "ERROR: test command timed out"

        output = ""
        if test_result.stdout:
            output += test_result.stdout
        if test_result.stderr:
            output += ("\n" if output else "") + test_result.stderr
        return test_result.returncode == 0, output.strip() or "(no output)"

    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = task["instance_id"]
        self.task_success = False
        self.trace = []
        workspace = self._prepare_workspace(task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_issue(task)},
        ]
        started = time.monotonic()
        try:
            for step_idx in range(self.max_steps):
                if (time.monotonic() - started) > self.task_timeout_s:
                    break
                ts_start = time.time()
                llm_result = await self._call_llm(messages)
                ts_end = time.time()
                record = self.build_step_record(
                    step_idx=step_idx,
                    phase="reasoning",
                    llm_result=llm_result,
                    ts_start=ts_start,
                    ts_end=ts_end,
                )
                tool_call = self._parse_tool_call(llm_result.content)
                if tool_call is None:
                    self.trace.append(record)
                    break

                record.phase = "acting"
                record.tool_name = tool_call.name
                record.tool_args = tool_call.args
                tool_started = time.monotonic()
                if tool_call.name == "submit":
                    success, tool_output = await self._apply_and_test(tool_call.args, task)
                    record.tool_duration_ms = (time.monotonic() - tool_started) * 1000
                    record.tool_result = tool_output
                    record.tool_success = success
                    self.task_success = success
                    self.trace.append(record)
                    break

                tool_output = await self._execute_tool(tool_call, task)
                record.tool_duration_ms = (time.monotonic() - tool_started) * 1000
                record.tool_result = tool_output
                record.tool_success = not tool_output.startswith("ERROR")
                self.trace.append(record)
                messages.append({"role": "assistant", "content": llm_result.content})
                messages.append({"role": "user", "content": f"Tool output:\n{tool_output}"})
            return bool(self.task_success)
        finally:
            self._cleanup_workspace()
