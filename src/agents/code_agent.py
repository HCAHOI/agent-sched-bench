from __future__ import annotations

import json
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from agents.base import AgentBase
from agents.local_sandbox import LocalSandbox
from agents.tool_calling import strip_code_fences


SYSTEM_PROMPT = """You are a software engineer. You will be given a GitHub issue
and a repository snapshot. Fix the bug by modifying the code.

Always think step by step. Use grep/find to locate relevant files first,
then read the code, then make targeted edits, then run tests.

When you are done, call the submit tool with your fix as a unified diff."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the repository working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your fix as a unified diff patch. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The unified diff patch to apply.",
                    }
                },
                "required": ["patch"],
            },
        },
    },
]


class CodeAgent(AgentBase):
    """SWE-bench coding agent with local proot sandbox.

    Each task runs in an isolated temp directory with the target repo
    cloned at the correct base_commit.  The agent can install dependencies
    (``pip install``) and run real tests inside the sandbox.
    """

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        max_steps: int = 40,
        command_timeout_s: float = 120.0,
        task_timeout_s: float = 1200.0,
        max_tool_output_chars: int = 8000,
        repos_root: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, api_base=api_base, model=model, api_key=api_key)
        self.max_steps = max_steps
        self.command_timeout_s = command_timeout_s
        self.task_timeout_s = task_timeout_s
        self.max_tool_output_chars = max_tool_output_chars
        self._container_mgr = LocalSandbox(
            repos_root=Path(repos_root) if repos_root else None,
        )
        self._container_id: str | None = None
        self._prepared = False

    def _format_issue(self, task: dict[str, Any]) -> str:
        repo = task.get("repo", "unknown")
        return textwrap.dedent(
            f"""
            Instance ID: {task['instance_id']}
            Repository: {repo}
            Test command: {task['test_cmd']}

            Problem statement:
            {task['problem_statement']}
            """
        ).strip()

    async def prepare(self, task: dict[str, Any]) -> None:
        """Create the sandbox before the agent loop starts.

        When called before ``run()``, the expensive setup (sandbox
        creation, repo clone, pip install) happens in a separate phase
        so that all agents can start their LLM loops simultaneously.
        """
        self._container_id = await self._container_mgr.create_container(task)
        self._prepared = True

    async def _execute_bash(self, command: str) -> str:
        """Execute a bash command inside the container."""
        assert self._container_id is not None
        returncode, output = await self._container_mgr.exec_in_container(
            self._container_id,
            f"cd /workspace/repo && {command}",
            timeout_s=self.command_timeout_s,
        )
        if returncode != 0:
            return f"ERROR: command exited with {returncode}\n{output}".strip()
        return output.strip() or "(no output)"

    async def _apply_and_test(
        self, patch_text: str, task: dict[str, Any],
    ) -> tuple[bool, str]:
        """Apply patch and run tests inside the container."""
        assert self._container_id is not None
        cleaned_patch = strip_code_fences(patch_text)

        # Write patch to a temp file and copy into container
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", delete=False,
        ) as f:
            f.write(cleaned_patch + "\n")
            patch_path = f.name

        try:
            await self._container_mgr.copy_to_container(
                self._container_id,
                patch_path,
                "/workspace/repo/.agent_patch.diff",
            )
        finally:
            Path(patch_path).unlink(missing_ok=True)

        # Apply patch
        returncode, output = await self._container_mgr.exec_in_container(
            self._container_id,
            "cd /workspace/repo && git apply --whitespace=nowarn .agent_patch.diff",
            timeout_s=self.command_timeout_s,
        )
        if returncode != 0:
            return False, (output or "git apply failed").strip()

        # Run tests
        returncode, output = await self._container_mgr.exec_in_container(
            self._container_id,
            f"cd /workspace/repo && {task['test_cmd']}",
            timeout_s=self.command_timeout_s,
        )
        return returncode == 0, output.strip() or "(no output)"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_output_chars:
            return text
        half = self.max_tool_output_chars // 2
        return (
            text[:half]
            + f"\n[... truncated {len(text) - self.max_tool_output_chars} chars ...]\n"
            + text[-half:]
        )

    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = task["instance_id"]
        self.task_success = False
        self.trace = []

        # Setup: skip if prepare() was already called (two-phase mode)
        if not self._prepared:
            await self.prepare(task)

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
                llm_result = await self._call_llm(messages, tools=TOOLS)
                ts_end = time.time()
                record = self.build_step_record(
                    step_idx=step_idx,
                    phase="reasoning",
                    llm_result=llm_result,
                    ts_start=ts_start,
                    ts_end=ts_end,
                )

                if not llm_result.tool_calls:
                    self.trace.append(record)
                    break

                tc = llm_result.tool_calls[0]
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    args = (
                        {"command": tc.arguments}
                        if tc.name == "bash"
                        else {"patch": tc.arguments}
                    )

                record.phase = "acting"
                record.tool_name = tc.name
                record.tool_args = json.dumps(args)
                tool_started = time.monotonic()

                if tc.name == "submit":
                    patch_text = args.get("patch", tc.arguments)
                    success, tool_output = await self._apply_and_test(
                        patch_text, task,
                    )
                    raw_ms = (time.monotonic() - tool_started) * 1000
                    record.tool_duration_ms = raw_ms
                    record.tool_result = tool_output
                    record.tool_success = success
                    self.task_success = success
                    self.trace.append(record)
                    break

                command = args.get("command", tc.arguments)
                tool_output = await self._execute_bash(command)
                raw_ms = (time.monotonic() - tool_started) * 1000
                record.tool_duration_ms = raw_ms
                record.tool_result = tool_output
                record.tool_success = not tool_output.startswith("ERROR")
                self.trace.append(record)

                # Build assistant message with tool_calls for multi-turn
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_result.content or "",
                }
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                ]
                messages.append(assistant_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": self._truncate(tool_output),
                })
            return bool(self.task_success)
        finally:
            await self._container_mgr.destroy_container(self._container_id)
            self._container_id = None
            self._prepared = False
