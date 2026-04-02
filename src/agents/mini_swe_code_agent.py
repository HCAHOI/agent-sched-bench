"""SWE-bench coding agent backed by mini-swe-agent's DefaultAgent.

Uses mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) as the
underlying LLM loop, preserving the AgentBase interface for trace collection.

prepare() clones the repo at base_commit into a temp directory.
run()     wraps DefaultAgent.run() via asyncio.to_thread, then converts
          the trajectory into StepRecords for our trace logger.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.base import AgentBase, LLMCallResult

# ---------------------------------------------------------------------------
# Templates (adapted from mini-swe-agent swebench.yaml)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer shell to solve"
    " programming tasks."
)

_INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
Your task is to make changes to non-test files in the current working directory to fix the
issue described in the PR description in a way that is general and consistent with the codebase.

For each response:
1. Include a THOUGHT section explaining your reasoning.
2. Provide one or more bash tool calls to execute.

## Recommended Workflow

1. Analyse the codebase by finding and reading relevant files.
2. Create a script to reproduce the issue.
3. Edit the source code to resolve the issue.
4. Verify your fix works by running your script again.
5. Test edge cases to ensure your fix is robust.

## Constraints

- MODIFY: Regular source code files in the current working directory.
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.).

## Submission

When done, submit your changes as a git patch using SEPARATE commands:

  Step 1 – create patch:   git diff -- path/to/changed_file > patch.txt
  Step 2 – verify patch:   cat patch.txt
  Step 3 – submit (EXACT): echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt

You CANNOT continue working after submitting.
</instructions>"""


# ---------------------------------------------------------------------------
# Regex to parse returncode from mini-swe-agent observation template
# ---------------------------------------------------------------------------
_RETURNCODE_RE = re.compile(r"<returncode>(\d+)</returncode>")


class MiniSWECodeAgent(AgentBase):
    """SWE-bench coding agent backed by mini-swe-agent's DefaultAgent."""

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
        self.repos_root = Path(repos_root) if repos_root else None
        self._workdir: Path | None = None
        self._prepared = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _repo_dir_name(self, task: dict[str, Any]) -> str:
        owner, name = task["repo"].split("/")
        return f"{owner}__{name}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_output_chars:
            return text
        half = self.max_tool_output_chars // 2
        return (
            text[:half]
            + f"\n[... truncated {len(text) - self.max_tool_output_chars} chars ...]\n"
            + text[-half:]
        )

    # ------------------------------------------------------------------
    # Two-phase lifecycle
    # ------------------------------------------------------------------

    async def prepare(self, task: dict[str, Any]) -> None:
        """Clone repo at base_commit into an isolated temp directory."""
        workdir = Path(tempfile.mkdtemp(prefix="miniswe_"))
        repo_dir = workdir / "repo"
        base_commit: str = task["base_commit"]

        if self.repos_root:
            local_repo = self.repos_root / self._repo_dir_name(task)
            clone_cmd = f"git clone {local_repo} {repo_dir}"
        else:
            repo_url = f"https://github.com/{task['repo']}.git"
            clone_cmd = f"git clone {repo_url} {repo_dir}"

        checkout_cmd = f"git -C {repo_dir} checkout {base_commit}"
        install_cmd = (
            f"cd {repo_dir}"
            " && if [ -f setup.py ] || [ -f pyproject.toml ]; then"
            "   pip install -e . 2>&1 | tail -5;"
            " fi"
        )

        for cmd in (clone_cmd, checkout_cmd):
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300.0,
            )
            if result.returncode != 0:
                shutil.rmtree(workdir, ignore_errors=True)
                raise RuntimeError(
                    f"Repo setup failed for {task['instance_id']}: "
                    f"{(result.stdout + result.stderr)[:300]}"
                )

        # pip install – best effort
        await asyncio.to_thread(
            subprocess.run,
            install_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600.0,
        )

        self._workdir = workdir
        self._prepared = True

    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = task["instance_id"]
        self.task_success = False
        self.trace = []

        if not self._prepared:
            await self.prepare(task)

        assert self._workdir is not None
        repo_dir = self._workdir / "repo"

        try:
            return await self._run_mini_agent(task, repo_dir)
        finally:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
            self._prepared = False

    # ------------------------------------------------------------------
    # Inner agent loop
    # ------------------------------------------------------------------

    async def _run_mini_agent(self, task: dict[str, Any], repo_dir: Path) -> bool:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        lm = LitellmModel(
            model_name=f"openai/{self.model}",
            model_kwargs={
                "api_base": self.api_base,
                "api_key": self.api_key,
                "drop_params": True,
                "temperature": 0.0,
            },
            cost_tracking="ignore_errors",
        )
        env = LocalEnvironment(
            cwd=str(repo_dir),
            timeout=int(self.command_timeout_s),
            env={"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"},
        )
        mini_agent = DefaultAgent(
            lm,
            env,
            system_template=_SYSTEM_TEMPLATE,
            instance_template=_INSTANCE_TEMPLATE,
            step_limit=self.max_steps,
            cost_limit=0.0,
            output_path=str(self._workdir / "trajectory.json"),
        )

        wall_start = time.time()

        result: dict[str, Any] = {"exit_status": "error", "submission": ""}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(mini_agent.run, task["problem_statement"]),
                timeout=self.task_timeout_s,
            )
        except TimeoutError:
            result = {"exit_status": "timeout", "submission": ""}
        except Exception as exc:
            result = {"exit_status": "error", "submission": "", "error": str(exc)}

        wall_end = time.time()

        success = (
            result.get("exit_status") == "Submitted"
            and bool(result.get("submission", "").strip())
        )
        self.task_success = success
        # Snapshot messages before conversion: the background thread may still
        # be running after a TimeoutError (threads cannot be cancelled), so
        # deep-copy to avoid data races where the thread mutates existing
        # message dicts (e.g. litellm adding usage to extra).
        msg_snapshot = copy.deepcopy(mini_agent.messages)
        self._convert_trajectory(msg_snapshot, wall_start, wall_end)
        return success

    # ------------------------------------------------------------------
    # Trajectory → StepRecord conversion
    # ------------------------------------------------------------------

    def _convert_trajectory(
        self,
        messages: list[dict[str, Any]],
        run_ts_start: float,
        run_ts_end: float,
    ) -> None:
        """Convert a snapshot of mini-swe-agent's message list into StepRecords."""

        # Messages layout:
        #   [0] system
        #   [1] user (task)
        #   [2] assistant (step 0 LLM response + tool calls)
        #   [3..k] tool result messages  (one per parallel tool call)
        #   [k+1] assistant (step 1)
        #   ...
        #   [-1] exit

        step_idx = 0
        prev_msg_len = 0  # for delta messages_in
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "exit":
                break

            if role != "assistant":
                i += 1
                continue

            # --- LLM call metadata ---
            extra = msg.get("extra", {})
            ts_end = extra.get("timestamp", run_ts_end)
            ts_start = (
                messages[i - 1].get("extra", {}).get("timestamp", run_ts_start)
                if i > 0
                else run_ts_start
            )
            usage = (extra.get("response") or {}).get("usage") or {}
            prompt_tokens: int = usage.get("prompt_tokens", 0) or 0
            completion_tokens: int = usage.get("completion_tokens", 0) or 0
            latency_ms = (ts_end - ts_start) * 1000

            self._emit_event("llm_start", {"step_idx": step_idx, "ts": ts_start})
            self._emit_event("llm_end", {
                "step_idx": step_idx,
                "ts": ts_end,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            })

            llm_result = LLMCallResult(
                content=msg.get("content") or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                llm_latency_ms=latency_ms,
                raw_response=extra.get("response") or {},
                tool_calls=[],
            )

            # delta messages_in: only messages added since last step
            messages_in = list(messages[prev_msg_len:i])

            record = self.build_step_record(
                step_idx=step_idx,
                phase="acting",
                llm_result=llm_result,
                ts_start=ts_start,
                ts_end=ts_end,
                messages_in=messages_in,
            )

            # --- Tool call info ---
            actions = extra.get("actions", [])
            j = i + 1
            tool_outputs: list[str] = []

            while j < len(messages) and messages[j].get("role") not in ("assistant", "exit"):
                content = messages[j].get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            tool_outputs.append(str(block.get("content") or block.get("text", "")))
                elif isinstance(content, str):
                    tool_outputs.append(content)
                j += 1

            # Submission path: mini-swe-agent raises Submitted before
            # format_toolcall_observation_messages is called, so the final
            # bash output (with returncode) lives in the exit message instead.
            _exit_ts: float | None = None
            if not tool_outputs and j < len(messages) and messages[j].get("role") == "exit":
                exit_msg = messages[j]
                exit_content = exit_msg.get("content", "")
                if exit_content:
                    tool_outputs.append(exit_content)
                    _exit_ts = exit_msg.get("extra", {}).get("timestamp")

            # Preserve per-call attribution when multiple tool calls are present.
            if len(tool_outputs) <= 1:
                tool_output = tool_outputs[0] if tool_outputs else ""
            else:
                tool_output = "\n".join(
                    f"[call {k}]\n{out}" for k, out in enumerate(tool_outputs)
                )

            if actions:
                # Record all commands; multiple parallel calls are serialized
                # as a JSON array so no command is silently dropped.
                commands = [a.get("command", "") for a in actions if isinstance(a, dict)]
                tool_args = (
                    json.dumps({"command": commands[0]}) if len(commands) == 1
                    else json.dumps({"commands": commands})
                )
                m = _RETURNCODE_RE.search(tool_output)
                # Default to -1 (failure) when regex doesn't match to avoid
                # silently marking failed tool calls as successful.
                returncode = int(m.group(1)) if m else -1
                tool_ts_start = ts_end
                tool_ts_end = (
                    _exit_ts
                    if _exit_ts is not None
                    else (messages[j - 1].get("extra", {}).get("timestamp") if j > i + 1 else None)
                )
                tool_duration_ms = (
                    (tool_ts_end - tool_ts_start) * 1000 if tool_ts_end is not None else None
                )
                record.tool_name = "bash"
                record.tool_args = tool_args
                record.tool_result = self._truncate(tool_output)
                record.tool_success = returncode == 0
                record.tool_timeout = False
                record.tool_ts_start = tool_ts_start
                record.tool_ts_end = tool_ts_end
                record.tool_duration_ms = tool_duration_ms
                self._emit_event("tool_start", {
                    "step_idx": step_idx,
                    "ts": tool_ts_start,
                    "tool_name": "bash",
                    "tool_args": record.tool_args,
                })
                self._emit_event("tool_end", {
                    "step_idx": step_idx,
                    "ts": tool_ts_end or tool_ts_start,
                    "tool_name": "bash",
                    "duration_ms": tool_duration_ms,
                    "success": record.tool_success,
                    "timeout": False,
                })

            self._emit_step(record)
            step_idx += 1
            prev_msg_len = j
            i = j
