"""SWE-bench coding agent backed by mini-swe-agent's DefaultAgent.

Uses mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) as the
underlying LLM loop, preserving the AgentBase interface for trace collection.

The agent runs inside the task's pre-built SWE-bench/SWE-rebench container
(``task["image_name"]``) via mini-swe-agent's ``DockerEnvironment``. The
container has the repo at ``/testbed`` with ``base_commit`` already checked
out and dependencies pre-installed, so no host-side clone or pip install is
required. The container runtime executable must be provided explicitly by the
caller when the agent uses ``runtime_mode="docker_container"``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.base import AgentBase, TraceAction
from harness.container_runtime import container_run_user_args
from llm_call import build_miniswe_litellm_model_name
from minisweagent.agents.default import DefaultAgent

if TYPE_CHECKING:
    from trace_collect.attempt_pipeline import AttemptContext

_logger = logging.getLogger(__name__)

class ContextManagedAgent(DefaultAgent):
    """DefaultAgent variant that trims old context but keeps a full transcript."""

    def __init__(
        self, *args: Any, max_context_tokens: int = 256_000, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_context_tokens = max_context_tokens
        self._full_messages: list[dict] = []

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        return (
            sum(
                len(
                    json.dumps(
                        {k: v for k, v in m.items() if k != "extra"},
                        ensure_ascii=False,
                    )
                )
                for m in messages
            )
            // 4
        )

    def query(self) -> dict:
        self._trim_context()
        return super().query()

    def add_messages(self, *messages: dict) -> None:
        super().add_messages(*messages)
        self._full_messages.extend(messages)

    def _trim_context(self) -> None:
        if len(self.messages) <= 2:
            return
        if self._estimate_tokens(self.messages) <= self._max_context_tokens:
            return
        prefix = self.messages[:2]
        suffix = self.messages[2:]
        while (
            suffix and self._estimate_tokens(prefix + suffix) > self._max_context_tokens
        ):
            suffix.pop(0)
        while suffix and suffix[0].get("role") != "assistant":
            suffix.pop(0)
        self.messages = prefix + suffix
        if self._estimate_tokens(self.messages) > self._max_context_tokens:
            import logging

            logging.getLogger("agent").warning(
                "Context (%d est. tokens) exceeds budget (%d) after "
                "trimming; prefix alone is too large",
                self._estimate_tokens(self.messages),
                self._max_context_tokens,
            )

_SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer shell to solve"
    " programming tasks."
)

def _load_default_instance_template() -> str:
    """Load ``configs/prompts/swe_rebench/default.md`` at import time.

    Falls back to an inline minimal template if the config file is missing,
    so early-boot test environments that have not yet created the configs
    directory still work.
    """
    try:
        from trace_collect.prompt_loader import load_prompt_template

        return load_prompt_template("default")
    except (FileNotFoundError, ValueError, ImportError):
        return (
            "<pr_description>\n"
            "Consider the following PR description:\n"
            "{{task}}\n"
            "</pr_description>\n"
            "\nSubmit your patch via: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        )

_INSTANCE_TEMPLATE = _load_default_instance_template()

_RETURNCODE_RE = re.compile(r"<returncode>(\d+)</returncode>")

class MiniSWECodeAgent(AgentBase):

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        provider_name: str | None = None,
        api_key: str = "EMPTY",
        max_iterations: int = 50,
        command_timeout_s: float = 120.0,
        task_timeout_s: float = 1200.0,
        max_tool_output_chars: int = 8000,
        repos_root: str | None = None,
        max_context_tokens: int = 256_000,
        prompt_template: str = "default",
        runtime_mode: str = "docker_container",
        container_executable: str | None = None,
        exec_working_dir: str = "/testbed",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_tool_output_chars=max_tool_output_chars,
        )
        self.max_iterations = max_iterations
        self.command_timeout_s = command_timeout_s
        self.task_timeout_s = task_timeout_s
        self.repos_root = Path(repos_root) if repos_root else None
        self.max_context_tokens = max_context_tokens
        self.prompt_template = prompt_template
        self.runtime_mode = runtime_mode
        self.container_executable = container_executable
        self.exec_working_dir = exec_working_dir
        self.provider_name = provider_name
        self._workdir: Path | None = None
        self._prepared = False

    async def prepare(self, task: dict[str, Any]) -> None:  # noqa: ARG002
        """Create the host tempdir used for the trajectory artifact."""
        self._workdir = Path(tempfile.mkdtemp(prefix="miniswe_"))
        self._prepared = True

    async def run(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: "AttemptContext | None" = None,
    ) -> bool:
        self.task_id = task["instance_id"]
        self.task_success = False
        self.task_submission = ""
        self.task_exit_status = None
        self.task_error = None
        self.trace = []

        if not self._prepared:
            await self.prepare(task)

        assert self._workdir is not None

        try:
            return await self._run_mini_agent(task, attempt_ctx=attempt_ctx)
        finally:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
            self._prepared = False

    async def _run_mini_agent(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: "AttemptContext | None" = None,
    ) -> bool:
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        from trace_collect.prompt_loader import load_prompt_template

        image = (
            attempt_ctx.fixed_image
            if attempt_ctx is not None and attempt_ctx.fixed_image
            else task.get("image_name")
        )
        if self.runtime_mode == "docker_container" and not image:
            raise RuntimeError(
                f"Task {task.get('instance_id')!r} has no 'image_name'; "
                "the SWE-rebench/SWE-bench plugin must populate it via "
                "normalize_task() before running mini-swe-agent."
            )

        template_name = (
            attempt_ctx.prompt_template
            if attempt_ctx is not None
            else self.prompt_template
        )
        try:
            instance_template = load_prompt_template(template_name)
        except (FileNotFoundError, ValueError) as exc:
            _logger.warning(
                "prompt template %r load failed (%s); falling back to default",
                template_name,
                exc,
            )
            instance_template = _INSTANCE_TEMPLATE

        lm = LitellmModel(
            model_name=build_miniswe_litellm_model_name(
                model=self.model,
                provider_name=self.provider_name,
                api_base=self.api_base,
            ),
            model_kwargs={
                "api_base": self.api_base,
                "api_key": self.api_key,
                "drop_params": True,
                "temperature": 0.0,
            },
            cost_tracking="ignore_errors",
        )
        home_dir = os.environ.get("HOME", "/root")
        shared_env = {
            "HOME": home_dir,
            "PATH": f"{home_dir}/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
        }

        if self.runtime_mode == "local_environment":
            env = LocalEnvironment(
                cwd=self.exec_working_dir,
                timeout=int(self.command_timeout_s),
                env=shared_env,
            )
        else:
            if not self.container_executable:
                raise RuntimeError(
                    "MiniSWE docker_container mode requires an explicit "
                    "container executable from the caller"
                )
            run_args = [
                "--rm",
                "--network=host",
                "-v", f"{home_dir}:{home_dir}",
            ]
            run_args.extend(container_run_user_args(self.container_executable))
            env = DockerEnvironment(
                image=image,
                cwd=self.exec_working_dir,
                timeout=int(self.command_timeout_s),
                executable=self.container_executable,
                pull_timeout=300,
                run_args=run_args,
                env=shared_env,
            )

            if attempt_ctx is not None:
                try:
                    env.execute({"command": "true"})
                except Exception as exc:
                    _logger.warning(
                        "env warmup execute failed: %s; proceeding anyway", exc
                    )
                container_id = getattr(env, "container_id", None)
                if container_id:
                    attempt_ctx.mark_container_ready(container_id)

        mini_agent = ContextManagedAgent(
            lm,
            env,
            system_template=_SYSTEM_TEMPLATE,
            instance_template=instance_template,
            step_limit=self.max_iterations,
            cost_limit=0.0,
            output_path=str(self._workdir / "trajectory.json"),
            max_context_tokens=self.max_context_tokens,
        )

        wall_start = time.time()

        result: dict[str, Any] = {"exit_status": "error", "submission": ""}
        try:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(mini_agent.run, task["problem_statement"]),
                    timeout=self.task_timeout_s,
                )
            except TimeoutError:
                result = {"exit_status": "timeout", "submission": ""}
            except Exception as exc:
                result = {"exit_status": "error", "submission": "", "error": str(exc)}
        finally:
            if attempt_ctx is not None and self.runtime_mode == "docker_container":
                container_id = getattr(env, "container_id", None)
                if container_id:
                    attempt_ctx.container_stdout = _capture_container_logs(
                        container_id,
                        self.container_executable,
                    )
            if hasattr(env, "cleanup"):
                env.cleanup()

        wall_end = time.time()

        self.task_exit_status = result.get("exit_status")
        self.task_submission = result.get("submission", "") or ""
        self.task_error = result.get("error")
        success = result.get("exit_status") == "Submitted" and bool(
            result.get("submission", "").strip()
        )
        self.task_success = success
        msg_snapshot = copy.deepcopy(mini_agent._full_messages)
        self._convert_trajectory(msg_snapshot, wall_start, wall_end)
        return success

    def _convert_trajectory(
        self,
        messages: list[dict[str, Any]],
        run_ts_start: float,
        run_ts_end: float,
    ) -> None:

        iteration = 0
        prev_msg_len = 0
        _prev_ts_end = run_ts_start
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "exit":
                break

            if role != "assistant":
                i += 1
                continue

            extra = msg.get("extra", {})
            ts_end = extra.get("timestamp", run_ts_end)
            ts_start = (
                messages[i - 1].get("extra", {}).get("timestamp") or _prev_ts_end
                if i > 0
                else _prev_ts_end
            )
            usage = (extra.get("response") or {}).get("usage") or {}
            prompt_tokens: int = usage.get("prompt_tokens", 0) or 0
            completion_tokens: int = usage.get("completion_tokens", 0) or 0
            latency_ms = (ts_end - ts_start) * 1000

            self._emit_event(
                "LLM", "llm_call_start", {},
                iteration=iteration, ts=ts_start,
            )
            self._emit_event(
                "LLM", "llm_call_end",
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_latency_ms": latency_ms,
                },
                iteration=iteration, ts=ts_end,
            )

            raw_response = extra.get("response") or {}

            messages_in = list(messages[prev_msg_len:i])

            llm_action = TraceAction(
                action_type="llm_call",
                action_id=f"llm_{iteration}",
                agent_id=self.agent_id,
                program_id=self.agent_id,
                iteration=iteration,
                ts_start=ts_start,
                ts_end=ts_end,
                data={
                    "messages_in": messages_in,
                    "raw_response": raw_response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_latency_ms": latency_ms,
                    "llm_content": (msg.get("content") or "")[:4000],
                },
            )
            self._emit_action(llm_action)

            actions = extra.get("actions", [])
            j = i + 1
            tool_outputs: list[str] = []

            while j < len(messages) and messages[j].get("role") not in (
                "assistant",
                "exit",
            ):
                content = messages[j].get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            tool_outputs.append(
                                str(block.get("content") or block.get("text", ""))
                            )
                elif isinstance(content, str):
                    tool_outputs.append(content)
                j += 1

            _exit_ts: float | None = None
            _from_exit_msg = False
            if (
                not tool_outputs
                and j < len(messages)
                and messages[j].get("role") == "exit"
            ):
                exit_msg = messages[j]
                exit_content = exit_msg.get("content", "")
                if exit_content:
                    tool_outputs.append(exit_content)
                    _exit_ts = exit_msg.get("extra", {}).get("timestamp")
                    _from_exit_msg = True

            if len(tool_outputs) <= 1:
                tool_output = tool_outputs[0] if tool_outputs else ""
            else:
                tool_output = "\n".join(
                    f"[call {k}]\n{out}" for k, out in enumerate(tool_outputs)
                )

            if actions:
                commands = [
                    a.get("command", "") for a in actions if isinstance(a, dict)
                ]
                tool_args = (
                    json.dumps({"command": commands[0]})
                    if len(commands) == 1
                    else json.dumps({"commands": commands})
                )
                m = _RETURNCODE_RE.search(tool_output)
                returncode = int(m.group(1)) if m else (0 if _from_exit_msg else -1)
                tool_ts_start = ts_end
                tool_ts_end = (
                    _exit_ts
                    if _exit_ts is not None
                    else (
                        messages[j - 1].get("extra", {}).get("timestamp")
                        if j > i + 1
                        else None
                    )
                )
                tool_duration_ms = (
                    (tool_ts_end - tool_ts_start) * 1000
                    if tool_ts_end is not None
                    else None
                )

                self._emit_event(
                    "TOOL", "tool_exec_start",
                    {"tool_name": "bash", "tool_args": tool_args},
                    iteration=iteration, ts=tool_ts_start,
                )
                self._emit_event(
                    "TOOL", "tool_exec_end",
                    {
                        "tool_name": "bash",
                        "duration_ms": tool_duration_ms,
                        "success": returncode == 0,
                        "timeout": False,
                    },
                    iteration=iteration, ts=tool_ts_end or tool_ts_start,
                )

                tool_action = TraceAction(
                    action_type="tool_exec",
                    action_id=f"tool_{iteration}_bash",
                    agent_id=self.agent_id,
                    program_id=self.agent_id,
                    iteration=iteration,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end or tool_ts_start,
                    data={
                        "tool_name": "bash",
                        "tool_args": tool_args,
                        "tool_result": self._truncate(tool_output),
                        "duration_ms": tool_duration_ms,
                        "success": returncode == 0,
                        "timeout": False,
                    },
                )
                self._emit_action(tool_action)

            iteration += 1
            prev_msg_len = j
            _prev_ts_end = ts_end
            i = j

def _capture_container_logs(container_id: str, executable: str) -> str:
    """Capture ``<executable> logs <container_id>`` output as UTF-8 text."""
    try:
        result = subprocess.run(
            [executable, "logs", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    return (result.stdout or "") + (result.stderr or "")
