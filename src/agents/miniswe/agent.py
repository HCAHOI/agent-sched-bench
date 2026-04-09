"""SWE-bench coding agent backed by mini-swe-agent's DefaultAgent.

Uses mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) as the
underlying LLM loop, preserving the AgentBase interface for trace collection.

The agent runs inside the task's pre-built SWE-bench/SWE-rebench container
(``task["image_name"]``) via mini-swe-agent's ``DockerEnvironment``. The
container has the repo at ``/testbed`` with ``base_commit`` already checked
out and dependencies pre-installed, so no host-side clone or pip install is
required. The container runtime defaults to ``podman`` but can be overridden
via the ``MSWEA_DOCKER_EXECUTABLE`` environment variable.
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
from minisweagent.agents.default import DefaultAgent

if TYPE_CHECKING:
    from trace_collect.attempt_pipeline import AttemptContext, AttemptResult

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContextManagedAgent — sliding window context management
# ---------------------------------------------------------------------------


class ContextManagedAgent(DefaultAgent):
    """DefaultAgent with sliding window context management.

    Always preserves system message [0] + task message [1].
    Trims oldest assistant-tool groups from the middle to fit
    within budget. Maintains a separate _full_messages list
    that is never trimmed, for post-hoc trajectory conversion.
    """

    def __init__(
        self, *args: Any, max_context_tokens: int = 256_000, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_context_tokens = max_context_tokens
        self._full_messages: list[dict] = []

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        # Strip 'extra' metadata (not sent to the model) to avoid
        # ~7x overestimation from response objects and timestamps.
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
        # Keep an untrimmed copy of all messages for trajectory
        # conversion. add_messages is the sole entry point for new
        # messages (called by query() and run()), so this is the
        # single place to maintain the full record.
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
        # Don't leave an orphaned tool result at the start of suffix;
        # trim until we hit an assistant message to keep pairs intact.
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


# ---------------------------------------------------------------------------
# Templates (adapted from mini-swe-agent swebench.yaml)
# ---------------------------------------------------------------------------

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
        max_steps: int = 50,
        command_timeout_s: float = 120.0,
        task_timeout_s: float = 1200.0,
        max_tool_output_chars: int = 8000,
        repos_root: str | None = None,
        max_context_tokens: int = 256_000,
        prompt_template: str = "default",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_tool_output_chars=max_tool_output_chars,
        )
        self.max_steps = max_steps
        self.command_timeout_s = command_timeout_s
        self.task_timeout_s = task_timeout_s
        self.repos_root = Path(repos_root) if repos_root else None
        self.max_context_tokens = max_context_tokens
        self.prompt_template = prompt_template
        self._workdir: Path | None = None
        self._prepared = False

    # ------------------------------------------------------------------
    # Two-phase lifecycle
    # ------------------------------------------------------------------

    async def prepare(self, task: dict[str, Any]) -> None:  # noqa: ARG002
        """Create a host tempdir for the mini-swe-agent trajectory file.

        The repo itself lives inside the task container at ``/testbed``; no
        host-side clone or dependency install is required.
        """
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

    # ------------------------------------------------------------------
    # Inner agent loop
    # ------------------------------------------------------------------

    async def _run_mini_agent(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: "AttemptContext | None" = None,
    ) -> bool:
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        from trace_collect.prompt_loader import load_prompt_template

        image = (
            attempt_ctx.fixed_image
            if attempt_ctx is not None and attempt_ctx.fixed_image
            else task.get("image_name")
        )
        if not image:
            raise RuntimeError(
                f"Task {task.get('instance_id')!r} has no 'image_name'; "
                "the SWE-rebench/SWE-bench plugin must populate it via "
                "normalize_task() before running mini-swe-agent."
            )

        # Load the instance template by name (falls back to module constant
        # if the configs dir is missing — keeps existing tests happy).
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
            model_name=f"openai/{self.model}",
            model_kwargs={
                "api_base": self.api_base,
                "api_key": self.api_key,
                "drop_params": True,
                "temperature": 0.0,
            },
            cost_tracking="ignore_errors",
        )
        # Default to podman; override with MSWEA_DOCKER_EXECUTABLE=docker if needed.
        executable = os.getenv("MSWEA_DOCKER_EXECUTABLE", "podman")

        # run_args mirror the Claude Code harness container launch
        # (agentcgroup/scripts/run_swebench.py::_run_claude_with_monitoring):
        # --userns=keep-id + host /home mount so /testbed is writable by the
        # host uid (the fixed derivative image was chown'd for this), and
        # binaries under ~/.local are accessible inside the container.
        home_dir = os.environ.get("HOME", "/root")
        run_args = [
            "--rm",
            "--userns=keep-id",
            "--network=host",
            "-v", f"{home_dir}:{home_dir}",
        ]
        env = DockerEnvironment(
            image=image,
            cwd="/testbed",
            timeout=int(self.command_timeout_s),
            executable=executable,
            pull_timeout=300,
            run_args=run_args,
            env={
                "HOME": home_dir,
                "PATH": f"{home_dir}/.local/bin:/usr/local/bin:/usr/bin:/bin",
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
            },
        )

        # Force container start so env.container_id is populated, then
        # hand the id to the attempt pipeline so its resource sampler can
        # begin polling the right container.
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
            step_limit=self.max_steps,
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
            # Capture container stdout BEFORE cleanup (cleanup backgrounds
            # podman stop/rm, which may race with a subsequent logs call).
            if attempt_ctx is not None:
                container_id = getattr(env, "container_id", None)
                if container_id:
                    attempt_ctx.claude_output = _capture_container_logs(
                        container_id, executable
                    )
            # Stop and remove the container explicitly; do not rely on __del__.
            env.cleanup()

        wall_end = time.time()

        self.task_exit_status = result.get("exit_status")
        self.task_submission = result.get("submission", "") or ""
        self.task_error = result.get("error")
        success = result.get("exit_status") == "Submitted" and bool(
            result.get("submission", "").strip()
        )
        self.task_success = success
        # Use _full_messages (never trimmed) for trajectory conversion so
        # that all steps are recorded even when context was trimmed.
        # Deep-copy to avoid data races: the background thread may still
        # be running after TimeoutError and could mutate message dicts.
        msg_snapshot = copy.deepcopy(mini_agent._full_messages)
        self._convert_trajectory(msg_snapshot, wall_start, wall_end)
        return success

    # ------------------------------------------------------------------
    # Trajectory → TraceAction conversion
    # ------------------------------------------------------------------

    def _convert_trajectory(
        self,
        messages: list[dict[str, Any]],
        run_ts_start: float,
        run_ts_end: float,
    ) -> None:
        """Convert a snapshot of mini-swe-agent's message list into TraceActions."""

        # Messages layout:
        #   [0] system
        #   [1] user (task)
        #   [2] assistant (step 0 LLM response + tool calls)
        #   [3..k] tool result messages  (one per parallel tool call)
        #   [k+1] assistant (step 1)
        #   ...
        #   [-1] exit

        iteration = 0
        prev_msg_len = 0  # for delta messages_in
        _prev_ts_end = (
            run_ts_start  # fallback ts_start when predecessor lacks timestamp
        )
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
            # Use the previous message's timestamp when available; fall back
            # to the previous step's ts_end rather than run_ts_start to avoid
            # inflating LLM latency for steps whose predecessor lacks a timestamp
            # (e.g. the submission step following a tool result without extra.ts).
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

            # delta messages_in: only messages added since last step
            messages_in = list(messages[prev_msg_len:i])

            # Emit llm_call TraceAction
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

            # --- Tool call info ---
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

            # Submission path: exit message may carry final tool output
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

                # Emit tool_exec_start/end observability events
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

                # Emit tool_exec TraceAction
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
    """Capture ``<executable> logs <container_id>`` output as UTF-8 text.

    Returns an empty string on any failure; the caller is expected to treat
    log capture as best-effort (the container may already be gone by the
    time we get here if cleanup raced).
    """
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
    # podman merges stdout/stderr for logs unless --err is passed; capture both.
    return (result.stdout or "") + (result.stderr or "")
