"""Replay a trace from a specific step with new LLM parameters.

Loads a completed (or failed) trace, replays bash commands to restore
filesystem state, reconstructs the agent message list, then resumes
the LLM loop from the specified step.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.mini_swe_code_agent import (
    ContextManagedAgent,
    MiniSWECodeAgent,
    _INSTANCE_TEMPLATE,
    _SYSTEM_TEMPLATE,
)
from harness.trace_logger import TraceLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TraceLoadError(Exception):
    """Raised when a trace file is malformed or the agent_id is not found."""


class ReplayError(Exception):
    """Raised when bash replay detects environment divergence."""


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_trace_steps(
    trace_path: Path,
    agent_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load step records and optional summary for one agent from a JSONL trace.

    Returns:
        (steps sorted by step_idx, summary or None)

    Raises:
        TraceLoadError: if agent_id not found or no step records exist.
    """
    steps: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("agent_id") != agent_id:
                continue
            if record.get("type") == "step":
                steps.append(record)
            elif record.get("type") == "summary":
                summary = record

    if not steps:
        raise TraceLoadError(f"No step records found for agent_id={agent_id!r}")

    steps.sort(key=lambda s: s["step_idx"])

    # Validate no gaps
    indices = [s["step_idx"] for s in steps]
    expected = list(range(len(indices)))
    if indices != expected:
        raise TraceLoadError(
            f"Step index gaps for {agent_id}: got {indices[:5]}... expected 0..{len(indices)-1}"
        )

    return steps, summary


# ---------------------------------------------------------------------------
# Message reconstruction
# ---------------------------------------------------------------------------


def _extract_actions(step: dict[str, Any]) -> list[dict[str, str]]:
    """Extract action dicts from a step record's tool_args + raw_response."""
    tool_args = json.loads(step.get("tool_args") or "{}")
    raw_msg = (step.get("raw_response") or {}).get("choices", [{}])[0].get("message", {})
    tool_calls = raw_msg.get("tool_calls") or []

    if "command" in tool_args:
        commands = [tool_args["command"]]
    elif "commands" in tool_args:
        commands = tool_args["commands"]
    else:
        return []

    actions = []
    for i, cmd in enumerate(commands):
        tc_id = tool_calls[i]["id"] if i < len(tool_calls) else f"call_{i}"
        actions.append({"command": cmd, "tool_call_id": tc_id})
    return actions


def _build_tool_messages(
    step: dict[str, Any],
    tool_content: str,
) -> list[dict[str, Any]]:
    """Reconstruct tool result messages for a step."""
    raw_msg = (step.get("raw_response") or {}).get("choices", [{}])[0].get("message", {})
    tool_calls = raw_msg.get("tool_calls") or []
    ts = step.get("tool_ts_end") or step.get("ts_end")

    # Single tool call (common case)
    if len(tool_calls) <= 1:
        tc_id = tool_calls[0]["id"] if tool_calls else "call_0"
        return [{
            "role": "tool",
            "tool_call_id": tc_id,
            "content": tool_content,
            "extra": {"timestamp": ts, "raw_output": "", "returncode": 0 if step.get("tool_success") else 1},
        }]

    # Multi-call: split on [call N] markers
    parts = tool_content.split("[call ")
    outputs: list[str] = []
    for part in parts:
        if part and part[0].isdigit() and "]\n" in part:
            outputs.append(part.split("]\n", 1)[1])

    # Fallback: if splitting failed, give full content to first call
    if len(outputs) != len(tool_calls):
        outputs = [tool_content] + [""] * (len(tool_calls) - 1)

    msgs = []
    for i, tc in enumerate(tool_calls):
        msgs.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": outputs[i] if i < len(outputs) else "",
            "extra": {"timestamp": ts, "raw_output": "", "returncode": 0 if step.get("tool_success") else 1},
        })
    return msgs


def reconstruct_messages(
    steps: list[dict[str, Any]],
    up_to_step: int,
    bash_outputs: dict[int, str],
) -> list[dict[str, Any]]:
    """Rebuild the full message list as it existed after step `up_to_step - 1`.

    Args:
        steps: All step records for the agent, sorted by step_idx.
        up_to_step: Reconstruct messages up to (but not including) this step.
        bash_outputs: Mapping of step_idx -> full bash output from replay.
            Used instead of step["tool_result"] (which may be truncated).
    """
    messages: list[dict[str, Any]] = []

    # Seed with system + user from step 0's messages_in
    messages_in = steps[0].get("messages_in") or []
    if len(messages_in) < 2:
        raise TraceLoadError("Step 0 messages_in must contain system + user messages")
    messages.append(messages_in[0])  # system
    messages.append(messages_in[1])  # user

    for step in steps[:up_to_step]:
        raw_response = step.get("raw_response") or {}
        choices = raw_response.get("choices", [{}])
        raw_msg = choices[0].get("message", {}) if choices else {}

        # Reconstruct assistant message
        assistant_msg: dict[str, Any] = {
            "content": raw_msg.get("content") or "",
            "role": "assistant",
        }
        if raw_msg.get("tool_calls"):
            assistant_msg["tool_calls"] = raw_msg["tool_calls"]
        if raw_msg.get("function_call"):
            assistant_msg["function_call"] = raw_msg["function_call"]
        if raw_msg.get("provider_specific_fields"):
            assistant_msg["provider_specific_fields"] = raw_msg["provider_specific_fields"]

        assistant_msg["extra"] = {
            "actions": _extract_actions(step),
            "response": raw_response,
            "cost": 0.0,
            "timestamp": step.get("ts_end", 0.0),
        }
        messages.append(assistant_msg)

        # Reconstruct tool result message(s)
        if step.get("tool_name"):
            # Prefer full bash output over truncated tool_result
            tool_content = bash_outputs.get(step["step_idx"], step.get("tool_result") or "")
            tool_msgs = _build_tool_messages(step, tool_content)
            messages.extend(tool_msgs)

    return messages


# ---------------------------------------------------------------------------
# Bash replay
# ---------------------------------------------------------------------------

_OBSERVATION_TEMPLATE = "<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"


async def replay_bash_commands(
    steps: list[dict[str, Any]],
    up_to_step: int,
    repo_dir: Path,
    *,
    command_timeout_s: float = 120.0,
) -> dict[int, str]:
    """Execute bash commands from steps 0..up_to_step-1, return full outputs.

    Returns:
        Mapping of step_idx -> formatted observation string (with returncode).
    """
    outputs: dict[int, str] = {}
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}

    for step in steps[:up_to_step]:
        if not step.get("tool_name"):
            continue

        tool_args = json.loads(step.get("tool_args") or "{}")
        if "command" in tool_args:
            commands = [tool_args["command"]]
        elif "commands" in tool_args:
            commands = tool_args["commands"]
        else:
            continue

        all_output = []
        last_returncode = 0
        for cmd in commands:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    shell=True,
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=command_timeout_s,
                    env=env,
                )
                all_output.append(result.stdout + result.stderr)
                last_returncode = result.returncode
            except subprocess.TimeoutExpired:
                all_output.append("[timeout]")
                last_returncode = 124

        # Add [call N] markers for multi-command steps so _build_tool_messages
        # can split per-call output (matches _convert_trajectory's format).
        if len(commands) > 1:
            combined = "\n".join(f"[call {k}]\n{out}" for k, out in enumerate(all_output))
        else:
            combined = all_output[0] if all_output else ""
        observation = _OBSERVATION_TEMPLATE.format(returncode=last_returncode, output=combined)
        outputs[step["step_idx"]] = observation

        # Warn on returncode mismatch
        expected_success = step.get("tool_success", True)
        actual_success = last_returncode == 0
        if actual_success != expected_success:
            logger.warning(
                "Step %d: returncode %d (success=%s) != expected success=%s",
                step["step_idx"], last_returncode, actual_success, expected_success,
            )

        logger.debug("Replayed step %d/%d: %s", step["step_idx"] + 1, up_to_step, cmd[:60])

    return outputs


# ---------------------------------------------------------------------------
# ReplayAgent
# ---------------------------------------------------------------------------


class ReplayAgent(MiniSWECodeAgent):
    """Agent that replays from a trace and resumes with new LLM calls.

    prepare() is inherited and handles repo cloning.
    run() is NOT used; the replay() orchestration function drives it.
    """

    def __init__(self, *, replay_from_step: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._replay_from_step = replay_from_step

    def _emit_step(self, record: Any) -> None:
        # Suppress prefix steps from being written to TraceLogger
        if record.step_idx < self._replay_from_step:
            return
        super()._emit_step(record)

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        # Suppress prefix events
        if data.get("step_idx", 0) < self._replay_from_step:
            return
        super()._emit_event(event_type, data)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _find_task(task_source: Path, agent_id: str) -> dict[str, Any]:
    """Look up a task by instance_id from the tasks JSON file."""
    tasks = json.loads(task_source.read_text(encoding="utf-8"))
    for task in tasks:
        if task["instance_id"] == agent_id:
            return task
    raise TraceLoadError(f"Task {agent_id!r} not found in {task_source}")


def _build_replay_run_id(agent_id: str, from_step: int) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_id = agent_id.replace("/", "-").replace(":", "-")
    return f"{safe_id}_replay_from{from_step}_{ts}"


async def replay(
    *,
    trace_path: Path,
    agent_id: str,
    from_step: int,
    task_source: Path,
    repos_root: Path,
    output_dir: Path,
    max_steps: int = 80,
    api_base: str,
    api_key: str,
    model: str | None = None,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    max_context_tokens: int = 256_000,
) -> Path:
    """Replay an agent run from a trace, resuming from `from_step`.

    Args:
        trace_path: Path to the original JSONL trace file.
        agent_id: The agent/instance ID to replay (= instance_id).
        from_step: Step index to resume from (0-indexed).
        task_source: Path to tasks JSON.
        repos_root: Path to pre-cloned repos.
        output_dir: Output directory for the new trace file.
        max_steps: New maximum steps (total, not additional).
        model: Model for new steps. If None, uses original from trace metadata.
        api_base: OpenAI-compatible API base URL.
        api_key: API key.
        command_timeout_s: Timeout per bash command.
        task_timeout_s: Timeout for the entire resumed run.
        max_context_tokens: Sliding window token budget.

    Returns:
        Path to the new replay trace JSONL file.
    """
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    # 1. Load trace + task
    logger.info("Loading trace for %s from %s", agent_id, trace_path)
    steps, summary = load_trace_steps(trace_path, agent_id)
    if from_step > len(steps):
        raise ValueError(f"from_step={from_step} exceeds available steps ({len(steps)})")
    task = _find_task(task_source, agent_id)
    if model is None:
        model = (summary or {}).get("extra", {}).get("model") or steps[0].get("extra", {}).get("model") or "qwen3.5-plus"

    logger.info("Replaying %s: steps 0..%d, then resuming with model=%s max_steps=%d",
                agent_id, from_step - 1, model, max_steps)

    # 2. Create ReplayAgent, clone repo
    agent = ReplayAgent(
        agent_id=agent_id,
        api_base=api_base,
        model=model,
        api_key=api_key,
        max_steps=max_steps,
        command_timeout_s=command_timeout_s,
        task_timeout_s=task_timeout_s,
        repos_root=str(repos_root),
        max_context_tokens=max_context_tokens,
        replay_from_step=from_step,
    )
    await agent.prepare(task)
    repo_dir = agent._workdir / "repo"

    # 3. Replay bash commands (returns full outputs)
    logger.info("Replaying %d bash commands...", from_step)
    bash_outputs = await replay_bash_commands(
        steps, from_step, repo_dir, command_timeout_s=command_timeout_s,
    )

    # 4. Reconstruct messages
    messages = reconstruct_messages(steps, from_step, bash_outputs)
    logger.info("Reconstructed %d messages for context", len(messages))

    # 5. Create ContextManagedAgent, inject state
    lm = LitellmModel(
        model_name=f"openai/{model}",
        model_kwargs={
            "api_base": api_base,
            "api_key": api_key,
            "drop_params": True,
            "temperature": 0.0,
        },
        cost_tracking="ignore_errors",
    )
    env = LocalEnvironment(
        cwd=str(repo_dir),
        timeout=int(command_timeout_s),
        env={"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"},
    )
    mini_agent = ContextManagedAgent(
        lm, env,
        system_template=_SYSTEM_TEMPLATE,
        instance_template=_INSTANCE_TEMPLATE,
        step_limit=max_steps,
        cost_limit=0.0,
        output_path=str(agent._workdir / "trajectory.json"),
        max_context_tokens=max_context_tokens,
    )

    # Inject reconstructed state (bypass DefaultAgent.run init)
    mini_agent.messages = list(messages)
    mini_agent._full_messages = list(messages)
    mini_agent.n_calls = from_step
    mini_agent.extra_template_vars = {"task": task["problem_statement"]}

    # 6. Set up trace logger
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = _build_replay_run_id(agent_id, from_step)
    trace_logger = TraceLogger(output_path, run_id)

    # Write replay metadata header
    trace_logger.log_event(agent_id, "replay_metadata", {
        "original_trace": str(trace_path),
        "from_step": from_step,
        "original_steps": len(steps),
        "model": model,
        "max_steps": max_steps,
        "max_context_tokens": max_context_tokens,
        "ts": time.time(),
    })

    agent._trace_logger = trace_logger
    agent.run_metadata = {"model": model, "api_provider": "dashscope", "replay_from": from_step}

    # 7. Run step loop (mirrors DefaultAgent.run but without init)
    wall_start = time.time()
    try:
        result: dict[str, Any] = {"exit_status": "error", "submission": ""}
        try:
            loop_result = await asyncio.wait_for(
                asyncio.to_thread(_run_step_loop, mini_agent, task["problem_statement"]),
                timeout=task_timeout_s,
            )
            if loop_result is not None:
                result = loop_result
        except TimeoutError:
            result = {"exit_status": "timeout", "submission": ""}
        except Exception as exc:
            result = {"exit_status": "error", "submission": "", "error": str(exc)}

        wall_end = time.time()

        success = (
            result.get("exit_status") == "Submitted"
            and bool(result.get("submission", "").strip())
        )
        agent.task_success = success

        # Convert trajectory (only steps >= from_step will be emitted).
        # Normalize the last prefix message's timestamp to wall_start so
        # the first resumed step doesn't compute ts_start from an old
        # timestamp (which would inflate llm_latency_ms massively).
        msg_snapshot = copy.deepcopy(mini_agent._full_messages)
        # Find the boundary: count assistant messages to locate prefix end
        asst_count = 0
        for idx, m in enumerate(msg_snapshot):
            if m.get("role") == "assistant":
                asst_count += 1
                if asst_count == from_step:
                    # The message just before the next assistant is the last
                    # prefix message; patch its timestamp to wall_start.
                    # Scan forward to find the tool result(s) after this assistant.
                    boundary = idx + 1
                    while boundary < len(msg_snapshot) and msg_snapshot[boundary].get("role") not in ("assistant", "exit"):
                        boundary += 1
                    # Patch the message right before the first new assistant
                    if boundary > 0 and boundary < len(msg_snapshot):
                        msg_snapshot[boundary - 1].setdefault("extra", {})["timestamp"] = wall_start
                    break
        agent._convert_trajectory(msg_snapshot, wall_start, wall_end)

        # Log summary
        agent_summary = agent.summary()
        agent_summary["elapsed_s"] = wall_end - wall_start
        agent_summary["replay_from_step"] = from_step
        trace_logger.log_summary(agent_id, agent_summary)

        logger.info(
            "Replay complete: %s success=%s new_steps=%d elapsed=%.1fs",
            agent_id, success, len(agent.trace), wall_end - wall_start,
        )
        return output_path / f"{run_id}.jsonl"

    finally:
        trace_logger.close()
        if agent._workdir:
            import shutil
            shutil.rmtree(agent._workdir, ignore_errors=True)


def _run_step_loop(mini_agent: ContextManagedAgent, problem_statement: str) -> dict[str, Any] | None:
    """Run the step loop synchronously (called via asyncio.to_thread)."""
    from minisweagent.agents.default import InterruptAgentFlow

    while True:
        try:
            mini_agent.step()
        except InterruptAgentFlow as e:
            mini_agent.add_messages(*e.messages)
        except Exception:
            raise
        finally:
            mini_agent.save(mini_agent.config.output_path)
        if mini_agent.messages[-1].get("role") == "exit":
            return mini_agent.messages[-1].get("extra", {})
    return None
