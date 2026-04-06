"""SWE-bench evaluation runner — minimal OpenClaw scheduling mode.

Wires up MessageBus + AgentLoop + ResultCollector, preserving full
trace authenticity (memory, skills, MCP tools, hooks). Strips channels,
cron, heartbeat, and interactive commands.

Trace format is aligned with agent-sched-bench's StepRecord schema
for cross-benchmark comparability.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._loop import AgentLoop
from agents.openclaw.bus.events import InboundMessage
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.eval.collector import ResultCollector
from agents.openclaw.eval.prepare import prepare_workspace
from agents.openclaw.eval.types import (
    LLM,
    MCP,
    SUBAGENT,
    TOOL,
    EvalResult,
    EvalTask,
    EvalTraceEvent,
    EvalTraceSummary,
    EvalTraceStep,
)
from agents.openclaw.providers.base import LLMProvider
from agents.openclaw.session.manager import SessionManager


class TraceCollectorHook(AgentHook):
    """Collects per-iteration checkpoints and fine-grained events as JSONL.

    Emits three record types:
    - ``step``: per-iteration checkpoint (LLM call + tool results)
    - ``event``: fine-grained event from any subsystem
    - ``summary``: aggregate stats at end of run
    """

    def __init__(
        self,
        trace_file: Path,
        instance_id: str,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
    ):
        self.trace_file = trace_file
        self.instance_id = instance_id
        self.agent_id = agent_id or instance_id
        self.program_id = self.agent_id
        self.task_id = task_id or instance_id
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._wall_start = time.monotonic()
        self._total_tokens = 0
        self._n_steps = 0
        self._tool_times: dict[str, float] = {}
        self._tool_timeouts: dict[str, int] = {}
        self._tool_start_ts: dict[str, float] = {}  # wall-clock start per tool
        self._iter_start_ts: float = 0.0  # iteration start (for LLM latency)
        self._iter_start_wall: float = 0.0  # wall-clock iteration start
        self._before_exec_ts: float = 0.0  # before tools execute (LLM done)
        self._steps: list[dict[str, Any]] = []  # accumulated step dicts for summary
        self._fh = open(trace_file, "a", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def emit_event(
        self, category: str, event: str, data: dict[str, Any], *, iteration: int = 0
    ) -> None:
        """Emit a fine-grained event to the trace file."""
        entry = EvalTraceEvent(
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            event=event,
            category=category,
            data=data,
            ts=time.time(),
            iteration=iteration,
        )
        self._fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._iter_start_ts = time.monotonic()
        self._iter_start_wall = time.time()

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Record tool execution start times and LLM completion time."""
        self._before_exec_ts = time.monotonic()
        if context.tool_calls:
            for tc in context.tool_calls:
                self._tool_start_ts[tc.name] = time.monotonic()

    async def after_iteration(self, context: AgentHookContext) -> None:
        ts_now = time.time()
        self._n_steps += 1

        usage = context.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += prompt_tokens + completion_tokens

        # Approximate LLM latency: time from iteration start to before tool execution
        # If no tool calls, LLM response IS the end of the iteration
        llm_latency_ms = 0.0
        if self._before_exec_ts > self._iter_start_ts:
            llm_latency_ms = (self._before_exec_ts - self._iter_start_ts) * 1000
        elif not context.tool_calls and self._iter_start_ts > 0:
            # No tools — LLM response completed the iteration
            llm_latency_ms = (time.monotonic() - self._iter_start_ts) * 1000

        resp = context.response
        resp_dict = self._build_raw_response(
            context=context,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Build step record
        step = EvalTraceStep(
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            step_idx=context.iteration,
            phase="acting",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_latency_ms=llm_latency_ms,
            llm_output=resp.content if resp else "",
            messages_in=self._clone_messages(context.messages),
            raw_response=resp_dict,
            extra={},
            ts_start=self._iter_start_wall or ts_now,
            ts_end=ts_now,
        )

        # Extract tool execution details
        if context.tool_calls:
            # Tool calls detected in this iteration (from LLM response)
            tool_names = [tc.name for tc in context.tool_calls]
            step.tool_name = ", ".join(tool_names)
            step.tool_args = json.dumps(
                {tc.name: tc.arguments for tc in context.tool_calls},
                ensure_ascii=False,
            )[:2000]

            # Emit per-tool events
            for tc in context.tool_calls:
                is_mcp = tc.name.startswith("mcp_")
                category = MCP if is_mcp else TOOL
                event_name = "mcp_tool_call" if is_mcp else "tool_execute"

                # Parse server name from MCP tool name (mcp_{server}_{tool})
                mcp_data: dict[str, Any] = {}
                if is_mcp:
                    parts = tc.name.split("_", 2)
                    if len(parts) >= 3:
                        mcp_data["server_name"] = parts[1]
                        mcp_data["tool_name"] = parts[2]
                    else:
                        mcp_data["tool_name"] = tc.name
                else:
                    mcp_data["tool_name"] = tc.name

                mcp_data["args_preview"] = json.dumps(tc.arguments, ensure_ascii=False)[
                    :200
                ]
                self.emit_event(
                    category, event_name, mcp_data, iteration=context.iteration
                )

        # Extract tool results from context.messages (last tool-role messages)
        tool_results_from_messages = self._extract_tool_results(context.messages)
        per_tool_results: list[dict[str, Any]] = []
        if tool_results_from_messages:
            for tool_name, tool_content, tool_ok in tool_results_from_messages:
                tool_start = self._tool_start_ts.pop(tool_name, None)
                duration_ms = (
                    (time.monotonic() - tool_start) * 1000 if tool_start else 0.0
                )

                # Store per-tool result
                per_tool_results.append(
                    {
                        "tool_name": tool_name,
                        "success": tool_ok,
                        "duration_ms": round(duration_ms, 1),
                        "result": tool_content[:4000],
                    }
                )

                self._tool_times[tool_name] = (
                    self._tool_times.get(tool_name, 0.0) + duration_ms
                )
                if not tool_ok:
                    self._tool_timeouts[tool_name] = (
                        self._tool_timeouts.get(tool_name, 0) + 1
                    )

                # Emit tool result event
                is_mcp = tool_name.startswith("mcp_")
                category = MCP if is_mcp else TOOL
                event_name = "tool_complete" if tool_ok else "tool_error"
                self.emit_event(
                    category,
                    event_name,
                    {
                        "tool_name": tool_name,
                        "success": tool_ok,
                        "duration_ms": round(duration_ms, 1),
                        "result_preview": tool_content[:200],
                    },
                    iteration=context.iteration,
                )

                # Subagent completion detection
                if tool_name == "spawn":
                    self.emit_event(
                        SUBAGENT,
                        "subagent_complete",
                        {
                            "task_preview": tool_content[:200],
                        },
                        iteration=context.iteration,
                    )

        # Set step-level tool result fields
        if per_tool_results:
            if len(per_tool_results) == 1:
                # Single tool: store result directly (backward compatible)
                tr = per_tool_results[0]
                step.tool_result = tr["result"]
                step.tool_success = tr["success"]
                step.tool_duration_ms = tr["duration_ms"]
            else:
                # Multiple tools: store as JSON array with per-tool details
                step.tool_result = json.dumps(per_tool_results, ensure_ascii=False)
                # Use last tool's success/duration for step-level summary
                last = per_tool_results[-1]
                step.tool_success = last["success"]
                step.tool_duration_ms = last["duration_ms"]

        # LLM response events
        if context.response:
            finish_reason = context.response.finish_reason
            if finish_reason == "error":
                self.emit_event(
                    LLM,
                    "llm_error",
                    {
                        "error_message": context.response.content[:500]
                        if context.response.content
                        else "",
                        "finish_reason": finish_reason,
                    },
                    iteration=context.iteration,
                )
            elif finish_reason == "max_iterations":
                self.emit_event(
                    LLM,
                    "max_iterations",
                    {
                        "total_tokens": self._total_tokens,
                    },
                    iteration=context.iteration,
                )

        self._write_step(step)

    @staticmethod
    def _extract_tool_results(messages: list[dict]) -> list[tuple[str, str, bool]]:
        """Extract tool result tuples from the messages list.

        Looks for tool-role messages at the end of the list that correspond
        to the most recent tool execution round.
        """
        results = []
        # Find the last contiguous block of tool-role messages
        i = len(messages) - 1
        while i >= 0 and messages[i].get("role") == "tool":
            m = messages[i]
            name = m.get("name", "unknown")
            content = str(m.get("content", ""))
            ok = not content.startswith("Error")
            results.append((name, content, ok))
            i -= 1
        results.reverse()
        return results

    def _write_step(self, step: EvalTraceStep) -> None:
        d = step.to_dict()
        self._steps.append(d)
        self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._fh.flush()

    @staticmethod
    def _clone_messages(messages: list[dict] | None) -> list[dict[str, Any]] | None:
        """Take a JSON-safe snapshot of the current message list."""
        if not messages:
            return None
        return json.loads(json.dumps(messages, ensure_ascii=False, default=str))

    def _build_raw_response(
        self,
        *,
        context: AgentHookContext,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        """Shape openclaw responses into the trace_collect raw_response contract."""
        resp = context.response
        message: dict[str, Any] = {
            "role": "assistant",
            "content": resp.content if resp else "",
        }
        if resp and resp.reasoning_content:
            message["reasoning_content"] = resp.reasoning_content
        if context.tool_calls:
            tool_calls = []
            for idx, tc in enumerate(context.tool_calls):
                arguments = tc.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": f"call_{context.iteration}_{idx}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": arguments,
                        },
                    }
                )
            message["tool_calls"] = tool_calls

        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": resp.finish_reason if resp else None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    def write_summary(
        self,
        *,
        success: bool | None = None,
        elapsed_s: float = 0.0,
        prepare_ms: float | None = None,
    ) -> None:
        """Write summary record at the end of the task run."""
        summary = EvalTraceSummary(
            agent_id=self.agent_id,
            program_id=self.program_id,
            task_id=self.task_id,
            instance_id=self.instance_id,
            n_steps=self._n_steps,
            total_llm_ms=sum(s.get("llm_latency_ms", 0) for s in self._steps),
            total_tool_ms=sum(self._tool_times.values()),
            total_tokens=self._total_tokens,
            tool_ms_by_name=self._tool_times,
            tool_timeouts=self._tool_timeouts,
            success=success,
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )
        self._fh.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()
        self.close()


class SWEBenchRunner:
    """Runs SWE-bench tasks through OpenClaw's AgentLoop scheduling mode.

    Preserves full trace authenticity: memory consolidation, skill loading,
    MCP tools, and lifecycle hooks all participate identically to gateway mode.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace_base: Path,
        mcp_servers: dict | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
        repos_root: Path | None = None,
    ):
        self.provider = provider
        self.workspace_base = Path(workspace_base).resolve()
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.model = model or provider.get_default_model()
        self.repos_root = repos_root

    @staticmethod
    def _eval_result_route(instance_id: str) -> tuple[str, str]:
        """Return the outbound route used by eval-triggered system messages."""
        channel = "cli"
        return f"{channel}:{instance_id}", f"{channel}:{instance_id}"

    async def run_task(self, task: EvalTask) -> EvalResult:
        """Run a single evaluation task.

        Lifecycle:
        1. prepare_workspace() — git clone + checkout + pip install (if needed)
        2. AgentLoop.run() — full scheduling mode with memory/skills/MCP
        3. Extract result + model_patch from agent output
        """
        ws = task.workspace_dir
        ws.mkdir(parents=True, exist_ok=True)

        trace_file = ws / "trace.jsonl"

        # Phase 1: Prepare workspace (git clone + checkout)
        prepare_ms: float | None = None
        if task.needs_prepare:
            try:
                prepare_ms = await prepare_workspace(
                    ws,
                    repo=task.repo,
                    base_commit=task.base_commit,
                    repos_root=self.repos_root,
                )
            except Exception as e:
                logger.error("Prepare failed for {id}: {e}", id=task.instance_id, e=e)
                return EvalResult(
                    instance_id=task.instance_id,
                    content=None,
                    stop_reason="prepare_error",
                    error=str(e),
                    prepare_ms=prepare_ms,
                    trace_file=trace_file,
                    workspace_dir=ws,
                    base_commit=task.base_commit,
                )
        else:
            logger.info(
                "Skipping prepare for {id} (no repo/commit)",
                id=task.instance_id,
            )

        trace_hook = TraceCollectorHook(
            trace_file,
            task.instance_id,
            agent_id=task.instance_id,
            task_id=task.instance_id,
        )

        # Phase 2: Run agent loop
        bus = MessageBus()
        collector = ResultCollector(bus)
        session_manager = SessionManager(ws)

        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=ws,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            mcp_servers=self.mcp_servers,
            session_manager=session_manager,
            hooks=[trace_hook],
        )

        session_key = f"eval:{task.instance_id}"
        wall_start = time.monotonic()

        # Inject event callbacks into subsystems
        self._inject_event_callbacks(agent, trace_hook)

        async with AsyncExitStack() as stack:
            # Start collector
            await collector.start()
            stack.callback(collector.stop)

            # Start agent loop (runs in background, consumes from bus)
            agent_task = asyncio.create_task(agent.run())
            stack.callback(agent.stop)

            inbound_chat_id, result_key = self._eval_result_route(task.instance_id)

            # Publish the task
            msg = InboundMessage(
                channel="system",
                sender_id="eval",
                chat_id=inbound_chat_id,
                content=task.problem_statement,
                session_key_override=session_key,
            )
            await bus.publish_inbound(msg)

            # Wait for result
            content = await collector.wait_for_result(result_key)

        elapsed_s = time.monotonic() - wall_start
        run_ms = elapsed_s * 1000

        # Gather agent task to avoid orphaned tasks
        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # Write summary (this also closes the trace file handle)
        trace_hook.write_summary(
            success=bool(content and content.strip()),
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )

        # Build result from session history
        session = session_manager.get_or_create(session_key)
        tools_used = []
        tool_events = []
        usage = {}

        # Extract tool usage from session messages
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

        stop_reason = "completed"
        error = None
        if content is None and session.messages:
            # Try to get final content from last assistant message
            for m in reversed(session.messages):
                if m.get("role") == "assistant" and not m.get("tool_calls"):
                    content = m.get("content")
                    break

        return EvalResult(
            instance_id=task.instance_id,
            content=content,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            trace_file=trace_file,
            prepare_ms=prepare_ms,
            run_ms=run_ms,
            workspace_dir=ws,
            base_commit=task.base_commit,
        )

    @staticmethod
    def _inject_event_callbacks(agent: AgentLoop, hook: TraceCollectorHook) -> None:
        """Inject event callback hooks into subsystems for trace emission."""

        def emit(category: str, event: str, data: dict, iteration: int = 0) -> None:
            hook.emit_event(category, event, data, iteration=iteration)

        # Memory consolidation callbacks
        if hasattr(agent, "memory_consolidator") and hasattr(
            agent.memory_consolidator, "_event_callback"
        ):
            agent.memory_consolidator._event_callback = lambda cat, evt, d, it=0: emit(
                cat, evt, d, it
            )

        # Skill loading callbacks
        if (
            hasattr(agent, "context")
            and hasattr(agent.context, "skills")
            and hasattr(agent.context.skills, "_event_callback")
        ):
            agent.context.skills._event_callback = lambda cat, evt, d, it=0: emit(
                cat, evt, d, it
            )

        # MCP connection callbacks
        if hasattr(agent, "_mcp_event_callback"):
            agent._mcp_event_callback = lambda cat, evt, d: emit(cat, evt, d)

        # Session callbacks
        if hasattr(agent, "sessions") and hasattr(agent.sessions, "_event_callback"):
            agent.sessions._event_callback = lambda cat, evt, d: emit(cat, evt, d)

        # Top-level event callback (openclaw AgentLoop)
        if hasattr(agent, "_event_callback"):
            agent._event_callback = lambda cat, evt, d: emit(cat, evt, d)

    async def run_batch(
        self,
        tasks: list[EvalTask],
        max_concurrent: int = 1,
        on_progress: Callable[[EvalResult], None] | None = None,
    ) -> list[EvalResult]:
        """Run multiple tasks concurrently using a semaphore.

        Args:
            tasks: List of tasks to run.
            max_concurrent: Maximum concurrent tasks (maps to semaphore).
            on_progress: Optional callback called with each completed result.

        Returns:
            List of EvalResult in the same order as input tasks.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[EvalResult | None] = [None] * len(tasks)

        async def _run_one(idx: int, task: EvalTask) -> None:
            async with semaphore:
                try:
                    result = await self.run_task(task)
                    results[idx] = result
                    if on_progress:
                        on_progress(result)
                except Exception as e:
                    logger.error("Task {} failed: {}", task.instance_id, e)
                    results[idx] = EvalResult(
                        instance_id=task.instance_id,
                        content=None,
                        stop_reason="error",
                        error=str(e),
                    )
                    if on_progress:
                        on_progress(results[idx])

        await asyncio.gather(*[_run_one(i, t) for i, t in enumerate(tasks)])
        return [r for r in results if r is not None]
