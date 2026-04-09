"""Shared bus-based session runner for CLI and evaluation flows."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._loop import AgentLoop
from agents.openclaw.bus.events import InboundMessage
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.eval.collector import ResultCollector
from agents.base import TraceAction
from agents.openclaw.eval.types import (
    LLM,
    MCP,
    SUBAGENT,
    TOOL,
    EvalTraceEvent,
    EvalTraceSummary,
)
from agents.openclaw.providers.base import LLMProvider
from agents.openclaw.session.manager import SessionManager

class TraceCollectorHook(AgentHook):
    """Collect per-iteration actions, events, and summaries as JSONL."""

    def __init__(
        self,
        trace_file: Path,
        instance_id: str,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self.trace_file = trace_file
        self.instance_id = instance_id
        self.agent_id = agent_id or instance_id
        self.program_id = self.agent_id
        self.task_id = task_id or instance_id
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._wall_start = time.monotonic()
        self._total_tokens = 0
        self._n_iterations = 0
        self._tool_times: dict[str, float] = {}
        self._tool_timeouts: dict[str, int] = {}
        self._tool_start_ts: dict[str, float] = {}
        self._iter_start_wall: float = 0.0
        self._iter_messages_snapshot: list[dict[str, Any]] | None = None
        self._before_exec_wall: float = 0.0
        self._actions: list[dict[str, Any]] = []
        self._fh = open(trace_file, "a", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def emit_event(
        self,
        category: str,
        event: str,
        data: dict[str, Any],
        *,
        iteration: int = 0,
    ) -> None:
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
        self._iter_start_wall = time.time()
        self._iter_messages_snapshot = self._clone_messages(context.messages)
        self.emit_event(
            LLM, "llm_call_start",
            {"messages_in": self._iter_messages_snapshot},
            iteration=context.iteration,
        )

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        self._before_exec_wall = time.time()
        if context.tool_calls:
            for tc in context.tool_calls:
                self._tool_start_ts[tc.name] = time.monotonic()
                is_mcp = tc.name.startswith("mcp_")
                self.emit_event(
                    MCP if is_mcp else TOOL,
                    "tool_exec_start",
                    {
                        "tool_name": tc.name,
                        "args_preview": json.dumps(
                            tc.arguments, ensure_ascii=False
                        )[:200],
                    },
                    iteration=context.iteration,
                )

    async def after_iteration(self, context: AgentHookContext) -> None:
        ts_now = time.time()
        self._n_iterations += 1

        usage = context.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += prompt_tokens + completion_tokens

        llm_ts_end = self._before_exec_wall or ts_now
        llm_latency_ms = (llm_ts_end - self._iter_start_wall) * 1000
        resp_dict = self._build_raw_response(
            context=context,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.emit_event(
            LLM, "llm_call_end",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "llm_latency_ms": round(llm_latency_ms, 2),
                "finish_reason": context.response.finish_reason if context.response else None,
            },
            iteration=context.iteration,
        )
        llm_action = TraceAction(
            action_type="llm_call",
            action_id=f"llm_{context.iteration}",
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            iteration=context.iteration,
            ts_start=self._iter_start_wall,
            ts_end=llm_ts_end,
            data={
                "messages_in": self._iter_messages_snapshot,
                "raw_response": resp_dict,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "llm_latency_ms": round(llm_latency_ms, 2),
            },
        )
        self._write_action(llm_action)
        self._before_exec_wall = 0.0
        self._iter_messages_snapshot = None

        tool_results_from_messages = self._extract_tool_results(context.messages)
        if tool_results_from_messages:
            tool_args_map: dict[str, str] = {}
            if context.tool_calls:
                for tc in context.tool_calls:
                    tool_args_map[tc.name] = json.dumps(
                        tc.arguments, ensure_ascii=False
                    )[:2000]

            for tool_name, tool_content, tool_ok in tool_results_from_messages:
                tool_start_mono = self._tool_start_ts.pop(tool_name, None)
                duration_ms = (
                    (time.monotonic() - tool_start_mono) * 1000
                    if tool_start_mono
                    else 0.0
                )
                self._tool_times[tool_name] = (
                    self._tool_times.get(tool_name, 0.0) + duration_ms
                )
                if not tool_ok:
                    self._tool_timeouts[tool_name] = (
                        self._tool_timeouts.get(tool_name, 0) + 1
                    )

                is_mcp = tool_name.startswith("mcp_")
                self.emit_event(
                    MCP if is_mcp else TOOL,
                    "tool_exec_end",
                    {
                        "tool_name": tool_name,
                        "success": tool_ok,
                        "duration_ms": round(duration_ms, 1),
                        "result_preview": tool_content[:200],
                    },
                    iteration=context.iteration,
                )
                if tool_name == "spawn":
                    self.emit_event(
                        SUBAGENT,
                        "subagent_complete",
                        {"task_preview": tool_content[:200]},
                        iteration=context.iteration,
                    )

                tool_ts_end = time.time()
                tool_ts_start = (
                    tool_ts_end - duration_ms / 1000 if duration_ms else tool_ts_end
                )
                tool_action = TraceAction(
                    action_type="tool_exec",
                    action_id=f"tool_{context.iteration}_{tool_name}",
                    agent_id=self.agent_id,
                    program_id=self.program_id,
                    instance_id=self.instance_id,
                    iteration=context.iteration,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end,
                    data={
                        "tool_name": tool_name,
                        "tool_args": tool_args_map.get(tool_name, ""),
                        "tool_result": tool_content[:4000],
                        "duration_ms": round(duration_ms, 1),
                        "success": tool_ok,
                    },
                )
                self._write_action(tool_action)

        if context.response:
            finish_reason = context.response.finish_reason
            if finish_reason == "error":
                error_data: dict[str, Any] = {
                    "error_message": context.response.content[:500]
                    if context.response.content
                    else "",
                    "finish_reason": finish_reason,
                }
                if context.response.extra:
                    error_data.update(context.response.extra)
                self.emit_event(
                    LLM,
                    "llm_error",
                    error_data,
                    iteration=context.iteration,
                )
            elif finish_reason == "max_iterations":
                self.emit_event(
                    LLM,
                    "max_iterations",
                    {"total_tokens": self._total_tokens},
                    iteration=context.iteration,
                )

    @staticmethod
    def _extract_tool_results(messages: list[dict]) -> list[tuple[str, str, bool]]:
        results: list[tuple[str, str, bool]] = []
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

    def _write_action(self, action: TraceAction) -> None:
        d = action.to_dict()
        self._actions.append(d)
        self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._fh.flush()

    @staticmethod
    def _clone_messages(messages: list[dict] | None) -> list[dict[str, Any]] | None:
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
        summary = EvalTraceSummary(
            agent_id=self.agent_id,
            program_id=self.program_id,
            task_id=self.task_id,
            instance_id=self.instance_id,
            n_iterations=self._n_iterations,
            total_llm_ms=sum(
                a.get("data", {}).get("llm_latency_ms", 0)
                for a in self._actions
                if a.get("action_type") == "llm_call"
            ),
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

def inject_event_callbacks(agent: AgentLoop, hook: TraceCollectorHook) -> None:
    def emit(category: str, event: str, data: dict, iteration: int = 0) -> None:
        hook.emit_event(category, event, data, iteration=iteration)

    if hasattr(agent, "memory_consolidator") and hasattr(
        agent.memory_consolidator, "_event_callback"
    ):
        agent.memory_consolidator._event_callback = lambda cat, evt, d, si=0: emit(
            cat, evt, d, si
        )

    if (
        hasattr(agent, "context")
        and hasattr(agent.context, "skills")
        and hasattr(agent.context.skills, "_event_callback")
    ):
        agent.context.skills._event_callback = lambda cat, evt, d, si=0: emit(
            cat, evt, d, si
        )

    if hasattr(agent, "_mcp_event_callback"):
        agent._mcp_event_callback = lambda cat, evt, d: emit(cat, evt, d)

    if hasattr(agent, "sessions") and hasattr(agent.sessions, "_event_callback"):
        agent.sessions._event_callback = lambda cat, evt, d: emit(cat, evt, d)

    if hasattr(agent, "_event_callback"):
        agent._event_callback = lambda cat, evt, d: emit(cat, evt, d)

@dataclass
class SessionRunResult:

    content: str | None
    elapsed_s: float
    trace_file: Path | None = None
    session_key: str = ""
    session_manager: SessionManager | None = None

class SessionRunner:

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        mcp_servers: dict | None = None,
        extra_hooks: list[AgentHook] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens or 65536
        self.max_tool_result_chars = max_tool_result_chars
        self.mcp_servers = mcp_servers or {}
        self.extra_hooks = extra_hooks or []

    async def run(
        self,
        prompt: str,
        workspace: Path,
        *,
        session_key: str,
        trace_file: Path,
        instance_id: str | None = None,
        channel: str = "cli",
        prepare_ms: float | None = None,
        container_workspace: Any = None,
    ) -> SessionRunResult:
        workspace.mkdir(parents=True, exist_ok=True)
        iid = instance_id or session_key

        trace_hook = TraceCollectorHook(trace_file, iid, agent_id=iid, task_id=iid)

        import json as _json

        metadata = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "mode": "collect",
            "model": self.model,
            "instance_id": iid,
            "session_key": session_key,
            "max_iterations": self.max_iterations,
            "scaffold_capabilities": {
                "tools": [
                    "read_file",
                    "write_file",
                    "edit_file",
                    "list_dir",
                    "exec",
                    "web_search",
                    "web_fetch",
                    "message",
                    "spawn",
                ],
                "memory": True,
                "skills": True,
                "file_ops": "structured",
            },
        }
        trace_hook._fh.write(_json.dumps(metadata, ensure_ascii=False) + "\n")
        trace_hook._fh.flush()

        bus = MessageBus()
        collector = ResultCollector(bus)
        session_manager = SessionManager(workspace)

        all_hooks: list[AgentHook] = [trace_hook, *self.extra_hooks]
        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=workspace,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            mcp_servers=self.mcp_servers,
            session_manager=session_manager,
            hooks=all_hooks,
            container_workspace=container_workspace,
        )

        inject_event_callbacks(agent, trace_hook)

        wall_start = time.monotonic()

        chat_id = session_key.split(":", 1)[-1] if ":" in session_key else session_key
        result_key = f"{channel}:{chat_id}"

        async with AsyncExitStack() as stack:
            await collector.start()
            stack.callback(collector.stop)

            agent_task = asyncio.create_task(agent.run())
            stack.callback(agent.stop)

            msg = InboundMessage(
                channel="system",
                sender_id="user",
                chat_id=f"{channel}:{chat_id}",
                content=prompt,
                session_key_override=session_key,
            )
            await bus.publish_inbound(msg)

            content = await collector.wait_for_result(result_key)

        elapsed_s = time.monotonic() - wall_start

        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        trace_hook.write_summary(
            success=bool(content and content.strip()),
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )

        return SessionRunResult(
            content=content,
            elapsed_s=elapsed_s,
            trace_file=trace_file,
            session_key=session_key,
            session_manager=session_manager,
        )
