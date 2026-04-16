"""Multi-phase research-agent runner for host-mode research benchmarks."""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.base import TraceAction
from agents.benchmarks._research import render_research_prompt
from agents.research_agent.evidence import Evidence
from agents.research_agent.phases import (
    ExtractPhase,
    FetchPhase,
    PlanPhase,
    SearchPhase,
    SynthesizePhase,
)
from agents.research_agent.tools import TracedWebFetch, TracedWebSearch
from harness.trace_logger import TraceLogger
from llm_call import create_async_openai_client
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult
from trace_collect.latency_metrics import summarize_llm_latencies

logger = logging.getLogger(__name__)


class ResearchAgentRunner:
    """Run research-style benchmark tasks through a multi-phase pipeline.

    Phases: plan -> search -> fetch -> extract -> synthesize.
    Each phase emits canonical v5 trace records.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        max_iterations: int,
        benchmark_slug: str,
        client: Any | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.max_iterations = max_iterations
        self.benchmark_slug = benchmark_slug
        self.client = client or create_async_openai_client(
            api_base=api_base,
            api_key=api_key,
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        trace_logger = TraceLogger(attempt_ctx.attempt_dir, "trace")
        agent_id = attempt_ctx.instance_id
        instance_id = attempt_ctx.instance_id

        trace_logger.log_metadata(
            scaffold="research-agent",
            execution_environment="host",
            benchmark=self.benchmark_slug,
            model=self.model,
            api_base=self.api_base,
            max_iterations=self.max_iterations,
            instance_id=instance_id,
            prompt_template=prompt_template,
            agent_runtime_mode=attempt_ctx.agent_runtime_mode,
            scaffold_capabilities={
                "tools": ["web_search", "web_fetch"],
                "memory": False,
                "skills": False,
                "file_ops": "none",
            },
        )

        all_actions: list[TraceAction] = []
        exit_status = "completed"
        error_msg: str | None = None

        try:
            # Render the benchmark prompt (drops reference_answer, keeps task +
            # inference-time metadata like topic/difficulty/domain/source_urls).
            # This is the single source of task framing for every phase.
            task_prompt = render_research_prompt(
                self.benchmark_slug,
                task,
                prompt_template=prompt_template,
            )

            # Build tools
            search_tool = TracedWebSearch()
            fetch_tool = TracedWebFetch()

            prev_phase: str | None = None

            # --- Phase 1: Plan ---
            self._log_phase_transition(
                trace_logger, agent_id, "plan", prev_phase,
            )
            prev_phase = "plan"

            plan_phase = PlanPhase(
                self.client, self.model,
                agent_id=agent_id, instance_id=instance_id,
            )
            queries, plan_actions = await plan_phase.execute(task_prompt)
            all_actions.extend(plan_actions)
            self._log_actions(trace_logger, agent_id, plan_actions)

            # --- Phase 2: Search ---
            self._log_phase_transition(
                trace_logger, agent_id, "search", prev_phase,
            )
            prev_phase = "search"

            search_phase = SearchPhase(
                search_tool, agent_id=agent_id, instance_id=instance_id,
            )
            search_results, search_actions = await search_phase.execute(queries)
            all_actions.extend(search_actions)
            self._log_actions(trace_logger, agent_id, search_actions)

            # Graceful degradation: if search returned nothing useful
            urls = FetchPhase.extract_urls(search_results)
            fetched_pages: list[dict[str, Any]] = []
            evidence: list[Evidence] = []

            if urls:
                # --- Phase 3: Fetch ---
                self._log_phase_transition(
                    trace_logger, agent_id, "fetch", prev_phase,
                )
                prev_phase = "fetch"

                fetch_phase = FetchPhase(
                    fetch_tool, agent_id=agent_id, instance_id=instance_id,
                )
                fetched_pages, fetch_actions = await fetch_phase.execute(urls)
                all_actions.extend(fetch_actions)
                self._log_actions(trace_logger, agent_id, fetch_actions)

                # --- Phase 4: Extract ---
                self._log_phase_transition(
                    trace_logger, agent_id, "extract", prev_phase,
                )
                prev_phase = "extract"

                extract_phase = ExtractPhase(
                    self.client, self.model,
                    agent_id=agent_id, instance_id=instance_id,
                )
                evidence, extract_actions = await extract_phase.execute(
                    task_prompt, fetched_pages,
                )
                all_actions.extend(extract_actions)
                self._log_actions(trace_logger, agent_id, extract_actions)
            else:
                logger.info(
                    "No URLs found in search results; skipping fetch/extract"
                )

            # --- Phase 5: Synthesize ---
            self._log_phase_transition(
                trace_logger, agent_id, "synthesize", prev_phase,
            )

            synth_phase = SynthesizePhase(
                self.client, self.model,
                agent_id=agent_id, instance_id=instance_id,
            )
            final_answer, synth_actions = await synth_phase.execute(
                task_prompt, evidence,
            )
            all_actions.extend(synth_actions)
            self._log_actions(trace_logger, agent_id, synth_actions)

        except Exception as exc:
            logger.exception("Research agent pipeline failed: %s", exc)
            exit_status = "error"
            error_msg = str(exc)
            final_answer = ""

        # Build summary
        success = exit_status == "completed"
        summary = self._build_summary(
            agent_id=agent_id,
            instance_id=instance_id,
            actions=all_actions,
            final_answer=final_answer if success else "",
            success=success,
        )
        trace_logger.log_summary(agent_id, summary)
        trace_logger.close()

        trace_path = trace_logger.path

        return AttemptResult(
            success=success,
            exit_status=exit_status,
            trace_path=trace_path,
            model_patch="",
            summary=summary,
            error=error_msg,
            n_iterations=summary.get("n_iterations"),
            total_llm_ms=summary.get("total_llm_ms"),
            total_tool_ms=summary.get("total_tool_ms"),
            total_tokens=summary.get("total_tokens"),
            runtime_proof={
                "agent_runtime_mode": "host_controller",
                "benchmark": self.benchmark_slug,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_phase_transition(
        trace_logger: TraceLogger,
        agent_id: str,
        phase: str,
        prev_phase: str | None,
    ) -> None:
        trace_logger.log_event(
            agent_id,
            category="SESSION",
            event="phase_transition",
            data={"phase": phase, "prev_phase": prev_phase},
        )

    @staticmethod
    def _log_actions(
        trace_logger: TraceLogger,
        agent_id: str,
        actions: list[TraceAction],
    ) -> None:
        for action in actions:
            trace_logger.log_trace_action(agent_id, action)

    @staticmethod
    def _build_summary(
        *,
        agent_id: str,
        instance_id: str,
        actions: list[TraceAction],
        final_answer: str,
        success: bool,
    ) -> dict[str, Any]:
        llm_records = [a.data for a in actions if a.action_type == "llm_call"]
        llm_summary = summarize_llm_latencies(llm_records)

        total_tool_ms = sum(
            (a.data.get("duration_ms") or 0)
            for a in actions
            if a.action_type == "tool_exec"
        )
        total_tokens = sum(
            (a.data.get("prompt_tokens") or 0) + (a.data.get("completion_tokens") or 0)
            for a in actions
            if a.action_type == "llm_call"
        )

        tool_ms_by_name: dict[str, float] = {}
        for a in actions:
            if a.action_type != "tool_exec":
                continue
            tool_name = a.data.get("tool_name")
            if tool_name:
                tool_ms_by_name[tool_name] = tool_ms_by_name.get(tool_name, 0.0) + (
                    a.data.get("duration_ms") or 0.0
                )

        n_iterations = len({a.iteration for a in actions})

        return {
            "agent_id": agent_id,
            "instance_id": instance_id,
            "n_iterations": n_iterations,
            "total_llm_ms": llm_summary["total_llm_ms"],
            "total_llm_wall_ms": llm_summary["total_llm_wall_ms"],
            "total_llm_call_time_ms": llm_summary["total_llm_call_time_ms"],
            "llm_call_time_count": llm_summary["llm_call_time_count"],
            "llm_timing_source": llm_summary["llm_timing_source"],
            "total_tool_ms": total_tool_ms,
            "total_tokens": total_tokens,
            "tool_ms_by_name": tool_ms_by_name,
            "final_answer": final_answer,
            "success": success,
        }
