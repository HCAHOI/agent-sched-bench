"""OpenClaw host runner for BrowseComp tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.deep_research.runner import DeepResearchRunner, _model_tool_result_chars
from agents.deep_research.web_tools import (
    DeepResearchWebFetchTool,
    DeepResearchWebSearchTool,
    ToolExecutionRecord,
    WebToolRuntimeState,
)
from agents.openclaw._session_runner import SessionRunner
from agents.openclaw.config.schema import ExecToolConfig
from trace_collect.attempt_pipeline import AttemptArtifact, AttemptContext, AttemptResult
from trace_collect.prompt_loader import load_prompt_template, render_prompt
from trace_collect.trace_data import trace_summary_totals


_WEB_ONLY_TOOLS = frozenset({"web_search", "web_fetch"})
_BROWSECOMP_PARENT_TOOLS = frozenset(
    {"web_search", "web_fetch", "spawn", "sessions_yield"}
)


class BrowseCompOpenClawRunner(DeepResearchRunner):
    """Run BrowseComp tasks through OpenClaw with a web-only host tool surface."""

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        problem = str(task["problem_statement"])
        reference_answer = str(task["reference_answer"])
        template = load_prompt_template(prompt_template, self.benchmark_slug)
        user_prompt = render_prompt(template, problem)

        attempt_ctx.attempt_dir.mkdir(parents=True, exist_ok=True)
        trace_path = attempt_ctx.attempt_dir / "trace.jsonl"
        workspace = self.workspace_base / attempt_ctx.instance_id
        runtime_dir = attempt_ctx.attempt_dir / "openclaw-runtime"
        artifacts: list[AttemptArtifact] = []
        tool_state = WebToolRuntimeState(
            max_search_calls=self.web_config.max_search_calls,
            max_fetch_calls=self.web_config.max_fetch_calls,
        )
        preview_chars = int(
            self.benchmark_extras.get("tool_result_preview_chars", 1200)
        )
        max_tool_result_chars = _model_tool_result_chars(
            self.benchmark_extras,
            fetch_max_chars=self.web_config.fetch_max_chars,
            preview_chars=preview_chars,
        )
        max_concurrent_research_units = _max_concurrent_research_units(
            self.benchmark_extras
        )
        run_config = {
            "provider": self.provider_name,
            "api_base": self.api_base,
            "env_key": self.env_key,
            "generation_config": dict(self.generation_config),
            "scorer_provider": self.scorer_provider_name,
            "scorer_model": self.scorer_model,
            "scorer_temperature": self.scorer_temperature,
            "web_search_provider": self.web_config.search_provider,
            "web_search_api_key_env": self.web_config.search_api_key_env,
            "web_search_fallback_provider": self.web_config.search_fallback_provider,
            "web_search_base_url": self.web_config.search_base_url,
            "web_fetch_provider": self.web_config.fetch_provider,
            "web_fetch_api_key_env": self.web_config.fetch_api_key_env,
            "web_fetch_fallback_provider": self.web_config.fetch_fallback_provider,
            "web_fetch_max_chars": self.web_config.fetch_max_chars,
            "max_search_calls": self.web_config.max_search_calls,
            "max_fetch_calls": self.web_config.max_fetch_calls,
            "max_concurrent_research_units": max_concurrent_research_units,
            "tool_result_preview_chars": preview_chars,
            "max_tool_result_chars": max_tool_result_chars,
        }
        runner = SessionRunner(
            self.provider,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=max_tool_result_chars,
            mcp_servers={},
            exec_config=ExecToolConfig(enable=False),
            enabled_tools=_BROWSECOMP_PARENT_TOOLS,
            subagent_enabled_tools=_WEB_ONLY_TOOLS,
            subagent_max_active_per_session=max_concurrent_research_units,
            tool_overrides=[
                DeepResearchWebSearchTool(
                    config=self.web_config,
                    state=tool_state,
                    environ=self.environ,
                ),
                DeepResearchWebFetchTool(
                    config=self.web_config,
                    state=tool_state,
                    environ=self.environ,
                ),
            ],
        )
        result = await runner.run(
            prompt=user_prompt,
            workspace=workspace,
            tool_workspace=workspace,
            project_workspace=workspace,
            session_key=f"browsecomp-openclaw:{attempt_ctx.instance_id}",
            trace_file=trace_path,
            runtime_dir=runtime_dir,
            instance_id=attempt_ctx.instance_id,
            channel="cli",
            runtime_label="host:web-only",
            metadata_overrides={
                "benchmark": self.benchmark_slug,
                "execution_environment": "host",
                "agent_runtime_mode": "host_controller",
                "prompt_template": prompt_template,
                "run_config": run_config,
                "sensitive_artifacts": bool(
                    self.benchmark_extras.get("sensitive_artifacts", True)
                ),
                "task_source_kind": task.get("task_source_kind"),
                "task_source_id": task.get("task_source_id"),
                "task_source_path": task.get("task_source_path"),
                "task_source_sha256": task.get("task_source_sha256"),
                "task_source_schema": task.get("task_source_schema"),
                "task_source_row_count": task.get("task_source_row_count"),
                "parent_tool_policy": sorted(_BROWSECOMP_PARENT_TOOLS),
                "child_tool_policy": sorted(_WEB_ONLY_TOOLS),
            },
        )
        _enrich_trace_tool_actions(trace_path, tool_state.records)

        summary, exit_status, success, error = await self._summarize_attempt(
            task=task,
            model_response=result.content or "",
            stop_reason=result.stop_reason,
            tool_state=tool_state,
            artifacts=artifacts,
            attempt_dir=attempt_ctx.attempt_dir,
            reference_answer=reference_answer,
        )
        existing_summary = _last_summary(trace_path, attempt_ctx.instance_id)
        total_llm_ms, total_tool_ms, total_tokens = trace_summary_totals(trace_path)
        summary.update(
            {
                "n_iterations": existing_summary.get("n_iterations", 0),
                "total_llm_ms": total_llm_ms or 0.0,
                "total_tool_ms": total_tool_ms or 0.0,
                "total_tokens": total_tokens or 0,
                "tool_ms_by_name": existing_summary.get("tool_ms_by_name", {}),
                "search_calls_used": tool_state.search_calls_used,
                "fetch_calls_used": tool_state.fetch_calls_used,
                "tool_budget_exhausted": tool_state.tool_budget_exhausted,
                "budget_exhausted_tool": tool_state.budget_exhausted_tool,
                "tool_backend_failed": tool_state.tool_backend_failed,
                "backend_failed_tool": tool_state.backend_failed_tool,
                "backend_failure_error": tool_state.backend_failure_error,
                "success": success,
                "exit_status": exit_status,
            }
        )
        _append_summary(trace_path, attempt_ctx.instance_id, existing_summary, summary)

        return AttemptResult(
            success=success,
            exit_status=exit_status,
            trace_path=trace_path,
            model_patch=result.content or "",
            tool_calls=_tool_payloads_from_trace(trace_path),
            summary=summary,
            error=error or result.error,
            n_iterations=int(summary["n_iterations"]),
            total_llm_ms=summary["total_llm_ms"],
            total_tool_ms=summary["total_tool_ms"],
            total_tokens=summary["total_tokens"],
            runtime_proof={
                "scaffold": "openclaw",
                "runtime": "host_web_only",
                "tools": sorted(_BROWSECOMP_PARENT_TOOLS),
                "child_tools": sorted(_WEB_ONLY_TOOLS),
                "agent_runner": "agents.openclaw._session_runner.SessionRunner",
            },
            artifacts=artifacts,
        )

def _max_concurrent_research_units(extras: dict[str, Any]) -> int:
    raw_value = extras.get("max_concurrent_research_units", 1)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_concurrent_research_units must be an integer, got {raw_value!r}"
        ) from exc
    if value < 1:
        raise ValueError(f"max_concurrent_research_units must be >= 1, got {value}")
    return value


def _enrich_trace_tool_actions(
    trace_path: Path, records: list[ToolExecutionRecord]
) -> None:
    if not records or not trace_path.exists():
        return
    web_records_by_call_id = {
        record.tool_call_id: record.to_trace_payload()
        for record in records
        if record.tool_call_id
    }
    if not web_records_by_call_id:
        return

    trace_records: list[dict[str, Any]] = []
    changed = False
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec":
            data = record.get("data")
            if isinstance(data, dict):
                web_record = web_records_by_call_id.get(data.get("tool_call_id"))
                if web_record is not None:
                    for key in (
                        "requested_provider",
                        "actual_provider",
                        "fallback_used",
                        "budget_exhausted",
                        "artifact_path",
                        "artifact_sha256",
                        "original_size",
                        "preview",
                        "truncated_preview",
                    ):
                        if key in web_record:
                            data[key] = web_record[key]
                    web_success = bool(web_record.get("success", True))
                    data["success"] = bool(data.get("success", True)) and web_success
                    web_error = web_record.get("error")
                    if web_error:
                        data["error"] = web_error
                    elif web_record.get("budget_exhausted"):
                        data["error"] = "tool_budget_exhausted"
                    changed = True
        trace_records.append(record)

    if changed:
        trace_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in trace_records),
            encoding="utf-8",
        )



def _last_summary(trace_path: Path, agent_id: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not trace_path.exists():
        return latest
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "summary" and record.get("agent_id") == agent_id:
            latest = dict(record)
    return latest


def _append_summary(
    trace_path: Path,
    agent_id: str,
    existing_summary: dict[str, Any],
    browsecomp_summary: dict[str, Any],
) -> None:
    merged = {**existing_summary, **browsecomp_summary}
    merged["type"] = "summary"
    merged["agent_id"] = agent_id
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(merged, ensure_ascii=False) + "\n")


def _tool_payloads_from_trace(trace_path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not trace_path.exists():
        return payloads
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "action" or record.get("action_type") != "tool_exec":
            continue
        data = dict(record.get("data") or {})
        data.setdefault("action_id", record.get("action_id"))
        payloads.append(data)
    return payloads
