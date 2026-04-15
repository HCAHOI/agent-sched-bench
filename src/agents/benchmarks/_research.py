"""Shared helpers for host-mode research-style benchmark plugins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult


def _require_text(raw: dict[str, Any], field: str, *, label: str) -> str:
    value = raw.get(field)
    if value is None:
        raise ValueError(f"Missing required {label} field {field!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty required {label} field {field!r}")
    return text


def _optional_text(raw: dict[str, Any], field: str | None) -> str | None:
    if not field:
        return None
    value = raw.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_urls(raw: dict[str, Any], field: str | None) -> list[str]:
    if not field:
        return []
    value = raw.get(field)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        value = decoded
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _load_hf_rows(dataset: str, split: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[import]

    if split is None:
        raise ValueError("Host-mode research benchmarks require harness_split")
    ds = load_dataset(dataset, split=split)
    return [dict(row) for row in ds]


class HostResearchOpenClawRunner:
    """Run one research-style task through OpenClaw in host-controller mode."""

    def __init__(
        self,
        *,
        provider: Any,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        model: str,
        benchmark_slug: str,
        mcp_servers: dict[str, Any] | None = None,
        mcp_config: str | None = None,
        **_: Any,
    ) -> None:
        from agents.openclaw._session_runner import SessionRunner

        self.workspace_base = Path(workspace_base)
        self.model = model
        self.benchmark_slug = benchmark_slug
        self.mcp_config = mcp_config
        self._session_runner = SessionRunner(
            provider,
            model=model,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            mcp_servers=mcp_servers or {},
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        workspace = self.workspace_base / attempt_ctx.instance_id
        trace_path = attempt_ctx.attempt_dir / "trace.jsonl"
        prompt = self._render_prompt(task, prompt_template=prompt_template)
        result = await self._session_runner.run(
            prompt=prompt,
            workspace=workspace,
            tool_workspace=workspace,
            project_workspace=workspace,
            session_key=f"research:{attempt_ctx.instance_id}",
            trace_file=trace_path,
            instance_id=attempt_ctx.instance_id,
            channel="collect",
        )
        self._stamp_trace_metadata(
            trace_path,
            instance_id=attempt_ctx.instance_id,
            prompt_template=prompt_template,
        )
        n_iterations, total_llm_ms, total_tool_ms, total_tokens = (
            self._trace_summary_totals(trace_path)
        )
        success = result.stop_reason == "completed" and result.error is None
        return AttemptResult(
            success=success,
            exit_status=result.stop_reason,
            trace_path=trace_path,
            model_patch="",
            error=result.error,
            n_iterations=n_iterations,
            total_llm_ms=total_llm_ms,
            total_tool_ms=total_tool_ms,
            total_tokens=total_tokens,
            runtime_proof={
                "agent_runtime_mode": "host_controller",
                "benchmark": self.benchmark_slug,
            },
        )

    def _stamp_trace_metadata(
        self,
        trace_path: Path,
        *,
        instance_id: str,
        prompt_template: str,
    ) -> None:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
        stamped: list[str] = []
        replaced = False
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if not replaced and record.get("type") == "trace_metadata":
                record.update(
                    {
                        "benchmark": self.benchmark_slug,
                        "execution_environment": "host",
                        "instance_id": instance_id,
                        "prompt_template": prompt_template,
                    }
                )
                replaced = True
            stamped.append(json.dumps(record, ensure_ascii=False))
        if not replaced:
            stamped.insert(
                0,
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "openclaw",
                        "trace_format_version": 5,
                        "benchmark": self.benchmark_slug,
                        "execution_environment": "host",
                        "instance_id": instance_id,
                        "prompt_template": prompt_template,
                    },
                    ensure_ascii=False,
                ),
            )
        trace_path.write_text("\n".join(stamped) + "\n", encoding="utf-8")

    def _render_prompt(self, task: dict[str, Any], *, prompt_template: str) -> str:
        template = self._load_prompt_template(prompt_template)
        return template.replace("{{task}}", str(task["problem_statement"]))

    def _load_prompt_template(self, name: str) -> str:
        prompt_dir = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "prompts"
            / self.benchmark_slug.replace("-", "_")
        )
        path = prompt_dir / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template {name!r} not found at {path}"
            )
        text = path.read_text(encoding="utf-8")
        if "{{task}}" not in text:
            raise ValueError(f"Prompt template {path} is missing '{{{{task}}}}'")
        return text

    @staticmethod
    def _trace_summary_totals(
        trace_path: Path,
    ) -> tuple[int | None, float | None, float | None, int | None]:
        n_iterations: int | None = None
        total_llm_ms: float | None = None
        total_tool_ms: float | None = None
        total_tokens: int | None = None
        if not trace_path.exists():
            return n_iterations, total_llm_ms, total_tool_ms, total_tokens
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "summary":
                continue
            n_iterations = record.get("n_iterations")
            total_llm_ms = record.get("total_llm_ms")
            total_tool_ms = record.get("total_tool_ms")
            total_tokens = record.get("total_tokens")
        return n_iterations, total_llm_ms, total_tool_ms, total_tokens


class ResearchBenchmark(Benchmark):
    """Base for host-mode QA/research benchmark plugins."""

    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"openclaw", "qwen-deep-research"}

    @property
    def execution_environment(self) -> str:
        return "host"

    def validate_config(self) -> None:
        if not self.config.harness_dataset:
            raise ValueError(f"{self.slug} requires harness_dataset")
        if not self.config.harness_split:
            raise ValueError(f"{self.slug} requires harness_split")

    def validate_scaffold_support(self, scaffold: str) -> None:
        if scaffold not in self.SUPPORTED_SCAFFOLDS:
            raise NotImplementedError(
                f"{self.config.display_name} does not support scaffold={scaffold!r}"
            )

    def runtime_mode_for(self, scaffold: str) -> str:
        self.validate_scaffold_support(scaffold)
        return "host_controller"

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        return None

    def load_tasks(self) -> list[dict[str, Any]]:
        assert self.config.harness_dataset is not None
        rows = _load_hf_rows(self.config.harness_dataset, self.config.harness_split)
        return [self.normalize_task(row) for row in rows]

    def build_runner(self, *, scaffold: str, **kwargs: Any) -> Any:
        self.validate_scaffold_support(scaffold)
        if scaffold == "openclaw":
            return HostResearchOpenClawRunner(
                benchmark_slug=self.config.slug,
                **kwargs,
            )
        raise NotImplementedError(
            "qwen-deep-research runner is implemented in Phase 3"
        )
