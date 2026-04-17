from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from trace_collect.attempt_pipeline import AttemptContext, AttemptResult


class TerminalBenchRunner:
    AGENT_IMPORT_PATH = "agents.terminal_bench.openclaw_agent:TerminalBenchOpenClawAgent"
    TRACE_FILENAME = "openclaw-trace.jsonl"

    def __init__(
        self,
        *,
        provider_name: str | None,
        env_key: str | None,
        api_base: str,
        api_key: str,
        model: str,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        benchmark_slug: str,
        benchmark_extras: dict[str, Any],
        mcp_config: str | None = None,
    ) -> None:
        self.provider_name = provider_name or "openai"
        self.env_key = env_key or "OPENAI_API_KEY"
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.workspace_base = Path(workspace_base)
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.benchmark_slug = benchmark_slug
        self.benchmark_extras = dict(benchmark_extras)
        self.mcp_config = self._resolve_mcp_config(mcp_config)
        self.mcp_config_label = self._mcp_config_label(mcp_config)

    async def run_openclaw_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        return await asyncio.to_thread(
            self._run_openclaw_task_sync,
            task,
            attempt_ctx,
            prompt_template,
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        return await self.run_openclaw_task(
            task,
            attempt_ctx=attempt_ctx,
            prompt_template=prompt_template,
        )

    def _run_openclaw_task_sync(
        self,
        task: dict[str, Any],
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        proof = self._preflight()
        run_root = attempt_ctx.attempt_dir / "_terminal_bench_run"
        run_id = attempt_ctx.instance_id.replace("/", "_")
        run_root.mkdir(parents=True, exist_ok=True)

        command = self._build_tb_command(
            task=task,
            run_root=run_root,
            run_id=run_id,
            prompt_template=prompt_template,
        )
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[3]
        env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env[self.env_key] = self.api_key
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "terminal-bench run failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        tb_run_path = run_root / run_id
        success = self._extract_success(tb_run_path)
        trace_path = self._find_trace_path(tb_run_path)
        summary = {
            "tb_version": proof["tb_version"],
            "tb_dataset": task.get("tb_dataset"),
            "tb_registry_source": task.get("tb_registry_source"),
            "adapter_kind": "terminal_bench_openclaw",
            "agent_import_path": self.AGENT_IMPORT_PATH,
            "tb_run_path": str(tb_run_path),
        }
        if self.mcp_config_label is not None:
            summary["mcp_config"] = self.mcp_config_label
        if not trace_path.exists():
            raise RuntimeError(f"terminal-bench trace file not found under {tb_run_path}")
        normalized_trace = run_root / f"{attempt_ctx.instance_id}-terminal-bench-trace.jsonl"
        self._augment_trace_metadata(
            src=trace_path,
            dst=normalized_trace,
            task=task,
            prompt_template=prompt_template,
            tb_version=proof["tb_version"],
        )
        return AttemptResult(
            success=success,
            exit_status="completed" if success else "failed",
            trace_path=normalized_trace,
            model_patch="",
            error=None if success else "Terminal-Bench task did not resolve",
            summary=summary,
            runtime_proof=proof,
        )

    def _preflight(self) -> dict[str, Any]:
        if importlib.util.find_spec("terminal_bench") is None:
            raise RuntimeError("terminal-bench Python package is not installed")
        tb_bin = shutil.which("tb")
        if not tb_bin:
            raise RuntimeError("tb CLI is not available on PATH")
        docker_bin = shutil.which("docker")
        if not docker_bin:
            raise RuntimeError("docker is not available on PATH")
        return {
            "tb_version": importlib.metadata.version("terminal-bench"),
            "tb_path": tb_bin,
            "docker_path": docker_bin,
            "agent_runtime_mode": "host_controller",
        }

    def _build_tb_command(
        self,
        *,
        task: dict[str, Any],
        run_root: Path,
        run_id: str,
        prompt_template: str,
    ) -> list[str]:
        dataset_root = task["dataset_root"]
        task_id = task["task_id"]
        command = [
            "tb",
            "run",
            "--dataset-path",
            str(dataset_root),
            "--task-id",
            str(task_id),
            "--output-path",
            str(run_root),
            "--run-id",
            run_id,
            "--n-concurrent",
            "1",
            "--agent-import-path",
            self.AGENT_IMPORT_PATH,
            "--model",
            self.model,
            "--agent-kwarg",
            f"provider_name={self.provider_name}",
            "--agent-kwarg",
            f"api_base={self.api_base}",
            "--agent-kwarg",
            f"env_key={self.env_key}",
            "--agent-kwarg",
            "api_key=",
            "--agent-kwarg",
            f"max_iterations={self.max_iterations}",
        ]
        prompt_template_path = self._materialize_prompt_template(
            prompt_template=prompt_template,
            run_root=run_root,
        )
        command.extend(
            [
                "--agent-kwarg",
                f"prompt_template={prompt_template_path}",
            ]
        )
        if self.mcp_config is not None:
            command.extend(
                [
                    "--agent-kwarg",
                    f"mcp_config_path={self.mcp_config}",
                ]
            )
        return command

    def _extract_success(self, tb_run_path: Path) -> bool:
        results_path = tb_run_path / "results.json"
        if not results_path.exists():
            raise RuntimeError(f"terminal-bench results.json missing at {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if not results:
            return False
        first = results[0]
        return bool(first.get("is_resolved"))

    def _find_trace_path(self, tb_run_path: Path) -> Path:
        # Trace is written to /logs (CONTAINER_SESSION_LOGS_PATH) inside the container,
        # which mounts to sessions/ on the host. Fall back to agent-logs/ for old runs.
        for pattern in (
            f"**/sessions/{self._trace_filename()}",
            f"**/agent-logs/{self._trace_filename()}",
        ):
            traces = sorted(tb_run_path.glob(pattern))
            if traces:
                return traces[0]
        return tb_run_path / "missing-trace.jsonl"

    def _augment_trace_metadata(
        self,
        *,
        src: Path,
        dst: Path,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
    ) -> None:
        lines = src.read_text(encoding="utf-8").splitlines()
        source_metadata: dict[str, Any] | None = None
        body_start = 0
        for idx, line in enumerate(lines):
            if not line.strip():
                body_start = idx + 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                body_start = idx + 1
                continue
            if record.get("type") == "trace_metadata":
                source_metadata = record
                body_start = idx + 1
            else:
                body_start = idx
            break

        merged: dict[str, Any] = source_metadata.copy() if source_metadata else {}
        merged.update(
            {
                "type": "trace_metadata",
                "trace_format_version": 5,
                "mode": "collect",
                "scaffold": "openclaw",
                "execution_environment": "host",
                "benchmark": self.benchmark_slug,
                "model": self.model,
                "max_iterations": self.max_iterations,
                "instance_id": task["instance_id"],
                "agent_runtime_mode": "host_controller",
                "prompt_template": prompt_template,
                "task_source_kind": task.get("task_source_kind"),
                "task_source_id": task.get("task_source_id"),
                "task_source_path": task.get("task_source_path"),
                "tb_version": tb_version,
                "tb_dataset": task.get("tb_dataset"),
                "tb_registry_source": task.get("tb_registry_source"),
                "adapter_kind": "terminal_bench_openclaw",
                "agent_import_path": self.AGENT_IMPORT_PATH,
            }
        )
        if self.mcp_config_label is not None:
            run_config = merged.get("run_config") or {}
            run_config["mcp_config"] = self.mcp_config_label
            merged["run_config"] = run_config

        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            for line in lines[body_start:]:
                if not line.strip():
                    continue
                handle.write(line + "\n")

    def _trace_filename(self) -> str:
        return self.TRACE_FILENAME

    def _materialize_prompt_template(
        self,
        *,
        prompt_template: str,
        run_root: Path,
    ) -> Path:
        from trace_collect.prompt_loader import load_prompt_template

        run_root.mkdir(parents=True, exist_ok=True)
        template_text = load_prompt_template(prompt_template).replace(
            "{{task}}",
            "{{ instruction }}",
        )
        safe_name = prompt_template.replace("/", "_")
        template_path = run_root / f"prompt-template-{safe_name}.j2"
        template_path.write_text(template_text, encoding="utf-8")
        return template_path.resolve()

    @staticmethod
    def _resolve_mcp_config(mcp_config: str | None) -> str | None:
        if mcp_config in {None, "none"}:
            return None
        return str(Path(mcp_config).expanduser().resolve())

    @staticmethod
    def _mcp_config_label(mcp_config: str | None) -> str | None:
        if mcp_config is None:
            return None
        if mcp_config == "none":
            return "none"
        return Path(mcp_config).name
