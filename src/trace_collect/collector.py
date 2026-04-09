"""SWE-rebench trace collection pipeline.

Two entry points dispatched by ``--scaffold``:
  * ``collect_miniswe_traces`` — mini-swe-agent runs inside the task
    container via ``DockerEnvironment``.
  * ``collect_openclaw_traces`` — openclaw runs its structured tool loop
    inside the task container via ``ContainerToolBackend``.

Both produce the same ``<run_dir>/<instance_id>/attempt_1/{trace.jsonl,
run_manifest.json, results.json, resources.json, tool_calls.json,
container_stdout.txt}`` layout, orchestrated by
``attempt_pipeline.run_attempt``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trace_collect.attempt_pipeline import (
    AttemptContext,
    AttemptResult,
    run_attempt,
    start_task_container,
    stop_task_container,
)

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP config parsing (shared between scaffolds)
# ---------------------------------------------------------------------------


def _mcp_config_label(mcp_config: str | None) -> str | None:
    """Map an ``--mcp-config`` value to its trace-header label.

    ``None`` → flag absent, ``"none"`` → affirmative opt-out,
    ``<basename>`` → YAML at that basename was loaded.
    """
    if mcp_config is None:
        return None
    if mcp_config == "none":
        return "none"
    return Path(mcp_config).name


def load_mcp_servers(mcp_config: str | None) -> dict:
    """Parse a ``--mcp-config`` argument into ``dict[str, MCPServerConfig]``."""
    if mcp_config is None or mcp_config == "none":
        return {}

    import yaml

    path = Path(mcp_config)
    if not path.exists():
        raise FileNotFoundError(f"--mcp-config path does not exist: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"--mcp-config root must be a mapping, got {type(raw).__name__}"
        )

    from agents.openclaw.config.schema import MCPServerConfig

    servers: dict[str, Any] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"--mcp-config entry {name!r} must be a mapping, "
                f"got {type(entry).__name__}"
            )
        servers[name] = MCPServerConfig.model_validate(entry)
    return servers


# ---------------------------------------------------------------------------
# Per-task result + run directory
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CollectedTaskResult:
    """Per-task result emitted alongside each attempt's artifacts.

    Intentionally slim: ``success`` is derived from ``bool(model_patch)``
    at construction time, so there is no ``success_basis`` enum or
    redundant ``patch_generated`` flag.
    """

    instance_id: str
    attempt_dir: Path
    model_patch: str
    exit_status: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    n_iterations: int | None = None

    @property
    def success(self) -> bool:
        return bool(self.model_patch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "attempt_dir": str(self.attempt_dir),
            "trace_file": str(self.attempt_dir / "trace.jsonl"),
            "success": self.success,
            "model_patch": self.model_patch,
            "exit_status": self.exit_status,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "n_iterations": self.n_iterations,
        }


def build_run_dir(benchmark: "Benchmark", model: str) -> Path:
    """Build run directory from benchmark plugin config: ``trace_root/model/ts/``."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    return benchmark.config.trace_root / safe_model / ts


def load_completed_ids(run_dir: Path) -> set[str]:
    """Return instance_ids whose ``attempt_*/run_manifest.json`` is ``completed``.

    Only the nested attempt layout is supported — no legacy flat scan.
    """
    completed: set[str] = set()
    if not run_dir.exists():
        return completed
    for instance_dir in run_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        for attempt_dir in sorted(instance_dir.glob("attempt_*")):
            manifest_path = attempt_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("status") == "completed":
                completed.add(instance_dir.name)
                break
    return completed


def write_results_jsonl(
    results: list[CollectedTaskResult], results_path: Path
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Shared per-task loop
# ---------------------------------------------------------------------------


async def _run_scaffold_tasks(
    *,
    benchmark: "Benchmark",
    tasks: list[dict[str, Any]],
    run_dir: Path,
    model: str,
    scaffold: str,
    prompt_template: str,
    min_free_disk_gb: float,
    inner_factory,
) -> Path:
    """Iterate over tasks, wrapping each in ``run_attempt``.

    ``inner_factory`` is a callable ``(task: dict, ctx: AttemptContext) ->
    Awaitable[AttemptResult]`` that the scaffold provides to run its agent
    loop. Results are collected into ``<run_dir>/results.jsonl`` at the end.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(run_dir)
    if completed:
        logger.info("Resuming: %d tasks already completed", len(completed))

    results: list[CollectedTaskResult] = []
    total = len(tasks)

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        if instance_id in completed:
            logger.info(
                "[%d/%d] SKIP %s (already completed)", i + 1, total, instance_id
            )
            continue

        logger.info(
            "[%d/%d] START %s (%s)", i + 1, total, instance_id, scaffold
        )
        t0 = time.monotonic()

        attempt_ctx = AttemptContext(
            run_dir=run_dir,
            instance_id=instance_id,
            attempt=1,
            task=task,
            model=model,
            requested_model=model,
            scaffold=scaffold,
            source_image=task.get("image_name") or "",
            prompt_template=prompt_template,
        )

        # Bind `task` into the inner via a default arg to avoid the closure
        # capturing a stale reference in the async iterator.
        _inner = inner_factory(task)

        try:
            result = await run_attempt(
                attempt_ctx,
                inner=_inner,
                min_free_disk_gb=min_free_disk_gb,
            )
        except Exception as exc:
            logger.exception("FAILED %s", instance_id)
            results.append(
                CollectedTaskResult(
                    instance_id=instance_id,
                    attempt_dir=attempt_ctx.attempt_dir,
                    model_patch="",
                    exit_status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_s=time.monotonic() - t0,
                )
            )
            continue

        results.append(
            CollectedTaskResult(
                instance_id=instance_id,
                attempt_dir=attempt_ctx.attempt_dir,
                model_patch=getattr(result, "model_patch", "") or "",
                exit_status=result.exit_status,
                error=result.error,
                elapsed_s=time.monotonic() - t0,
                n_iterations=result.n_iterations,
            )
        )
        logger.info(
            "[%d/%d] DONE %s success=%s elapsed=%.1fs",
            i + 1,
            total,
            instance_id,
            results[-1].success,
            results[-1].elapsed_s,
        )

    write_results_jsonl(results, run_dir / "results.jsonl")
    logger.info("Results written to %s", run_dir / "results.jsonl")
    return run_dir


# ---------------------------------------------------------------------------
# mini-swe-agent scaffold
# ---------------------------------------------------------------------------


async def collect_miniswe_traces(
    *,
    api_base: str,
    api_key: str,
    model: str,
    benchmark: "Benchmark",
    max_iterations: int = 60,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    sample: int | None = None,
    instance_ids: list[str] | None = None,
    run_id: str | None = None,
    max_context_tokens: int = 256_000,
    prompt_template: str = "default",
    min_free_disk_gb: float = 30.0,
) -> Path:
    """Collect mini-swe-agent traces inside the SWE-rebench task container."""
    if benchmark.task_shape != "swe_patch":
        raise ValueError(
            f"mini-swe-agent only supports swe_patch benchmarks; "
            f"{benchmark.config.slug!r} has task_shape={benchmark.task_shape!r}"
        )

    tasks = benchmark.load_tasks()
    tasks = [benchmark.normalize_task(t) for t in tasks]
    if instance_ids is not None:
        id_set = set(instance_ids)
        tasks = [t for t in tasks if t["instance_id"] in id_set]
        if not tasks:
            raise ValueError(f"No tasks matched instance_ids: {instance_ids}")
    if sample is not None:
        tasks = tasks[:sample]

    run_dir = Path(run_id) if run_id else build_run_dir(benchmark, model)

    from agents.miniswe import MiniSWECodeAgent

    def make_inner(task: dict):
        async def inner(ctx: AttemptContext) -> AttemptResult:
            from harness.trace_logger import TraceLogger

            trace_logger = TraceLogger(ctx.attempt_dir, "trace")
            trace_logger.log_metadata(
                scaffold="mini-swe-agent",
                benchmark=benchmark.config.slug,
                benchmark_split=benchmark.config.harness_split,
                model=model,
                api_base=api_base,
                max_iterations=max_iterations,
                instance_id=ctx.instance_id,
                scaffold_capabilities={
                    "tools": ["bash"],
                    "memory": False,
                    "skills": False,
                    "file_ops": "bash_only",
                },
            )

            agent = MiniSWECodeAgent(
                agent_id=ctx.instance_id,
                api_base=api_base,
                model=model,
                api_key=api_key,
                max_iterations=max_iterations,
                command_timeout_s=command_timeout_s,
                task_timeout_s=task_timeout_s,
                max_context_tokens=max_context_tokens,
                prompt_template=ctx.prompt_template,
            )
            agent._trace_logger = trace_logger
            agent.run_metadata = {"model": model}

            try:
                await agent.prepare(task)
                success = await agent.run(task, attempt_ctx=ctx)
            finally:
                summary = agent.summary()
                trace_logger.log_summary(agent.agent_id, summary)
                trace_logger.close()

            return AttemptResult(
                success=bool(success),
                exit_status=agent.task_exit_status,
                trace_path=trace_logger.path,
                model_patch=(agent.task_submission or "").strip(),
                n_iterations=summary.get("n_iterations") or len(agent.trace),
                total_llm_ms=summary.get("total_llm_ms"),
                total_tool_ms=summary.get("total_tool_ms"),
                total_tokens=summary.get("total_tokens"),
                error=agent.task_error,
            )

        return inner

    return await _run_scaffold_tasks(
        benchmark=benchmark,
        tasks=tasks,
        run_dir=run_dir,
        model=model,
        scaffold="mini-swe-agent",
        prompt_template=prompt_template,
        min_free_disk_gb=min_free_disk_gb,
        inner_factory=make_inner,
    )


# ---------------------------------------------------------------------------
# openclaw scaffold
# ---------------------------------------------------------------------------


async def collect_openclaw_traces(
    *,
    api_base: str,
    api_key: str,
    model: str,
    benchmark: "Benchmark",
    max_iterations: int = 80,
    sample: int | None = None,
    instance_ids: list[str] | None = None,
    run_id: str | None = None,
    max_context_tokens: int = 256_000,
    mcp_config: str | None = None,
    prompt_template: str = "default",
    min_free_disk_gb: float = 30.0,
) -> Path:
    """Collect openclaw traces inside the SWE-rebench task container."""
    if benchmark.task_shape != "swe_patch":
        raise ValueError(
            f"openclaw SWE-rebench collection only supports swe_patch; "
            f"{benchmark.config.slug!r} has task_shape={benchmark.task_shape!r}"
        )

    tasks = benchmark.load_tasks()
    tasks = [benchmark.normalize_task(t) for t in tasks]
    if instance_ids is not None:
        id_set = set(instance_ids)
        tasks = [t for t in tasks if t["instance_id"] in id_set]
        if not tasks:
            raise ValueError(f"No tasks matched instance_ids: {instance_ids}")
    if sample is not None:
        tasks = tasks[:sample]

    run_dir = Path(run_id) if run_id else build_run_dir(benchmark, model)
    mcp_servers = load_mcp_servers(mcp_config)
    mcp_config_label = _mcp_config_label(mcp_config)

    from agents.openclaw.eval.types import EvalTask
    from agents.openclaw.tools.container_backend import ContainerWorkspace
    from agents.openclaw.unified_provider import UnifiedProvider

    provider = UnifiedProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model,
    )
    runner = benchmark.build_runner(
        scaffold="openclaw",
        provider=provider,
        workspace_base=run_dir / "_workspaces",
        max_iterations=max_iterations,
        context_window_tokens=max_context_tokens,
        model=model,
        mcp_servers=mcp_servers,
    )

    def make_inner(task: dict):
        async def inner(ctx: AttemptContext) -> AttemptResult:
            fixed_image = ctx.fixed_image or task.get("image_name") or ""
            if not fixed_image:
                raise RuntimeError(
                    f"Task {ctx.instance_id!r} has no image_name"
                )
            container_id = start_task_container(fixed_image)
            ctx.mark_container_ready(container_id)
            cw = ContainerWorkspace(container_id=container_id, cwd="/testbed")

            try:
                eval_task = EvalTask.from_benchmark_instance(
                    task, run_dir / "_workspaces", benchmark=benchmark
                )
                eval_result = await runner.run_task(
                    eval_task,
                    container_workspace=cw,
                    prompt_template=ctx.prompt_template,
                )
            finally:
                ctx.container_stdout = stop_task_container(container_id)

            model_patch = eval_result.model_patch or ""
            if (
                eval_result.trace_file
                and Path(eval_result.trace_file).exists()
            ):
                _normalize_openclaw_trace(
                    src=Path(eval_result.trace_file),
                    dst=ctx.attempt_dir / "trace.jsonl",
                    benchmark=benchmark,
                    model=model,
                    api_base=api_base,
                    max_iterations=max_iterations,
                    instance_id=ctx.instance_id,
                    mcp_config_label=mcp_config_label,
                )

            return AttemptResult(
                success=bool(model_patch),
                exit_status=eval_result.stop_reason,
                trace_path=ctx.attempt_dir / "trace.jsonl",
                model_patch=model_patch,
                n_iterations=eval_result.n_iterations,
                error=eval_result.error,
            )

        return inner

    return await _run_scaffold_tasks(
        benchmark=benchmark,
        tasks=tasks,
        run_dir=run_dir,
        model=model,
        scaffold="openclaw",
        prompt_template=prompt_template,
        min_free_disk_gb=min_free_disk_gb,
        inner_factory=make_inner,
    )


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


async def collect_traces(
    *,
    scaffold: str,
    **kwargs: Any,
) -> Path:
    """Dispatch to ``collect_miniswe_traces`` or ``collect_openclaw_traces``."""
    if scaffold == "mini-swe-agent":
        kwargs.pop("mcp_config", None)
        return await collect_miniswe_traces(**kwargs)
    if scaffold == "openclaw":
        kwargs.pop("command_timeout_s", None)
        kwargs.pop("task_timeout_s", None)
        return await collect_openclaw_traces(**kwargs)
    raise ValueError(f"Unknown scaffold: {scaffold!r}")


# ---------------------------------------------------------------------------
# openclaw trace metadata merge
# ---------------------------------------------------------------------------


def _normalize_openclaw_trace(
    src: Path,
    dst: Path,
    *,
    benchmark: "Benchmark",
    model: str,
    api_base: str,
    max_iterations: int,
    instance_id: str,
    mcp_config_label: str | None = None,
) -> None:
    """Copy an OpenClaw trace into the attempt dir, merging trace_metadata.

    Runner-stamped fields win on conflict so ``scaffold_capabilities`` from
    the live session survives. Missing fields are filled from the collector's
    knowledge (benchmark slug, split, model, etc.).
    """
    lines = src.read_text(encoding="utf-8").splitlines()
    source_metadata: dict[str, Any] | None = None
    body_start = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            body_start = idx + 1
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            body_start = idx + 1
            continue
        if rec.get("type") == "trace_metadata":
            source_metadata = rec
            body_start = idx + 1
        else:
            body_start = idx
        break

    merged: dict[str, Any] = {
        "type": "trace_metadata",
        "scaffold": "openclaw",
        "trace_format_version": 5,
        "mode": "collect",
        "scaffold_capabilities": {
            "unknown": True,
            "reason": "source trace had no metadata record",
        },
    }
    if source_metadata is not None:
        merged.update(source_metadata)
    merged.setdefault("benchmark", benchmark.config.slug)
    merged.setdefault("benchmark_split", benchmark.config.harness_split)
    merged.setdefault("model", model)
    merged.setdefault("api_base", api_base)
    merged.setdefault("max_iterations", max_iterations)
    merged.setdefault("instance_id", instance_id)
    merged["type"] = "trace_metadata"
    merged["trace_format_version"] = 5
    if mcp_config_label is not None:
        run_config = merged.get("run_config") or {}
        run_config["mcp_config"] = mcp_config_label
        merged["run_config"] = run_config

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged, ensure_ascii=False) + "\n")
        for line in lines[body_start:]:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "trace_metadata":
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
