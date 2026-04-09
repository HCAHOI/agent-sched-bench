"""Trace collection entrypoints for SWE-style benchmarks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.container_image_prep import (
    drop_cached_fixed_image,
    ensure_source_image,
    normalize_image_reference,
    prune_dangling_images,
    remove_image,
)
from trace_collect.attempt_pipeline import (
    AttemptContext,
    AttemptResult,
    run_attempt,
    start_task_container,
    stop_task_container,
)
from trace_collect.runtime.task_container import (
    preflight_task_container_runtime,
    project_mount_args,
    run_task_container_agent,
)

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


def _mcp_config_label(mcp_config: str | None) -> str | None:
    """Map ``--mcp-config`` to the value stored in trace metadata."""
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


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass(slots=True)
class CollectedTaskResult:
    """Per-task summary emitted alongside attempt artifacts."""

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


def _select_tasks(
    tasks: list[dict[str, Any]],
    *,
    instance_ids: list[str] | None,
    sample: int | None,
) -> list[dict[str, Any]]:
    """Filter tasks while preserving the explicit ``instance_ids`` order."""
    selected = list(tasks)
    if instance_ids is not None:
        by_id = {task["instance_id"]: task for task in tasks}
        missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
        if missing:
            raise ValueError(f"No tasks matched instance_ids: {missing}")
        selected = [by_id[instance_id] for instance_id in instance_ids]
    if sample is not None:
        selected = selected[:sample]
    return selected


def _task_source_image(task: dict[str, Any]) -> str:
    return normalize_image_reference(str(task.get("image_name") or ""))


def _next_pending_source_image(
    tasks: list[dict[str, Any]],
    *,
    current_index: int,
    completed: set[str],
) -> str | None:
    """Return the next incomplete task's source image, if any."""
    for next_task in tasks[current_index + 1 :]:
        if next_task["instance_id"] in completed:
            continue
        source_image = _task_source_image(next_task)
        if source_image:
            return source_image
    return None


def _ensure_task_source_ready(
    *,
    instance_id: str,
    source_image: str,
    prefetched_source_image: str | None,
    prefetch_future: Future[None] | None,
    executable: str = "podman",
) -> None:
    """Ensure the current task's source image is available locally."""
    if not source_image:
        return
    if (
        prefetch_future is not None
        and prefetched_source_image == source_image
    ):
        try:
            prefetch_future.result()
            logger.info("prefetch ready for %s image=%s", instance_id, source_image)
            return
        except Exception as exc:
            logger.warning(
                "prefetch failed for %s image=%s; retrying foreground: %s",
                instance_id,
                source_image,
                exc,
            )
    ensure_source_image(source_image, executable=executable)


def _cleanup_task_images(
    *,
    instance_id: str,
    source_image: str,
    fixed_image: str | None,
    keep_source_image: str | None,
    executable: str = "podman",
) -> None:
    """Best-effort cleanup that keeps only the current/next-image budget."""
    removed_any = False
    try:
        if fixed_image and fixed_image != source_image:
            removed_fixed = remove_image(fixed_image, executable=executable)
            removed_any = removed_fixed or removed_any
            if removed_fixed:
                logger.info(
                    "cleanup %s removed fixed image %s",
                    instance_id,
                    fixed_image,
                )
    except Exception as exc:
        logger.warning(
            "cleanup %s failed removing fixed image %s: %s",
            instance_id,
            fixed_image,
            exc,
        )

    try:
        if source_image and source_image != keep_source_image:
            removed_source = remove_image(source_image, executable=executable)
            removed_any = removed_source or removed_any
            if removed_source:
                logger.info(
                    "cleanup %s removed source image %s",
                    instance_id,
                    source_image,
                )
    except Exception as exc:
        logger.warning(
            "cleanup %s failed removing source image %s: %s",
            instance_id,
            source_image,
            exc,
        )

    if source_image:
        drop_cached_fixed_image(source_image)

    if not removed_any:
        return
    try:
        prune_dangling_images(executable=executable)
    except Exception as exc:
        logger.warning("cleanup %s prune failed: %s", instance_id, exc)


async def _run_scaffold_tasks(
    *,
    benchmark: "Benchmark",
    tasks: list[dict[str, Any]],
    run_dir: Path,
    model: str,
    scaffold: str,
    prompt_template: str | None,
    min_free_disk_gb: float,
    inner_factory,
) -> Path:
    """Iterate over tasks, wrapping each in ``run_attempt``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(run_dir)
    if completed:
        logger.info("Resuming: %d tasks already completed", len(completed))

    results: list[CollectedTaskResult] = []
    total = len(tasks)
    prefetched_source_image: str | None = None
    prefetch_future: Future[None] | None = None

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-prefetch") as executor:
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
            source_image = _task_source_image(task)
            next_source_image = _next_pending_source_image(
                tasks,
                current_index=i,
                completed=completed,
            )

            attempt_ctx = AttemptContext(
                run_dir=run_dir,
                instance_id=instance_id,
                attempt=1,
                task=task,
                model=model,
                scaffold=scaffold,
                source_image=source_image,
                prompt_template=_resolve_prompt_template(
                    benchmark=benchmark,
                    prompt_template=prompt_template,
                ),
                agent_runtime_mode=benchmark.runtime_mode_for(scaffold),
            )

            _inner = inner_factory(task)

            try:
                _ensure_task_source_ready(
                    instance_id=instance_id,
                    source_image=source_image,
                    prefetched_source_image=prefetched_source_image,
                    prefetch_future=prefetch_future,
                )
                prefetched_source_image = None
                prefetch_future = None

                if next_source_image and next_source_image != source_image:
                    logger.info(
                        "prefetch start for next task after %s image=%s",
                        instance_id,
                        next_source_image,
                    )
                    prefetched_source_image = next_source_image
                    prefetch_future = executor.submit(
                        ensure_source_image,
                        next_source_image,
                    )

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
            else:
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
            finally:
                _cleanup_task_images(
                    instance_id=instance_id,
                    source_image=source_image,
                    fixed_image=attempt_ctx.fixed_image,
                    keep_source_image=next_source_image,
                )

    write_results_jsonl(results, run_dir / "results.jsonl")
    logger.info("Results written to %s", run_dir / "results.jsonl")
    return run_dir


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
    prompt_template: str | None = None,
    min_free_disk_gb: float = 30.0,
) -> Path:
    """Collect miniswe traces inside the SWE-rebench task container."""
    tasks = _select_tasks(
        benchmark.load_tasks(),
        instance_ids=instance_ids,
        sample=sample,
    )

    run_dir = Path(run_id) if run_id else build_run_dir(benchmark, model)

    from agents.miniswe import MiniSWECodeAgent

    def make_inner(task: dict):
        async def inner(ctx: AttemptContext) -> AttemptResult:
            if ctx.agent_runtime_mode == "task_container_agent":
                return await _run_miniswe_in_task_container(
                    ctx=ctx,
                    task=task,
                    benchmark=benchmark,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    max_iterations=max_iterations,
                    command_timeout_s=command_timeout_s,
                    task_timeout_s=task_timeout_s,
                    max_context_tokens=max_context_tokens,
                )

            from harness.trace_logger import TraceLogger

            trace_logger = TraceLogger(ctx.attempt_dir, "trace")
            trace_logger.log_metadata(
                scaffold="miniswe",
                benchmark=benchmark.config.slug,
                benchmark_split=benchmark.config.harness_split,
                model=model,
                api_base=api_base,
                max_iterations=max_iterations,
                instance_id=ctx.instance_id,
                prompt_template=ctx.prompt_template,
                agent_runtime_mode=ctx.agent_runtime_mode,
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
                runtime_proof={},
            )

        return inner

    return await _run_scaffold_tasks(
        benchmark=benchmark,
        tasks=tasks,
        run_dir=run_dir,
        model=model,
        scaffold="miniswe",
        prompt_template=prompt_template,
        min_free_disk_gb=min_free_disk_gb,
        inner_factory=make_inner,
    )


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
    prompt_template: str | None = None,
    min_free_disk_gb: float = 30.0,
) -> Path:
    """Collect openclaw traces inside the SWE-rebench task container."""
    tasks = _select_tasks(
        benchmark.load_tasks(),
        instance_ids=instance_ids,
        sample=sample,
    )

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
            if ctx.agent_runtime_mode == "task_container_agent":
                return await _run_openclaw_in_task_container(
                    ctx=ctx,
                    task=task,
                    benchmark=benchmark,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    max_iterations=max_iterations,
                    max_context_tokens=max_context_tokens,
                    mcp_config=mcp_config,
                )

            fixed_image = ctx.fixed_image or task.get("image_name") or ""
            if not fixed_image:
                raise RuntimeError(
                    f"Task {ctx.instance_id!r} has no image_name"
                )
            container_id = start_task_container(fixed_image)
            ctx.mark_container_ready(container_id)
            cw = ContainerWorkspace(container_id=container_id, cwd="/testbed")

            try:
                eval_task = EvalTask(
                    instance_id=task["instance_id"],
                    problem_statement=task.get("problem_statement", ""),
                    workspace_dir=run_dir / "_workspaces" / task["instance_id"],
                    repo=task.get("repo"),
                    base_commit=task.get("base_commit"),
                    image_name=task.get("image_name"),
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
                    prompt_template=ctx.prompt_template,
                    agent_runtime_mode=ctx.agent_runtime_mode,
                    runtime_proof={"container_id": container_id},
                )

            return AttemptResult(
                success=bool(model_patch),
                exit_status=eval_result.stop_reason,
                trace_path=ctx.attempt_dir / "trace.jsonl",
                model_patch=model_patch,
                n_iterations=eval_result.n_iterations,
                error=eval_result.error,
                runtime_proof={"container_id": container_id},
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


async def collect_traces(
    *,
    scaffold: str,
    **kwargs: Any,
) -> Path:
    """Dispatch to ``collect_miniswe_traces`` or ``collect_openclaw_traces``."""
    if scaffold == "miniswe":
        kwargs.pop("mcp_config", None)
        return await collect_miniswe_traces(**kwargs)
    if scaffold == "openclaw":
        kwargs.pop("command_timeout_s", None)
        kwargs.pop("task_timeout_s", None)
        return await collect_openclaw_traces(**kwargs)
    raise ValueError(f"Unknown scaffold: {scaffold!r}")


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
    prompt_template: str | None = None,
    agent_runtime_mode: str | None = None,
    runtime_proof: dict[str, Any] | None = None,
) -> None:
    """Copy an OpenClaw trace into the attempt dir, merging trace metadata."""
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
        "scaffold_capabilities": {"unknown": True},
    }
    if source_metadata is not None:
        merged.update(source_metadata)
    merged.setdefault("benchmark", benchmark.config.slug)
    merged.setdefault("benchmark_split", benchmark.config.harness_split)
    merged.setdefault("model", model)
    merged.setdefault("api_base", api_base)
    merged.setdefault("max_iterations", max_iterations)
    merged.setdefault("instance_id", instance_id)
    if prompt_template is not None:
        merged["prompt_template"] = prompt_template
    if agent_runtime_mode is not None:
        merged["agent_runtime_mode"] = agent_runtime_mode
    if runtime_proof:
        merged["runtime_proof"] = runtime_proof
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


def _resolve_prompt_template(
    *,
    benchmark: "Benchmark",
    prompt_template: str | None,
) -> str:
    return prompt_template or benchmark.config.default_prompt_template


async def _run_miniswe_in_task_container(
    *,
    ctx: AttemptContext,
    task: dict[str, Any],
    benchmark: "Benchmark",
    api_base: str,
    api_key: str,
    model: str,
    max_iterations: int,
    command_timeout_s: float,
    task_timeout_s: float,
    max_context_tokens: int,
) -> AttemptResult:
    fixed_image = ctx.fixed_image or task.get("image_name") or ""
    if not fixed_image:
        raise RuntimeError(f"Task {ctx.instance_id!r} has no image_name")

    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "miniswe"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    proof = None
    runtime = None
    container_id = start_task_container(
        fixed_image,
        extra_args=project_mount_args(ctx.attempt_dir),
    )
    ctx.mark_container_ready(container_id)
    try:
        proof = preflight_task_container_runtime(
            container_id=container_id,
            attempt_dir=ctx.attempt_dir,
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime = run_task_container_agent(
            container_id=container_id,
            timeout=task_timeout_s + 120,
            request={
                "kind": "run_miniswe",
                "scaffold": "miniswe",
                "result_path": str(runtime_dir / "run.result.json"),
                "container_id": container_id,
                "benchmark": benchmark.config.slug,
                "benchmark_split": benchmark.config.harness_split,
                "api_base": api_base,
                "api_key": api_key,
                "model": model,
                "max_iterations": max_iterations,
                "command_timeout_s": command_timeout_s,
                "task_timeout_s": task_timeout_s,
                "max_context_tokens": max_context_tokens,
                "prompt_template": ctx.prompt_template,
                "agent_runtime_mode": ctx.agent_runtime_mode,
                "exec_working_dir": "/testbed",
                "task": task,
                "trace_file": str(runtime_dir / "trace.jsonl"),
                "raw_stdout_path": str(stdout_path),
                "raw_stderr_path": str(stderr_path),
            },
        )
    finally:
        container_logs = stop_task_container(container_id)
        ctx.container_stdout = "\n".join(
            part
            for part in [
                stdout_path.read_text(encoding="utf-8")
                if stdout_path.exists()
                else "",
                stderr_path.read_text(encoding="utf-8")
                if stderr_path.exists()
                else "",
                container_logs,
            ]
            if part
        )

    assert proof is not None
    assert runtime is not None
    runtime_proof = {
        **asdict(proof),
        **runtime.runtime_proof,
    }
    return AttemptResult(
        success=runtime.success,
        exit_status=runtime.exit_status,
        trace_path=runtime.trace_path,
        model_patch=runtime.model_patch,
        n_iterations=runtime.n_iterations,
        total_llm_ms=runtime.total_llm_ms,
        total_tool_ms=runtime.total_tool_ms,
        total_tokens=runtime.total_tokens,
        error=runtime.error,
        runtime_proof=runtime_proof,
    )


async def _run_openclaw_in_task_container(
    *,
    ctx: AttemptContext,
    task: dict[str, Any],
    benchmark: "Benchmark",
    api_base: str,
    api_key: str,
    model: str,
    max_iterations: int,
    max_context_tokens: int,
    mcp_config: str | None,
) -> AttemptResult:
    fixed_image = ctx.fixed_image or task.get("image_name") or ""
    if not fixed_image:
        raise RuntimeError(f"Task {ctx.instance_id!r} has no image_name")

    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    proof = None
    runtime = None
    runtime_proof = None
    container_id = start_task_container(
        fixed_image,
        extra_args=project_mount_args(ctx.attempt_dir),
    )
    ctx.mark_container_ready(container_id)
    try:
        proof = preflight_task_container_runtime(
            container_id=container_id,
            attempt_dir=ctx.attempt_dir,
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime = run_task_container_agent(
            container_id=container_id,
            timeout=(max_iterations * 120) + 300,
            request={
                "kind": "run_openclaw",
                "scaffold": "openclaw",
                "result_path": str(runtime_dir / "run.result.json"),
                "container_id": container_id,
                "benchmark": benchmark.config.slug,
                "api_base": api_base,
                "api_key": api_key,
                "model": model,
                "max_iterations": max_iterations,
                "max_context_tokens": max_context_tokens,
                "prompt_template": ctx.prompt_template,
                "agent_runtime_mode": ctx.agent_runtime_mode,
                "mcp_config": (
                    str(Path(mcp_config).resolve())
                    if mcp_config not in {None, "none"}
                    else mcp_config
                ),
                "task": task,
                "workspace_base": str(runtime_dir / "workspace_base"),
                "workspace_dir": str(runtime_dir / "workspace_base" / ctx.instance_id),
                "tool_workspace": "/testbed",
                "exec_working_dir": "/testbed",
                "trace_file": str(runtime_dir / "trace.raw.jsonl"),
                "raw_stdout_path": str(stdout_path),
                "raw_stderr_path": str(stderr_path),
            },
        )
        runtime_proof = {
            **asdict(proof),
            **runtime.runtime_proof,
        }
        _normalize_openclaw_trace(
            src=runtime.trace_path,
            dst=ctx.attempt_dir / "trace.jsonl",
            benchmark=benchmark,
            model=model,
            api_base=api_base,
            max_iterations=max_iterations,
            instance_id=ctx.instance_id,
            mcp_config_label=_mcp_config_label(mcp_config),
            prompt_template=ctx.prompt_template,
            agent_runtime_mode=ctx.agent_runtime_mode,
            runtime_proof=runtime_proof,
        )
    finally:
        container_logs = stop_task_container(container_id)
        ctx.container_stdout = "\n".join(
            part
            for part in [
                stdout_path.read_text(encoding="utf-8")
                if stdout_path.exists()
                else "",
                stderr_path.read_text(encoding="utf-8")
                if stderr_path.exists()
                else "",
                container_logs,
            ]
            if part
        )
    assert runtime is not None
    assert runtime_proof is not None
    return AttemptResult(
        success=bool(runtime.model_patch),
        exit_status=runtime.exit_status,
        trace_path=ctx.attempt_dir / "trace.jsonl",
        model_patch=runtime.model_patch,
        n_iterations=runtime.n_iterations,
        total_llm_ms=runtime.total_llm_ms,
        total_tool_ms=runtime.total_tool_ms,
        total_tokens=runtime.total_tokens,
        error=runtime.error,
        runtime_proof=runtime_proof,
    )
