"""Trace collection entrypoints for SWE-style benchmarks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_call import UnifiedProvider
from agents.openclaw.runtime_deps import OPENCLAW_MCP_RUNTIME_REQUIREMENTS

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
    configure_task_container_apt_mirror,
    mcp_config_label,
    next_attempt_number_in,
    run_attempt,
    sanitize_path_segment,
    start_task_container,
    stop_task_container,
)
from trace_collect.runtime.task_container import (
    bootstrap_task_container_python,
    preflight_task_container_runtime,
    resolve_task_container_exec_config,
    resolve_running_container_exec_config,
    run_task_container_agent,
)

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)
_DOCKER_HOST_GATEWAY = "172.17.0.1"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSY_ENV_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class _CollectModelBackend:
    """Materialized cloud model client and endpoint exposed to the scaffold."""

    provider: Any
    provider_name: str | None
    api_base: str
    api_key: str
    trace_run_config: dict[str, Any]


def _prepare_collect_model_backend(
    *,
    model: str,
    api_base: str,
    api_key: str,
    provider_name: str | None,
    generation_config: dict[str, Any],
) -> _CollectModelBackend:
    """Build the cloud/OpenAI-compatible model-provider path."""
    provider = UnifiedProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model,
        **generation_config,
    )
    return _CollectModelBackend(
        provider=provider,
        provider_name=provider_name,
        api_base=api_base,
        api_key=api_key,
        trace_run_config={},
    )


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


def _generation_config(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }.items()
        if value is not None
    }


@dataclass(slots=True)
class CollectedTaskResult:
    """Per-task summary emitted alongside attempt artifacts."""

    instance_id: str
    attempt_dir: Path
    success: bool
    model_patch: str = ""
    exit_status: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    n_iterations: int | None = None

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
    safe_model = sanitize_path_segment(model)
    return benchmark.config.trace_root / safe_model / ts


_RESUME_TERMINAL_STATUSES = frozenset({"completed", "exhausted"})


def _is_resume_terminal_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("status") in _RESUME_TERMINAL_STATUSES


def load_completed_ids(run_dir: Path) -> set[str]:
    """Return instance_ids whose attempts are terminal for ``--run-id`` resume.

    ``completed`` attempts and ``exhausted`` max-iteration attempts should not be
    rerun when resuming the same run directory. ``error`` manifests are not
    resume-terminal, even if their error text mentions max-iteration exhaustion.
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
            if _is_resume_terminal_manifest(manifest):
                completed.add(instance_dir.name)
                break
    return completed


def next_attempt_number(run_dir: Path, instance_id: str) -> int:
    """Return the next attempt index for ``run_dir/instance_id``."""
    return next_attempt_number_in(run_dir / instance_id)


def write_results_jsonl(results: list[CollectedTaskResult], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def _select_tasks(
    tasks: list[dict[str, Any]],
    *,
    instance_ids: list[str] | None,
    sample: int | None,
    skip: int = 0,
) -> list[dict[str, Any]]:
    """Filter tasks, then apply ``skip`` and ``sample`` in that order."""
    selected = list(tasks)
    if instance_ids is not None:
        by_id = {task["instance_id"]: task for task in tasks}
        missing = [
            instance_id for instance_id in instance_ids if instance_id not in by_id
        ]
        if missing:
            raise ValueError(f"No tasks matched instance_ids: {missing}")
        selected = [by_id[instance_id] for instance_id in instance_ids]
    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
    if skip:
        selected = selected[skip:]
    if sample is not None:
        if sample < 0:
            raise ValueError(f"sample must be non-negative, got {sample}")
        selected = selected[:sample]
    return selected


def _task_source_image(benchmark: "Benchmark", task: dict[str, Any]) -> str | None:
    image_name = benchmark.image_name_for(task)
    if not image_name:
        return None
    return normalize_image_reference(str(image_name))


def _next_pending_source_image(
    benchmark: "Benchmark",
    tasks: list[dict[str, Any]],
    *,
    current_index: int,
    completed: set[str],
) -> str | None:
    """Return the next incomplete task's source image, if any."""
    for next_task in tasks[current_index + 1 :]:
        if next_task["instance_id"] in completed:
            continue
        source_image = _task_source_image(benchmark, next_task)
        if source_image:
            return source_image
    return None


def _ensure_task_source_ready(
    *,
    instance_id: str,
    source_image: str | None,
    prefetched_source_image: str | None,
    prefetch_future: Future[None] | None,
    container_executable: str | None,
) -> None:
    """Ensure the current task's source image is available locally."""
    if not source_image:
        return
    if container_executable is None:
        raise ValueError("container_executable is required when source_image is set")
    if prefetch_future is not None and prefetched_source_image == source_image:
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
    ensure_source_image(
        source_image,
        container_executable=container_executable,
    )


def _task_image_cleanup_enabled() -> bool:
    value = os.environ.get("TASK_CONTAINER_CLEANUP_IMAGES", "").strip().lower()
    if not value:
        return False
    if value in _TRUTHY_ENV_VALUES:
        return True
    if value in _FALSY_ENV_VALUES:
        return False
    raise ValueError(
        "TASK_CONTAINER_CLEANUP_IMAGES must be one of 1/true/yes/on or 0/false/no/off"
    )


def _cleanup_task_images(
    *,
    instance_id: str,
    source_image: str | None,
    fixed_image: str | None,
    keep_source_image: str | None,
    container_executable: str | None,
    run_dir: Path | None = None,
) -> None:
    """Best-effort cleanup that keeps only the current/next-image budget.

    Image removal is disabled by default so benchmark runs can reuse pre-pulled
    task images across smoke tests and sweeps. Set
    ``TASK_CONTAINER_CLEANUP_IMAGES=1`` to enable removal. When enabled, set env
    ``KEEP_IMAGES_ABOVE_GB`` (e.g. ``30``) to skip image removal when free disk
    exceeds the threshold.
    """
    if not source_image and not fixed_image:
        return
    if not _task_image_cleanup_enabled():
        logger.info(
            "cleanup %s skipped: TASK_CONTAINER_CLEANUP_IMAGES is not enabled",
            instance_id,
        )
        return
    if container_executable is None:
        raise ValueError("container_executable is required for image cleanup")

    keep_gb_str = os.environ.get("KEEP_IMAGES_ABOVE_GB", "")
    if keep_gb_str and run_dir is not None:
        keep_gb = float(keep_gb_str)
        try:
            free_gb = shutil.disk_usage(run_dir).free / (1024**3)
        except OSError:
            free_gb = None
        if free_gb is not None and free_gb > keep_gb:
            logger.info(
                "cleanup %s skipped: %.1f GB free > %.1f GB threshold",
                instance_id,
                free_gb,
                keep_gb,
            )
            return

    removed_any = False
    try:
        if fixed_image and fixed_image != source_image:
            removed_fixed = remove_image(
                fixed_image,
                container_executable=container_executable,
            )
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
            removed_source = remove_image(
                source_image,
                container_executable=container_executable,
            )
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
        prune_dangling_images(
            container_executable=container_executable,
        )
    except Exception as exc:
        logger.warning("cleanup %s prune failed: %s", instance_id, exc)


async def _run_scaffold_tasks(
    *,
    benchmark: "Benchmark",
    tasks: list[dict[str, Any]],
    run_dir: Path,
    model: str,
    scaffold: str,
    container_executable: str | None,
    prompt_template: str | None,
    min_free_disk_gb: float,
    inner_factory,
    concurrency: int = 1,
) -> Path:
    """Iterate over tasks, wrapping each in ``run_attempt``."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(run_dir)
    if completed:
        logger.info("Resuming: %d tasks already terminal", len(completed))

    results: list[CollectedTaskResult] = []
    total = len(tasks)
    prefetched_source_image: str | None = None
    prefetch_future: Future[None] | None = None
    resolved_prompt_template = _resolve_prompt_template(
        benchmark=benchmark,
        prompt_template=prompt_template,
    )

    if concurrency > 1:
        next_attempt_by_instance: dict[str, int] = {}
        scheduled: list[tuple[int, dict[str, Any], int]] = []
        for i, task in enumerate(tasks):
            instance_id = task["instance_id"]
            if instance_id in completed:
                logger.info(
                    "[%d/%d] SKIP %s (already terminal)", i + 1, total, instance_id
                )
                continue
            attempt = next_attempt_by_instance.get(instance_id)
            if attempt is None:
                attempt = next_attempt_number(run_dir, instance_id)
            next_attempt_by_instance[instance_id] = attempt + 1
            scheduled.append((i, task, attempt))

        semaphore = asyncio.Semaphore(concurrency)

        async def run_scheduled(
            index: int,
            task: dict[str, Any],
            attempt: int,
        ) -> tuple[int, CollectedTaskResult, str | None, str | None]:
            async with semaphore:
                instance_id = task["instance_id"]
                logger.info(
                    "[%d/%d] START %s (%s)", index + 1, total, instance_id, scaffold
                )
                t0 = time.monotonic()
                source_image = _task_source_image(benchmark, task)
                attempt_ctx = AttemptContext(
                    run_dir=run_dir,
                    instance_id=instance_id,
                    attempt=attempt,
                    task=task,
                    model=model,
                    scaffold=scaffold,
                    source_image=source_image,
                    prompt_template=resolved_prompt_template,
                    agent_runtime_mode=benchmark.runtime_mode_for(scaffold),
                    execution_environment=benchmark.execution_environment,
                )
                try:
                    _ensure_task_source_ready(
                        instance_id=instance_id,
                        source_image=source_image,
                        prefetched_source_image=None,
                        prefetch_future=None,
                        container_executable=container_executable,
                    )
                    run_attempt_kwargs: dict[str, Any] = {
                        "inner": inner_factory(task),
                        "min_free_disk_gb": min_free_disk_gb,
                        "container_executable": container_executable,
                    }

                    def run_attempt_sync() -> AttemptResult:
                        return asyncio.run(
                            run_attempt(attempt_ctx, **run_attempt_kwargs)
                        )

                    result = await asyncio.to_thread(run_attempt_sync)
                except Exception as exc:
                    logger.exception("FAILED %s", instance_id)
                    collected = CollectedTaskResult(
                        instance_id=instance_id,
                        attempt_dir=attempt_ctx.attempt_dir,
                        success=False,
                        model_patch="",
                        exit_status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        elapsed_s=time.monotonic() - t0,
                    )
                else:
                    collected = CollectedTaskResult(
                        instance_id=instance_id,
                        attempt_dir=attempt_ctx.attempt_dir,
                        success=result.success,
                        model_patch=result.model_patch,
                        exit_status=result.exit_status,
                        error=result.error,
                        elapsed_s=time.monotonic() - t0,
                        n_iterations=result.n_iterations,
                    )
                    logger.info(
                        "[%d/%d] DONE %s success=%s elapsed=%.1fs",
                        index + 1,
                        total,
                        instance_id,
                        collected.success,
                        collected.elapsed_s,
                    )
                return index, collected, source_image, attempt_ctx.fixed_image

        task_results = await asyncio.gather(
            *(run_scheduled(index, task, attempt) for index, task, attempt in scheduled)
        )
        for _, collected, _, _ in sorted(task_results, key=lambda item: item[0]):
            results.append(collected)
        for _, collected, source_image, fixed_image in task_results:
            _cleanup_task_images(
                instance_id=collected.instance_id,
                source_image=source_image,
                fixed_image=fixed_image,
                keep_source_image=None,
                container_executable=container_executable,
                run_dir=run_dir,
            )
        write_results_jsonl(results, run_dir / "results.jsonl")
        logger.info("Results written to %s", run_dir / "results.jsonl")
        return run_dir

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="image-prefetch"
    ) as executor:
        for i, task in enumerate(tasks):
            instance_id = task["instance_id"]
            if instance_id in completed:
                logger.info(
                    "[%d/%d] SKIP %s (already terminal)", i + 1, total, instance_id
                )
                continue

            logger.info("[%d/%d] START %s (%s)", i + 1, total, instance_id, scaffold)
            t0 = time.monotonic()
            source_image = _task_source_image(benchmark, task)
            next_source_image = _next_pending_source_image(
                benchmark,
                tasks,
                current_index=i,
                completed=completed,
            )

            attempt_ctx = AttemptContext(
                run_dir=run_dir,
                instance_id=instance_id,
                attempt=next_attempt_number(run_dir, instance_id),
                task=task,
                model=model,
                scaffold=scaffold,
                source_image=source_image,
                prompt_template=resolved_prompt_template,
                agent_runtime_mode=benchmark.runtime_mode_for(scaffold),
                execution_environment=benchmark.execution_environment,
            )

            _inner = inner_factory(task)

            try:
                _ensure_task_source_ready(
                    instance_id=instance_id,
                    source_image=source_image,
                    prefetched_source_image=prefetched_source_image,
                    prefetch_future=prefetch_future,
                    container_executable=container_executable,
                )
                prefetched_source_image = None
                prefetch_future = None

                if (
                    next_source_image
                    and next_source_image != source_image
                    and container_executable is not None
                ):
                    logger.info(
                        "prefetch start for next task after %s image=%s",
                        instance_id,
                        next_source_image,
                    )
                    prefetched_source_image = next_source_image
                    prefetch_future = executor.submit(
                        ensure_source_image,
                        next_source_image,
                        container_executable=container_executable,
                    )

                run_attempt_kwargs: dict[str, Any] = {
                    "inner": _inner,
                    "min_free_disk_gb": min_free_disk_gb,
                    "container_executable": container_executable,
                }
                result = await run_attempt(attempt_ctx, **run_attempt_kwargs)
            except Exception as exc:
                logger.exception("FAILED %s", instance_id)
                results.append(
                    CollectedTaskResult(
                        instance_id=instance_id,
                        attempt_dir=attempt_ctx.attempt_dir,
                        success=False,
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
                        success=result.success,
                        model_patch=result.model_patch,
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
                    container_executable=container_executable,
                    run_dir=run_dir,
                )

    write_results_jsonl(results, run_dir / "results.jsonl")
    logger.info("Results written to %s", run_dir / "results.jsonl")
    return run_dir


async def collect_traces(
    *,
    scaffold: str,
    api_base: str,
    api_key: str,
    model: str,
    benchmark: "Benchmark",
    provider_name: str | None = None,
    env_key: str | None = None,
    max_iterations: int = 100,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    sample: int | None = None,
    skip: int = 0,
    concurrency: int = 1,
    instance_ids: list[str] | None = None,
    run_id: str | None = None,
    max_context_tokens: int = 256_000,
    container_executable: str | None = None,
    mcp_config: str | None = None,
    prompt_template: str | None = None,
    min_free_disk_gb: float = 30.0,
) -> Path:
    """Collect traces for any scaffold supported by the benchmark plugin."""
    benchmark.validate_scaffold_support(scaffold)
    execution_environment = benchmark.execution_environment
    if execution_environment not in {"container", "host"}:
        raise ValueError(
            f"Unsupported benchmark.execution_environment: {execution_environment!r}"
        )

    runtime_mode = benchmark.runtime_mode_for(scaffold)
    if runtime_mode not in {"host_controller", "task_container_agent"}:
        raise NotImplementedError(
            f"Unsupported benchmark.runtime_mode_for({scaffold!r}): {runtime_mode!r}"
        )
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if (
        execution_environment == "container" or runtime_mode == "task_container_agent"
    ) and container_executable is None:
        raise ValueError("--container required for container-mode benchmarks")

    run_dir = Path(run_id) if run_id else build_run_dir(benchmark, model)
    generation_config = _generation_config(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    model_backend = _prepare_collect_model_backend(
        model=model,
        api_base=api_base,
        api_key=api_key,
        provider_name=provider_name,
        generation_config=generation_config,
    )
    runner = None
    if runtime_mode == "host_controller":
        runner = benchmark.build_runner(
            scaffold=scaffold,
            provider=model_backend.provider,
            workspace_base=run_dir / "_workspace_base",
            max_iterations=max_iterations,
            context_window_tokens=max_context_tokens,
            model=model,
            provider_name=model_backend.provider_name,
            env_key=env_key,
            api_base=model_backend.api_base,
            api_key=model_backend.api_key,
            mcp_config=mcp_config,
            mcp_servers=load_mcp_servers(mcp_config),
            generation_config=generation_config,
        )

    tasks = _select_tasks(
        benchmark.load_tasks(),
        instance_ids=instance_ids,
        sample=sample,
        skip=skip,
    )

    def make_inner(task: dict[str, Any]):
        async def inner(ctx: AttemptContext) -> AttemptResult:
            if ctx.agent_runtime_mode == "task_container_agent":
                if scaffold != "openclaw":
                    raise NotImplementedError(
                        "task-container collection currently supports "
                        f"scaffold='openclaw', got {scaffold!r}"
                    )
                if container_executable is None:
                    raise ValueError(
                        "container_executable is required for task-container runs"
                    )
                return await _run_openclaw_in_task_container(
                    ctx=ctx,
                    task=task,
                    benchmark=benchmark,
                    provider_name=model_backend.provider_name,
                    api_base=model_backend.api_base,
                    api_key=model_backend.api_key,
                    model=model,
                    max_iterations=max_iterations,
                    generation_config=generation_config,
                    max_context_tokens=max_context_tokens,
                    mcp_config=mcp_config,
                    container_executable=container_executable,
                    run_config_overrides=model_backend.trace_run_config,
                )

            assert runner is not None
            ctx.agent_start_time = datetime.now(tz=timezone.utc)
            try:
                result = await runner.run_task(
                    task,
                    attempt_ctx=ctx,
                    prompt_template=ctx.prompt_template,
                )
            finally:
                ctx.agent_end_time = datetime.now(tz=timezone.utc)
            if not isinstance(result, AttemptResult):
                raise TypeError(
                    "benchmark runner returned "
                    f"{type(result).__name__}, expected AttemptResult"
                )
            if model_backend.trace_run_config and result.trace_path is not None:
                _stamp_trace_run_config(
                    result.trace_path,
                    model_backend.trace_run_config,
                )
            return result

        return inner

    return await _run_scaffold_tasks(
        benchmark=benchmark,
        tasks=tasks,
        run_dir=run_dir,
        model=model,
        scaffold=scaffold,
        container_executable=container_executable,
        prompt_template=prompt_template,
        min_free_disk_gb=min_free_disk_gb,
        inner_factory=make_inner,
        concurrency=concurrency,
    )


def _set_run_config(merged: dict[str, Any], key: str, value: Any) -> None:
    run_config = merged.get("run_config") or {}
    run_config[key] = value
    merged["run_config"] = run_config


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
    run_config_overrides: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
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

    execution_environment = benchmark.execution_environment
    merged: dict[str, Any] = {
        "scaffold": "openclaw",
        "mode": "collect",
        "scaffold_capabilities": {"unknown": True},
    }
    if source_metadata is not None:
        merged.update(source_metadata)
    merged.setdefault("benchmark", benchmark.config.slug)
    if benchmark.config.harness_split is not None:
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
    merged["execution_environment"] = execution_environment
    if mcp_config_label is not None:
        _set_run_config(merged, "mcp_config", mcp_config_label)
    if run_config_overrides:
        for key, value in run_config_overrides.items():
            _set_run_config(merged, key, value)
    if generation_config:
        _set_run_config(merged, "generation", dict(generation_config))

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dst.parent,
            prefix=f".{dst.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
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
        os.replace(tmp_path, dst)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _resolve_prompt_template(
    *,
    benchmark: "Benchmark",
    prompt_template: str | None,
) -> str:
    return prompt_template or benchmark.config.default_prompt_template


def _stamp_trace_run_config(trace_path: Path, values: dict[str, Any]) -> None:
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if not replaced and rec.get("type") == "trace_metadata":
            run_config = dict(rec.get("run_config") or {})
            run_config.update(values)
            rec["run_config"] = run_config
            replaced = True
        out.append(json.dumps(rec, ensure_ascii=False))
    trace_path.write_text("\n".join(out) + "\n", encoding="utf-8")


async def _run_openclaw_in_task_container(
    *,
    ctx: AttemptContext,
    task: dict[str, Any],
    benchmark: "Benchmark",
    provider_name: str | None,
    api_base: str,
    api_key: str,
    model: str,
    container_executable: str,
    max_iterations: int,
    generation_config: dict[str, Any] | None,
    max_context_tokens: int,
    mcp_config: str | None,
    run_config_overrides: dict[str, Any] | None = None,
) -> AttemptResult:
    fixed_image = ctx.fixed_image or task.get("image_name") or ""
    if not fixed_image:
        raise RuntimeError(f"Task {ctx.instance_id!r} has no image_name")

    runtime_dir = ctx.attempt_dir.resolve() / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    proof = None
    runtime = None
    runtime_proof = None
    exec_config = resolve_task_container_exec_config(
        attempt_dir=ctx.attempt_dir,
        image=fixed_image,
        container_executable=container_executable,
    )
    bootstrap_userbase_bin: str | None = None
    if exec_config.bootstrap_site_dir is not None:
        userbase_bin = exec_config.bootstrap_site_dir.parent / ".pyuserbase" / "bin"
        bootstrap_userbase_bin = str(userbase_bin)
    container_id = start_task_container(
        fixed_image,
        executable=container_executable,
        extra_args=list(exec_config.start_extra_args),
        bootstrap_userbase_bin=bootstrap_userbase_bin,
    )
    ctx.mark_container_ready(container_id)
    try:
        apt_mirror = configure_task_container_apt_mirror(
            container_id,
            executable=container_executable,
        )
        if apt_mirror is not None:
            logger.info("task-container apt mirror: %s", apt_mirror["stdout"])
        exec_config = resolve_running_container_exec_config(
            container_id=container_id,
            exec_config=exec_config,
            container_executable=container_executable,
        )
        preflight_imports = [
            "trace_collect.runtime.entrypoint",
            "agents.openclaw.eval.runner",
            "harness.trace_logger",
        ]
        bootstrap_requirements: tuple[str, ...] = ()
        if mcp_config not in {None, "none"}:
            preflight_imports.append("agents.openclaw.tools.mcp")
            bootstrap_requirements = OPENCLAW_MCP_RUNTIME_REQUIREMENTS
        exec_config = bootstrap_task_container_python(
            container_id=container_id,
            exec_config=exec_config,
            extra_requirements=bootstrap_requirements,
            container_executable=container_executable,
        )
        proof = preflight_task_container_runtime(
            container_id=container_id,
            attempt_dir=ctx.attempt_dir,
            imports=preflight_imports,
            runtime=exec_config.runtime,
            pythonpath=exec_config.pythonpath,
            container_executable=container_executable,
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        exec_path_append = ""
        if exec_config.bootstrap_site_dir is not None:
            exec_path_append = ":".join(
                [
                    str(exec_config.bootstrap_site_dir.parent / ".pyuserbase" / "bin"),
                    str(exec_config.bootstrap_site_dir / "bin"),
                ]
            )
        ctx.agent_start_time = datetime.now(tz=timezone.utc)
        try:
            runtime = run_task_container_agent(
                container_id=container_id,
                timeout=(max_iterations * 120) + 300,
                runtime=exec_config.runtime,
                pythonpath=exec_config.pythonpath,
                request={
                    "kind": "run_openclaw",
                    "scaffold": "openclaw",
                    "result_path": str(runtime_dir / "run.result.json"),
                    "container_id": container_id,
                    "benchmark": benchmark.config.slug,
                    "provider_name": provider_name,
                    "api_base": api_base,
                    "api_key": api_key,
                    "model": model,
                    "max_iterations": max_iterations,
                    "generation_config": generation_config or {},
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
                    "workspace_dir": str(
                        runtime_dir / "workspace_base" / ctx.instance_id
                    ),
                    "tool_workspace": "/testbed",
                    "exec_path_append": exec_path_append,
                    "bootstrap_userbase": (
                        str(exec_config.bootstrap_site_dir.parent / ".pyuserbase")
                        if exec_config.bootstrap_site_dir is not None
                        else None
                    ),
                    "exec_working_dir": "/testbed",
                    "trace_file": str((ctx.attempt_dir / "trace.jsonl").resolve()),
                    "raw_stdout_path": str(stdout_path),
                    "raw_stderr_path": str(stderr_path),
                    "container_executable": container_executable,
                },
                container_executable=container_executable,
            )
        finally:
            ctx.agent_end_time = datetime.now(tz=timezone.utc)
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
            mcp_config_label=mcp_config_label(mcp_config),
            prompt_template=ctx.prompt_template,
            agent_runtime_mode=ctx.agent_runtime_mode,
            runtime_proof=runtime_proof,
            run_config_overrides=run_config_overrides,
            generation_config=generation_config,
        )
    finally:
        container_logs = stop_task_container(
            container_id,
            executable=container_executable,
        )
        ctx.container_stdout = "\n".join(
            part
            for part in [
                stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
                stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
                container_logs,
            ]
            if part
        )
    assert runtime is not None
    assert runtime_proof is not None
    return AttemptResult(
        success=runtime.success,
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
