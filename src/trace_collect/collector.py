"""SWE-Bench trace collector using an external LLM API + Docker sandbox."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.trace_logger import TraceLogger
from trace_collect.swebench_harness import (
    build_eval_run_id,
    is_swebench_available,
    run_official_evaluation,
)

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark
    from agents.miniswe import MiniSWECodeAgent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CollectedTaskResult:
    """Durable per-task result emitted by the trace collector."""

    instance_id: str
    trace_file: Path
    success: bool
    success_basis: str
    patch_generated: bool
    model_patch: str
    exit_status: str | None = None
    error: str | None = None
    n_steps: int | None = None
    elapsed_s: float | None = None
    prepare_ms: float | None = None
    total_llm_ms: float | None = None
    total_tool_ms: float | None = None
    total_tokens: int | None = None
    official_resolved: bool | None = None
    evaluation_run_id: str | None = None
    evaluation_report_path: str | None = None
    evaluation_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "trace_file": str(self.trace_file),
            "success": self.success,
            "success_basis": self.success_basis,
            "patch_generated": self.patch_generated,
            "model_patch": self.model_patch,
            "exit_status": self.exit_status,
            "error": self.error,
            "n_steps": self.n_steps,
            "elapsed_s": self.elapsed_s,
            "prepare_ms": self.prepare_ms,
            "total_llm_ms": self.total_llm_ms,
            "total_tool_ms": self.total_tool_ms,
            "total_tokens": self.total_tokens,
            "official_resolved": self.official_resolved,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluation_report_path": self.evaluation_report_path,
            "evaluation_report": self.evaluation_report,
            "resolved": bool(self.official_resolved),
        }

    def to_prediction(self, model_name: str) -> dict[str, Any] | None:
        if not self.model_patch:
            return None
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": model_name,
            "model_patch": self.model_patch,
        }


def load_tasks(task_source: str | Path) -> list[dict[str, Any]]:
    """Load tasks from a JSON file."""
    path = Path(task_source)
    return json.loads(path.read_text(encoding="utf-8"))


def load_completed_ids(run_dir: Path) -> set[str]:
    """Scan a run directory for already-completed agent IDs.

    Each task has its own JSONL file: {run_dir}/{agent_id}.jsonl.
    A task is complete if its file contains a summary record.
    """
    completed: set[str] = set()
    if not run_dir.exists():
        return completed
    for trace_file in run_dir.glob("*.jsonl"):
        agent_id = trace_file.stem
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "summary":
                        completed.add(agent_id)
                        break
                except json.JSONDecodeError:
                    continue
    return completed


def build_run_dir(benchmark: "Benchmark", model: str) -> Path:
    """Build run directory from benchmark plugin config: trace_root/model/ts/."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    return benchmark.config.trace_root / safe_model / ts


def load_existing_results(results_path: Path) -> dict[str, CollectedTaskResult]:
    """Load previously written per-task results for resume support."""
    if not results_path.exists():
        return {}
    loaded: dict[str, CollectedTaskResult] = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = payload.get("instance_id")
            trace_file = payload.get("trace_file")
            if not instance_id or not trace_file:
                continue
            loaded[instance_id] = CollectedTaskResult(
                instance_id=instance_id,
                trace_file=Path(trace_file),
                success=bool(payload.get("success")),
                success_basis=str(payload.get("success_basis") or "patch_generated"),
                patch_generated=bool(payload.get("patch_generated")),
                model_patch=payload.get("model_patch", "") or "",
                exit_status=payload.get("exit_status"),
                error=payload.get("error"),
                n_steps=payload.get("n_steps"),
                elapsed_s=payload.get("elapsed_s"),
                prepare_ms=payload.get("prepare_ms"),
                total_llm_ms=payload.get("total_llm_ms"),
                total_tool_ms=payload.get("total_tool_ms"),
                total_tokens=payload.get("total_tokens"),
                official_resolved=payload.get("official_resolved"),
                evaluation_run_id=payload.get("evaluation_run_id"),
                evaluation_report_path=payload.get("evaluation_report_path"),
                evaluation_report=payload.get("evaluation_report"),
            )
    return loaded


def _build_result_record(
    *,
    agent: "MiniSWECodeAgent",
    summary: dict[str, Any],
    trace_file: Path,
) -> CollectedTaskResult:
    model_patch = (agent.task_submission or "").strip()
    return CollectedTaskResult(
        instance_id=agent.task_id or trace_file.stem,
        trace_file=trace_file,
        success=bool(summary.get("success")),
        success_basis="patch_generated",
        patch_generated=bool(model_patch),
        model_patch=model_patch,
        exit_status=agent.task_exit_status,
        error=summary.get("error") or agent.task_error,
        n_steps=summary.get("n_steps"),
        elapsed_s=summary.get("elapsed_s"),
        prepare_ms=summary.get("prepare_ms"),
        total_llm_ms=summary.get("total_llm_ms"),
        total_tool_ms=summary.get("total_tool_ms"),
        total_tokens=summary.get("total_tokens"),
    )


def _write_results(results: list[CollectedTaskResult], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def _write_predictions(
    results: list[CollectedTaskResult],
    *,
    model_name: str,
    predictions_path: Path,
) -> int:
    predictions = {}
    for result in results:
        pred = result.to_prediction(model_name)
        if pred is not None:
            predictions[result.instance_id] = pred
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    return len(predictions)


def _rewrite_trace_summary(trace_file: Path, result: CollectedTaskResult) -> None:
    """Update the per-task summary row with post-collection evaluation fields."""
    rewritten: list[str] = []
    with open(trace_file, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rewritten.append(raw_line.rstrip("\n"))
                continue
            if (
                record.get("type") == "summary"
                and record.get("agent_id") == result.instance_id
            ):
                record["success"] = result.success
                record["success_basis"] = result.success_basis
                record["patch_generated"] = result.patch_generated
                record["official_resolved"] = result.official_resolved
                record["resolved"] = bool(result.official_resolved)
                record["exit_status"] = result.exit_status
                record["error"] = result.error
                record["evaluation_run_id"] = result.evaluation_run_id
                record["evaluation_report_path"] = result.evaluation_report_path
            rewritten.append(json.dumps(record, ensure_ascii=False))
    trace_file.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _ordered_results(
    *,
    task_ids: list[str],
    results_by_id: dict[str, CollectedTaskResult],
) -> list[CollectedTaskResult]:
    ordered = [
        results_by_id[task_id] for task_id in task_ids if task_id in results_by_id
    ]
    extra_ids = sorted(set(results_by_id) - set(task_ids))
    ordered.extend(results_by_id[task_id] for task_id in extra_ids)
    return ordered


async def collect_traces(
    *,
    api_base: str,
    api_key: str,
    model: str,
    benchmark: "Benchmark",
    task_source: str | Path | None = None,
    max_steps: int = 60,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    sample: int | None = None,
    instance_ids: list[str] | None = None,
    scaffold: str = "mini-swe-agent",
    run_id: str | None = None,
    max_context_tokens: int = 256_000,
    evaluate: bool = False,
    harness_max_workers: int = 1,
    harness_timeout: int = 1800,
    harness_run_id: str | None = None,
    harness_report_dir: str | Path | None = None,
) -> Path:
    """Collect agent traces for a benchmark.

    The benchmark plugin is the single source of truth for dataset name,
    split, namespace, repos_root, and trace_root. Runtime overrides
    (max_steps, timeouts, sample size) stay on the signature.

    Args:
        api_base: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model: Model name (e.g. "qwen-plus-latest").
        benchmark: Benchmark plugin instance — drives dataset, paths, harness.
        task_source: Optional override for the tasks JSON file. Defaults to
            ``benchmark.config.data_root / "tasks.json"``.
        max_steps: Maximum agent steps per task.
        command_timeout_s: Timeout per bash command.
        task_timeout_s: Timeout per task overall.
        sample: If set, only run the first N tasks.
        run_id: Explicit run ID for resuming an interrupted run. If None,
            a new timestamped ID is generated.
        max_context_tokens: Token budget for sliding window context management.

    Returns:
        Path to the run directory containing per-task JSONL files.
    """
    # Resolve defaults from the benchmark
    if task_source is None:
        task_source = benchmark.config.data_root / "tasks.json"
    repos_root = benchmark.config.repos_root
    if repos_root is None:
        raise ValueError(
            f"Benchmark {benchmark.config.slug!r} has no repos_root configured; "
            f"cannot run code-patching scaffolds without local repos."
        )

    tasks = load_tasks(task_source)
    if instance_ids is not None:
        id_set = set(instance_ids)
        tasks = [t for t in tasks if t["instance_id"] in id_set]
        if not tasks:
            raise ValueError(f"No tasks matched instance_ids: {instance_ids}")
    if sample is not None:
        tasks = tasks[:sample]

    if scaffold == "openclaw":
        return await _collect_openclaw(
            api_base=api_base,
            api_key=api_key,
            model=model,
            tasks=tasks,
            benchmark=benchmark,
            max_steps=max_steps,
            run_id=run_id,
            max_context_tokens=max_context_tokens,
            evaluate=evaluate,
            harness_max_workers=harness_max_workers,
            harness_timeout=harness_timeout,
            harness_run_id=harness_run_id,
            harness_report_dir=harness_report_dir,
        )

    from agents.miniswe import MiniSWECodeAgent

    if run_id is not None:
        # Resume: run_id is the full path to an existing run directory
        run_dir = Path(run_id)
    else:
        run_dir = build_run_dir(benchmark, model)
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    predictions_path = run_dir / "preds.json"
    results_by_id = load_existing_results(results_path)
    completed = load_completed_ids(run_dir)
    if completed:
        logger.info("Resuming: %d tasks already completed", len(completed))

    total = len(tasks)
    succeeded = 0
    failed = 0

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        if instance_id in completed:
            logger.info(
                "[%d/%d] SKIP %s (already completed)", i + 1, total, instance_id
            )
            continue

        logger.info("[%d/%d] START %s", i + 1, total, instance_id)
        t0 = time.monotonic()

        # One TraceLogger per task → one JSONL file per task
        trace_logger = TraceLogger(run_dir, instance_id)
        trace_logger.log_metadata(
            scaffold="mini-swe-agent",
            benchmark=benchmark.config.slug,
            benchmark_split=benchmark.config.harness_split,
            model=model,
            api_base=api_base,
            max_steps=max_steps,
            instance_id=instance_id,
            scaffold_capabilities={
                "tools": ["bash"],
                "memory": False,
                "skills": False,
                "file_ops": "bash_only",
            },
        )

        agent = MiniSWECodeAgent(
            agent_id=instance_id,
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_steps=max_steps,
            command_timeout_s=command_timeout_s,
            task_timeout_s=task_timeout_s,
            repos_root=str(repos_root),
            max_context_tokens=max_context_tokens,
        )
        agent._trace_logger = trace_logger
        agent.run_metadata = {"model": model, "api_provider": "dashscope"}

        prepare_ms = 0.0
        prepare_t0 = time.monotonic()
        try:
            await agent.prepare(task)
            prepare_ms = (time.monotonic() - prepare_t0) * 1000
            agent.run_metadata["prepare_ms"] = prepare_ms

            success = await agent.run(task)
        except Exception as exc:
            logger.exception("FAILED %s", instance_id)
            failed += 1
            elapsed = time.monotonic() - t0
            if prepare_ms == 0.0:
                prepare_ms = (time.monotonic() - prepare_t0) * 1000
            error_summary = agent.summary()
            error_summary["elapsed_s"] = elapsed
            error_summary["prepare_ms"] = prepare_ms
            error_summary["error"] = str(exc)
            error_summary["error_type"] = type(exc).__name__
            trace_logger.log_summary(agent.agent_id, error_summary)
            trace_logger.close()
            task_result = _build_result_record(
                agent=agent,
                summary=error_summary,
                trace_file=trace_logger.path,
            )
            _rewrite_trace_summary(trace_logger.path, task_result)
            results_by_id[instance_id] = task_result
            continue

        elapsed = time.monotonic() - t0

        summary = agent.summary()
        summary["elapsed_s"] = elapsed
        summary["prepare_ms"] = prepare_ms
        trace_logger.log_summary(agent.agent_id, summary)
        trace_logger.close()
        task_result = _build_result_record(
            agent=agent,
            summary=summary,
            trace_file=trace_logger.path,
        )
        _rewrite_trace_summary(trace_logger.path, task_result)
        results_by_id[instance_id] = task_result

        if success:
            succeeded += 1
        else:
            failed += 1

        steps = len(agent.trace)
        logger.info(
            "[%d/%d] DONE %s success=%s steps=%d elapsed=%.1fs",
            i + 1,
            total,
            instance_id,
            success,
            steps,
            elapsed,
        )

    logger.info(
        "Collection complete: %d/%d succeeded, %d failed, traces -> %s",
        succeeded,
        total,
        failed,
        run_dir,
    )
    task_ids = [task["instance_id"] for task in tasks]
    ordered_results = _ordered_results(task_ids=task_ids, results_by_id=results_by_id)

    prediction_count = _write_predictions(
        ordered_results,
        model_name=model,
        predictions_path=predictions_path,
    )
    logger.info(
        "Predictions written to %s (%d patches)", predictions_path, prediction_count
    )

    if evaluate:
        if not is_swebench_available():
            raise RuntimeError(
                "Official SWE-bench evaluation requested, but the 'swebench' package is not installed."
            )
        if prediction_count == 0:
            logger.warning(
                "Official evaluation requested, but no patch predictions were produced. Marking all tasks unresolved."
            )
            for result in ordered_results:
                result.success_basis = "official_resolved"
                result.official_resolved = False
                result.success = False
                _rewrite_trace_summary(result.trace_file, result)
        else:
            evaluation_run_id = harness_run_id or build_eval_run_id(run_dir.name)
            evaluation_report_path = (
                Path(harness_report_dir).expanduser().resolve()
                if harness_report_dir is not None
                else (run_dir / "swebench_eval").resolve()
            )
            harness_args = benchmark.build_harness_args(
                predictions_path=predictions_path,
                run_id=evaluation_run_id,
                max_workers=harness_max_workers,
                timeout=harness_timeout,
                report_dir=evaluation_report_path,
            )
            evaluation = run_official_evaluation(
                instance_ids=task_ids,
                **harness_args,
            )
            if evaluation.returncode != 0:
                raise RuntimeError(
                    "Official SWE-bench harness failed "
                    f"(exit {evaluation.returncode}): {(evaluation.stderr or evaluation.stdout)[-4000:]}"
                )
            for result in ordered_results:
                report = evaluation.instance_reports.get(result.instance_id)
                report_payload = (
                    report.get(result.instance_id)
                    if isinstance(report, dict) and result.instance_id in report
                    else report
                )
                result.success_basis = "official_resolved"
                result.official_resolved = bool(
                    isinstance(report_payload, dict) and report_payload.get("resolved")
                )
                result.success = bool(result.official_resolved)
                result.evaluation_run_id = evaluation.run_id
                report_path = evaluation.instance_report_paths.get(result.instance_id)
                result.evaluation_report_path = (
                    str(report_path) if report_path else None
                )
                result.evaluation_report = report
                _rewrite_trace_summary(result.trace_file, result)
            if evaluation.report_path is not None:
                logger.info(
                    "Official SWE-bench report written to %s", evaluation.report_path
                )

    _write_results(ordered_results, results_path)
    logger.info("Results written to %s", results_path)
    return run_dir


async def _collect_openclaw(
    *,
    api_base: str,
    api_key: str,
    model: str,
    tasks: list[dict[str, Any]],
    benchmark: "Benchmark",
    max_steps: int = 80,
    run_id: str | None = None,
    max_context_tokens: int = 256_000,
    evaluate: bool = False,
    harness_max_workers: int = 1,
    harness_timeout: int = 1800,
    harness_run_id: str | None = None,
    harness_report_dir: str | Path | None = None,
) -> Path:
    """Collect traces using the OpenClaw (nanobot) scaffold via SWEBenchRunner."""
    from agents.openclaw.unified_provider import UnifiedProvider
    from agents.openclaw.eval.runner import SWEBenchRunner
    from agents.openclaw.eval.types import EvalTask

    if run_id is not None:
        run_dir = Path(run_id)
    else:
        run_dir = build_run_dir(benchmark, model)
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    completed = load_completed_ids(run_dir)
    if completed:
        logger.info("Resuming: %d tasks already completed", len(completed))

    provider = UnifiedProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model,
    )
    runner = SWEBenchRunner(
        provider=provider,
        workspace_base=run_dir / "_workspaces",
        max_iterations=max_steps,
        context_window_tokens=max_context_tokens,
        model=model,
        repos_root=benchmark.config.repos_root,
        benchmark=benchmark,
    )

    total = len(tasks)
    results: list[CollectedTaskResult] = []

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        if instance_id in completed:
            logger.info(
                "[%d/%d] SKIP %s (already completed)", i + 1, total, instance_id
            )
            continue

        logger.info("[%d/%d] START %s (openclaw)", i + 1, total, instance_id)

        try:
            eval_task = EvalTask.from_benchmark_instance(
                task, run_dir / "_workspaces", benchmark=benchmark
            )
            eval_result = await runner.run_task(eval_task)
        except Exception as exc:
            logger.exception("FAILED %s (openclaw)", instance_id)
            # Persist any partial trace that was written before the crash
            ws_trace = run_dir / "_workspaces" / instance_id / "trace.jsonl"
            dest_trace = run_dir / f"{instance_id}.jsonl"
            if ws_trace.exists() and not dest_trace.exists():
                _normalize_openclaw_trace(
                    src=ws_trace,
                    dst=dest_trace,
                    benchmark=benchmark,
                    model=model,
                    api_base=api_base,
                    max_steps=max_steps,
                    instance_id=instance_id,
                )
            results.append(
                CollectedTaskResult(
                    instance_id=instance_id,
                    trace_file=str(dest_trace),
                    success=False,
                    success_basis="patch_generated",
                    patch_generated=False,
                    model_patch="",
                    exit_status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        # Read summary from trace file to get aggregate metrics
        trace_summary: dict[str, Any] = {}
        if eval_result.trace_file:
            trace_path = Path(eval_result.trace_file)
            if trace_path.exists():
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("type") == "summary":
                        trace_summary = rec
                        break

        # Map EvalResult -> CollectedTaskResult
        model_patch = eval_result.model_patch or ""
        task_result = CollectedTaskResult(
            instance_id=instance_id,
            trace_file=run_dir / f"{instance_id}.jsonl",
            success=bool(model_patch),
            success_basis="patch_generated",
            patch_generated=bool(model_patch),
            model_patch=model_patch,
            exit_status=eval_result.stop_reason,
            error=eval_result.error,
            elapsed_s=eval_result.run_ms / 1000 if eval_result.run_ms else 0.0,
            n_steps=trace_summary.get("n_steps", 0),
            total_llm_ms=trace_summary.get("total_llm_ms", 0.0),
            total_tool_ms=trace_summary.get("total_tool_ms", 0.0),
            total_tokens=trace_summary.get("total_tokens", 0),
            prepare_ms=eval_result.prepare_ms or 0.0,
        )

        # Normalize trace: copy to benchmark run-dir layout + inject metadata
        if eval_result.trace_file and Path(eval_result.trace_file).exists():
            _normalize_openclaw_trace(
                src=Path(eval_result.trace_file),
                dst=run_dir / f"{instance_id}.jsonl",
                benchmark=benchmark,
                model=model,
                api_base=api_base,
                max_steps=max_steps,
                instance_id=instance_id,
            )

        results.append(task_result)
        logger.info(
            "[%d/%d] DONE %s  steps=%d patch=%s",
            i + 1,
            total,
            instance_id,
            task_result.n_steps,
            task_result.patch_generated,
        )

    # Write results + predictions
    _write_results(results, results_path)
    _write_predictions(
        results, model_name=model, predictions_path=run_dir / "preds.json"
    )
    logger.info("Results written to %s", results_path)

    # Harness evaluation
    if evaluate and is_swebench_available():
        preds_path = run_dir / "preds.json"
        if preds_path.exists():
            harness_args = benchmark.build_harness_args(
                predictions_path=preds_path,
                run_id=harness_run_id or build_eval_run_id(run_dir.name),
                max_workers=harness_max_workers,
                timeout=harness_timeout,
                report_dir=Path(harness_report_dir) if harness_report_dir else None,
            )
            evaluation = run_official_evaluation(**harness_args)
            if evaluation.returncode != 0:
                logger.error(
                    "Official SWE-bench harness failed (exit %d): %s",
                    evaluation.returncode,
                    (evaluation.stderr or evaluation.stdout)[-4000:],
                )
            else:
                for result in results:
                    report = evaluation.instance_reports.get(result.instance_id)
                    report_payload = (
                        report.get(result.instance_id)
                        if isinstance(report, dict) and result.instance_id in report
                        else report
                    )
                    result.success_basis = "official_resolved"
                    result.official_resolved = bool(
                        isinstance(report_payload, dict)
                        and report_payload.get("resolved")
                    )
                    result.success = bool(result.official_resolved)
                    result.evaluation_run_id = evaluation.run_id
                    report_path = evaluation.instance_report_paths.get(
                        result.instance_id
                    )
                    result.evaluation_report_path = (
                        str(report_path) if report_path else None
                    )
                    result.evaluation_report = report
                _write_results(results, results_path)

    return run_dir


def _normalize_openclaw_trace(
    src: Path,
    dst: Path,
    *,
    benchmark: "Benchmark",
    model: str,
    api_base: str,
    max_steps: int,
    instance_id: str,
) -> None:
    """Copy an OpenClaw trace to benchmark run-dir layout with metadata injection."""
    lines = src.read_text(encoding="utf-8").splitlines()
    with open(dst, "w", encoding="utf-8") as f:
        # Inject trace_metadata header. trace_format_version: 5 is stamped
        # at write time — every openclaw trace routed through the collector
        # must carry the version field so downstream TraceData.load() can
        # enforce its strict v5 check. v4 support was dropped during the
        # SWE-rebench plugin refactor (no backfill, no tolerance).
        metadata = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "benchmark": benchmark.config.slug,
            "benchmark_split": benchmark.config.harness_split,
            "model": model,
            "api_base": api_base,
            "max_steps": max_steps,
            "instance_id": instance_id,
            "mode": "collect",
            "scaffold_capabilities": {
                "tools": [
                    "bash",
                    "file_read",
                    "file_write",
                    "file_edit",
                    "list_dir",
                    "web_search",
                    "web_fetch",
                    "send_message",
                ],
                "memory": True,
                "skills": True,
                "file_ops": "structured",
            },
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Skip any existing trace_metadata from nanobot
            if rec.get("type") == "trace_metadata":
                continue
            # Normalize step tool_args: ensure it's a string
            if rec.get("type") == "step" and isinstance(rec.get("tool_args"), dict):
                rec["tool_args"] = json.dumps(rec["tool_args"], ensure_ascii=False)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
