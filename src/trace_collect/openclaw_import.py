"""Import nanobot/OpenClaw results into the benchmark run layout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_import_run_dir(
    output_dir: str | Path, model_name: str, run_id: str | None
) -> Path:
    safe_model = model_name.replace("/", "-").replace(":", "-")
    if run_id is None:
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return Path(output_dir) / "openclaw_import" / safe_model / run_id


def _load_results(results_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not payload.get("instance_id"):
                raise ValueError(f"Missing instance_id in {results_path}")
            if not payload.get("trace_file"):
                raise ValueError(
                    f"Missing trace_file for {payload.get('instance_id')} in {results_path}"
                )
            records.append(payload)
    if not records:
        raise ValueError(f"No records found in {results_path}")
    return records


def _validate_trace(trace_path: Path, *, instance_id: str) -> None:
    saw_step = False
    saw_summary = False
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("agent_id") != instance_id:
                continue
            if record.get("type") == "step":
                saw_step = True
            elif record.get("type") == "summary":
                saw_summary = True
    if not saw_step or not saw_summary:
        raise ValueError(
            f"Trace {trace_path} is not benchmark-compatible for {instance_id}: "
            f"saw_step={saw_step} saw_summary={saw_summary}"
        )


def _build_imported_result(
    *,
    payload: dict[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    official_resolved = payload.get("official_resolved")
    success_basis = (
        "official_resolved" if official_resolved is not None else "patch_generated"
    )
    success = (
        bool(official_resolved)
        if official_resolved is not None
        else bool(payload.get("patch_generated"))
    )
    return {
        "instance_id": payload["instance_id"],
        "trace_file": str(trace_path),
        "success": success,
        "success_basis": success_basis,
        "patch_generated": bool(payload.get("patch_generated")),
        "model_patch": payload.get("model_patch", "") or "",
        "stop_reason": payload.get("stop_reason") or payload.get("exit_status"),
        "error": payload.get("error"),
        "prepare_ms": payload.get("prepare_ms"),
        "run_ms": payload.get("run_ms"),
        "official_resolved": official_resolved,
        "evaluation_run_id": payload.get("evaluation_run_id"),
        "evaluation_report_path": payload.get("evaluation_report_path"),
        "evaluation_report": payload.get("evaluation_report"),
        "resolved": bool(official_resolved),
    }


def _write_results(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _copy_trace_for_import(
    *,
    source_trace: Path,
    target_trace: Path,
    imported_result: dict[str, Any],
    model_name: str = "unknown",
    benchmark: str = "swe-bench-verified",
    benchmark_split: str = "test",
) -> None:
    """Copy a trace while aligning summary success semantics to benchmark results.

    The written trace_metadata record carries ``trace_format_version: 5`` plus
    the ``benchmark`` / ``benchmark_split`` fields required by the strict
    :func:`trace_collect.trace_inspector.TraceData.load` check. Without this,
    any imported trace would fail to load through the v5 reader. Benchmark
    defaults to ``swe-bench-verified`` / ``test`` — the historical use case
    for this importer — and can be overridden when importing traces from a
    different benchmark.
    """
    target_trace.parent.mkdir(parents=True, exist_ok=True)
    benchmark_success = bool(imported_result.get("success"))
    with (
        open(source_trace, encoding="utf-8") as src,
        open(target_trace, "w", encoding="utf-8") as dst,
    ):
        # Inject trace_metadata as the first record. trace_format_version: 5
        # is required by the strict v5 reader introduced during the
        # SWE-rebench refactor (no backfill, no tolerance).
        metadata = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "benchmark": benchmark,
            "benchmark_split": benchmark_split,
            "mode": "import",
            "model": model_name,
            "source_trace": str(source_trace),
            "instance_id": imported_result.get("instance_id", ""),
        }
        dst.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("type") == "step":
                _normalize_step_tool_args(record)
            if record.get("type") == "summary":
                if "success" in record:
                    record["source_success"] = record["success"]
                record["success"] = benchmark_success
                record["success_basis"] = imported_result.get("success_basis")
                record["official_resolved"] = imported_result.get("official_resolved")
                record["patch_generated"] = imported_result.get("patch_generated")
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_step_tool_args(record: dict[str, Any]) -> None:
    tool_args = record.get("tool_args")
    if not isinstance(tool_args, str):
        return
    try:
        json.loads(tool_args)
        return
    except json.JSONDecodeError:
        pass

    raw_response = record.get("raw_response") or {}
    message = (raw_response.get("choices") or [{}])[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return

    function = (tool_calls[0] or {}).get("function") or {}
    tool_name = function.get("name")
    arguments = function.get("arguments")
    if not tool_name or not arguments:
        return

    try:
        parsed_args = json.loads(arguments)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed_args, dict):
        return

    record["tool_args"] = json.dumps({tool_name: parsed_args}, ensure_ascii=False)


def _write_predictions(
    results: list[dict[str, Any]], *, model_name: str, path: Path
) -> int:
    predictions = {}
    for result in results:
        model_patch = result.get("model_patch", "")
        if not model_patch:
            continue
        predictions[result["instance_id"]] = {
            "instance_id": result["instance_id"],
            "model_name_or_path": model_name,
            "model_patch": model_patch,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    return len(predictions)


def import_openclaw_run(
    *,
    results_path: Path,
    output_dir: str | Path,
    model_name: str = "Qwen3.6-Plus",
    run_id: str | None = None,
    benchmark: str = "swe-bench-verified",
    benchmark_split: str = "test",
) -> Path:
    """Copy nanobot results/traces into the benchmark run layout.

    Args:
        benchmark: Benchmark slug stamped into each imported trace's
            ``trace_metadata`` record. Defaults to the historical use case
            (``swe-bench-verified``); pass ``swe-rebench`` (or any other
            registered slug) when importing traces from a different benchmark.
        benchmark_split: Dataset split for the stamped metadata. Defaults to
            the Verified test split.
    """
    records = _load_results(results_path)
    run_dir = _build_import_run_dir(output_dir, model_name, run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    imported_results: list[dict[str, Any]] = []
    for payload in records:
        instance_id = payload["instance_id"]
        source_trace = Path(payload["trace_file"]).expanduser().resolve()
        if not source_trace.exists():
            raise FileNotFoundError(
                f"Trace file not found for {instance_id}: {source_trace}"
            )
        _validate_trace(source_trace, instance_id=instance_id)
        target_trace = run_dir / f"{instance_id}.jsonl"
        imported_result = _build_imported_result(
            payload=payload, trace_path=target_trace
        )
        _copy_trace_for_import(
            source_trace=source_trace,
            target_trace=target_trace,
            imported_result=imported_result,
            model_name=model_name,
            benchmark=benchmark,
            benchmark_split=benchmark_split,
        )
        imported_results.append(imported_result)

    _write_results(imported_results, run_dir / "results.jsonl")
    prediction_count = _write_predictions(
        imported_results,
        model_name=model_name,
        path=run_dir / "preds.json",
    )
    manifest = {
        "source_results": str(results_path.resolve()),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "task_count": len(imported_results),
        "prediction_count": prediction_count,
    }
    (run_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir
