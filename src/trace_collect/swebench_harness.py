"""Helpers for running the official SWE-bench harness from trace collection."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def is_swebench_available() -> bool:
    """Return True when the official SWE-bench package is importable."""
    return importlib.util.find_spec("swebench") is not None


def build_eval_run_id(prefix: str = "trace-collect") -> str:
    """Build a stable UTC timestamped run id for official harness runs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = prefix.replace(" ", "-").replace("/", "-")
    return f"{safe_prefix}-{ts}"


def _load_predictions(predictions_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Predictions file must contain a JSON object: {predictions_path}")
    return payload


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


@dataclass(slots=True)
class HarnessEvaluationResult:
    run_id: str
    report_dir: Path
    report_path: Path | None
    stdout: str
    stderr: str
    returncode: int
    summary_report: dict[str, Any] | None
    instance_reports: dict[str, dict[str, Any]]
    instance_report_paths: dict[str, Path]


def run_official_evaluation(
    *,
    predictions_path: Path,
    dataset_name: str,
    split: str,
    run_id: str,
    max_workers: int = 1,
    timeout: int = 1800,
    instance_ids: list[str] | None = None,
    namespace: str | None = "swebench",
    report_dir: Path | None = None,
    python_executable: str | None = None,
) -> HarnessEvaluationResult:
    """Run the official SWE-bench harness and collect per-instance reports."""
    report_dir = (report_dir or predictions_path.parent).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable or sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path",
        str(predictions_path),
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    if namespace is not None:
        command.extend(["--namespace", namespace])

    completed = subprocess.run(
        command,
        cwd=str(report_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    predictions = _load_predictions(predictions_path)
    if predictions:
        model_name = _safe_model_name(
            next(iter(predictions.values())).get("model_name_or_path", "unknown")
        )
    else:
        model_name = "unknown"

    summary_report_path = report_dir / f"{model_name}.{run_id}.json"
    summary_report = None
    if summary_report_path.exists():
        summary_report = json.loads(summary_report_path.read_text(encoding="utf-8"))

    logs_root = report_dir / "logs" / "run_evaluation" / run_id / model_name
    instance_reports: dict[str, dict[str, Any]] = {}
    instance_report_paths: dict[str, Path] = {}
    for instance_id in predictions:
        report_path = logs_root / instance_id / "report.json"
        if not report_path.exists():
            continue
        instance_report_paths[instance_id] = report_path
        instance_reports[instance_id] = json.loads(report_path.read_text(encoding="utf-8"))

    return HarnessEvaluationResult(
        run_id=run_id,
        report_dir=report_dir,
        report_path=summary_report_path if summary_report_path.exists() else None,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        summary_report=summary_report,
        instance_reports=instance_reports,
        instance_report_paths=instance_report_paths,
    )
