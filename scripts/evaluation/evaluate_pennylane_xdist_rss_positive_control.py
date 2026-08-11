#!/usr/bin/env python3
"""Evaluate the frozen fit-calibrated pytest-xdist RSS positive control."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    Row,
    _row_from_clause,
    command_resource_bucket_label,
)
from scripts.evaluation.evaluate_pennylane_temporal_rss_ceiling import (  # noqa: E402
    _EXPECTED_TASK_IDS,
)
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_RESOURCE_BUCKET_EDGES,
    ClauseResourceKB,
)
from tool_resource_eval.labels import repo_of  # noqa: E402


_PROTOCOL_GIT_SHA = "81b7ff1c1cd715283d5ee944f369985fa60424a6"
_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
_OUTPUT = _ROOT / "analysis/results/pennylane-xdist-rss-positive-control-v1"
_RSS = "sampled_peak_rss_mb"
_HOST_CPUS = 8
_MIN_CALIBRATION = 50
_MIN_CALIBRATION_TASKS = 10
_MIN_CARRIERS = 20
_MIN_CARRIER_TASKS = 5
_MIN_HIGH_RECALL_GAIN = 0.20


def parse_pytest_workers(argv: Sequence[str]) -> int | str | None:
    """Return worker count, ``serial``, ``invalid``, or None for non-pytest."""

    words = tuple(str(value) for value in argv)
    if not words:
        return None
    executable = PurePosixPath(words[0]).name
    if executable == "pytest":
        tail = words[1:]
    elif executable.startswith("python") and words[1:3] == ("-m", "pytest"):
        tail = words[3:]
    else:
        return None
    values: list[str] = []
    index = 0
    while index < len(tail):
        token = tail[index]
        if token == "--":
            break
        if token in {"-n", "--numprocesses"}:
            index += 1
            if index >= len(tail):
                return "invalid"
            values.append(tail[index])
        elif token.startswith("--numprocesses="):
            values.append(token.split("=", 1)[1])
        elif token.startswith("-n") and len(token) > 2:
            values.append(token[2:])
        index += 1
    if not values:
        return "serial"
    parsed = {
        _HOST_CPUS
        if value in {"auto", "logical"}
        else int(value)
        if value.isdecimal() and 1 <= int(value) <= 64
        else -1
        for value in values
    }
    return next(iter(parsed)) if len(parsed) == 1 and -1 not in parsed else "invalid"


def scaled_rss_pmf(values: Sequence[float], workers: int) -> tuple[float, float, float]:
    edges = CANONICAL_RESOURCE_BUCKET_EDGES[_RSS]
    counts = Counter(
        0 if (workers + 1) * value <= edges[0]
        else 1 if (workers + 1) * value <= edges[1]
        else 2
        for value in values
    )
    return tuple(counts[index] / len(values) for index in range(3))


def max_bucket_convolution(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    result = [0.0, 0.0, 0.0]
    for left_index, left_probability in enumerate(left):
        for right_index, right_probability in enumerate(right):
            result[max(left_index, right_index)] += left_probability * right_probability
    return tuple(result)


def _clean_head() -> str:
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ValueError("formal evaluation requires a clean committed checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        cwd=_ROOT,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_tasks() -> tuple[list[Row], dict[str, list[Row]], dict[str, list[CommandRow]]]:
    relative = _SPLIT.relative_to(_ROOT).as_posix()
    frozen = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if _SPLIT.read_bytes() != frozen:
        raise ValueError("frozen PennyLane split changed")
    split = json.loads(frozen)
    replay = {str(item["task_id"]): item for item in split["replay"]}
    selected = [*split["fit"], *(replay[task_id] for task_id in _EXPECTED_TASK_IDS)]
    fit_rows: list[Row] = []
    clauses: dict[str, list[Row]] = {}
    commands: dict[str, list[CommandRow]] = {}
    for manifest_index, item in enumerate(selected):
        task_id = str(item["task_id"])
        attempt = (_ROOT / str(item["trace"])).parent
        artifact = json.loads(
            (attempt / "resource_observations.json").read_text(encoding="utf-8")
        )
        if (
            artifact.get("collection_validity"),
            artifact.get("workload_execution"),
            artifact.get("telemetry_quality"),
            artifact.get("cleanup"),
        ) != ("valid", "completed", "ok", "ok"):
            raise ValueError(f"{task_id}: telemetry is not evidence-valid")
        task_clauses: list[Row] = []
        task_commands: list[CommandRow] = []
        for call_index, call in enumerate(artifact.get("calls", ())):
            if not isinstance(call, Mapping) or call.get("eligible_for_kb") is not True:
                continue
            call_id, command = call.get("tool_call_id"), call.get("command")
            if not isinstance(call_id, str) or not isinstance(command, str):
                raise ValueError(f"{task_id}: eligible call lacks identity")
            retained = tuple(
                row
                for clause in call.get("clauses", ())
                if isinstance(clause, Mapping)
                and (row := _row_from_clause(task_id, manifest_index, clause)) is not None
            )
            if not retained:
                raise ValueError(f"{task_id}:{call_id}: no eligible clauses")
            task_clauses.extend(row for row in retained if row.pipeline_position <= 0)
            task_commands.append(
                CommandRow(
                    task_id,
                    repo_of(task_id),
                    manifest_index,
                    call_index,
                    call_id,
                    command,
                    0.0,
                    retained,
                )
            )
        clauses[task_id] = task_clauses
        commands[task_id] = task_commands
        if manifest_index < len(split["fit"]):
            fit_rows.extend(task_clauses)
    return fit_rows, clauses, commands


def _carrier_count(row: CommandRow) -> int | None:
    parsed = [parse_pytest_workers(clause.argv) for clause in row.clauses]
    values = {value for value in parsed if isinstance(value, int)}
    return next(iter(values)) if len(values) == 1 and "invalid" not in parsed else None


def _hard(pmf: Sequence[float] | None) -> int | None:
    return None if pmf is None else max(range(3), key=pmf.__getitem__)


def fit_clause_kb(rows: Sequence[Row]) -> ClauseResourceKB:
    """Build the fit fallback and the settled repository-local hierarchy."""

    kb = ClauseResourceKB.fit_public(row.observation(0.0, 1.0) for row in rows)
    for row in rows:
        kb.observe_completed_clause(row.observation(0.0, 1.0))
    return kb


def _metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    eligible = [row for row in rows if row["label"] is not None]
    high = [row for row in eligible if row["label"] == 2]
    return {
        "commands": len(rows),
        "eligible_labels": len(eligible),
        "available": sum(row[arm] is not None for row in rows),
        "accuracy": sum(_hard(row[arm]) == row["label"] for row in eligible)
        / len(eligible),
        "high_labels": len(high),
        "high_recall": sum(_hard(row[arm]) == 2 for row in high) / len(high),
    }


def _changes(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    changed = [
        row for row in rows
        if row["label"] is not None and _hard(row[arm]) != _hard(row["clause_kb"])
    ]
    helpful = [
        row for row in changed
        if _hard(row[arm]) == row["label"] and _hard(row["clause_kb"]) != row["label"]
    ]
    harmful = [
        row for row in changed
        if _hard(row[arm]) != row["label"] and _hard(row["clause_kb"]) == row["label"]
    ]
    return {
        "changed": len(changed),
        "helpful": len(helpful),
        "harmful": len(harmful),
        "helpful_tasks": sorted({str(row["task_id"]) for row in helpful}),
    }


def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head = _clean_head()
    fit, clauses, commands = _load_tasks()
    calibration = [
        row
        for row in fit
        if parse_pytest_workers(row.argv) == "serial"
        and row.sampled_peak_rss_mb is not None
    ]
    calibration_values = [float(row.sampled_peak_rss_mb) for row in calibration]
    calibration_tasks = {row.task_id for row in calibration}
    if len(calibration) < _MIN_CALIBRATION or len(calibration_tasks) < _MIN_CALIBRATION_TASKS:
        raise ValueError("fit calibration misses the frozen coverage gate")

    kb = fit_clause_kb(fit)
    rows: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(_EXPECTED_TASK_IDS):
        query_ts = float(task_index * 2 + 3)
        for command in commands[task_id]:
            prediction = kb.predict_command_resource_buckets(
                command.repo, command.command, query_ts
            ).classifications[_RSS]
            baseline = None if prediction is None else prediction.probability_by_bucket
            workers = _carrier_count(command)
            if workers is None:
                scaled = baseline
                presence = baseline
            else:
                worker_pmf = scaled_rss_pmf(calibration_values, workers)
                presence_pmf = scaled_rss_pmf(calibration_values, 1)
                scaled = worker_pmf if baseline is None else max_bucket_convolution(baseline, worker_pmf)
                presence = presence_pmf if baseline is None else max_bucket_convolution(baseline, presence_pmf)
            label, source = command_resource_bucket_label(command, _RSS)
            rows.append(
                {
                    "sample_id": f"{task_id}:{command.call_index}",
                    "task_id": task_id,
                    "call_id": command.call_id,
                    "command": command.command,
                    "workers": workers,
                    "carrier": workers is not None,
                    "label": label,
                    "label_source": source,
                    "clause_kb": None if baseline is None else list(baseline),
                    "presence_only": None if presence is None else list(presence),
                    "worker_scaling": None if scaled is None else list(scaled),
                }
            )
        settle_ts = query_ts + 0.5
        for clause in clauses[task_id]:
            kb.observe_completed_clause(clause.observation(query_ts, settle_ts))

    carriers = [row for row in rows if row["carrier"]]
    carrier_tasks = {str(row["task_id"]) for row in carriers}
    if len(carriers) < _MIN_CARRIERS or len(carrier_tasks) < _MIN_CARRIER_TASKS:
        raise ValueError("replay carriers miss the frozen coverage gate")
    noncarrier_identity = all(
        row["clause_kb"] == row["presence_only"] == row["worker_scaling"]
        for row in rows if not row["carrier"]
    )
    metrics = {
        scope: {
            arm: _metrics(selected, arm)
            for arm in ("clause_kb", "presence_only", "worker_scaling")
        }
        for scope, selected in (("overall", rows), ("carriers", carriers))
    }
    changes = {
        arm: _changes(carriers, arm)
        for arm in ("presence_only", "worker_scaling")
    }
    baseline = metrics["overall"]["clause_kb"]
    scaled = metrics["overall"]["worker_scaling"]
    carrier_scaled = metrics["carriers"]["worker_scaling"]
    carrier_presence = metrics["carriers"]["presence_only"]
    representation_go = (
        scaled["accuracy"] > baseline["accuracy"]
        and scaled["high_recall"] - baseline["high_recall"] >= _MIN_HIGH_RECALL_GAIN
        and changes["worker_scaling"]["helpful"] > changes["worker_scaling"]["harmful"]
        and len(changes["worker_scaling"]["helpful_tasks"]) >= 3
        and noncarrier_identity
    )
    result = {
        "schema": "pennylane-xdist-rss-positive-control-v1",
        "status": "development_go" if representation_go else "development_no_go",
        "claim_bearing": False,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "evaluation_git_sha": head,
        "evidence": {
            "fit_tasks": 15,
            "replay_tasks": list(_EXPECTED_TASK_IDS),
            "calibration_clauses": len(calibration),
            "calibration_tasks": len(calibration_tasks),
            "carrier_commands": len(carriers),
            "carrier_tasks": len(carrier_tasks),
        },
        "method": {
            "host_cpus": _HOST_CPUS,
            "scaling": "(workers + 1) * fit_serial_pytest_clause_rss",
            "composition": "max_bucket_convolution_with_clause_kb_command_pmf",
            "causal_update": "after_whole_task_settlement",
            "prediction_time_agent_calls": 0,
        },
        "metrics": metrics,
        "changes_vs_clause_kb": changes,
        "gates": {
            "minimum_high_recall_gain": _MIN_HIGH_RECALL_GAIN,
            "noncarrier_pmf_bit_identical": noncarrier_identity,
            "representation_go": representation_go,
            "worker_count_specificity": carrier_scaled["accuracy"] > carrier_presence["accuracy"],
        },
        "limitations": [
            "development-exposed tasks",
            "positive-control process scaling, not a learned worker-memory model",
            "canonical clause RSS target only; no scheduler or interference result",
        ],
    }
    return result, rows


def main() -> None:
    if _OUTPUT.exists():
        raise FileExistsError(_OUTPUT)
    started = time.monotonic()
    result, rows = run()
    result["cost"] = {"evaluation_seconds": time.monotonic() - started}
    _OUTPUT.mkdir(parents=True)
    (_OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (_OUTPUT / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
