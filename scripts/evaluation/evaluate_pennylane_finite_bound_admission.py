#!/usr/bin/env python3
"""Evaluate fit-only finite resource bounds for causal PennyLane admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    RESOURCE_BUCKET_LABELS,
    command_resource_bucket_label,
    load_run_rows,
)
from scripts.evaluation.evaluate_pennylane_causal_joint_admission import (  # noqa: E402
    Program,
    _CPU_PAGES,
    _INPUT_TREE_SHA256,
    _RSS_PAGES,
    _reservations,
    _segments,
    simulate,
)
from scripts.evaluation.evaluate_pennylane_joint_phase_packing import (  # noqa: E402
    Capacities,
    _load_profiles,
    simulate as simulate_oracle,
)
from scripts.evaluation.evaluate_pennylane_multitarget import (  # noqa: E402
    RUN_DIR,
    _write_results_view,
)

_PROTOCOL = (
    _ROOT / "analysis/development/pennylane-finite-bound-joint-admission-protocol.md"
)
_HARD_RESULT = (
    _ROOT / "analysis/results/pennylane-causal-joint-tool-admission-v1/result.json"
)
_OUTPUT = (
    _ROOT / "analysis/results/pennylane-finite-bound-joint-admission-v1/result.json"
)
_TARGETS = ("peak_cpu_cores", "sampled_peak_rss_mb")
_DEFAULTS = {
    "peak_cpu_cores": _CPU_PAGES,
    "sampled_peak_rss_mb": _RSS_PAGES,
}
_CAPACITIES = {
    "peak_cpu_cores": 43.0,
    "sampled_peak_rss_mb": 80_000.0,
}
_EPS = 1e-8


def _require_clean_checkout() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal evaluation requires a clean committed checkout")
    relative = _PROTOCOL.relative_to(_ROOT).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if _PROTOCOL.read_bytes() != committed:
        raise ValueError("finite-bound protocol differs from HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _finite_class_bounds(
    values: Mapping[int, Sequence[float]],
    defaults: Sequence[float],
    capacity: float,
) -> tuple[float, float, float]:
    bounds: list[float] = []
    for bucket in range(3):
        observed = values.get(bucket, ())
        bound = max(observed) if observed else float(defaults[bucket])
        bound = min(capacity, bound)
        if bounds:
            bound = max(bounds[-1], bound)
        bounds.append(bound)
    return bounds[0], bounds[1], bounds[2]


def _fit_rows(task_ids: Sequence[str]) -> list[CommandRow]:
    with tempfile.TemporaryDirectory() as temporary:
        view = Path(temporary) / "results.jsonl"
        _write_results_view(view, task_ids)
        loaded, _clauses, commands = load_run_rows(RUN_DIR, results_path=view)
    if loaded != list(task_ids):
        raise ValueError("finite-bound fit tasks differ from frozen split")
    return commands


def _fit_bounds(
    rows: Sequence[CommandRow], programs: Sequence[Program]
) -> tuple[dict[str, tuple[float, float, float]], dict[str, Any]]:
    action_peaks: dict[tuple[str, str], dict[str, float]] = {}
    for program in programs:
        for segment in program.segments:
            if segment.kind != "exec" or segment.action_id is None:
                continue
            peaks = action_peaks.setdefault(
                (program.task_id, segment.action_id),
                {target: 0.0 for target in _TARGETS},
            )
            peaks["peak_cpu_cores"] = max(peaks["peak_cpu_cores"], segment.cpu_cores)
            peaks["sampled_peak_rss_mb"] = max(
                peaks["sampled_peak_rss_mb"], segment.rss_mb
            )

    values = {target: {bucket: [] for bucket in range(3)} for target in _TARGETS}
    tasks = {target: {bucket: set() for bucket in range(3)} for target in _TARGETS}
    for row in rows:
        peaks = action_peaks.get((row.task_id, row.call_id))
        if peaks is None:
            raise ValueError(f"fit exec is absent from action segments: {row.call_id}")
        for target in _TARGETS:
            bucket, source = command_resource_bucket_label(row, target)
            if (
                bucket is None
                or source != f"observed_composed_{RESOURCE_BUCKET_LABELS[bucket]}"
            ):
                continue
            value = peaks[target]
            if value < 0.0 or not math.isfinite(value):
                raise ValueError("fit action peak is invalid")
            values[target][bucket].append(value)
            tasks[target][bucket].add(row.task_id)

    bounds = {
        target: _finite_class_bounds(
            values[target], _DEFAULTS[target], _CAPACITIES[target]
        )
        for target in _TARGETS
    }
    return bounds, {
        target: {
            "bounds": list(bounds[target]),
            "support_commands": [len(values[target][bucket]) for bucket in range(3)],
            "support_tasks": [len(tasks[target][bucket]) for bucket in range(3)],
            "observed_max": [
                max(values[target][bucket], default=None) for bucket in range(3)
            ],
        }
        for target in _TARGETS
    }


def _finite_requests(
    hard: Mapping[tuple[str, str], tuple[float, float]],
    evidence: Mapping[str, Any],
    bounds: Mapping[str, Sequence[float]],
) -> dict[tuple[str, str], tuple[float, float]]:
    missing_cpu = set(evidence["cpu_prediction_unavailable_command_ids"])
    missing_rss = set(evidence["rss_prediction_unavailable_command_ids"])
    requests = {}
    for key, (cpu, rss) in hard.items():
        text_key = f"{key[0]}:{key[1]}"
        cpu_request = (
            _CPU_PAGES[-1]
            if text_key in missing_cpu
            else bounds["peak_cpu_cores"][_CPU_PAGES.index(cpu)]
        )
        rss_request = (
            _RSS_PAGES[-1]
            if text_key in missing_rss
            else bounds["sampled_peak_rss_mb"][_RSS_PAGES.index(rss)]
        )
        requests[key] = float(cpu_request), float(rss_request)
    return requests


def _gate(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate = arms["finite_fit_feedback"]
    baseline = arms["serial_tool"]

    def safe(arm: Mapping[str, Any]) -> bool:
        return bool(arm["completed"]) and not any(
            float(value) > _EPS for value in arm["capacity_violation_s"].values()
        )

    candidate_safe = safe(candidate)
    baseline_safe = safe(baseline)
    reduction = (
        1.0
        - float(candidate["mean_task_completion_s"])
        / float(baseline["mean_task_completion_s"])
        if candidate_safe and baseline_safe
        else None
    )
    checks = {
        "candidate_completed_and_safe": candidate_safe,
        "serial_tool_completed_and_safe": baseline_safe,
        "mean_completion_at_least_5pct_below_serial_tool": (
            reduction is not None and reduction >= 0.05
        ),
        "makespan_no_higher_than_serial_tool": (
            candidate_safe
            and baseline_safe
            and candidate["makespan_s"] <= baseline["makespan_s"]
        ),
        "overlap_spans_at_least_20_tasks": candidate["tasks_with_overlapped_exec"]
        >= 20,
    }
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "mean_completion_reduction_vs_serial_tool": reduction,
        "checks": checks,
    }


def evaluate(git_sha: str) -> dict[str, Any]:
    profiles, source_evidence = _load_profiles()
    ordered = sorted(profiles, key=lambda profile: profile.task_id)
    fit_profiles, replay_profiles = ordered[:35], ordered[35:]
    fit_ids = [profile.task_id for profile in fit_profiles]
    replay_ids = [profile.task_id for profile in replay_profiles]
    fit_programs = [_segments(profile) for profile in fit_profiles]
    replay_programs = [_segments(profile) for profile in replay_profiles]
    hard_requests, prediction_evidence = _reservations(fit_ids, replay_ids)
    bounds, bound_evidence = _fit_bounds(_fit_rows(fit_ids), fit_programs)
    finite_requests = _finite_requests(hard_requests, prediction_evidence, bounds)
    arms = {
        "serial_tool": simulate(replay_programs, "serial_tool", hard_requests),
        "finite_fit_static": simulate(
            replay_programs, "finite_fit_static", finite_requests
        ),
        "finite_fit_feedback": simulate(
            replay_programs, "finite_fit_feedback", finite_requests
        ),
        "joint_oracle_cap8": simulate_oracle(
            replay_profiles, "joint", active_cap=8, capacities=Capacities()
        ),
    }
    hard_bytes = _HARD_RESULT.read_bytes()
    hard_reference = json.loads(hard_bytes)
    if hard_reference.get("schema") != "pennylane-causal-joint-tool-admission-v1":
        raise ValueError("hard-page reference schema changed")
    gate = _gate(arms)
    return {
        "schema": "pennylane-finite-bound-joint-admission-v1",
        "status": gate["status"],
        "corpus_role": "development_exposed",
        "git_sha": git_sha,
        "protocol": _PROTOCOL.relative_to(_ROOT).as_posix(),
        "run_dir": RUN_DIR.relative_to(_ROOT).as_posix(),
        "input_tree_sha256": _INPUT_TREE_SHA256,
        "fit_task_ids": fit_ids,
        "replay_task_ids": replay_ids,
        "evidence": {
            "source": source_evidence,
            "prediction": prediction_evidence,
            "finite_bounds": bound_evidence,
            "prediction_time_agent_calls": 0,
            "hard_page_reference": {
                "path": _HARD_RESULT.relative_to(_ROOT).as_posix(),
                "sha256": hashlib.sha256(hard_bytes).hexdigest(),
                "status": hard_reference["status"],
            },
        },
        "arms": arms,
        "gate": gate,
        "interpretation_boundary": (
            "Fit-only bucket-to-reservation carrier on development-exposed fixed "
            "trajectories. Zero modeled violations are required; this is not a "
            "physical concurrency or fresh-data result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    result = evaluate(_require_clean_checkout())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
