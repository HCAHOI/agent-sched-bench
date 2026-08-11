#!/usr/bin/env python3
"""Evaluate the frozen PennyLane time-varying RSS packing ceiling."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from harness.container_stats_sampler import _parse_memory_mb  # noqa: E402


_PROTOCOL_GIT_SHA = "348c98a8e3e8fa8e3cd21030ae71702f06fbf347"
_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
_CAPACITY_MB = 16_000.0
_MINIMUM_REDUCTION = 0.05
_EXPECTED_TASK_IDS = (
    "PennyLaneAI__pennylane-2601",
    "PennyLaneAI__pennylane-2603",
    "PennyLaneAI__pennylane-2654",
    "PennyLaneAI__pennylane-2668",
    "PennyLaneAI__pennylane-2834",
    "PennyLaneAI__pennylane-2947",
    "PennyLaneAI__pennylane-2964",
    "PennyLaneAI__pennylane-3024",
    "PennyLaneAI__pennylane-3033",
    "PennyLaneAI__pennylane-3057",
    "PennyLaneAI__pennylane-3182",
    "PennyLaneAI__pennylane-3266",
    "PennyLaneAI__pennylane-3278",
    "PennyLaneAI__pennylane-3381",
    "PennyLaneAI__pennylane-3386",
)


@dataclass(frozen=True)
class RssJob:
    task_id: str
    offsets_s: tuple[float, ...]
    rss_mb: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return self.offsets_s[-1]

    @property
    def peak_rss_mb(self) -> float:
        return max(self.rss_mb)


def _require_clean_checkout() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal evaluation requires a clean committed checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _valid_artifact(artifact: Mapping[str, Any]) -> bool:
    return (
        artifact.get("collection_validity"),
        artifact.get("workload_execution"),
        artifact.get("telemetry_quality"),
        artifact.get("cleanup"),
    ) == ("valid", "completed", "ok", "ok")


def _load_jobs() -> tuple[list[RssJob], dict[str, Any]]:
    split = json.loads(_SPLIT.read_text(encoding="utf-8"))
    jobs: list[RssJob] = []
    gaps: list[float] = []
    for item in split["replay"]:
        task_id = str(item["task_id"])
        trace_path = _ROOT / str(item["trace"])
        artifact_path = trace_path.parent / "resource_observations.json"
        if not artifact_path.exists():
            continue
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not _valid_artifact(artifact):
            continue
        resources = json.loads(
            (trace_path.parent / "resources.json").read_text(encoding="utf-8")
        )
        samples = sorted(resources["samples"], key=lambda row: float(row["epoch"]))
        if len(samples) < 2:
            raise ValueError(f"{task_id}: fewer than two RSS samples")
        first_epoch = float(samples[0]["epoch"])
        offsets = tuple(float(row["epoch"]) - first_epoch for row in samples)
        values = tuple(_parse_memory_mb(str(row["mem_usage"])) for row in samples)
        if any(value is None or value < 0.0 for value in values):
            raise ValueError(f"{task_id}: invalid RSS sample")
        if any(right <= left for left, right in zip(offsets, offsets[1:])):
            raise ValueError(f"{task_id}: non-increasing sample timestamps")
        gaps.extend(right - left for left, right in zip(offsets, offsets[1:]))
        jobs.append(RssJob(task_id, offsets, tuple(float(value) for value in values)))
    jobs.sort(key=lambda job: job.task_id)
    actual_ids = tuple(job.task_id for job in jobs)
    if actual_ids != _EXPECTED_TASK_IDS:
        raise ValueError(f"evidence-valid cohort changed: {actual_ids!r}")
    return jobs, {
        "task_count": len(jobs),
        "sample_count": sum(len(job.offsets_s) for job in jobs),
        "median_sample_gap_s": statistics.median(gaps),
        "maximum_sample_gap_s": max(gaps),
    }


def _rss_at(job: RssJob, elapsed_s: float) -> float:
    if elapsed_s < 0.0 or elapsed_s >= job.duration_s:
        return 0.0
    index = min(bisect_right(job.offsets_s, elapsed_s) - 1, len(job.rss_mb) - 2)
    return job.rss_mb[max(0, index)]


def _rss_before(job: RssJob, elapsed_s: float) -> float:
    if elapsed_s <= 0.0 or elapsed_s > job.duration_s:
        return 0.0
    if elapsed_s == job.duration_s:
        return job.rss_mb[-1]
    return job.rss_mb[max(0, bisect_right(job.offsets_s, elapsed_s) - 1)]


def _snapshot_rss(
    jobs: Mapping[str, RssJob],
    starts: Mapping[str, float],
    time_s: float,
    *,
    before_transitions: bool,
) -> float:
    rss = _rss_before if before_transitions else _rss_at
    return sum(rss(jobs[task_id], time_s - start_s) for task_id, start_s in starts.items())


def _aggregate_peak(
    jobs: Mapping[str, RssJob],
    starts: Mapping[str, float],
    *,
    start_at_s: float | None = None,
    end_at_s: float | None = None,
) -> tuple[float, int]:
    boundaries = sorted(
        {
            start_s + offset_s
            for task_id, start_s in starts.items()
            for offset_s in jobs[task_id].offsets_s
            if (start_at_s is None or start_s + offset_s >= start_at_s)
            and (end_at_s is None or start_s + offset_s <= end_at_s)
        }
    )
    values = [
        _snapshot_rss(jobs, starts, time_s, before_transitions=before)
        for time_s in boundaries
        for before in (True, False)
    ]
    return max(values, default=0.0), sum(value > _CAPACITY_MB for value in values)


def _temporal_fit(
    candidate: RssJob,
    now_s: float,
    jobs: Mapping[str, RssJob],
    starts: Mapping[str, float],
) -> bool:
    tentative_jobs = dict(jobs)
    tentative_jobs[candidate.task_id] = candidate
    tentative_starts = dict(starts)
    tentative_starts[candidate.task_id] = now_s
    peak, _violations = _aggregate_peak(
        tentative_jobs,
        tentative_starts,
        start_at_s=now_s,
        end_at_s=now_s + candidate.duration_s,
    )
    return peak <= _CAPACITY_MB


def simulate(jobs: list[RssJob], arm: str) -> dict[str, Any]:
    by_id = {job.task_id: job for job in jobs}
    if arm == "unconstrained":
        starts = {job.task_id: 0.0 for job in jobs}
    else:
        starts: dict[str, float] = {}
        running: dict[str, float] = {}
        pending = list(jobs)
        now_s = 0.0
        while pending:
            running = {
                task_id: start_s
                for task_id, start_s in running.items()
                if start_s + by_id[task_id].duration_s > now_s
            }
            admitted: list[RssJob] = []
            for job in pending:
                if arm == "serial":
                    fits = not running
                elif arm == "static_peak":
                    fits = (
                        sum(by_id[task_id].peak_rss_mb for task_id in running)
                        + job.peak_rss_mb
                        <= _CAPACITY_MB
                    )
                elif arm == "temporal_oracle":
                    fits = _temporal_fit(job, now_s, by_id, running)
                else:
                    raise ValueError(f"unknown arm: {arm}")
                if fits:
                    starts[job.task_id] = now_s
                    running[job.task_id] = now_s
                    admitted.append(job)
            pending = [job for job in pending if job not in admitted]
            if pending:
                if not running:
                    raise ValueError(f"{pending[0].task_id}: cannot fit into capacity")
                now_s = min(
                    start_s + by_id[task_id].duration_s
                    for task_id, start_s in running.items()
                )

    completions = {
        task_id: start_s + by_id[task_id].duration_s
        for task_id, start_s in starts.items()
    }
    first_completion = min(completions.values())
    events = sorted(
        [(start_s, 1) for start_s in starts.values()]
        + [(completion_s, -1) for completion_s in completions.values()],
        key=lambda event: (event[0], event[1]),
    )
    concurrency = current = 0
    for _time_s, delta in events:
        current += delta
        concurrency = max(concurrency, current)
    peak_rss_mb, violation_points = _aggregate_peak(by_id, starts)
    return {
        "mean_task_completion_s": statistics.mean(completions.values()),
        "makespan_s": max(completions.values()),
        "maximum_concurrency": concurrency,
        "starts_before_first_completion": sum(
            start_s < first_completion for start_s in starts.values()
        ),
        "maximum_sampled_aggregate_rss_mb": peak_rss_mb,
        "sampled_capacity_violation_points": violation_points,
        "capacity_violation": violation_points > 0,
        "start_s_by_task": starts,
        "completion_s_by_task": completions,
    }


def _gate(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = arms["static_peak"]
    candidate = arms["temporal_oracle"]
    reduction = 1.0 - float(candidate["mean_task_completion_s"]) / float(
        baseline["mean_task_completion_s"]
    )
    checks = {
        "mean_completion_reduction_at_least_5pct": reduction
        >= _MINIMUM_REDUCTION,
        "makespan_strictly_lower": candidate["makespan_s"] < baseline["makespan_s"],
        "additional_start_before_first_completion": candidate[
            "starts_before_first_completion"
        ]
        > baseline["starts_before_first_completion"],
        "temporal_zero_sampled_capacity_violations": not candidate[
            "capacity_violation"
        ],
        "unconstrained_has_sampled_capacity_violation": arms["unconstrained"][
            "capacity_violation"
        ],
    }
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "mean_completion_reduction_vs_static_peak": reduction,
        "checks": checks,
    }


def evaluate(eval_git_sha: str) -> dict[str, Any]:
    jobs, telemetry = _load_jobs()
    arms = {
        arm: simulate(jobs, arm)
        for arm in ("serial", "static_peak", "temporal_oracle", "unconstrained")
    }
    gate = _gate(arms)
    return {
        "schema_version": 1,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "evaluation_git_sha": eval_git_sha,
        "corpus_role": "development_exposed",
        "capacity_mb": _CAPACITY_MB,
        "task_ids": [job.task_id for job in jobs],
        "telemetry": telemetry,
        "arms": arms,
        "gate": gate,
        "status": gate["status"],
        "interpretation_boundary": (
            "Memory-only hindsight ceiling with fixed recorded service; no CPU "
            "contention, predictor, or deployable causal action."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(_require_clean_checkout())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
