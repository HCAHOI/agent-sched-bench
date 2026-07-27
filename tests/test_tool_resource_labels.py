from __future__ import annotations

import json
from pathlib import Path

from tool_resource.labels import extract_resource_call_samples, load_resource_corpus


def test_resource_labels_keep_peak_ambient_missing_and_censored_separate(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(tmp_path)
    (tmp_path / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"epoch": 100.2, "mem_usage": "256MiB / 8GiB"},
                    {"epoch": 101.0, "mem_usage": "0.5GiB / 8GiB"},
                    {"epoch": 103.0, "mem_usage": "300MiB / 8GiB"},
                    {"epoch": 106.5, "mem_usage": "700MiB / 8GiB"},
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = {
        sample.action_id: sample for sample in extract_resource_call_samples(trace_path)
    }

    measured = samples["measured"]
    assert measured.cpu_core_seconds == 6.4
    assert measured.cpu_core_seconds_eligible
    assert measured.peak_cpu_cores == 8.0
    assert measured.peak_cpu_cores_eligible
    assert measured.peak_cpu_clipped_sample_count == 1
    assert measured.peak_memory_mb == 512.0
    assert measured.peak_memory_mb_eligible
    assert measured.ambient_memory_mb is None
    assert measured.memory_window_sample_count == 2

    non_exec = samples["non-exec"]
    assert non_exec.cpu_core_seconds == 0.0
    assert non_exec.cpu_core_seconds_kind == "non_exec_zero"
    assert non_exec.ambient_memory_mb == 300.0
    assert non_exec.ambient_memory_mb_eligible
    assert non_exec.peak_memory_mb is None

    censored = samples["censored"]
    assert censored.censored
    assert censored.cpu_core_seconds is None
    assert not censored.cpu_core_seconds_eligible
    assert censored.peak_memory_mb is None
    assert censored.ambient_memory_mb is None

    missing = samples["missing"]
    assert missing.cpu_core_seconds is None
    assert missing.cpu_core_seconds_kind == "missing_exec_timeline"


def test_short_or_single_interval_exec_has_no_peak_cpu_label(tmp_path: Path) -> None:
    trace_path = _write_trace(tmp_path)
    samples = {
        sample.action_id: sample for sample in extract_resource_call_samples(trace_path)
    }

    assert samples["short"].cpu_core_seconds_eligible
    assert not samples["short"].peak_cpu_cores_eligible
    assert samples["short"].peak_cpu_cores is None
    assert samples["single"].cpu_core_seconds_eligible
    assert not samples["single"].peak_cpu_cores_eligible
    assert samples["single"].peak_cpu_cores is None


def test_ambient_before_never_uses_at_or_after_call_start(tmp_path: Path) -> None:
    trace_path = _write_trace(tmp_path)
    (tmp_path / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"epoch": 98.0, "mem_usage": "50MiB / 8GiB"},
                    {"epoch": 99.0, "mem_usage": "100MiB / 8GiB"},
                    {"epoch": 100.0, "mem_usage": "900MiB / 8GiB"},
                    {"epoch": 100.1, "mem_usage": "800MiB / 8GiB"},
                ]
            }
        ),
        encoding="utf-8",
    )

    measured = next(
        sample
        for sample in extract_resource_call_samples(trace_path)
        if sample.action_id == "measured"
    )

    assert measured.ambient_before_mb == 100.0
    assert measured.ambient_before_age_s == 1.0


def test_corpus_loader_uses_declared_membership(tmp_path: Path) -> None:
    _write_trace(tmp_path)
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps({"expected_task_count": 1, "task_ids": ["task-a"]}),
        encoding="utf-8",
    )

    samples_by_task, task_ids = load_resource_corpus(tmp_path, manifest)
    assert task_ids == ["task-a"]
    assert len(samples_by_task["task-a"]) == 6


def _write_trace(root: Path) -> Path:
    trace_path = root / "trace.jsonl"
    timeline = {
        "samples": [
            {"dt_s": 0.5, "cpu_core_s": 5.0, "cpu_quota_cores": 8.0},
            {"dt_s": 0.7, "cpu_core_s": 1.4, "cpu_quota_cores": 8.0},
        ],
        "summary": {"cpu_core_s": 6.4},
    }
    records = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "instance_id": "task-a",
        },
        _action("measured", 1, 100.0, 101.2, "exec", timeline=timeline),
        _action("non-exec", 2, 102.0, 102.1, "read_file"),
        _action(
            "censored",
            3,
            106.0,
            107.0,
            "exec",
            timeline=timeline,
            success=False,
            result="Error: [timeout] stopped\nExit code: 124",
        ),
        _action("missing", 4, 110.0, 110.2, "exec"),
        _action(
            "short",
            5,
            112.0,
            112.8,
            "exec",
            timeline=timeline,
        ),
        _action(
            "single",
            6,
            114.0,
            115.2,
            "exec",
            timeline={
                "samples": [{"dt_s": 1.2, "cpu_core_s": 1.2, "cpu_quota_cores": 8.0}],
                "summary": {"cpu_core_s": 1.2},
            },
        ),
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return trace_path


def _action(
    action_id: str,
    iteration: int,
    start: float,
    end: float,
    tool_name: str,
    *,
    timeline: dict[str, object] | None = None,
    success: bool = True,
    result: str = "ok",
) -> dict[str, object]:
    data: dict[str, object] = {
        "tool_name": tool_name,
        "tool_call_id": action_id,
        "tool_args": json.dumps({"command": "pytest -q"}),
        "success": success,
        "tool_result": result,
    }
    if timeline is not None:
        data["resource_timeline"] = timeline
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "agent_id": "agent-a",
        "instance_id": "task-a",
        "iteration": iteration,
        "ts_start": start,
        "ts_end": end,
        "data": data,
    }
