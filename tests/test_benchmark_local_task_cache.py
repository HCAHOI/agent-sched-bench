from __future__ import annotations

import json
from pathlib import Path

from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.swe_rebench import SWERebenchBenchmark


def _config(tmp_path: Path, *, slug: str) -> BenchmarkConfig:
    data_root = tmp_path / "data"
    data_root.mkdir()
    return BenchmarkConfig(
        slug=slug,
        display_name=slug,
        trace_root=tmp_path / "traces",
        default_max_iterations=100,
        selection_n=1,
        selection_seed=1,
        harness_dataset="should-not-be-loaded",
        harness_split="test",
        data_root=data_root,
    )


def test_local_tasks_json_is_opt_in(tmp_path: Path) -> None:
    config = _config(tmp_path, slug="swe-rebench")
    assert config.data_root is not None
    (config.data_root / "tasks.json").write_text("[]", encoding="utf-8")

    assert SWERebenchBenchmark(config).load_tasks_from_local_json() is None


def test_local_tasks_json_requires_instance_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SCHED_BENCH_USE_LOCAL_TASK_CACHE", "1")
    config = _config(tmp_path, slug="swe-rebench")
    assert config.data_root is not None
    (config.data_root / "tasks.json").write_text(
        json.dumps([{"repo": "repo/project"}]),
        encoding="utf-8",
    )

    try:
        SWERebenchBenchmark(config).load_tasks_from_local_json()
    except ValueError as exc:
        assert "missing instance_id" in str(exc)
    else:
        raise AssertionError("expected missing instance_id failure")


def test_swe_rebench_loads_local_tasks_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SCHED_BENCH_USE_LOCAL_TASK_CACHE", "1")
    config = _config(tmp_path, slug="swe-rebench")
    assert config.data_root is not None
    (config.data_root / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "instance_id": "repo__issue-2",
                    "repo": "repo/project",
                    "problem_statement": "Fix it",
                    "FAIL_TO_PASS": ["tests/test_bug.py::test_fix"],
                    "docker_image": "swerebench/example:latest",
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = SWERebenchBenchmark(config).load_tasks()

    assert len(tasks) == 1
    assert tasks[0]["test_cmd"] == "python -m pytest tests/test_bug.py::test_fix -x --no-header -q"
    assert tasks[0]["image_name"] == "swerebench/example:latest"
    assert tasks[0]["task_source_kind"] == "benchmark_local_json"
    assert tasks[0]["task_source_id"] == "repo__issue-2"
    assert tasks[0]["task_source_path"] == str(config.data_root / "tasks.json")
