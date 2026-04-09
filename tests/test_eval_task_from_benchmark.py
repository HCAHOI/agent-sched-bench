"""EvalTask.from_benchmark_instance correctness."""
from __future__ import annotations

from pathlib import Path

from agents.openclaw.eval.types import EvalTask


def test_from_benchmark_instance_without_plugin() -> None:
    """When no benchmark is provided, behaves like the old from_swebench_instance."""
    row = {
        "instance_id": "test-1",
        "problem_statement": "Fix it",
        "repo": "foo/bar",
        "base_commit": "abc1234",
        "FAIL_TO_PASS": ["tests/a.py::test_x"],
        "PASS_TO_PASS": ["tests/b.py::test_y"],
    }
    task = EvalTask.from_benchmark_instance(row, Path("/tmp/ws"))
    assert task.instance_id == "test-1"
    assert task.fail_to_pass == ["tests/a.py::test_x"]
    assert task.pass_to_pass == ["tests/b.py::test_y"]


def test_from_benchmark_instance_with_rebench_plugin_pins_image() -> None:
    """When SWE-rebench plugin is provided, its normalize_task runs first
    and the explicit docker_image becomes image_name on the EvalTask."""
    from agents.benchmarks import get_benchmark_class
    from agents.benchmarks.base import BenchmarkConfig

    config = BenchmarkConfig(
        slug="swe-rebench",
        display_name="SWE-rebench",
        harness_dataset="nebius/SWE-rebench",
        harness_split="filtered",
        data_root=Path("data/swe-rebench"),
        repos_root=Path("data/swe-rebench/repos"),
        trace_root=Path("traces/swe-rebench"),
        default_max_iterations=50,
        selection_n=32,
        selection_seed=42,
    )
    plugin = get_benchmark_class("swe-rebench")(config)

    row = {
        "instance_id": "nebius-42",
        "problem_statement": "Fix X",
        "repo": "nebius/foo",
        "base_commit": "cafe",
        "FAIL_TO_PASS": ["tests/x.py::test_y"],
        "PASS_TO_PASS": [],
        "docker_image": "swerebench/sweb.eval.x86_64.nebius_foo-42",
    }
    task = EvalTask.from_benchmark_instance(row, Path("/tmp/ws"), benchmark=plugin)
    assert task.image_name == "swerebench/sweb.eval.x86_64.nebius_foo-42"


def test_from_swebench_instance_no_longer_exists() -> None:
    """'least compensations': the old method name is removed, not aliased."""
    assert not hasattr(EvalTask, "from_swebench_instance"), (
        "from_swebench_instance must be fully renamed; no deprecation alias."
    )
