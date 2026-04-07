"""Phase 4 regression: --benchmark flag resolves from configs/benchmarks YAML."""
from __future__ import annotations

from pathlib import Path

from trace_collect.cli import parse_collect_args

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_benchmark_is_swe_bench_verified() -> None:
    args = parse_collect_args([])
    assert args.benchmark == "swe-bench-verified"


def test_benchmark_flag_accepts_swe_rebench() -> None:
    args = parse_collect_args(["--benchmark", "swe-rebench"])
    assert args.benchmark == "swe-rebench"


def test_harness_dataset_flag_removed() -> None:
    """--harness-dataset was removed in Phase 4; parser must reject it."""
    import pytest
    with pytest.raises(SystemExit):
        parse_collect_args(["--harness-dataset", "foo"])


def test_swe_bench_verified_yaml_exists_and_loads() -> None:
    from agents.benchmarks.base import BenchmarkConfig
    yaml_path = REPO_ROOT / "configs" / "benchmarks" / "swe-bench-verified.yaml"
    assert yaml_path.exists()
    config = BenchmarkConfig.from_yaml(yaml_path)
    assert config.slug == "swe-bench-verified"
    assert config.display_name == "SWE-Bench Verified"
    assert config.harness_dataset == "princeton-nlp/SWE-bench_Verified"
    assert str(config.trace_root) == "traces/swebench_verified"  # legacy path preserved
    assert config.docker_namespace == "swebench"


def test_swe_rebench_yaml_exists_and_loads() -> None:
    from agents.benchmarks.base import BenchmarkConfig
    yaml_path = REPO_ROOT / "configs" / "benchmarks" / "swe-rebench.yaml"
    assert yaml_path.exists()
    config = BenchmarkConfig.from_yaml(yaml_path)
    assert config.slug == "swe-rebench"
    assert config.harness_dataset == "nebius/SWE-rebench"
    assert config.harness_split == "filtered"
    assert config.docker_namespace is None
    assert config.exclude_lite is False
