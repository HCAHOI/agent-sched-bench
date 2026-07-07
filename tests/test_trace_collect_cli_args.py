"""Tests for trace_collect CLI argument parsing."""

from __future__ import annotations

import pytest
from pathlib import Path
from types import SimpleNamespace


from llm_call.config import ResolvedLLMConfig
from trace_collect.cli import _run_collect, parse_collect_args, parse_simulate_args


def test_parse_collect_args_accepts_skip_and_concurrency() -> None:
    args = parse_collect_args([
        "--provider",
        "openrouter",
        "--model",
        "z-ai/glm-5.1",
        "--skip",
        "7",
        "--selection-seed",
        "43",
        "--concurrency",
        "3",
    ])

    assert args.skip == 7
    assert args.selection_seed == 43
    assert args.concurrency == 3


def test_parse_collect_args_rejects_negative_skip() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--skip",
            "-1",
        ])


def test_parse_collect_args_rejects_negative_sample() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--sample",
            "-1",
        ])


def test_parse_collect_args_rejects_zero_concurrency() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--concurrency",
            "0",
        ])


def test_parse_simulate_args_does_not_default_task_source() -> None:
    args = parse_simulate_args([
        "--manifest",
        "/abs/path/to/manifest.yaml",
    ])

    assert args.task_source is None


def test_parse_simulate_args_accepts_explicit_task_source() -> None:
    args = parse_simulate_args([
        "--manifest",
        "/abs/path/to/manifest.yaml",
        "--task-source",
        "/abs/path/to/tasks.json",
    ])

    assert args.task_source == "/abs/path/to/tasks.json"


def test_parse_collect_args_accepts_deep_research_scaffold() -> None:
    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "openai/gpt-4.1",
            "--benchmark",
            "browsecomp",
            "--scaffold",
            "deep-research",
        ]
    )

    assert args.scaffold == "deep-research"
    assert args.benchmark == "browsecomp"
    assert args.mcp_config is None


def test_run_collect_normalizes_omitted_mcp_config_for_deep_research(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agents.benchmarks
    import trace_collect.collector
    from agents.benchmarks.base import BenchmarkConfig

    benchmark_yaml = tmp_path / "configs" / "benchmarks" / "browsecomp.yaml"
    benchmark_yaml.parent.mkdir(parents=True)
    benchmark_yaml.write_text("slug: browsecomp\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr("trace_collect.cli.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **kwargs: ResolvedLLMConfig(
            name="openrouter",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-key",
            model="openai/gpt-4.1",
            env_key="OPENROUTER_API_KEY",
        ),
    )
    monkeypatch.setattr(
        BenchmarkConfig,
        "from_yaml",
        classmethod(
            lambda cls, path: SimpleNamespace(
                slug="browsecomp",
                trace_root=tmp_path / "traces",
            )
        ),
    )

    class FakeBenchmark:
        def __init__(self, config) -> None:
            self.config = config

    async def fake_collect_traces(**kwargs):
        captured.update(kwargs)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        return run_dir

    monkeypatch.setattr(
        agents.benchmarks,
        "get_benchmark_class",
        lambda slug: FakeBenchmark,
    )
    monkeypatch.setattr(trace_collect.collector, "collect_traces", fake_collect_traces)

    _run_collect(
        parse_collect_args(
            [
                "--provider",
                "openrouter",
                "--model",
                "openai/gpt-4.1",
                "--benchmark",
                "browsecomp",
                "--scaffold",
                "deep-research",
                "--run-id",
                str(tmp_path / "run"),
            ]
        )
    )

    assert captured["scaffold"] == "deep-research"
    assert captured["mcp_config"] == "none"
