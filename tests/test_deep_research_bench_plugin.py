from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks._research import HostResearchOpenClawRunner
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.deep_research_bench import DeepResearchBenchBenchmark


def _make_config(**extras: object) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="deep-research-bench",
        display_name="DeepResearchBench",
        harness_dataset="example/deepresearchbench",
        harness_split="test",
        trace_root=Path("traces/deep-research-bench"),
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
        extras={
            "id_field": "id",
            "question_field": "question",
            "answer_field": "answer",
            "topic_field": "topic",
            "difficulty_field": "difficulty",
            "domain_field": "domain",
            **extras,
        },
    )


def test_deep_research_bench_registered() -> None:
    assert REGISTRY["deep-research-bench"] is DeepResearchBenchBenchmark
    assert get_benchmark_class("deep-research-bench") is DeepResearchBenchBenchmark


def test_deep_research_bench_normalize_task_preserves_research_fields() -> None:
    plugin = DeepResearchBenchBenchmark(_make_config())

    normalized = plugin.normalize_task(
        {
            "id": "drb-1",
            "question": "What happened?",
            "answer": "A documented event.",
            "topic": "history",
            "difficulty": "hard",
            "domain": "humanities",
        }
    )

    assert normalized == {
        "instance_id": "drb-1",
        "problem_statement": "What happened?",
        "reference_answer": "A documented event.",
        "topic": "history",
        "difficulty": "hard",
        "domain": "humanities",
        "repo": None,
        "image_name": None,
        "docker_image": None,
    }


def test_deep_research_bench_load_tasks_uses_configured_hf_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    datasets_mod = types.ModuleType("datasets")

    def load_dataset(dataset: str, *, split: str):
        seen["dataset"] = dataset
        seen["split"] = split
        return [
            {
                "id": "drb-1",
                "question": "Question",
                "answer": "Answer",
            }
        ]

    datasets_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    plugin = DeepResearchBenchBenchmark(_make_config())
    tasks = plugin.load_tasks()

    assert seen == {"dataset": "example/deepresearchbench", "split": "test"}
    assert tasks[0]["instance_id"] == "drb-1"


def test_deep_research_bench_runtime_and_runner_gating() -> None:
    plugin = DeepResearchBenchBenchmark(_make_config())

    assert plugin.execution_environment == "host"
    assert plugin.runtime_mode_for("openclaw") == "host_controller"
    assert plugin.runtime_mode_for("qwen-deep-research") == "host_controller"
    with pytest.raises(NotImplementedError, match="Phase 3"):
        plugin.build_runner(scaffold="qwen-deep-research")


def test_host_research_runner_stamps_host_trace_metadata(tmp_path: Path) -> None:
    runner = object.__new__(HostResearchOpenClawRunner)
    runner.benchmark_slug = "deep-research-bench"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "scaffold": "openclaw",
                "trace_format_version": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner._stamp_trace_metadata(
        trace_path,
        instance_id="drb-1",
        prompt_template="default",
    )

    metadata = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["benchmark"] == "deep-research-bench"
    assert metadata["execution_environment"] == "host"
    assert metadata["instance_id"] == "drb-1"
    assert metadata["prompt_template"] == "default"
