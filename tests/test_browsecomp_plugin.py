from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.browsecomp import BrowseCompBenchmark


def _make_config(**extras: object) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="browsecomp",
        display_name="BrowseComp",
        harness_dataset="example/browsecomp",
        harness_split="test",
        trace_root=Path("traces/browsecomp"),
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
        extras={
            "id_field": "id",
            "question_field": "question",
            "answer_field": "answer",
            "source_urls_field": "source_urls",
            **extras,
        },
    )


def test_browsecomp_registered() -> None:
    assert REGISTRY["browsecomp"] is BrowseCompBenchmark
    assert get_benchmark_class("browsecomp") is BrowseCompBenchmark


def test_browsecomp_normalize_task_preserves_source_urls() -> None:
    plugin = BrowseCompBenchmark(_make_config())

    normalized = plugin.normalize_task(
        {
            "id": "bc-1",
            "question": "Which source states the fact?",
            "answer": "The cited source.",
            "source_urls": ["https://example.com/a", "https://example.com/b"],
        }
    )

    assert normalized == {
        "instance_id": "bc-1",
        "problem_statement": "Which source states the fact?",
        "reference_answer": "The cited source.",
        "source_urls": ["https://example.com/a", "https://example.com/b"],
        "repo": None,
        "image_name": None,
        "docker_image": None,
    }


def test_browsecomp_normalize_task_decodes_json_url_list() -> None:
    plugin = BrowseCompBenchmark(_make_config())

    normalized = plugin.normalize_task(
        {
            "id": "bc-2",
            "question": "Question",
            "answer": "Answer",
            "source_urls": '["https://example.com/a"]',
        }
    )

    assert normalized["source_urls"] == ["https://example.com/a"]


def test_browsecomp_load_tasks_uses_configured_hf_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    datasets_mod = types.ModuleType("datasets")

    def load_dataset(dataset: str, *, split: str):
        seen["dataset"] = dataset
        seen["split"] = split
        return [
            {
                "id": "bc-1",
                "question": "Question",
                "answer": "Answer",
                "source_urls": [],
            }
        ]

    datasets_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    plugin = BrowseCompBenchmark(_make_config())
    tasks = plugin.load_tasks()

    assert seen == {"dataset": "example/browsecomp", "split": "test"}
    assert tasks[0]["instance_id"] == "bc-1"


def test_browsecomp_runtime_and_runner_gating() -> None:
    plugin = BrowseCompBenchmark(_make_config())

    assert plugin.execution_environment == "host"
    assert plugin.runtime_mode_for("openclaw") == "host_controller"
    assert plugin.runtime_mode_for("qwen-deep-research") == "host_controller"
    with pytest.raises(NotImplementedError, match="Phase 3"):
        plugin.build_runner(scaffold="qwen-deep-research")

