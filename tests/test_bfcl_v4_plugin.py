"""Tests for :class:`agents.benchmarks.bfcl_v4.BFCLv4Benchmark`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.bfcl_v4 import BFCLv4Benchmark


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="bfcl-v4",
        display_name="BFCL v4 (test)",
        harness_dataset="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        harness_split="v4",
        data_root=tmp_path / "data",
        repos_root=None,
        trace_root=tmp_path / "traces",
        default_max_iterations=20,
        selection_n=4,
        selection_seed=42,
        docker_namespace=None,
    )


@pytest.fixture
def plugin(tmp_path: Path) -> BFCLv4Benchmark:
    return BFCLv4Benchmark(_make_config(tmp_path))


# ── Registry ───────────────────────────────────────────────────────────


def test_registry_lookup() -> None:
    assert "bfcl-v4" in REGISTRY
    assert REGISTRY["bfcl-v4"] is BFCLv4Benchmark
    assert get_benchmark_class("bfcl-v4") is BFCLv4Benchmark


def test_task_shape_is_function_call(plugin: BFCLv4Benchmark) -> None:
    assert plugin.task_shape == "function_call"
    assert plugin.slug == "bfcl-v4"


# ── normalize_task ─────────────────────────────────────────────────────


def test_normalize_task_maps_bfcl_schema_to_canonical(
    plugin: BFCLv4Benchmark,
) -> None:
    raw = {
        "id": "simple_0",
        "category": "simple",
        "question": [[{"role": "user", "content": "What is 2+2?"}]],
        "function": [
            {
                "name": "add",
                "description": "Adds two numbers",
                "parameters": {"type": "dict", "properties": {}},
            }
        ],
        "ground_truth": [{"add": {"a": [2], "b": [2]}}],
    }
    result = plugin.normalize_task(raw)
    assert result["instance_id"] == "simple_0"
    assert result["problem_statement"] == "What is 2+2?"
    assert result["category"] == "simple"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "add"
    assert result["question"] == [[{"role": "user", "content": "What is 2+2?"}]]
    assert result["ground_truth"] == [{"add": {"a": [2], "b": [2]}}]
    # Intentionally absent so EvalTask.needs_prepare returns False.
    assert "repo" not in result
    assert "base_commit" not in result


def test_normalize_task_handles_missing_user_content(
    plugin: BFCLv4Benchmark,
) -> None:
    raw = {"id": "x", "category": "simple", "question": [], "function": []}
    result = plugin.normalize_task(raw)
    assert result["problem_statement"] == ""
    assert result["tools"] == []


# ── load_tasks ─────────────────────────────────────────────────────────


def _write_tasks(data_root: Path, rows: list[dict]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "tasks.json").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_load_tasks_filters_deferred_categories(
    plugin: BFCLv4Benchmark, caplog: pytest.LogCaptureFixture
) -> None:
    rows = [
        {
            "id": "s0",
            "category": "simple_python",
            "question": [[{"role": "user", "content": "x"}]],
            "function": [],
            "ground_truth": [],
        },
        {
            "id": "mt0",
            "category": "multi_turn_base",
            "question": [[{"role": "user", "content": "y"}]],
            "function": [],
            "ground_truth": [],
        },
        {
            "id": "mem0",
            "category": "memory",
            "question": [[{"role": "user", "content": "z"}]],
            "function": [],
            "ground_truth": [],
        },
    ]
    _write_tasks(plugin.config.data_root, rows)

    with caplog.at_level(logging.WARNING, logger="agents.benchmarks.bfcl_v4"):
        tasks = plugin.load_tasks()

    assert len(tasks) == 1
    assert tasks[0]["instance_id"] == "s0"
    assert any("deferred categories" in rec.message for rec in caplog.records)
    # Both deferred categories named in the warning summary.
    warnings = "\n".join(rec.message for rec in caplog.records)
    assert "multi_turn_base=1" in warnings
    assert "memory=1" in warnings


def test_load_tasks_missing_file_raises(plugin: BFCLv4Benchmark) -> None:
    with pytest.raises(FileNotFoundError, match="bfcl-v4"):
        plugin.load_tasks()


def test_load_tasks_skips_malformed_lines(
    plugin: BFCLv4Benchmark, caplog: pytest.LogCaptureFixture
) -> None:
    plugin.config.data_root.mkdir(parents=True, exist_ok=True)
    (plugin.config.data_root / "tasks.json").write_text(
        '{"id": "s0", "category": "simple_python", "question": [], "function": []}\n'
        "NOT VALID JSON {\n"
        '{"id": "s1", "category": "simple_python", "question": [], "function": []}\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="agents.benchmarks.bfcl_v4"):
        tasks = plugin.load_tasks()
    assert len(tasks) == 2
    assert any("malformed JSON" in rec.message for rec in caplog.records)


def test_load_tasks_drops_unknown_categories_loudly(
    plugin: BFCLv4Benchmark, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown categories must be dropped with a loud warning, not silently kept."""
    # CLAUDE.md §5 completeness: silently keeping unknown rows biases results
    rows = [
        {"id": "s0", "category": "simple_python",
         "question": [[{"role": "user", "content": "x"}]], "function": []},
        {"id": "u0", "category": "brand_new_category_upstream",
         "question": [[{"role": "user", "content": "y"}]], "function": []},
        {"id": "u1", "category": "brand_new_category_upstream",
         "question": [[{"role": "user", "content": "z"}]], "function": []},
    ]
    _write_tasks(plugin.config.data_root, rows)
    with caplog.at_level(logging.WARNING, logger="agents.benchmarks.bfcl_v4"):
        tasks = plugin.load_tasks()
    # Only the supported row survives.
    assert len(tasks) == 1
    assert tasks[0]["instance_id"] == "s0"
    # The warning names the unknown category + count explicitly.
    warnings = "\n".join(rec.message for rec in caplog.records)
    assert "UNKNOWN categories" in warnings
    assert "brand_new_category_upstream=2" in warnings


def test_normalize_task_is_idempotent(plugin: BFCLv4Benchmark) -> None:
    """normalize_task applied twice must produce the same result as once."""
    # collector hoists normalize_task before dispatch; from_benchmark_instance calls it again
    raw = {
        "id": "simple_0",
        "category": "simple_python",
        "question": [[{"role": "user", "content": "What is 2+2?"}]],
        "function": [{"name": "add", "description": "", "parameters": {}}],
        "ground_truth": [{"add": {"a": [2], "b": [2]}}],
    }
    once = plugin.normalize_task(raw)
    twice = plugin.normalize_task(once)
    assert once == twice, "normalize_task is not idempotent"


# ── select_subset ──────────────────────────────────────────────────────


def test_select_subset_stratified_across_categories(
    plugin: BFCLv4Benchmark,
) -> None:
    tasks = []
    for cat in ("simple", "multiple", "parallel"):
        for i in range(10):
            tasks.append({"instance_id": f"{cat}_{i:02d}", "category": cat})

    selected = plugin.select_subset(tasks, n=9, seed=None)
    assert len(selected) == 9
    counts: dict[str, int] = {}
    for t in selected:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    # Proportional: 10/30 each → 3 per category.
    assert counts == {"simple": 3, "multiple": 3, "parallel": 3}
    # Deterministic ordering: (category, instance_id).
    assert selected == sorted(
        selected, key=lambda t: (t["category"], t["instance_id"])
    )


def test_select_subset_all_when_n_exceeds_total(
    plugin: BFCLv4Benchmark,
) -> None:
    tasks = [{"instance_id": f"s_{i}", "category": "simple"} for i in range(5)]
    selected = plugin.select_subset(tasks, n=100)
    assert len(selected) == 5


# ── SWE-bench-specific methods are disabled ───────────────────────────


def test_derive_test_cmd_raises(plugin: BFCLv4Benchmark) -> None:
    with pytest.raises(NotImplementedError, match="AST comparison"):
        plugin.derive_test_cmd({"instance_id": "x"})


def test_build_harness_args_raises(plugin: BFCLv4Benchmark, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="SWE-bench harness"):
        plugin.build_harness_args(
            predictions_path=tmp_path / "preds.json",
            run_id="test",
        )


def test_image_name_for_returns_none(plugin: BFCLv4Benchmark) -> None:
    assert plugin.image_name_for({"instance_id": "x"}) is None


# ── build_runner scaffold compatibility ───────────────────────────────


def test_build_runner_refuses_mini_swe_agent(plugin: BFCLv4Benchmark) -> None:
    with pytest.raises(
        NotImplementedError,
        match="mini-swe-agent.*bash-in-repo|function_call.*incompatible",
    ):
        plugin.build_runner(
            scaffold="mini-swe-agent",
            provider=object(),
            workspace_base=Path("/tmp/ws"),
            max_iterations=10,
            context_window_tokens=8192,
            model="test/model",
        )


def test_build_runner_refuses_unknown_scaffold(plugin: BFCLv4Benchmark) -> None:
    with pytest.raises(NotImplementedError, match="does not support scaffold"):
        plugin.build_runner(
            scaffold="bogus",
            provider=object(),
            workspace_base=Path("/tmp/ws"),
            max_iterations=10,
            context_window_tokens=8192,
            model="test/model",
        )


def test_build_runner_openclaw_forward_imports_bfcl_runner(
    plugin: BFCLv4Benchmark,
) -> None:
    """build_runner(scaffold='openclaw') should lazy-import BFCLRunner.

    Since BFCLRunner is implemented in US-006, this test uses
    pytest.importorskip so it becomes live once the runner file exists
    without blocking the current US-003 acceptance.
    """
    pytest.importorskip("agents.benchmarks.bfcl_runner")
    from agents.benchmarks.bfcl_runner import BFCLRunner

    runner = plugin.build_runner(
        scaffold="openclaw",
        provider=object(),
        workspace_base=Path("/tmp/ws"),
        max_iterations=10,
        context_window_tokens=8192,
        model="test/model",
    )
    assert isinstance(runner, BFCLRunner)
