from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.benchmarks._research import HostResearchOpenClawRunner
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.deep_research_bench import DeepResearchBenchBenchmark
from trace_collect.attempt_pipeline import AttemptContext


_DATASET = "muset-ai/DeepResearch-Bench-Dataset"
_REVISION = "deadbeef1234"
_DATA_FILE = "generated_reports/openai-deepresearch.jsonl"
_SCHEMA = ["article", "id", "prompt", "topic"]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _make_config(
    tmp_path: Path,
    *,
    data_sha256: str,
    expected_schema: list[str] | None = None,
    expected_row_count: int = 2,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="deep-research-bench",
        display_name="DeepResearchBench",
        trace_root=tmp_path / "traces",
        default_max_iterations=100,
        selection_n=2,
        selection_seed=42,
        harness_dataset=_DATASET,
        harness_split="test",
        default_prompt_template="default",
        extras={
            "data_files": _DATA_FILE,
            "revision": _REVISION,
            "data_sha256": data_sha256,
            "expected_schema": list(_SCHEMA if expected_schema is None else expected_schema),
            "expected_row_count": expected_row_count,
            "id_field": "id",
            "question_field": "prompt",
            "answer_field": "article",
            "reference_kind": "generated_report",
            "topic_field": "topic",
            "difficulty_field": None,
            "domain_field": None,
        },
    )


@pytest.mark.parametrize(
    "missing_key",
    [
        "data_files",
        "revision",
        "data_sha256",
        "expected_schema",
        "expected_row_count",
    ],
)
def test_deep_research_bench_config_requires_provenance_pins(
    tmp_path: Path,
    missing_key: str,
) -> None:
    config = _make_config(tmp_path, data_sha256="0" * 64)
    del config.extras[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        DeepResearchBenchBenchmark(config)


def _install_fake_hf_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]],
    payload: bytes,
) -> dict[str, list[dict[str, Any]]]:
    downloaded = tmp_path / "downloaded_deep_research_bench.jsonl"
    downloaded.write_bytes(payload)
    calls: dict[str, list[dict[str, Any]]] = {"download": [], "load": []}

    def hf_hub_download(**kwargs: Any) -> str:
        calls["download"].append(dict(kwargs))
        return str(downloaded)

    def load_dataset(dataset: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls["load"].append({"dataset": dataset, **kwargs})
        return [dict(row) for row in rows]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=hf_hub_download))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_deep_research_bench_load_tasks_pins_source_and_propagates_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": "drb-001",
            "prompt": "Which observatory first detected the sentinel signal?",
            "article": "The Vera Rubin Observatory detected the signal first.",
            "topic": "astronomy",
        },
        {
            "id": "drb-002",
            "prompt": "What policy changed after the sentinel report?",
            "article": "The review board required public provenance for all reports.",
            "topic": "policy",
        },
    ]
    payload = _jsonl_bytes(rows)
    sha256 = hashlib.sha256(payload).hexdigest()
    calls = _install_fake_hf_modules(
        monkeypatch,
        tmp_path,
        rows=rows,
        payload=payload,
    )
    plugin = DeepResearchBenchBenchmark(_make_config(tmp_path, data_sha256=sha256))

    tasks = plugin.load_tasks()

    assert [task["instance_id"] for task in tasks] == ["drb-001", "drb-002"]
    assert [task["problem_statement"] for task in tasks] == [
        "Which observatory first detected the sentinel signal?",
        "What policy changed after the sentinel report?",
    ]
    for task in tasks:
        assert task["task_source_kind"] == "huggingface_dataset"
        assert task["task_source_id"] == f"{_DATASET}@{_REVISION}"
        assert task["task_source_path"] == _DATA_FILE
        assert task["task_source_sha256"] == sha256
        assert task["task_source_schema"] == sorted(_SCHEMA)
        assert task["task_source_row_count"] == 2
    assert calls["download"] == [
        {
            "repo_id": _DATASET,
            "repo_type": "dataset",
            "filename": _DATA_FILE,
            "revision": _REVISION,
        }
    ]
    assert calls["load"] == [
        {
            "dataset": _DATASET,
            "split": "test",
            "revision": _REVISION,
            "data_files": {"test": _DATA_FILE},
        }
    ]

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        DeepResearchBenchBenchmark(
            _make_config(tmp_path, data_sha256="0" * 64)
        ).load_tasks()

    with pytest.raises(ValueError, match="row count mismatch"):
        DeepResearchBenchBenchmark(
            _make_config(tmp_path, data_sha256=sha256, expected_row_count=3)
        ).load_tasks()

    with pytest.raises(ValueError, match="schema mismatch"):
        DeepResearchBenchBenchmark(
            _make_config(
                tmp_path,
                data_sha256=sha256,
                expected_schema=["article", "id", "prompt"],
            )
        ).load_tasks()


class _CompletedSessionRunner:
    async def run(self, **kwargs: Any) -> SimpleNamespace:
        trace_file = Path(kwargs["trace_file"])
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            json.dumps(
                {
                    "type": "summary",
                    "n_iterations": 1,
                    "total_llm_ms": 25.0,
                    "total_tool_ms": 0.0,
                    "total_tokens": 42,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            content="Exact Answer: The public report is complete.\nConfidence: 100%",
            elapsed_s=0.1,
            trace_file=trace_file,
            stop_reason="completed",
            error=None,
        )


def _attempt_ctx(tmp_path: Path, task: dict[str, Any]) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id=str(task["instance_id"]),
        attempt=1,
        task=task,
        model="fake-openclaw-model",
        scaffold="openclaw",
        source_image=None,
        prompt_template="default",
        agent_runtime_mode="host_controller",
        execution_environment="host",
    )


def test_openclaw_drb_reference_task_fails_closed_when_grader_is_unavailable(
    tmp_path: Path,
) -> None:
    task = {
        "instance_id": "drb-graded-reference",
        "problem_statement": "Which public report should be checked?",
        "reference_answer": "The reference report is deliberately unavailable to OpenClaw.",
    }
    runner = HostResearchOpenClawRunner.__new__(HostResearchOpenClawRunner)
    runner.workspace_base = tmp_path / "workspaces"
    runner.model = "fake-openclaw-model"
    runner.benchmark_slug = "deep-research-bench"
    runner.mcp_config = None
    runner._session_runner = _CompletedSessionRunner()
    runner._render_prompt = lambda task, *, prompt_template: "rendered prompt"

    result = asyncio.run(
        runner.run_task(
            task,
            attempt_ctx=_attempt_ctx(tmp_path, task),
            prompt_template="default",
        )
    )

    assert result.success is False
    assert result.exit_status == "completed"
    assert result.error is None
    assert result.summary["correct"] is False
    assert result.summary["score"] == 0
    assert result.summary["grader_status"] == "not_run_no_grader"
    assert result.summary["answer_parse_error"] == "grader_unavailable"
    assert result.n_iterations == 1
    assert result.total_llm_ms == 25.0
    assert result.total_tool_ms == 0.0
    assert result.total_tokens == 42
