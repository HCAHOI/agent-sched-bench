from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


def _evict(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _isolate_modules():
    _evict("trace_collect.scaffold_registry", "agents.openclaw", "agents.miniswe")
    yield
    _evict("trace_collect.scaffold_registry", "agents.openclaw", "agents.miniswe")


def test_simulator_rejects_openclaw_trace_without_repo_fields(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_001",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_001",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps([{"instance_id": "fc_test_001", "problem_statement": "x"}]) + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulator import simulate

    with pytest.raises(
        NotImplementedError,
        match="Simulate mode requires repo-backed OpenClaw tasks",
    ):
        asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                repos_root=tmp_path / "repos",
                output_dir=tmp_path / "out",
                api_base="http://localhost:8000/v1",
                api_key="EMPTY",
                model="dummy",
            )
        )

    assert "agents.openclaw" not in sys.modules


def test_simulator_rejects_openclaw_trace_marked_non_prepareable(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_001",
                        "model": "dummy",
                        "mode": "collect",
                        "needs_prepare": False,
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_001",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "fc_test_001",
                    "problem_statement": "x",
                    "repo": "django/django",
                    "base_commit": "deadbeef",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulator import simulate

    with pytest.raises(
        NotImplementedError,
        match="Simulate mode requires repo-backed OpenClaw traces",
    ):
        asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                repos_root=tmp_path / "repos",
                output_dir=tmp_path / "out",
                api_base="http://localhost:8000/v1",
                api_key="EMPTY",
                model="dummy",
            )
        )

    assert "agents.openclaw" not in sys.modules
