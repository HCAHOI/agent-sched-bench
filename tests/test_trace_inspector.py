"""Unit tests for canonical trace inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path

from trace_collect.trace_inspector import TraceData, cmd_overview


def test_cmd_overview_counts_distinct_iterations(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "openclaw",
                        "trace_format_version": 5,
                        "model": "test-model",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "tool_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 2.0,
                        "ts_end": 2.1,
                        "data": {
                            "tool_name": "bash",
                            "duration_ms": 100.0,
                            "success": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_1",
                        "agent_id": "agent-1",
                        "iteration": 1,
                        "ts_start": 3.0,
                        "ts_end": 4.0,
                        "data": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_overview(TraceData.load(trace_path), as_json=True)
    output = json.loads(capsys.readouterr().out)
    assert output["n_iterations"] == 2
