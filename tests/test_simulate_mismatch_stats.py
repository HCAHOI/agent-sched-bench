from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.simulate_mismatch_stats import _output_diff_signature, main


def test_output_diff_signature_uses_replay_line_for_insertions() -> None:
    assert _output_diff_signature("  same\n- <missing>\n+ pid 456") == "+ pid <N>"
    assert _output_diff_signature(
        "raw output differs without line-content difference"
    ) == "raw output differs without line-content difference"


def test_simulate_mismatch_stats_reports_output_diff_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_trace = tmp_path / "source.jsonl"
    source_trace.write_text("", encoding="utf-8")
    trace_path = tmp_path / "simulate.jsonl"
    records = [
        {"type": "trace_metadata", "source_traces": [str(source_trace)]},
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "replay_outcome_match": False,
                "mismatch_reason": "command_output_mismatch",
                "output_diff_snippet": "- pid 123\n+ pid 456",
            },
        },
        {"type": "summary", "elapsed_s": 1.0, "unresolved_mismatches": 1},
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["simulate_mismatch_stats.py", str(trace_path)])
    main()

    output = capsys.readouterr().out
    assert "Output diff patterns (top 10):" in output
    assert "[1x] pid <N> -> pid <N>" in output
