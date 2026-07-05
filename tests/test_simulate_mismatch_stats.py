from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.simulate_mismatch_stats import _output_diff_signature, main
from trace_collect.output_normalize import normalize_tool_output


def test_normalize_tool_output_general_runtime_volatility() -> None:
    source = (
        "pid 123 at 2026-07-05T12:34:56Z epoch 1782470400 "
        "addr 0x7ffdeadbeef tmp /tmp/run-456/a.txt proc /proc/789/status"
    )
    replay = (
        "pid 999 at 2027-08-06T01:02:03Z epoch 1782470999 "
        "addr 0xabc tmp /tmp/run-000/a.txt proc /proc/111/status"
    )

    assert normalize_tool_output(source) == normalize_tool_output(replay)
    assert normalize_tool_output(source) == (
        "pid <N> at <TS> epoch <TS> addr <HEX> tmp <TMP> proc <PROC>"
    )


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
                "normalized_output_match": False,
                "cas_mode_mismatch_count": 2,
                "forced_sync_attempted": True,
                "forced_sync_success": True,
                "forced_sync_verified": True,
                "forced_sync_reapplied_action_count": 2,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "replay_outcome_match": True,
                "normalized_output_match": True,
                "cas_mode_mismatch_count": 0,
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
    assert (
        "  Match rate among exit-code-matching actions: 1/2 = 50.0%" in output
    )
    assert "CAS mode comparison:" in output
    assert "  Mode mismatches:  2 across 1 actions" in output
    assert "  Verified:        1" in output
    assert "  Reapplied acts:  2" in output


def test_simulate_mismatch_stats_report_compares_oracle_and_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_trace = tmp_path / "source.jsonl"
    source_trace.write_text("", encoding="utf-8")
    trace_path = tmp_path / "simulate.jsonl"
    labels_path = tmp_path / "labels.csv"
    records = [
        {"type": "trace_metadata", "source_traces": [str(source_trace)]},
        _tool_record(
            "cosmetic-raw-mismatch",
            {
                "tool_name": "exec",
                "replay_outcome_match": False,
                "mismatch_reason": "command_output_mismatch",
                "source_returncode": 0,
                "replay_returncode": 0,
                "source_timed_out": False,
                "replay_timed_out": False,
                "normalized_output_match": True,
            },
        ),
        _tool_record(
            "output-content",
            {
                "tool_name": "exec",
                "replay_outcome_match": True,
                "source_returncode": 0,
                "replay_returncode": 0,
                "source_timed_out": False,
                "replay_timed_out": False,
                "normalized_output_match": False,
                "output_diff_snippet": "- alpha\n+ beta",
            },
        ),
        _tool_record(
            "exit-content",
            {
                "tool_name": "exec",
                "replay_outcome_match": False,
                "mismatch_reason": "command_exit_code_mismatch",
                "source_returncode": 0,
                "replay_returncode": 1,
                "source_timed_out": False,
                "replay_timed_out": False,
                "normalized_output_match": True,
            },
        ),
        _tool_record(
            "cas-content",
            {
                "tool_name": "exec",
                "replay_outcome_match": False,
                "mismatch_reason": "cas_state_mismatch",
                "source_returncode": 0,
                "replay_returncode": 0,
                "source_timed_out": False,
                "replay_timed_out": False,
                "normalized_output_match": True,
                "cas_manifest_match": False,
                "cas_modified_count": 1,
                "cas_removed_count": 0,
                "cas_added_count": 0,
            },
        ),
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "\n".join(
            [
                "source_action_id,human_label_different",
                "cosmetic-raw-mismatch,false",
                "output-content,true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_mismatch_stats.py",
            str(trace_path),
            "--report",
            "--labels",
            str(labels_path),
        ],
    )
    main()

    output = capsys.readouterr().out
    assert "TIERED SEMANTIC ORACLE REPORT" in output
    assert (
        "all compared actions: agreement 2/4 = 50.0%; raw match "
        "1/4 = 25.0%; oracle match 1/4 = 25.0%"
    ) in output
    assert "exit-code-matched: agreement 1/3 = 33.3%" in output
    assert "exit-code-mismatched: agreement 1/1 = 100.0%" in output
    assert "  Tier 1: 1" in output
    assert "  Tier 2: 1" in output
    assert "    cosmetic: 1" in output
    assert "  Tier 3: 1" in output
    assert "  oracle: FP 0/2 = 0.0%; FN 0/2 = 0.0%" in output
    assert "  raw replay_outcome_match: FP 1/2 = 50.0%; FN 1/2 = 50.0%" in output


def _tool_record(action_id: str, data: dict[str, object]) -> dict[str, object]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "agent_id": "agent-a",
        "data": data,
    }
