"""Smoke test: Gantt builder against a synthetic heterogeneous v5 trace.

Historically this module also loaded the most recent real OpenClaw CLI
trace from ``traces/openclaw_cli/`` as a smoke check. That test was
removed during the v4 → v5 hard cutover (SWE-rebench plugin refactor):
every on-disk trace predated the cutover and is therefore pre-v5 and
unsupported. Once a fresh v5 openclaw session has been run, a new
real-trace smoke test can be re-added without any backward-compat
machinery.
"""

from __future__ import annotations

from pathlib import Path

from trace_collect.gantt_data import build_gantt_payload_multi
from trace_collect.trace_inspector import TraceData


def test_gantt_payload_handles_empty_trace_dir(tmp_path: Path) -> None:
    """Defensive: a TraceData with zero actions should still produce a
    valid (empty) payload, not crash."""
    empty_trace = tmp_path / "empty.jsonl"
    empty_trace.write_text(
        '{"type":"trace_metadata","scaffold":"empty","trace_format_version":5}\n'
    )
    data = TraceData.load(empty_trace)
    payload = build_gantt_payload_multi([("empty", data)])
    assert payload["traces"][0]["metadata"]["n_actions"] == 0
    assert payload["traces"][0]["metadata"]["n_iterations"] == 0
