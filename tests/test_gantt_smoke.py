"""Smoke test: Gantt builder against a real heterogeneous v4 trace.

Loads the most recent OpenClaw CLI trace (produced by the Pacman playground
session) and verifies that the Gantt payload builder handles it without
crashing and emits the expected span types.

Skipped (not failed) when no trace is present, so the test suite stays
green on a fresh checkout. Run the OpenClaw CLI first to populate
``traces/openclaw_cli/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_collect.gantt_data import build_gantt_payload_multi
from trace_collect.trace_inspector import TraceData


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENCLAW_TRACES = REPO_ROOT / "traces" / "openclaw_cli"


def _latest_openclaw_trace() -> Path | None:
    """Find the most recent COMPLETE OpenClaw CLI trace, if any.

    A trace is considered complete when it contains a ``summary`` record —
    in-progress traces (the openclaw process is still writing) have no
    summary yet and would only have partial action coverage. Skipping them
    keeps the smoke test deterministic when run while a session is active.
    """
    if not OPENCLAW_TRACES.is_dir():
        return None
    candidates = sorted(
        OPENCLAW_TRACES.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open() as fh:
                if any('"type": "summary"' in line or '"type":"summary"' in line
                       for line in fh):
                    return path
        except OSError:
            continue
    return None


def test_gantt_payload_from_real_openclaw_trace() -> None:
    trace = _latest_openclaw_trace()
    if trace is None:
        pytest.skip(
            f"No OpenClaw CLI trace found under {OPENCLAW_TRACES}. "
            "Run `python -m agents.openclaw --prompt ...` first."
        )

    data = TraceData.load(trace)
    assert len(data.actions) > 0, f"Loaded zero actions from {trace}"

    payload = build_gantt_payload_multi([("openclaw_pacman", data)])

    # Payload structure invariants
    assert "registries" in payload
    assert "spans" in payload["registries"]
    assert "markers" in payload["registries"]
    assert len(payload["traces"]) == 1

    tp = payload["traces"][0]
    assert len(tp["lanes"]) >= 1, "Expected at least one lane"

    all_spans = [s for lane in tp["lanes"] for s in lane["spans"]]
    assert len(all_spans) > 0, "Expected at least one span"

    span_types = {s["type"] for s in all_spans}
    # OpenClaw should produce both LLM calls and tool exec actions
    assert "llm" in span_types, f"No llm spans found. Types: {span_types}"
    assert "tool" in span_types, f"No tool spans found. Types: {span_types}"

    # v4 metadata vocabulary
    meta = tp["metadata"]
    assert "n_actions" in meta
    assert "n_iterations" in meta
    assert "n_steps" not in meta
    assert meta["n_actions"] == len(data.actions)


def test_gantt_payload_handles_empty_trace_dir(tmp_path: Path) -> None:
    """Defensive: a TraceData with zero actions should still produce a
    valid (empty) payload, not crash."""
    empty_trace = tmp_path / "empty.jsonl"
    empty_trace.write_text(
        '{"type":"trace_metadata","scaffold":"empty","trace_format_version":4}\n'
    )
    data = TraceData.load(empty_trace)
    payload = build_gantt_payload_multi([("empty", data)])
    assert payload["traces"][0]["metadata"]["n_actions"] == 0
    assert payload["traces"][0]["metadata"]["n_iterations"] == 0
