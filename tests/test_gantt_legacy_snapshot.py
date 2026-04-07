"""Legacy gantt payload snapshot regression test.

The SWE-rebench plugin refactor (Phase 1 → 5) touches the benchmark plugin
layer, the trace metadata schema (v4 → v5 bump), and the gantt display-label
resolution. None of these should change the structural shape of the gantt
payload for a pre-refactor openclaw trace.

This test captures a snapshot of ``build_gantt_payload_multi`` output for a
real v4 openclaw trace (``oc-60184637.jsonl`` from the Pacman session),
saves it to ``tests/fixtures/legacy_gantt_payload.json``, and asserts the
post-refactor builder produces a byte-identical payload for the same input.

If this test fails, either:
  (a) the gantt_data.py refactor legitimately changed the payload shape —
      in which case regenerate the fixture deliberately with a commit that
      notes WHY, or
  (b) the refactor accidentally drifted — which is the exact regression
      this test is designed to catch.

Architect + Critic (ralplan iteration 0) required this snapshot because
``test_gantt_builder_parity.py`` only compares Python vs JS for the same
input — it cannot detect pre- vs post-refactor drift on the Python side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_collect.gantt_data import build_gantt_payload_multi
from trace_collect.trace_inspector import TraceData

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "legacy_gantt_payload.json"
LEGACY_TRACE = (
    REPO_ROOT
    / "traces"
    / "openclaw_cli"
    / "qwen_qwen3.6-plus_free"
    / "20260407T054101Z"
    / "oc-60184637.jsonl"
)


@pytest.mark.skipif(
    not LEGACY_TRACE.exists(),
    reason=f"legacy trace fixture missing at {LEGACY_TRACE}",
)
def test_legacy_payload_byte_identical() -> None:
    """Post-refactor builder must produce the same payload for the same input."""
    assert FIXTURE_PATH.exists(), (
        f"fixture missing at {FIXTURE_PATH}; regenerate via:\n"
        f"  python -c \"from trace_collect.gantt_data import build_gantt_payload_multi; "
        f"from trace_collect.trace_inspector import TraceData; import json; "
        f"t = TraceData.load('{LEGACY_TRACE}'); "
        f"print(json.dumps(build_gantt_payload_multi([('legacy_oc', t)]), indent=2, sort_keys=True))\" "
        f"> {FIXTURE_PATH}"
    )

    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    trace = TraceData.load(LEGACY_TRACE)
    actual = build_gantt_payload_multi([("legacy_oc", trace)])

    # Serialize both the same way (sorted keys) so dict-ordering noise
    # doesn't cause spurious diffs.
    expected_norm = json.dumps(expected, sort_keys=True)
    actual_norm = json.dumps(actual, sort_keys=True)
    assert actual_norm == expected_norm, (
        "Legacy gantt payload drifted from snapshot. "
        "If the refactor intentionally changed the shape, regenerate the "
        "fixture in a dedicated commit that documents the change."
    )
