"""Unit tests for analysis.sim_metrics_delta.

Phase 2 of the trace-sim-vastai-pipeline plan. Verifies that:
- compute_preemption_delta produces correct consecutive deltas for the
  num_preemptions_total counter.
- Empty input returns empty list.
- Partial-data inputs (missing snapshots, None counters) skip cleanly.
- compute_field_delta and the underlying _assert_field_is_counter helper
  raise TypeError for any gauge or ratio field.
- The whitelist is sourced from the documented field type tags
  (counter only) — verified by introspecting PreemptionSnapshot fields.
"""

from __future__ import annotations

import dataclasses

import pytest

from analysis.sim_metrics_delta import (
    _DELTA_VALID_FIELDS,
    compute_field_delta,
    compute_preemption_delta,
)
from harness.scheduler_hooks import PreemptionSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action_with_counter(value: float | None) -> dict:
    """Synthesize a TraceAction-shaped dict with only the counter field set."""
    return {
        "type": "action",
        "action_type": "llm_call",
        "data": {
            "sim_metrics": {
                "vllm_scheduler_snapshot": {
                    "num_preemptions_total": value,
                    "gpu_cache_usage_perc": None,
                    "cpu_cache_usage_perc": None,
                    "gpu_prefix_cache_hit_rate": None,
                    "cpu_prefix_cache_hit_rate": None,
                }
            }
        },
    }


def _action_no_sim_metrics() -> dict:
    """Synthesize a TraceAction with no sim_metrics blob (legacy v5 record)."""
    return {
        "type": "action",
        "action_type": "llm_call",
        "data": {"prompt_tokens": 100, "completion_tokens": 50},
    }


# ---------------------------------------------------------------------------
# Counter delta correctness
# ---------------------------------------------------------------------------


def test_compute_preemption_delta_canonical_sequence() -> None:
    """[0, 2, 2, 5, 5] → [2, 0, 3, 0]."""
    actions = [_action_with_counter(v) for v in [0, 2, 2, 5, 5]]
    assert compute_preemption_delta(actions) == [2, 0, 3, 0]


def test_compute_preemption_delta_empty_input() -> None:
    assert compute_preemption_delta([]) == []


def test_compute_preemption_delta_single_action_returns_empty() -> None:
    """Single action has nothing to subtract from."""
    actions = [_action_with_counter(7)]
    assert compute_preemption_delta(actions) == []


def test_compute_preemption_delta_skips_none_counters() -> None:
    """None values are skipped — the delta list shrinks accordingly."""
    actions = [
        _action_with_counter(0),
        _action_with_counter(None),  # skipped
        _action_with_counter(5),
    ]
    # After skipping the middle, the counters list is [0, 5] → delta [5]
    assert compute_preemption_delta(actions) == [5]


def test_compute_preemption_delta_skips_actions_without_sim_metrics() -> None:
    """Legacy actions (no sim_metrics) are silently ignored."""
    actions = [
        _action_with_counter(10),
        _action_no_sim_metrics(),
        _action_with_counter(13),
    ]
    assert compute_preemption_delta(actions) == [3]


def test_compute_preemption_delta_monotonic_growing_counter() -> None:
    """A typical real-world growing counter sequence."""
    actions = [_action_with_counter(v) for v in [0, 1, 3, 6, 10]]
    assert compute_preemption_delta(actions) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Gauge / ratio field rejection
# ---------------------------------------------------------------------------


def test_compute_field_delta_rejects_gauge_field() -> None:
    actions = [_action_with_counter(0)]
    with pytest.raises(TypeError) as exc_info:
        compute_field_delta(actions, "gpu_cache_usage_perc")
    msg = str(exc_info.value)
    assert "gpu_cache_usage_perc" in msg
    assert "gauge/ratio" in msg.lower() or "gauge" in msg.lower()


def test_compute_field_delta_rejects_ratio_field() -> None:
    actions = [_action_with_counter(0)]
    with pytest.raises(TypeError) as exc_info:
        compute_field_delta(actions, "gpu_prefix_cache_hit_rate")
    assert "gpu_prefix_cache_hit_rate" in str(exc_info.value)


def test_compute_field_delta_rejects_cpu_gauge() -> None:
    actions = [_action_with_counter(0)]
    with pytest.raises(TypeError):
        compute_field_delta(actions, "cpu_cache_usage_perc")


def test_compute_field_delta_rejects_cpu_ratio() -> None:
    actions = [_action_with_counter(0)]
    with pytest.raises(TypeError):
        compute_field_delta(actions, "cpu_prefix_cache_hit_rate")


def test_compute_field_delta_rejects_unknown_field() -> None:
    """Unknown field name → ValueError listing valid fields."""
    actions = [_action_with_counter(0)]
    with pytest.raises(ValueError) as exc_info:
        compute_field_delta(actions, "totally_made_up_field")
    msg = str(exc_info.value)
    assert "totally_made_up_field" in msg
    assert "PreemptionSnapshot" in msg


def test_compute_field_delta_accepts_whitelisted_counter() -> None:
    """num_preemptions_total IS in the whitelist; should not raise."""
    actions = [_action_with_counter(v) for v in [1, 4]]
    assert compute_field_delta(actions, "num_preemptions_total") == [3]


# ---------------------------------------------------------------------------
# Whitelist coherence — guards against accidental drift
# ---------------------------------------------------------------------------


def test_delta_whitelist_is_subset_of_preemption_snapshot_fields() -> None:
    """Every name in _DELTA_VALID_FIELDS must be a real PreemptionSnapshot field."""
    snapshot_fields = {f.name for f in dataclasses.fields(PreemptionSnapshot)}
    assert _DELTA_VALID_FIELDS.issubset(snapshot_fields), (
        f"_DELTA_VALID_FIELDS contains names not in PreemptionSnapshot: "
        f"{_DELTA_VALID_FIELDS - snapshot_fields}"
    )


def test_delta_whitelist_contains_only_counter() -> None:
    """Per Phase 0 schema audit, only num_preemptions_total is a counter."""
    assert _DELTA_VALID_FIELDS == frozenset({"num_preemptions_total"}), (
        "If the whitelist changes, .omc/plans/phase0-schemas.md section (b) "
        "MUST be updated with the rationale BEFORE the test is changed. "
        "See sim_metrics_delta.py module docstring."
    )
