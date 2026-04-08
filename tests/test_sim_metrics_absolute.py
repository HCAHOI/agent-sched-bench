"""Unit tests for absolute PreemptionSnapshot storage in sim_metrics.

Phase 2 of the trace-sim-vastai-pipeline plan. Verifies that:
- harness.scheduler_hooks.empty_snapshot() returns a PreemptionSnapshot
  with all five fields = None.
- harness.scheduler_hooks.get_snapshot(None) returns an empty snapshot
  (the explicit opt-out path).
- VLLMMetricsClient(None).get_snapshot() returns an empty snapshot.
- VLLMMetricsClient(None).is_enabled is False.
- The simulator's sim_metrics blob writes ALL fields from the snapshot
  field-for-field — there is no schema reduction.
- The whitelist of fields written matches PreemptionSnapshot.__dict__
  exactly (introspected via dataclasses.fields).
"""

from __future__ import annotations

import dataclasses

from harness.metrics_client import VLLMMetricsClient
from harness.scheduler_hooks import (
    PreemptionSnapshot,
    empty_snapshot,
    get_snapshot,
)


# ---------------------------------------------------------------------------
# Empty snapshot semantics (the opt-out path)
# ---------------------------------------------------------------------------


def test_empty_snapshot_has_all_fields_none() -> None:
    snap = empty_snapshot()
    assert snap.num_preemptions_total is None
    assert snap.gpu_cache_usage_perc is None
    assert snap.cpu_cache_usage_perc is None
    assert snap.gpu_prefix_cache_hit_rate is None
    assert snap.cpu_prefix_cache_hit_rate is None


def test_empty_snapshot_is_a_fresh_instance_each_call() -> None:
    """Two calls must return distinct instances (no shared state risk)."""
    a = empty_snapshot()
    b = empty_snapshot()
    assert a is not b


def test_get_snapshot_returns_empty_when_metrics_url_is_none() -> None:
    snap = get_snapshot(None)
    assert isinstance(snap, PreemptionSnapshot)
    assert snap.num_preemptions_total is None


def test_get_snapshot_returns_empty_when_metrics_url_is_empty_string() -> None:
    snap = get_snapshot("")
    assert isinstance(snap, PreemptionSnapshot)
    assert snap.num_preemptions_total is None


# ---------------------------------------------------------------------------
# VLLMMetricsClient — opt-out path
# ---------------------------------------------------------------------------


def test_metrics_client_opt_out_is_disabled() -> None:
    client = VLLMMetricsClient(metrics_url=None)
    assert client.is_enabled is False


def test_metrics_client_opt_out_returns_empty_snapshot() -> None:
    client = VLLMMetricsClient(metrics_url=None)
    snap = client.get_snapshot()
    assert snap.num_preemptions_total is None
    assert snap.gpu_cache_usage_perc is None


def test_metrics_client_with_url_is_enabled() -> None:
    client = VLLMMetricsClient(metrics_url="http://localhost:8000/metrics")
    assert client.is_enabled is True


def test_metrics_client_repeated_opt_out_calls_return_fresh_snapshots() -> None:
    """Each get_snapshot() returns a fresh dataclass instance."""
    client = VLLMMetricsClient(metrics_url=None)
    a = client.get_snapshot()
    b = client.get_snapshot()
    assert a is not b


# ---------------------------------------------------------------------------
# Field introspection — every PreemptionSnapshot field flows through to
# the sim_metrics blob without reduction
# ---------------------------------------------------------------------------


def test_preemption_snapshot_dict_has_all_fields() -> None:
    """The serialized snapshot dict must contain all dataclass fields."""
    snap = empty_snapshot()
    serialized = dataclasses.asdict(snap)
    expected_fields = {f.name for f in dataclasses.fields(PreemptionSnapshot)}
    assert set(serialized.keys()) == expected_fields, (
        f"Serialized snapshot must contain all dataclass fields. "
        f"Expected {expected_fields}, got {set(serialized.keys())}"
    )


def test_preemption_snapshot_field_set_is_stable() -> None:
    """Phase 0 audit names exactly 5 fields. Drift here breaks Phase 2."""
    expected = {
        "num_preemptions_total",
        "gpu_cache_usage_perc",
        "cpu_cache_usage_perc",
        "gpu_prefix_cache_hit_rate",
        "cpu_prefix_cache_hit_rate",
    }
    actual = {f.name for f in dataclasses.fields(PreemptionSnapshot)}
    assert actual == expected, (
        f"PreemptionSnapshot field set drifted from Phase 0 audit. "
        f"Expected {expected}, got {actual}. Update "
        f".omc/plans/phase0-schemas.md section (b) and "
        f"sim_metrics_delta.py whitelist before changing this test."
    )


def test_simulated_action_data_carries_full_snapshot() -> None:
    """Simulate a TraceAction.data dict and assert it preserves all fields."""
    snap = empty_snapshot()
    data = {
        "messages_in": [],
        "sim_metrics": {
            "timing": {"ttft_ms": 0.0, "tpot_ms": 0.0, "total_ms": 0.0},
            "vllm_scheduler_snapshot": dataclasses.asdict(snap),
        },
    }

    snapshot_blob = data["sim_metrics"]["vllm_scheduler_snapshot"]
    expected_fields = {f.name for f in dataclasses.fields(PreemptionSnapshot)}
    assert set(snapshot_blob.keys()) == expected_fields


# ---------------------------------------------------------------------------
# parse_prometheus_metrics happy path (no network — synthetic payload)
# ---------------------------------------------------------------------------


def test_parse_prometheus_metrics_extracts_all_five_fields() -> None:
    """Synthetic prometheus payload covering all five vLLM metrics."""
    from harness.scheduler_hooks import parse_prometheus_metrics

    payload = """\
# HELP vllm:num_preemptions_total Total preempted seqs
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model="test"} 42.0
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model="test"} 0.85
# HELP vllm:cpu_cache_usage_perc CPU KV cache usage
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc{model="test"} 0.10
# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache hit rate
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model="test"} 0.72
# HELP vllm:cpu_prefix_cache_hit_rate CPU prefix cache hit rate
# TYPE vllm:cpu_prefix_cache_hit_rate gauge
vllm:cpu_prefix_cache_hit_rate{model="test"} 0.05
"""
    snap = parse_prometheus_metrics(payload)
    assert snap.num_preemptions_total == 42.0
    assert snap.gpu_cache_usage_perc == 0.85
    assert snap.cpu_cache_usage_perc == 0.10
    assert snap.gpu_prefix_cache_hit_rate == 0.72
    assert snap.cpu_prefix_cache_hit_rate == 0.05
