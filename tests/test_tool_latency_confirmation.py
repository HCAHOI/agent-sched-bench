from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap
from trace_collect.tool_latency_dataset import (
    read_tool_latency_corpus_manifest,
    require_explicit_trace_task_ids,
)


_TEN_COSTS_MS = [float(cost) for cost in range(500, 5_001, 500)]


def test_task_cluster_bootstrap_preserves_repeated_call_contributions() -> None:
    decisions = [
        _decision("a-1", "task-a", latency_ms=80.0),
        _decision("a-2", "task-a", latency_ms=80.0),
        _decision("b-1", "task-b", latency_ms=150.0),
    ]

    result = paired_task_cluster_bootstrap(
        decisions,
        costs_ms=[100.0],
        replicates=50_000,
        confidence_level=0.95,
        seed=0,
    )

    assert result["sample_count"] == 3
    assert result["task_count"] == 2
    point = result["points"]["100.0"]
    assert point["paired_delta_ms"] == -60.0
    assert point["pointwise_interval_ms"] == {"low": -200.0, "high": 80.0}
    assert point["simultaneous_interval_ms"] == {
        "low": -200.0,
        "high": 80.0,
    }
    assert point["simultaneous_label"] == "inconclusive"
    assert point["baseline_early_fire_count"] == 3
    assert point["treatment_early_fire_count"] == 0
    assert point["baseline_early_short_fire_count"] == 2
    contributions = {
        row["task_id"]: row["paired_delta_ms_by_cost"]["100.0"]
        for row in result["task_contributions"]
    }
    assert contributions == {"task-a": 40.0, "task-b": -100.0}
    assert result["fold_paired_delta_ms_by_cost"] == {"f1": {"100.0": -60.0}}


def test_task_cluster_bootstrap_is_seed_reproducible() -> None:
    decisions = [
        _decision("a", "task-a", latency_ms=80.0),
        _decision("b", "task-b", latency_ms=150.0),
        _decision("c", "task-c", latency_ms=250.0),
    ]
    kwargs = {
        "costs_ms": [100.0],
        "replicates": 1_000,
        "confidence_level": 0.95,
        "seed": 17,
    }

    first = paired_task_cluster_bootstrap(decisions, **kwargs)
    second = paired_task_cluster_bootstrap(decisions, **kwargs)

    assert first == second


def test_task_cluster_bootstrap_rejects_incomplete_cost_panel() -> None:
    decisions = [
        _decision("a", "task-a", latency_ms=80.0, cost_ms=100.0),
        _decision("b", "task-b", latency_ms=80.0, cost_ms=100.0),
        _decision("b", "task-b", latency_ms=80.0, cost_ms=200.0),
    ]

    with pytest.raises(ValueError, match="incomplete cost panel"):
        paired_task_cluster_bootstrap(
            decisions,
            costs_ms=[100.0, 200.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_task_cluster_bootstrap_rejects_non_subpolicy_trigger() -> None:
    decision = _decision("a", "task-a", latency_ms=80.0)
    decision["offline_gated_robust_trigger_ms"] = 50.0

    with pytest.raises(ValueError, match="not baseline or deadline"):
        paired_task_cluster_bootstrap(
            [decision],
            costs_ms=[100.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_task_cluster_bootstrap_restore_cost_charges_early_short_fires() -> None:
    # task-a: baseline fires at 0 on a short call (exposed 20 + restore rho);
    # treatment waits at the deadline. task-b: both fire on a long call, so
    # its -100 delta carries no restore charge at any rho.
    decisions = [
        _decision("a", "task-a", latency_ms=80.0),
        _decision("b", "task-b", latency_ms=150.0),
    ]
    kwargs = {
        "costs_ms": [100.0],
        "replicates": 100,
        "confidence_level": 0.95,
        "seed": 0,
    }

    base = paired_task_cluster_bootstrap(decisions, **kwargs)
    charged = paired_task_cluster_bootstrap(
        decisions,
        restore_cost_fraction=0.3,
        **kwargs,
    )

    assert base["restore_cost_fraction"] == 0.0
    assert base["points"]["100.0"]["paired_delta_ms"] == -80.0
    assert charged["restore_cost_fraction"] == 0.3
    assert charged["points"]["100.0"]["paired_delta_ms"] == -50.0


def test_task_cluster_bootstrap_vs_deadline_requires_explicit_optout() -> None:
    decision = _decision("a", "task-a", latency_ms=80.0)
    decision["offline_gated_robust_trigger_ms"] = 50.0
    kwargs = {
        "costs_ms": [100.0],
        "replicates": 100,
        "confidence_level": 0.95,
        "seed": 0,
        "baseline_trigger_field": "threshold_ms",
    }

    with pytest.raises(ValueError, match="not baseline or deadline"):
        paired_task_cluster_bootstrap([decision], **kwargs)

    result = paired_task_cluster_bootstrap(
        [decision],
        enforce_gated_treatment=False,
        restore_cost_fraction=0.1,
        **kwargs,
    )

    # Treatment fires at 50 on a short call: exposed 70 + restore 10; the
    # deadline baseline never fires.
    assert result["enforce_gated_treatment"] is False
    assert result["points"]["100.0"]["paired_delta_ms"] == -80.0


def test_task_cluster_bootstrap_uses_ten_cost_bonferroni_family() -> None:
    decisions = _decision_panel(
        task_ids=[f"task-{index}" for index in range(5)],
        costs=_TEN_COSTS_MS,
        latency_ms=80.0,
    )

    result = paired_task_cluster_bootstrap(
        decisions,
        costs_ms=_TEN_COSTS_MS,
        replicates=50_000,
        confidence_level=0.95,
        seed=0,
    )

    assert result["bootstrap"]["simultaneous_family_size"] == 10
    assert result["bootstrap"]["simultaneous_tail_probability"] == pytest.approx(0.0025)
    assert {point["simultaneous_label"] for point in result["points"].values()} == {
        "positive"
    }


def test_task_cluster_bootstrap_harmful_and_zero_boundaries() -> None:
    harmful = paired_task_cluster_bootstrap(
        _decision_panel(
            task_ids=[f"task-{index}" for index in range(5)],
            costs=[100.0],
            latency_ms=150.0,
        ),
        costs_ms=[100.0],
        replicates=1_000,
        confidence_level=0.95,
        seed=0,
    )
    zero_decisions = _decision_panel(
        task_ids=[f"task-{index}" for index in range(5)],
        costs=[100.0],
        latency_ms=150.0,
    )
    for row in zero_decisions:
        row["offline_gated_robust_trigger_ms"] = row["robust_trigger_ms"]
    zero = paired_task_cluster_bootstrap(
        zero_decisions,
        costs_ms=[100.0],
        replicates=1_000,
        confidence_level=0.95,
        seed=0,
    )

    assert harmful["points"]["100.0"]["simultaneous_label"] == "harmful"
    assert zero["points"]["100.0"]["simultaneous_interval_ms"] == {
        "low": 0.0,
        "high": 0.0,
    }
    assert zero["points"]["100.0"]["simultaneous_label"] == "inconclusive"


def test_permutation_certificate_off_by_default_is_byte_identical() -> None:
    decisions = _decision_panel(
        task_ids=[f"task-{index}" for index in range(5)],
        costs=[100.0],
        latency_ms=80.0,
    )
    kwargs = {
        "costs_ms": [100.0],
        "replicates": 1_000,
        "confidence_level": 0.95,
        "seed": 0,
    }

    off = paired_task_cluster_bootstrap(decisions, **kwargs)
    on = paired_task_cluster_bootstrap(decisions, permutation_draws=2_000, **kwargs)

    assert "permutation" not in off
    assert "permutation_label" not in off["points"]["100.0"]
    assert paired_task_cluster_bootstrap(decisions, **kwargs) == off  # reproducible
    assert on["permutation"]["method"] == "paired_signflip_randomization"
    # Percentile certifies a clean 5-task win; the randomization test cannot at
    # family tail 0.0025 because the smallest achievable p-value is 1/2^5 ~ 0.031.
    assert on["points"]["100.0"]["simultaneous_label"] == "positive"
    assert on["points"]["100.0"]["permutation_label"] == "inconclusive"
    assert on["points"]["100.0"]["permutation_p_positive"] == pytest.approx(
        1.0 / 32.0, abs=0.01
    )
    # The exact-test floor 1/2^5 binds (> the 1/2001 Monte-Carlo floor), so no
    # 5-task result can ever certify at family tail 0.0025.
    assert on["permutation"]["min_achievable_p_value"] == pytest.approx(1.0 / 32.0)
    assert on["permutation"]["min_achievable_p_value"] > 0.0025


def test_permutation_certificate_certifies_a_clean_wide_win() -> None:
    # 40 tasks, each a strictly positive paired delta: min achievable p-value is
    # 1/2^40 << 0.0025, so an unambiguous win now certifies under the exact test.
    decisions = _decision_panel(
        task_ids=[f"task-{index}" for index in range(40)],
        costs=[100.0],
        latency_ms=80.0,
    )
    result = paired_task_cluster_bootstrap(
        decisions,
        costs_ms=[100.0],
        replicates=1_000,
        confidence_level=0.95,
        seed=0,
        permutation_draws=5_000,
    )
    assert result["points"]["100.0"]["permutation_label"] == "positive"


def test_corpus_manifest_resolves_inputs_and_validates_types(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    manifest = read_tool_latency_corpus_manifest(manifest_path, repo_root=tmp_path)

    assert manifest["costs_ms"] == _TEN_COSTS_MS
    assert manifest["expected_task_count"] == 5
    assert Path(manifest["task_ids_file"]).is_file()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["skip_leading_cd"] = 0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a bool"):
        read_tool_latency_corpus_manifest(manifest_path, repo_root=tmp_path)


def test_corpus_reader_uses_explicit_logical_task_metadata() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures/openclaw_minimal_v5.jsonl"

    task_by_trace = require_explicit_trace_task_ids([fixture])

    assert task_by_trace[str(fixture.resolve())] == "test-openclaw-1"


def _decision(
    sample_id: str,
    task_id: str,
    *,
    latency_ms: float,
    cost_ms: float = 100.0,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "outer_fold": "f1",
        "latency_ms": latency_ms,
        "kv_cost_ms": cost_ms,
        "threshold_ms": cost_ms,
        "robust_trigger_ms": 0.0,
        "offline_gated_robust_trigger_ms": cost_ms,
    }


def _decision_panel(
    *,
    task_ids: list[str],
    costs: list[float],
    latency_ms: float,
) -> list[dict[str, Any]]:
    return [
        _decision(
            f"sample-{task_id}",
            task_id,
            latency_ms=latency_ms,
            cost_ms=cost,
        )
        for task_id in task_ids
        for cost in costs
    ]


def _write_manifest(tmp_path: Path) -> Path:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    task_ids = tmp_path / "task_ids.txt"
    task_ids.write_text("task-a\ntask-b\ntask-c\ntask-d\ntask-e\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "collection_id": "corpus",
        "trace_root": str(trace_root),
        "task_ids_file": str(task_ids),
        "expected_task_count": 5,
        "fold_count": 5,
        "inner_folds": 4,
        "costs_ms": _TEN_COSTS_MS,
        "guard_ms": 0,
        "min_tool_history": 1,
        "min_profile_tasks": 1,
        "command_field": "command",
        "max_prefix_depth": 4,
        "skip_leading_cd": False,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest
