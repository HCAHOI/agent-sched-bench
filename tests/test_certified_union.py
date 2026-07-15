"""Hand-computable tests for the certified gate-union combiner.

All fixtures use kv_cost = threshold = 100 ms and deadline_trigger = 100, so a
gate that fires at 0 on a long call (latency 150) hides the full 100 ms
(utility +100 vs the deadline's 0), and a gate that fires at 0 on a short call
(latency 50) is exposed for 50 ms (utility -50 vs the deadline's 0). The
leave-fold-out inclusion for fold ``f`` uses only rows with ``outer_fold != f``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.restore_cost_analysis import run_certified_union_analysis


def _row(
    sample_id: str,
    task_id: str,
    fold: str,
    *,
    latency_ms: float,
    hazard_trigger_ms: float,
    trie_trigger_ms: float,
    cost_ms: float = 100.0,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "outer_fold": fold,
        "latency_ms": latency_ms,
        "kv_cost_ms": cost_ms,
        "threshold_ms": cost_ms,
        "deadline_trigger_ms": cost_ms,
        "offline_gated_hazard_trigger_ms": hazard_trigger_ms,
        "offline_gated_robust_trigger_ms": trie_trigger_ms,
    }


def _write_hazard_root(root: Path, rows_by_fraction: dict[str, list[dict[str, Any]]]) -> None:
    root.mkdir(parents=True)
    for key, rows in rows_by_fraction.items():
        (root / f"rho_{key}_decisions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _delta(result: dict[str, Any], name: str, key: str = "0.0") -> float:
    return result["comparisons"][name]["by_restore_cost_fraction"][key]["points"][
        "100.0"
    ]["paired_delta_ms"]


def test_both_gates_certified_reduces_to_naive_union(tmp_path: Path) -> None:
    # SWE-ReBench-like: every long call is caught by exactly one gate, so each
    # gate's leave-fold-out delta is strictly positive on both folds -> both
    # included -> certified union == naive union.
    rows = [
        _row("a1", "task-a1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
        _row("b1", "task-b1", "f1", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0),
        _row("a2", "task-a2", "f2", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
        _row("b2", "task-b2", "f2", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0),
    ]
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": rows})

    result = run_certified_union_analysis(
        hazard_root,
        output_root=tmp_path / "certified",
        restore_cost_fractions=[0.0],
        replicates=100,
        confidence_level=0.95,
        seed=0,
    )

    inclusion = result["gate_inclusion_by_restore_cost_fraction"]["0.0"]
    # Fold f1's inclusion is computed from f2 rows only, and vice versa. In each
    # LOO partition one task has a +100 gate delta and the other 0.
    for fold in ("f1", "f2"):
        assert inclusion[fold]["hazard"]["included"] is True
        assert inclusion[fold]["trie"]["included"] is True
        assert inclusion[fold]["hazard"]["leave_fold_out_delta_ms"] == 100.0
        assert inclusion[fold]["trie"]["leave_fold_out_delta_ms"] == 100.0
    # Both gates in -> certified trigger = min(hazard, trie) = naive union.
    assert _delta(result, "certified_union_vs_naive_union") == 0.0
    # Every row is a long call caught at 0 by one gate: +100 each over deadline.
    assert _delta(result, "certified_union_vs_deadline") == 400.0

    decisions = [
        json.loads(line)
        for line in (tmp_path / "certified" / "rho_0.0_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["certified_union_trigger_ms"] for row in decisions} == {0.0}


def test_harmful_gate_excluded_reduces_to_other_gate(tmp_path: Path) -> None:
    # Terminal-Bench-like: the trie fires on short calls (net-harmful) while the
    # hazard gate fires on long calls. The trie's leave-fold-out delta is
    # negative -> excluded -> certified union drops it and equals the hazard.
    rows = [
        _row("h1", "task-h1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
        _row("t1", "task-t1", "f1", latency_ms=50.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0),
        _row("h2", "task-h2", "f2", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
        _row("t2", "task-t2", "f2", latency_ms=50.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0),
    ]
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": rows})

    result = run_certified_union_analysis(
        hazard_root,
        output_root=tmp_path / "certified",
        restore_cost_fractions=[0.0],
        replicates=100,
        confidence_level=0.95,
        seed=0,
    )

    inclusion = result["gate_inclusion_by_restore_cost_fraction"]["0.0"]
    for fold in ("f1", "f2"):
        assert inclusion[fold]["hazard"]["included"] is True
        assert inclusion[fold]["trie"]["included"] is False
        assert inclusion[fold]["hazard"]["leave_fold_out_delta_ms"] == 100.0
        # trie fires at 0 on the LOO short call: exposed 50, no restore at rho 0.
        assert inclusion[fold]["trie"]["leave_fold_out_delta_ms"] == -50.0
    # Certified == hazard alone.
    assert _delta(result, "certified_union_vs_gated_hazard") == 0.0
    # Only the two long calls (hazard fires at 0) beat the deadline: +200.
    assert _delta(result, "certified_union_vs_deadline") == 200.0
    # vs naive union >= 0 exactly because the dropped gate was net-harmful: the
    # naive union eats the trie's -50 on each short call; certified waits (0).
    assert _delta(result, "certified_union_vs_naive_union") == 100.0


def test_no_gate_certified_falls_back_to_deadline(tmp_path: Path) -> None:
    # Neither gate ever fires early (triggers == deadline), so every
    # leave-fold-out delta is exactly 0 (not strictly positive) -> both
    # excluded -> certified union == the deadline everywhere.
    rows = [
        _row("x1", "task-x1", "f1", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=100.0),
        _row("y1", "task-y1", "f1", latency_ms=50.0, hazard_trigger_ms=100.0, trie_trigger_ms=100.0),
        _row("x2", "task-x2", "f2", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=100.0),
        _row("y2", "task-y2", "f2", latency_ms=50.0, hazard_trigger_ms=100.0, trie_trigger_ms=100.0),
    ]
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": rows})

    result = run_certified_union_analysis(
        hazard_root,
        output_root=tmp_path / "certified",
        restore_cost_fractions=[0.0],
        replicates=100,
        confidence_level=0.95,
        seed=0,
    )

    inclusion = result["gate_inclusion_by_restore_cost_fraction"]["0.0"]
    for fold in ("f1", "f2"):
        assert inclusion[fold]["hazard"]["included"] is False
        assert inclusion[fold]["trie"]["included"] is False
        assert inclusion[fold]["hazard"]["leave_fold_out_delta_ms"] == 0.0
        assert inclusion[fold]["trie"]["leave_fold_out_delta_ms"] == 0.0
    assert _delta(result, "certified_union_vs_deadline") == 0.0
    decisions = [
        json.loads(line)
        for line in (tmp_path / "certified" / "rho_0.0_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["certified_union_trigger_ms"] for row in decisions} == {100.0}


def test_loo_lcb_includes_when_every_task_strictly_positive(tmp_path: Path) -> None:
    # Both gates fire at 0 on every task's long call, so every leave-fold-out
    # task has a strictly positive delta for both gates -> every bootstrap
    # resample total is positive -> the LCB is > 0 -> both included, even under
    # the conservative criterion with a tiny replicate count.
    rows = [
        _row("p1", "task-p1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=0.0),
        _row("q1", "task-q1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=0.0),
        _row("p2", "task-p2", "f2", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=0.0),
        _row("q2", "task-q2", "f2", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=0.0),
    ]
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": rows})

    result = run_certified_union_analysis(
        hazard_root,
        output_root=tmp_path / "certified",
        restore_cost_fractions=[0.0],
        replicates=64,
        confidence_level=0.95,
        seed=0,
        inclusion_criterion="loo_lcb",
    )

    assert result["inclusion_criterion"] == "loo_lcb"
    inclusion = result["gate_inclusion_by_restore_cost_fraction"]["0.0"]
    for fold in ("f1", "f2"):
        for gate in ("hazard", "trie"):
            assert inclusion[fold][gate]["included"] is True
            assert inclusion[fold][gate]["leave_fold_out_lcb_ms"] > 0.0
    # Both gates fire at 0 on every long call -> certified == naive union.
    assert _delta(result, "certified_union_vs_naive_union") == 0.0
    assert _delta(result, "certified_union_vs_deadline") == 400.0


def test_rejects_unknown_inclusion_criterion(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown inclusion_criterion"):
        run_certified_union_analysis(
            tmp_path / "hazard",
            output_root=tmp_path / "certified",
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
            inclusion_criterion="oracle",
        )


def test_rejects_missing_gate_field(tmp_path: Path) -> None:
    row = _row("a1", "task-a1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0)
    del row["offline_gated_hazard_trigger_ms"]
    other = _row("b2", "task-b2", "f2", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0)
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": [row, other]})

    with pytest.raises(ValueError, match="lacks offline_gated_hazard_trigger_ms"):
        run_certified_union_analysis(
            hazard_root,
            output_root=tmp_path / "certified",
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_rejects_empty_decisions(tmp_path: Path) -> None:
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": []})

    with pytest.raises(ValueError, match="no decisions"):
        run_certified_union_analysis(
            hazard_root,
            output_root=tmp_path / "certified",
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_rejects_differing_decision_counts_across_fractions(tmp_path: Path) -> None:
    two = [
        _row("a1", "task-a1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
        _row("b2", "task-b2", "f2", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0),
    ]
    one = [
        _row("a1", "task-a1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
    ]
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": two, "0.5": one})

    with pytest.raises(AssertionError, match="differing decision counts"):
        run_certified_union_analysis(
            hazard_root,
            output_root=tmp_path / "certified",
            restore_cost_fractions=[0.0, 0.5],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_rejects_existing_output_root(tmp_path: Path) -> None:
    rows = [
        _row("a1", "task-a1", "f1", latency_ms=150.0, hazard_trigger_ms=0.0, trie_trigger_ms=100.0),
        _row("b2", "task-b2", "f2", latency_ms=150.0, hazard_trigger_ms=100.0, trie_trigger_ms=0.0),
    ]
    hazard_root = tmp_path / "hazard"
    _write_hazard_root(hazard_root, {"0.0": rows})
    output_root = tmp_path / "certified"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="refusing to mix stale output"):
        run_certified_union_analysis(
            hazard_root,
            output_root=output_root,
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )
