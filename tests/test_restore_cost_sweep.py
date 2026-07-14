from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.analyze_restore_cost_sweep import (
    analyze_restore_cost_sweep,
    render_summary_markdown,
)


def test_restore_cost_sweep_rescoring_matches_hand_computed_deltas(
    tmp_path: Path,
) -> None:
    _write_confirmation_cv(
        tmp_path,
        {
            "f1": [_decision("a", "task-a", latency_ms=80.0)],
            "f2": [_decision("b", "task-b", latency_ms=150.0)],
        },
    )

    result = analyze_restore_cost_sweep(
        tmp_path,
        restore_cost_fractions=[0.0, 0.5],
        replicates=100,
        confidence_level=0.95,
        seed=0,
    )

    assert result["fold_names"] == ["f1", "f2"]
    assert result["decision_row_count"] == 2
    assert result["costs_ms"] == [100.0]
    totals = {
        name: {
            fraction: payload["points"]["100.0"]["paired_delta_ms"]
            for fraction, payload in comparison["by_restore_cost_fraction"].items()
        }
        for name, comparison in result["comparisons"].items()
    }
    # task-a: robust fires at 0 on a short call (-20 - 100f), gated waits (0);
    # task-b: robust fires at 0 on a long call (+100), gated equals deadline.
    assert totals["gated_vs_robust"] == {"0.0": -80.0, "0.5": -30.0}
    assert totals["gated_vs_deadline"] == {"0.0": 0.0, "0.5": 0.0}
    assert totals["robust_vs_deadline"] == {"0.0": 80.0, "0.5": 30.0}

    summary = render_summary_markdown(result)
    assert "## gated_vs_robust" in summary
    assert "| 0.5 |" in summary


def test_restore_cost_sweep_rejects_conflicting_fold_stamp(tmp_path: Path) -> None:
    row = _decision("a", "task-a", latency_ms=80.0)
    row["outer_fold"] = "f9"
    _write_confirmation_cv(tmp_path, {"f1": [row]})

    with pytest.raises(ValueError, match="conflicts with fold file"):
        analyze_restore_cost_sweep(
            tmp_path,
            restore_cost_fractions=[0.0],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


def test_restore_cost_sweep_rejects_duplicate_fractions(tmp_path: Path) -> None:
    _write_confirmation_cv(
        tmp_path,
        {"f1": [_decision("a", "task-a", latency_ms=80.0)]},
    )

    with pytest.raises(ValueError, match="unique"):
        analyze_restore_cost_sweep(
            tmp_path,
            restore_cost_fractions=[0.5, 0.5],
            replicates=100,
            confidence_level=0.95,
            seed=0,
        )


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
        "latency_ms": latency_ms,
        "kv_cost_ms": cost_ms,
        "threshold_ms": cost_ms,
        "deadline_trigger_ms": cost_ms,
        "robust_trigger_ms": 0.0,
        "offline_gated_robust_trigger_ms": cost_ms,
    }


def _write_confirmation_cv(
    root: Path,
    rows_by_fold: dict[str, list[dict[str, Any]]],
) -> None:
    cv_root = root / "cv"
    cv_root.mkdir(parents=True)
    for fold, rows in rows_by_fold.items():
        (cv_root / f"{fold}_decisions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
