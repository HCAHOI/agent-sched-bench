from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.aggregate_offline_probe_cv import main as aggregate_main
from scripts.evaluate_offline_probe_clock import main as offline_probe_main
from trace_collect.tool_latency_offline_probe import (
    aggregate_offline_probe_cv,
    evaluate_offline_probe_clock,
    mean_clock_region_stats,
    select_probe_guard,
)


def test_mean_clock_region_stats_partition_surviving_calls() -> None:
    stats = mean_clock_region_stats(
        [80.0, 80.0, 150.0],
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )

    assert stats.trigger_ms == 0.0
    assert stats.normalized_band_gain == pytest.approx(1.0 / 3.0)
    assert stats.normalized_short_penalty == pytest.approx(2.0 / 15.0)
    assert stats.normalized_margin == pytest.approx(0.2)
    assert stats.survivor_count == 3
    assert stats.probability_short_given_survival == pytest.approx(2.0 / 3.0)
    assert stats.probability_band_given_survival == pytest.approx(1.0 / 3.0)
    assert stats.probability_far_given_survival == 0.0


def test_select_probe_guard_uses_observed_utility_breakpoints() -> None:
    calibration = select_probe_guard(
        [
            _probe_decision("short", "task-a", latency_ms=80.0, score=0.1),
            _probe_decision("band", "task-b", latency_ms=150.0, score=0.5),
        ]
    )

    assert calibration["selected_guard_normalized"] == 0.1
    assert calibration["probe_objective_normalized"] == 1.0
    assert calibration["accepted_early_decision_count"] == 1
    assert calibration["candidate_guard_count"] == 4


def test_select_probe_guard_can_disable_all_early_actions() -> None:
    calibration = select_probe_guard(
        [
            _probe_decision("short-a", "task-a", latency_ms=80.0, score=0.1),
            _probe_decision("short-b", "task-b", latency_ms=90.0, score=0.5),
        ]
    )

    assert calibration["selected_guard_normalized"] is None
    assert calibration["probe_objective_normalized"] == 0.0
    assert calibration["accepted_early_decision_count"] == 0


def test_offline_probe_cross_fits_profile_tasks_and_applies_learned_guard() -> None:
    result = evaluate_offline_probe_clock(
        _eval_rows(),
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
    )

    assert result["profile_task_count"] == 8
    assert result["calibration"]["probe_task_count"] == 8
    assert result["calibration"]["selected_guard_normalized"] == 0.0
    assert result["robust_calibration"]["probe_task_count"] == 8
    assert result["robust_calibration"]["selected_guard_normalized"] == 0.0
    assert "dataset_name" not in result
    point = result["points"]["100.0"]
    calibrated = point["policies"]["offline_probe_guard"]
    assert calibrated["delta_vs_deadline_ms"] == 80.0
    assert calibrated["early_trigger_count"] == 2
    gated = point["policies"]["offline_gated_robust_clock"]
    assert gated["delta_vs_deadline_ms"] == 80.0
    assert gated["early_trigger_count"] == 2
    for decision in result["decisions"]:
        assert decision["offline_probe_trigger_ms"] == 0.0
        assert decision["offline_gated_robust_trigger_ms"] in {
            decision["robust_trigger_ms"],
            decision["threshold_ms"],
        }
        assert decision["probe_robust_margin_normalized"] > 0.0
        probabilities = [
            decision["probe_probability_short_given_survival"],
            decision["probe_probability_band_given_survival"],
            decision["probe_probability_far_given_survival"],
        ]
        assert sum(probabilities) == pytest.approx(1.0)


def test_offline_robust_guard_filters_low_margin_and_retains_high_margin() -> None:
    profile_by_task = {
        "p0": {"alpha": [180.0, 220.0, 120.0], "beta": [220.0, 80.0, 80.0]},
        "p1": {"alpha": [80.0, 150.0], "beta": [80.0]},
        "p2": {"alpha": [220.0, 20.0], "beta": [60.0, 60.0]},
        "p3": {"alpha": [180.0, 20.0, 80.0], "beta": [80.0, 80.0, 180.0]},
        "p4": {"alpha": [180.0, 80.0, 180.0], "beta": [180.0]},
        "p5": {"alpha": [60.0], "beta": [80.0]},
        "p6": {"alpha": [180.0, 120.0, 120.0], "beta": [80.0, 20.0, 60.0]},
        "p7": {"alpha": [60.0, 150.0], "beta": [20.0, 20.0, 20.0]},
    }
    result = evaluate_offline_probe_clock(
        [
            _row(
                "alpha-band",
                150.0,
                task_id="eval-a",
                source_trace="eval-a",
                command="alpha",
            ),
            _row(
                "beta-band",
                150.0,
                task_id="eval-b",
                source_trace="eval-b",
                command="beta",
            ),
            _row(
                "alpha-short",
                80.0,
                task_id="eval-c",
                source_trace="eval-c",
                command="alpha",
            ),
            _row(
                "beta-short",
                80.0,
                task_id="eval-d",
                source_trace="eval-d",
                command="beta",
            ),
        ],
        profile_rows=_grouped_profile_rows(profile_by_task),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
        command_field="command",
    )

    calibration = result["robust_calibration"]
    assert calibration["selected_guard_normalized"] == pytest.approx(
        0.03636363636363638
    )
    assert calibration["eligible_early_decision_count"] == 28
    assert calibration["accepted_early_decision_count"] == 19
    decisions = {row["sample_id"]: row for row in result["decisions"]}
    for sample_id in ("alpha-band", "alpha-short"):
        assert decisions[sample_id]["robust_trigger_ms"] == 80.0
        assert decisions[sample_id]["offline_gated_robust_trigger_ms"] == 80.0
    for sample_id in ("beta-band", "beta-short"):
        assert decisions[sample_id]["robust_trigger_ms"] == 80.0
        assert decisions[sample_id]["offline_gated_robust_trigger_ms"] == 100.0
    policies = result["points"]["100.0"]["policies"]
    assert policies["robust_clock"]["delta_vs_deadline_ms"] == 80.0
    assert policies["offline_gated_robust_clock"]["delta_vs_deadline_ms"] == 40.0


def test_offline_robust_guard_can_select_never_early() -> None:
    latencies_by_task = {
        "p0": [220.0, 20.0],
        "p1": [20.0, 20.0, 20.0, 220.0],
        "p2": [60.0, 60.0, 60.0, 20.0],
        "p3": [60.0, 220.0],
        "p4": [80.0, 180.0],
        "p5": [150.0],
        "p6": [60.0],
        "p7": [20.0, 220.0, 20.0, 80.0],
    }
    profile_by_task = {
        task_id: {"alpha": latencies}
        for task_id, latencies in latencies_by_task.items()
    }
    result = evaluate_offline_probe_clock(
        [
            _row(
                "band", 150.0, task_id="eval-a", source_trace="eval-a", command="alpha"
            ),
            _row(
                "short", 80.0, task_id="eval-b", source_trace="eval-b", command="alpha"
            ),
        ],
        profile_rows=_grouped_profile_rows(profile_by_task),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
        command_field="command",
    )

    calibration = result["robust_calibration"]
    assert calibration["selected_guard_normalized"] is None
    assert calibration["eligible_early_decision_count"] == 9
    assert calibration["accepted_early_decision_count"] == 0
    for decision in result["decisions"]:
        assert decision["robust_trigger_ms"] == 80.0
        assert decision["offline_gated_robust_trigger_ms"] == 100.0
    policies = result["points"]["100.0"]["policies"]
    assert policies["robust_clock"]["delta_vs_deadline_ms"] == 40.0
    assert policies["offline_gated_robust_clock"]["delta_vs_deadline_ms"] == 0.0


def test_offline_probe_rejects_outer_task_overlap() -> None:
    overlapping_eval = [
        _row(
            "eval-overlap",
            150.0,
            task_id="profile-0",
            source_trace="eval-overlap",
        )
    ]

    with pytest.raises(ValueError, match="disjoint logical tasks"):
        evaluate_offline_probe_clock(
            overlapping_eval,
            profile_rows=_profile_rows(),
            kv_costs_ms=[100.0],
            guard_ms=0.0,
            inner_folds=4,
        )


def test_offline_probe_cli_and_aggregator_write_recomputed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    summary_path = tmp_path / "f1_summary.json"
    decisions_path = tmp_path / "f1_decisions.jsonl"
    pooled_path = tmp_path / "pooled.json"
    _write_jsonl(profile_path, _profile_rows())
    _write_jsonl(eval_path, _eval_rows())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_offline_probe_clock.py",
            "--profile-latencies",
            str(profile_path),
            "--eval-latencies",
            str(eval_path),
            "--kv-costs-ms",
            "100",
            "--inner-folds",
            "4",
            "--output",
            str(summary_path),
            "--decisions-output",
            str(decisions_path),
        ],
    )

    offline_probe_main()

    assert (
        "Calibrated guard=0.0 and robust_guard=0.0 and evaluated 2 rows"
        in capsys.readouterr().out
    )
    pooled = aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)
    assert pooled["fold_count"] == 1
    assert pooled["sample_count"] == 2
    assert (
        pooled["points"]["100.0"]["policies"]["offline_probe_guard"][
            "delta_vs_deadline_ms"
        ]
        == 80.0
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_offline_probe_cv.py",
            str(tmp_path),
            "--expected-fold-count",
            "1",
            "--output",
            str(pooled_path),
        ],
    )
    aggregate_main()
    assert "Pooled 2 samples from 1 folds" in capsys.readouterr().out
    assert json.loads(pooled_path.read_text())["sample_count"] == 2


def test_aggregator_accepts_exact_task_disjoint_fold_set(tmp_path: Path) -> None:
    _write_fold(
        tmp_path,
        1,
        _evaluate_one_task("sample-a", "outer-a", latency_ms=150.0),
    )
    _write_fold(
        tmp_path,
        2,
        _evaluate_one_task("sample-b", "outer-b", latency_ms=80.0),
    )

    pooled = aggregate_offline_probe_cv(tmp_path, expected_fold_count=2)

    assert pooled["fold_count"] == 2
    assert pooled["sample_count"] == 2
    assert pooled["task_count"] == 2


def test_aggregator_rejects_outer_task_overlap(tmp_path: Path) -> None:
    _write_fold(
        tmp_path,
        1,
        _evaluate_one_task("sample-a", "shared-task", latency_ms=150.0),
    )
    _write_fold(
        tmp_path,
        2,
        _evaluate_one_task("sample-b", "shared-task", latency_ms=80.0),
    )

    with pytest.raises(ValueError, match="disjoint across folds"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=2)


def test_aggregator_rejects_incomplete_sample_cost_panel(tmp_path: Path) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
        costs=(100.0, 200.0),
    )
    result["decisions"].pop()
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match=r"sample_count \* cost_count"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_stale_unexpected_fold_files(tmp_path: Path) -> None:
    _write_fold(
        tmp_path,
        1,
        _evaluate_one_task("sample-a", "outer-a", latency_ms=150.0),
    )
    _write_fold(
        tmp_path,
        2,
        _evaluate_one_task("sample-b", "outer-b", latency_ms=80.0),
    )

    with pytest.raises(ValueError, match="do not match the expected set"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_summary_decision_mismatch(tmp_path: Path) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    result["row_count"] = 999
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="row_count differs"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_missing_calibration_field(tmp_path: Path) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    del result["calibration"]["probe_objective_normalized"]
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="calibration is missing fields"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_missing_robust_calibration(tmp_path: Path) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    del result["robust_calibration"]
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="missing robust_calibration"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_gated_robust_trigger_inconsistent_with_guard(
    tmp_path: Path,
) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    result["decisions"][0]["offline_gated_robust_trigger_ms"] = 50.0
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="inconsistent with its guard"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_changed_robust_probe_candidate(tmp_path: Path) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    result["decisions"][0]["probe_robust_candidate_trigger_ms"] = 50.0
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="robust probe candidate changed"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_non_null_guard_without_acceptance(
    tmp_path: Path,
) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    calibration = result["calibration"]
    calibration["accepted_early_decision_count"] = 0
    calibration["accepted_task_count"] = 0
    calibration["worst_accepted_task_normalized_delta"] = None
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="non-null guard requires"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_aggregator_rejects_probe_count_inconsistent_with_profile(
    tmp_path: Path,
) -> None:
    result = _evaluate_one_task(
        "sample-a",
        "outer-a",
        latency_ms=150.0,
    )
    result["calibration"]["probe_decision_count"] -= 1
    _write_fold(tmp_path, 1, result)

    with pytest.raises(ValueError, match="probe decision count is inconsistent"):
        aggregate_offline_probe_cv(tmp_path, expected_fold_count=1)


def test_runner_snapshots_result_driving_dependencies() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "analysis/tool-time-offline-gated-robust-20260711/run_probe.sh"
    ).read_text(encoding="utf-8")
    dependencies = (
        "src/trace_collect/causal_history.py",
        "src/trace_collect/classification_metrics.py",
        "src/trace_collect/cli_helpers.py",
        "src/trace_collect/command_features.py",
        "src/trace_collect/latency_validation.py",
        "src/trace_collect/tool_latency_dataset.py",
        "src/trace_collect/tool_latency_offline_probe.py",
        "src/trace_collect/tool_latency_profiled.py",
        "src/trace_collect/tool_latency_utility_clock.py",
        "tests/test_tool_latency_offline_probe.py",
        "tests/test_tool_latency_utility_clock.py",
    )
    for dependency in dependencies:
        assert runner.count(dependency) == 2


def _probe_decision(
    sample_id: str,
    task_id: str,
    *,
    latency_ms: float,
    score: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "latency_ms": latency_ms,
        "kv_cost_ms": 100.0,
        "threshold_ms": 100.0,
        "probe_candidate_trigger_ms": 0.0,
        "probe_margin_normalized": score,
    }


def _evaluate_one_task(
    sample_id: str,
    task_id: str,
    *,
    latency_ms: float,
    costs: tuple[float, ...] = (100.0,),
) -> dict[str, Any]:
    return evaluate_offline_probe_clock(
        [
            _row(
                sample_id,
                latency_ms,
                task_id=task_id,
                source_trace=f"trace-{sample_id}",
            )
        ],
        profile_rows=_profile_rows(),
        kv_costs_ms=costs,
        guard_ms=0.0,
        inner_folds=4,
    )


def _write_fold(
    root: Path,
    fold: int,
    result: dict[str, Any],
) -> None:
    decisions = result["decisions"]
    summary = {key: value for key, value in result.items() if key != "decisions"}
    (root / f"f{fold}_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / f"f{fold}_decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )


def _profile_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_index in range(8):
        task_id = f"profile-{task_index}"
        for call_index, latency_ms in enumerate((80.0, 80.0, 150.0)):
            rows.append(
                _row(
                    f"{task_id}-{call_index}",
                    latency_ms,
                    task_id=task_id,
                    source_trace=f"trace-{task_id}",
                )
            )
    return rows


def _eval_rows() -> list[dict[str, object]]:
    return [
        _row("eval-short", 80.0, task_id="eval-a", source_trace="eval-a"),
        _row("eval-band", 150.0, task_id="eval-b", source_trace="eval-b"),
    ]


def _row(
    sample_id: str,
    latency_ms: float,
    *,
    task_id: str,
    source_trace: str,
    command: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "task_id": task_id,
        "tool_name": "exec",
        "tool_ts_start": 0.0,
        "tool_ts_end": latency_ms / 1000.0,
        "latency_ms": latency_ms,
    }
    if command is not None:
        row["tool_args"] = {"command": command}
    return row


def _grouped_profile_rows(
    profile_by_task: dict[str, dict[str, list[float]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id, groups in profile_by_task.items():
        call_index = 0
        for command, latencies in groups.items():
            for latency_ms in latencies:
                rows.append(
                    _row(
                        f"{task_id}-{call_index}",
                        latency_ms,
                        task_id=task_id,
                        source_trace=f"trace-{task_id}",
                        command=command,
                    )
                )
                call_index += 1
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
