import json

import pytest

from scripts.evaluation.evaluate_command_history_residual import BUCKETS, DISK, Row
from scripts.evaluation.evaluate_survival_disk import (
    FROZEN_FIT_ROWS,
    FROZEN_FIT_TRACE_ROOT,
    FROZEN_VALIDATION_ROWS,
    FROZEN_VALIDATION_TRACE_ROOT,
    SPLIT_MANIFEST,
    _decision,
    run,
)


def _row(
    sample: str,
    task: str,
    command: str,
    *,
    latency: int = 1,
    cpu: int = 0,
    rss: int = 0,
    disk: int = 0,
) -> Row:
    return Row(
        sample,
        task,
        command,
        {
            "latency": latency,
            "peak_cpu_cores": cpu,
            "sampled_peak_rss_mb": rss,
            DISK: disk,
        },
        {
            "latency": 0,
            "peak_cpu_cores": "low",
            "sampled_peak_rss_mb": "low",
            DISK: "low",
        },
        {
            target: tuple([1.0, *([0.0] * (buckets - 1))])
            for target, buckets in BUCKETS.items()
        },
    )


def _action(*, duration: float = 2.0) -> dict:
    return {
        "ts_start": 10.0,
        "ts_end": 10.0 + duration,
        "data": {
            "tool_name": "exec",
            "resource_timeline": {
                "version": 1,
                "sample_interval_s": 0.5,
                "samples": [
                    {
                        "offset_s": 0.505,
                        "dt_s": 0.5,
                    }
                ],
            },
        },
    }


def test_decision_uses_only_timing_and_survival() -> None:
    decision = _decision(_action())

    assert decision is not None
    assert decision["effective_offset_s"] == pytest.approx(0.64632007875)
    assert decision["latency_floor"] == 1
    assert _decision(_action(duration=0.6)) is None

    invalid = _action()
    invalid["data"]["resource_timeline"]["samples"][0].pop("offset_s")
    invalid["data"]["resource_timeline"]["samples"].append(
        {"offset_s": 0.505, "dt_s": 0.5, "net_rx_bytes": 1}
    )
    assert _decision(invalid) is None


def test_survival_disk_uses_fit_pmf_and_preserves_cpu_rss() -> None:
    fit = _row("fit:0", "fit", "echo fit", disk=1)
    validation = _row("t1:0", "t1", "echo validation", disk=1)

    result, rows = run(
        [fit],
        [validation],
        {fit.sample_id: _action()},
        {validation.sample_id: _action()},
    )

    primary = rows[0]["arms"]["survival_disk"]
    ablation = rows[0]["arms"]["semantic_exact"]
    assert primary["candidate"][DISK] == "medium"
    assert primary["candidate_probability_by_bucket"][DISK] == [0.0, 1.0, 0.0]
    assert primary["provenance"][DISK] == {
        "source": "survival_by_current_disk",
        "current_disk_bucket": 0,
        "support": 1,
        "effective_offset_s": pytest.approx(0.64632007875),
    }
    for target in ("peak_cpu_cores", "sampled_peak_rss_mb"):
        assert primary["candidate"][target] == ablation["candidate"][target]
        assert primary["candidate_probability_by_bucket"][target] == ablation[
            "candidate_probability_by_bucket"
        ][target]
    assert result["gate"]["cpu_rss_bit_identical_to_semantic"] is True
    assert result["protocol"]["resource_values_read"] is False


def test_survival_fit_tie_uses_lower_disk_bucket_and_does_not_update() -> None:
    fit = [
        _row("fit-a:0", "fit-a", "echo a", disk=0),
        _row("fit-b:0", "fit-b", "echo b", disk=1),
    ]
    validation = [
        _row("t1:0", "t1", "echo c", disk=2),
        _row("t2:0", "t2", "echo d", disk=2),
    ]

    result, rows = run(
        fit,
        validation,
        {row.sample_id: _action() for row in fit},
        {row.sample_id: _action() for row in validation},
    )

    assert result["fit"]["disk_groups"]["low"] == {
        "support": 2,
        "label_counts": [1, 1, 0],
        "pmf": [0.5, 0.5, 0.0],
    }
    for row in rows:
        arm = row["arms"]["survival_disk"]
        assert arm["candidate"][DISK] == "low"
        assert arm["provenance"][DISK]["support"] == 2


def test_finished_command_uses_exact_disk_evidence() -> None:
    fit = _row("fit:0", "fit", "echo same", disk=2)
    validation = _row("t1:0", "t1", "echo same", disk=2)

    result, rows = run(
        [fit],
        [validation],
        {fit.sample_id: _action(duration=0.6)},
        {validation.sample_id: _action(duration=0.6)},
    )

    for arm in ("semantic_exact", "survival_disk"):
        assert rows[0]["arms"][arm]["candidate"][DISK] == "high"
        assert rows[0]["arms"][arm]["provenance"][DISK] == {
            "source": "exact",
            "support": 1,
        }
    assert result["coverage"]["live_at_decision"] == 0
    assert result["coverage"]["disk_survival_overrides"] == 0


def test_gate_is_component_only_and_rows_remain_identical() -> None:
    fit = _row("fit:0", "fit", "echo fit", disk=1)
    validation = _row("t1:0", "t1", "echo validation", disk=1)

    result, rows = run(
        [fit],
        [validation],
        {fit.sample_id: _action()},
        {validation.sample_id: _action()},
    )

    assert len(rows) == 1
    assert result["gate"]["minimum_disk_gain_percentage_points"] == 5.0
    assert result["gate"]["minimum_disk_helpful_tasks"] == 10
    assert result["gate"]["row_identity"] is True
    assert result["gate"]["cpu_still_unresolved"] is True
    assert result["gate"]["go"] is False


def test_frozen_inputs_split_and_trace_roots_are_present() -> None:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    assert FROZEN_FIT_ROWS.is_file()
    assert FROZEN_VALIDATION_ROWS.is_file()
    assert FROZEN_FIT_TRACE_ROOT.is_dir()
    assert FROZEN_VALIDATION_TRACE_ROOT.is_dir()
    assert len(split["development"]) == 100
    assert len(split["validation"]) == 50
    assert len(split["final_test"]) == 50
