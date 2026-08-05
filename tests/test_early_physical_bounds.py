import json

import pytest

from scripts.evaluation.evaluate_command_history_residual import BUCKETS, DISK, Row
from scripts.evaluation.evaluate_early_physical_bounds import (
    CPU_UPDATE_P95_S,
    FROZEN_FIT_ROWS,
    FROZEN_TRACE_ROOT,
    FROZEN_VALIDATION_ROWS,
    SAMPLE_AVAILABILITY_PAD_S,
    SPLIT_MANIFEST,
    _prefix,
    _project,
    run,
)


def _row(
    sample: str,
    task: str,
    command: str,
    *,
    latency: int = 0,
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


def _action(*, duration: float = 2.0, cpu_rate: float = 5.0) -> dict:
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
                        "cpu_core_s": 0.5 * cpu_rate,
                        "cpu_quota_cores": 8.0,
                    }
                ],
            },
        },
    }


def test_projection_only_raises_predictions_and_renormalizes() -> None:
    hard, pmf, changed = _project("latency", 0, [0.6, 0.3, 0.1, 0.0, 0.0], 1)
    assert changed is True
    assert hard == 1
    assert pmf == pytest.approx([0.0, 0.75, 0.25, 0.0, 0.0])

    hard, pmf, changed = _project("peak_cpu_cores", "low", [1.0, 0.0, 0.0], 2)
    assert (hard, pmf, changed) == ("high", [0.0, 0.0, 1.0], True)

    hard, pmf, changed = _project("peak_cpu_cores", "high", [0.2, 0.1, 0.7], 1)
    assert (hard, pmf, changed) == ("high", [0.2, 0.1, 0.7], False)


def test_prefix_uses_frozen_decision_time_and_requires_live_action() -> None:
    prefix = _prefix(_action(cpu_rate=6.0))

    assert prefix is not None
    assert prefix["effective_offset_s"] == pytest.approx(
        0.505 + SAMPLE_AVAILABILITY_PAD_S + CPU_UPDATE_P95_S
    )
    assert prefix["latency_floor"] == 1
    assert prefix["cpu_floor"] == 2
    assert prefix["cpu_rate_cores"] == pytest.approx(6.0)
    assert _prefix(_action(duration=0.6)) is None


def test_run_changes_only_latency_cpu_and_keeps_disk_bit_identical() -> None:
    row = _row("t1:0", "t1", "python -m pytest", latency=1, cpu=2)

    result, rows = run([], [row], {row.sample_id: _action()})

    early = rows[0]["arms"]["early_bounds"]
    assert early["candidate"]["latency"] == 1
    assert early["candidate"]["peak_cpu_cores"] == "high"
    assert early["candidate"]["sampled_peak_rss_mb"] == "low"
    assert early["candidate"][DISK] == rows[0]["current_dynamic"][DISK]
    assert early["candidate_probability_by_bucket"][DISK] == rows[0][
        "current_probability_by_bucket"
    ][DISK]
    assert result["coverage"] == {
        "validation_rows": 1,
        "live_at_decision": 1,
        "latency_projection_changes": 1,
        "cpu_projection_changes": 1,
    }
    assert result["physical_validity"] == {"violations": 0, "rows": []}
    assert result["gate"]["disk_bit_identical_to_current"] is True
    assert result["gate"]["row_identity"] is True


def test_physical_scope_violation_fails_closed() -> None:
    row = _row("t1:0", "t1", "python -m pytest", latency=1, cpu=0)

    result, _rows = run([], [row], {row.sample_id: _action()})

    assert result["physical_validity"] == {
        "violations": 1,
        "rows": [
            {
                "sample_id": row.sample_id,
                "target": "peak_cpu_cores",
                "lower_bound_bucket": 2,
                "label": 0,
            }
        ],
    }
    assert result["gate"]["physical_lower_bounds_valid"] is False
    assert result["gate"]["go"] is False


def test_physical_scope_is_checked_even_when_projection_is_already_high() -> None:
    command = "python -m pytest"
    fit = _row("fit:0", "fit", command, latency=1, cpu=2)
    validation = _row("t1:0", "t1", command, latency=1, cpu=0)

    result, rows = run([fit], [validation], {validation.sample_id: _action()})

    assert rows[0]["arms"]["semantic_only"]["candidate"]["peak_cpu_cores"] == "high"
    assert rows[0]["arms"]["early_bounds"]["candidate"]["peak_cpu_cores"] == "high"
    assert result["coverage"]["cpu_projection_changes"] == 0
    assert result["physical_validity"]["violations"] == 1
    assert result["physical_validity"]["rows"][0]["target"] == "peak_cpu_cores"
    assert result["gate"]["physical_lower_bounds_valid"] is False


def test_frozen_inputs_split_and_trace_root_are_present() -> None:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    assert FROZEN_FIT_ROWS.is_file()
    assert FROZEN_VALIDATION_ROWS.is_file()
    assert FROZEN_TRACE_ROOT.is_dir()
    assert len(split["development"]) == 100
    assert len(split["validation"]) == 50
    assert len(split["final_test"]) == 50
