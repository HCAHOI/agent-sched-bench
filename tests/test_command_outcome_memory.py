from collections.abc import Mapping

from scripts.evaluation.evaluate_command_history_residual import BUCKETS, Row
from scripts.evaluation.evaluate_command_outcome_memory import run


def _row(
    sample: str,
    task: str,
    command: str,
    label: int | Mapping[str, int],
    *,
    latency_available: bool = True,
) -> Row:
    labels = (
        {target: label for target in BUCKETS}
        if isinstance(label, int)
        else dict(label)
    )
    current = {
        "latency": 0 if latency_available else None,
        "peak_cpu_cores": "low",
        "sampled_peak_rss_mb": "low",
        "disk_read_write_bytes_total": "low",
    }
    pmfs = {
        target: tuple([1.0, *([0.0] * (buckets - 1))])
        for target, buckets in BUCKETS.items()
    }
    if not latency_available:
        pmfs["latency"] = None
    return Row(sample, task, command, labels, current, pmfs)


def test_command_outcome_memory_updates_only_after_task_settlement() -> None:
    fit = [_row("fit:0", "fit", "pytest a.py", 0)]
    validation = [
        _row("t1:0", "t1", "pytest b.py", 2),
        _row("t1:1", "t1", "pytest b.py", 2),
        _row("t2:0", "t2", "pytest b.py", 2),
    ]

    result, rows = run(fit, validation)

    # Shape evidence from fit is visible, but neither t1 call can see its sibling.
    assert [row["arms"]["hierarchy"]["candidate"]["latency"] for row in rows] == [
        0,
        0,
        2,
    ]
    assert rows[0]["arms"]["exact"]["provenance"]["latency"] == {
        "source": "current",
        "support": 0,
    }
    assert rows[2]["arms"]["exact"]["provenance"]["latency"] == {
        "source": "exact",
        "support": 2,
    }
    assert result["gate"]["row_identity"] is True


def test_hierarchy_ablation_pmfs_targets_fallback_and_gate_are_frozen() -> None:
    exact_labels = {
        "latency": 4,
        "peak_cpu_cores": 2,
        "sampled_peak_rss_mb": 1,
        "disk_read_write_bytes_total": 0,
    }
    fit = [
        _row("fit-a:0", "fit-a", "pytest a.py", exact_labels),
        _row("fit-b:0", "fit-b", "pytest b.py", 0),
        _row("fit-c:0", "fit-c", "pytest c.py", 0),
    ]
    validation = [
        _row("t1:0", "t1", "pytest a.py", exact_labels),
        _row("t1:1", "t1", "git status", 0, latency_available=False),
    ]

    result, rows = run(fit, validation)
    exact = rows[0]["arms"]["exact"]
    shape = rows[0]["arms"]["shape"]
    hierarchy = rows[0]["arms"]["hierarchy"]

    assert exact["candidate"] == hierarchy["candidate"] == {
        "latency": 4,
        "peak_cpu_cores": "high",
        "sampled_peak_rss_mb": "medium",
        "disk_read_write_bytes_total": "low",
    }
    assert shape["candidate"] == rows[0]["current_dynamic"]
    assert hierarchy["candidate_probability_by_bucket"] == {
        "latency": [0.0, 0.0, 0.0, 0.0, 1.0],
        "peak_cpu_cores": [0.0, 0.0, 1.0],
        "sampled_peak_rss_mb": [0.0, 1.0, 0.0],
        "disk_read_write_bytes_total": [1.0, 0.0, 0.0],
    }
    assert shape["provenance"]["latency"] == {"source": "shape", "support": 3}
    assert hierarchy["provenance"]["latency"] == {
        "source": "exact",
        "support": 1,
    }

    fallback = rows[1]["arms"]["hierarchy"]
    assert fallback["candidate"]["latency"] is None
    assert fallback["candidate_probability_by_bucket"]["latency"] is None
    assert fallback["provenance"]["latency"] == {
        "source": "current",
        "support": 0,
    }
    assert result["gate"]["minimum_gain_percentage_points_each"] == 5.0
    assert result["gate"]["minimum_helpful_tasks"] == 10
    assert result["gate"]["all_targets_meet_gain"] is False
    assert result["gate"]["go"] is False


def test_shape_pmf_tie_uses_lower_bucket() -> None:
    fit = [
        _row("fit-a:0", "fit-a", "pytest a.py", 0),
        _row("fit-b:0", "fit-b", "pytest b.py", 2),
    ]

    _result, rows = run(fit, [_row("t1:0", "t1", "pytest c.py", 0)])

    shape = rows[0]["arms"]["shape"]
    assert shape["candidate_probability_by_bucket"]["latency"] == [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
    ]
    assert shape["candidate"]["latency"] == 0
    assert shape["candidate"]["peak_cpu_cores"] == "low"
