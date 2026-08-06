import argparse
from copy import deepcopy
import json

import pytest

from scripts.evaluation import evaluate_joint_prediction_state as joint


def _pmfs(value=0):
    return {
        "latency": [1.0 if index == value else 0.0 for index in range(5)],
        "peak_cpu_cores": [1.0, 0.0, 0.0],
        "sampled_peak_rss_mb": [1.0, 0.0, 0.0],
        "disk_read_write_bytes_total": [1.0, 0.0, 0.0],
    }


def _fit_row(task_id, *, high=False):
    label = 2 if high else 0
    return {
        "sample_id": f"{task_id}:0",
        "task_id": task_id,
        "command": "python -m pytest",
        "latency_label": 0,
        "resource_labels": {
            "peak_cpu_cores": label,
            "sampled_peak_rss_mb": 0,
            "disk_read_write_bytes_total": label,
        },
        "current_dynamic": {
            "latency": 0,
            "peak_cpu_cores": "low",
            "sampled_peak_rss_mb": "low",
            "disk_read_write_bytes_total": "low",
            "probability_by_bucket": _pmfs(),
        },
    }


def _validation_row(task_id):
    current_pmfs = _pmfs()
    composition_pmfs = _pmfs()
    composition_pmfs["latency"] = [0.0, 0.0, 0.0, 0.0, 1.0]
    composition_pmfs["sampled_peak_rss_mb"] = [0.0, 0.0, 1.0]
    return {
        "sample_id": f"{task_id}:0",
        "task_id": task_id,
        "command": "python -m pytest",
        "labels": {
            "latency": 4,
            "peak_cpu_cores": 2,
            "sampled_peak_rss_mb": 2,
            "disk_read_write_bytes_total": 2,
        },
        "current_dynamic": {
            "latency": 0,
            "peak_cpu_cores": "low",
            "sampled_peak_rss_mb": "low",
            "disk_read_write_bytes_total": "low",
        },
        "current_probability_by_bucket": current_pmfs,
        "arms": {
            "composition": {
                "candidate": {
                    "latency": 4,
                    "peak_cpu_cores": "low",
                    "sampled_peak_rss_mb": "high",
                    "disk_read_write_bytes_total": "low",
                },
                "candidate_probability_by_bucket": composition_pmfs,
            }
        },
    }


def test_forward_gate_and_frozen_composition() -> None:
    fit = [
        _fit_row(f"task-{index:02d}", high=index < 40 or index >= 40)
        for index in range(80)
    ]

    audit, audit_rows = joint.forward_audit(fit)
    validation, rows = joint.validation(
        fit, [_validation_row(f"validation-{index:02d}") for index in range(12)]
    )

    assert audit["go"] is True
    assert len(audit_rows) == 40
    assert audit["target_coverage"] == {
        "peak_cpu_cores": 1.0,
        "disk_read_write_bytes_total": 1.0,
    }
    assert validation["go"] is True
    assert validation["latency_rss_bit_identical_to_composition"] is True
    assert all(row["candidate"]["peak_cpu_cores"] == "high" for row in rows)
    assert all(
        row["candidate"]["disk_read_write_bytes_total"] == "high" for row in rows
    )


def test_missing_state_falls_back_and_harmful_audit_stops() -> None:
    calibration = [_fit_row(f"task-{index:02d}", high=True) for index in range(40)]
    audit_rows = [_fit_row(f"task-{index:02d}", high=False) for index in range(40, 80)]
    result, _rows = joint.forward_audit([*calibration, *audit_rows])

    unseen = _fit_row("unseen", high=False)
    unseen["current_dynamic"]["latency"] = 1
    unseen["current_dynamic"]["probability_by_bucket"]["latency"] = [0, 1, 0, 0, 0]
    applied = joint._apply([unseen], joint._fit_counts(calibration), composition=False)[
        0
    ]

    assert result["go"] is False
    assert applied["state_support"] == {
        "peak_cpu_cores": 0,
        "disk_read_write_bytes_total": 0,
    }
    assert applied["candidate"] == applied["base"]
    assert (
        applied["candidate_probability_by_bucket"]
        == applied["base_probability_by_bucket"]
    )


def test_fit_rejects_inconsistent_current_prediction() -> None:
    row = _fit_row("bad", high=True)
    row["current_dynamic"]["peak_cpu_cores"] = "high"

    with pytest.raises(ValueError, match="hard prediction differs"):
        joint._fit_counts([row])

    row = _fit_row("bad-latency", high=True)
    row["current_dynamic"]["latency"] = 0.5
    with pytest.raises(ValueError, match="invalid latency hard prediction"):
        joint._fit_counts([row])


def test_row_identity_covers_all_frozen_inputs() -> None:
    source = _fit_row("task", high=False)
    output = joint._apply([source], {}, composition=False)
    changed = deepcopy(output)
    changed[0]["command"] = "different"

    assert joint._row_identity([source], output) is True
    assert joint._row_identity([source], changed) is False


def test_forward_no_go_does_not_read_validation(tmp_path, monkeypatch) -> None:
    fit_path = tmp_path / "fit.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    out_dir = tmp_path / "result"
    fit_path.write_text("", encoding="utf-8")
    validation_path.write_text("reserved", encoding="utf-8")
    accessed = []

    def fake_sha(path):
        accessed.append(path)
        if path == validation_path:
            raise AssertionError("validation contents were accessed")
        return "fit-digest"

    def fake_load(path):
        if path == validation_path:
            raise AssertionError("validation contents were accessed")
        return []

    monkeypatch.setattr(joint, "FIT_ROWS", fit_path)
    monkeypatch.setattr(joint, "VALIDATION_ROWS", validation_path)
    monkeypatch.setattr(joint, "FIT_SHA256", "fit-digest")
    monkeypatch.setattr(joint, "_sha256", fake_sha)
    monkeypatch.setattr(joint, "_load", fake_load)
    monkeypatch.setattr(joint, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(
        joint,
        "forward_audit",
        lambda _rows: ({"go": False}, [{"sample_id": "audit"}]),
    )

    joint._run(
        argparse.Namespace(
            fit_rows=fit_path,
            validation_rows=validation_path,
            out_dir=out_dir,
        )
    )

    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert fit_path in accessed
    assert validation_path not in accessed
    assert result["inputs"]["validation_rows"] == "unopened"
