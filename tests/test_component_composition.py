from scripts.evaluation.evaluate_component_composition import _compose


def _rows() -> tuple[dict, dict]:
    current = {
        "latency": 0,
        "peak_cpu_cores": "low",
        "sampled_peak_rss_mb": "low",
        "disk_read_write_bytes_total": "low",
    }
    pmfs = {
        "latency": [1, 0, 0, 0, 0],
        "peak_cpu_cores": [1, 0, 0],
        "sampled_peak_rss_mb": [1, 0, 0],
        "disk_read_write_bytes_total": [1, 0, 0],
    }
    base = {
        "candidate": dict(current),
        "candidate_probability_by_bucket": {k: list(v) for k, v in pmfs.items()},
        "provenance": {k: {"source": "current"} for k in current},
    }
    shared = {
        "sample_id": "task:0",
        "task_id": "task",
        "command": "pytest",
        "labels": {k: 0 for k in current},
        "current_dynamic": current,
        "current_probability_by_bucket": pmfs,
    }
    survival = {**shared, "arms": {"survival_disk": base}}
    phase = {
        **shared,
        "phase_applied_targets": ["latency", "peak_cpu_cores"],
        "full_test_phase": 2,
        "candidate": {
            **current,
            "latency": 4,
            "peak_cpu_cores": "high",
        },
        "candidate_probability_by_bucket": {
            **pmfs,
            "latency": [0, 0, 0, 0, 1],
            "peak_cpu_cores": [0, 0, 1],
        },
    }
    return survival, phase


def test_phase_monotonically_raises_only_declared_targets() -> None:
    survival, phase = _rows()

    row = _compose(survival, phase)

    assert row["phase_applied_targets"] == ["latency", "peak_cpu_cores"]
    assert row["arms"]["composition"]["candidate"] == {
        "latency": 4,
        "peak_cpu_cores": "high",
        "sampled_peak_rss_mb": "low",
        "disk_read_write_bytes_total": "low",
    }
    assert row["arms"]["survival_ablation"]["candidate"]["latency"] == 0


def test_phase_never_lowers_a_semantic_prediction() -> None:
    survival, phase = _rows()
    survival["arms"]["survival_disk"]["candidate"]["peak_cpu_cores"] = "high"
    survival["arms"]["survival_disk"]["candidate_probability_by_bucket"][
        "peak_cpu_cores"
    ] = [0, 0, 1]

    row = _compose(survival, phase)

    assert row["phase_applied_targets"] == ["latency"]
    assert row["arms"]["composition"]["candidate"]["peak_cpu_cores"] == "high"
