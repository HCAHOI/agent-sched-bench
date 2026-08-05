from __future__ import annotations

import torch

from scripts.evaluation.evaluate_command_history_residual import (
    Row,
    command_shape,
    dataset,
    fit_model,
    run,
)


def test_residual_features_hide_literals_and_use_only_prior_labels() -> None:
    left = command_shape("pytest tests/secret_a.py --maxfail=1")
    right = command_shape("pytest tests/secret_b.py --maxfail=9")
    assert left == right
    assert command_shape("x && x").executables == ("x",)

    def row(sample: str, label: int) -> Row:
        return Row(
            sample_id=sample,
            task_id="owner__repo-1",
            command="pytest tests/secret_a.py --maxfail=1",
            labels={"latency": label},
            current={"latency": 0},
            pmfs={"latency": (0.5, 0.5, 0.0, 0.0, 0.0)},
        )

    x, base, labels = dataset([row("a", 0), row("b", 1)], "latency")
    assert torch.equal(x[0, -33:], torch.zeros(33, dtype=torch.float64))
    assert not torch.equal(x[1, -33:], x[0, -33:])
    weight, bias, initial, final = fit_model(x, base, labels)
    assert weight.shape == (5, x.shape[1])
    assert bias.shape == (5,)
    assert final < initial

    missing = row("c", 1)
    missing = Row(
        sample_id=missing.sample_id,
        task_id="owner__repo-2",
        command=missing.command,
        labels=missing.labels,
        current={"latency": None},
        pmfs={"latency": None},
    )
    _missing_x, missing_base, _missing_y = dataset([missing], "latency")
    assert torch.equal(missing_base, torch.full((1, 5), 0.2, dtype=torch.float64))


def test_residual_run_is_deterministic_fail_closed_and_preserves_disk() -> None:
    def row(task: int, call: int, high: bool, *, missing_latency: bool = False) -> Row:
        label = 4 if high else 0
        resource = 2 if high else 0
        current = {
            "latency": None if missing_latency else 0,
            "peak_cpu_cores": "low",
            "sampled_peak_rss_mb": "low",
            "disk_read_write_bytes_total": "medium",
        }
        return Row(
            sample_id=f"owner__repo-{task}:{call}",
            task_id=f"owner__repo-{task}",
            command="pytest tests/<PATH>" if high else "git status --short",
            labels={
                "latency": label,
                "peak_cpu_cores": resource,
                "sampled_peak_rss_mb": resource,
                "disk_read_write_bytes_total": 1,
            },
            current=current,
            pmfs={
                "latency": None if missing_latency else (0.8, 0.05, 0.05, 0.05, 0.05),
                "peak_cpu_cores": (0.8, 0.1, 0.1),
                "sampled_peak_rss_mb": (0.8, 0.1, 0.1),
                "disk_read_write_bytes_total": (0.1, 0.8, 0.1),
            },
        )

    fit = [
        row(task, call, high=(task + call) % 2 == 0)
        for task in range(1, 4)
        for call in range(2)
    ]
    validation = [
        row(4, 0, True, missing_latency=True),
        row(4, 1, False),
        row(5, 0, True),
    ]
    first, first_rows = run(fit, validation)
    second, second_rows = run(fit, validation)

    assert first["models"] == second["models"]
    assert first_rows == second_rows
    assert first["targets"]["latency"]["current"]["prediction_unavailable"] == 1
    assert first["targets"]["latency"]["current"]["exact_class_accuracy"] is not None
    assert first["targets"]["latency"]["candidate"]["prediction_unavailable"] == 0
    assert first["gate"]["minimum_gain_percentage_points_each"] == 5.0
    assert first["gate"]["minimum_helpful_tasks"] == 10
    assert first["gate"]["disk_bit_identical"] is True
    assert all(
        item["candidate"]["disk_read_write_bytes_total"]
        == item["current_dynamic"]["disk_read_write_bytes_total"]
        and item["candidate_probability_by_bucket"]["disk_read_write_bytes_total"]
        == item["current_probability_by_bucket"]["disk_read_write_bytes_total"]
        for item in first_rows
    )

    x, _base, _labels = dataset(validation, "latency")
    history_width = 3 * (1 + 2 * 5)
    assert torch.equal(x[0, -history_width:], x[2, -history_width:])
