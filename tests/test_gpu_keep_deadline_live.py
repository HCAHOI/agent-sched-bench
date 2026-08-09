from __future__ import annotations

import copy
import datetime as dt
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _module():
    return importlib.import_module(
        "scripts.evaluation.evaluate_gpu_keep_deadline_live"
    )


def test_causal_reuse_requires_free_admission_restore_order_and_block_overlap() -> None:
    deadline = {
        "requests": [
            {
                "request_id": "deadline:0:0",
                "program_index": 0,
                "turn_index": 0,
            },
            {
                "request_id": "deadline:0:1",
                "program_index": 0,
                "turn_index": 1,
            },
            {
                "request_id": "deadline:1:0",
                "program_index": 1,
                "turn_index": 0,
            },
        ],
        "transfers": [
            {
                "phase": "retention_offload",
                "request_id": "deadline:0:0",
                "program_id": "program:0",
                "completed_monotonic_s": 2.0,
            },
            {
                "phase": "retention_restore",
                "request_id": "deadline:0:1",
                "program_id": "program:0",
                "started_monotonic_s": 5.0,
            },
        ],
        "retention_events": [
            {
                "phase": "retention_blocks_freed",
                "request_id": "deadline:0:0",
                "monotonic_s": 2.1,
                "block_ids": [1, 2],
                "free_blocks_before": 7,
                "free_blocks_after": 9,
            },
            {
                "phase": "retention_request_admitted",
                "request_id": "deadline:1:0",
                "monotonic_s": 3.0,
                "block_ids": [2, 3],
                "free_blocks_before": 9,
                "free_blocks_after": 7,
            },
            {
                "phase": "retention_blocks_freed",
                "request_id": "deadline:1:0",
                "monotonic_s": 4.0,
                "block_ids": [2],
                "free_blocks_before": 7,
                "free_blocks_after": 8,
            },
            {
                "phase": "retention_request_admitted",
                "request_id": "deadline:1:0",
                "monotonic_s": 5.1,
                "block_ids": [1],
                "free_blocks_before": 9,
                "free_blocks_after": 8,
            },
        ],
    }

    events = _module().causal_reuse_events(deadline)

    assert len(events) == 1
    assert events[0]["owner_program_index"] == 0
    assert events[0]["admitted_request_key"] == [1, 0]
    assert events[0]["reused_block_ids"] == [2]
    assert events[0]["freed_block_seconds"] == pytest.approx(2 * 2.9)
    assert events[0]["reused_block_seconds"] == pytest.approx(1.0)


def test_output_parity_rejects_a_changed_generated_token() -> None:
    reference = {
        "requests": [
            {
                "request_id": "keep:0:0",
                "program_index": 0,
                "turn_index": 0,
                "task_id": "t0",
                "messages_in": [{"role": "user", "content": "x"}],
                "prompt_token_ids_sha256": "abc",
                "output_token_ids": [1, 2],
                "finish_reason": "length",
            }
        ],
        "programs": [{"program_index": 0, "task_id": "t0", "status": "replayed_complete"}],
    }
    changed = {
        "requests": [
            {
                **reference["requests"][0],
                "request_id": "deadline:0:0",
                "output_token_ids": [1, 3],
            }
        ],
        "programs": list(reference["programs"]),
    }

    with pytest.raises(ValueError, match="output parity"):
        _module().validate_output_parity(reference, changed)


def test_frozen_cells_reject_a_different_replay_root() -> None:
    module = _module()
    config = yaml.safe_load(Path("configs/serving/w5_multitenant.yaml").read_text())
    workload = next(
        row
        for row in config["workloads"]
        if row["name"] == "swe-rebench-277-development-exposed"
    )
    task_ids = Path(workload["task_ids_file"]).read_text().splitlines()
    base = dt.datetime(2026, 8, 9, tzinfo=dt.timezone.utc)
    cells = []
    for index, policy in enumerate(("keep", "deadline", "deadline", "keep")):
        cells.append(
            {
                "status": "complete",
                "policy": policy,
                "load": 8,
                "program_count": 277,
                "request_count": 13_048,
                "limit_programs": None,
                "max_turns": None,
                "git_sha": module._git_sha(),
                "vllm_version": "0.11.2",
                "runtime": {
                    "host_name": "0091-dsm2-sma100-prxmx70124",
                    "device_name": "NVIDIA A100 80GB PCIe",
                },
                "config": copy.deepcopy(config),
                "workload": copy.deepcopy(workload),
                "replay_task_ids": task_ids,
                "programs": [
                    {"program_index": number, "task_id": task_id}
                    for number, task_id in enumerate(task_ids)
                ],
                "transfers": [],
                "cell_started_at": (base + dt.timedelta(minutes=2 * index)).isoformat(),
                "cell_finished_at": (
                    base + dt.timedelta(minutes=2 * index + 1)
                ).isoformat(),
            }
        )

    module.validate_frozen_cells(cells)
    cells[1]["workload"]["replay_trace_root"] = "traces/other"
    with pytest.raises(ValueError, match="workload"):
        module.validate_frozen_cells(cells)


def test_vllm_011_request_metrics_use_state_stats_timestamps(monkeypatch) -> None:
    module = importlib.import_module("spike.run_multitenant")
    metrics = SimpleNamespace(
        queued_ts=1.0,
        scheduled_ts=1.25,
        first_token_ts=1.75,
        last_token_ts=3.0,
    )
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)

    row = module._request_metrics(
        SimpleNamespace(metrics=metrics), submitted_at=5.0, first_token_at=6.5
    )

    assert row == {
        "queue_ms": 250.0,
        "prefill_ms": 500.0,
        "ttft_ms": 1500.0,
        "latency_ms": 5000.0,
    }
