from __future__ import annotations
import hashlib
import json

from pathlib import Path
import subprocess
import tempfile

from scripts.exploration.analyze_prequential_profile_updates import (
    _PAIR_FIELDS,
    _ZstdJsonlWriter,
    _cleanup_partial_outputs,
    _aggregate_fold_run,
    _fold_task_sets,
    _load_config,
    _merge_cost_arm_outputs,
    _paired_point_estimates,
    _source_snapshot_records,
    build_parser,
)


def test_task_only_prequential_config_loads() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config, digest = _load_config(
        repo_root / "configs/experiments/prequential_profile_update.yaml"
    )
    assert config["task_order_seed"] == 0
    assert "order_sensitivity" not in config
    assert config["score_kv_costs_ms"] == [3500, 5000]
    assert config["schema_version"] == 3
    assert config["outer_folds"] == 5
    assert config["arms"] == [
        "frozen_100",
        "fresh4_static",
        "warmup_snapshot",
        "task",
    ]
    assert config["initialization_trace_inventory"]["trace_count"] == 100
    assert config["development_trace_inventory"]["trace_count"] == 277
    assert config["outputs"]["json"].endswith(
        "prequential-task-update-task-only/prequential-task-update.json"
    )
    assert len(digest) == 64


def test_historical_five_arm_config_is_preserved_exactly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    frozen = (
        repo_root
        / "analysis/results/prequential-task-update-20260721/prequential_profile_update.yaml"
    ).read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == (
        "f0b49926af9a4d8cf5029e53df24f27e700ee003876d9738555fd7b60b30bf20"
    )
    assert b"  - call\n" in frozen


def test_workers_cli_accepts_remote_pool_size() -> None:
    assert build_parser().parse_args(["--workers", "25"]).workers == 25


def test_adaptive_point_estimates_are_descriptive_per_order() -> None:
    rows = [
        {
            "sample_id": "a",
            "task_id": "t0",
            "latency_ms": 150.0,
            "kv_cost_ms": 100.0,
            "threshold_ms": 100.0,
            "restore_cost_ms": 94.0,
            "frozen_100_trigger_ms": 100.0,
            "fresh4_static_trigger_ms": 100.0,
            "warmup_snapshot_trigger_ms": 100.0,
            "task_trigger_ms": 50.0,
        },
        {
            "sample_id": "b",
            "task_id": "t1",
            "latency_ms": 150.0,
            "kv_cost_ms": 100.0,
            "threshold_ms": 100.0,
            "restore_cost_ms": 94.0,
            "frozen_100_trigger_ms": 100.0,
            "fresh4_static_trigger_ms": 100.0,
            "warmup_snapshot_trigger_ms": 100.0,
            "task_trigger_ms": 100.0,
        },
    ]
    points = _paired_point_estimates(rows, costs=[100.0])
    task = points["task_vs_frozen_100"]["100.0"]
    assert task["paired_delta_ms"] == 100.0
    assert (task["positive_task_count"], task["zero_task_count"]) == (1, 1)


def test_five_outer_folds_cover_each_task_once() -> None:
    task_ids = [f"t{i}" for i in range(277)]
    test_sets = [_fold_task_sets(task_ids, outer_fold=fold)[0] for fold in range(1, 6)]
    assert set().union(*test_sets) == set(task_ids)
    assert sum(len(test) for test in test_sets) == len(task_ids)
    assert all(
        test_sets[left].isdisjoint(test_sets[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )


def test_fold_aggregation_pools_tasks_instead_of_averaging_folds() -> None:
    folds = []
    for fold, task_count in enumerate((56, 56, 55, 55, 55), 1):
        point_estimates = {
            comparison: {
                str(cost): {
                    "kv_cost_ms": cost,
                    "call_count": task_count * 2,
                    "task_count": task_count,
                    "paired_delta_ms": float(fold * task_count),
                    "positive_task_count": task_count,
                    "negative_task_count": 0,
                    "zero_task_count": 0,
                }
                for cost in (3500.0, 5000.0)
            }
            for comparison in _PAIR_FIELDS
        }
        folds.append(
            {
                "runs": [
                    {
                        "order_run": "primary",
                        "point_estimates": point_estimates,
                        "timing": {},
                    }
                ]
            }
        )
    combined = _aggregate_fold_run(folds, run_name="primary", costs=[3500.0, 5000.0])
    point = combined["point_estimates"]["task_vs_warmup_snapshot"]["3500.0"]
    assert point["task_count"] == 277
    assert point["paired_delta_ms"] == sum(
        fold * task_count for fold, task_count in enumerate((56, 56, 55, 55, 55), 1)
    )
    assert "call_update_readiness" not in combined


def test_cost_split_reassembles_sample_major_panel_and_sums_cpu_time() -> None:
    common = {
        "arm": "frozen",
        "initial_profile_row_count": 10,
        "initial_profile_task_count": 2,
        "final_profile_row_count": 10,
        "final_profile_task_count": 2,
        "final_model_version": 0,
        "final_model_state_hash": "abc",
        "updates": [],
    }
    outputs = [
        {
            "test": {
                **common,
                "decisions": [
                    {
                        "sample_id": sample_id,
                        "kv_cost_ms": cost,
                        "score_panel_runtime_ms": runtime,
                    }
                    for sample_id, runtime in (("a", 1.0), ("b", 2.0))
                ],
            }
        }
        for cost in (3500.0, 5000.0)
    ]
    merged = _merge_cost_arm_outputs(outputs)["test"]["decisions"]
    assert [(row["sample_id"], row["kv_cost_ms"]) for row in merged] == [
        ("a", 3500.0),
        ("a", 5000.0),
        ("b", 3500.0),
        ("b", 5000.0),
    ]
    assert [row["score_panel_runtime_ms"] for row in merged] == [2.0, 2.0, 4.0, 4.0]


def test_source_snapshot_archives_exact_dirty_tree_bytes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    relative_path = "src/trace_collect/tool_latency_prequential.py"
    raw = (repo_root / relative_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    [record] = list(_source_snapshot_records({relative_path: digest}))
    assert record["sha256"] == digest
    assert record["content"].encode("utf-8") == raw


def test_partial_cleanup_removes_pid_staging_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        partial = Path(directory) / "records.jsonl.zst.partial"
        staging = partial.with_name(f".{partial.name}.123.tmp")
        partial.write_text("partial", encoding="utf-8")
        staging.write_text("staging", encoding="utf-8")
        _cleanup_partial_outputs([partial])
        assert not partial.exists()
        assert not staging.exists()


def test_zstd_jsonl_writer_streams_complete_records() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "records.jsonl.zst"
        writer = _ZstdJsonlWriter(path)
        writer.write_many([{"record_type": "a"}, {"record_type": "b", "value": 2}])
        writer.finish()
        completed = subprocess.run(
            ["zstd", "-q", "-d", "-c", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert [json.loads(line) for line in completed.stdout.splitlines()] == [
        {"record_type": "a"},
        {"record_type": "b", "value": 2},
    ]
    assert writer.count == 2
