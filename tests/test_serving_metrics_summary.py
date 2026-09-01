from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.summarize_serving_metrics import summarize


METRICS = {
    "vllm:prefix_cache_queries_total": (10, 310),
    "vllm:prefix_cache_hits_total": (4, 84),
    "vllm:num_preemptions_total": (2, 3),
    "vllm:prompt_tokens_total": (100, 400),
    "vllm:prompt_tokens_cached_total": (20, 155),
    "vllm:prompt_tokens_recomputed_total": (1, 2),
    "vllm:generation_tokens_total": (50, 56),
}


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _artifact(tmp_path: Path) -> dict[str, Path]:
    output_dir = tmp_path / "output"
    trace_path = output_dir / "combined.jsonl"
    requests = [
        ("task-a", "request-a0", 0, 100, None, 3, 1_001.0, 1_003.0),
        ("task-a", "request-a1", 1, 120, 119, 2, 1_010.0, 1_014.0),
        ("task-b", "request-b0", 0, 80, 16, 1, 1_020.0, 1_023.5),
    ]
    events = [{"type": "trace_metadata"}]
    for task, request_id, index, prompt, cached, generated, started, ended in requests:
        events.append(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": f"llm_{index}",
                "ts_start": started,
                "ts_end": ended,
                "data": {
                    "run_instance_id": task,
                    "shadow_generation": {
                        "request_id": request_id,
                        "source_action_index": index,
                        "prompt_tokens": prompt,
                        "cached_prompt_tokens": cached,
                        "requested_completion_tokens": generated,
                        "returned_completion_tokens": generated,
                        "finish_reason": "length",
                        "ttft_ms": index * 1_000 + (1_000 if task == "task-a" else 3_000),
                        "latency_ms": (ended - started) * 1_000,
                    },
                },
            }
        )
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    summary_path = output_dir / "throughput_summary.json"
    _json(
        summary_path,
        {
            "attempted_traces": 2,
            "completed_traces": 2,
            "failed_traces": 0,
            "llm_call_count": 3,
            "arrival_zero_wall_time_s": 1_000.0,
            "trace_file": str(trace_path),
            "tasks": [
                {
                    "run_instance_id": "task-a",
                    "arrival_s": 0.0,
                    "ready_to_terminal_s": 30.0,
                    "success": True,
                },
                {
                    "run_instance_id": "task-b",
                    "arrival_s": 10.0,
                    "ready_to_terminal_s": 40.0,
                    "success": True,
                },
            ],
        },
    )
    for task, second in (("task-a", "00"), ("task-b", "10")):
        _json(
            output_dir / task / "attempt_1" / "container_startup.json",
            {
                "status": "success",
                "run_instance_id": task,
                "started_at": f"2026-09-01T00:00:{second}Z",
            },
        )

    gpu_path = tmp_path / "gpu.csv"
    gpu_path.write_text(
        "timestamp_s,power_w,memory_mib,utilization_pct,memory_activity_pct\n"
        + "".join(
            f"{timestamp},200,20000,50,25\n"
            for timestamp in range(998, 1_053, 2)
        )
    )
    dram_bandwidth_path = tmp_path / "dram-bandwidth.csv"
    dram_bandwidth_path.write_text(
        "start_timestamp_ns,end_timestamp_ns,gpu_id,"
        "read_bytes_per_s,write_bytes_per_s\n"
        + "".join(
            f"{start * 1_000_000_000},{end * 1_000_000_000},0,1000000000,2000000000\n"
            for start, end in [(start, start + 1) for start in range(998, 1052)]
        )
    )
    start_prom = tmp_path / "start.prom"
    final_prom = tmp_path / "final.prom"
    start_prom.write_text(
        "".join(f'{name}{{engine="0"}} {values[0]}.0\n' for name, values in METRICS.items())
    )
    final_prom.write_text(
        "".join(f'{name}{{engine="0"}} {values[1]}.0\n' for name, values in METRICS.items())
    )
    kv_path = tmp_path / "kv.json"
    _json(
        kv_path,
        {
            "stored_blocks": 20,
            "removed_blocks": 3,
            "removed_tokens": 48,
            "clear_count": 0,
            "first_seq": 1,
            "last_seq": 2,
            "sequence_gaps": [],
            "batch_count": 2,
            "event_count": 23,
            "replayed_tail_batches": 1,
            "tail_replay_rounds": 3,
            "tail_replay_complete": True,
        },
    )
    return {
        "throughput_summary_path": summary_path,
        "gpu_csv_path": gpu_path,
        "dram_bandwidth_csv_path": dram_bandwidth_path,
        "prometheus_start_path": start_prom,
        "prometheus_final_path": final_prom,
        "kv_events_summary_path": kv_path,
        "output_path": tmp_path / "serving_metrics.json",
        "requests_output_path": tmp_path / "request_metrics.jsonl",
    }


def test_summarizes_serving_metrics(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)

    summary = summarize(**paths)

    assert summary["request_count"] == 3
    assert summary["window"]["scheduled_makespan_s"] == 50.0
    assert summary["request_token_totals"] == {
        "prompt_tokens": 300,
        "cached_prompt_tokens": 135,
        "recomputed_prompt_tokens": 1,
        "generation_tokens": 6,
    }
    assert summary["headline"]["request_usage_cached_prompt_token_ratio"] == pytest.approx(
        135 / 300
    )
    counters = summary["prometheus_counter_deltas"]
    assert counters["prefix_lookup_token_hit_ratio"] == pytest.approx(80 / 300)
    assert counters["recomputed_prompt_tokens"] == 1
    assert counters["preemptions"] == 1
    assert summary["ttft"]["first_request_per_task"]["count"] == 2
    assert summary["ttft"]["subsequent_requests"]["mean_s"] == 2.0
    assert summary["whole_run_generation_tokens_per_s"] == pytest.approx(6 / 50)
    assert summary["gpu"]["sample_count"] == 26
    assert summary["gpu"]["utilization"]["mean_pct"] == 50.0
    assert summary["gpu"]["memory_activity"]["mean_pct"] == 25.0
    assert summary["schema_version"] == 3
    bandwidth = summary["gpu"]["dram_bandwidth"]
    assert bandwidth["sample_count"] == 50
    assert bandwidth["max_sample_gap_s"] == 0
    assert bandwidth["scope"] == "CUDA context"
    assert bandwidth["read"]["mean_gb_per_s"] == 1.0
    assert bandwidth["write"]["p95_gb_per_s"] == 2.0
    assert bandwidth["total"]["max_gb_per_s"] == 3.0
    assert bandwidth["integrated_bytes"] == {
        "read": 50_000_000_000.0,
        "write": 100_000_000_000.0,
        "total": 150_000_000_000.0,
    }
    assert summary["kv_events"]["removed_tokens"] == 48
    assert summary["task_preparation_started_at"]["task-b"].endswith("10Z")

    rows = [json.loads(line) for line in paths["requests_output_path"].read_text().splitlines()]
    assert [row["request_id"] for row in rows] == ["request-a0", "request-a1", "request-b0"]
    assert rows[0]["cached_prompt_tokens"] == 0
    assert rows[0]["cached_prompt_tokens_omitted_zero"] is True
    assert rows[0]["first_request"] is True
    assert rows[0]["tpot_s"] == 0.5
    assert rows[0]["decode_tokens_per_s"] == 2.0
    assert rows[2]["tpot_s"] is None


def test_rejects_inconsistent_cached_usage_counter(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    final = paths["prometheus_final_path"]
    final.write_text(
        final.read_text().replace(
            'vllm:prompt_tokens_cached_total{engine="0"} 155.0',
            'vllm:prompt_tokens_cached_total{engine="0"} 156.0',
        )
    )

    with pytest.raises(ValueError, match="cached-token total differs"):
        summarize(**paths)


def test_accepts_backends_without_optional_prompt_counters(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    for name in ("prometheus_start_path", "prometheus_final_path"):
        path = paths[name]
        path.write_text(
            "".join(
                line
                for line in path.read_text().splitlines(keepends=True)
                if "prompt_tokens_cached" not in line
                and "prompt_tokens_recomputed" not in line
            )
        )

    summary = summarize(**paths)

    assert summary["prometheus_counter_deltas"]["cached_prompt_tokens"] is None
    assert summary["prometheus_counter_deltas"]["recomputed_prompt_tokens"] is None
    assert summary["request_token_totals"]["recomputed_prompt_tokens"] == 1


def test_uses_common_ready_for_staged_replay_window(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    throughput = paths["throughput_summary_path"]
    summary = json.loads(throughput.read_text())
    summary.pop("arrival_zero_wall_time_s")
    summary["common_ready_wall_time_s"] = 1_000.0
    _json(throughput, summary)

    result = summarize(**paths)

    assert result["window"]["start_source"] == "common_ready_wall_time_s"
    assert result["window"]["scheduled_makespan_s"] == 40.0


def test_records_gpu_telemetry_gap(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    gpu = paths["gpu_csv_path"]
    lines = gpu.read_text().splitlines()
    gpu.write_text("\n".join(line for line in lines if not line.startswith("1026,")) + "\n")

    summary = summarize(**paths)
    assert summary["gpu"]["max_sample_gap_s"] == 4


@pytest.mark.parametrize(
    "drop_start_s",
    [
        1000,
        1026,
        1049,
    ],
)
def test_records_dram_bandwidth_telemetry_gap(
    tmp_path: Path, drop_start_s: int
) -> None:
    paths = _artifact(tmp_path)
    bandwidth = paths["dram_bandwidth_csv_path"]
    lines = bandwidth.read_text().splitlines()
    prefix = f"{drop_start_s * 1_000_000_000},"
    bandwidth.write_text(
        "\n".join(line for line in lines if not line.startswith(prefix)) + "\n"
    )

    summary = summarize(**paths)
    bandwidth_summary = summary["gpu"]["dram_bandwidth"]
    assert bandwidth_summary["max_sample_gap_s"] == 1


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (",0,1000000000,", ",1,1000000000,", "row 2 is invalid"),
        (
            "998000000000,999000000000",
            "998000000000,998000000000",
            "row 2 is invalid",
        ),
        (",1000000000,2000000000", ",-1,2000000000", "rate is invalid"),
        (",1000000000,2000000000", ",nan,2000000000", "rate is invalid"),
    ],
)
def test_rejects_invalid_dram_bandwidth_sample(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    paths = _artifact(tmp_path)
    bandwidth = paths["dram_bandwidth_csv_path"]
    bandwidth.write_text(bandwidth.read_text().replace(old, new, 1))

    with pytest.raises(ValueError, match=message):
        summarize(**paths)


def test_rejects_overlapping_dram_bandwidth_samples(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    bandwidth = paths["dram_bandwidth_csv_path"]
    bandwidth.write_text(
        bandwidth.read_text().replace(
            "1002000000000,1003000000000", "1001500000000,1003000000000"
        )
    )

    with pytest.raises(ValueError, match="increasing and non-overlapping"):
        summarize(**paths)


def test_rejects_dram_bandwidth_without_window_coverage(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    bandwidth = paths["dram_bandwidth_csv_path"]
    bandwidth.write_text(bandwidth.read_text().splitlines()[0] + "\n")

    with pytest.raises(ValueError, match="has no samples in the scheduled run window"):
        summarize(**paths)


def test_rejects_cache_clear_during_measurement(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    kv = paths["kv_events_summary_path"]
    summary = json.loads(kv.read_text())
    summary["clear_count"] = 1
    _json(kv, summary)

    with pytest.raises(ValueError, match="cache was cleared"):
        summarize(**paths)


def test_rejects_incomplete_kv_tail_replay(tmp_path: Path) -> None:
    paths = _artifact(tmp_path)
    kv = paths["kv_events_summary_path"]
    summary = json.loads(kv.read_text())
    summary["tail_replay_complete"] = False
    _json(kv, summary)

    with pytest.raises(ValueError, match="tail replay did not complete"):
        summarize(**paths)
