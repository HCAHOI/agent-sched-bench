from __future__ import annotations

from harness.scheduler_hooks import parse_prometheus_metrics


def test_parse_prometheus_metrics_extracts_preemption_fields() -> None:
    metrics_payload = "\n".join(
        [
            "vllm:num_preemptions_total 3",
            "vllm:gpu_cache_usage_perc 75.0",
            "vllm:cpu_cache_usage_perc 0.0",
            "vllm:gpu_prefix_cache_hit_rate 0.5",
            "vllm:cpu_prefix_cache_hit_rate 0.0",
        ]
    )
    snapshot = parse_prometheus_metrics(metrics_payload)
    assert snapshot.num_preemptions_total == 3.0
    assert snapshot.gpu_cache_usage_perc == 75.0
    assert snapshot.gpu_prefix_cache_hit_rate == 0.5
