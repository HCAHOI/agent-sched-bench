from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class PreemptionSnapshot:
    """Subset of vLLM metrics relevant to ENV-5 preemption analysis."""

    num_preemptions_total: float | None
    gpu_cache_usage_perc: float | None
    cpu_cache_usage_perc: float | None
    gpu_prefix_cache_hit_rate: float | None
    cpu_prefix_cache_hit_rate: float | None


@dataclass(slots=True)
class EvictionEvent:
    """Structured eviction event parsed from scheduler instrumentation logs."""

    seq_id: str
    tokens: int
    reason: str
    gpu_usage: float


EVICTION_PATTERN = re.compile(
    r"EVICT seq_id=(?P<seq_id>\S+) tokens=(?P<tokens>\d+) reason=(?P<reason>\S+) gpu_usage=(?P<gpu_usage>[0-9.]+)"
)


def parse_prometheus_metrics(metrics_payload: str) -> PreemptionSnapshot:
    """Extract the preemption-related metrics from a Prometheus payload."""
    wanted = {
        "vllm:num_preemptions_total": "num_preemptions_total",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc": "cpu_cache_usage_perc",
        "vllm:gpu_prefix_cache_hit_rate": "gpu_prefix_cache_hit_rate",
        "vllm:cpu_prefix_cache_hit_rate": "cpu_prefix_cache_hit_rate",
    }
    values: dict[str, float | None] = {field: None for field in wanted.values()}
    for line in metrics_payload.splitlines():
        for metric_name, field_name in wanted.items():
            if line.startswith(metric_name):
                values[field_name] = float(line.split()[-1])
    return PreemptionSnapshot(**values)


def parse_eviction_events(log_text: str) -> list[EvictionEvent]:
    """Parse scheduler hook eviction logs into structured events."""
    events: list[EvictionEvent] = []
    for line in log_text.splitlines():
        match = EVICTION_PATTERN.search(line)
        if not match:
            continue
        events.append(
            EvictionEvent(
                seq_id=match.group("seq_id"),
                tokens=int(match.group("tokens")),
                reason=match.group("reason"),
                gpu_usage=float(match.group("gpu_usage")),
            )
        )
    return events


def scheduler_log_snippet() -> str:
    """Return the recommended vLLM scheduler instrumentation snippet."""
    return (
        'logger.info(f"EVICT seq_id={seq.seq_id} tokens={seq.get_len()} '
        'reason={reason} gpu_usage={self.block_manager.gpu_utilization}")'
    )


def build_report(metrics_payload: str, log_text: str) -> dict[str, Any]:
    """Build a JSON-serializable ENV-5 report from metrics and scheduler logs."""
    snapshot = parse_prometheus_metrics(metrics_payload)
    events = parse_eviction_events(log_text)
    return {
        "metrics_fetch_ok": True,
        "baseline_provided": False,
        "preemption_counter_delta": None,
        "scheduler_log_provided": bool(log_text),
        "scheduler_hook_runtime_confirmed": bool(events),
        "evidence_scope": "current_run_log" if log_text else "cumulative_metrics_only",
        "preemption_snapshot": asdict(snapshot),
        "eviction_events": [asdict(event) for event in events],
        "instrumentation_snippet": scheduler_log_snippet(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and parse vLLM preemption metrics plus scheduler-hook logs."
    )
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--baseline-preemptions-total", type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = httpx.get(args.metrics_url, timeout=10.0)
    response.raise_for_status()
    metrics_payload = response.text
    log_text = Path(args.log_file).read_text(encoding="utf-8") if args.log_file else ""
    report = build_report(metrics_payload=metrics_payload, log_text=log_text)
    report["baseline_provided"] = args.baseline_preemptions_total is not None
    current_total = report["preemption_snapshot"]["num_preemptions_total"]
    if args.baseline_preemptions_total is not None and current_total is not None:
        report["preemption_counter_delta"] = current_total - args.baseline_preemptions_total
        report["evidence_scope"] = "baseline_delta"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
