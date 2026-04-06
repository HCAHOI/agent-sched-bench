from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import httpx


class VLLMMetricsCollector:
    """Periodic collector for Prometheus-style vLLM metrics snapshots."""

    METRICS_OF_INTEREST = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:e2e_request_latency_seconds",
        "vllm:time_to_first_token_seconds",
    ]

    HISTOGRAM_METRICS = {
        "vllm:e2e_request_latency_seconds",
        "vllm:time_to_first_token_seconds",
    }

    def __init__(
        self,
        metrics_url: str = "http://localhost:8000/metrics",
        *,
        gpu_sample_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.url = metrics_url
        self.snapshots: list[dict[str, Any]] = []
        self.gpu_sample_provider = gpu_sample_provider or sample_nvidia_smi

    def _parse_prometheus(self, metrics_payload: str) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for line in metrics_payload.splitlines():
            if not line or line.startswith("#"):
                continue
            for metric_name in self.METRICS_OF_INTEREST:
                if metric_name in self.HISTOGRAM_METRICS:
                    if line.startswith(f"{metric_name}_sum"):
                        entry = snapshot.setdefault(metric_name, {})
                        entry["_sum"] = float(line.split()[-1])
                    elif line.startswith(f"{metric_name}_count"):
                        entry = snapshot.setdefault(metric_name, {})
                        entry["_count"] = float(line.split()[-1])
                elif line.startswith(metric_name):
                    snapshot[metric_name] = float(line.split()[-1])
        return snapshot

    def _validate_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        missing = [
            metric for metric in self.METRICS_OF_INTEREST if metric not in snapshot
        ]
        if missing:
            logging.warning(f"Incomplete metrics snapshot, missing: {missing}")
        return snapshot

    async def poll(
        self, interval_s: float = 1.0, max_samples: int | None = None
    ) -> list[dict[str, Any]]:
        """Poll the metrics endpoint until cancelled or max_samples reached."""
        self.snapshots = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            while max_samples is None or len(self.snapshots) < max_samples:
                response = await client.get(self.url)
                response.raise_for_status()
                snapshot = self._parse_prometheus(response.text)
                self._validate_snapshot(snapshot)
                snapshot["timestamp"] = time.time()  # noqa: DTZ003
                snapshot["gpu_samples"] = self.gpu_sample_provider()
                self.snapshots.append(snapshot)
                await asyncio.sleep(interval_s)
        return self.snapshots

    def dump_json(self, output_path: Path) -> None:
        """Write collected snapshots to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.snapshots, indent=2) + "\n", encoding="utf-8"
        )


def parse_nvidia_smi_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output into records."""
    rows: list[dict[str, Any]] = []
    reader = csv.reader(line for line in csv_text.splitlines() if line.strip())
    for row in reader:
        if len(row) < 2:
            continue
        rows.append(
            {
                "utilization_gpu": float(row[0].strip()),
                "memory_used_mib": float(row[1].strip()),
            }
        )
    return rows


def sample_nvidia_smi() -> list[dict[str, Any]]:
    """Collect one GPU sample via nvidia-smi."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return parse_nvidia_smi_csv(output)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def dump_nvidia_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Write GPU utilization samples to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect vLLM metrics snapshots.")
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = VLLMMetricsCollector(metrics_url=args.metrics_url)
    asyncio.run(
        collector.poll(interval_s=args.interval_s, max_samples=args.max_samples)
    )
    collector.dump_json(Path(args.output))


if __name__ == "__main__":
    main()
