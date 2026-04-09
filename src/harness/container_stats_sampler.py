"""Background sampler for container CPU and memory statistics.

The summary format matches the harness resources.json schema.
"""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any

# The CC harness format — pipe-delimited, three fields.
_PODMAN_FORMAT = "{{.MemUsage}}|{{.MemPerc}}|{{.CPUPerc}}"

def _parse_pipe_stats(raw: str) -> dict[str, Any] | None:
    parts = (raw or "").strip().split("|")
    if len(parts) < 3:
        return None
    now = datetime.now(tz=timezone.utc)
    return {
        "timestamp": now.isoformat().replace("+00:00", ""),
        "epoch": now.timestamp(),
        "mem_usage": parts[0],
        "mem_percent": parts[1],
        "cpu_percent": parts[2],
    }

def _parse_memory_mb(mem_usage: str) -> float | None:
    if not mem_usage:
        return None
    left = mem_usage.split("/")[0].strip()
    try:
        if left.endswith("GiB"):
            return float(left[:-3]) * 1024
        if left.endswith("MiB"):
            return float(left[:-3])
        if left.endswith("KiB"):
            return float(left[:-3]) / 1024
        if left.endswith("GB"):
            return float(left[:-2]) * 1000
        if left.endswith("MB"):
            return float(left[:-2])
        if left.endswith("KB"):
            return float(left[:-2]) / 1000
        if left.endswith("kB"):
            return float(left[:-2]) / 1000
        if left.endswith("B"):
            return float(left[:-1]) / (1024 * 1024)
    except ValueError:
        return None
    return None

def _parse_percent(value: str) -> float | None:
    try:
        return float(value.replace("%", "").strip())
    except (ValueError, AttributeError):
        return None

def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute CC-compatible resources.json summary from a sample list.

    Matches ``ResourceMonitor.get_summary()`` in the agentcgroup reference
    repo byte-for-byte: {sample_count, duration_seconds, memory_mb, cpu_percent}
    where memory_mb/cpu_percent each contain min/max/avg.
    """
    if not samples:
        return {
            "sample_count": 0,
            "duration_seconds": 0,
            "memory_mb": {"min": 0, "max": 0, "avg": 0},
            "cpu_percent": {"min": 0, "max": 0, "avg": 0},
        }

    mem_values: list[float] = []
    cpu_values: list[float] = []
    for sample in samples:
        mem_mb = _parse_memory_mb(sample.get("mem_usage", ""))
        if mem_mb is not None:
            mem_values.append(mem_mb)
        cpu_val = _parse_percent(sample.get("cpu_percent", ""))
        if cpu_val is not None:
            cpu_values.append(cpu_val)

    duration = 0.0
    if len(samples) > 1:
        duration = float(samples[-1]["epoch"]) - float(samples[0]["epoch"])

    return {
        "sample_count": len(samples),
        "duration_seconds": duration,
        "memory_mb": {
            "min": min(mem_values) if mem_values else 0,
            "max": max(mem_values) if mem_values else 0,
            "avg": sum(mem_values) / len(mem_values) if mem_values else 0,
        },
        "cpu_percent": {
            "min": min(cpu_values) if cpu_values else 0,
            "max": max(cpu_values) if cpu_values else 0,
            "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
        },
    }

class ContainerStatsSampler(threading.Thread):
    """Background thread that samples ``podman stats`` for one container.

    Usage::

        sampler = ContainerStatsSampler(container_id="abc123", interval_s=1.0)
        sampler.start()
        # ... agent runs ...
        samples = sampler.stop()
        summary = summarize_samples(samples)

    ``stop()`` is idempotent and safe to call multiple times.
    """

    def __init__(
        self,
        container_id: str,
        *,
        interval_s: float = 1.0,
        executable: str = "podman",
        subprocess_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(daemon=True, name=f"stats-{container_id[:12]}")
        self.container_id = container_id
        self.interval_s = interval_s
        self.executable = executable
        self.subprocess_timeout_s = subprocess_timeout_s
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            try:
                result = subprocess.run(
                    [
                        self.executable,
                        "stats",
                        "--no-stream",
                        "--format",
                        _PODMAN_FORMAT,
                        self.container_id,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout_s,
                    check=False,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                break
            if result.returncode != 0:
                break
            sample = _parse_pipe_stats(result.stdout)
            if sample is not None:
                self._samples.append(sample)
            elapsed = time.monotonic() - tick_start
            remainder = max(0.0, self.interval_s - elapsed)
            if self._stop_event.wait(remainder):
                break

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=self.subprocess_timeout_s + self.interval_s + 1.0)
        return list(self._samples)

    def get_summary(self) -> dict[str, Any]:
        return summarize_samples(self._samples)
