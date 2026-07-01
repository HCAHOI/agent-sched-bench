"""Action-level resource timeline collection for trace replay.

The collector stores per-tool deltas instead of only aggregate container stats so
simulate can later reason about source-equivalent progress under different CPU
and network conditions.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RESOURCE_TIMELINE_VERSION = 1
_DEFAULT_SAMPLE_INTERVAL_S = 0.5
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_NET_DEV = Path("/proc/net/dev")


@dataclass(frozen=True, slots=True)
class ResourceReading:
    """One cumulative resource reading from the current runtime."""

    monotonic_s: float
    cpu_usage_s: float | None
    net_rx_bytes: int | None
    net_tx_bytes: int | None
    cpu_quota_cores: float | None


@dataclass(frozen=True, slots=True)
class ResourceDelta:
    """One interval delta suitable for trace serialization."""

    offset_s: float
    dt_s: float
    cpu_core_s: float | None
    net_rx_bytes: int | None
    net_tx_bytes: int | None
    cpu_quota_cores: float | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "offset_s": round(self.offset_s, 6),
            "dt_s": round(self.dt_s, 6),
        }
        if self.cpu_core_s is not None:
            payload["cpu_core_s"] = round(self.cpu_core_s, 6)
        if self.net_rx_bytes is not None:
            payload["net_rx_bytes"] = int(self.net_rx_bytes)
        if self.net_tx_bytes is not None:
            payload["net_tx_bytes"] = int(self.net_tx_bytes)
        if self.cpu_quota_cores is not None:
            payload["cpu_quota_cores"] = round(self.cpu_quota_cores, 6)
            payload["cpu_opportunity_core_s"] = round(
                self.cpu_quota_cores * self.dt_s,
                6,
            )
        return payload


def read_cgroup_cpu_usage_s(cgroup_root: Path = _CGROUP_ROOT) -> float | None:
    """Return cumulative cgroup CPU usage in seconds, if available."""

    try:
        text = (cgroup_root / "cpu.stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "usage_usec":
            try:
                return int(parts[1]) / 1_000_000.0
            except ValueError:
                return None
    return None


def read_cgroup_cpu_quota_cores(cgroup_root: Path = _CGROUP_ROOT) -> float | None:
    """Return cgroup CPU quota as cores, falling back to cpuset/host count."""

    quota_path = cgroup_root / "cpu.max"
    try:
        parts = quota_path.read_text(encoding="utf-8").strip().split()
    except (FileNotFoundError, PermissionError, OSError):
        parts = []
    if len(parts) >= 2:
        quota_raw, period_raw = parts[0], parts[1]
        if quota_raw != "max":
            try:
                quota = float(quota_raw)
                period = float(period_raw)
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return quota / period

    cpuset_count = _read_cpuset_cpu_count(cgroup_root)
    if cpuset_count is not None:
        return float(cpuset_count)
    cpu_count = os.cpu_count()
    return float(cpu_count) if cpu_count and cpu_count > 0 else None


def _read_cpuset_cpu_count(cgroup_root: Path) -> int | None:
    for filename in ("cpuset.cpus.effective", "cpuset.cpus"):
        try:
            text = (cgroup_root / filename).read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        count = _count_cpuset_cpus(text)
        if count is not None:
            return count
    return None


def _count_cpuset_cpus(value: str) -> int | None:
    if not value:
        return None
    total = 0
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except ValueError:
                return None
            if end < start:
                return None
            total += end - start + 1
        else:
            try:
                int(item)
            except ValueError:
                return None
            total += 1
    return total or None


def read_proc_net_bytes(path: Path = _PROC_NET_DEV) -> tuple[int, int] | None:
    """Return cumulative RX/TX bytes across non-loopback interfaces."""

    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    rx_total = 0
    tx_total = 0
    found = False
    for line in text.splitlines()[2:]:
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        if iface.strip() == "lo":
            continue
        fields = rest.split()
        if len(fields) < 16:
            continue
        try:
            rx_total += int(fields[0])
            tx_total += int(fields[8])
        except ValueError:
            continue
        found = True
    return (rx_total, tx_total) if found else None


def read_resource_reading(
    *,
    cgroup_root: Path = _CGROUP_ROOT,
    proc_net_dev: Path = _PROC_NET_DEV,
) -> ResourceReading:
    """Read the current cumulative CPU and network counters."""

    net_bytes = read_proc_net_bytes(proc_net_dev)
    return ResourceReading(
        monotonic_s=time.monotonic(),
        cpu_usage_s=read_cgroup_cpu_usage_s(cgroup_root),
        net_rx_bytes=net_bytes[0] if net_bytes is not None else None,
        net_tx_bytes=net_bytes[1] if net_bytes is not None else None,
        cpu_quota_cores=read_cgroup_cpu_quota_cores(cgroup_root),
    )


def resource_delta(previous: ResourceReading, current: ResourceReading) -> ResourceDelta | None:
    """Build a non-negative delta between two cumulative readings."""

    dt_s = current.monotonic_s - previous.monotonic_s
    if dt_s <= 0:
        return None

    cpu_core_s: float | None = None
    if previous.cpu_usage_s is not None and current.cpu_usage_s is not None:
        cpu_core_s = max(0.0, current.cpu_usage_s - previous.cpu_usage_s)

    net_rx_bytes: int | None = None
    if previous.net_rx_bytes is not None and current.net_rx_bytes is not None:
        net_rx_bytes = max(0, current.net_rx_bytes - previous.net_rx_bytes)

    net_tx_bytes: int | None = None
    if previous.net_tx_bytes is not None and current.net_tx_bytes is not None:
        net_tx_bytes = max(0, current.net_tx_bytes - previous.net_tx_bytes)

    if cpu_core_s is None and net_rx_bytes is None and net_tx_bytes is None:
        return None

    return ResourceDelta(
        offset_s=current.monotonic_s,
        dt_s=dt_s,
        cpu_core_s=cpu_core_s,
        net_rx_bytes=net_rx_bytes,
        net_tx_bytes=net_tx_bytes,
        cpu_quota_cores=current.cpu_quota_cores,
    )


class ResourceTimelineRecorder:
    """Async sampler for one tool action's CPU and network deltas."""

    def __init__(
        self,
        *,
        sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL_S,
        cgroup_root: Path = _CGROUP_ROOT,
        proc_net_dev: Path = _PROC_NET_DEV,
        enabled: bool = True,
        scope: str = "cgroup_interval",
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        self.sample_interval_s = sample_interval_s
        self.cgroup_root = cgroup_root
        self.proc_net_dev = proc_net_dev
        self.enabled = enabled
        self.scope = scope
        self._started_at_monotonic: float | None = None
        self._last_reading: ResourceReading | None = None
        self._samples: list[ResourceDelta] = []
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "ResourceTimelineRecorder":
        if not self.enabled:
            return self
        initial = read_resource_reading(
            cgroup_root=self.cgroup_root,
            proc_net_dev=self.proc_net_dev,
        )
        self._started_at_monotonic = initial.monotonic_s
        self._last_reading = initial
        self._task = asyncio.create_task(self._sample_loop())
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._task is not None:
            await self._task
        self._append_current_delta()

    async def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.sample_interval_s,
                )
            except asyncio.TimeoutError:
                self._append_current_delta()

    def _append_current_delta(self) -> None:
        previous = self._last_reading
        started_at = self._started_at_monotonic
        if previous is None or started_at is None:
            return
        current = read_resource_reading(
            cgroup_root=self.cgroup_root,
            proc_net_dev=self.proc_net_dev,
        )
        delta = resource_delta(previous, current)
        self._last_reading = current
        if delta is None:
            return
        self._samples.append(
            ResourceDelta(
                offset_s=current.monotonic_s - started_at,
                dt_s=delta.dt_s,
                cpu_core_s=delta.cpu_core_s,
                net_rx_bytes=delta.net_rx_bytes,
                net_tx_bytes=delta.net_tx_bytes,
                cpu_quota_cores=delta.cpu_quota_cores,
            )
        )

    def to_trace_dict(self) -> dict[str, Any] | None:
        """Return JSON-serializable trace data, or None when unavailable."""

        if not self._samples:
            return None
        samples = [sample.to_dict() for sample in self._samples]
        cpu_core_s = sum(
            float(sample.get("cpu_core_s", 0.0)) for sample in samples
        )
        net_rx_bytes = sum(int(sample.get("net_rx_bytes", 0)) for sample in samples)
        net_tx_bytes = sum(int(sample.get("net_tx_bytes", 0)) for sample in samples)
        wall_s = sum(float(sample.get("dt_s", 0.0)) for sample in samples)
        return {
            "version": _RESOURCE_TIMELINE_VERSION,
            "source": "cgroup_cpu_proc_net",
            "scope": self.scope,
            "sample_interval_s": self.sample_interval_s,
            "samples": samples,
            "summary": {
                "sample_count": len(samples),
                "wall_s": round(wall_s, 6),
                "cpu_core_s": round(cpu_core_s, 6),
                "net_rx_bytes": net_rx_bytes,
                "net_tx_bytes": net_tx_bytes,
            },
        }


def valid_resource_timeline(value: Any) -> dict[str, Any] | None:
    """Return value when it looks like a v1 resource timeline."""

    if not isinstance(value, dict):
        return None
    if value.get("version") != _RESOURCE_TIMELINE_VERSION:
        return None
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    valid_samples = [sample for sample in samples if isinstance(sample, dict)]
    if not valid_samples:
        return None
    return value
