from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

logger = logging.getLogger(__name__)

EVENT_SOURCE_ROOT = Path("/sys/bus/event_source/devices")
DEFAULT_PERF_EXECUTABLE = "perf"
INTEL_CAS_BYTES = 64.0


@dataclass(frozen=True, slots=True)
class MemoryBandwidthReading:
    available: bool
    source: str | None = None
    total_mb_s: float | None = None
    read_mb_s: float | None = None
    write_mb_s: float | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PerfEventBackend:
    kind: Literal["intel_imc_cas", "explicit_byte_events"]
    source: str
    read_specs: tuple[str, ...]
    write_specs: tuple[str, ...]
    bytes_per_count: float


@dataclass(frozen=True, slots=True)
class CgroupMemoryAccessBackend:
    source: str
    event_specs: tuple[str, ...]

    @property
    def event_spec(self) -> str:
        return ",".join(self.event_specs)


@dataclass(frozen=True, slots=True)
class CgroupMemoryAccessMeasurement:
    cgroup: str
    events: float
    events_per_s: float


@dataclass(frozen=True, slots=True)
class CgroupMemoryAccessReading:
    available: bool
    source: str | None = None
    measurements: Mapping[str, CgroupMemoryAccessMeasurement] = field(
        default_factory=dict
    )
    reason: str | None = None
    started_epoch: float | None = None
    ended_epoch: float | None = None


def _event_aliases(device_path: Path) -> set[str]:
    events_dir = device_path / "events"
    if not events_dir.is_dir():
        return set()
    return {child.name for child in events_dir.iterdir() if child.is_file()}


def _detect_intel_imc_backend(
    root: Path = EVENT_SOURCE_ROOT,
) -> PerfEventBackend | None:
    read_specs: list[str] = []
    write_specs: list[str] = []
    for device in sorted(root.glob("uncore_imc_*")):
        aliases = _event_aliases(device)
        if "cas_count_read" not in aliases or "cas_count_write" not in aliases:
            continue
        read_specs.append(f"{device.name}/cas_count_read/")
        write_specs.append(f"{device.name}/cas_count_write/")
    if not read_specs or not write_specs:
        return None
    return PerfEventBackend(
        kind="intel_imc_cas",
        source="perf:intel-imc-cas",
        read_specs=tuple(read_specs),
        write_specs=tuple(write_specs),
        bytes_per_count=INTEL_CAS_BYTES,
    )


_READ_BYTE_EVENT_NAMES = (
    "read_bytes",
    "bytes_read",
    "data_read_bytes",
    "data_read",
)
_WRITE_BYTE_EVENT_NAMES = (
    "write_bytes",
    "bytes_write",
    "data_write_bytes",
    "data_write",
)


def _pick_alias(aliases: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in aliases:
            return name
    return None


def _detect_explicit_byte_backend(
    root: Path = EVENT_SOURCE_ROOT,
) -> PerfEventBackend | None:
    read_specs: list[str] = []
    write_specs: list[str] = []
    for device in sorted(root.iterdir()) if root.exists() else []:
        aliases = _event_aliases(device)
        if not aliases:
            continue
        read_alias = _pick_alias(aliases, _READ_BYTE_EVENT_NAMES)
        write_alias = _pick_alias(aliases, _WRITE_BYTE_EVENT_NAMES)
        if read_alias is None or write_alias is None:
            continue
        read_specs.append(f"{device.name}/{read_alias}/")
        write_specs.append(f"{device.name}/{write_alias}/")
    if not read_specs or not write_specs:
        return None
    return PerfEventBackend(
        kind="explicit_byte_events",
        source="perf:explicit-byte-events",
        read_specs=tuple(read_specs),
        write_specs=tuple(write_specs),
        bytes_per_count=1.0,
    )


def detect_perf_backend(root: Path = EVENT_SOURCE_ROOT) -> PerfEventBackend | None:
    return _detect_intel_imc_backend(root) or _detect_explicit_byte_backend(root)


def detect_cgroup_memory_access_backend(
    root: Path = EVENT_SOURCE_ROOT,
) -> CgroupMemoryAccessBackend | None:
    """Find a PMU event usable for per-cgroup memory-access counting.

    This deliberately reports access events, not memory bandwidth bytes. On ARM
    hosts the common architectural PMU exposes ``mem_access`` but not a portable
    read/write byte counter.
    """
    specs: list[str] = []
    devices: list[str] = []
    for device in sorted(root.iterdir()) if root.exists() else []:
        aliases = _event_aliases(device)
        if "mem_access" not in aliases:
            continue
        devices.append(device.name)
        specs.append(f"{device.name}/mem_access/")
    if not specs:
        return None
    return CgroupMemoryAccessBackend(
        source=f"perf:{'+'.join(devices)}:mem_access:cgroup",
        event_specs=tuple(specs),
    )


def _parse_perf_count(raw: str) -> float | None:
    value = raw.strip()
    if not value or value in {"<not counted>", "<not supported>"}:
        return None
    value = value.replace(" ", "")
    try:
        return float(value)
    except ValueError:
        return None


def _parse_perf_stat_output(
    text: str,
    event_specs: tuple[str, ...],
) -> dict[str, float] | None:
    counts: dict[str, float] = {}
    for line in text.splitlines():
        for spec in event_specs:
            if spec not in line:
                continue
            first_field = line.split(",", 1)[0]
            value = _parse_perf_count(first_field)
            if value is None:
                return None
            counts[spec] = value
            break
    if len(counts) != len(event_specs):
        return None
    return counts


def _classify_perf_failure(stderr: str) -> str:
    message = stderr.lower()
    if "permission" in message or "access to performance monitoring" in message:
        return "permission_denied"
    if "not supported" in message:
        return "pmu_unsupported"
    if "not found" in message:
        return "perf_missing"
    return "perf_error"


def _parse_perf_cgroup_count_output(
    text: str,
    *,
    event_specs: tuple[str, ...],
    cgroups: Mapping[str, str],
) -> tuple[dict[str, float], bool]:
    counts: dict[str, float] = {}
    saw_matching_cgroup = False
    for line in text.splitlines():
        if not any(spec in line for spec in event_specs):
            continue
        fields = [field.strip() for field in line.split(",") if field.strip()]
        for key, cgroup in cgroups.items():
            normalized = cgroup.lstrip("/")
            if not any(field.lstrip("/") == normalized for field in fields):
                continue
            saw_matching_cgroup = True
            value = _parse_perf_count(line.split(",", 1)[0])
            if value is not None:
                counts[key] = counts.get(key, 0.0) + value
                break
    return counts, saw_matching_cgroup


def sample_cgroup_memory_access_once(
    backend: CgroupMemoryAccessBackend,
    *,
    cgroups: Mapping[str, str],
    interval_s: float,
    perf_executable: str = DEFAULT_PERF_EXECUTABLE,
) -> CgroupMemoryAccessReading:
    if not cgroups:
        return CgroupMemoryAccessReading(
            available=False,
            source=backend.source,
            reason="no_cgroups",
        )
    started_epoch = time.time()
    try:
        result = subprocess.run(
            [
                perf_executable,
                "stat",
                "-x,",
                "--no-big-num",
                "-a",
                "-e",
                backend.event_spec,
                "--for-each-cgroup",
                ",".join(dict.fromkeys(cgroups.values())),
                "--",
                "sleep",
                f"{interval_s:.6f}",
            ],
            capture_output=True,
            text=True,
            timeout=max(5.0, interval_s + 5.0),
            check=False,
            env={"LC_ALL": "C"},
        )
    except FileNotFoundError:
        return CgroupMemoryAccessReading(
            available=False,
            source=backend.source,
            reason="perf_missing",
            started_epoch=started_epoch,
            ended_epoch=time.time(),
        )
    except subprocess.TimeoutExpired:
        return CgroupMemoryAccessReading(
            available=False,
            source=backend.source,
            reason="perf_timeout",
            started_epoch=started_epoch,
            ended_epoch=time.time(),
        )
    ended_epoch = time.time()

    perf_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        return CgroupMemoryAccessReading(
            available=False,
            source=backend.source,
            reason=_classify_perf_failure(perf_output),
            started_epoch=started_epoch,
            ended_epoch=ended_epoch,
        )
    counts, saw_matching_cgroup = _parse_perf_cgroup_count_output(
        perf_output,
        event_specs=backend.event_specs,
        cgroups=cgroups,
    )
    if not counts:
        return CgroupMemoryAccessReading(
            available=False,
            source=backend.source,
            reason="not_counted" if saw_matching_cgroup else "parse_error",
            started_epoch=started_epoch,
            ended_epoch=ended_epoch,
        )
    divisor = max(interval_s, 1e-9)
    measurements = {
        key: CgroupMemoryAccessMeasurement(
            cgroup=cgroups[key],
            events=value,
            events_per_s=value / divisor,
        )
        for key, value in counts.items()
    }
    return CgroupMemoryAccessReading(
        available=True,
        source=backend.source,
        measurements=measurements,
        started_epoch=started_epoch,
        ended_epoch=ended_epoch,
    )


def sample_memory_bandwidth_once(
    backend: PerfEventBackend,
    *,
    interval_s: float,
    perf_executable: str = DEFAULT_PERF_EXECUTABLE,
) -> MemoryBandwidthReading:
    event_specs = (*backend.read_specs, *backend.write_specs)
    try:
        result = subprocess.run(
            [
                perf_executable,
                "stat",
                "-x,",
                "--no-big-num",
                "-a",
                "-e",
                ",".join(event_specs),
                "--",
                "sleep",
                f"{interval_s:.6f}",
            ],
            capture_output=True,
            text=True,
            timeout=max(5.0, interval_s + 5.0),
            check=False,
            env={"LC_ALL": "C"},
        )
    except FileNotFoundError:
        return MemoryBandwidthReading(
            available=False,
            source=backend.source,
            reason="perf_missing",
        )
    except subprocess.TimeoutExpired:
        return MemoryBandwidthReading(
            available=False,
            source=backend.source,
            reason="perf_timeout",
        )

    perf_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        return MemoryBandwidthReading(
            available=False,
            source=backend.source,
            reason=_classify_perf_failure(perf_output),
        )

    counts = _parse_perf_stat_output(perf_output, event_specs)
    if counts is None:
        return MemoryBandwidthReading(
            available=False,
            source=backend.source,
            reason="parse_error",
        )

    read_bytes = sum(counts[spec] for spec in backend.read_specs) * backend.bytes_per_count
    write_bytes = sum(counts[spec] for spec in backend.write_specs) * backend.bytes_per_count
    divisor = max(interval_s, 1e-9) * 1024 * 1024
    read_mb_s = read_bytes / divisor
    write_mb_s = write_bytes / divisor
    return MemoryBandwidthReading(
        available=True,
        source=backend.source,
        total_mb_s=read_mb_s + write_mb_s,
        read_mb_s=read_mb_s,
        write_mb_s=write_mb_s,
    )


class HostMemoryBandwidthCollector(threading.Thread):
    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        perf_executable: str = DEFAULT_PERF_EXECUTABLE,
        event_source_root: Path = EVENT_SOURCE_ROOT,
    ) -> None:
        super().__init__(daemon=True, name="host-memory-bandwidth")
        self.interval_s = interval_s
        self.perf_executable = perf_executable
        self.event_source_root = event_source_root
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = MemoryBandwidthReading(
            available=False,
            reason="initializing",
        )

    def latest(self) -> MemoryBandwidthReading:
        with self._lock:
            return self._latest

    def _set_latest(self, reading: MemoryBandwidthReading) -> None:
        with self._lock:
            self._latest = reading

    def run(self) -> None:
        if sys.platform != "linux":
            self._set_latest(
                MemoryBandwidthReading(
                    available=False,
                    reason="unsupported_platform",
                )
            )
            return

        resolved_perf = shutil.which(self.perf_executable)
        if resolved_perf is None:
            self._set_latest(
                MemoryBandwidthReading(
                    available=False,
                    reason="perf_missing",
                )
            )
            return

        backend = detect_perf_backend(self.event_source_root)
        if backend is None:
            self._set_latest(
                MemoryBandwidthReading(
                    available=False,
                    reason="pmu_unsupported",
                )
            )
            return

        while not self._stop_event.is_set():
            reading = sample_memory_bandwidth_once(
                backend,
                interval_s=self.interval_s,
                perf_executable=resolved_perf,
            )
            self._set_latest(reading)
            if not reading.available and reading.reason in {
                "permission_denied",
                "pmu_unsupported",
                "perf_missing",
            }:
                return

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=max(2.0, self.interval_s + 1.0))


_collector_lock = threading.Lock()
_collector: HostMemoryBandwidthCollector | None = None
_TERMINAL_REASONS = {
    "unsupported_platform",
    "perf_missing",
    "pmu_unsupported",
    "permission_denied",
}


def get_host_memory_bandwidth_collector(
    *,
    interval_s: float = 1.0,
) -> HostMemoryBandwidthCollector:
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = HostMemoryBandwidthCollector(interval_s=interval_s)
            _collector.start()
        elif not _collector.is_alive():
            latest = _collector.latest()
            if latest.available or latest.reason in _TERMINAL_REASONS:
                return _collector
            _collector = HostMemoryBandwidthCollector(interval_s=interval_s)
            _collector.start()
        return _collector


def get_host_memory_bandwidth_sample(
    *,
    interval_s: float = 1.0,
) -> MemoryBandwidthReading:
    return get_host_memory_bandwidth_collector(interval_s=interval_s).latest()


def attach_host_memory_bandwidth(
    sample: dict[str, object],
    *,
    interval_s: float,
) -> None:
    reading = get_host_memory_bandwidth_sample(interval_s=interval_s)
    sample["memory_bandwidth_available"] = reading.available
    if reading.source is not None:
        sample["memory_bandwidth_source"] = reading.source
    if reading.reason is not None:
        sample["memory_bandwidth_reason"] = reading.reason
    if reading.total_mb_s is not None:
        sample["memory_total_mb_s"] = reading.total_mb_s
    if reading.read_mb_s is not None:
        sample["memory_read_mb_s"] = reading.read_mb_s
    if reading.write_mb_s is not None:
        sample["memory_write_mb_s"] = reading.write_mb_s


def reset_host_memory_bandwidth_collector_for_tests() -> None:
    global _collector
    with _collector_lock:
        if _collector is not None:
            _collector.stop()
        _collector = None
