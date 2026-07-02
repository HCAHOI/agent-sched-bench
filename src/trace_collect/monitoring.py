"""Resolve monitoring policy for trace simulation.

The policy is intentionally separate from replay semantics: resource timelines
remain the replay-facing source of truth, while this module controls expensive
run-level telemetry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

MonitoringMode = Literal["auto", "on", "off"]
MONITORING_CHOICES = ("auto", "on", "off")


@dataclass(frozen=True, slots=True)
class SimulateMonitoringPolicy:
    """Requested and resolved monitoring settings for one simulate run."""

    resource_requested: MonitoringMode
    pmu_requested: MonitoringMode
    memory_bandwidth_requested: MonitoringMode
    resource_enabled: bool
    pmu_enabled: bool
    pmu_reason: str
    memory_bandwidth_enabled: bool
    memory_bandwidth_reason: str
    global_container_resource_enabled: bool
    per_task_resource_enabled: bool
    concurrent: bool
    workers: int
    concurrency: int
    has_container_session: bool
    has_host_session: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_simulate_monitoring(
    *,
    resource: MonitoringMode,
    pmu: MonitoringMode,
    memory_bandwidth: MonitoringMode,
    concurrency: int,
    workers: int,
    has_container_session: bool,
    has_host_session: bool,
) -> SimulateMonitoringPolicy:
    """Resolve simulate monitoring defaults and reject unsafe combinations."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    _validate_mode("resource", resource)
    _validate_mode("pmu", pmu)
    _validate_mode("memory_bandwidth", memory_bandwidth)

    concurrent = concurrency > 1 or workers > 1
    if has_host_session and resource == "on":
        raise ValueError(
            "--resource-monitoring on is unsupported for host simulate sessions "
            "because replay has no isolated host agent PID"
        )

    resource_enabled = resource == "on" or (
        resource == "auto" and has_container_session
    )

    if pmu == "on" and concurrent:
        raise ValueError(
            "--pmu-monitoring on is forbidden for concurrent simulate replay"
        )
    if pmu == "on" and not resource_enabled:
        raise ValueError(
            "--pmu-monitoring on requires --resource-monitoring on or auto with "
            "container sessions"
        )
    if memory_bandwidth == "on" and not resource_enabled:
        raise ValueError(
            "--memory-bandwidth-monitoring on requires --resource-monitoring on "
            "or auto with container sessions"
        )

    pmu_enabled = resource_enabled and not concurrent and pmu != "off"
    memory_bandwidth_enabled = (
        resource_enabled and not concurrent and memory_bandwidth != "off"
    )
    pmu_reason = _resolved_reason(
        requested=pmu,
        enabled=pmu_enabled,
        resource_enabled=resource_enabled,
        concurrent=concurrent,
    )
    memory_bandwidth_reason = _resolved_reason(
        requested=memory_bandwidth,
        enabled=memory_bandwidth_enabled,
        resource_enabled=resource_enabled,
        concurrent=concurrent,
    )
    per_task_resource_enabled = resource_enabled and has_container_session
    global_container_resource_enabled = (
        resource_enabled and has_container_session and workers == 1
    )

    return SimulateMonitoringPolicy(
        resource_requested=resource,
        pmu_requested=pmu,
        memory_bandwidth_requested=memory_bandwidth,
        resource_enabled=resource_enabled,
        pmu_enabled=pmu_enabled,
        pmu_reason=pmu_reason,
        memory_bandwidth_enabled=memory_bandwidth_enabled,
        memory_bandwidth_reason=memory_bandwidth_reason,
        global_container_resource_enabled=global_container_resource_enabled,
        per_task_resource_enabled=per_task_resource_enabled,
        concurrent=concurrent,
        workers=workers,
        concurrency=concurrency,
        has_container_session=has_container_session,
        has_host_session=has_host_session,
    )


def _resolved_reason(
    *,
    requested: MonitoringMode,
    enabled: bool,
    resource_enabled: bool,
    concurrent: bool,
) -> str:
    if enabled:
        return "enabled_serial_container"
    if requested == "off":
        return "disabled_by_request"
    if not resource_enabled:
        return "disabled_resource_monitoring_off"
    if concurrent:
        return "disabled_concurrent_replay"
    return "disabled_unavailable"


def _validate_mode(name: str, value: str) -> None:
    if value not in MONITORING_CHOICES:
        choices = ", ".join(MONITORING_CHOICES)
        raise ValueError(f"{name} monitoring must be one of: {choices}")
