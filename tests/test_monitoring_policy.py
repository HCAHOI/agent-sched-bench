from __future__ import annotations

import pytest

from trace_collect.monitoring import resolve_simulate_monitoring


def test_simulate_monitoring_serial_auto_enables_expensive_telemetry() -> None:
    policy = resolve_simulate_monitoring(
        resource="auto",
        pmu="auto",
        memory_bandwidth="auto",
        concurrency=1,
        workers=1,
        has_container_session=True,
        has_host_session=False,
    )

    assert policy.resource_enabled is True
    assert policy.pmu_enabled is True
    assert policy.pmu_reason == "enabled_serial_container"
    assert policy.memory_bandwidth_enabled is True
    assert policy.memory_bandwidth_reason == "enabled_serial_container"
    assert policy.global_container_resource_enabled is True
    assert policy.per_task_resource_enabled is True


def test_simulate_monitoring_concurrent_auto_disables_pmu_and_memory_bandwidth() -> None:
    policy = resolve_simulate_monitoring(
        resource="auto",
        pmu="auto",
        memory_bandwidth="auto",
        concurrency=8,
        workers=4,
        has_container_session=True,
        has_host_session=False,
    )

    assert policy.resource_enabled is True
    assert policy.pmu_enabled is False
    assert policy.pmu_reason == "disabled_concurrent_replay"
    assert policy.memory_bandwidth_enabled is False
    assert policy.memory_bandwidth_reason == "disabled_concurrent_replay"
    assert policy.global_container_resource_enabled is False
    assert policy.per_task_resource_enabled is True


def test_simulate_monitoring_rejects_pmu_when_concurrent() -> None:
    with pytest.raises(ValueError, match="forbidden for concurrent simulate replay"):
        resolve_simulate_monitoring(
            resource="auto",
            pmu="on",
            memory_bandwidth="auto",
            concurrency=2,
            workers=1,
            has_container_session=True,
            has_host_session=False,
        )


@pytest.mark.parametrize(
    ("concurrency", "workers"),
    [(2, 1), (1, 2)],
)
def test_simulate_monitoring_disables_explicit_memory_bandwidth_when_concurrent(
    concurrency: int,
    workers: int,
) -> None:
    policy = resolve_simulate_monitoring(
        resource="auto",
        pmu="auto",
        memory_bandwidth="on",
        concurrency=concurrency,
        workers=workers,
        has_container_session=True,
        has_host_session=False,
    )

    assert policy.memory_bandwidth_requested == "on"
    assert policy.memory_bandwidth_enabled is False
    assert policy.memory_bandwidth_reason == "disabled_concurrent_replay"


def test_simulate_monitoring_resource_off_disables_all_resource_paths() -> None:
    policy = resolve_simulate_monitoring(
        resource="off",
        pmu="auto",
        memory_bandwidth="auto",
        concurrency=1,
        workers=1,
        has_container_session=True,
        has_host_session=False,
    )

    assert policy.resource_enabled is False
    assert policy.pmu_enabled is False
    assert policy.memory_bandwidth_enabled is False
    assert policy.global_container_resource_enabled is False
    assert policy.per_task_resource_enabled is False


def test_simulate_monitoring_rejects_explicit_host_resource_monitoring() -> None:
    with pytest.raises(ValueError, match="unsupported for host simulate sessions"):
        resolve_simulate_monitoring(
            resource="on",
            pmu="off",
            memory_bandwidth="off",
            concurrency=1,
            workers=1,
            has_container_session=False,
            has_host_session=True,
        )
