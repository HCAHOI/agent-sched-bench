from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.memory_bandwidth import (
    CgroupMemoryAccessBackend,
    INTEL_CAS_BYTES,
    MemoryBandwidthReading,
    attach_host_memory_bandwidth,
    detect_cgroup_memory_access_backend,
    detect_perf_backend,
    reset_host_memory_bandwidth_collector_for_tests,
    sample_cgroup_memory_access_once,
    sample_memory_bandwidth_once,
)


@pytest.fixture(autouse=True)
def _reset_collector() -> None:
    reset_host_memory_bandwidth_collector_for_tests()
    yield
    reset_host_memory_bandwidth_collector_for_tests()


def test_detect_perf_backend_prefers_intel_imc(tmp_path: Path) -> None:
    events = tmp_path / "uncore_imc_0" / "events"
    events.mkdir(parents=True)
    (events / "cas_count_read").write_text("event=0x01\n", encoding="utf-8")
    (events / "cas_count_write").write_text("event=0x02\n", encoding="utf-8")

    backend = detect_perf_backend(tmp_path)

    assert backend is not None
    assert backend.kind == "intel_imc_cas"
    assert backend.read_specs == ("uncore_imc_0/cas_count_read/",)
    assert backend.write_specs == ("uncore_imc_0/cas_count_write/",)


def test_detect_perf_backend_falls_back_to_explicit_byte_events(tmp_path: Path) -> None:
    events = tmp_path / "ddrc0" / "events"
    events.mkdir(parents=True)
    (events / "read_bytes").write_text("event=0x11\n", encoding="utf-8")
    (events / "write_bytes").write_text("event=0x12\n", encoding="utf-8")

    backend = detect_perf_backend(tmp_path)

    assert backend is not None
    assert backend.kind == "explicit_byte_events"
    assert backend.read_specs == ("ddrc0/read_bytes/",)
    assert backend.write_specs == ("ddrc0/write_bytes/",)


def test_detect_cgroup_memory_access_backend(tmp_path: Path) -> None:
    events = tmp_path / "armv8_pmuv3_0" / "events"
    events.mkdir(parents=True)
    (events / "mem_access").write_text("event=0x13\n", encoding="utf-8")

    backend = detect_cgroup_memory_access_backend(tmp_path)

    assert backend is not None
    assert backend.source == "perf:armv8_pmuv3_0:mem_access:cgroup"
    assert backend.event_specs == ("armv8_pmuv3_0/mem_access/",)
    assert backend.event_spec == "armv8_pmuv3_0/mem_access/"


def test_detect_cgroup_memory_access_backend_collects_all_mem_access_pmus(
    tmp_path: Path,
) -> None:
    for name in ("armv8_pmuv3_0", "armv8_pmuv3_1"):
        events = tmp_path / name / "events"
        events.mkdir(parents=True)
        (events / "mem_access").write_text("event=0x13\n", encoding="utf-8")

    backend = detect_cgroup_memory_access_backend(tmp_path)

    assert backend is not None
    assert backend.event_specs == (
        "armv8_pmuv3_0/mem_access/",
        "armv8_pmuv3_1/mem_access/",
    )
    assert backend.event_spec == (
        "armv8_pmuv3_0/mem_access/,armv8_pmuv3_1/mem_access/"
    )


def test_sample_cgroup_memory_access_once_parses_for_each_cgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/",),
    )
    cgroups = {
        "container-a": "/system.slice/docker-a.scope",
        "container-b": "/system.slice/docker-b.scope",
    }

    def fake_run(cmd, **kwargs):
        assert cmd[:5] == ["perf", "stat", "-x,", "--no-big-num", "-a"]
        assert cmd[cmd.index("-e") + 1] == "armv8_pmuv3_0/mem_access/"
        assert cmd[cmd.index("--for-each-cgroup") + 1] == (
            "/system.slice/docker-a.scope,/system.slice/docker-b.scope"
        )
        stderr = "\n".join(
            [
                "1000,,armv8_pmuv3_0/mem_access/,1.00,100.00,,/system.slice/docker-a.scope",
                "2500,,armv8_pmuv3_0/mem_access/,1.00,100.00,,system.slice/docker-b.scope",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)

    reading = sample_cgroup_memory_access_once(
        backend,
        cgroups=cgroups,
        interval_s=2.0,
    )

    assert reading.available is True
    assert reading.measurements["container-a"].events == pytest.approx(1000.0)
    assert reading.measurements["container-a"].events_per_s == pytest.approx(500.0)
    assert reading.measurements["container-b"].events == pytest.approx(2500.0)
    assert reading.measurements["container-b"].events_per_s == pytest.approx(1250.0)
    assert reading.started_epoch is not None
    assert reading.ended_epoch is not None


def test_sample_cgroup_memory_access_once_sums_multiple_pmus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0+armv8_pmuv3_1:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/", "armv8_pmuv3_1/mem_access/"),
    )

    def fake_run(cmd, **kwargs):
        assert cmd[cmd.index("-e") + 1] == (
            "armv8_pmuv3_0/mem_access/,armv8_pmuv3_1/mem_access/"
        )
        stderr = "\n".join(
            [
                "1000,,armv8_pmuv3_0/mem_access/,1.00,100.00,,system.slice/docker-a.scope",
                "2000,,armv8_pmuv3_1/mem_access/,1.00,100.00,,system.slice/docker-a.scope",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)

    reading = sample_cgroup_memory_access_once(
        backend,
        cgroups={"container-a": "/system.slice/docker-a.scope"},
        interval_s=2.0,
    )

    assert reading.available is True
    assert reading.measurements["container-a"].events == pytest.approx(3000.0)
    assert reading.measurements["container-a"].events_per_s == pytest.approx(1500.0)


def test_sample_cgroup_memory_access_once_does_not_prefix_match_cgroups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/",),
    )
    cgroups = {
        "short": "/system.slice/docker-a.scope",
        "long": "/system.slice/docker-a.scope-extra",
    }

    def fake_run(cmd, **kwargs):
        stderr = (
            "2500,,armv8_pmuv3_0/mem_access/,1.00,100.00,,"
            "system.slice/docker-a.scope-extra\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)

    reading = sample_cgroup_memory_access_once(
        backend,
        cgroups=cgroups,
        interval_s=1.0,
    )

    assert reading.available is True
    assert "short" not in reading.measurements
    assert reading.measurements["long"].events == pytest.approx(2500.0)


def test_sample_cgroup_memory_access_once_reports_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/",),
    )

    def fake_run(cmd, **kwargs):
        stderr = (
            "<not counted>,,armv8_pmuv3_0/mem_access/,"
            "system.slice/docker-a.scope,0,100.00,,\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)

    reading = sample_cgroup_memory_access_once(
        backend,
        cgroups={"container-a": "/system.slice/docker-a.scope"},
        interval_s=1.0,
    )

    assert reading.available is False
    assert reading.reason == "not_counted"


def test_sample_memory_bandwidth_once_intel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = tmp_path / "uncore_imc_0" / "events"
    events.mkdir(parents=True)
    (events / "cas_count_read").write_text("", encoding="utf-8")
    (events / "cas_count_write").write_text("", encoding="utf-8")
    backend = detect_perf_backend(tmp_path)
    assert backend is not None

    def fake_run(cmd, **kwargs):
        assert "perf" in cmd[0]
        stderr = "\n".join(
            [
                "1000,,uncore_imc_0/cas_count_read/,1.00,100.00",
                "2000,,uncore_imc_0/cas_count_write/,1.00,100.00",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)
    reading = sample_memory_bandwidth_once(backend, interval_s=2.0)

    assert reading.available is True
    expected_read = 1000 * INTEL_CAS_BYTES / (2.0 * 1024 * 1024)
    expected_write = 2000 * INTEL_CAS_BYTES / (2.0 * 1024 * 1024)
    assert reading.read_mb_s == pytest.approx(expected_read)
    assert reading.write_mb_s == pytest.approx(expected_write)
    assert reading.total_mb_s == pytest.approx(expected_read + expected_write)


def test_sample_memory_bandwidth_once_permission_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    events = tmp_path / "uncore_imc_0" / "events"
    events.mkdir(parents=True)
    (events / "cas_count_read").write_text("", encoding="utf-8")
    (events / "cas_count_write").write_text("", encoding="utf-8")
    backend = detect_perf_backend(tmp_path)
    assert backend is not None

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            255,
            stdout="",
            stderr="Error: No permission to enable uncore_imc_0/cas_count_read/ event.\n",
        )

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)
    reading = sample_memory_bandwidth_once(backend, interval_s=1.0)

    assert reading.available is False
    assert reading.reason == "permission_denied"


def test_attach_host_memory_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harness.memory_bandwidth.get_host_memory_bandwidth_sample",
        lambda **kwargs: MemoryBandwidthReading(
            available=True,
            source="perf:intel-imc-cas",
            total_mb_s=12.5,
            read_mb_s=7.5,
            write_mb_s=5.0,
        ),
    )
    sample: dict[str, object] = {}

    attach_host_memory_bandwidth(sample, interval_s=1.0)

    assert sample["memory_bandwidth_available"] is True
    assert sample["memory_bandwidth_source"] == "perf:intel-imc-cas"
    assert sample["memory_total_mb_s"] == 12.5
    assert sample["memory_read_mb_s"] == 7.5
    assert sample["memory_write_mb_s"] == 5.0
