from __future__ import annotations

from pathlib import Path

from trace_collect.resource_timeline import (
    ResourceReading,
    read_cgroup_cpu_quota_cores,
    read_cgroup_cpu_usage_s,
    read_proc_net_bytes,
    resource_delta,
)


def test_read_cgroup_cpu_usage_s(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat").write_text("usage_usec 1234567\n", encoding="utf-8")

    assert read_cgroup_cpu_usage_s(tmp_path) == 1.234567


def test_read_cgroup_cpu_quota_cores_from_cpu_max(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("200000 100000\n", encoding="utf-8")

    assert read_cgroup_cpu_quota_cores(tmp_path) == 2.0


def test_read_cgroup_cpu_quota_cores_from_cpuset(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (tmp_path / "cpuset.cpus.effective").write_text("0-1,4\n", encoding="utf-8")

    assert read_cgroup_cpu_quota_cores(tmp_path) == 3.0


def test_read_proc_net_bytes_excludes_loopback(tmp_path: Path) -> None:
    proc_net = tmp_path / "dev"
    proc_net.write_text(
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes\n"
        "    lo: 100 0 0 0 0 0 0 0 200 0 0 0 0 0 0 0\n"
        "  eth0: 300 0 0 0 0 0 0 0 400 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )

    assert read_proc_net_bytes(proc_net) == (300, 400)


def test_resource_delta_serializes_cpu_and_network() -> None:
    previous = ResourceReading(
        monotonic_s=10.0,
        cpu_usage_s=100.0,
        net_rx_bytes=1000,
        net_tx_bytes=2000,
        cpu_quota_cores=4.0,
    )
    current = ResourceReading(
        monotonic_s=10.5,
        cpu_usage_s=101.5,
        net_rx_bytes=1500,
        net_tx_bytes=2200,
        cpu_quota_cores=4.0,
    )

    delta = resource_delta(previous, current)

    assert delta is not None
    assert delta.to_dict() == {
        "offset_s": 10.5,
        "dt_s": 0.5,
        "cpu_core_s": 1.5,
        "net_rx_bytes": 500,
        "net_tx_bytes": 200,
        "cpu_quota_cores": 4.0,
        "cpu_opportunity_core_s": 2.0,
    }
