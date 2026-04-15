from __future__ import annotations

import os
import sys
import time
import types
from types import SimpleNamespace

from harness.container_stats_sampler import summarize_samples
from harness.process_stats_sampler import ProcessStatsSampler, _sample_with_psutil


def test_process_stats_sampler_collects_current_process_sample() -> None:
    sampler = ProcessStatsSampler(pid=os.getpid(), interval_s=0.01)
    sampler.start()
    time.sleep(0.03)
    samples = sampler.stop()

    assert samples
    assert "epoch" in samples[0]
    assert "cpu_percent" in samples[0]
    assert "mem_usage" in samples[0]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == len(samples)


def test_process_stats_sampler_stop_samples_fast_process_attempt() -> None:
    sampler = ProcessStatsSampler(pid=os.getpid(), interval_s=60.0)
    samples = sampler.stop()

    assert len(samples) == 1
    assert "mem_usage" in samples[0]


def test_psutil_sampler_aggregates_recursive_children(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int, rss: int, cpu: float) -> None:
            self.pid = pid
            self._rss = rss
            self._cpu = cpu

        def children(self, *, recursive: bool):
            assert recursive is True
            return [
                FakeProcess(2, rss=20 * 1024 * 1024, cpu=2.5),
                FakeProcess(3, rss=30 * 1024 * 1024, cpu=3.5),
            ]

        def memory_info(self):
            return SimpleNamespace(rss=self._rss)

        def cpu_percent(self, *, interval=None):
            assert interval is None
            return self._cpu

        def io_counters(self):
            return SimpleNamespace(read_bytes=self.pid * 100, write_bytes=self.pid * 10)

        def num_ctx_switches(self):
            return SimpleNamespace(voluntary=self.pid, involuntary=self.pid + 1)

    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Process = lambda pid: FakeProcess(pid, rss=10 * 1024 * 1024, cpu=1.5)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    sample = _sample_with_psutil(1)

    assert sample is not None
    assert sample["mem_usage"] == "60.000MiB"
    assert sample["cpu_percent"] == "7.500%"
    assert sample["disk_read_bytes"] == 600
    assert sample["disk_write_bytes"] == 60
    assert sample["context_switches"] == 15
    assert sample["process_count"] == 3
