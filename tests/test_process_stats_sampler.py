from __future__ import annotations

import os
import time

from harness.container_stats_sampler import summarize_samples
from harness.process_stats_sampler import ProcessStatsSampler


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
