"""Tests for harness.disk_preflight, container_image_prep, container_stats_sampler."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_image_prep import (  # noqa: E402
    clear_image_cache,
    ensure_fixed_image,
)
from harness.container_stats_sampler import (  # noqa: E402
    ContainerStatsSampler,
    _parse_pipe_stats,
    summarize_samples,
)
from harness.disk_preflight import (  # noqa: E402
    DiskSpaceError,
    preflight_disk,
)


# ---------------------------------------------------------------------------
# disk_preflight
# ---------------------------------------------------------------------------


def test_preflight_disk_raises_on_shortfall(tmp_path: Path) -> None:
    with pytest.raises(DiskSpaceError) as exc:
        preflight_disk(tmp_path, min_free_gb=10**12)
    assert "GB required" in str(exc.value)


# ---------------------------------------------------------------------------
# container_image_prep
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_image_cache() -> None:
    clear_image_cache()
    yield
    clear_image_cache()


def test_ensure_fixed_image_builds_when_derivative_missing() -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # existence probe → missing (returncode=1)
        if cmd[:3] == ["podman", "image", "exists"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        # run -d → container id
        if cmd[1:3] == ["run", "-d"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid_xyz\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        fixed, elapsed = ensure_fixed_image(
            "swerebench/foo:latest", host_uid=1000, host_gid=1000
        )
    assert fixed == "swebench-fixed-swerebench_foo_latest"
    assert elapsed >= 0.0
    # Expect: exists (miss), run -d, exec chown, commit, stop, rm
    verbs = [" ".join(c[1:3]) for c in calls]
    assert "image exists" in verbs
    assert any("run -d" in v for v in verbs)
    assert any("exec cid_xyz" == " ".join(c[1:3]) for c in calls)
    assert any("commit cid_xyz" == " ".join(c[1:3]) for c in calls)


def test_ensure_fixed_image_raises_on_build_failure() -> None:
    def boom(cmd, **kwargs):
        if cmd[:3] == ["podman", "image", "exists"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[1:3] == ["run", "-d"]:
            raise subprocess.CalledProcessError(1, cmd, stderr="kaboom")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=boom,
    ):
        with pytest.raises(RuntimeError, match="Failed to build"):
            ensure_fixed_image("swerebench/img")


# ---------------------------------------------------------------------------
# container_stats_sampler
# ---------------------------------------------------------------------------


def test_parse_pipe_stats_parses_pipe_format() -> None:
    pipe = _parse_pipe_stats("1MB / 1GB|0.1%|0.5%")
    assert pipe is not None and pipe["mem_usage"] == "1MB / 1GB"


def test_stats_sampler_collects_samples_and_stops_cleanly() -> None:
    # Emit the new pipe format.
    pipe_output = "1MB / 1GB|0.1%|0.5%"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=pipe_output, stderr="")

    sampler = ContainerStatsSampler(
        container_id="abc",
        interval_s=0.05,
    )
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        sampler.start()
        time.sleep(0.2)
        samples = sampler.stop()
    assert len(samples) >= 2
    assert samples[0]["mem_usage"] == "1MB / 1GB"


def test_summarize_samples_computes_min_max_avg() -> None:
    samples = [
        {
            "epoch": 1000.0,
            "mem_usage": "100MB / 1GB",
            "mem_percent": "10%",
            "cpu_percent": "5%",
        },
        {
            "epoch": 1002.0,
            "mem_usage": "200MB / 1GB",
            "mem_percent": "20%",
            "cpu_percent": "15%",
        },
        {
            "epoch": 1004.0,
            "mem_usage": "300MB / 1GB",
            "mem_percent": "30%",
            "cpu_percent": "25%",
        },
    ]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == 3
    assert summary["duration_seconds"] == 4.0
    assert summary["memory_mb"]["min"] == 100.0
    assert summary["memory_mb"]["max"] == 300.0
    assert summary["memory_mb"]["avg"] == 200.0
    assert summary["cpu_percent"]["min"] == 5.0
    assert summary["cpu_percent"]["max"] == 25.0
    assert summary["cpu_percent"]["avg"] == 15.0
