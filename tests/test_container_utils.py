"""Tests for harness.disk_preflight, container_image_prep, container_stats_sampler."""

from __future__ import annotations

from collections.abc import Callable
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_image_prep import (  # noqa: E402
    clear_image_cache,
    drop_cached_fixed_image,
    ensure_fixed_image,
    ensure_source_image,
    normalize_image_reference,
    prune_dangling_images,
    remove_image,
)
from harness.container_stats_sampler import (  # noqa: E402
    ContainerResourceRecorder,
    ContainerStatsSampler,
    _parse_pipe_stats,
    summarize_samples,
)
from harness.disk_preflight import (  # noqa: E402
    DiskSpaceError,
    preflight_disk,
)
from harness.memory_bandwidth import (  # noqa: E402
    CgroupMemoryAccessBackend,
    CgroupMemoryAccessMeasurement,
    CgroupMemoryAccessReading,
)


def _wait_until(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    if predicate():
        return
    raise AssertionError("condition did not become true before timeout")


def _recorder_empty_tick_count(recorder: ContainerResourceRecorder) -> int:
    with recorder._lock:
        return recorder._empty_tick_count


def _recorder_error_count(recorder: ContainerResourceRecorder) -> int:
    with recorder._lock:
        return len(recorder._errors)


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


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_fixed_image_passthrough_pulls_source_no_build(
    container_executable: str,
) -> None:
    # Passthrough contract: ensure_fixed_image no longer builds a chown/commit
    # derivative. It pulls the (missing) source image and returns it unchanged;
    # the container runs as root so /testbed is already writable.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == ["podman", "image", "exists"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        fixed, elapsed = ensure_fixed_image(
            "swerebench/foo:latest",
            container_executable=container_executable,
            host_uid=1000,
            host_gid=1000,
        )

    assert fixed == "docker.io/swerebench/foo:latest"
    assert elapsed == 0.0
    verbs = [" ".join(c[1:3]) for c in calls]
    assert "pull docker.io/swerebench/foo:latest" in [" ".join(c[1:4]) for c in calls]
    assert not any(v == "run -d" for v in verbs)
    assert not any(v.startswith("commit") for v in verbs)
    assert not any(v.startswith("exec") for v in verbs)


def test_ensure_fixed_image_ignores_derivative_args() -> None:
    # fixed_image_name / rebuild are accepted for call-site compatibility but
    # ignored: the source image is always returned as-is (no rebuild, no commit).
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        fixed, elapsed = ensure_fixed_image(
            "swerebench/foo:latest",
            container_executable="docker",
            fixed_image_name="swebench-fixed-custom:attempt",
            rebuild=True,
        )

    assert fixed == "docker.io/swerebench/foo:latest"
    assert elapsed == 0.0
    verbs = [" ".join(c[1:3]) for c in calls]
    assert not any(v == "run -d" for v in verbs)
    assert not any(v.startswith("commit") for v in verbs)
    assert not any(v.startswith("image rm") for v in verbs)


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_source_image_pulls_when_missing(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        ensure_source_image(
            "swerebench/source:latest",
            container_executable=container_executable,
        )

    assert calls[0] == (
        [container_executable, "image", "exists", "docker.io/swerebench/source:latest"]
        if container_executable == "podman"
        else [
            container_executable,
            "image",
            "inspect",
            "docker.io/swerebench/source:latest",
        ]
    )
    assert calls[1] == [
        container_executable,
        "pull",
        "docker.io/swerebench/source:latest",
    ]


def test_immutable_image_id_is_not_rewritten_as_registry_name() -> None:
    image_id = "sha256:" + "a" * 64
    assert normalize_image_reference(image_id) == image_id


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_missing_immutable_image_id_is_never_pulled(
    container_executable: str,
) -> None:
    image_id = "sha256:" + "a" * 64

    with patch(
        "harness.container_image_prep.subprocess.run",
        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="missing"),
    ) as run:
        with pytest.raises(RuntimeError, match="unavailable locally"):
            ensure_source_image(image_id, container_executable=container_executable)

    assert all("pull" not in call.args[0] for call in run.call_args_list)


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_source_image_retries_transient_pull_failure(
    container_executable: str,
) -> None:
    calls = []
    pull_attempts = 0

    def fake_run(cmd, **kwargs):
        nonlocal pull_attempts
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[:2] == [container_executable, "pull"]:
            pull_attempts += 1
            if pull_attempts < 3:
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="failed to do request: EOF",
                )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch(
            "harness.container_image_prep.subprocess.run",
            side_effect=fake_run,
        ),
        patch("harness.container_image_prep.time.sleep", lambda *_: None),
    ):
        ensure_source_image(
            "swerebench/source:latest",
            container_executable=container_executable,
        )

    assert pull_attempts == 3


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_source_image_does_not_retry_nonretryable_pull_failure(
    container_executable: str,
) -> None:
    pull_attempts = 0

    def fake_run(cmd, **kwargs):
        nonlocal pull_attempts
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[:2] == [container_executable, "pull"]:
            pull_attempts += 1
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="manifest unknown",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch(
            "harness.container_image_prep.subprocess.run",
            side_effect=fake_run,
        ),
        patch("harness.container_image_prep.time.sleep", lambda *_: None),
    ):
        with pytest.raises(RuntimeError, match="manifest unknown"):
            ensure_source_image(
                "swerebench/source:latest",
                container_executable=container_executable,
            )

    assert pull_attempts == 1


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_drop_cached_fixed_image_forces_reprobe(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        ensure_fixed_image(
            "swerebench/cached:latest",
            container_executable=container_executable,
        )
        drop_cached_fixed_image("swerebench/cached:latest")
        ensure_fixed_image(
            "swerebench/cached:latest",
            container_executable=container_executable,
        )

    exists_calls = [
        cmd
        for cmd in calls
        if (
            container_executable == "podman"
            and cmd[:3] == ["podman", "image", "exists"]
        )
        or (
            container_executable == "docker"
            and cmd[:3] == ["docker", "image", "inspect"]
        )
    ]
    assert len(exists_calls) == 2


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_remove_image_and_prune_dangling_images(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        assert (
            remove_image(
                "docker.io/swerebench/source:latest",
                container_executable=container_executable,
            )
            is True
        )
        prune_dangling_images(container_executable=container_executable)

    assert [
        container_executable,
        "image",
        "rm",
        "-f",
        "docker.io/swerebench/source:latest",
    ] in calls
    assert [container_executable, "image", "prune", "-f"] in calls


def test_normalize_image_reference_qualifies_short_names() -> None:
    assert normalize_image_reference("hello-world") == "docker.io/library/hello-world"
    assert normalize_image_reference("alpine:3.20") == "docker.io/library/alpine:3.20"
    assert (
        normalize_image_reference("alpine@sha256:deadbeef")
        == "docker.io/library/alpine@sha256:deadbeef"
    )
    assert (
        normalize_image_reference("swerebench/foo:latest")
        == "docker.io/swerebench/foo:latest"
    )
    assert (
        normalize_image_reference("docker.io/swerebench/foo:latest")
        == "docker.io/swerebench/foo:latest"
    )


def test_normalize_image_reference_respects_registry_prefix_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TASK_CONTAINER_IMAGE_REGISTRY_PREFIX", "docker.1ms.run")

    assert (
        normalize_image_reference("hello-world") == "docker.1ms.run/library/hello-world"
    )
    assert (
        normalize_image_reference("swerebench/foo:latest")
        == "docker.1ms.run/swerebench/foo:latest"
    )
    assert (
        normalize_image_reference("docker.io/swerebench/foo:latest")
        == "docker.io/swerebench/foo:latest"
    )


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_remove_image_keeps_local_fixed_tags_unqualified(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        assert (
            remove_image(
                "swebench-fixed-local_tag",
                container_executable=container_executable,
            )
            is True
        )

    assert calls[0] == (
        [container_executable, "image", "exists", "swebench-fixed-local_tag"]
        if container_executable == "podman"
        else [container_executable, "image", "inspect", "swebench-fixed-local_tag"]
    )
    assert calls[1] == [
        container_executable,
        "image",
        "rm",
        "-f",
        "swebench-fixed-local_tag",
    ]


# ---------------------------------------------------------------------------
# container_stats_sampler
# ---------------------------------------------------------------------------


def test_parse_pipe_stats_parses_pipe_format() -> None:
    pipe = _parse_pipe_stats("1MB / 1GB|0.1%|0.5%|0B / 0B")
    assert pipe is not None and pipe["mem_usage"] == "1MB / 1GB"


def test_stats_sampler_collects_samples_and_stops_cleanly() -> None:
    # Emit the new pipe format.
    pipe_output = "1MB / 1GB|0.1%|0.5%|0B / 0B"

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


@pytest.mark.parametrize(
    ("container_executable", "expected_stats_id_field"),
    [
        ("docker", "{{.Container}}"),
        ("podman", "{{.ID}}"),
    ],
)
def test_container_resource_recorder_appends_global_container_samples(
    tmp_path: Path,
    container_executable: str,
    expected_stats_id_field: str,
) -> None:
    container_id = "abcdef1234567890"
    other_container_id = "1111111111112222"

    def fake_run(cmd, **kwargs):
        if cmd[:2] == [container_executable, "ps"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    f"{container_id}|swerebench/task-a|task-a\n"
                    f"{other_container_id}|swerebench/task-b|task-b\n"
                ),
                stderr="",
            )
        if cmd[:2] == [container_executable, "stats"]:
            stats_format = cmd[cmd.index("--format") + 1]
            assert expected_stats_id_field in stats_format
            assert container_id in cmd
            assert other_container_id not in cmd
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "abcdef123456|task-a|10MiB / 1GiB|1.0%|12.5%|"
                    "1kB / 2kB\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id=f"simulate-test-{container_executable}",
        interval_s=0.01,
        executable=container_executable,
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
        collect_cgroup_memory_access=False,
        monitoring_policy={
            "pmu_enabled": False,
            "pmu_reason": "disabled_by_policy",
        },
    )
    recorder.register_container(container_id)
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        recorder.start()
        _wait_until(
            lambda: recorder.jsonl_path.exists()
            and bool(recorder.jsonl_path.read_text(encoding="utf-8").strip())
        )
        summary = recorder.stop()

    records = [
        json.loads(line)
        for line in recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    persisted_summary = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
    assert records
    assert records[0]["resource_scope"] == "global_container"
    assert records[0]["container_id"] == container_id
    assert records[0]["container_short_id"] == container_id[:12]
    assert records[0]["container_image"] == "swerebench/task-a"
    assert records[0]["container_name"] == "task-a"
    assert records[0]["net_rx_bytes"] == 1000
    assert records[0]["net_tx_bytes"] == 2000
    assert summary["sample_count"] == len(records)
    assert summary["monitoring"] == {
        "pmu_enabled": False,
        "pmu_reason": "disabled_by_policy",
    }
    assert persisted_summary["monitoring"] == summary["monitoring"]
    assert persisted_summary["sampling"]["stop_complete"] is True
    assert persisted_summary["sampling"]["sample_all_containers"] is False
    assert len(persisted_summary["containers"]) == 1
    assert persisted_summary["containers"][0]["summary"]["sample_count"] == len(records)


def test_container_resource_recorder_attaches_cgroup_memory_access(
    tmp_path: Path,
) -> None:
    container_id = "abcdef1234567890"
    cgroup_path = Path("/sys/fs/cgroup/system.slice/docker-test.scope")
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/",),
    )

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"{container_id}|swerebench/task-a|task-a\n",
                stderr="",
            )
        if cmd[:2] == ["docker", "stats"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    f"{container_id}|task-a|10MiB / 1GiB|1.0%|12.5%|"
                    "1kB / 2kB\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    def fake_sample_memory_access(
        sample_backend,
        *,
        cgroups,
        interval_s,
    ):
        assert sample_backend is backend
        assert cgroups == {container_id: "/system.slice/docker-test.scope"}
        assert interval_s == pytest.approx(0.01)
        return CgroupMemoryAccessReading(
            available=True,
            source=backend.source,
            measurements={
                container_id: CgroupMemoryAccessMeasurement(
                    cgroup="/system.slice/docker-test.scope",
                    events=1200.0,
                    events_per_s=120000.0,
                )
            },
            started_epoch=1000.0,
            ended_epoch=1000.01,
        )

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-memory-access",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
    )
    recorder.register_container(container_id)
    with (
        patch(
            "harness.container_stats_sampler.subprocess.run",
            side_effect=fake_run,
        ),
        patch(
            "harness.container_stats_sampler.detect_cgroup_memory_access_backend",
            return_value=backend,
        ),
        patch("harness.container_stats_sampler.sys.platform", "linux"),
        patch(
            "harness.container_stats_sampler._resolve_container_pid",
            return_value=42,
        ),
        patch(
            "harness.container_stats_sampler._resolve_cgroup_path",
            return_value=cgroup_path,
        ),
        patch(
            "harness.container_stats_sampler.sample_cgroup_memory_access_once",
            side_effect=fake_sample_memory_access,
        ),
    ):
        recorder.start()
        _wait_until(
            lambda: recorder.jsonl_path.exists()
            and bool(recorder.jsonl_path.read_text(encoding="utf-8").strip())
        )
        summary = recorder.stop()

    records = [
        json.loads(line)
        for line in recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert records[0]["memory_access_available"] is True
    assert records[0]["memory_access_source"] == backend.source
    assert records[0]["memory_access_cgroup"] == "/system.slice/docker-test.scope"
    assert records[0]["memory_access_events"] == pytest.approx(1200.0)
    assert records[0]["memory_access_events_per_s"] == pytest.approx(120000.0)
    assert records[0]["memory_access_window_start_epoch"] == pytest.approx(1000.0)
    assert records[0]["memory_access_window_end_epoch"] == pytest.approx(1000.01)
    container_summary = summary["containers"][0]["summary"]
    assert container_summary["memory_access_available"] is True
    assert container_summary["memory_access_source"] == backend.source
    assert container_summary["memory_access_events_per_s"]["avg"] == pytest.approx(
        120000.0
    )
    assert summary["sampling"]["memory_access"]["source"] == backend.source


def test_container_resource_recorder_disables_terminal_memory_access_failures(
    tmp_path: Path,
) -> None:
    container_id = "abcdef1234567890"
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/",),
    )
    calls = 0

    def fake_sample_memory_access(
        sample_backend,
        *,
        cgroups,
        interval_s,
    ):
        nonlocal calls
        calls += 1
        return CgroupMemoryAccessReading(
            available=False,
            source=sample_backend.source,
            reason="permission_denied",
            started_epoch=1000.0,
            ended_epoch=1001.0,
        )

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-memory-access-terminal",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
    )
    recorder.register_container(container_id)
    containers = {container_id: {"container_id": container_id}}
    with (
        patch(
            "harness.container_stats_sampler.detect_cgroup_memory_access_backend",
            return_value=backend,
        ),
        patch(
            "harness.container_stats_sampler._resolve_container_pid",
            return_value=42,
        ),
        patch(
            "harness.container_stats_sampler._resolve_cgroup_path",
            return_value=Path("/sys/fs/cgroup/system.slice/docker-test.scope"),
        ),
        patch(
            "harness.container_stats_sampler.sample_cgroup_memory_access_once",
            side_effect=fake_sample_memory_access,
        ),
        patch("harness.container_stats_sampler.sys.platform", "linux"),
    ):
        first = recorder._sample_memory_access(containers)
        second = recorder._sample_memory_access(containers)

    assert calls == 1
    assert first.reason == "permission_denied"
    assert second.reason == "permission_denied"
    assert second.source == backend.source


def test_container_resource_recorder_keeps_retrying_not_counted(
    tmp_path: Path,
) -> None:
    container_id = "abcdef1234567890"
    backend = CgroupMemoryAccessBackend(
        source="perf:armv8_pmuv3_0:mem_access:cgroup",
        event_specs=("armv8_pmuv3_0/mem_access/",),
    )
    calls = 0

    def fake_sample_memory_access(
        sample_backend,
        *,
        cgroups,
        interval_s,
    ):
        nonlocal calls
        calls += 1
        return CgroupMemoryAccessReading(
            available=False,
            source=sample_backend.source,
            reason="not_counted",
            started_epoch=1000.0 + calls,
            ended_epoch=1001.0 + calls,
        )

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-memory-access-not-counted",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
    )
    recorder.register_container(container_id)
    containers = {container_id: {"container_id": container_id}}
    with (
        patch(
            "harness.container_stats_sampler.detect_cgroup_memory_access_backend",
            return_value=backend,
        ),
        patch(
            "harness.container_stats_sampler._resolve_container_pid",
            return_value=42,
        ),
        patch(
            "harness.container_stats_sampler._resolve_cgroup_path",
            return_value=Path("/sys/fs/cgroup/system.slice/docker-test.scope"),
        ),
        patch(
            "harness.container_stats_sampler.sample_cgroup_memory_access_once",
            side_effect=fake_sample_memory_access,
        ),
        patch("harness.container_stats_sampler.sys.platform", "linux"),
    ):
        first = recorder._sample_memory_access(containers)
        second = recorder._sample_memory_access(containers)
        third = recorder._sample_memory_access(containers)

    assert calls == 3
    assert first.reason == "not_counted"
    assert second.reason == "not_counted"
    assert third.reason == "not_counted"
    assert third.source == backend.source
    assert recorder._memory_access_backend_reason is None
    assert recorder._memory_access_consecutive_not_counted == 3


def test_container_resource_recorder_writes_empty_summary_without_containers(
    tmp_path: Path,
) -> None:
    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-empty",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
    )
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        recorder.start()
        _wait_until(lambda: _recorder_empty_tick_count(recorder) >= 1)
        summary = recorder.stop()

    assert recorder.jsonl_path.read_text(encoding="utf-8") == ""
    assert summary["sample_count"] == 0
    assert summary["containers"] == []
    assert summary["sampling"]["empty_tick_count"] >= 1
    assert summary["sampling"]["stop_complete"] is True


def test_container_resource_recorder_treats_stats_eof_as_transient(
    tmp_path: Path,
) -> None:
    container_id = "abcdef1234567890"
    stats_calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal stats_calls
        if cmd[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"{container_id}|swerebench/task-a|task-a\n",
                stderr="",
            )
        if cmd[:2] == ["docker", "stats"]:
            stats_calls += 1
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="EOF\nEOF\n")
        raise AssertionError(f"unexpected command: {cmd}")

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-eof",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
    )
    recorder.register_container(container_id)
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        recorder.start()
        _wait_until(lambda: stats_calls >= 1)
        summary = recorder.stop()

    assert stats_calls >= 1
    assert summary["sample_count"] == 0
    assert summary["errors"] == []
    assert summary["sampling"]["empty_tick_count"] >= 1


def test_container_resource_recorder_treats_missing_container_stats_as_transient(
    tmp_path: Path,
) -> None:
    container_id = "abcdef1234567890"
    stats_calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal stats_calls
        if cmd[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"{container_id}|swerebench/task-a|task-a\n",
                stderr="",
            )
        if cmd[:2] == ["docker", "stats"]:
            stats_calls += 1
            message = (
                "Error response from daemon: No such container: "
                f"{container_id}\nEOF\n"
            )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=message)
        raise AssertionError(f"unexpected command: {cmd}")

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-missing-container",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
    )
    recorder.register_container(container_id)
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        recorder.start()
        _wait_until(lambda: stats_calls >= 1)
        summary = recorder.stop()

    assert summary["sample_count"] == 0
    assert summary["errors"] == []
    assert summary["sampling"]["empty_tick_count"] >= 1


def test_container_resource_recorder_records_non_transient_stats_errors(
    tmp_path: Path,
) -> None:
    container_id = "abcdef1234567890"

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"{container_id}|swerebench/task-a|task-a\n",
                stderr="",
            )
        if cmd[:2] == ["docker", "stats"]:
            message = (
                "Error response from daemon: No such container: "
                f"{container_id}\npermission denied\n"
            )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=message)
        raise AssertionError(f"unexpected command: {cmd}")

    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-stats-error",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
        sample_all_containers=False,
    )
    recorder.register_container(container_id)
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        recorder.start()
        _wait_until(lambda: _recorder_error_count(recorder) >= 1)
        summary = recorder.stop()

    assert summary["sample_count"] == 0
    assert summary["errors"]
    assert summary["errors"][0]["type"] == "RuntimeError"
    assert "No such container" in summary["errors"][0]["message"]
    assert "permission denied" in summary["errors"][0]["message"]


def test_container_resource_recorder_marks_stop_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = ContainerResourceRecorder(
        output_dir=tmp_path,
        run_id="simulate-timeout",
        interval_s=0.01,
        executable="docker",
        subprocess_timeout_s=0.1,
    )
    joins: list[float | None] = []
    monkeypatch.setattr(recorder, "is_alive", lambda: True)
    monkeypatch.setattr(recorder, "join", lambda timeout=None: joins.append(timeout))

    summary = recorder.stop()

    assert joins
    assert summary["sampling"]["stop_complete"] is False
    assert summary["errors"][-1]["type"] == "RuntimeError"
    assert "did not stop before summary" in summary["errors"][-1]["message"]


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
