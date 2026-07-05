from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest

from agents.openclaw import _checkpoint_container, _session_runner
from agents.sandbox_runtime import (
    AgentTransportRequest,
    FakeBackend,
    OverlayBackend,
    SandboxBackend,
    WalkBackend,
    get_sandbox_backend_class,
)


@pytest.fixture
def checkpoint_cas_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(_session_runner, "_CHECKPOINT_CAS_ROOT", cas_root)
    monkeypatch.setattr(_checkpoint_container, "_CHECKPOINT_CAS_ROOT", cas_root)
    return cas_root


def test_sandbox_runtime_exports_checkpoint_backends() -> None:
    assert issubclass(WalkBackend, SandboxBackend)
    assert issubclass(OverlayBackend, SandboxBackend)


def test_sandbox_runtime_registry_includes_fake_and_docker() -> None:
    assert get_sandbox_backend_class("fake") is FakeBackend
    assert issubclass(get_sandbox_backend_class("docker"), SandboxBackend)


def test_fake_runtime_exec_and_filesystem_tools() -> None:
    async def run() -> None:
        backend = FakeBackend()
        await backend.start()
        write = await backend.execute(
            AgentTransportRequest(
                tool="write_file",
                args={"path": "/testbed/pkg/file.txt", "content": "hello\nworld\n"},
            )
        )
        assert write.ok is True

        read = await backend.execute(
            AgentTransportRequest(
                tool="read_file",
                args={"path": "/testbed/pkg/file.txt"},
            )
        )
        assert read.ok is True
        assert read.result == "1| hello\n2| world"

        listing = await backend.execute(
            AgentTransportRequest(tool="list_dir", args={"path": "/testbed"})
        )
        assert listing.result == "pkg"

        exec_result = await backend.execute(
            AgentTransportRequest(tool="exec", args={"command": "echo hi"})
        )
        assert exec_result.ok is True
        assert exec_result.returncode == 0
        assert exec_result.metadata["fake_exec_counter"] == 0
        assert exec_result.result.endswith("Exit code: 0")

        await backend.stop()

    asyncio.run(run())


def test_fake_runtime_snapshot_restore_and_change_probe() -> None:
    async def run() -> None:
        backend = FakeBackend()
        await backend.execute(
            AgentTransportRequest(
                tool="write_file",
                args={"path": "a.txt", "content": "alpha"},
            )
        )
        snapshot = await backend.capture_snapshot()
        assert await backend.probe_changes_since(snapshot.timestamp_ns) is False

        await backend.execute(
            AgentTransportRequest(
                tool="edit_file",
                args={"path": "a.txt", "old_text": "alpha", "new_text": "beta"},
            )
        )
        assert await backend.probe_changes_since(snapshot.timestamp_ns) is True

        await backend.restore_snapshot(snapshot)
        restored = await backend.execute(
            AgentTransportRequest(tool="read_file", args={"path": "a.txt"})
        )
        assert restored.ok is True
        assert restored.result == "1| alpha"

    asyncio.run(run())


def test_walk_backend_host_snapshot_and_probe(
    tmp_path: Path,
    checkpoint_cas_root: Path,
) -> None:
    async def run() -> None:
        root = tmp_path / "testbed"
        checkpoint_dir = tmp_path / "checkpoints"
        root.mkdir()
        (root / "a.txt").write_text("alpha", encoding="utf-8")

        backend = WalkBackend(root=str(root), checkpoint_dir=checkpoint_dir)
        snapshot = await backend.capture_snapshot()

        expected_hash = hashlib.sha256(b"alpha").hexdigest()
        assert snapshot.disk_state["entries"]["a.txt"]["hash"] == expected_hash
        assert snapshot.disk_state["deleted_paths"] == []
        assert snapshot.disk_state["kind"] == "cas_manifest_full"
        assert await backend.probe_changes_since(snapshot.timestamp_ns) is False

        (root / "a.txt").write_text("beta", encoding="utf-8")
        os.utime(root / "a.txt", ns=(snapshot.timestamp_ns + 1, snapshot.timestamp_ns + 1))
        assert await backend.probe_changes_since(snapshot.timestamp_ns) is True

    asyncio.run(run())


def test_overlay_backend_upperdir_snapshot_skips_git(
    tmp_path: Path,
    checkpoint_cas_root: Path,
) -> None:
    async def run() -> None:
        upperdir = tmp_path / "upper"
        testbed = upperdir / "testbed"
        checkpoint_dir = tmp_path / "checkpoints"
        testbed.mkdir(parents=True)
        (testbed / "a.txt").write_text("alpha", encoding="utf-8")
        git_dir = testbed / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("ignored", encoding="utf-8")

        backend = OverlayBackend(
            upperdir=upperdir,
            checkpoint_dir=checkpoint_dir,
        )
        snapshot = await backend.capture_snapshot()

        expected_hash = hashlib.sha256(b"alpha").hexdigest()
        assert snapshot.disk_state["entries"]["a.txt"]["hash"] == expected_hash
        assert ".git/config" not in snapshot.disk_state["entries"]
        assert snapshot.disk_state["deleted_paths"] == []

    asyncio.run(run())


def test_overlay_backend_empty_upperdir_snapshot(
    tmp_path: Path,
    checkpoint_cas_root: Path,
) -> None:
    async def run() -> None:
        upper = tmp_path / "upper"
        (upper / "testbed").mkdir(parents=True, exist_ok=True)
        backend = OverlayBackend(
            upperdir=upper,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        snapshot = await backend.capture_snapshot()

        assert snapshot.disk_state["entries"] == {}
        assert snapshot.disk_state["deleted_paths"] == []

    asyncio.run(run())
