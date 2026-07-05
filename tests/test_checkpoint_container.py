"""Tests for container-side CAS checkpoint (_checkpoint_container.py)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

import agents.openclaw._checkpoint_container as ccp
from agents.openclaw._checkpoint_container import (
    _CONTAINER_PYTHON_CANDIDATES,
    _compute_deleted_paths,
    _probe_container_python,
    _run_snapshot,
    _fetch_blobs,
    run_container_checkpoint,
)

_FAKE_CONTAINER = {"id": "test-cid", "executable": "docker"}


def _build_snapshot_json(
    *,
    entries: dict[str, Any],
    changed: dict[str, dict[str, Any]] | None = None,
    symlinks: list[str] | None = None,
    changed_during_walk: bool = False,
    now_ns: int | None = None,
) -> str:
    if now_ns is None:
        now_ns = int(time.time() * 1e9)
    payload = {
        "now_ns": now_ns,
        "entries": entries,
        "changed": changed or {},
        "symlinks": symlinks or [],
        "changed_during_walk": changed_during_walk,
    }
    return json.dumps(payload) + "\n"


def _make_tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.type = tarfile.REGTYPE
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_run_sequence(responses: list) -> Any:
    """Build a fake_run that returns each response in sequence."""
    call_idx = [0]

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        idx = call_idx[0]
        call_idx[0] += 1
        if idx >= len(responses):
            raise RuntimeError(f"unexpected call #{idx}: {cmd[:5]}")
        resp = responses[idx]
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, subprocess.CompletedProcess):
            return resp
        if isinstance(resp, dict):
            # {"stdout": str, "stderr": str, "rc": int}
            return subprocess.CompletedProcess(
                cmd, resp.get("rc", 0),
                stdout=resp.get("stdout", ""),
                stderr=resp.get("stderr", ""),
            )
        if isinstance(resp, bytes):
            return subprocess.CompletedProcess(cmd, 0, stdout=resp, stderr=b"")
        if isinstance(resp, str):
            return subprocess.CompletedProcess(cmd, 0, stdout=resp, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


class _FakePopen:
    def __init__(
        self,
        *,
        stdout: bytes,
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode

    def wait(self) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _fake_popen_sequence(responses: list) -> Any:
    """Build a fake Popen constructor that returns each response in sequence."""
    call_idx = [0]

    def fake_popen(cmd, **kwargs):  # noqa: ANN001
        idx = call_idx[0]
        call_idx[0] += 1
        if idx >= len(responses):
            raise RuntimeError(f"unexpected Popen call #{idx}: {cmd[:5]}")
        resp = responses[idx]
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, dict):
            return _FakePopen(
                stdout=resp.get("stdout", b""),
                stderr=resp.get("stderr", b""),
                returncode=resp.get("rc", 0),
            )
        if isinstance(resp, bytes):
            return _FakePopen(stdout=resp)
        return _FakePopen(stdout=b"")

    return fake_popen


def _probe_response(python_path: str = "/usr/bin/python3\n") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=python_path, stderr="")

def test_probe_container_python_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="/usr/bin/python3\n", stderr="")

    monkeypatch.setattr(ccp.subprocess, "run", fake_run)
    ccp._probe_container_python._probe_cache_test_cid = None  # type: ignore[attr-defined]

    r1 = _probe_container_python({"id": "test-cid", "executable": "docker"})
    r2 = _probe_container_python({"id": "test-cid", "executable": "docker"})
    assert r1 == "/usr/bin/python3"
    assert r2 == "/usr/bin/python3"
    assert len(calls) == 1


def test_probe_container_python_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no python found")

    monkeypatch.setattr(ccp.subprocess, "run", fake_run)
    ccp._probe_container_python._probe_cache_test_cid2 = None  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="no usable python in container"):
        _probe_container_python({"id": "test-cid2", "executable": "docker"})


# ---------------------------------------------------------------------------
# _run_snapshot
# ---------------------------------------------------------------------------

def test_run_snapshot_full(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = {"src": "dir", "src/main.py": "file"}
    changed = {
        "src/main.py": {"hash": _sha256(b"hello"), "mode": 0o644, "size": 5, "mtime_ns": 1000}
    }
    snapshot_json = _build_snapshot_json(entries=entries, changed=changed)
    monkeypatch.setattr(ccp.subprocess, "run", _fake_run_sequence([snapshot_json]))
    result = _run_snapshot(_FAKE_CONTAINER, "/usr/bin/python3", None)
    assert result["entries"] == entries
    assert result["changed"] == changed
    assert result["changed_during_walk"] is False


def test_run_snapshot_with_since_ns_passes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                kv = cmd[i + 1]
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    captured_env[k] = v
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_build_snapshot_json(entries={}, changed={}), stderr=""
        )

    monkeypatch.setattr(ccp.subprocess, "run", fake_run)
    _run_snapshot(_FAKE_CONTAINER, "/usr/bin/python3", 42)
    assert captured_env.get("CHECKPOINT_SINCE_NS") == "42"


def test_run_snapshot_without_since_ns_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    has_since_env = False

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        nonlocal has_since_env
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                if cmd[i + 1].startswith("CHECKPOINT_SINCE_NS="):
                    has_since_env = True
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_build_snapshot_json(entries={}, changed={}), stderr=""
        )

    monkeypatch.setattr(ccp.subprocess, "run", fake_run)
    _run_snapshot(_FAKE_CONTAINER, "/usr/bin/python3", None)
    assert has_since_env is False


def test_container_snapshot_script_detects_backdated_size_change(
    tmp_path: Path,
) -> None:
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    changed_file = testbed / "changed.txt"
    unchanged_file = testbed / "unchanged.txt"
    changed_file.write_bytes(b"old")
    unchanged_file.write_bytes(b"same")
    old_mtime_ns = time.time_ns() - 1_000_000_000
    os.utime(changed_file, ns=(old_mtime_ns, old_mtime_ns))
    os.utime(unchanged_file, ns=(old_mtime_ns, old_mtime_ns))
    prev_entries = {
        "changed.txt": ["file", len(b"old"), old_mtime_ns],
        "unchanged.txt": ["file", len(b"same"), old_mtime_ns],
    }

    changed_content = b"changed with different size"
    changed_file.write_bytes(changed_content)
    os.utime(changed_file, ns=(old_mtime_ns, old_mtime_ns))

    script = ccp._CONTAINER_SNAPSHOT_SCRIPT.replace(
        'root = "/testbed"',
        f"root = {json.dumps(str(testbed))}",
        1,
    )
    env = dict(os.environ)
    env["CHECKPOINT_SINCE_NS"] = str(old_mtime_ns + 1)
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(prev_entries),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    snapshot = json.loads(result.stdout)

    assert set(snapshot["changed"]) == {"changed.txt"}
    assert snapshot["changed"]["changed.txt"]["size"] == len(changed_content)
    assert snapshot["changed"]["changed.txt"]["mtime_ns"] == old_mtime_ns
    assert snapshot["entries"]["changed.txt"] == [
        "file",
        len(changed_content),
        old_mtime_ns,
    ]
    assert snapshot["entries"]["unchanged.txt"] == [
        "file",
        len(b"same"),
        old_mtime_ns,
    ]


def test_run_snapshot_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccp.subprocess, "run",
                        _fake_run_sequence([{"stdout": "", "stderr": "walk error", "rc": 1}]))
    with pytest.raises(RuntimeError, match="checkpoint snapshot failed"):
        _run_snapshot(_FAKE_CONTAINER, "/usr/bin/python3", None)


# ---------------------------------------------------------------------------
# _fetch_blobs
# ---------------------------------------------------------------------------

def test_fetch_blobs_empty_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        nonlocal called
        called = True
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(ccp.subprocess, "run", fake_run)
    result = _fetch_blobs(_FAKE_CONTAINER, "/usr/bin/python3", [])
    assert result == {}
    assert not called


def test_fetch_blobs_returns_file_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    tar_bytes = _make_tar_bytes({"src/main.py": b"hello", "README.md": b"world"})
    monkeypatch.setattr(ccp.subprocess, "Popen", _fake_popen_sequence([tar_bytes]))
    result = _fetch_blobs(_FAKE_CONTAINER, "/usr/bin/python3", ["src/main.py", "README.md"])
    assert result == {"src/main.py": b"hello", "README.md": b"world"}


def test_fetch_blobs_skips_directories_in_tar(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        dir_info = tarfile.TarInfo("src")
        dir_info.type = tarfile.DIRTYPE
        tf.addfile(dir_info)
        file_info = tarfile.TarInfo("src/main.py")
        content = b"hello"
        file_info.size = len(content)
        file_info.type = tarfile.REGTYPE
        tf.addfile(file_info, io.BytesIO(content))
    tar_bytes = buf.getvalue()
    monkeypatch.setattr(ccp.subprocess, "Popen", _fake_popen_sequence([tar_bytes]))
    result = _fetch_blobs(_FAKE_CONTAINER, "/usr/bin/python3", ["src", "src/main.py"])
    assert list(result.keys()) == ["src/main.py"]


# ---------------------------------------------------------------------------
# _compute_deleted_paths
# ---------------------------------------------------------------------------

def test_compute_deleted_paths_no_prev() -> None:
    assert _compute_deleted_paths(None, {"a": "file"}) == []


def test_compute_deleted_paths_file_deleted() -> None:
    prev = {"a": "file", "b": "file"}
    curr = {"a": "file"}
    assert _compute_deleted_paths(prev, curr) == ["b"]


def test_compute_deleted_paths_type_changed() -> None:
    prev = {"a": "file", "d": "dir"}
    curr = {"a": "dir", "d": "dir"}
    assert _compute_deleted_paths(prev, curr) == ["a"]


def test_compute_deleted_paths_file_added_not_in_deleted() -> None:
    prev = {"a": "file"}
    curr = {"a": "file", "b": "file"}
    assert _compute_deleted_paths(prev, curr) == []


def test_compute_deleted_paths_ignores_file_size_mtime_changes() -> None:
    prev = {"a": ("file", 10, 100)}
    curr = {"a": ("file", 20, 100)}
    assert _compute_deleted_paths(prev, curr) == []


def test_compute_deleted_paths_symlink_target_change() -> None:
    prev = {"link": {"type": "symlink", "target": "a"}}
    curr = {"link": {"type": "symlink", "target": "b"}}
    assert _compute_deleted_paths(prev, curr) == ["link"]


def test_compute_deleted_paths_skips_git() -> None:
    prev = {".git": "dir", ".git/config": "file", "a": "file"}
    curr: dict[str, str] = {}
    assert _compute_deleted_paths(prev, curr) == ["a"]


def _clear_probe_cache() -> None:
    for attr in dir(ccp._probe_container_python):
        if attr.startswith("_probe_cache_"):
            try:
                delattr(ccp._probe_container_python, attr)
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# run_container_checkpoint — full snapshot (first checkpoint)
# ---------------------------------------------------------------------------

def test_first_checkpoint_produces_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    content = b"hello world\n"
    digest = _sha256(content)
    entries = {"src": "dir", "src/main.py": "file"}
    changed = {
        "src/main.py": {"hash": digest, "mode": 0o644, "size": len(content), "mtime_ns": 1000}
    }
    now_ns = int(time.time() * 1e9)

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries, changed=changed, now_ns=now_ns),
        ]),
    )
    monkeypatch.setattr(
        ccp.subprocess,
        "Popen",
        _fake_popen_sequence([_make_tar_bytes({"src/main.py": content})]),
    )

    manifest_path = tmp_path / "manifest.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-f1", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )

    assert "error" not in result
    assert "skipped" not in result
    assert result["kind"] == "cas_manifest_full"
    assert result["incremental"] is False
    assert result["incremental_since_ns"] is None
    assert result["root"] == "/testbed"
    assert result["overhead_excluded"] is True
    assert result["elapsed_ms"] >= 0
    assert result["size_bytes"] == manifest_path.stat().st_size
    assert result["chain_bytes"] == len(content)

    state = result["_state"]
    assert state["now_ns"] == now_ns
    assert state["current_snapshot"] == entries
    assert state["is_full"] is True

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sorted(manifest) == ["deleted_paths", "entries"]
    assert manifest["deleted_paths"] == []
    assert set(manifest["entries"]) == {"src/main.py"}
    assert manifest["entries"]["src/main.py"]["hash"] == digest
    assert manifest["entries"]["src/main.py"]["size"] == len(content)
    assert manifest["entries"]["src/main.py"]["mode"] == 0o644
    assert manifest["entries"]["src/main.py"]["mtime_ns"] == 1000

    blob_path = cas_root / "blobs" / digest[:2] / digest[2:]
    assert blob_path.read_bytes() == content


# ---------------------------------------------------------------------------
# run_container_checkpoint — manifest structure match with host
# ---------------------------------------------------------------------------

def test_container_manifest_matches_host_manifest_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.openclaw._session_runner as session_runner

    host_root = tmp_path / "host_testbed"
    host_root.mkdir()
    (host_root / "src").mkdir()
    (host_root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (host_root / "README.md").write_text("# Test\n", encoding="utf-8")
    (host_root / "empty_dir").mkdir()

    host_cas_root = tmp_path / "host_cas"
    monkeypatch.setattr(session_runner, "_CHECKPOINT_CAS_ROOT", host_cas_root)
    host_manifest_path = tmp_path / "host_manifest.json"
    session_runner._write_cas_manifest(
        root=host_root, manifest_path=host_manifest_path,
        incremental_since_ns=None, deleted_paths=[],
    )
    host_manifest = json.loads(host_manifest_path.read_text(encoding="utf-8"))

    container_cas_root = tmp_path / "container_cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", container_cas_root)

    entries = {"src": "dir", "src/main.py": "file", "empty_dir": "dir", "README.md": "file"}
    changed = {}
    tar_files = {}
    for rel, content_bytes in [
        ("src/main.py", b"print('hello')\n"),
        ("README.md", b"# Test\n"),
    ]:
        st = os.lstat(host_root / rel)
        changed[rel] = {
            "hash": _sha256(content_bytes),
            "mode": stat.S_IMODE(st.st_mode),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
        tar_files[rel] = content_bytes

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries, changed=changed),
        ]),
    )
    monkeypatch.setattr(
        ccp.subprocess,
        "Popen",
        _fake_popen_sequence([_make_tar_bytes(tar_files)]),
    )

    container_manifest_path = tmp_path / "container_manifest.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-m", "executable": "docker"},
        manifest_path=container_manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )
    assert "error" not in result

    container_manifest = json.loads(container_manifest_path.read_text(encoding="utf-8"))
    assert sorted(container_manifest) == sorted(host_manifest)
    assert container_manifest["deleted_paths"] == host_manifest["deleted_paths"]
    assert set(container_manifest["entries"]) == set(host_manifest["entries"])
    for rel in host_manifest["entries"]:
        ce = container_manifest["entries"][rel]
        he = host_manifest["entries"][rel]
        assert ce["hash"] == he["hash"], f"hash mismatch for {rel}"
        assert ce["size"] == he["size"], f"size mismatch for {rel}"
        assert ce["mode"] == he["mode"], f"mode mismatch for {rel}"
        assert ce["mtime_ns"] == he["mtime_ns"], f"mtime_ns mismatch for {rel}"


def test_source_checkpoint_walk_skips_git(tmp_path: Path) -> None:
    import agents.openclaw._session_runner as session_runner

    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git_dir = testbed / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n", encoding="utf-8")

    entries = session_runner._snapshot_checkpoint_entries(testbed)
    assert set(entries) == {"tracked.txt"}
    assert entries["tracked.txt"][0] == "file"

    marker_mtime_ns = time.time_ns()
    old_mtime_ns = marker_mtime_ns - 1_000_000_000
    new_mtime_ns = marker_mtime_ns + 1_000_000_000
    os.utime(testbed / "tracked.txt", ns=(old_mtime_ns, old_mtime_ns))
    os.utime(git_dir / "config", ns=(new_mtime_ns, new_mtime_ns))
    os.utime(git_dir, ns=(new_mtime_ns, new_mtime_ns))
    os.utime(testbed, ns=(old_mtime_ns, old_mtime_ns))

    assert session_runner._any_file_newer_than(testbed, marker_mtime_ns) is False


def test_source_checkpoint_manifest_records_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.openclaw._session_runner as session_runner

    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "target.txt").write_text("target\n", encoding="utf-8")
    (testbed / "link.txt").symlink_to("target.txt")
    (testbed / "real-dir").mkdir()
    (testbed / "dir-link").symlink_to("real-dir", target_is_directory=True)

    snapshot = session_runner._snapshot_checkpoint_entries(testbed)
    assert snapshot["link.txt"][0] == {"type": "symlink", "target": "target.txt"}
    assert snapshot["dir-link"][0] == {"type": "symlink", "target": "real-dir"}

    monkeypatch.setattr(session_runner, "_CHECKPOINT_CAS_ROOT", tmp_path / "cas")
    manifest_path = tmp_path / "manifest.json"
    chain_bytes = session_runner._write_cas_manifest(
        root=testbed,
        manifest_path=manifest_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"]["link.txt"] == {
        "type": "symlink",
        "target": "target.txt",
    }
    assert manifest["entries"]["dir-link"] == {
        "type": "symlink",
        "target": "real-dir",
    }
    assert manifest["entries"]["target.txt"]["size"] == len("target\n")
    assert chain_bytes == len("target\n")


# ---------------------------------------------------------------------------
# run_container_checkpoint — incremental
# ---------------------------------------------------------------------------

def test_incremental_checkpoint_only_contains_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    prev_entries = {"src": "dir", "src/main.py": "file", "README.md": "file"}
    changed = {"src/main.py": {"hash": _sha256(b"updated"), "mode": 0o644, "size": 7, "mtime_ns": 2000}}
    now_ns = 2000

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=prev_entries, changed=changed, now_ns=now_ns),
        ]),
    )
    monkeypatch.setattr(
        ccp.subprocess,
        "Popen",
        _fake_popen_sequence([_make_tar_bytes({"src/main.py": b"updated"})]),
    )

    manifest_path = tmp_path / "inc_manifest.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-i", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=1000,
        prev_snapshot_entries=prev_entries,
    )

    assert "error" not in result
    assert result["kind"] == "cas_manifest_incremental"
    assert result["incremental"] is True
    assert result["incremental_since_ns"] == 1000

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["entries"]) == {"src/main.py"}
    assert manifest["deleted_paths"] == []


def test_incremental_checkpoint_detects_deleted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    prev_entries = {"src": "dir", "src/main.py": "file", "README.md": "file"}
    current_entries = {"src": "dir", "src/main.py": "file"}

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=current_entries, changed={}),
        ]),
    )

    manifest_path = tmp_path / "del_manifest.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-d", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=1000,
        prev_snapshot_entries=prev_entries,
    )

    assert "error" not in result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["deleted_paths"] == ["README.md"]
    assert manifest["entries"] == {}


def test_incremental_detects_type_change_as_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    prev_entries = {"x": "file"}
    current_entries = {"x": "dir"}

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=current_entries, changed={}),
        ]),
    )

    manifest_path = tmp_path / "type_manifest.json"
    run_container_checkpoint(
        container_runtime={"id": "test-cid-t", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=1000,
        prev_snapshot_entries=prev_entries,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["deleted_paths"] == ["x"]


# ---------------------------------------------------------------------------
# run_container_checkpoint — no-change skip
# ---------------------------------------------------------------------------

def test_no_change_returns_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    entries = {"src": "dir", "src/main.py": "file"}

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries, changed={}),
        ]),
    )

    manifest_path = tmp_path / "skip_manifest.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-s", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=1000,
        prev_snapshot_entries=entries,
    )

    assert "skipped" in result
    assert "_state" not in result
    assert result["skipped"] == "no filesystem changes since last checkpoint"
    assert result["overhead_excluded"] is True
    assert not manifest_path.exists()


def test_first_checkpoint_never_skips_even_with_empty_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    entries = {"README.md": "file"}

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries, changed={}),
        ]),
    )

    manifest_path = tmp_path / "first_empty.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-fe", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )

    assert "skipped" not in result
    assert result["kind"] == "cas_manifest_full"


# ---------------------------------------------------------------------------
# run_container_checkpoint — error cases
# ---------------------------------------------------------------------------

def test_blob_hash_mismatch_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas_root = tmp_path / "cas"
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    digest = _sha256(b"real content")
    entries = {"f.txt": "file"}
    changed = {"f.txt": {"hash": digest, "mode": 0o644, "size": 12, "mtime_ns": 1000}}
    wrong_content = b"wrong content"

    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries, changed=changed),
        ]),
    )
    monkeypatch.setattr(
        ccp.subprocess,
        "Popen",
        _fake_popen_sequence([_make_tar_bytes({"f.txt": wrong_content})]),
    )

    manifest_path = tmp_path / "mismatch.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-hm", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )

    assert "error" in result
    assert "_state" not in result
    assert "digest mismatch" in result["error"]
    assert not manifest_path.exists()


def test_python_probe_failure_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([{"stdout": "", "stderr": "no python", "rc": 1}]),
    )

    manifest_path = tmp_path / "nopy.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-pf", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )

    assert "error" in result
    assert "_state" not in result
    assert "no usable python" in result["error"]


def test_changed_during_walk_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(
                entries={"a": "file"},
                changed={"a": {"hash": _sha256(b"x"), "mode": 0o644, "size": 1, "mtime_ns": 1}},
                changed_during_walk=True,
            ),
        ]),
    )

    manifest_path = tmp_path / "cdw.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-cw", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )

    assert "error" in result
    assert "_state" not in result
    assert "filesystem changed during checkpoint" in result["error"]
    assert not manifest_path.exists()


def test_symlinks_in_snapshot_are_manifested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = {"link.txt": {"type": "symlink", "target": "target.txt"}}
    changed = {"link.txt": {"type": "symlink", "target": "target.txt"}}
    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries, changed=changed, symlinks=["link.txt"]),
        ]),
    )

    manifest_path = tmp_path / "sym.json"
    result = run_container_checkpoint(
        container_runtime={"id": "test-cid-sl", "executable": "docker"},
        manifest_path=manifest_path,
        incremental_since_ns=None,
        prev_snapshot_entries=None,
    )

    assert "error" not in result
    assert result["kind"] == "cas_manifest_full"
    assert result["chain_bytes"] == 0
    assert result["_state"]["current_snapshot"] == entries
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"] == {
        "link.txt": {"type": "symlink", "target": "target.txt"}
    }
    assert manifest["deleted_paths"] == []


# ---------------------------------------------------------------------------
# run_container_checkpoint — rebaseline (via hook)
# ---------------------------------------------------------------------------

def test_rebaseline_when_chain_bytes_exceeds_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When chain_bytes_since_full >= rebaseline_bytes, next checkpoint promotes to full."""
    import agents.openclaw._session_runner as session_runner

    cas_root = tmp_path / "cas"
    monkeypatch.setattr(session_runner, "_CHECKPOINT_CAS_ROOT", cas_root)
    monkeypatch.setattr(ccp, "_CHECKPOINT_CAS_ROOT", cas_root)

    trace_file = tmp_path / "trace.jsonl"
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    content = b"x" * 100
    digest = _sha256(content)
    entries = {"a.txt": "file"}
    changed = {"a.txt": {"hash": digest, "mode": 0o644, "size": 100, "mtime_ns": 2000}}
    now_ns = 2000

    def make_fake_run():
        _clear_probe_cache()
        monkeypatch.setattr(
            ccp.subprocess, "run",
            _fake_run_sequence([
                _probe_response(),
                _build_snapshot_json(entries=entries, changed=changed, now_ns=now_ns),
            ]),
        )
        monkeypatch.setattr(
            ccp.subprocess,
            "Popen",
            _fake_popen_sequence([_make_tar_bytes({"a.txt": content})]),
        )

    hook = session_runner.TraceCollectorHook(
        trace_file,
        instance_id="test-rebase",
        checkpoint_root=Path("/testbed"),
        checkpoint_dir=checkpoints_dir,
        checkpoint_root_label="/testbed",
        checkpoint_rebaseline_bytes=80,
        container_runtime={"id": "test-cid-rb", "executable": "docker"},
    )

    # First checkpoint: full
    make_fake_run()
    result1 = hook._checkpoint_after_tool(
        tool_call_id="call_1", tool_name="exec",
        tool_args_json='{"command":"ls"}',
    )
    assert result1 is not None
    assert result1["kind"] == "cas_manifest_full"
    assert result1["chain_bytes"] == 100
    assert hook._checkpoint_chain_bytes_since_full == 0

    # Simulate chain accumulation: manually push past threshold
    hook._checkpoint_chain_bytes_since_full = 100  # >= 80 → rebaseline

    # Next checkpoint triggers rebaseline (promoted to full)
    entries2 = {"a.txt": "file", "b.txt": "file"}
    changed2 = {"b.txt": {"hash": _sha256(b"yy"), "mode": 0o644, "size": 2, "mtime_ns": 3000}}
    now_ns2 = 3000
    _clear_probe_cache()
    monkeypatch.setattr(
        ccp.subprocess, "run",
        _fake_run_sequence([
            _probe_response(),
            _build_snapshot_json(entries=entries2, changed=changed2, now_ns=now_ns2),
        ]),
    )
    monkeypatch.setattr(
        ccp.subprocess,
        "Popen",
        _fake_popen_sequence([_make_tar_bytes({"b.txt": b"yy"})]),
    )
    hook._last_full_checkpoint_ns = 1000
    hook._last_incremental_checkpoint_ns = 1000
    hook._checkpoint_snapshot_entries = entries

    result2 = hook._checkpoint_after_tool(
        tool_call_id="call_2", tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert result2 is not None
    assert result2["kind"] == "cas_manifest_full"
    assert result2["rebaseline"] is True
    assert hook._checkpoint_chain_bytes_since_full == 0


# ---------------------------------------------------------------------------
# _CONTAINER_PYTHON_CANDIDATES re-export
# ---------------------------------------------------------------------------

def test_container_python_candidates_is_importable_from_task_container() -> None:
    from trace_collect.runtime.task_container import _CONTAINER_PYTHON_CANDIDATES as tc_candidates
    assert tc_candidates == _CONTAINER_PYTHON_CANDIDATES


def test_container_python_candidates_is_importable_from_openclaw_tools() -> None:
    from trace_collect.openclaw_tools import _CONTAINER_PYTHON_CANDIDATES as ot_candidates
    assert ot_candidates == _CONTAINER_PYTHON_CANDIDATES


# ---------------------------------------------------------------------------
# Docker integration test
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
def test_container_checkpoint_end_to_end_docker(tmp_path: Path) -> None:
    """Full -> incremental -> delete chain using a real python:3.11-alpine container."""
    cas_root = tmp_path / "cas"
    original_cas = ccp._CHECKPOINT_CAS_ROOT

    try:
        # Start a container with /testbed
        container_name = f"ckpt-test-{os.getpid()}"
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name,
             "python:3.11-alpine", "sleep", "3600"],
            capture_output=True, check=True, timeout=60,
        )
        cr = {"id": container_name, "executable": "docker"}

        # Create initial files
        subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", "/testbed/src"],
            capture_output=True, check=True, timeout=10,
        )
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-c",
             "echo 'hello' > /testbed/src/main.py"],
            capture_output=True, check=True, timeout=10,
        )
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-c",
             "echo 'readme' > /testbed/README.md"],
            capture_output=True, check=True, timeout=10,
        )

        # Override CAS root for this test
        import agents.openclaw._checkpoint_container as ccp_mod
        ccp_mod._CHECKPOINT_CAS_ROOT = cas_root

        # Clear probe cache
        for attr in dir(ccp_mod._probe_container_python):
            if attr.startswith("_probe_cache_"):
                try:
                    delattr(ccp_mod._probe_container_python, attr)
                except AttributeError:
                    pass

        # Full checkpoint
        mf1 = tmp_path / "full.json"
        result1 = run_container_checkpoint(
            container_runtime=cr,
            manifest_path=mf1,
            incremental_since_ns=None,
            prev_snapshot_entries=None,
        )
        assert "error" not in result1, f"Full checkpoint error: {result1.get('error')}"
        assert result1["kind"] == "cas_manifest_full"
        m1 = json.loads(mf1.read_text(encoding="utf-8"))
        assert "src/main.py" in m1["entries"]
        assert "README.md" in m1["entries"]
        assert m1["deleted_paths"] == []

        ent1_digest = m1["entries"]["src/main.py"]["hash"]
        blob1 = cas_root / "blobs" / ent1_digest[:2] / ent1_digest[2:]
        assert blob1.exists()
        assert blob1.read_bytes() == b"hello\n"

        full_entries = result1["_state"]["current_snapshot"]
        full_ns = result1["_state"]["now_ns"]

        # Modify a file
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-c",
             "echo 'updated' > /testbed/src/main.py"],
            capture_output=True, check=True, timeout=10,
        )

        # Incremental checkpoint
        mf2 = tmp_path / "inc.json"
        result2 = run_container_checkpoint(
            container_runtime=cr,
            manifest_path=mf2,
            incremental_since_ns=full_ns,
            prev_snapshot_entries=full_entries,
        )
        assert "error" not in result2, f"Incremental error: {result2.get('error')}"
        assert result2["kind"] == "cas_manifest_incremental"
        m2 = json.loads(mf2.read_text(encoding="utf-8"))
        assert set(m2["entries"]) == {"src/main.py"}
        assert m2["entries"]["src/main.py"]["hash"] != ent1_digest
        assert m2["deleted_paths"] == []

        inc_entries = result2["_state"]["current_snapshot"]
        inc_ns = result2["_state"]["now_ns"]

        # Delete a file
        subprocess.run(
            ["docker", "exec", container_name, "rm", "/testbed/README.md"],
            capture_output=True, check=True, timeout=10,
        )

        # Incremental after delete
        mf3 = tmp_path / "del.json"
        result3 = run_container_checkpoint(
            container_runtime=cr,
            manifest_path=mf3,
            incremental_since_ns=inc_ns,
            prev_snapshot_entries=inc_entries,
        )
        assert "error" not in result3, f"Delete checkpoint error: {result3.get('error')}"
        m3 = json.loads(mf3.read_text(encoding="utf-8"))
        assert m3["deleted_paths"] == ["README.md"]

        # Verify simulator can read the manifest
        from trace_collect.simulator import _load_source_manifest_entries
        loaded = _load_source_manifest_entries(mf1)
        assert "src/main.py" in loaded
        assert "README.md" in loaded

    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=10,
        )
        ccp_mod._CHECKPOINT_CAS_ROOT = original_cas
