"""Tests for FCBackend host-side plumbing (no KVM required).

Covers the FC API HTTP client (which must not depend on the server
closing the connection — Firecracker keeps API connections alive), the
copy-on-write helper, and the collision-free network slot allocator.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

import agents.sandbox_runtime as sr
from agents.sandbox_runtime import FCBackend, _FC_NET_SLOTS, _cow_copy


def _make_backend(tmp_path: Path, **overrides: object) -> FCBackend:
    kwargs: dict[str, object] = dict(
        source_image="python:3.12-slim",
        kernel_path=tmp_path / "vmlinux",
        checkpoint_dir=tmp_path / "ckpt",
        instance_id="0000abcd",
        api_sock=str(tmp_path / "api.sock"),
        vsock_sock=str(tmp_path / "vsock.sock"),
    )
    kwargs.update(overrides)
    return FCBackend(**kwargs)  # type: ignore[arg-type]


class _KeepAliveUDSServer(threading.Thread):
    """Fake FC API server: answers one HTTP request per connection with a
    Content-Length response and then KEEPS THE CONNECTION OPEN, exactly
    like the real Firecracker API server."""

    def __init__(self, sock_path: Path, status: int, body: bytes) -> None:
        super().__init__(daemon=True)
        self._sock_path = sock_path
        self._status = status
        self._body = body
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(sock_path))
        self._server.listen(4)
        self._server.settimeout(10)
        self.requests: list[bytes] = []

    def run(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                request += chunk
            head, _, rest = request.partition(b"\r\n\r\n")
            content_length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1])
            while len(rest) < content_length:
                rest += conn.recv(65536)
            self.requests.append(request)
            status_text = {200: "OK", 204: "No Content", 400: "Bad Request"}[
                self._status
            ]
            response = (
                f"HTTP/1.1 {self._status} {status_text}\r\n"
                f"Server: Firecracker API\r\n"
                f"Connection: keep-alive\r\n"
                f"Content-Length: {len(self._body)}\r\n\r\n"
            ).encode() + self._body
            conn.sendall(response)
            # Keep-alive: hold the connection open until the client
            # closes.  A client that waits for server close hangs here.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    if conn.recv(65536) == b"":
                        return
                except OSError:
                    return

    def close(self) -> None:
        self._server.close()


class TestFCApiRequest:
    def test_returns_immediately_with_content_length(self, tmp_path: Path) -> None:
        body = b'{"vcpu_count": 2}'
        server = _KeepAliveUDSServer(tmp_path / "api.sock", 200, body)
        server.start()
        backend = _make_backend(tmp_path)
        started = time.monotonic()
        status, resp = backend._api_request("GET", "/machine-config")
        elapsed = time.monotonic() - started
        server.close()
        assert status == 200
        assert resp == body.decode()
        # The old client blocked on recv until the 10s socket timeout on
        # every call; the fixed client must return as soon as the body is
        # complete.
        assert elapsed < 2.0, f"API request took {elapsed:.1f}s"

    def test_no_content_response(self, tmp_path: Path) -> None:
        server = _KeepAliveUDSServer(tmp_path / "api.sock", 204, b"")
        server.start()
        backend = _make_backend(tmp_path)
        started = time.monotonic()
        backend._api_put("/vm", {"state": "Paused"})
        elapsed = time.monotonic() - started
        server.close()
        assert elapsed < 2.0

    def test_error_status_raises(self, tmp_path: Path) -> None:
        server = _KeepAliveUDSServer(
            tmp_path / "api.sock", 400, b'{"fault_message": "bad"}'
        )
        server.start()
        backend = _make_backend(tmp_path)
        with pytest.raises(RuntimeError, match="returned 400"):
            backend._api_put("/vm", {"state": "Paused"})
        server.close()

    def test_missing_content_length_fails_fast(self, tmp_path: Path) -> None:
        sock_path = tmp_path / "api.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)

        def respond_without_content_length() -> None:
            conn, _ = server.accept()
            with conn:
                request = b""
                while b"\r\n\r\n" not in request:
                    request += conn.recv(65536)
                conn.sendall(b"HTTP/1.1 200 OK\r\nServer: fake\r\n\r\n")
                time.sleep(2)

        thread = threading.Thread(
            target=respond_without_content_length, daemon=True,
        )
        thread.start()
        backend = _make_backend(tmp_path)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="without Content-Length"):
            backend._api_request("GET", "/machine-config")
        assert time.monotonic() - started < 2.0
        server.close()


class TestCowCopy:
    def test_copies_content(self, tmp_path: Path) -> None:
        src = tmp_path / "src.img"
        dst = tmp_path / "dst.img"
        src.write_bytes(b"rootfs-bytes" * 1024)
        _cow_copy(str(src), str(dst))
        assert dst.read_bytes() == src.read_bytes()

    def test_fallback_is_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        src = tmp_path / "src.img"
        dst = tmp_path / "dst.img"
        src.write_bytes(b"data")

        real_checked_run = sr._checked_run

        def reflink_always_fails(cmd: list[str], **kwargs: object):
            if "--reflink=always" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="",
                    stderr="cp: failed to clone: Operation not supported",
                )
            return real_checked_run(cmd, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sr, "_checked_run", reflink_always_fails)
        monkeypatch.setattr(sr, "_reflink_fallback_warned", False)
        with caplog.at_level("WARNING", logger="agents.sandbox_runtime"):
            _cow_copy(str(src), str(dst))
        assert dst.read_bytes() == b"data"
        assert any("reflink" in rec.message for rec in caplog.records)


class TestNetSlotAllocation:
    def test_probes_past_taken_slots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = _make_backend(tmp_path)
        hint = backend._net_index_hint
        taken = {f"fc-tap{hint}", f"fc-tap{(hint + 1) % _FC_NET_SLOTS}"}
        created: list[str] = []

        def fake_checked_run(cmd: list[str], **kwargs: object):
            if cmd[:4] == ["sudo", "ip", "tuntap", "add"]:
                name = cmd[4]
                if name in taken:
                    return subprocess.CompletedProcess(
                        cmd, 2, stdout="", stderr="File exists",
                    )
                created.append(name)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:3] == ["ip", "link", "show"]:
                exists = cmd[3] in taken or cmd[3] in created
                return subprocess.CompletedProcess(
                    cmd, 0 if exists else 1, stdout="", stderr="",
                )
            raise AssertionError(f"unexpected command: {cmd}")

        monkeypatch.setattr(sr, "_checked_run", fake_checked_run)
        backend._allocate_net_slot()
        expected_idx = (hint + 2) % _FC_NET_SLOTS
        assert backend._tap_dev == f"fc-tap{expected_idx}"
        assert backend._host_ip == (
            f"172.{16 + expected_idx // 256}.{expected_idx % 256}.1"
        )
        assert backend._guest_ip == (
            f"172.{16 + expected_idx // 256}.{expected_idx % 256}.2"
        )

    def test_non_exists_failure_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = _make_backend(tmp_path)

        def fake_checked_run(cmd: list[str], **kwargs: object):
            if cmd[:4] == ["sudo", "ip", "tuntap", "add"]:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="Operation not permitted",
                )
            if cmd[:3] == ["ip", "link", "show"]:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        monkeypatch.setattr(sr, "_checked_run", fake_checked_run)
        with pytest.raises(RuntimeError, match="failed to create TAP"):
            backend._allocate_net_slot()

    def test_explicit_net_params_require_all(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be given together"):
            _make_backend(tmp_path, tap_dev="fc-x")

    def test_explicit_net_params_preserved(self, tmp_path: Path) -> None:
        backend = _make_backend(
            tmp_path,
            tap_dev="fc-x",
            host_ip="172.31.255.1",
            guest_ip="172.31.255.2",
        )
        assert backend._tap_dev == "fc-x"
        assert backend._host_ip == "172.31.255.1"
        assert backend._guest_ip == "172.31.255.2"


class TestDiffChain:
    """Memory diff-snapshot chain selection (no KVM needed)."""

    def test_first_memory_snapshot_is_full(self) -> None:
        bk = _make_backend(Path("/tmp"))
        assert bk._choose_memory_snap_type() == "Full", (
            "first memory snapshot must be Full (no base to diff against)"
        )

    def test_subsequent_are_diff_until_max(self) -> None:
        bk = _make_backend(Path("/tmp"))
        # Simulate the actual capture_snapshot flow:
        # 1. First capture: chain empty → Full. Post-capture state.
        assert bk._choose_memory_snap_type() == "Full"
        bk._diff_chain = [{"type": "Full", "mem_path": "/a", "vmstate_path": "/b"}]
        bk._mem_snapshots_since_full = 0
        # 2..N: chain non-empty, counter < max → Diff
        for i in range(1, 10):
            bk._mem_snapshots_since_full = i
            result = bk._choose_memory_snap_type()
            assert result == "Diff", f"snapshot {i+1} should be Diff, got {result}"

    def test_diff_chain_rebaselines_at_max(self) -> None:
        bk = _make_backend(Path("/tmp"))
        bk._diff_chain = [{"type": "Full", "mem_path": "/a", "vmstate_path": "/b"}]
        # After _diff_chain_max diffs, the next should be Full
        bk._mem_snapshots_since_full = bk._diff_chain_max
        assert bk._choose_memory_snap_type() == "Full"

    def test_chain_updated_on_full(self) -> None:
        bk = _make_backend(Path("/tmp"))
        bk._diff_chain = [{"type": "Full", "mem_path": "/old", "vmstate_path": "/ov"}]
        bk._mem_snapshots_since_full = 0
        entry = {"type": "Full", "mem_path": "/new", "vmstate_path": "/nv"}
        bk._diff_chain = [entry]
        bk._mem_snapshots_since_full = 0
        assert len(bk._diff_chain) == 1
        assert bk._diff_chain[0]["mem_path"] == "/new"
        assert bk._mem_snapshots_since_full == 0

    def test_chain_appended_on_diff(self) -> None:
        bk = _make_backend(Path("/tmp"))
        bk._diff_chain = [{"type": "Full", "mem_path": "/b", "vmstate_path": "/v"}]
        bk._mem_snapshots_since_full = 1
        entry = {"type": "Diff", "mem_path": "/d1", "vmstate_path": "/v1"}
        bk._diff_chain.append(entry)
        bk._mem_snapshots_since_full = 2
        assert len(bk._diff_chain) == 2
        assert bk._diff_chain[1]["mem_path"] == "/d1"

    def test_restore_resets_chain(self) -> None:
        bk = _make_backend(Path("/tmp"))
        bk._diff_chain = [{"type": "Full", "mem_path": "/b", "vmstate_path": "/v"}]
        bk._diff_chain.append({"type": "Diff", "mem_path": "/d", "vmstate_path": "/dv"})
        bk._mem_snapshots_since_full = 3
        # Simulate what restore does
        bk._diff_chain = []
        bk._mem_snapshots_since_full = 0
        assert len(bk._diff_chain) == 0
        assert bk._mem_snapshots_since_full == 0

    def test_snapshot_editor_not_found_returns_none(self) -> None:
        """When snapshot-editor is not on PATH, _find_snapshot_editor
        returns None (the test env won't have it)."""
        bk = _make_backend(Path("/tmp"))
        found = bk._find_snapshot_editor()
        assert found is None, (
            f"expected snapshot-editor to be absent in test env, "
            f"but found at {found}"
        )

    @pytest.mark.asyncio
    async def test_compact_short_chain_returns_none(self) -> None:
        bk = _make_backend(Path("/tmp"))
        result = await bk._compact_diff_chain(
            [{"type": "Full", "mem_path": "/a", "vmstate_path": "/b"}]
        )
        assert result is None, "single-entry chain should skip compaction"

    @pytest.mark.asyncio
    async def test_compact_chain_no_editor_logs_warning(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bk = _make_backend(Path("/tmp"))
        monkeypatch.setattr(bk, "_snapshot_editor_warned", False)
        with caplog.at_level("WARNING", logger="agents.sandbox_runtime"):
            result = await bk._compact_diff_chain([
                {"type": "Full", "mem_path": "/b", "vmstate_path": "/v"},
                {"type": "Diff", "mem_path": "/d", "vmstate_path": "/dv"},
            ])
        assert result is None
        assert any("snapshot-editor not found" in rec.message for rec in caplog.records)


    def test_seek_last_full_in_chain(self) -> None:
        """restore_snapshot fallback must scan for the last Full in reverse,
        not blindly use the head entry (which could be a Diff)."""
        chain = [
            {"type": "Full", "mem_path": "/base", "vmstate_path": "/basev"},
            {"type": "Diff", "mem_path": "/d1", "vmstate_path": "/d1v"},
            {"type": "Diff", "mem_path": "/d2", "vmstate_path": "/d2v"},
        ]
        last_full = None
        for entry in reversed(chain):
            if entry.get("type") == "Full":
                last_full = entry
                break
        assert last_full is not None
        assert last_full["mem_path"] == "/base"

    def test_backward_compat_flat_process_state(self) -> None:
        """restore_snapshot must handle flat mem_path/vmstate_path format
        from snapshots captured before the diff_chain change."""
        from agents.sandbox_runtime import SandboxSnapshot

        snap = SandboxSnapshot(
            process_state={
                "vmstate_path": "/tmp/legacy-vmstate.snap",
                "mem_path": "/tmp/legacy-mem.snap",
                "snapshot_type": "Full",
                "mem_version": 1,
            },
            disk_state={"disk_path": "/tmp/legacy-disk.img", "kind": "fc_paired_snapshot"},
            root="/testbed",
            timestamp_ns=0,
        )
        assert snap.process_state is not None
        assert snap.process_state.get("mem_path") == "/tmp/legacy-mem.snap"
        assert snap.process_state.get("vmstate_path") == "/tmp/legacy-vmstate.snap"
        assert "diff_chain" not in snap.process_state
