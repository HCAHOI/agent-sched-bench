from __future__ import annotations

import json
import socket
import socketserver
import stat
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tool_resource.sidecar_protocol import (
    PROTOCOL_VERSION,
    SidecarProtocolError,
    SidecarUnavailableError,
    UnixSocketTransport,
    receive_message,
    send_message,
)
from tool_resource.sidecar_server import ToolResourceSidecarServer
from trace_collect.openclaw_host_runtime import _start_tool_resource_sidecar


class _Collector:
    def __init__(self, **kwargs: Any) -> None:
        self.artifact_path = Path(kwargs["artifact_path"])
        self.container_id = kwargs["container_id"]
        self.repo = kwargs["repo"]
        self.calls: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def begin_tool_call(self, tool_call_id: str, command: str) -> tuple[str, str]:
        return tool_call_id, command

    def finish_tool_call(
        self,
        token: tuple[str, str],
        *,
        replay_response: dict[str, Any] | None,
    ) -> dict[str, Any]:
        call = {
            "tool_call_id": token[0],
            "command": token[1],
            "workload_returncode": (replay_response or {}).get("returncode"),
            "telemetry_quality": "ok",
            "eligible_for_kb": True,
            "clauses": [],
        }
        self.calls.append(call)
        return call

    def add_integrity_error(self, message: str) -> None:
        self.errors.append(message)

    def finalize(self, *, replay_execution: str) -> None:
        self.artifact_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "container_id": self.container_id,
                    "calls": self.calls,
                    "replay_execution": replay_execution,
                    "telemetry_quality": "ok",
                    "formal_completeness": "complete",
                    "call_coverage": {
                        "total_call_count": len(self.calls),
                        "eligible_call_count": len(self.calls),
                        "withheld_call_count": 0,
                        "eligible_fraction": 1.0,
                    },
                    "collection_validity": "valid",
                    "integrity": {"status": "ok", "errors": self.errors},
                    "provenance": {"repo": self.repo},
                }
            ),
            encoding="utf-8",
        )


def _serve(server: socketserver.BaseServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_versioned_protocol_returns_only_finalized_data(tmp_path: Path) -> None:
    socket_path = tmp_path / "sidecar.sock"
    server = ToolResourceSidecarServer(
        socket_path,
        collector_factory=_Collector,
        state_dir=tmp_path / "state",
    )
    thread = _serve(server)
    client = UnixSocketTransport(socket_path)
    try:
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        assert client.request("ping") == {"protocol_version": PROTOCOL_VERSION}
        opened = client.request(
            "open",
            {
                "container_id": "container-1",
                "repo": "repo-1",
                "source_actions": [],
            },
        )
        session_id = opened["session_id"]
        begun = client.request(
            "begin",
            {
                "session_id": session_id,
                "tool_call_id": "call-1",
                "command": "echo hi",
            },
        )
        finished = client.request(
            "finish",
            {
                "session_id": session_id,
                "token_id": begun["token_id"],
                "replay_response": {"returncode": 0, "result": "hi"},
            },
        )
        finalized = client.request(
            "finalize",
            {
                "session_id": session_id,
                "replay_execution": "completed",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert finished["call"]["tool_call_id"] == "call-1"
    assert finished["call"]["workload_returncode"] == 0
    assert finalized["artifact"]["calls"] == [finished["call"]]
    assert not list((tmp_path / "state").glob("*.json"))


def test_protocol_rejects_wrong_version_and_oversized_message(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "sidecar.sock"
    server = ToolResourceSidecarServer(
        socket_path,
        collector_factory=_Collector,
        state_dir=tmp_path / "state",
    )
    thread = _serve(server)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
            send_message(
                connection,
                {
                    "version": PROTOCOL_VERSION + 1,
                    "id": "bad-version",
                    "operation": "ping",
                    "payload": {},
                },
            )
            response = receive_message(connection)
        assert response["ok"] is False
        assert "unsupported protocol version" in response["error"]
        with (
            pytest.raises(SidecarProtocolError, match="exceeds limit"),
            socket.socket() as connection,
        ):
            send_message(connection, {"too_large": "value"}, max_message_bytes=4)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_finalize_closes_stale_concurrent_session_reference(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "sidecar.sock"
    server = ToolResourceSidecarServer(
        socket_path,
        collector_factory=_Collector,
        state_dir=tmp_path / "state",
    )
    session_id = server._open(
        {"container_id": "container-1", "repo": "repo-1", "source_actions": []}
    )["session_id"]
    original_session = server._session
    stale_reference_ready = threading.Event()
    release_stale_reference = threading.Event()
    errors: list[BaseException] = []

    def intercept_session(payload: dict[str, Any]) -> Any:
        session = original_session(payload)
        if threading.current_thread().name == "stale-begin":
            stale_reference_ready.set()
            release_stale_reference.wait(timeout=2.0)
        return session

    server._session = intercept_session  # type: ignore[method-assign]

    def begin() -> None:
        try:
            server._begin(
                {
                    "session_id": session_id,
                    "tool_call_id": "late-call",
                    "command": "true",
                }
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=begin, name="stale-begin")
    thread.start()
    try:
        assert stale_reference_ready.wait(timeout=2.0)
        server._finalize({"session_id": session_id, "replay_execution": "completed"})
        release_stale_reference.set()
        thread.join(timeout=2.0)
    finally:
        release_stale_reference.set()
        server.server_close()

    assert len(errors) == 1
    assert isinstance(errors[0], SidecarProtocolError)
    assert "closed" in str(errors[0])


def test_timeout_eviction_closes_stale_concurrent_session_reference(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "sidecar.sock"
    server = ToolResourceSidecarServer(
        socket_path,
        collector_factory=_Collector,
        state_dir=tmp_path / "state",
        session_timeout_s=0.01,
    )
    session_id = server._open(
        {"container_id": "container-1", "repo": "repo-1", "source_actions": []}
    )["session_id"]
    server._sessions[session_id].last_used = 0.0
    original_session = server._session
    stale_reference_ready = threading.Event()
    release_stale_reference = threading.Event()
    errors: list[BaseException] = []

    def intercept_session(payload: dict[str, Any]) -> Any:
        session = original_session(payload)
        if threading.current_thread().name == "stale-begin":
            stale_reference_ready.set()
            release_stale_reference.wait(timeout=2.0)
        return session

    server._session = intercept_session  # type: ignore[method-assign]

    def begin() -> None:
        try:
            server._begin(
                {
                    "session_id": session_id,
                    "tool_call_id": "late-call",
                    "command": "true",
                }
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=begin, name="stale-begin")
    thread.start()
    try:
        assert stale_reference_ready.wait(timeout=2.0)
        server.service_actions()
        release_stale_reference.set()
        thread.join(timeout=2.0)
    finally:
        release_stale_reference.set()
        server.server_close()

    assert len(errors) == 1
    assert isinstance(errors[0], SidecarProtocolError)
    assert "closed" in str(errors[0])


class _SlowHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        time.sleep(0.2)


def test_transport_timeout_is_reported_as_unavailable(tmp_path: Path) -> None:
    socket_path = tmp_path / "slow.sock"
    server = socketserver.UnixStreamServer(str(socket_path), _SlowHandler)
    thread = _serve(server)
    try:
        with pytest.raises(SidecarUnavailableError):
            UnixSocketTransport(socket_path, timeout_s=0.02).ping()
    finally:
        server.shutdown()
        server.server_close()
        socket_path.unlink(missing_ok=True)
        thread.join(timeout=2.0)


def test_sidecar_module_runs_as_real_subprocess() -> None:
    sidecar = _start_tool_resource_sidecar("docker", timeout_s=5.0)
    temporary_directory = Path(sidecar.temporary_directory.name)
    try:
        UnixSocketTransport(sidecar.socket_path).ping()
        assert stat.S_IMODE(sidecar.socket_path.stat().st_mode) == 0o600
    finally:
        sidecar.close()
    assert not temporary_directory.exists()
