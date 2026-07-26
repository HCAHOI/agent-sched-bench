"""Privileged clause telemetry sidecar over a Unix-domain socket."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socketserver
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_resource.sidecar_protocol import (
    DEFAULT_TIMEOUT_S,
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    SidecarProtocolError,
    receive_message,
    send_message,
)

CollectorFactory = Callable[..., Any]
DEFAULT_SESSION_TIMEOUT_S = 900.0


def _collector_factory(**kwargs: Any) -> Any:
    from tool_resource.telemetry import ClauseTelemetryCollector

    try:
        return ClauseTelemetryCollector(**kwargs)
    except Exception as exc:  # noqa: BLE001 - attach failures become unavailable
        return ClauseTelemetryCollector.unavailable(
            container_id=str(kwargs["container_id"]),
            repo=str(kwargs["repo"]),
            artifact_path=Path(kwargs["artifact_path"]),
            source_actions=kwargs["source_actions"],
            reason=f"collector attach failed: {type(exc).__name__}: {exc}",
        )


@dataclass
class _Session:
    collector: Any
    artifact_path: Path
    tokens: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)
    closed: bool = False


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, ToolResourceSidecarServer)
        self.request.settimeout(server.request_timeout_s)
        request_id: object = None
        try:
            request = receive_message(
                self.request,
                max_message_bytes=server.max_message_bytes,
            )
            request_id = request.get("id")
            result = server.dispatch(request)
            response = {
                "version": PROTOCOL_VERSION,
                "id": request_id,
                "ok": True,
                "result": result,
            }
        except Exception as exc:  # noqa: BLE001 - protocol boundary returns errors
            response = {
                "version": PROTOCOL_VERSION,
                "id": request_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            send_message(
                self.request,
                response,
                max_message_bytes=server.max_message_bytes,
            )
        except (OSError, TimeoutError):
            pass


class ToolResourceSidecarServer(
    socketserver.ThreadingMixIn,
    socketserver.UnixStreamServer,
):
    """Own collector sessions; protocol responses contain summaries only."""

    daemon_threads = False

    def __init__(
        self,
        socket_path: str | Path,
        *,
        container_executable: str = "docker",
        collector_factory: CollectorFactory = _collector_factory,
        request_timeout_s: float = DEFAULT_TIMEOUT_S,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        socket_mode: int = 0o600,
        socket_gid: int | None = None,
        state_dir: str | Path | None = None,
        parent_pid: int | None = None,
    ) -> None:
        if (
            not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
            or not math.isfinite(session_timeout_s)
            or session_timeout_s <= 0
        ):
            raise ValueError("sidecar timeouts must be finite and positive")
        if not container_executable:
            raise ValueError("container_executable is required")
        if (
            not isinstance(max_message_bytes, int)
            or isinstance(max_message_bytes, bool)
            or max_message_bytes <= 0
        ):
            raise ValueError("max_message_bytes must be a positive integer")
        if socket_mode < 0 or socket_mode & ~0o660:
            raise ValueError("socket_mode may grant only user/group rw")
        if socket_gid is not None and socket_gid < 0:
            raise ValueError("socket_gid must be non-negative")
        if parent_pid is not None and parent_pid <= 0:
            raise ValueError("parent_pid must be positive")
        self.socket_path = Path(socket_path)
        self.container_executable = container_executable
        self.collector_factory = collector_factory
        self.request_timeout_s = float(request_timeout_s)
        self.session_timeout_s = float(session_timeout_s)
        self.max_message_bytes = int(max_message_bytes)
        self.parent_pid = parent_pid
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        self._state_tmp = (
            tempfile.TemporaryDirectory(prefix="tool-resource-sidecar-")
            if state_dir is None
            else None
        )
        self.state_dir = Path(
            self._state_tmp.name if self._state_tmp is not None else state_dir
        )
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.socket_path.exists():
            raise FileExistsError(f"sidecar socket already exists: {self.socket_path}")
        previous_umask = os.umask(0o177)
        try:
            super().__init__(str(self.socket_path), _RequestHandler)
        finally:
            os.umask(previous_umask)
        try:
            if socket_gid is not None:
                os.chown(self.socket_path, -1, socket_gid)
            os.chmod(self.socket_path, socket_mode)
        except Exception:
            super().server_close()
            self.socket_path.unlink(missing_ok=True)
            if self._state_tmp is not None:
                self._state_tmp.cleanup()
            raise

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("version") != PROTOCOL_VERSION:
            raise SidecarProtocolError(
                f"unsupported protocol version {request.get('version')!r}"
            )
        if not isinstance(request.get("id"), str):
            raise SidecarProtocolError("request id must be a string")
        operation = request.get("operation")
        payload = request.get("payload")
        if not isinstance(operation, str) or not isinstance(payload, Mapping):
            raise SidecarProtocolError("operation and payload are required")
        if operation == "ping":
            return {"protocol_version": PROTOCOL_VERSION}
        if operation == "open":
            return self._open(payload)
        if operation == "begin":
            return self._begin(payload)
        if operation == "finish":
            return self._finish(payload)
        if operation == "safety_guard":
            return self._safety_guard(payload)
        if operation == "add_error":
            return self._add_error(payload)
        if operation == "finalize":
            return self._finalize(payload)
        raise SidecarProtocolError(f"unsupported operation {operation!r}")

    def _open(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        container_id = _required_string(payload, "container_id")
        repo = _required_string(payload, "repo")
        raw_actions = payload.get("source_actions", [])
        if not isinstance(raw_actions, list) or not all(
            isinstance(action, Mapping) for action in raw_actions
        ):
            raise SidecarProtocolError("source_actions must be a list of objects")
        session_id = uuid.uuid4().hex
        artifact_path = self.state_dir / f"{session_id}.json"
        collector = self.collector_factory(
            container_id=container_id,
            container_executable=self.container_executable,
            repo=repo,
            artifact_path=artifact_path,
            source_actions=list(raw_actions),
        )
        with self._sessions_lock:
            self._sessions[session_id] = _Session(collector, artifact_path)
        return {"session_id": session_id}

    def _begin(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        with session.lock:
            _require_open_session(session)
            token = session.collector.begin_tool_call(
                _required_string(payload, "tool_call_id"),
                _required_string(payload, "command", allow_empty=True),
            )
            token_id = uuid.uuid4().hex
            session.tokens[token_id] = token
            session.last_used = time.monotonic()
        return {"token_id": token_id}

    def _finish(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        token_id = _required_string(payload, "token_id")
        response = payload.get("replay_response")
        if response is not None and not isinstance(response, Mapping):
            raise SidecarProtocolError("replay_response must be an object or null")
        with session.lock:
            _require_open_session(session)
            try:
                token = session.tokens.pop(token_id)
            except KeyError as exc:
                raise SidecarProtocolError("unknown sidecar token") from exc
            call = session.collector.finish_tool_call(
                token,
                replay_response=response,
            )
            session.last_used = time.monotonic()
        return {"call": _mapping(call, "finalized call")}

    def _safety_guard(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        with session.lock:
            _require_open_session(session)
            call = session.collector.record_safety_guard_blocked(
                _required_string(payload, "tool_call_id", allow_empty=True),
                _required_string(payload, "command", allow_empty=True),
                _required_string(payload, "replay_result", allow_empty=True),
            )
            session.last_used = time.monotonic()
        return {"call": _mapping(call, "finalized call")}

    def _add_error(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        with session.lock:
            _require_open_session(session)
            session.collector.add_integrity_error(_required_string(payload, "message"))
            session.last_used = time.monotonic()
        return {}

    def _finalize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _required_string(payload, "session_id")
        replay_execution = _required_string(payload, "replay_execution")
        if replay_execution not in {"completed", "failed", "incomplete"}:
            raise SidecarProtocolError("invalid replay_execution")
        session = self._session(payload)
        with session.lock:
            _require_open_session(session)
            with self._sessions_lock:
                if self._sessions.get(session_id) is not session:
                    raise SidecarProtocolError("unknown sidecar session")
                del self._sessions[session_id]
                session.closed = True
            try:
                session.collector.finalize(replay_execution=replay_execution)
                artifact = json.loads(session.artifact_path.read_text(encoding="utf-8"))
            finally:
                session.artifact_path.unlink(missing_ok=True)
        return {"artifact": _mapping(artifact, "finalized artifact")}

    def _session(self, payload: Mapping[str, Any]) -> _Session:
        session_id = _required_string(payload, "session_id")
        with self._sessions_lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise SidecarProtocolError("unknown sidecar session") from exc

    def service_actions(self) -> None:
        if self.parent_pid is not None and os.getppid() != self.parent_pid:
            self._BaseServer__shutdown_request = True
        cutoff = time.monotonic() - self.session_timeout_s
        with self._sessions_lock:
            stale_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.last_used < cutoff
            ]
        for session_id in stale_ids:
            self._close_registered_session(session_id, stale_before=cutoff)

    def server_close(self) -> None:
        super().server_close()
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._close_registered_session(session_id)
        self.socket_path.unlink(missing_ok=True)
        if self._state_tmp is not None:
            self._state_tmp.cleanup()

    def _close_registered_session(
        self,
        session_id: str,
        *,
        stale_before: float | None = None,
    ) -> None:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            return
        with session.lock:
            with self._sessions_lock:
                if self._sessions.get(session_id) is not session:
                    return
                if (
                    stale_before is not None
                    and session.last_used >= stale_before
                ):
                    return
                del self._sessions[session_id]
                session.closed = True
            with suppress(Exception):
                session.collector.finalize(replay_execution="incomplete")
            session.artifact_path.unlink(missing_ok=True)


def _required_string(
    payload: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise SidecarProtocolError(f"{name} must be a string")
    return value


def _require_open_session(session: _Session) -> None:
    if session.closed:
        raise SidecarProtocolError("sidecar session is closed")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SidecarProtocolError(f"{name} must be an object")
    return dict(value)


def _parse_mode(value: str) -> int:
    mode = int(value, 8)
    if mode & ~0o660:
        raise argparse.ArgumentTypeError("socket mode may grant only user/group rw")
    return mode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--container-executable", default="docker")
    parser.add_argument("--socket-mode", type=_parse_mode, default=0o600)
    parser.add_argument("--socket-gid", type=int)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=DEFAULT_SESSION_TIMEOUT_S,
    )
    args = parser.parse_args(argv)
    with ToolResourceSidecarServer(
        args.socket,
        container_executable=args.container_executable,
        socket_mode=args.socket_mode,
        socket_gid=args.socket_gid,
        session_timeout_s=args.session_timeout,
        state_dir=args.state_dir,
        parent_pid=args.parent_pid,
    ) as server:
        previous_sigterm = signal.signal(
            signal.SIGTERM,
            lambda *_args: setattr(server, "_BaseServer__shutdown_request", True),
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SESSION_TIMEOUT_S",
    "ToolResourceSidecarServer",
    "main",
]
