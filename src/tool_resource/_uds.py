"""Shared bounded Unix-socket framing for the two tool-resource protocols."""

from __future__ import annotations

import json
import math
import os
import socket
import socketserver
import struct
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Set
from pathlib import Path
from typing import Any

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_S = 10.0
_HEADER = struct.Struct("!I")
_PEERCRED = struct.Struct("3i")
_ENVELOPE_KEYS = frozenset({"protocol_version", "request_id", "operation", "payload"})


class WireError(RuntimeError):
    """Invalid or unavailable bounded UDS interaction."""


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise WireError("peer disconnected mid-message")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def receive_message(
    connection: socket.socket,
    *,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    (size,) = _HEADER.unpack(_read_exact(connection, _HEADER.size))
    if size > max_message_bytes:
        raise WireError(f"message size {size} exceeds limit {max_message_bytes}")
    try:
        value = json.loads(
            _read_exact(connection, size),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise WireError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WireError("message must be a JSON object")
    return value


def send_message(
    connection: socket.socket,
    message: Mapping[str, Any],
    *,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
) -> None:
    try:
        payload = json.dumps(
            dict(message),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as exc:
        raise WireError(f"message is not valid JSON: {exc}") from exc
    if len(payload) > max_message_bytes:
        raise WireError(
            f"message size {len(payload)} exceeds limit {max_message_bytes}"
        )
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    return _PEERCRED.unpack(
        connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
    )


def validate_envelope(
    message: Mapping[str, Any],
    *,
    protocol_version: int,
    error_type: type[RuntimeError],
) -> tuple[str, str, Mapping[str, Any]]:
    if set(message) != _ENVELOPE_KEYS:
        raise error_type("message envelope fields are invalid")
    if message["protocol_version"] != protocol_version:
        raise error_type(
            f"unsupported protocol version {message['protocol_version']!r}"
        )
    request_id = message["request_id"]
    operation = message["operation"]
    payload = message["payload"]
    if not isinstance(request_id, str) or not request_id:
        raise error_type("request_id must be a non-empty string")
    if not isinstance(operation, str) or not operation:
        raise error_type("operation must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise error_type("payload must be an object")
    return request_id, operation, payload


def require_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: Set[str] = frozenset(),
    error_type: type[RuntimeError],
) -> None:
    missing = required - set(payload)
    unknown = set(payload) - required - optional
    if missing:
        raise error_type(f"missing payload fields: {', '.join(sorted(missing))}")
    if unknown:
        raise error_type(f"unknown payload fields: {', '.join(sorted(unknown))}")


class UnixTransport:
    """One strict request per UDS connection."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        protocol_version: int,
        error_type: type[RuntimeError],
        unavailable_type: type[RuntimeError],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        expected_peer_uids: set[int] | None = None,
        expected_peer_socket_owner: bool = False,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        self.socket_path = Path(socket_path)
        self.protocol_version = protocol_version
        self.error_type = error_type
        self.unavailable_type = unavailable_type
        self.timeout_s = float(timeout_s)
        self.max_message_bytes = max_message_bytes
        self.expected_peer_uids = expected_peer_uids
        self.expected_peer_socket_owner = expected_peer_socket_owner

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or uuid.uuid4().hex
        request = {
            "protocol_version": self.protocol_version,
            "request_id": request_id,
            "operation": operation,
            "payload": dict(payload or {}),
        }
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_s)
                connection.connect(str(self.socket_path))
                expected_peer_uids = self.expected_peer_uids
                if self.expected_peer_socket_owner:
                    expected_peer_uids = {self.socket_path.stat().st_uid}
                if expected_peer_uids is not None:
                    _pid, uid, _gid = peer_credentials(connection)
                    if uid not in expected_peer_uids:
                        raise self.error_type(f"unexpected peer uid {uid}")
                send_message(
                    connection,
                    request,
                    max_message_bytes=self.max_message_bytes,
                )
                response = receive_message(
                    connection,
                    max_message_bytes=self.max_message_bytes,
                )
        except self.error_type:
            raise
        except (OSError, TimeoutError, WireError) as exc:
            raise self.unavailable_type(f"{operation} failed: {exc}") from exc
        response_id, response_operation, response_payload = validate_envelope(
            response,
            protocol_version=self.protocol_version,
            error_type=self.error_type,
        )
        if response_id != request_id:
            raise self.error_type("response request_id mismatch")
        if response_operation != operation:
            raise self.error_type("response operation mismatch")
        if set(response_payload) == {"ok", "result"}:
            if response_payload["ok"] is not True or not isinstance(
                response_payload["result"], Mapping
            ):
                raise self.error_type("invalid success payload")
            return dict(response_payload["result"])
        if set(response_payload) == {"ok", "error"}:
            if response_payload["ok"] is not False or not isinstance(
                response_payload["error"], str
            ):
                raise self.error_type("invalid error payload")
            raise self.error_type(response_payload["error"])
        raise self.error_type("invalid response payload fields")


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, StrictUnixServer)
        self.request.settimeout(server.request_timeout_s)
        request_id = ""
        operation = ""
        try:
            _pid, peer_uid, _gid = peer_credentials(self.request)
            if peer_uid not in server.allowed_uids:
                raise server.protocol_error(f"peer uid {peer_uid} is not allowed")
            request = receive_message(
                self.request,
                max_message_bytes=server.max_message_bytes,
            )
            request_id, operation, _payload = validate_envelope(
                request,
                protocol_version=server.protocol_version,
                error_type=server.protocol_error,
            )
            response = server.handle_request_message(peer_uid, request)
        except Exception as exc:  # noqa: BLE001 - protocol boundary
            response = {
                "protocol_version": server.protocol_version,
                "request_id": request_id,
                "operation": operation,
                "payload": {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }
        try:
            send_message(
                self.request,
                response,
                max_message_bytes=server.max_message_bytes,
            )
        except (OSError, TimeoutError, WireError):
            pass


class StrictUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Strict peer-checked server with bounded idempotency cache."""

    daemon_threads = True

    def __init__(
        self,
        socket_path: str | Path,
        *,
        protocol_version: int,
        protocol_error: type[RuntimeError],
        dispatch: Callable[
            [str, Mapping[str, Any], tuple[int, str]], dict[str, Any]
        ],
        allowed_uids: set[int],
        request_timeout_s: float = DEFAULT_TIMEOUT_S,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        socket_mode: int = 0o600,
        socket_gid: int | None = None,
        socket_dir_mode: int = 0o700,
        socket_dir_gid: int | None = None,
        response_cache_size: int = 1024,
    ) -> None:
        if not allowed_uids or any(uid < 0 for uid in allowed_uids):
            raise ValueError("allowed_uids must contain non-negative UIDs")
        if socket_mode < 0 or socket_mode & ~0o660:
            raise ValueError("socket_mode may grant only user/group rw")
        if socket_dir_mode < 0 or socket_dir_mode & ~0o770:
            raise ValueError("socket_dir_mode may grant only user/group rwx")
        if response_cache_size <= 0:
            raise ValueError("response_cache_size must be positive")
        self.socket_path = Path(socket_path)
        self.protocol_version = protocol_version
        self.protocol_error = protocol_error
        self.dispatch = dispatch
        self.allowed_uids = frozenset(allowed_uids)
        self.request_timeout_s = request_timeout_s
        self.max_message_bytes = max_message_bytes
        self.response_cache_size = response_cache_size
        self._responses: OrderedDict[
            tuple[int, str], tuple[str, str, dict[str, Any]]
        ] = OrderedDict()
        # ponytail: serialize requests until service throughput warrants
        # per-request in-flight coordination.
        self._request_lock = threading.Lock()
        socket_dir_missing = not self.socket_path.parent.exists()
        self.socket_path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=socket_dir_mode,
        )
        if socket_dir_missing:
            if socket_dir_gid is not None:
                os.chown(self.socket_path.parent, -1, socket_dir_gid)
            os.chmod(self.socket_path.parent, socket_dir_mode)
        if self.socket_path.exists():
            raise FileExistsError(f"socket already exists: {self.socket_path}")
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
            raise

    def handle_request_message(
        self,
        peer_uid: int,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id, operation, payload = validate_envelope(
            request,
            protocol_version=self.protocol_version,
            error_type=self.protocol_error,
        )
        key = (peer_uid, request_id)
        payload_identity = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._request_lock:
            cached = self._responses.get(key)
            if cached is not None:
                cached_operation, cached_payload, response = cached
                if cached_operation != operation or cached_payload != payload_identity:
                    raise self.protocol_error(
                        "request_id was already used for a different request"
                    )
                self._responses.move_to_end(key)
                return response
            try:
                result = self.dispatch(operation, payload, key)
                response_payload: dict[str, Any] = {"ok": True, "result": result}
            except Exception as exc:  # noqa: BLE001 - protocol error data
                response_payload = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            response = {
                "protocol_version": self.protocol_version,
                "request_id": request_id,
                "operation": operation,
                "payload": response_payload,
            }
            self._responses[key] = (operation, payload_identity, response)
            self._responses.move_to_end(key)
            while len(self._responses) > self.response_cache_size:
                self._responses.popitem(last=False)
        return response

    def server_close(self) -> None:
        super().server_close()
        self.socket_path.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_MESSAGE_BYTES",
    "StrictUnixServer",
    "UnixTransport",
    "WireError",
    "receive_message",
    "require_fields",
    "send_message",
    "validate_envelope",
]
