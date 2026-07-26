"""Bounded versioned Unix-socket transport for tool-resource telemetry."""

from __future__ import annotations

import json
import math
import socket
import struct
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_S = 10.0
_HEADER = struct.Struct("!I")


class SidecarError(RuntimeError):
    """Base error for unavailable or invalid sidecar interactions."""


class SidecarUnavailableError(SidecarError):
    """The sidecar could not complete a transport operation."""


class SidecarProtocolError(SidecarError):
    """The peer violated the versioned protocol."""


class SidecarTransport(Protocol):
    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def receive_message(
    connection: socket.socket,
    *,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    header = _read_exact(connection, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if size > max_message_bytes:
        raise SidecarProtocolError(
            f"sidecar message size {size} exceeds limit {max_message_bytes}"
        )
    try:
        value = json.loads(
            _read_exact(connection, size),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise SidecarProtocolError(f"sidecar returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SidecarProtocolError("sidecar message must be a JSON object")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


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
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SidecarProtocolError(f"message is not valid JSON: {exc}") from exc
    if len(payload) > max_message_bytes:
        raise SidecarProtocolError(
            f"sidecar message size {len(payload)} exceeds limit {max_message_bytes}"
        )
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise SidecarUnavailableError("sidecar disconnected mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class UnixSocketTransport:
    """One bounded request per Unix-domain-socket connection."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("sidecar timeout_s must be finite and positive")
        if (
            not isinstance(max_message_bytes, int)
            or isinstance(max_message_bytes, bool)
            or max_message_bytes <= 0
        ):
            raise ValueError("max_message_bytes must be a positive integer")
        self.socket_path = Path(socket_path)
        self.timeout_s = float(timeout_s)
        self.max_message_bytes = int(max_message_bytes)

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        request = {
            "version": PROTOCOL_VERSION,
            "id": request_id,
            "operation": operation,
            "payload": dict(payload or {}),
        }
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_s)
                connection.connect(str(self.socket_path))
                send_message(
                    connection,
                    request,
                    max_message_bytes=self.max_message_bytes,
                )
                response = receive_message(
                    connection,
                    max_message_bytes=self.max_message_bytes,
                )
        except SidecarError:
            raise
        except (OSError, TimeoutError) as exc:
            raise SidecarUnavailableError(
                f"sidecar request {operation!r} failed: {exc}"
            ) from exc
        if response.get("version") != PROTOCOL_VERSION:
            raise SidecarProtocolError(
                f"sidecar protocol version mismatch: {response.get('version')!r}"
            )
        if response.get("id") != request_id:
            raise SidecarProtocolError("sidecar response id mismatch")
        if response.get("ok") is not True:
            raise SidecarError(str(response.get("error") or "sidecar request failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise SidecarProtocolError("sidecar result must be a JSON object")
        return result

    def ping(self) -> None:
        self.request("ping")


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_MESSAGE_BYTES",
    "PROTOCOL_VERSION",
    "SidecarError",
    "SidecarProtocolError",
    "SidecarTransport",
    "SidecarUnavailableError",
    "UnixSocketTransport",
    "receive_message",
    "send_message",
]
