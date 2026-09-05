"""Strict Resource Protocol used only by unprivileged clients."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from tool_resource._uds import DEFAULT_TIMEOUT_S, UnixTransport

RESOURCE_PROTOCOL_VERSION = 3


class ResourceError(RuntimeError):
    """Resource service rejected a request or response."""


class ResourceUnavailableError(ResourceError):
    """Resource service could not complete a transport operation."""


class ResourceProtocolError(ResourceError):
    """Resource peer violated the strict protocol."""


class ResourceTransport(Protocol):
    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]: ...


#: AwaitTraceReady is a pre-workload instrumentation barrier, CloseTrace is the
#: post-workload settlement barrier, and CloseRun settles the store. Online
#: OpenTrace, BeginCall, and EndCall retain the short default timeout.
RESOURCE_OPERATION_TIMEOUTS_S = {
    "AwaitTraceReady": 300.0,
    "CloseTrace": 900.0,
    "CloseRun": 300.0,
}


class ResourceUnixTransport(UnixTransport):
    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        expected_peer_uid: int | None = None,
    ) -> None:
        super().__init__(
            socket_path,
            protocol_version=RESOURCE_PROTOCOL_VERSION,
            error_type=ResourceProtocolError,
            unavailable_type=ResourceUnavailableError,
            timeout_s=timeout_s,
            operation_timeouts_s=RESOURCE_OPERATION_TIMEOUTS_S,
            expected_peer_uids=(
                None if expected_peer_uid is None else {expected_peer_uid}
            ),
            expected_peer_socket_owner=expected_peer_uid is None,
        )

    def ping(self) -> dict[str, Any]:
        return self.request("Ping")


__all__ = [
    "RESOURCE_PROTOCOL_VERSION",
    "ResourceError",
    "ResourceProtocolError",
    "ResourceTransport",
    "ResourceUnavailableError",
    "ResourceUnixTransport",
]
