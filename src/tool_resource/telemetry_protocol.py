"""Strict Telemetry Protocol used only between the two services."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from tool_resource._uds import DEFAULT_TIMEOUT_S, UnixTransport

TELEMETRY_PROTOCOL_VERSION = 2


class TelemetryError(RuntimeError):
    """Telemetry service rejected a request or response."""


class TelemetryUnavailableError(TelemetryError):
    """Telemetry service could not complete a transport operation."""


class TelemetryProtocolError(TelemetryError):
    """Telemetry peer violated the strict protocol."""


class TelemetryTransport(Protocol):
    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]: ...


#: AttachTarget loads and verifies the eBPF program for a new target;
#: FinalizeSession tears it down and serializes the whole trace artifact. Both
#: routinely outlast a timeout sized for cheap RPCs, and timing them out
#: discards evidence the collector gathered correctly.
TELEMETRY_OPERATION_TIMEOUTS_S = {
    "AttachTarget": 120.0,
    "FinalizeSession": 120.0,
}


class TelemetryUnixTransport(UnixTransport):
    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        expected_peer_uid: int = 0,
    ) -> None:
        super().__init__(
            socket_path,
            protocol_version=TELEMETRY_PROTOCOL_VERSION,
            error_type=TelemetryProtocolError,
            unavailable_type=TelemetryUnavailableError,
            timeout_s=timeout_s,
            operation_timeouts_s=TELEMETRY_OPERATION_TIMEOUTS_S,
            expected_peer_uids={expected_peer_uid},
        )

    def ping(self) -> dict[str, Any]:
        return self.request("Ping")


__all__ = [
    "TELEMETRY_PROTOCOL_VERSION",
    "TelemetryError",
    "TelemetryProtocolError",
    "TelemetryTransport",
    "TelemetryUnavailableError",
    "TelemetryUnixTransport",
]
