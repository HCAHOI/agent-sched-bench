"""Vsock transport for Firecracker guest agent communication.

Host (CID 2) connects to guest (CID 3) on a configured port.  The
guest agent speaks the same JSON-lines protocol as the container-based
``ContainerAgent`` (one JSON request per line, one JSON response per
line), so the host just sends and receives serialised dicts.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

# Guest CID is always 3 in Firecracker; the host is VMADDR_CID_HOST (2).
_GUEST_CID = 3
_DEFAULT_VSOCK_PORT = 5678
_DEFAULT_CONNECT_TIMEOUT_S = 5.0


class VsockTransport:
    """Host-side vsock client for the Firecracker guest agent."""

    def __init__(
        self,
        *,
        port: int = _DEFAULT_VSOCK_PORT,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        response_timeout_s: float = 600.0,
    ) -> None:
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._response_timeout_s = response_timeout_s
        self._sock: socket.socket | None = None
        self._reader: Any = None
        self._writer: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a vsock connection to the guest agent.

        Raises ``ConnectionError`` when the agent is not listening yet.
        """
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        try:
            sock.connect((_GUEST_CID, self._port))
        except (OSError, TimeoutError) as exc:
            sock.close()
            raise ConnectionError(
                f"vsock connect to guest CID {_GUEST_CID} port {self._port} failed"
            ) from exc
        sock.settimeout(self._response_timeout_s)
        self._sock = sock
        self._reader = sock.makefile("r", buffering=1, errors="replace")
        self._writer = sock.makefile("w", buffering=1)

    def close(self) -> None:
        """Close the vsock connection (idempotent)."""
        sock = self._sock
        self._sock = None
        self._reader = None
        self._writer = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    @property
    def connected(self) -> bool:
        return self._sock is not None

    # ------------------------------------------------------------------
    # Request / response
    # ------------------------------------------------------------------

    def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON-line request and return the parsed JSON-line response.

        Per the ContainerAgent protocol the agent writes exactly one JSON
        object (followed by a newline) for every request it receives.
        """
        if self._writer is None or self._reader is None:
            raise RuntimeError("VsockTransport not connected")

        payload = json.dumps(request, ensure_ascii=False) + "\n"
        self._writer.write(payload)
        self._writer.flush()

        for _skip in range(50):
            raw = self._reader.readline()
            if not raw:
                raise ConnectionError("vsock agent closed connection")
            decoded = raw.strip()
            if decoded.startswith("{"):
                return json.loads(decoded)  # type: ignore[no-any-return]
            # Skip stray non-JSON lines (e.g. init messages).
            continue

        raise RuntimeError("vsock agent emitted no JSON response")

    def try_connect(self) -> bool:
        """Test whether the guest agent is listening (does not keep the connection)."""
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        try:
            sock.connect((_GUEST_CID, self._port))
            return True
        except (OSError, TimeoutError):
            return False
        finally:
            sock.close()


def poll_vsock_ready(
    port: int = _DEFAULT_VSOCK_PORT,
    timeout_s: float = 60.0,
    interval_s: float = 0.2,
) -> bool:
    """Block until the vsock agent accepts a connection or *timeout_s* expires.

    Returns ``True`` when the agent is ready.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(min(interval_s, 1.0))
        try:
            sock.connect((_GUEST_CID, port))
            sock.close()
            return True
        except (OSError, TimeoutError):
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass
    return False
