"""Vsock transport for Firecracker guest agent communication.

Firecracker does NOT support host-side AF_VSOCK.  Instead, it exposes a
host-side Unix socket at the UDS path configured via ``PUT /vsock``.
The host communicates by sending ``CONNECT <port>\\n`` over that UDS,
then using the resulting connection for the JSON-lines request/response
protocol (same as the container-based ``ContainerAgent``).
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

_DEFAULT_VSOCK_PORT = 5678
_DEFAULT_CONNECT_TIMEOUT_S = 5.0


class VsockTransport:
    """Host-side UDS client for the Firecracker guest vsock agent.

    Firecracker creates a host-side Unix socket at *vsock_sock_path*
    after ``PUT /vsock`` + ``InstanceStart``.  The transport connects to
    that socket and performs a ``CONNECT <port>`` handshake to reach the
    guest agent listening on *port* inside the VM.
    """

    def __init__(
        self,
        *,
        vsock_sock_path: str,
        port: int = _DEFAULT_VSOCK_PORT,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        response_timeout_s: float = 600.0,
    ) -> None:
        self._vsock_sock_path = vsock_sock_path
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
        """Open a UDS connection to the FC vsock socket and perform the
        ``CONNECT <port>`` handshake.

        Raises ``ConnectionError`` when the UDS socket does not exist or
        the guest agent is not listening yet.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        try:
            sock.connect(self._vsock_sock_path)
        except (OSError, TimeoutError) as exc:
            sock.close()
            raise ConnectionError(
                f"UDS connect to FC vsock socket {self._vsock_sock_path} failed"
            ) from exc

        try:
            # Firecracker vsock UDS handshake: CONNECT <port>\n
            handshake = f"CONNECT {self._port}\n".encode()
            sock.sendall(handshake)

            # Read the response line (OK <port> or error message).
            resp_chunks: list[bytes] = []
            while True:
                try:
                    chunk = sock.recv(1024)
                except socket.timeout:
                    break
                if not chunk:
                    break
                resp_chunks.append(chunk)
                if b"\n" in chunk:
                    break

            response = (
                b"".join(resp_chunks).decode("utf-8", errors="replace").strip()
            )
            if not response.startswith("OK"):
                sock.close()
                raise ConnectionError(
                    f"FC vsock CONNECT {self._port} handshake failed: {response}"
                )
        except (OSError, TimeoutError):
            sock.close()
            raise ConnectionError(
                f"FC vsock CONNECT {self._port} handshake I/O error"
            )

        sock.settimeout(self._response_timeout_s)
        self._sock = sock
        self._reader = sock.makefile("r", buffering=1, errors="replace")
        self._writer = sock.makefile("w", buffering=1)

    def close(self) -> None:
        """Close the UDS connection (idempotent)."""
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
        """Test whether the guest agent is listening (does not keep the
        connection)."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        try:
            sock.connect(self._vsock_sock_path)
            handshake = f"CONNECT {self._port}\n".encode()
            sock.sendall(handshake)
            resp = sock.recv(1024)
            return resp.startswith(b"OK")
        except (OSError, TimeoutError):
            return False
        finally:
            sock.close()


def poll_vsock_ready(
    *,
    vsock_sock_path: str,
    port: int = _DEFAULT_VSOCK_PORT,
    timeout_s: float = 60.0,
    interval_s: float = 0.2,
) -> bool:
    """Block until the vsock UDS socket appears and the guest agent accepts
    a ``CONNECT <port>`` handshake, or *timeout_s* expires.

    Returns ``True`` when the agent is ready.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(min(interval_s, 1.0))
        try:
            sock.connect(vsock_sock_path)
            handshake = f"CONNECT {port}\n".encode()
            sock.sendall(handshake)
            resp = sock.recv(1024)
            if resp.startswith(b"OK"):
                sock.close()
                return True
        except (OSError, TimeoutError):
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass
        time.sleep(interval_s)
    return False


def quiesce_vsock(
    *,
    vsock_sock_path: str,
    port: int = _DEFAULT_VSOCK_PORT,
    timeout_s: float = 10.0,
) -> bool:
    """Send a ``sync`` command to the guest agent to flush journals and
    page cache before a snapshot pause.

    Opens a short-lived UDS connection, performs the ``CONNECT <port>``
    handshake, sends an ``exec`` tool request for ``sync``, and reads the
    response.  Returns ``True`` on success.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(min(timeout_s, 5.0))
    try:
        sock.connect(vsock_sock_path)
    except (OSError, TimeoutError) as exc:
        sock.close()
        raise ConnectionError(
            f"quiesce UDS connect to {vsock_sock_path} failed"
        ) from exc

    try:
        handshake = f"CONNECT {port}\n".encode()
        sock.sendall(handshake)
        resp = b""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(1024)
            except socket.timeout:
                break
            if not chunk:
                break
            resp += chunk
            if b"\n" in chunk:
                break
        if not resp.startswith(b"OK"):
            sock.close()
            raise ConnectionError(
                f"quiesce CONNECT {port} handshake failed: "
                f"{resp.decode('utf-8', errors='replace')!r}"
            )
    except (OSError, TimeoutError) as exc:
        sock.close()
        raise ConnectionError(
            f"quiesce CONNECT {port} handshake I/O error"
        ) from exc

    sock.settimeout(timeout_s)
    try:
        request = json.dumps(
            {"tool": "exec", "args": {"command": "sync"}},
            ensure_ascii=False,
        ) + "\n"
        sock.sendall(request.encode())

        chunks: list[bytes] = []
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        sock.close()

    for i in range(len(raw)):
        if raw[i] == "{":
            raw = raw[i:]
            break

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                response = json.loads(stripped)
                return bool(response.get("ok", False))
            except json.JSONDecodeError:
                pass
    return False
