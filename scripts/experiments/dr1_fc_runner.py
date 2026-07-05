#!/usr/bin/env python3
"""DR1: Memory-diff cost reality check — Stage 2 (Firecracker VM runner).

Stage 2 replays a collected trace inside a Firecracker microVM and measures
actual dirty-memory bytes per turn via KVM dirty-page tracking + VM snapshots.
File-CAS delta bytes are computed from the trace's checkpoint manifests (reusing
Stage 1 logic).  The output CSV compares estimated vs. measured costs.

Usage::

    # On the KVM host:
    ssh ubuntu@51.158.203.248 \\
      "cd ~/workspace/agent-sched-bench && source .venv/bin/activate && \\
       PYTHONPATH=src python scripts/experiments/dr1_fc_runner.py \\
       --trace-jsonl path/to/trace.jsonl"

    # Dry-run: check environment without launching a VM
    PYTHONPATH=src python scripts/experiments/dr1_fc_runner.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Firecracker binary, API socket, and kernel/rootfs locations.
_FC_BINARY = "firecracker"
_FC_API_SOCK = "/tmp/fc-dr1.sock"
_FC_CACHE_DIR = Path("/tmp/fc-cache")
_FC_KERNEL_URL = (
    "https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.11/"
    "x86_64/vmlinux-5.10.225"
)
_FC_KERNEL_PATH = _FC_CACHE_DIR / "vmlinux-5.10.225"
_FC_ROOTFS_PATH = _FC_CACHE_DIR / "rootfs.ext4"
_FC_ROOTFS_SIZE_MB = 2048  # 2 GiB rootfs for command execution (generous)

# VM configuration.
_FC_VCPU_COUNT = 2
_FC_MEM_SIZE_MIB = 1024
_FC_GUEST_IP = "172.16.0.2"
_FC_HOST_IP = "172.16.0.1"
_FC_TAP_DEV = "fc-tap0"
_FC_NETMASK = 24
_FC_SSH_USER = "root"
_FC_SSH_PASSWORD = "root"  # ephemeral VM, hardcoded is fine

# Timing (seconds).
_VM_BOOT_TIMEOUT_S = 30
_VM_SHUTDOWN_TIMEOUT_S = 10
_SNAPSHOT_TIMEOUT_S = 30
_COMMAND_TIMEOUT_S = 60
_SSH_CONNECT_TIMEOUT_S = 5

# Stage 1 estimate constants (mirrored from dr1_memory_diff_cost.py).
_ESTIMATE_DIRTY_OVERHEAD_RATIO = 1.15
_DIRTY_PAGE_WRITE_MB_PER_S = 80.0  # MB/s
_SNAPSHOT_FIXED_COST_MS = 8.0

# CAS manifest skip dirs (mirrored).
_SKIP_DIRS = frozenset({".git"})

# CSV output columns.
CSV_FIELDNAMES = [
    "turn_number",
    "file_cas_delta_bytes",
    "dirty_memory_bytes_est",
    "dirty_memory_bytes_meas",
    "snapshot_pause_ms_est",
    "snapshot_pause_ms_meas",
    "wall_time_ms",
    "tool_exec_count",
    "error",
]


# ---------------------------------------------------------------------------
# Environment / setup
# ---------------------------------------------------------------------------


def check_environment() -> list[str]:
    """Verify that the host is ready for Firecracker VM execution.

    Returns a list of human-readable issues; empty means ready.
    """
    issues: list[str] = []

    kvm = Path("/dev/kvm")
    if not kvm.exists():
        issues.append("/dev/kvm not found — KVM support required")
    elif not os.access(str(kvm), os.R_OK | os.W_OK):
        issues.append("/dev/kvm exists but is not readable/writable")

    if not shutil.which(_FC_BINARY):
        issues.append(f"'{_FC_BINARY}' not found on PATH")

    for tool in ("ssh", "ssh-keygen", "sshpass", "mkfs.ext4", "docker"):
        if not shutil.which(tool):
            issues.append(f"'{tool}' not found on PATH")

    result = subprocess.run(
        ["lsmod"], capture_output=True, text=True
    )
    if "kvm" not in result.stdout.lower():
        issues.append("kvm kernel module not loaded")

    # Verify the current user can sudo (needed for TAP / mount / iptables).
    rc = subprocess.run(
        ["sudo", "-n", "true"], capture_output=True
    ).returncode
    if rc != 0:
        issues.append("passwordless sudo not available (needed for TAP/networking)")

    return issues


def ensure_fc_cache() -> tuple[Path, Path]:
    """Download kernel and build rootfs if not already cached.

    Returns ``(kernel_path, rootfs_path)``.
    """
    _FC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Kernel download.
    if not _FC_KERNEL_PATH.exists():
        print(f"Downloading kernel from {_FC_KERNEL_URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(_FC_KERNEL_URL, _FC_KERNEL_PATH)
        _FC_KERNEL_PATH.chmod(0o755)
        print(f"Kernel cached at {_FC_KERNEL_PATH}", file=sys.stderr)

    # Rootfs build from Docker Alpine.
    if not _FC_ROOTFS_PATH.exists():
        print("Building rootfs from alpine:latest via Docker export ...", file=sys.stderr)
        _build_rootfs(_FC_ROOTFS_PATH)
        print(f"Rootfs cached at {_FC_ROOTFS_PATH}", file=sys.stderr)

    return _FC_KERNEL_PATH, _FC_ROOTFS_PATH


def _build_rootfs(output_path: Path) -> None:
    """Create an ext4 rootfs image from an Alpine Docker container.

    The image includes an SSH server, basic shell tools, and Python so that
    tool-exec commands (shell scripts, pip installs, etc.) can run inside the
    microVM.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="fc-rootfs-"))
    try:
        dockerfile = tmpdir / "Dockerfile"
        dockerfile.write_text(
            """FROM alpine:latest
RUN apk add --no-cache openssh bash coreutils util-linux \\
    python3 py3-pip curl wget git openssl \\
    && ssh-keygen -A \\
    && echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config \\
    && echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config \\
    && echo 'root:root' | chpasswd \\
    && mkdir -p /run/sshd /etc/init.d
# Write a minimal init that starts sshd then waits forever.
RUN printf '#!/bin/sh\\n/sbin/syslogd\\n/usr/sbin/sshd\\nsleep 2147483647\\n' \\
    > /init && chmod 755 /init
CMD ["/init"]
"""
        )
        tag = "fc-dr1-rootfs-builder"
        subprocess.run(
            ["docker", "build", "-t", tag, str(tmpdir)],
            check=True,
            capture_output=True,
        )

        # Export container filesystem as a tar stream.
        cid = subprocess.run(
            ["docker", "create", tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tar_path = tmpdir / "rootfs.tar"
        with open(tar_path, "wb") as fh:
            subprocess.run(
                ["docker", "export", cid],
                check=True,
                stdout=fh,
            )
        subprocess.run(["docker", "rm", cid], check=True, capture_output=True)

        # Create a blank ext4 image, mount it, and extract the tar.
        subprocess.run(
            [
                "dd", "if=/dev/zero", f"of={output_path}",
                "bs=1M", f"count={_FC_ROOTFS_SIZE_MB}",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["mkfs.ext4", "-F", str(output_path)],
            check=True,
            capture_output=True,
        )

        mnt = tmpdir / "mnt"
        mnt.mkdir()
        subprocess.run(
            ["sudo", "mount", "-o", "loop", str(output_path), str(mnt)],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(
                ["sudo", "tar", "-xf", str(tar_path), "-C", str(mnt)],
                check=True,
                capture_output=True,
            )
        finally:
            subprocess.run(
                ["sudo", "umount", str(mnt)],
                check=False,
                capture_output=True,
            )

        # Clean up Docker image.
        subprocess.run(
            ["docker", "rmi", tag],
            check=False,
            capture_output=True,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _setup_tap_device() -> None:
    """Create and configure the TAP device with NAT for the guest VM."""
    # Create TAP device.
    subprocess.run(
        ["sudo", "ip", "tuntap", "add", _FC_TAP_DEV, "mode", "tap"],
        capture_output=True,
    )
    subprocess.run(
        [
            "sudo", "ip", "addr", "add",
            f"{_FC_HOST_IP}/{_FC_NETMASK}", "dev", _FC_TAP_DEV,
        ],
        capture_output=True,
    )
    subprocess.run(
        ["sudo", "ip", "link", "set", _FC_TAP_DEV, "up"],
        capture_output=True,
    )

    # Enable NAT and forwarding (idempotent — may already be enabled).
    subprocess.run(
        ["sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"],
        capture_output=True,
    )
    # Delete any pre-existing rule to avoid duplicates, then re-add.
    subprocess.run(
        [
            "sudo", "iptables", "-t", "nat", "-D", "POSTROUTING",
            "-s", f"{_FC_GUEST_IP}/32", "-j", "MASQUERADE",
        ],
        capture_output=True,
    )
    subprocess.run(
        [
            "sudo", "iptables", "-t", "nat", "-A", "POSTROUTING",
            "-s", f"{_FC_GUEST_IP}/32", "-j", "MASQUERADE",
        ],
        capture_output=True,
    )


def _teardown_tap_device() -> None:
    """Remove the TAP device and NAT rules."""
    subprocess.run(
        [
            "sudo", "iptables", "-t", "nat", "-D", "POSTROUTING",
            "-s", f"{_FC_GUEST_IP}/32", "-j", "MASQUERADE",
        ],
        capture_output=True,
    )
    subprocess.run(
        ["sudo", "ip", "link", "delete", _FC_TAP_DEV],
        capture_output=True,
    )


def _generate_ephemeral_ssh_key() -> Path:
    """Return path to an ephemeral SSH key, generating it if needed."""
    key_path = Path(tempfile.gettempdir()) / "fc-dr1-id_rsa"
    if not key_path.exists():
        subprocess.run(
            [
                "ssh-keygen", "-t", "rsa", "-b", "2048",
                "-f", str(key_path), "-N", "", "-q",
            ],
            check=True,
            capture_output=True,
        )
    return key_path


# ---------------------------------------------------------------------------
# Minimal HTTP-over-Unix-socket client for the Firecracker API
# ---------------------------------------------------------------------------


class _FcApi:
    """Minimal Firecracker API client over a Unix-domain socket."""

    def __init__(self, sock_path: str) -> None:
        self._sock_path = sock_path
        self._bufsize = 65536

    def _request(self, method: str, path: str, body: str | None = None) -> tuple[int, str]:
        """Send an HTTP request, return ``(status_code, response_body)``."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(self._sock_path)
        try:
            headers = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            if body is not None:
                body_bytes = body.encode("utf-8")
                headers += (
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_bytes)}\r\n"
                )
            else:
                body_bytes = b""
            headers += "Connection: close\r\n\r\n"
            sock.sendall(headers.encode("utf-8") + body_bytes)

            # Read response.
            chunks: list[bytes] = []
            while True:
                try:
                    chunk = sock.recv(self._bufsize)
                except socket.timeout:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks).decode("utf-8", errors="replace")
        finally:
            sock.close()

        # Parse status line.
        parts = response.split("\r\n\r\n", 1)
        header_section = parts[0]
        resp_body = parts[1] if len(parts) > 1 else ""
        status_line = header_section.split("\r\n")[0]
        try:
            status_code = int(status_line.split(" ")[1])
        except (IndexError, ValueError):
            status_code = 0
        return status_code, resp_body

    def put(self, path: str, data: dict[str, Any]) -> None:
        """PUT *path* with JSON body.  Raises RuntimeError on non-2xx."""
        body = json.dumps(data)
        status, resp = self._request("PUT", path, body)
        if status not in (200, 204):
            raise RuntimeError(
                f"FC API PUT {path} returned {status}: {resp[:200]}"
            )

    def patch(self, path: str, data: dict[str, Any]) -> None:
        """PATCH *path* with JSON body.  Raises RuntimeError on non-2xx."""
        body = json.dumps(data)
        status, resp = self._request("PATCH", path, body)
        if status not in (200, 204):
            raise RuntimeError(
                f"FC API PATCH {path} returned {status}: {resp[:200]}"
            )

    def get(self, path: str) -> dict[str, Any]:
        """GET *path*, parse JSON response."""
        status, body = self._request("GET", path)
        if status != 200:
            raise RuntimeError(
                f"FC API GET {path} returned {status}: {body[:200]}"
            )
        if not body.strip():
            return {}
        return json.loads(body)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Firecracker VM management
# ---------------------------------------------------------------------------


class FirecrackerVM:
    """Manage a Firecracker microVM lifecycle.

    On ``start()``, configures the VM via the FC API socket, boots it, and
    waits for SSH readiness.  ``stop()`` tears everything down.
    """

    def __init__(
        self,
        kernel_path: Path,
        rootfs_path: Path,
        api_sock: str = _FC_API_SOCK,
        vcpu: int = _FC_VCPU_COUNT,
        mem_mib: int = _FC_MEM_SIZE_MIB,
    ) -> None:
        self._kernel = kernel_path
        self._rootfs = rootfs_path
        self._api_sock = api_sock
        self._vcpu = vcpu
        self._mem_mib = mem_mib
        self._process: subprocess.Popen[str] | None = None
        self._api: _FcApi | None = None
        self._snap_dir: Path | None = None
        self._full_snapshot_mem: Path | None = None
        self._full_snapshot_state: Path | None = None

    # -- VM lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Boot the Firecracker VM and wait for SSH."""
        _setup_tap_device()

        # Remove stale socket.
        if os.path.exists(self._api_sock):
            os.unlink(self._api_sock)

        # Launch firecracker.
        self._process = subprocess.Popen(
            [_FC_BINARY, "--api-sock", self._api_sock],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # Wait for the API socket to appear.
        deadline = time.monotonic() + _VM_BOOT_TIMEOUT_S
        while not os.path.exists(self._api_sock):
            if time.monotonic() > deadline:
                raise TimeoutError("Firecracker API socket did not appear")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Firecracker exited early (rc={self._process.returncode})"
                )
            time.sleep(0.1)

        self._api = _FcApi(self._api_sock)

        # Configure the VM.
        self._configure_vm()

        # Start instance.
        self._api.put("/actions", {"action_type": "InstanceStart"})

        # Wait for SSH.
        self._wait_for_ssh()

        # Take an initial full snapshot as diff baseline.
        self._snap_dir = Path(tempfile.mkdtemp(prefix="fc-snapshots-"))
        self._take_full_snapshot()
        print(f"VM started (PID {self._process.pid})", file=sys.stderr)

    def _configure_vm(self) -> None:
        assert self._api is not None
        # Machine config — track_dirty_pages enables KVM dirty logging for
        # differential snapshots.
        self._api.put(
            "/machine-config",
            {
                "vcpu_count": self._vcpu,
                "mem_size_mib": self._mem_mib,
                "track_dirty_pages": True,
            },
        )
        # Boot source.
        self._api.put(
            "/boot-source",
            {
                "kernel_image_path": str(self._kernel),
                "boot_args": (
                    "console=ttyS0 reboot=k panic=1 pci=off "
                    "root=/dev/vda rw quiet"
                ),
            },
        )
        # Root drive.
        self._api.put(
            "/drives/rootfs",
            {
                "drive_id": "rootfs",
                "path_on_host": str(self._rootfs),
                "is_root_device": True,
                "is_read_only": False,
            },
        )
        # Network interface.
        self._api.put(
            "/network-interfaces/eth0",
            {
                "iface_id": "eth0",
                "guest_mac": "AA:FC:00:00:00:01",
                "host_dev_name": _FC_TAP_DEV,
            },
        )

    def _wait_for_ssh(self, timeout: int = _VM_BOOT_TIMEOUT_S) -> None:
        """Poll SSH until the guest is reachable."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    "sshpass", "-p", _FC_SSH_PASSWORD,
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", f"ConnectTimeout={_SSH_CONNECT_TIMEOUT_S}",
                    f"{_FC_SSH_USER}@{_FC_GUEST_IP}",
                    "echo ready",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "ready" in result.stdout:
                return
            time.sleep(1.0)
        raise TimeoutError(f"SSH did not become available within {timeout}s")

    def stop(self) -> None:
        """Shut down the VM and clean up resources."""
        if self._process is not None and self._process.poll() is None:
            try:
                if self._api is not None:
                    self._api.put("/actions", {"action_type": "SendCtrlAltDel"})
                self._process.wait(timeout=_VM_SHUTDOWN_TIMEOUT_S)
            except (subprocess.TimeoutExpired, Exception):
                self._process.kill()
                self._process.wait(timeout=5)

        if os.path.exists(self._api_sock):
            os.unlink(self._api_sock)

        _teardown_tap_device()

        # Clean up snapshot directory.
        if self._snap_dir is not None:
            shutil.rmtree(self._snap_dir, ignore_errors=True)

        print("VM stopped", file=sys.stderr)

    # -- Command execution --------------------------------------------------

    def execute(self, command: str, timeout: int = _COMMAND_TIMEOUT_S) -> tuple[str, int, float]:
        """Execute *command* inside the VM via SSH.

        Returns ``(stdout, returncode, elapsed_seconds)``.
        """
        t0 = time.monotonic()
        result = subprocess.run(
            [
                "sshpass", "-p", _FC_SSH_PASSWORD,
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", f"ConnectTimeout={_SSH_CONNECT_TIMEOUT_S}",
                f"{_FC_SSH_USER}@{_FC_GUEST_IP}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # slight margin over the logical timeout
        )
        elapsed = time.monotonic() - t0
        stdout = result.stdout
        if result.returncode != 0 and result.stderr:
            stdout += "\n[stderr]\n" + result.stderr
        return stdout, result.returncode, elapsed

    # -- Snapshot / measurement ---------------------------------------------

    def _take_full_snapshot(self) -> None:
        """Pause VM, create a full snapshot, and resume.

        The snapshot files become the diff baseline for subsequent per-turn
        diff snapshots.
        """
        assert self._snap_dir is not None
        assert self._api is not None

        mem_path = self._snap_dir / "full-mem.snap"
        state_path = self._snap_dir / "full-vmstate.snap"

        # Remove previous full snapshot if present.
        if mem_path.exists():
            mem_path.unlink()
        if state_path.exists():
            state_path.unlink()

        self._api.patch("/vm", {"state": "Paused"})
        try:
            self._api.put(
                "/snapshot/create",
                {
                    "snapshot_type": "Full",
                    "snapshot_path": str(mem_path),
                    "mem_file_path": str(mem_path),
                    "version": "1.1.0",
                },
            )
        finally:
            self._api.patch("/vm", {"state": "Resumed"})

        self._full_snapshot_mem = mem_path
        self._full_snapshot_state = state_path

    def snapshot_diff_and_reset_baseline(self) -> dict[str, Any]:
        """Take a diff snapshot, measure dirty pages, then reset the baseline.

        Returns a dict with ``dirty_memory_bytes`` and ``snapshot_pause_ms``.
        After this call a fresh full snapshot is taken so the next turn starts
        from a clean baseline.
        """
        assert self._snap_dir is not None
        assert self._api is not None

        # Take a diff snapshot relative to the last full snapshot.
        diff_mem = self._snap_dir / "diff-mem.snap"
        if diff_mem.exists():
            diff_mem.unlink()

        t0 = time.monotonic()
        self._api.patch("/vm", {"state": "Paused"})
        try:
            self._api.put(
                "/snapshot/create",
                {
                    "snapshot_type": "Diff",
                    "snapshot_path": str(diff_mem),
                    "mem_file_path": str(diff_mem),
                    "version": "1.1.0",
                },
            )
        finally:
            self._api.patch("/vm", {"state": "Resumed"})
        pause_ms = (time.monotonic() - t0) * 1000.0

        # The diff snapshot memory file size approximates the dirty page set
        # (with minor metadata overhead).
        dirty_bytes = diff_mem.stat().st_size if diff_mem.exists() else 0

        # Reset baseline for the next turn.
        self._take_full_snapshot()

        return {
            "dirty_memory_bytes": dirty_bytes,
            "snapshot_pause_ms": round(pause_ms, 3),
        }


# ---------------------------------------------------------------------------
# Trace parsing and command extraction
# ---------------------------------------------------------------------------


def parse_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL trace file, returning all records."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def iter_turns(
    records: Sequence[dict[str, Any]],
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield ``(turn_index, tool_exec_actions)`` tuples.

    A *turn* is the set of ``tool_exec`` actions that follow an ``llm_call``
    action.  Actions before the first ``llm_call`` (e.g. startup events) are
    skipped.  Empty turns are yielded with an empty action list.
    """
    turn_index = 0
    current_tool_actions: list[dict[str, Any]] = []
    in_turn = False
    for record in records:
        rtype = record.get("type")
        if rtype == "action" and record.get("action_type") == "llm_call":
            if in_turn:
                yield turn_index, current_tool_actions
                turn_index += 1
                current_tool_actions = []
            in_turn = True
        elif rtype == "action" and record.get("action_type") == "tool_exec":
            if in_turn:
                current_tool_actions.append(record)
        elif rtype == "summary":
            if in_turn:
                yield turn_index, current_tool_actions
                turn_index += 1
                current_tool_actions = []
            in_turn = False
    if in_turn:
        yield turn_index, current_tool_actions


def extract_commands(tool_actions: list[dict[str, Any]]) -> list[str]:
    """Extract shell commands from tool_exec actions.

    Only extracts commands from ``exec`` / ``bash`` / ``shell`` tool
    invocations where the tool args contain a ``command`` field.
    """
    commands: list[str] = []
    for action in tool_actions:
        data = action.get("data") or {}
        tool_name = data.get("tool_name", "")

        # OpenClaw uses 'exec' for shell commands.
        if tool_name in ("exec", "bash", "shell"):
            tool_args = data.get("tool_args") or {}
            if isinstance(tool_args, dict):
                cmd = tool_args.get("command") or tool_args.get("cmd")
                if isinstance(cmd, str) and cmd.strip():
                    commands.append(cmd.strip())

    return commands


# ---------------------------------------------------------------------------
# File-CAS delta computation (inlined from Stage 1 to keep this script
# self-contained — the KVM host may not have trace_collect importable).
# ---------------------------------------------------------------------------


def _relpath_is_skipped(relpath: str) -> bool:
    return any(part in _SKIP_DIRS for part in relpath.split("/"))


def _read_manifest_entries(manifest_path: str) -> dict[str, dict[str, Any]] | None:
    mpath = Path(manifest_path)
    if not mpath.is_file():
        return None
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_entries = data.get("entries", {})
    if not isinstance(raw_entries, dict):
        return None
    result: dict[str, dict[str, Any]] = {}
    for rel, entry in raw_entries.items():
        if not isinstance(rel, str):
            continue
        if _relpath_is_skipped(rel):
            continue
        if not isinstance(entry, dict):
            continue
        result[rel] = dict(entry)
    return result


def _fold_manifest(
    manifest_path: str,
    prev_folded: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    entries = _read_manifest_entries(manifest_path)
    if entries is None:
        return prev_folded
    mpath = Path(manifest_path)
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return prev_folded
    is_incremental = data.get("incremental", False) is True
    if not is_incremental:
        return entries
    folded = dict(prev_folded)
    folded.update(entries)
    deleted = data.get("deleted_paths", [])
    if isinstance(deleted, list):
        for dpath in deleted:
            if isinstance(dpath, str) and not _relpath_is_skipped(dpath):
                folded.pop(dpath, None)
    return folded


def _entry_hash(entry: dict[str, Any]) -> str:
    entry_type = entry.get("type", "file")
    if entry_type == "symlink":
        target = entry.get("target", "")
        return json.dumps({"type": "symlink", "target": target}, sort_keys=True)
    return str(entry.get("hash", ""))


def _entry_size_bytes(entry: dict[str, Any]) -> int:
    if entry.get("type") == "symlink":
        return 0
    size = entry.get("size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        return max(0, int(size))
    return 0


def _checkpoint_after_path(
    action_data: dict[str, Any],
    source_trace: Path,
) -> str | None:
    """Resolve the ``checkpoint_after`` CAS manifest path from a tool_exec action.

    Returns ``None`` when the action has no checkpoint or the root is not
    ``/testbed`` (the only root we currently handle).
    """
    raw = action_data.get("checkpoint_after")
    if raw is None:
        return None
    if isinstance(raw, str):
        spec_path = raw
    elif isinstance(raw, dict):
        spec_path = raw.get("path")
    else:
        return None
    if not spec_path:
        return None
    checkpoint_path = Path(str(spec_path))
    if not checkpoint_path.is_absolute():
        checkpoint_path = source_trace.parent / checkpoint_path
    if isinstance(raw, dict) and raw.get("root", "/testbed") != "/testbed":
        return None
    return str(checkpoint_path)


def compute_file_cas_delta_bytes(
    turn_actions: list[dict[str, Any]],
    source_trace: Path,
    previous_folded: dict[str, dict[str, Any]],
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Compute file-CAS delta bytes for a turn.

    Returns ``(delta_bytes, new_folded_state)``.
    """
    folded = dict(previous_folded)
    if not turn_actions:
        return 0, folded

    for action in turn_actions:
        data = action.get("data") or {}
        manifest_path = _checkpoint_after_path(data, source_trace)
        if manifest_path is None:
            continue
        folded = _fold_manifest(manifest_path, folded)

    previous_keys = set(previous_folded.keys())
    current_keys = set(folded.keys())

    delta_bytes = 0
    for key in current_keys - previous_keys:
        delta_bytes += _entry_size_bytes(folded[key])
    for key in current_keys & previous_keys:
        prev_entry = previous_folded[key]
        curr_entry = folded[key]
        if _entry_hash(prev_entry) != _entry_hash(curr_entry):
            delta_bytes += _entry_size_bytes(curr_entry)
            delta_bytes += _entry_size_bytes(prev_entry)

    return delta_bytes, folded


def estimate_dirty_bytes(file_cas_delta: int) -> int:
    """Estimate dirty-memory bytes from file-CAS delta bytes.

    See ``dr1_memory_diff_cost.py`` for the derivation.
    """
    return round(file_cas_delta * _ESTIMATE_DIRTY_OVERHEAD_RATIO)


def estimate_pause_ms(dirty_bytes: int) -> float:
    """Estimate snapshot pause duration (ms)."""
    transfer_ms = (dirty_bytes / (_DIRTY_PAGE_WRITE_MB_PER_S * 1_000_000)) * 1000.0
    return round(_SNAPSHOT_FIXED_COST_MS + transfer_ms, 3)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    trace_path: Path,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run the Stage 2 experiment for a single trace file.

    Returns a list of per-turn measurement rows (suitable for CSV output).
    """
    records = parse_trace_jsonl(trace_path)

    turns = list(iter_turns(records))
    total_turns = len(turns)
    if total_turns == 0:
        print(f"No turns found in {trace_path}", file=sys.stderr)
        return []

    if dry_run:
        total_commands = sum(len(extract_commands(actions)) for _, actions in turns)
        print(
            f"[DRY-RUN] {trace_path}: {len(records)} records, "
            f"{total_turns} turns, ~{total_commands} exec commands",
            file=sys.stderr,
        )
        return []

    # Ensure kernel + rootfs are cached.
    kernel_path, rootfs_path = ensure_fc_cache()

    # Start the VM.
    vm = FirecrackerVM(kernel_path, rootfs_path)
    rows: list[dict[str, Any]] = []
    try:
        vm.start()

        folded: dict[str, dict[str, Any]] = {}
        for turn_idx, turn_actions in turns:
            turn_number = turn_idx + 1
            error: str = ""

            # -- Stage 1: compute file-CAS delta from checkpoint manifests --
            cas_delta, folded = compute_file_cas_delta_bytes(
                turn_actions, trace_path, folded
            )
            dirty_est = estimate_dirty_bytes(cas_delta)
            pause_est = estimate_pause_ms(dirty_est)

            # -- Replay commands inside the VM --
            commands = extract_commands(turn_actions)
            wall_start = time.monotonic()
            for cmd in commands:
                try:
                    vm.execute(cmd, timeout=_COMMAND_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    error = f"command timed out after {_COMMAND_TIMEOUT_S}s"
                    break
                except RuntimeError as exc:
                    error = str(exc)[:200]
                    break
            wall_time_ms = (time.monotonic() - wall_start) * 1000.0

            # -- Snapshot and measure dirty memory --
            snap_info: dict[str, Any] = {}
            if not error:
                try:
                    snap_info = vm.snapshot_diff_and_reset_baseline()
                except RuntimeError as exc:
                    error = f"snapshot failed: {exc}"

            rows.append({
                "turn_number": turn_number,
                "file_cas_delta_bytes": cas_delta,
                "dirty_memory_bytes_est": dirty_est,
                "dirty_memory_bytes_meas": snap_info.get("dirty_memory_bytes", 0),
                "snapshot_pause_ms_est": pause_est,
                "snapshot_pause_ms_meas": snap_info.get("snapshot_pause_ms", 0.0),
                "wall_time_ms": round(wall_time_ms, 3),
                "tool_exec_count": len(turn_actions),
                "error": error,
            })

            if error:
                print(
                    f"  turn {turn_number}: ERROR — {error}", file=sys.stderr,
                )
            else:
                print(
                    f"  turn {turn_number}: "
                    f"cas={cas_delta}B, dirty_est={dirty_est}B, "
                    f"dirty_meas={snap_info.get('dirty_memory_bytes', 0)}B, "
                    f"pause={snap_info.get('snapshot_pause_ms', 0):.1f}ms, "
                    f"wall={wall_time_ms:.0f}ms",
                    file=sys.stderr,
                )
    finally:
        vm.stop()

    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], output_path: Path | None) -> None:
    """Write experiment results to CSV (stdout or file)."""
    out_fh = output_path.open("w", encoding="utf-8") if output_path else sys.stdout
    writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    if output_path:
        out_fh.close()
        print(f"Wrote {len(rows)} rows → {output_path}", file=sys.stderr)
    else:
        print(f"# Wrote {len(rows)} rows", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DR1 Stage 2: Firecracker VM runner for memory-diff cost measurement",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=None,
        help="Path to a single trace.jsonl file to replay inside Firecracker",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check environment readiness without launching a VM or processing traces",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write CSV to file instead of stdout",
    )
    args = parser.parse_args()

    # -- Environment check --
    issues = check_environment()
    if issues:
        print("Environment issues found:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        if not args.dry_run:
            print(
                "ERROR: Environment not ready. Use --dry-run to check without running.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print("Environment OK", file=sys.stderr)

    if args.dry_run:
        if args.trace_jsonl is not None:
            if not args.trace_jsonl.exists():
                print(
                    f"ERROR: --trace-jsonl file not found: {args.trace_jsonl}",
                    file=sys.stderr,
                )
                sys.exit(1)
            _ = run_experiment(args.trace_jsonl, dry_run=True)
        print("[DRY-RUN] All checks passed.", file=sys.stderr)
        return

    if args.trace_jsonl is None:
        parser.error("--trace-jsonl is required (unless --dry-run)")
    if not args.trace_jsonl.exists():
        print(
            f"ERROR: --trace-jsonl file not found: {args.trace_jsonl}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = run_experiment(args.trace_jsonl)
    _write_csv(rows, args.output)


if __name__ == "__main__":
    main()
