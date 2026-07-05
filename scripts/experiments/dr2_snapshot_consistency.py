#!/usr/bin/env python3
"""DR2: Host-mount-and-hash consistency experiment.

Hypothesis: dm-thin snapshot after in-guest syncfs/fsfreeze yields host-side
hashes matching in-guest ground truth 100%, while snapshots taken without
explicit sync may show divergence under K1-adversary workloads.

The experiment provisions a thin-provisioned volume, boots a Firecracker VM
with it mounted as a data disk, runs workloads (including mtime-preserving
moves, tar extractions, and touch backdating), then compares host-side hashes
(from a dm-thin snapshot) against in-guest ground-truth hashes at each step,
under three sync regimes: none, syncfs, and fsfreeze.

Output: CSV with columns workload, sync_method, guest_hash, host_hash, match,
changed_blocks, time_ms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# In-guest test directory — the thin volume is mounted here inside the VM.
_GUEST_TEST_DIR = "/testbed"

# File to place inside the VM for in-guest hashing.
_GUEST_HASH_SCRIPT = "/tmp/dr2_hash.py"

# Host-side mount point for snapshot read-only verification.
_HOST_SNAP_MOUNT = "/mnt/dr2-snap"

# dm-thin pool name and volume naming.
_THIN_POOL_NAME = "dr2-thin-pool"
_THIN_VOLUME_NAME = "dr2-thin-vol"
_THIN_SNAP_PREFIX = "dr2-snap"

# Firecracker socket path.
_FC_SOCKET_PATH = "/tmp/dr2-fc.sock"
_FC_LOG_PATH = "/tmp/dr2-fc.log"
_FC_PID_PATH = "/tmp/dr2-fc.pid"

# Device mapper paths derived from names.
_THIN_POOL_DEV = f"/dev/mapper/{_THIN_POOL_NAME}"
_THIN_VOL_DEV = f"/dev/mapper/{_THIN_VOLUME_NAME}"

# Thin volume size (4 GiB).
_THIN_VOLUME_SIZE_SECTORS = 4 * 1024 * 1024 * 1024 // 512  # 4 GiB in 512-byte sectors
_THIN_POOL_DATA_SIZE = "5G"
_THIN_POOL_META_SIZE = "256M"

# Block size for thin provisioning.
_THIN_BLOCK_SIZE = 128  # 64 KiB blocks (128 sectors × 512 bytes)

# Timeout defaults (seconds).
_SSH_TIMEOUT = 30
_FC_BOOT_TIMEOUT = 120
_DMSETUP_TIMEOUT = 30
_MOUNT_TIMEOUT = 15
_WORKLOAD_TIMEOUT = 60

# Sync methods tested.
SYNC_METHODS = ("none", "syncfs", "fsfreeze")

# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Workload:
    """A named workload with its shell commands to execute inside the guest."""

    name: str
    description: str
    setup_commands: tuple[str, ...] = ()  # Run once before the workload family
    commands: tuple[str, ...] = ()  # The actual workload steps

    def guest_shell_script(self) -> str:
        """Produce a self-contained POSIX shell script that runs the workload.

        The script writes the expected hash manifest to stdout as a JSON object
        mapping relative paths (under /testbed) to SHA-256 hex digests.
        """
        lines: list[str] = [
            "#!/bin/sh",
            "set -eu",
            "",
            # Ensure /testbed exists and is writable.
            f'test -d {_GUEST_TEST_DIR} || mkdir -p {_GUEST_TEST_DIR}',
            f"cd {_GUEST_TEST_DIR}",
            "",
            "# -- workload begin --",
        ]
        for cmd in self.setup_commands:
            lines.append(cmd)
        for cmd in self.commands:
            lines.append(cmd)
        # Emit JSON hash manifest.
        lines.extend(
            [
                "",
                "# -- hash manifest --",
                'python3 -c "',
                "import json, os, hashlib",
                "manifest = {}",
                f'for root, dirs, files in os.walk({_GUEST_TEST_DIR!r}):',
                "    for fname in sorted(files):",
                "        fpath = os.path.join(root, fname)",
                "        rel = os.path.relpath(fpath, start={!r})".format(
                    _GUEST_TEST_DIR
                ),
                "        h = hashlib.sha256()",
                "        with open(fpath, 'rb') as f:",
                "            while True:",
                "                chunk = f.read(65536)",
                "                if not chunk:",
                "                    break",
                "                h.update(chunk)",
                "        manifest[rel] = h.hexdigest()",
                "print(json.dumps(manifest, sort_keys=True))",
                '"',
            ]
        )
        return "\n".join(lines)


# Regular file write and read-back (control workload).
_WL_CONTROL = Workload(
    name="control",
    description="Regular file write and read-back (control)",
    commands=(
        "echo 'DR2 control payload' > control_a.txt",
        "dd if=/dev/urandom of=control_b.bin bs=4096 count=16 status=none",
        "echo 'appended' >> control_a.txt",
    ),
)

# mv — mtime-preserving move.
_WL_MOVE = Workload(
    name="k1_move",
    description="mv old_file to /testbed (mtime-preserving move)",
    setup_commands=(
        # Pre-create a file with a known old mtime outside /testbed.
        "mkdir -p /tmp/dr2-k1",
        "echo 'k1 move source' > /tmp/dr2-k1/move_src.txt",
        "touch -d '2020-01-15 10:30:00' /tmp/dr2-k1/move_src.txt",
    ),
    commands=(
        "mv /tmp/dr2-k1/move_src.txt move_dst.txt",
        # Verify mtime was preserved.
        'test "$(stat -c %Y move_dst.txt)" = "1579084200" || '
        'echo "WARNING: mtime not preserved" >&2',
    ),
)

# tar -xp — archive extraction (sets old mtimes).
_WL_TAR = Workload(
    name="k1_tar",
    description="tar -xp archive extraction (sets old mtimes)",
    setup_commands=(
        # Pre-create a tar archive with files having old mtimes.
        "mkdir -p /tmp/dr2-k1-tar",
        "echo 'tar file a' > /tmp/dr2-k1-tar/a.txt",
        "echo 'tar file b' > /tmp/dr2-k1-tar/b.txt",
        "touch -d '2019-06-01 12:00:00' /tmp/dr2-k1-tar/a.txt",
        "touch -d '2019-06-01 12:00:00' /tmp/dr2-k1-tar/b.txt",
        "tar -C /tmp/dr2-k1-tar -cf /tmp/dr2-k1.tar a.txt b.txt",
    ),
    commands=(
        "mkdir -p tar_extract",
        "tar -xp -C tar_extract -f /tmp/dr2-k1.tar",
    ),
)

# touch -d — backdating file mtimes.
_WL_TOUCH = Workload(
    name="k1_touch",
    description="touch -d backdating",
    commands=(
        "echo 'backdated content' > touch_backdated.txt",
        "touch -d '2018-03-20 08:00:00' touch_backdated.txt",
        # Verify.
        'test "$(stat -c %Y touch_backdated.txt)" = "1521532800" || '
        'echo "WARNING: backdate not applied" >&2',
    ),
)

ALL_WORKLOADS: tuple[Workload, ...] = (
    _WL_CONTROL,
    _WL_MOVE,
    _WL_TAR,
    _WL_TOUCH,
)

# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


def ssh(
    host: str,
    command: str,
    *,
    port: int = 22,
    user: str = "root",
    identity_file: str | None = None,
    timeout: float = _SSH_TIMEOUT,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a remote host via the system ``ssh`` binary."""
    cmd: list[str] = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={int(timeout)}",
        "-o",
        "ServerAliveInterval=5",
        "-p",
        str(port),
    ]
    if identity_file:
        cmd.extend(["-i", identity_file])
    cmd.extend([f"{user}@{host}", command])
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout + 5,
        check=False,
    )


def ssh_script(
    host: str,
    script: str,
    *,
    port: int = 22,
    user: str = "root",
    identity_file: str | None = None,
    timeout: float = _SSH_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Pipe a script via SSH stdin to the remote host."""
    cmd: list[str] = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={int(timeout)}",
        "-o",
        "ServerAliveInterval=5",
        "-p",
        str(port),
    ]
    if identity_file:
        cmd.extend(["-i", identity_file])
    cmd.extend([f"{user}@{host}", "/bin/sh -s"])
    return subprocess.run(
        cmd,
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout + 10,
        check=False,
    )


# ---------------------------------------------------------------------------
# dm-thin helpers
# ---------------------------------------------------------------------------


def _dmsetup(*args: str, timeout: float = _DMSETUP_TIMEOUT) -> None:
    """Run a dmsetup command, raising on failure."""
    cmd = ["dmsetup", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"dmsetup {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )


def _thin_check(metadata_dev: str, timeout: float = _DMSETUP_TIMEOUT) -> None:
    """Verify thin pool metadata consistency."""
    result = subprocess.run(
        ["thin_check", metadata_dev],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"thin_check failed: {result.stderr.strip()}")


def thin_pool_create(
    data_dev: str,
    metadata_dev: str,
    *,
    volume_size_sectors: int = _THIN_VOLUME_SIZE_SECTORS,
    block_size: int = _THIN_BLOCK_SIZE,
    timeout: float = _DMSETUP_TIMEOUT,
) -> None:
    """Create a dm-thin pool from data and metadata block devices."""
    pool_table = (
        f"0 {volume_size_sectors} thin-pool "
        f"{metadata_dev} {data_dev} {block_size} 0"
    )
    _dmsetup("create", _THIN_POOL_NAME, "--table", pool_table, timeout=timeout)
    # Verify.
    _thin_check(metadata_dev)


def thin_volume_create(
    thin_id: int,
    *,
    timeout: float = _DMSETUP_TIMEOUT,
) -> None:
    """Create a thin volume in the pool.

    The volume is activated as /dev/mapper/<name>.
    """
    _dmsetup(
        "message",
        _THIN_POOL_NAME,
        "0",
        f"create_thin {thin_id}",
        timeout=timeout,
    )
    table = f"0 {_THIN_VOLUME_SIZE_SECTORS} thin /dev/mapper/{_THIN_POOL_NAME} {thin_id}"
    _dmsetup("create", _THIN_VOLUME_NAME, "--table", table, timeout=timeout)


def thin_snapshot(
    snap_id: int,
    origin_id: int,
    *,
    timeout: float = _DMSETUP_TIMEOUT,
) -> str:
    """Create a dm-thin snapshot of an origin volume.

    Returns the device path of the new snapshot (e.g. /dev/mapper/dr2-snap-1).
    """
    snap_name = f"{_THIN_SNAP_PREFIX}-{snap_id}"
    _dmsetup(
        "message",
        _THIN_POOL_NAME,
        "0",
        f"create_snap {snap_id} {origin_id}",
        timeout=timeout,
    )
    table = f"0 {_THIN_VOLUME_SIZE_SECTORS} thin /dev/mapper/{_THIN_POOL_NAME} {snap_id}"
    _dmsetup("create", snap_name, "--table", table, timeout=timeout)
    return f"/dev/mapper/{snap_name}"


def thin_snapshot_remove(snap_id: int, *, timeout: float = _DMSETUP_TIMEOUT) -> None:
    """Remove a dm-thin snapshot."""
    snap_name = f"{_THIN_SNAP_PREFIX}-{snap_id}"
    _dmsetup("remove", snap_name, timeout=timeout)
    _dmsetup("message", _THIN_POOL_NAME, "0", f"delete {snap_id}", timeout=timeout)


def thin_volume_remove(*, timeout: float = _DMSETUP_TIMEOUT) -> None:
    """Remove the thin volume device."""
    _dmsetup("remove", _THIN_VOLUME_NAME, timeout=timeout)


def thin_pool_remove(*, timeout: float = _DMSETUP_TIMEOUT) -> None:
    """Remove the thin pool device."""
    _dmsetup("remove", _THIN_POOL_NAME, timeout=timeout)


def thin_delta(
    snap_id: int,
    origin_id: int,
    pool_metadata: str,
    *,
    timeout: float = _DMSETUP_TIMEOUT,
) -> int:
    """Count the number of changed blocks between origin and snapshot.

    Uses ``thin_delta --thin1`` to emit a binary diff.  Parses the number of
    differing data-mapping entries as the changed-block count.  Returns 0 if
    ``thin_delta`` is unavailable (e.g. on non-KVM hosts).
    """
    if not shutil.which("thin_delta"):
        return -1
    cmd = [
        "thin_delta",
        "--thin1",
        "-m",
        str(snap_id),
        "-o",
        str(origin_id),
        pool_metadata,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        return -1
    if not result.stdout:
        return 0
    # thin_delta --thin1 outputs one 24-byte header plus 24-byte records per
    # differing mapping.  Each record encodes (begin_block, end_block, mapped).
    header_size = 24
    record_size = 24
    if len(result.stdout) < header_size:
        return 0
    data_len = len(result.stdout) - header_size
    return max(0, data_len // record_size)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _mount_ro(snap_dev: str, mount_point: str, *, timeout: float = _MOUNT_TIMEOUT) -> None:
    """Mount a block device read-only with noload (no journal replay)."""
    os.makedirs(mount_point, exist_ok=True)
    result = subprocess.run(
        ["mount", "-o", "ro,noload", snap_dev, mount_point],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mount {snap_dev} → {mount_point} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _umount(mount_point: str, *, timeout: float = _MOUNT_TIMEOUT) -> None:
    """Unmount a filesystem."""
    result = subprocess.run(
        ["umount", mount_point],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        # Lazy unmount as fallback.
        subprocess.run(
            ["umount", "-l", mount_point],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


def host_hash_directory(mount_point: str) -> dict[str, str]:
    """Recursively hash all regular files under *mount_point*.

    Returns a dict mapping relative paths to SHA-256 hex digests.
    """
    manifest: dict[str, str] = {}
    if not os.path.isdir(mount_point):
        return manifest
    mp = Path(mount_point)
    for fpath in sorted(mp.rglob("*")):
        if not fpath.is_file():
            continue
        rel = str(fpath.relative_to(mp))
        h = hashlib.sha256()
        try:
            with open(fpath, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
        except (OSError, PermissionError):
            continue
        manifest[rel] = h.hexdigest()
    return manifest


# ---------------------------------------------------------------------------
# In-guest sync operations
# ---------------------------------------------------------------------------


def guest_sync(
    host: str,
    sync_method: str,
    *,
    port: int = 22,
    user: str = "root",
    identity_file: str | None = None,
) -> str | None:
    """Execute an in-guest sync operation.

    Returns the stdout (for verification) or raises on failure.
    """
    if sync_method == "none":
        return None
    elif sync_method == "syncfs":
        # sync -f <path> flushes the filesystem containing that path.
        result = ssh(
            host,
            f"sync -f {_GUEST_TEST_DIR} && echo 'syncfs_ok'",
            port=port,
            user=user,
            identity_file=identity_file,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"syncfs failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout.strip()
    elif sync_method == "fsfreeze":
        # fsfreeze --freeze flushes and halts writes; --unfreeze resumes.
        freeze = ssh(
            host,
            f"fsfreeze --freeze {_GUEST_TEST_DIR} && echo 'freeze_ok'",
            port=port,
            user=user,
            identity_file=identity_file,
        )
        if freeze.returncode != 0:
            raise RuntimeError(
                f"fsfreeze --freeze failed: {freeze.stderr.strip() or freeze.stdout.strip()}"
            )
        # Caller must unfreeze after the snapshot is taken.
        return freeze.stdout.strip()
    else:
        raise ValueError(f"Unknown sync method: {sync_method}")


def guest_unfreeze(
    host: str,
    *,
    port: int = 22,
    user: str = "root",
    identity_file: str | None = None,
) -> None:
    """Thaw a frozen filesystem inside the guest."""
    result = ssh(
        host,
        f"fsfreeze --unfreeze {_GUEST_TEST_DIR}",
        port=port,
        user=user,
        identity_file=identity_file,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fsfreeze --unfreeze failed: {result.stderr.strip() or result.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# Firecracker lifecycle
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FirecrackerVM:
    """Handle for a running Firecracker microVM."""

    pid: int
    socket_path: str
    log_path: str
    guest_ip: str
    ssh_port: int
    ssh_user: str
    ssh_identity_file: str | None
    kernel_path: str
    rootfs_path: str
    thin_volume_dev: str
    _cleaned_up: bool = False

    def stop(self) -> None:
        """Gracefully shut down the Firecracker VM."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            # Send shutdown via SSH first.
            ssh(
                self.guest_ip,
                "sync; poweroff || shutdown -h now || halt",
                port=self.ssh_port,
                user=self.ssh_user,
                identity_file=self.ssh_identity_file,
                timeout=10,
            )
        except Exception:
            pass
        # Kill Firecracker process if still running.
        try:
            os.kill(self.pid, 0)
            os.kill(self.pid, 9)  # SIGKILL
        except OSError:
            pass
        # Clean up socket.
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

    def wait_ready(self, timeout: float = _FC_BOOT_TIMEOUT) -> bool:
        """Poll SSH until the VM accepts connections."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = ssh(
                self.guest_ip,
                "echo ready",
                port=self.ssh_port,
                user=self.ssh_user,
                identity_file=self.ssh_identity_file,
                timeout=5,
            )
            if result.returncode == 0 and "ready" in result.stdout:
                return True
            time.sleep(1)
        return False


def _firecracker_launch(
    *,
    kernel_path: str,
    rootfs_path: str,
    socket_path: str,
    log_path: str,
    thin_volume_dev: str,
    fc_binary: str = "firecracker",
    guest_ip: str = "192.168.100.2",
    ssh_port: int = 22,
    ssh_user: str = "root",
    ssh_identity_file: str | None = None,
) -> FirecrackerVM:
    """Launch a Firecracker microVM.

    The VM boots from *kernel_path* with *rootfs_path* as the root block device.
    *thin_volume_dev* is passed through as a secondary block device mounted at
    ``/testbed`` inside the guest.
    """
    # Build Firecracker config.
    # The config uses a kernel image + rootfs as root drive, plus the thin
    # volume as a second drive.
    config: dict[str, Any] = {
        "boot-source": {
            "kernel_image_path": kernel_path,
            "boot_args": (
                "console=ttyS0 reboot=k panic=1 pci=off "
                f"ip={guest_ip}::{_gateway_from_ip(guest_ip)}:255.255.255.0::eth0:off"
            ),
        },
        "drives": [
            {
                "drive_id": "rootfs",
                "path_on_host": rootfs_path,
                "is_root_device": True,
                "is_read_only": False,
            },
            {
                "drive_id": "thin_volume",
                "path_on_host": thin_volume_dev,
                "is_root_device": False,
                "is_read_only": False,
            },
        ],
        "network-interfaces": [
            {
                "iface_id": "eth0",
                "guest_mac": "AA:BB:CC:DD:EE:01",
                "host_dev_name": _tap_device_name(),
            }
        ],
        "machine-config": {
            "vcpu_count": 2,
            "mem_size_mib": 1024,
            "smt": False,
        },
    }

    # Write config to a temp file.
    config_path = f"/tmp/dr2-fc-config-{os.getpid()}.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # Ensure socket path is clear.
    try:
        os.unlink(socket_path)
    except OSError:
        pass

    # Launch Firecracker.
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [fc_binary, "--api-sock", socket_path, "--config-file", config_path],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    # Give Firecracker a moment to start; the PID might be the jailer or
    # firecracker itself. Wait for the socket to appear.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if os.path.exists(socket_path):
            break
        time.sleep(0.1)

    if not os.path.exists(socket_path):
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"Firecracker did not create API socket at {socket_path} "
            f"within 10s — check {log_path}"
        )

    vm = FirecrackerVM(
        pid=proc.pid,
        socket_path=socket_path,
        log_path=log_path,
        guest_ip=guest_ip,
        ssh_port=ssh_port,
        ssh_user=ssh_user,
        ssh_identity_file=ssh_identity_file,
        kernel_path=kernel_path,
        rootfs_path=rootfs_path,
        thin_volume_dev=thin_volume_dev,
    )
    return vm


def _tap_device_name() -> str:
    """Return the tap device name for Firecracker networking.

    The host must have a pre-configured tap device (e.g. ``tap0``).
    """
    # Check common tap device names.
    for name in ("tap0", "fc-tap0", "dr2-tap"):
        if os.path.exists(f"/sys/class/net/{name}"):
            return name
    # Default — the caller must ensure this exists.
    return "tap0"


def _gateway_from_ip(ip: str) -> str:
    """Derive a gateway address from a guest IP (assumes /24)."""
    parts = ip.rsplit(".", 1)
    if len(parts) == 2:
        return f"{parts[0]}.1"
    return "192.168.100.1"


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "workload",
    "sync_method",
    "guest_hash",
    "host_hash",
    "match",
    "changed_blocks",
    "time_ms",
]


@dataclass(slots=True)
class ExperimentConfig:
    """All tunables for a DR2 run."""

    # Block device configuration.
    thin_pool_data_dev: str
    thin_pool_meta_dev: str

    # Firecracker configuration.
    kernel_path: str
    rootfs_path: str
    fc_binary: str = "firecracker"

    # SSH configuration.
    guest_ip: str = "192.168.100.2"
    ssh_port: int = 22
    ssh_user: str = "root"
    ssh_identity_file: str | None = None

    # Experiment control.
    sync_methods: Sequence[str] = SYNC_METHODS
    workloads: Sequence[Workload] = ALL_WORKLOADS
    dry_run: bool = False
    keep_snapshots: bool = False
    output: Path | None = None

    # Internal state (populated at run time).
    _thin_origin_id: int = 0
    _snap_counter: int = 0
    _vm: FirecrackerVM | None = None
    _snap_devs: list[str] = field(default_factory=list)


def _run_experiment(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Run the full DR2 experiment and return rows for the CSV."""
    if config.dry_run:
        return _run_dry(config)

    rows: list[dict[str, Any]] = []
    origin_id = config._thin_origin_id
    snap_counter = config._snap_counter

    try:
        # 1. Create thin pool and volume.
        thin_pool_create(config.thin_pool_data_dev, config.thin_pool_meta_dev)
        thin_volume_create(thin_id=origin_id)

        # 2. Format the thin volume with ext4 (or verify it's already formatted).
        _ensure_ext4(_THIN_VOL_DEV)

        # 3. Launch Firecracker VM.
        vm = _firecracker_launch(
            kernel_path=config.kernel_path,
            rootfs_path=config.rootfs_path,
            socket_path=_FC_SOCKET_PATH,
            log_path=_FC_LOG_PATH,
            thin_volume_dev=_THIN_VOL_DEV,
            fc_binary=config.fc_binary,
            guest_ip=config.guest_ip,
            ssh_port=config.ssh_port,
            ssh_user=config.ssh_user,
            ssh_identity_file=config.ssh_identity_file,
        )
        config._vm = vm
        if not vm.wait_ready():
            raise RuntimeError(f"Firecracker VM at {config.guest_ip}:{config.ssh_port} "
                               f"did not become SSH-ready within {_FC_BOOT_TIMEOUT}s")

        # 4. Mount the thin volume inside the guest.
        _guest_mount_thin_volume(config)

        # 5. For each sync method, run all workloads.
        for sync_method in config.sync_methods:
            for workload in config.workloads:
                row = _run_workload_trial(
                    config=config,
                    workload=workload,
                    sync_method=sync_method,
                    origin_id=origin_id,
                    snap_counter=snap_counter,
                )
                rows.append(row)
                snap_counter += 1

        # 6. Cleanup snapshots.
        if not config.keep_snapshots:
            for i, snap_dev in enumerate(config._snap_devs):
                try:
                    _umount(_HOST_SNAP_MOUNT)
                except Exception:
                    pass
                thin_snapshot_remove(snap_id=origin_id + i + 1)

    finally:
        # Best-effort cleanup.
        if config._vm is not None:
            try:
                config._vm.stop()
            except Exception:
                pass
        try:
            _umount(_HOST_SNAP_MOUNT)
        except Exception:
            pass
        try:
            thin_volume_remove()
        except Exception:
            pass
        try:
            thin_pool_remove()
        except Exception:
            pass
        # Remove stale config files.
        for pattern in ("/tmp/dr2-fc-config-*.json",):
            for p in Path("/tmp").glob(pattern):
                try:
                    p.unlink()
                except OSError:
                    pass

    return rows


def _guest_mount_thin_volume(config: ExperimentConfig) -> None:
    """Mount the thin volume inside the guest at /testbed.

    The guest kernel sees the secondary block device as /dev/vdb (first
    non-root virtio-blk device).  If the device is not already formatted,
    format it; otherwise mount it.
    """
    mount_script = (
        f"#!/bin/sh\n"
        f"set -eu\n"
        f"# Find the second virtio-blk device (first is rootfs).\n"
        f"BLKDEV=$(ls /dev/vd* 2>/dev/null | grep -v vda | head -1)\n"
        f'if [ -z "$BLKDEV" ]; then BLKDEV=/dev/vdb; fi\n'
        f"mkdir -p {_GUEST_TEST_DIR}\n"
        f"# Check if already formatted.\n"
        f"if blkid \"$BLKDEV\" >/dev/null 2>&1; then\n"
        f"    mount \"$BLKDEV\" {_GUEST_TEST_DIR} || true\n"
        f"else\n"
        f"    mkfs.ext4 -F \"$BLKDEV\"\n"
        f"    mount \"$BLKDEV\" {_GUEST_TEST_DIR}\n"
        f"fi\n"
    )
    result = ssh_script(
        config.guest_ip,
        mount_script,
        port=config.ssh_port,
        user=config.ssh_user,
        identity_file=config.ssh_identity_file,
        timeout=_WORKLOAD_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Guest mount of thin volume failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _ensure_ext4(dev: str) -> None:
    """Ensure *dev* has an ext4 filesystem; format if not."""
    result = subprocess.run(
        ["blkid", dev],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0 or "ext4" not in result.stdout:
        subprocess.run(
            ["mkfs.ext4", "-F", dev],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )


def _run_workload_trial(
    *,
    config: ExperimentConfig,
    workload: Workload,
    sync_method: str,
    origin_id: int,
    snap_counter: int,
) -> dict[str, Any]:
    """Execute a single workload trial and return a CSV row."""
    t_start = time.monotonic()

    # 1. Run the workload inside the guest via SSH.
    script = workload.guest_shell_script()
    result = ssh_script(
        config.guest_ip,
        script,
        port=config.ssh_port,
        user=config.ssh_user,
        identity_file=config.ssh_identity_file,
        timeout=_WORKLOAD_TIMEOUT,
    )

    # 2. Execute in-guest sync (none / syncfs / fsfreeze).
    frozen = False
    if sync_method == "fsfreeze":
        guest_sync(
            config.guest_ip,
            "fsfreeze",
            port=config.ssh_port,
            user=config.ssh_user,
            identity_file=config.ssh_identity_file,
        )
        frozen = True
    elif sync_method == "syncfs":
        guest_sync(
            config.guest_ip,
            "syncfs",
            port=config.ssh_port,
            user=config.ssh_user,
            identity_file=config.ssh_identity_file,
        )

    try:
        # 3. Take a dm-thin snapshot.
        snap_id = origin_id + snap_counter + 1
        snap_dev = thin_snapshot(snap_id=snap_id, origin_id=origin_id)
        config._snap_devs.append(snap_dev)
    finally:
        if frozen:
            guest_unfreeze(
                config.guest_ip,
                port=config.ssh_port,
                user=config.ssh_user,
                identity_file=config.ssh_identity_file,
            )

    # 4. In-guest hash (from the workload script's JSON stdout).
    guest_manifest: dict[str, str] = {}
    if result.returncode == 0 and result.stdout.strip():
        # The last non-empty line of stdout should be the JSON manifest.
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    guest_manifest = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break

    # 5. Host-side: mount snapshot read-only and hash.
    _mount_ro(snap_dev, _HOST_SNAP_MOUNT)
    try:
        host_manifest = host_hash_directory(_HOST_SNAP_MOUNT)
    finally:
        _umount(_HOST_SNAP_MOUNT)

    # 6. thin_delta: count changed blocks.
    changed_blocks = thin_delta(
        snap_id=snap_id,
        origin_id=origin_id,
        pool_metadata=config.thin_pool_meta_dev,
    )

    # 7. Compare hashes.
    guest_hash_agg = _aggregate_hash(guest_manifest)
    host_hash_agg = _aggregate_hash(host_manifest)
    match = guest_hash_agg == host_hash_agg

    elapsed_ms = int((time.monotonic() - t_start) * 1000)

    return {
        "workload": workload.name,
        "sync_method": sync_method,
        "guest_hash": guest_hash_agg,
        "host_hash": host_hash_agg,
        "match": match,
        "changed_blocks": changed_blocks,
        "time_ms": elapsed_ms,
    }


def _aggregate_hash(manifest: dict[str, str]) -> str:
    """Produce a single deterministic hash over a file-hash manifest.

    Sorts the entries by path, concatenates path:hash pairs, and SHA-256's
    the result.  This gives a single comparable hash for the entire directory.
    """
    if not manifest:
        return hashlib.sha256(b"").hexdigest()
    parts = sorted(manifest.items())
    h = hashlib.sha256()
    for path, file_hash in parts:
        h.update(f"{path}:{file_hash}\n".encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


def _run_dry(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Simulate the experiment flow without executing real system commands."""
    print("=== DR2 DRY RUN ===", file=sys.stderr)
    print(f"thin_pool_data_dev: {config.thin_pool_data_dev}", file=sys.stderr)
    print(f"thin_pool_meta_dev: {config.thin_pool_meta_dev}", file=sys.stderr)
    print(f"kernel_path: {config.kernel_path}", file=sys.stderr)
    print(f"rootfs_path: {config.rootfs_path}", file=sys.stderr)
    print(f"guest_ip: {config.guest_ip}", file=sys.stderr)
    print(f"ssh_port: {config.ssh_port}", file=sys.stderr)
    print(f"ssh_user: {config.ssh_user}", file=sys.stderr)
    print(f"sync_methods: {list(config.sync_methods)}", file=sys.stderr)
    print(f"workloads: {[w.name for w in config.workloads]}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for sync_method in config.sync_methods:
        for workload in config.workloads:
            print(
                f"  [DRY-RUN] workload={workload.name} sync={sync_method}",
                file=sys.stderr,
            )
            rows.append(
                {
                    "workload": workload.name,
                    "sync_method": sync_method,
                    "guest_hash": "-",
                    "host_hash": "-",
                    "match": True,
                    "changed_blocks": 0,
                    "time_ms": 1,
                }
            )

    print(f"=== DRY-RUN complete: {len(rows)} trials ===", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DR2: Host-mount-and-hash consistency experiment"
    )
    # Required block device paths.
    parser.add_argument(
        "--thin-pool-data-dev",
        default="/dev/loop0",
        help="Block device for thin pool data (default: /dev/loop0)",
    )
    parser.add_argument(
        "--thin-pool-meta-dev",
        default="/dev/loop1",
        help="Block device for thin pool metadata (default: /dev/loop1)",
    )
    # Firecracker configuration.
    parser.add_argument(
        "--kernel-path",
        default="/var/lib/firecracker/vmlinux.bin",
        help="Path to Firecracker-compatible kernel image",
    )
    parser.add_argument(
        "--rootfs-path",
        default="/var/lib/firecracker/rootfs.ext4",
        help="Path to root filesystem image for the VM",
    )
    parser.add_argument(
        "--fc-binary",
        default="firecracker",
        help="Path to the Firecracker binary (default: firecracker)",
    )
    # SSH configuration.
    parser.add_argument(
        "--guest-ip",
        default="192.168.100.2",
        help="IP address of the Firecracker guest (default: 192.168.100.2)",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port on the guest (default: 22)",
    )
    parser.add_argument(
        "--ssh-user",
        default="root",
        help="SSH user for guest access (default: root)",
    )
    parser.add_argument(
        "--ssh-identity-file",
        default=None,
        help="SSH private key for guest access",
    )
    # Experiment control.
    parser.add_argument(
        "--sync-methods",
        nargs="+",
        default=list(SYNC_METHODS),
        choices=list(SYNC_METHODS),
        help="Sync methods to test (default: none syncfs fsfreeze)",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=[w.name for w in ALL_WORKLOADS],
        choices=[w.name for w in ALL_WORKLOADS],
        help="Workloads to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment plan without executing system commands",
    )
    parser.add_argument(
        "--keep-snapshots",
        action="store_true",
        help="Do not remove snapshots after the experiment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write CSV to file instead of stdout",
    )
    parser.add_argument(
        "--thin-origin-id",
        type=int,
        default=0,
        help="Thin device ID for the origin volume (default: 0)",
    )
    args = parser.parse_args()

    # Resolve workload objects.
    wl_map = {w.name: w for w in ALL_WORKLOADS}
    workloads = tuple(wl_map[name] for name in args.workloads)

    config = ExperimentConfig(
        thin_pool_data_dev=args.thin_pool_data_dev,
        thin_pool_meta_dev=args.thin_pool_meta_dev,
        kernel_path=args.kernel_path,
        rootfs_path=args.rootfs_path,
        fc_binary=args.fc_binary,
        guest_ip=args.guest_ip,
        ssh_port=args.ssh_port,
        ssh_user=args.ssh_user,
        ssh_identity_file=args.ssh_identity_file,
        sync_methods=tuple(args.sync_methods),
        workloads=workloads,
        dry_run=args.dry_run,
        keep_snapshots=args.keep_snapshots,
        output=args.output,
        _thin_origin_id=args.thin_origin_id,
    )

    rows = _run_experiment(config)

    out_fh = config.output.open("w", encoding="utf-8") if config.output else sys.stdout
    writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    if config.output:
        out_fh.close()
        print(f"Wrote {len(rows)} rows → {config.output}", file=sys.stderr)
    else:
        print(
            f"# Wrote {len(rows)} rows from {len(config.sync_methods)} sync methods × "
            f"{len(config.workloads)} workloads",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
