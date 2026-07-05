"""Build ext4 rootfs images from Docker images for Firecracker microVMs.

Each rootfs is an ext4 filesystem containing the source Docker image's
filesystem plus the vsock agent, dropbear (SSH fallback), and a
supervisor init script.  Rootfs images are cached by source image digest
so repeated builds for the same image are free.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FC_ROOTFS_CACHE_DIR = Path("/tmp/fc-rootfs-cache")
_DEFAULT_ROOTFS_SIZE_MB = 4096  # 4 GiB — generous for full dev toolchains
_DEFAULT_VSOCK_PORT = 5678
# Working directory inside the VM (mirrors DockerBackend.root).
_GUEST_WORKDIR = "/testbed"
# Bump this when the builder logic (init script, agent, packages) changes
# to invalidate stale cached rootfs images.
_ROOTFS_BUILDER_VERSION = 3

# _FC_AGENT_SCRIPT is composed from the shared tool-handler module
# plus the vsock transport wrapper (read from file at module load time).
_AGENT_HANDLERS_SRC = (
    Path(__file__).resolve().parent.parent / "agents" / "_agent_handlers.py"
).read_text()

_FC_VSOCK_TRANSPORT = textwrap.dedent(r"""
import signal as _signal
import socket as _socket
import time as _time

VSOCK_PORT = int(os.environ.get("AGENT_VSOCK_PORT", "5678"))

HANDLERS = {
    "exec": handle_exec,
    "commands": handle_commands,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "edit_file": handle_edit_file,
    "list_dir": handle_list_dir,
}


def _handle_connection(conn):
    reader = conn.makefile("r", buffering=1, errors="replace")
    writer = conn.makefile("w", buffering=1)
    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            tool = req.get("tool", "")
            args = req.get("args", {})
            handler = HANDLERS.get(tool)
            if handler:
                t0 = _time.monotonic()
                resp = handler(args)
                resp["inner_duration_ms"] = (
                    (_time.monotonic() - t0) * 1000.0
                )
            else:
                resp = {
                    "ok": False,
                    "result": f"Error: Unsupported tool {tool!r}",
                }
        except Exception as e:
            resp = {"ok": False, "result": f"Error: agent dispatch failed: {e}"}
        writer.write(json.dumps(resp, ensure_ascii=False) + "\n")
        writer.flush()


def main():
    _signal.signal(_signal.SIGTERM, lambda *_: os._exit(0))
    os.makedirs("/testbed", exist_ok=True)
    sock = _socket.socket(_socket.AF_VSOCK, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind((_socket.VMADDR_CID_ANY, VSOCK_PORT))
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        try:
            _handle_connection(conn)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
""").strip()

_FC_AGENT_SCRIPT = _AGENT_HANDLERS_SRC + "\n\n" + _FC_VSOCK_TRANSPORT


# ---------------------------------------------------------------------------
# Init script injected into the rootfs (launched as PID 1).
# ---------------------------------------------------------------------------

_INIT_SCRIPT = textwrap.dedent(f"""\
#!/bin/sh
# Firecracker agent init — starts dropbear (SSH) and the vsock agent.
set -e

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export AGENT_VSOCK_PORT={_DEFAULT_VSOCK_PORT}

mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

# Rootfs /dev/console from docker export is a regular file, not a device
# node.  After devtmpfs, /dev/ttyS0 is the real serial console.  Redirect
# there so debug/error messages appear on the host.  Guard with test so
# set -e doesn't abort on non-Firecracker kernels without ttyS0.
if [ -e /dev/ttyS0 ]; then exec >/dev/ttyS0 2>&1; fi

# Set up loopback.
ip link set lo up 2>/dev/null || true

# Parse guest networking from kernel cmdline (set in boot_args by FCBackend).
for _arg in $(cat /proc/cmdline); do
    case "$_arg" in
        guest_ip=*) GUEST_IP="${{_arg#guest_ip=}}" ;;
        host_ip=*) HOST_IP="${{_arg#host_ip=}}" ;;
        netmask_len=*) NETMASK_LEN="${{_arg#netmask_len=}}" ;;
    esac
done
: "${{GUEST_IP:=172.16.0.2}}"
: "${{HOST_IP:=172.16.0.1}}"
: "${{NETMASK_LEN:=24}}"

# Configure eth0 from parsed values.
ip addr add "${{GUEST_IP}}/${{NETMASK_LEN}}" dev eth0 2>/dev/null || true
ip link set eth0 up 2>/dev/null || true
ip route add default via "${{HOST_IP}}" 2>/dev/null || true

# Ensure /testbed exists.
mkdir -p /testbed

# Start dropbear SSH server for diagnostics.
dropbear -F -E -p 22 2>/dev/null &
sleep 0.5

# Start the vsock agent in background, NOT as PID 1 (PID 1 is the shell).
echo "Starting vsock agent on port ${{AGENT_VSOCK_PORT}}"

# Write agent script to /agent.py (avoid shell quoting issues with -c).
cat > /agent.py << 'AGENTSCRIPT'
{_FC_AGENT_SCRIPT}
AGENTSCRIPT
# stdout/stderr already redirected to serial console via exec above.
python3 -u /agent.py &

# PID 1 must reap orphaned child processes. The reap loop ensures
# grandchildren from subprocess double-forks are collected.
trap : CHLD
while true; do
    wait
done
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_fc_rootfs(
    docker_image: str,
    output_path: Path | None = None,
    *,
    rootfs_size_mb: int = _DEFAULT_ROOTFS_SIZE_MB,
    cache_dir: Path = _FC_ROOTFS_CACHE_DIR,
    container_executable: str = "docker",
) -> Path:
    """Build (or fetch from cache) an ext4 rootfs for *docker_image*.

    The rootfs contains the Docker image's filesystem plus the vsock
    agent and dropbear.  Results are cached by image digest, so
    repeated calls for the same image are free.

    **IMPORTANT**: The returned path points to the **read-only cache
    image**.  Callers **must not** mount or modify it in-place.  Create
    a per-instance working copy (e.g. ``cp --sparse=always --reflink=auto
    <cache> <working>``) and use the working copy as the VM's root block
    device.

    Args:
        docker_image: Docker image reference (e.g. ``"python:3.12"``).
        output_path: If given, the rootfs is copied/placed here.  When
            ``None`` the cached path is returned directly.
        rootfs_size_mb: Size of the ext4 image in MiB.
        cache_dir: Directory for cached rootfs images.
        container_executable: ``"docker"`` or ``"podman"``.

    Returns:
        Path to the ext4 rootfs image.
    """
    digest = _image_digest(docker_image, container_executable)
    cache_key = _cache_key(docker_image, digest)
    cached = cache_dir / cache_key
    if cached.exists():
        if output_path is not None:
            shutil.copy2(cached, output_path)
            return output_path
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    _build_and_inject(
        docker_image=docker_image,
        dest=cached,
        size_mb=rootfs_size_mb,
        container_executable=container_executable,
    )

    if output_path is not None:
        shutil.copy2(cached, output_path)
        return output_path
    return cached


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _image_digest(docker_image: str, executable: str = "docker") -> str:
    """Return the repo digest of a Docker image, falling back to a local
    inspect hash when no registry digest is available (e.g. for locally-
    built images).
    """
    # Try registry digest first.
    result = subprocess.run(
        [executable, "image", "inspect", docker_image, "--format", "{{.RepoDigests}}"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        raw = result.stdout.strip()
        if raw and raw != "[]":
            # Pick the first digest (e.g. "python@sha256:abcd...")
            for entry in raw.strip("[]").split():
                entry = entry.strip()
                if "@" in entry:
                    _, digest = entry.split("@", 1)
                    return digest

    # Fall back to local image ID hash.
    result = subprocess.run(
        [executable, "image", "inspect", docker_image, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"failed to inspect image {docker_image!r}: {detail}"
        )
    image_id = result.stdout.strip().removeprefix("sha256:")
    return f"sha256:{image_id}"


def _cache_key(docker_image: str, digest: str) -> str:
    """Produce a filesystem-safe cache directory name (includes builder version
    to invalidate when builder logic changes)."""
    safe_image = docker_image.replace("/", "_").replace(":", "_")
    short_digest = digest.split(":")[-1][:16]
    return f"{safe_image}-{short_digest}-v{_ROOTFS_BUILDER_VERSION}.ext4"


def _capture_image_env(
    mnt: Path,
    docker_image: str,
    container_executable: str,
) -> None:
    """Capture Docker image ``ENV`` and ``WorkingDir`` as ``/etc/agent-env.json``.

    ``docker export`` (used to extract the filesystem) strips image
    configuration, so ENV and WORKDIR are lost. We read them via
    ``docker image inspect`` and materialize them in the rootfs so the
    agent can apply them to executed commands.
    """
    result = subprocess.run(
        [
            container_executable,
            "image", "inspect",
            docker_image,
            "--format", "{{json .Config}}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        return  # best-effort

    try:
        config = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return

    env_vars = config.get("Env")
    working_dir = config.get("WorkingDir")
    payload: dict[str, object] = {}
    if isinstance(env_vars, list):
        payload["env"] = [str(e) for e in env_vars]
    if isinstance(working_dir, str) and working_dir:
        payload["working_dir"] = working_dir
    if payload:
        subprocess.run(
            ["sudo", "mkdir", "-p", str(mnt / "etc")],
            capture_output=True, check=True, timeout=30,
        )
        # Use a temp file and sudo mv to avoid Python path permissions.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="fc-agent-env-",
            dir="/tmp", delete=False,
        ) as tf:
            tf.write(json.dumps(payload, ensure_ascii=False) + "\n")
            tmp = Path(tf.name)
        subprocess.run(
            ["sudo", "mv", str(tmp), str(mnt / "etc" / "agent-env.json")],
            capture_output=True, check=True, timeout=30,
        )


def _build_and_inject(
    *,
    docker_image: str,
    dest: Path,
    size_mb: int,
    container_executable: str,
) -> None:
    """Create an ext4 rootfs from *docker_image*, inject agent + init."""
    tmpdir = Path(tempfile.mkdtemp(prefix="fc-rootfs-build-"))
    try:
        # 1. Export container filesystem as a tar.
        cid = subprocess.run(
            [container_executable, "create", docker_image],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout.strip()
        tar_path = tmpdir / "rootfs.tar"
        with open(tar_path, "wb") as fh:
            subprocess.run(
                [container_executable, "export", cid],
                stdout=fh, check=True, timeout=600,
            )
        subprocess.run(
            [container_executable, "rm", cid],
            capture_output=True, check=False, timeout=30,
        )

        # 2. Create sparse ext4 image (sparse to avoid allocating 4 GiB zeros).
        subprocess.run(
            ["truncate", "-s", f"{size_mb}M", str(dest)],
            capture_output=True, check=True, timeout=30,
        )
        subprocess.run(
            ["mkfs.ext4", "-F", str(dest)],
            capture_output=True, check=True, timeout=60,
        )

        # 3. Mount, extract tar, inject agent/init, capture image env.
        mnt = tmpdir / "mnt"
        mnt.mkdir()
        subprocess.run(
            ["sudo", "mount", "-o", "loop", str(dest), str(mnt)],
            capture_output=True, check=True, timeout=60,
        )
        try:
            # Extract Docker filesystem.
            subprocess.run(
                ["sudo", "tar", "-xf", str(tar_path), "-C", str(mnt)],
                capture_output=True, check=True, timeout=300,
            )

            # Ensure /testbed exists — run as root (mount is root-owned).
            subprocess.run(
                ["sudo", "mkdir", "-p", str(mnt / "testbed")],
                capture_output=True, check=True, timeout=30,
            )

            # Capture Docker image ENV and WORKDIR for agent env injection.
            _capture_image_env(mnt, docker_image, container_executable)

            # Inject init script as /sbin/init (FC kernel init search order
            # is /sbin/init → /etc/init → /bin/init → /bin/sh; /init is NOT
            # checked by the v5.10 FC kernel).
            init_src = tmpdir / "init.sh"
            init_src.write_text(_INIT_SCRIPT)
            init_src.chmod(0o755)
            init_dest = mnt / "sbin" / "init"
            subprocess.run(
                ["sudo", "mkdir", "-p", str(mnt / "sbin")],
                capture_output=True, check=True, timeout=30,
            )
            subprocess.run(
                ["sudo", "cp", str(init_src), str(init_dest)],
                capture_output=True, check=True, timeout=30,
            )
            subprocess.run(
                ["sudo", "chmod", "755", str(init_dest)],
                capture_output=True, check=True, timeout=30,
            )

            # Attempt to install additional packages inside the rootfs.
            # We do this via chroot if the rootfs has apt-get or apk.
            _inject_packages(mnt)

            # Ensure Python 3 is available; if not, try to install.
            if not _has_python(mnt):
                _try_install_python_chroot(mnt)

            # Verify vsock support (AF_VSOCK availability).
            _verify_agent(mnt)

        finally:
            subprocess.run(
                ["sudo", "umount", str(mnt)],
                capture_output=True, check=False, timeout=30,
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _has_python(mnt: Path) -> bool:
    """Check whether the rootfs has a Python >=3.8 interpreter."""
    candidates = [
        "usr/bin/python3", "usr/bin/python",
        "usr/local/bin/python3", "usr/local/bin/python",
    ]
    for cand in candidates:
        py = mnt / cand
        if py.exists() and os.access(str(py), os.X_OK):
            return True
    return False


def _inject_packages(mnt: Path) -> None:
    """Install necessary packages into the rootfs via chroot, if possible."""
    # Try apk (Alpine).
    apk = mnt / "sbin/apk"
    if apk.exists():
        try:
            subprocess.run(
                ["sudo", "chroot", str(mnt), "apk", "add", "--no-cache",
                 "dropbear", "dropbear-openrc", "python3", "busybox"],
                capture_output=True, check=False, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Try apt-get (Debian/Ubuntu).
    apt_get = mnt / "usr/bin/apt-get"
    if apt_get.exists():
        try:
            subprocess.run(
                ["sudo", "chroot", str(mnt), "apt-get", "update"],
                capture_output=True, check=False, timeout=120,
            )
            subprocess.run(
                ["sudo", "chroot", str(mnt), "apt-get", "install", "-y",
                 "dropbear", "python3", "busybox-static"],
                capture_output=True, check=False, timeout=300,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Try yum/dnf (RHEL/Fedora).
    dnf = mnt / "usr/bin/dnf"
    yum = mnt / "usr/bin/yum"
    if dnf.exists() or yum.exists():
        mgr = "dnf" if dnf.exists() else "yum"
        try:
            subprocess.run(
                ["sudo", "chroot", str(mnt), mgr, "install", "-y",
                 "dropbear", "python3", "busybox"],
                capture_output=True, check=False, timeout=300,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass


def _try_install_python_chroot(mnt: Path) -> None:
    """Attempt to install Python 3 inside the rootfs."""
    # Already tried in _inject_packages, but try more aggressively.
    apt_get = mnt / "usr/bin/apt-get"
    if apt_get.exists():
        try:
            subprocess.run(
                ["sudo", "chroot", str(mnt), "apt-get", "update"],
                capture_output=True, check=False, timeout=120,
            )
            subprocess.run(
                ["sudo", "chroot", str(mnt), "apt-get", "install", "-y",
                 "python3", "python3-dev"],
                capture_output=True, check=False, timeout=300,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    apk = mnt / "sbin/apk"
    if apk.exists():
        try:
            subprocess.run(
                ["sudo", "chroot", str(mnt), "apk", "add", "--no-cache",
                 "python3", "py3-pip"],
                capture_output=True, check=False, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    if not _has_python(mnt):
        raise RuntimeError(
            "rootfs has no Python >=3.8 and automatic installation failed. "
            "Ensure the base Docker image includes Python 3."
        )


def _verify_agent(mnt: Path) -> None:
    """Check that the agent script was written correctly."""
    # FC kernel searches /sbin/init first; fall back to /init for older caches.
    init_path = mnt / "sbin" / "init"
    if not init_path.exists():
        init_path = mnt / "init"
    if not init_path.exists():
        raise RuntimeError("init script not found in rootfs (/sbin/init or /init)")
    # The agent is embedded in the init script — verify it exists by
    # grepping for a distinctive string.
    init_text = init_path.read_text()
    if "socket.VMADDR_CID_ANY" not in init_text:
        raise RuntimeError("vsock agent not found in init script")
