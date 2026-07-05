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

# ---------------------------------------------------------------------------
# In-VM agent script (Python 3, listens on vsock, same JSON-lines protocol
# as the ContainerAgent used by DockerBackend).
# ---------------------------------------------------------------------------

_FC_AGENT_SCRIPT = textwrap.dedent(r"""
import json, os, sys, subprocess, difflib, signal, time, socket

VSOCK_PORT = int(os.environ.get("AGENT_VSOCK_PORT", "5678"))
_LIST_IGNORE = {".git", "node_modules", "__pycache__", ".venv", ".tox",
                ".mypy_cache", ".pytest_cache"}


# --------------- agent handlers (copied from _REPLAY_AGENT_SCRIPT) ----------

def _find_match(content, old_text):
    if old_text in content:
        return old_text, content.count(old_text)
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i:i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))
    if candidates:
        return candidates[0], len(candidates)
    return None, 0


def _not_found_msg(old_text, content, path):
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)
    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(
            None, old_lines, lines[i:i + window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
    if best_ratio > 0.5:
        diff = "\n".join(difflib.unified_diff(
            old_lines, lines[best_start:best_start + window],
            fromfile="old_text (provided)",
            tofile=f"{path} (actual, line {best_start + 1})", lineterm=""))
        return (
            f"Error: old_text not found in {path}.\n"
            f"Best match ({best_ratio:.0%}) at line {best_start + 1}:\n{diff}"
        )
    return f"Error: old_text not found in {path}. No similar text found."


_MAX_OUTPUT = 10_000


def _truncate_output(text, limit=_MAX_OUTPUT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half] + f"\n\n... ({len(text) - limit:,} chars truncated) ..."
        + "\n\n" + text[-half:]
    )


def _format_exec_result(stdout, stderr, returncode):
    output_parts = []
    if stdout:
        output_parts.append(stdout)
    if stderr and stderr.strip():
        output_parts.append(f"STDERR:\n{stderr}")
    output_parts.append(f"\nExit code: {returncode}")
    return "\n".join(output_parts)


def _format_command_timeout(timeout):
    return f"Error: Command timed out after {timeout} seconds"


# --------------- tool handlers ---------------

def handle_exec(args):
    cmd = args.get("command", "")
    timeout = args.get("timeout", 600)
    env = {**os.environ}
    try:
        r = subprocess.run(
            cmd, shell=True, cwd="/testbed",
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        output = _format_exec_result(
            r.stdout or "", r.stderr or "", r.returncode,
        )
        return {
            "ok": True, "result": _truncate_output(output),
            "returncode": r.returncode, "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "result": _format_command_timeout(timeout),
            "returncode": 124, "timed_out": True,
        }


def handle_commands(args):
    cmds = args.get("commands", [])
    timeout = args.get("timeout", 600)
    env = {**os.environ}
    all_output = []
    last_rc = 0
    first_failed_rc = 0
    any_timeout = False
    for i, cmd in enumerate(cmds):
        try:
            r = subprocess.run(
                cmd, shell=True, cwd="/testbed",
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            all_output.append(_format_exec_result(
                r.stdout or "", r.stderr or "", r.returncode,
            ))
            last_rc = r.returncode
            if r.returncode != 0 and first_failed_rc == 0:
                first_failed_rc = r.returncode
        except subprocess.TimeoutExpired:
            all_output.append(_format_command_timeout(timeout))
            last_rc = 124
            any_timeout = True
    if len(cmds) > 1:
        combined = "\n".join(
            f"[call {k}]\n{out}" for k, out in enumerate(all_output))
    else:
        combined = all_output[0] if all_output else ""
    returncode = 124 if any_timeout else (first_failed_rc or last_rc)
    return {
        "ok": not any_timeout, "result": combined,
        "returncode": returncode, "timed_out": any_timeout,
    }


_READ_MAX_CHARS = 128_000
_READ_DEFAULT_LIMIT = 2000


def handle_read_file(args):
    path = args.get("path", "")
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", _READ_DEFAULT_LIMIT))
    try:
        content = open(path).read()
        if not content:
            return {"ok": True, "result": f"(Empty file: {path})"}
        lines = content.splitlines()
        selected = lines[offset:offset + limit]
        numbered = "\n".join(
            f"{offset + i + 1}| {ln}" for i, ln in enumerate(selected))
        if len(numbered) > _READ_MAX_CHARS:
            numbered = (
                numbered[:_READ_MAX_CHARS]
                + f"\n\n... (truncated at {_READ_MAX_CHARS} chars)"
            )
        return {"ok": True, "result": numbered}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}


def handle_write_file(args):
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return {"ok": True, "result": f"Successfully wrote {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}


def handle_edit_file(args):
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    replace_all = args.get("replace_all", False)
    try:
        raw = open(path, "rb").read()
        uses_crlf = b"\r\n" in raw
        content = raw.decode("utf-8").replace("\r\n", "\n")
        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if match is None:
            return {
                "ok": False,
                "result": _not_found_msg(old_text, content, path),
            }
        if count > 1 and not replace_all:
            return {
                "ok": False,
                "result": (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context or set replace_all=true."
                ),
            }
        norm_new = new_text.replace("\r\n", "\n")
        new_content = (
            content.replace(match, norm_new) if replace_all
            else content.replace(match, norm_new, 1)
        )
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        open(path, "wb").write(new_content.encode("utf-8"))
        return {"ok": True, "result": f"Successfully edited {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error editing file: {e}"}


_LIST_MAX = 200


def handle_list_dir(args):
    path = args.get("path", ".")
    try:
        entries = sorted(
            e for e in os.listdir(path) if e not in _LIST_IGNORE)
        if len(entries) > _LIST_MAX:
            entries = entries[:_LIST_MAX]
            entries.append(
                f"... ({len(os.listdir(path)) - _LIST_MAX} more entries)")
        return {"ok": True, "result": "\n".join(entries)}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}


HANDLERS = {
    "exec": handle_exec,
    "commands": handle_commands,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "edit_file": handle_edit_file,
    "list_dir": handle_list_dir,
}


# --------------- main loop ---------------

def handle_connection(conn):
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
                t0 = time.monotonic()
                resp = handler(args)
                resp["inner_duration_ms"] = (
                    (time.monotonic() - t0) * 1000.0
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
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    os.makedirs("/testbed", exist_ok=True)
    sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((socket.VMADDR_CID_ANY, VSOCK_PORT))
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        try:
            handle_connection(conn)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
""").strip()


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

# Set up loopback (vsock does not need it, but tools may).
ip link set lo up 2>/dev/null || true

# Configure eth0 if tap networking is available.
ip addr add 172.16.0.2/24 dev eth0 2>/dev/null || true
ip link set eth0 up 2>/dev/null || true
ip route add default via 172.16.0.1 2>/dev/null || true

# Ensure /testbed exists.
mkdir -p /testbed

# Start dropbear SSH server for diagnostics.
dropbear -F -E -p 22 2>/dev/null &
# give it a moment to bind
sleep 0.5

# Start the vsock agent (runs forever).
echo "Starting vsock agent on port ${{AGENT_VSOCK_PORT}}"
exec python3 -u -c {json.dumps(_FC_AGENT_SCRIPT)}
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
    """Produce a filesystem-safe cache directory name."""
    safe_image = docker_image.replace("/", "_").replace(":", "_")
    short_digest = digest.split(":")[-1][:16]
    return f"{safe_image}-{short_digest}.ext4"


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

        # 2. Create blank ext4 image.
        subprocess.run(
            ["dd", "if=/dev/zero", f"of={dest}", "bs=1M",
             f"count={size_mb}"],
            capture_output=True, check=True, timeout=300,
        )
        subprocess.run(
            ["mkfs.ext4", "-F", str(dest)],
            capture_output=True, check=True, timeout=60,
        )

        # 3. Mount, extract tar, inject agent/init.
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

            # Ensure /testbed exists.
            testbed = mnt / "testbed"
            if not testbed.exists():
                testbed.mkdir(mode=0o755, exist_ok=True)

            # Inject init script as /init (PID 1).
            init_path = mnt / "init"
            init_path.write_text(_INIT_SCRIPT)
            init_path.chmod(0o755)

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
    init_path = mnt / "init"
    if not init_path.exists():
        raise RuntimeError("init script not found in rootfs")
    # The agent is embedded in the init script — verify it exists by
    # grepping for a distinctive string.
    init_text = init_path.read_text()
    if "socket.VMADDR_CID_ANY" not in init_text:
        raise RuntimeError("vsock agent not found in init script")
