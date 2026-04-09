"""Build a writable SWE-rebench derivative image (`swebench-fixed-<slug>`).

This mirrors the Claude Code harness (scripts/run_swebench.py::_fix_permissions
in the agentcgroup reference repo). The upstream CC flow runs the task
container with ``podman run --userns=keep-id`` so the in-container uid matches
the host user; for that to work ``/testbed`` must be ``chown``ed to the host
uid before the agent starts. We produce a cached derivative image exactly
once per source image and reuse it across runs.
"""

from __future__ import annotations

import os
import subprocess
import time


_IMAGE_CACHE: dict[str, tuple[str, float]] = {}


def _image_slug(source_image: str) -> str:
    """Stable filesystem-safe tag suffix matching CC's naming convention."""
    return source_image.replace("/", "_").replace(":", "_").replace("@", "_")


def fixed_image_name_for(source_image: str) -> str:
    """Return the ``swebench-fixed-*`` tag that would be produced for *source_image*."""
    return f"swebench-fixed-{_image_slug(source_image)}"


def _image_exists(image: str, executable: str) -> bool:
    result = subprocess.run(
        [executable, "image", "exists", image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _run(
    cmd: list[str], *, check: bool = True, timeout: int = 180
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _build_fixed_image(
    source_image: str,
    fixed_name: str,
    executable: str,
    uid: int,
    gid: int,
) -> None:
    """Commit a writable derivative with /testbed chowned to ``uid:gid``.

    Implementation mirrors agentcgroup/scripts/run_swebench.py::_fix_permissions.
    """
    start = _run(
        [executable, "run", "-d", source_image, "sleep", "120"],
        check=True,
        timeout=180,
    )
    container_id = start.stdout.strip()
    try:
        _run(
            [
                executable,
                "exec",
                container_id,
                "chown",
                "-R",
                f"{uid}:{gid}",
                "/testbed",
            ],
            check=True,
            timeout=120,
        )
        _run(
            [executable, "commit", container_id, fixed_name],
            check=True,
            timeout=240,
        )
    finally:
        _run([executable, "stop", container_id], check=False, timeout=30)
        _run([executable, "rm", "-f", container_id], check=False, timeout=30)


def ensure_fixed_image(
    source_image: str,
    *,
    executable: str = "podman",
    host_uid: int | None = None,
    host_gid: int | None = None,
) -> tuple[str, float]:
    """Return ``(fixed_image_name, elapsed_seconds)``.

    Reuses a cached derivative image for repeat calls within the same process
    and also skips the build step when the derivative already exists on the
    host (checked via ``podman image exists``). The CC reference always
    builds the fixed image once per source — there is no probe-first path,
    because the container is always launched with ``--userns=keep-id`` and
    always needs the ``/testbed`` ownership fix.
    """
    if source_image in _IMAGE_CACHE:
        return _IMAGE_CACHE[source_image]

    fixed_name = fixed_image_name_for(source_image)

    if _image_exists(fixed_name, executable):
        _IMAGE_CACHE[source_image] = (fixed_name, 0.0)
        return _IMAGE_CACHE[source_image]

    uid = host_uid if host_uid is not None else os.getuid()
    gid = host_gid if host_gid is not None else os.getgid()

    t0 = time.time()
    try:
        _build_fixed_image(source_image, fixed_name, executable, uid, gid)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # Fall back to the source image rather than leave the caller hanging.
        _IMAGE_CACHE[source_image] = (source_image, 0.0)
        raise RuntimeError(
            f"Failed to build fixed derivative image for {source_image}: {exc}"
        ) from exc
    elapsed = time.time() - t0
    _IMAGE_CACHE[source_image] = (fixed_name, elapsed)
    return _IMAGE_CACHE[source_image]


def clear_image_cache() -> None:
    """Test helper — drops the in-process cache."""
    _IMAGE_CACHE.clear()
