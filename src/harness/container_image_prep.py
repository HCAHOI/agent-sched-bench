"""Helpers for preparing writable derivative container images.

The derivative keeps the upstream image content while making ``/testbed``
writeable for agent runs.
"""

from __future__ import annotations

import os
import subprocess
import time

_IMAGE_CACHE: dict[str, tuple[str, float]] = {}

def _image_slug(source_image: str) -> str:
    return source_image.replace("/", "_").replace(":", "_").replace("@", "_")


def normalize_image_reference(image: str) -> str:
    """Return a Podman-safe fully qualified image reference when possible."""
    if not image:
        return ""
    if "/" not in image:
        return f"docker.io/library/{image}"
    head = image.split("/", 1)[0]
    if "." in head or ":" in head or head == "localhost":
        return image
    return f"docker.io/{image}"


def fixed_image_name_for(source_image: str) -> str:
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


def ensure_source_image(
    source_image: str,
    *,
    executable: str = "podman",
) -> None:
    """Ensure ``source_image`` exists locally, pulling when missing."""
    source_image = normalize_image_reference(source_image)
    if not source_image:
        return
    if _image_exists(source_image, executable):
        return
    try:
        _run(
            [executable, "pull", source_image],
            check=True,
            timeout=3600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Failed to pull source image {source_image}: {exc}"
        ) from exc


def remove_image(
    image: str,
    *,
    executable: str = "podman",
    normalize: bool = False,
) -> bool:
    """Best-effort local image removal.

    Returns ``True`` when an image existed and was removed, ``False`` when the
    image was already absent.
    """
    if normalize:
        image = normalize_image_reference(image)
    if not image or not _image_exists(image, executable):
        return False
    result = _run(
        [executable, "image", "rm", "-f", image],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to remove image {image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return True


def prune_dangling_images(*, executable: str = "podman") -> None:
    """Best-effort prune of dangling image layers."""
    result = _run(
        [executable, "image", "prune", "-f"],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to prune dangling images: "
            f"{result.stderr.strip() or result.stdout.strip()}"
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
    source_image = normalize_image_reference(source_image)
    if source_image in _IMAGE_CACHE:
        return _IMAGE_CACHE[source_image]

    fixed_name = fixed_image_name_for(source_image)

    if _image_exists(fixed_name, executable):
        _IMAGE_CACHE[source_image] = (fixed_name, 0.0)
        return _IMAGE_CACHE[source_image]

    ensure_source_image(source_image, executable=executable)

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
    _IMAGE_CACHE.clear()


def drop_cached_fixed_image(source_image: str) -> None:
    """Forget any cached fixed-image lookup for ``source_image``."""
    _IMAGE_CACHE.pop(normalize_image_reference(source_image), None)
