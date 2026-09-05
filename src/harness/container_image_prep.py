"""Helpers for preparing writable derivative container images.

The derivative keeps the upstream image content while making ``/testbed``
writeable for agent runs.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

from harness.container_runtime import image_exists_command

_IMAGE_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}
_PULL_ATTEMPTS = 3
_PULL_BACKOFF_SECONDS = 1.0
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _image_slug(source_image: str) -> str:
    return source_image.replace("/", "_").replace(":", "_").replace("@", "_")


def normalize_image_reference(image: str) -> str:
    """Return a fully qualified image reference when possible.

    MUST remain idempotent: simulator.py's fixed-vs-source cleanup guards
    rely on normalize_image_reference(out) == out.
    """
    if not image:
        return ""
    if _IMAGE_ID_RE.fullmatch(image):
        return image
    registry_prefix = os.environ.get("TASK_CONTAINER_IMAGE_REGISTRY_PREFIX", "").strip()
    if registry_prefix:
        registry_prefix = registry_prefix.rstrip("/")
    if "/" not in image:
        if registry_prefix:
            return f"{registry_prefix}/library/{image}"
        return f"docker.io/library/{image}"
    head = image.split("/", 1)[0]
    if "." in head or ":" in head or head == "localhost":
        return image
    if registry_prefix:
        return f"{registry_prefix}/{image}"
    return f"docker.io/{image}"


def fixed_image_name_for(source_image: str) -> str:
    return f"swebench-fixed-{_image_slug(source_image)}"


def _image_exists(image: str, executable: str) -> bool:
    result = subprocess.run(
        image_exists_command(image, container_executable=executable),
        capture_output=True,
        text=True,
        check=False,
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


def _is_retryable_pull_failure(text: str) -> bool:
    lowered = text.lower()
    retryable_markers = (
        "eof",
        "unexpected eof",
        "connection reset",
        "i/o timeout",
        "tls handshake timeout",
        "context deadline exceeded",
        "temporarily unavailable",
    )
    return any(marker in lowered for marker in retryable_markers)


def _pull_source_image(image: str, executable: str) -> None:
    last_error: str | None = None
    for attempt in range(1, _PULL_ATTEMPTS + 1):
        result = _run(
            [executable, "pull", image],
            check=False,
            timeout=3600,
        )
        if result.returncode == 0:
            return
        output = (result.stderr or result.stdout or "").strip()
        last_error = output or f"exit code {result.returncode}"
        if attempt >= _PULL_ATTEMPTS or not _is_retryable_pull_failure(last_error):
            raise RuntimeError(f"Failed to pull source image {image}: {last_error}")
        time.sleep(_PULL_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    if last_error is not None:
        raise RuntimeError(f"Failed to pull source image {image}: {last_error}")


def ensure_source_image(
    source_image: str,
    *,
    container_executable: str,
) -> None:
    """Ensure ``source_image`` exists locally, pulling when missing."""
    source_image = normalize_image_reference(source_image)
    if not source_image:
        return
    if _image_exists(source_image, container_executable):
        return
    if _IMAGE_ID_RE.fullmatch(source_image):
        raise RuntimeError(
            "Recorded immutable source image is unavailable locally: "
            f"{source_image}. Replay cannot pull an image ID; restore the "
            "collection image or provide an explicit image override."
        )
    _pull_source_image(source_image, container_executable)


def remove_image(
    image: str,
    *,
    container_executable: str,
    normalize: bool = False,
) -> bool:
    """Best-effort local image removal.

    Returns ``True`` when an image existed and was removed, ``False`` when the
    image was already absent.
    """
    if normalize:
        image = normalize_image_reference(image)
    if not image or not _image_exists(image, container_executable):
        return False
    result = _run(
        [container_executable, "image", "rm", "-f", image],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to remove image {image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return True


def prune_dangling_images(*, container_executable: str) -> None:
    """Best-effort prune of dangling image layers."""
    result = _run(
        [container_executable, "image", "prune", "-f"],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to prune dangling images: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def ensure_fixed_image(
    source_image: str,
    *,
    container_executable: str,
    host_uid: int | None = None,
    host_gid: int | None = None,
    fixed_image_name: str | None = None,
    rebuild: bool = False,
) -> tuple[str, float]:
    """Ensure the source image is local; run it DIRECTLY (no fixed derivative).

    The task container runs as root, so ``/testbed`` (root-owned in the swebench
    source image, which already contains the checked-out repo) is writable
    without a chown. The legacy chown-and-``docker commit`` derivative was
    therefore unnecessary, and that ``commit`` failed intermittently under
    concurrency on the overlayfs driver. We keep the pull and drop the build,
    returning the source image unchanged. ``host_uid``/``host_gid``/
    ``fixed_image_name``/``rebuild`` are accepted for call-site compatibility
    and ignored (matches origin/docs/cli-first-drop-verified-current@3da41b0).
    """
    source_image = normalize_image_reference(source_image)
    ensure_source_image(source_image, container_executable=container_executable)
    return source_image, 0.0


def clear_image_cache() -> None:
    # No-op under the passthrough (the cache is never populated); kept for
    # call-site compatibility.
    _IMAGE_CACHE.clear()


def drop_cached_fixed_image(source_image: str) -> None:
    """Forget any cached fixed-image lookup for ``source_image``."""
    normalized = normalize_image_reference(source_image)
    for cache_key in list(_IMAGE_CACHE):
        if cache_key[1] == normalized:
            _IMAGE_CACHE.pop(cache_key, None)
