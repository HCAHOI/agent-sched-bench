"""Container-runtime-specific command helpers."""

from __future__ import annotations


def image_exists_command(
    image: str,
    *,
    container_executable: str,
) -> list[str]:
    """Return a CLI-specific image existence probe command."""
    if container_executable == "podman":
        return [container_executable, "image", "exists", image]
    return [container_executable, "image", "inspect", image]
