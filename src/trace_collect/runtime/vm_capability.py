"""Capability probes for VM-backed trace replay runtimes.

The probe is intentionally read-only. It records whether a host can run a
same-architecture Firecracker microVM and whether a Docker image matches the
host architecture closely enough to be a candidate for KVM-backed replay.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_KERNEL_CANDIDATES = (
    Path(os.environ.get("FIRECRACKER_KERNEL", ""))
    if os.environ.get("FIRECRACKER_KERNEL")
    else None,
    Path("/opt/ear/firecracker/vmlinux"),
)
_DEFAULT_ROOTFS_CANDIDATES = (
    Path(os.environ.get("FIRECRACKER_ROOTFS", ""))
    if os.environ.get("FIRECRACKER_ROOTFS")
    else None,
    Path("/opt/ear/firecracker/rootfs.ext4"),
)


@dataclass(frozen=True, slots=True)
class FirecrackerCapability:
    """Resolved Firecracker capability for the current host."""

    host_arch: str
    normalized_host_arch: str
    kvm_device: str
    kvm_exists: bool
    kvm_readable_writable: bool
    kvm_open_error: str | None
    firecracker_path: str | None
    firecracker_version: str | None
    kernel_path: str | None
    kernel_exists: bool
    rootfs_path: str | None
    rootfs_exists: bool
    container_executable: str | None
    docker_server_arch: str | None
    docker_server_os: str | None
    image: str | None
    image_arch: str | None
    normalized_image_arch: str | None
    image_os: str | None
    same_arch_image: bool | None
    can_run_same_arch_microvm: bool
    can_run_image_microvm: bool | None
    warning_reasons: list[str] = field(default_factory=list)
    unavailable_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_firecracker_capability(
    *,
    image: str | None = None,
    container_executable: str | None = "docker",
    firecracker_bin: str = "firecracker",
    kernel_path: Path | None = None,
    rootfs_path: Path | None = None,
    kvm_device: Path = Path("/dev/kvm"),
) -> FirecrackerCapability:
    """Probe whether Firecracker is viable for this host and optional image."""
    host_arch = platform.machine()
    normalized_host_arch = normalize_arch(host_arch)
    kvm_exists = kvm_device.exists()
    kvm_readable_writable, kvm_open_error = _probe_kvm_access(kvm_device)
    firecracker_path = shutil.which(firecracker_bin)
    firecracker_version = _probe_firecracker_version(firecracker_path)
    resolved_kernel = kernel_path or _first_existing(_DEFAULT_KERNEL_CANDIDATES)
    resolved_rootfs = rootfs_path or _first_existing(_DEFAULT_ROOTFS_CANDIDATES)

    docker_server_arch: str | None = None
    docker_server_os: str | None = None
    image_arch: str | None = None
    image_os: str | None = None
    normalized_image_arch: str | None = None
    same_arch_image: bool | None = None
    if container_executable:
        docker_server_arch, docker_server_os = _probe_container_server(
            container_executable
        )
        if image:
            image_arch, image_os = _probe_image_arch(
                image,
                container_executable=container_executable,
            )
            if image_arch is not None:
                normalized_image_arch = normalize_arch(image_arch)
                same_arch_image = normalized_image_arch == normalized_host_arch

    warning_reasons: list[str] = []
    if firecracker_path is not None and firecracker_version is None:
        warning_reasons.append("firecracker_version_unavailable")
    if container_executable and docker_server_arch is None:
        warning_reasons.append("container_info_unavailable")
    if image and image_arch is None:
        warning_reasons.append("image_inspect_unavailable")

    unavailable_reasons: list[str] = []
    if not kvm_exists:
        unavailable_reasons.append("kvm_missing")
    elif not kvm_readable_writable:
        unavailable_reasons.append("kvm_permission_denied")
    if firecracker_path is None:
        unavailable_reasons.append("firecracker_missing")
    if resolved_kernel is None or not resolved_kernel.exists():
        unavailable_reasons.append("kernel_missing")
    if resolved_rootfs is None or not resolved_rootfs.exists():
        unavailable_reasons.append("rootfs_missing")

    can_run_same_arch_microvm = not unavailable_reasons
    can_run_image_microvm: bool | None
    if image is None:
        can_run_image_microvm = None
    elif same_arch_image is None:
        can_run_image_microvm = False
        unavailable_reasons.append("image_arch_unknown")
    elif same_arch_image:
        can_run_image_microvm = can_run_same_arch_microvm
    else:
        can_run_image_microvm = False
        unavailable_reasons.append("guest_arch_mismatch")

    return FirecrackerCapability(
        host_arch=host_arch,
        normalized_host_arch=normalized_host_arch,
        kvm_device=str(kvm_device),
        kvm_exists=kvm_exists,
        kvm_readable_writable=kvm_readable_writable,
        kvm_open_error=kvm_open_error,
        firecracker_path=firecracker_path,
        firecracker_version=firecracker_version,
        kernel_path=str(resolved_kernel) if resolved_kernel is not None else None,
        kernel_exists=bool(resolved_kernel and resolved_kernel.exists()),
        rootfs_path=str(resolved_rootfs) if resolved_rootfs is not None else None,
        rootfs_exists=bool(resolved_rootfs and resolved_rootfs.exists()),
        container_executable=container_executable,
        docker_server_arch=docker_server_arch,
        docker_server_os=docker_server_os,
        image=image,
        image_arch=image_arch,
        normalized_image_arch=normalized_image_arch,
        image_os=image_os,
        same_arch_image=same_arch_image,
        can_run_same_arch_microvm=can_run_same_arch_microvm,
        can_run_image_microvm=can_run_image_microvm,
        warning_reasons=warning_reasons,
        unavailable_reasons=unavailable_reasons,
    )


def normalize_arch(value: str | None) -> str:
    """Normalize common Linux/Docker architecture labels."""
    if not value:
        return "unknown"
    arch = value.strip().lower()
    if arch in {"x86_64", "amd64"}:
        return "amd64"
    if arch in {"aarch64", "arm64", "arm64/v8"}:
        return "arm64"
    return arch


def _first_existing(candidates: tuple[Path | None, ...]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _probe_kvm_access(kvm_device: Path) -> tuple[bool, str | None]:
    try:
        fd = os.open(kvm_device, os.O_RDWR)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc.strerror or exc}"
    os.close(fd)
    return True, None


def _probe_firecracker_version(firecracker_path: str | None) -> str | None:
    if firecracker_path is None:
        return None
    result = subprocess.run(
        [firecracker_path, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def _probe_container_server(
    container_executable: str,
) -> tuple[str | None, str | None]:
    result = subprocess.run(
        [container_executable, "info", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return None, None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, None
    return payload.get("Architecture"), payload.get("OSType")


def _probe_image_arch(
    image: str,
    *,
    container_executable: str,
) -> tuple[str | None, str | None]:
    result = subprocess.run(
        [container_executable, "image", "inspect", image, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return None, None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, None
    return payload.get("Architecture"), payload.get("Os")
