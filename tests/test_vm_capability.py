from __future__ import annotations

from pathlib import Path

from trace_collect.runtime import vm_capability
from trace_collect.runtime.vm_capability import (
    normalize_arch,
    probe_firecracker_capability,
)


def test_normalize_arch_handles_common_labels() -> None:
    assert normalize_arch("x86_64") == "amd64"
    assert normalize_arch("amd64") == "amd64"
    assert normalize_arch("aarch64") == "arm64"
    assert normalize_arch("arm64/v8") == "arm64"
    assert normalize_arch(None) == "unknown"


def test_firecracker_probe_flags_cross_arch_image(
    monkeypatch,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "vmlinux"
    rootfs = tmp_path / "rootfs.ext4"
    kvm = tmp_path / "kvm"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")
    kvm.write_bytes(b"")

    monkeypatch.setattr(vm_capability.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        vm_capability,
        "_probe_kvm_access",
        lambda _path: (True, None),
    )
    monkeypatch.setattr(vm_capability.shutil, "which", lambda _bin: "/usr/bin/firecracker")
    monkeypatch.setattr(
        vm_capability,
        "_probe_firecracker_version",
        lambda _path: "Firecracker v1.15.1",
    )
    monkeypatch.setattr(
        vm_capability,
        "_probe_container_server",
        lambda _exe: ("aarch64", "linux"),
    )
    monkeypatch.setattr(
        vm_capability,
        "_probe_image_arch",
        lambda _image, *, container_executable: ("amd64", "linux"),
    )

    capability = probe_firecracker_capability(
        image="swerebench/sweb.eval.x86_64.example",
        kernel_path=kernel,
        rootfs_path=rootfs,
        kvm_device=kvm,
    )

    assert capability.can_run_same_arch_microvm is True
    assert capability.can_run_image_microvm is False
    assert capability.same_arch_image is False
    assert capability.unavailable_reasons == ["guest_arch_mismatch"]


def test_firecracker_probe_records_kvm_permission_denied(
    monkeypatch,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "vmlinux"
    rootfs = tmp_path / "rootfs.ext4"
    kvm = tmp_path / "kvm"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")
    kvm.write_bytes(b"")

    monkeypatch.setattr(vm_capability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        vm_capability,
        "_probe_kvm_access",
        lambda _path: (False, "PermissionError: Permission denied"),
    )
    monkeypatch.setattr(vm_capability.shutil, "which", lambda _bin: "/usr/bin/firecracker")
    monkeypatch.setattr(
        vm_capability,
        "_probe_firecracker_version",
        lambda _path: "Firecracker v1.15.1",
    )
    monkeypatch.setattr(
        vm_capability,
        "_probe_container_server",
        lambda _exe: ("amd64", "linux"),
    )

    capability = probe_firecracker_capability(
        kernel_path=kernel,
        rootfs_path=rootfs,
        kvm_device=kvm,
    )

    assert capability.can_run_same_arch_microvm is False
    assert capability.kvm_open_error == "PermissionError: Permission denied"
    assert "kvm_permission_denied" in capability.unavailable_reasons


def test_firecracker_probe_allows_same_arch_image(
    monkeypatch,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "vmlinux"
    rootfs = tmp_path / "rootfs.ext4"
    kvm = tmp_path / "kvm"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")
    kvm.write_bytes(b"")

    monkeypatch.setattr(vm_capability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        vm_capability,
        "_probe_kvm_access",
        lambda _path: (True, None),
    )
    monkeypatch.setattr(vm_capability.shutil, "which", lambda _bin: "/usr/bin/firecracker")
    monkeypatch.setattr(
        vm_capability,
        "_probe_firecracker_version",
        lambda _path: "Firecracker v1.15.1",
    )
    monkeypatch.setattr(
        vm_capability,
        "_probe_container_server",
        lambda _exe: ("amd64", "linux"),
    )
    monkeypatch.setattr(
        vm_capability,
        "_probe_image_arch",
        lambda _image, *, container_executable: ("amd64", "linux"),
    )

    capability = probe_firecracker_capability(
        image="example/task:latest",
        kernel_path=kernel,
        rootfs_path=rootfs,
        kvm_device=kvm,
    )

    assert capability.can_run_same_arch_microvm is True
    assert capability.can_run_image_microvm is True
    assert capability.unavailable_reasons == []
