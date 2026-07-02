"""Tests for trace_collect CLI argument parsing."""

from __future__ import annotations

import pytest

from trace_collect.cli import parse_collect_args, parse_vm_probe_args


def test_parse_collect_args_accepts_skip_and_concurrency() -> None:
    args = parse_collect_args([
        "--provider",
        "openrouter",
        "--model",
        "z-ai/glm-5.1",
        "--skip",
        "7",
        "--concurrency",
        "3",
    ])

    assert args.skip == 7
    assert args.concurrency == 3


def test_parse_collect_args_rejects_negative_skip() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--skip",
            "-1",
        ])


def test_parse_collect_args_rejects_negative_sample() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--sample",
            "-1",
        ])


def test_parse_collect_args_rejects_zero_concurrency() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--concurrency",
            "0",
        ])


def test_parse_vm_probe_args_accepts_image_and_runtime_paths() -> None:
    args = parse_vm_probe_args([
        "--image",
        "example/task:latest",
        "--container",
        "docker",
        "--firecracker-bin",
        "/usr/local/bin/firecracker",
        "--kernel",
        "/opt/vmlinux",
        "--rootfs",
        "/opt/rootfs.ext4",
    ])

    assert args.image == "example/task:latest"
    assert args.container == "docker"
    assert args.firecracker_bin == "/usr/local/bin/firecracker"
    assert args.kernel == "/opt/vmlinux"
    assert args.rootfs == "/opt/rootfs.ext4"
