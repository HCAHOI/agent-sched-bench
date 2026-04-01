"""Tests for ContainerManager (Podman lifecycle management).

These tests verify the logic without requiring a running Podman daemon.
Integration tests that actually create containers are skipped unless
podman is available.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from agents.container_manager import ContainerManager


def _has_podman() -> bool:
    """Check if podman is available on the system."""
    return shutil.which("podman") is not None


def _has_swebench_image() -> bool:
    """Check if the swebench-base image exists."""
    if not _has_podman():
        return False
    result = subprocess.run(
        ["podman", "image", "exists", "swebench-base:latest"],
        capture_output=True,
    )
    return result.returncode == 0


class TestContainerManagerUnit:
    """Unit tests that don't require podman."""

    def test_container_name_derivation(self) -> None:
        mgr = ContainerManager()
        task = {"instance_id": "django__django-12345", "repo": "django/django"}
        name = mgr._container_name(task)
        assert name == "swebench-django-django-12345"
        # No slashes or double underscores
        assert "/" not in name

    def test_repo_dir_name(self) -> None:
        mgr = ContainerManager()
        task = {"repo": "scikit-learn/scikit-learn"}
        assert mgr._repo_dir_name(task) == "scikit-learn__scikit-learn"

    def test_init_without_repos_root(self) -> None:
        mgr = ContainerManager(base_image="test:latest")
        assert mgr.base_image == "test:latest"
        assert mgr.repos_root is None

    def test_init_with_repos_root(self, tmp_path) -> None:
        mgr = ContainerManager(
            base_image="test:latest",
            repos_root=tmp_path,
        )
        assert mgr.repos_root == tmp_path.resolve()


@pytest.mark.skipif(not _has_podman(), reason="podman not available")
@pytest.mark.skipif(not _has_swebench_image(), reason="swebench-base image not built")
class TestContainerManagerIntegration:
    """Integration tests that require podman + swebench-base image."""

    def test_create_exec_destroy_lifecycle(self) -> None:
        """Full lifecycle: create → exec → destroy."""
        mgr = ContainerManager(base_image="swebench-base:latest")
        task = {
            "instance_id": "test__lifecycle-001",
            "repo": "psf/requests",
            "base_commit": "main",
        }

        async def _run() -> None:
            container_id = await mgr.create_container(task)
            try:
                returncode, output = await mgr.exec_in_container(
                    container_id, "echo hello",
                )
                assert returncode == 0
                assert "hello" in output
            finally:
                await mgr.destroy_container(container_id)

        asyncio.run(_run())

    def test_exec_returns_nonzero_on_failure(self) -> None:
        mgr = ContainerManager(base_image="swebench-base:latest")

        async def _run() -> None:
            # Create a minimal container
            result = await mgr._run_podman([
                "run", "-d", "--name", "swebench-test-fail",
                "swebench-base:latest", "sleep", "infinity",
            ])
            assert result.returncode == 0
            try:
                returncode, output = await mgr.exec_in_container(
                    "swebench-test-fail", "exit 42",
                )
                assert returncode == 42
            finally:
                await mgr.destroy_container("swebench-test-fail")

        asyncio.run(_run())
