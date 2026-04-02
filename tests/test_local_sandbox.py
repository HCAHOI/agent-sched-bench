"""Tests for LocalSandbox (proot/subprocess-based task isolation).

Unit tests verify logic without requiring any repos or proot.
Integration tests run the full lifecycle against a real local repo clone
and are skipped only when the data/swebench_repos directory is absent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agents.local_sandbox import LocalSandbox

REPO_ROOT = Path(__file__).resolve().parents[1]
REPOS_ROOT = REPO_ROOT / "data" / "swebench_repos"
TASKS_FILE = REPO_ROOT / "data" / "swebench_verified" / "tasks.json"


def _has_local_repos() -> bool:
    return REPOS_ROOT.is_dir() and any(REPOS_ROOT.iterdir())


def _load_task(repo: str = "psf/requests") -> dict:
    """Load the first task for the given repo from tasks.json."""
    tasks = json.loads(TASKS_FILE.read_text())
    return next(t for t in tasks if t["repo"] == repo)


class TestLocalSandboxUnit:
    """Unit tests that don't require repos or proot."""

    def test_sandbox_name_derivation(self) -> None:
        sandbox = LocalSandbox()
        task = {"instance_id": "django__django-12345", "repo": "django/django"}
        name = sandbox._sandbox_name(task)
        assert name == "swebench-django-django-12345"
        assert "/" not in name

    def test_repo_dir_name(self) -> None:
        sandbox = LocalSandbox()
        task = {"repo": "scikit-learn/scikit-learn"}
        assert sandbox._repo_dir_name(task) == "scikit-learn__scikit-learn"

    def test_init_without_repos_root(self) -> None:
        sandbox = LocalSandbox()
        assert sandbox.repos_root is None

    def test_init_with_repos_root(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(repos_root=tmp_path)
        assert sandbox.repos_root == tmp_path.resolve()


@pytest.mark.skipif(not _has_local_repos(), reason="data/swebench_repos not present")
class TestLocalSandboxIntegration:
    """Integration tests using a real local repo clone."""

    def test_create_exec_destroy_lifecycle(self) -> None:
        """Full lifecycle: create sandbox → exec command → destroy."""
        sandbox = LocalSandbox(repos_root=REPOS_ROOT)
        task = _load_task("psf/requests")

        async def _run() -> None:
            sandbox_id = await sandbox.create_container(task)
            try:
                returncode, output = await sandbox.exec_in_container(
                    sandbox_id,
                    "cd /workspace/repo && echo hello && python3 --version",
                )
                assert returncode == 0, f"Command failed: {output}"
                assert "hello" in output
            finally:
                await sandbox.destroy_container(sandbox_id)
            # Verify temp dir is cleaned up
            assert not Path(sandbox_id).exists()

        asyncio.run(_run())

    def test_exec_returns_nonzero_on_failure(self) -> None:
        """Non-zero exit codes are correctly propagated."""
        sandbox = LocalSandbox(repos_root=REPOS_ROOT)
        task = _load_task("psf/requests")

        async def _run() -> None:
            sandbox_id = await sandbox.create_container(task)
            try:
                returncode, _ = await sandbox.exec_in_container(
                    sandbox_id, "exit 42",
                )
                assert returncode == 42
            finally:
                await sandbox.destroy_container(sandbox_id)

        asyncio.run(_run())

    def test_copy_to_container(self, tmp_path: Path) -> None:
        """copy_to_container places file at correct path in sandbox."""
        sandbox = LocalSandbox(repos_root=REPOS_ROOT)
        task = _load_task("psf/requests")
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello from host")

        async def _run() -> None:
            sandbox_id = await sandbox.create_container(task)
            try:
                await sandbox.copy_to_container(
                    sandbox_id, str(src_file), "/workspace/repo/test.txt"
                )
                returncode, output = await sandbox.exec_in_container(
                    sandbox_id, "cd /workspace/repo && cat test.txt",
                )
                assert returncode == 0
                assert "hello from host" in output
            finally:
                await sandbox.destroy_container(sandbox_id)

        asyncio.run(_run())
