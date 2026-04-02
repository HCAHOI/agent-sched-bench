"""proot-based local sandbox for per-task isolation (no container daemon needed).

Replaces ContainerManager (Podman) with a temp-directory + proot approach
that works in environments where user namespaces are unavailable.

If proot is not installed, falls back to plain subprocess with path
translation so the code runs everywhere.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cached proot availability check
_PROOT_AVAILABLE: bool | None = None


def _has_proot() -> bool:
    global _PROOT_AVAILABLE
    if _PROOT_AVAILABLE is None:
        _PROOT_AVAILABLE = shutil.which("proot") is not None
        if _PROOT_AVAILABLE:
            logger.info("proot found — using proot for sandbox isolation")
        else:
            logger.info("proot not found — falling back to plain subprocess")
    return _PROOT_AVAILABLE


class LocalSandbox:
    """Temp-directory sandbox for SWE-bench task isolation.

    Each task gets its own temp directory with:
    - A clone of the target repo at the correct base_commit
    - The repo's Python dependencies installed (best-effort, system pip)

    proot maps the temp directory to /workspace/repo so commands
    work identically to the former Podman-based setup.  Falls back to
    plain subprocess with path substitution if proot is not available.

    Lifecycle::

        sandbox_id = await sandbox.create_container(task)
        # ... agent loop: exec_in_container() for each tool call ...
        await sandbox.destroy_container(sandbox_id)

    Note: ``sandbox_id`` (returned by ``create_container``) is the temp
    directory path as a string.  The method names retain the container
    vocabulary for drop-in compatibility with CodeAgent.
    """

    def __init__(self, repos_root: Path | None = None) -> None:
        """
        Args:
            repos_root: Host directory containing pre-cloned repos.
                If provided, repos are cloned locally (fast, ~seconds).
                If None, repos are cloned from GitHub on-the-fly (slow).
        """
        self.repos_root = repos_root.resolve() if repos_root else None

    def _sandbox_name(self, task: dict[str, Any]) -> str:
        """Derive a unique sandbox prefix from the task instance_id."""
        instance_id = task["instance_id"]
        return f"swebench-{instance_id.replace('/', '-').replace('__', '-')}"

    def _repo_dir_name(self, task: dict[str, Any]) -> str:
        """Derive the local repo directory name: owner__name."""
        owner, name = task["repo"].split("/")
        return f"{owner}__{name}"

    async def _run(
        self,
        args: list[str],
        *,
        cwd: Path | str | None = None,
        timeout_s: float = 300.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a subprocess asynchronously in a thread."""
        try:
            return await asyncio.to_thread(
                subprocess.run,
                args,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(cwd) if cwd else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Command timed out after {timeout_s}s: {' '.join(str(a) for a in args)}"
            ) from exc

    async def cleanup_stale(self, prefix: str = "swebench-") -> int:
        """Remove leftover sandbox directories from prior runs.

        Orphaned directories can accumulate when a process is killed before
        the ``finally`` block executes.  Call this before a sweep to
        guarantee a clean slate.

        Returns:
            Number of directories removed.
        """
        pattern = os.path.join(tempfile.gettempdir(), f"{prefix}*")
        stale = glob.glob(pattern)
        removed = 0
        for path in stale:
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.info("Cleaned up stale sandbox %s", path)
                removed += 1
            except Exception as exc:
                logger.warning("Failed to clean up %s: %s", path, exc)
        return removed

    async def create_container(self, task: dict[str, Any]) -> str:
        """Create a sandbox directory for the given task.

        Steps:
        1. Create a fresh temp directory prefixed with the instance_id
        2. Clone the repo at ``base_commit`` (local or GitHub)
        3. Install the repo's Python dependencies (``pip install -e .``,
           best-effort — failure is logged but does not abort)

        Returns:
            The sandbox ID (temp directory path as a string).

        Raises:
            RuntimeError: If git clone or checkout fails.
        """
        sandbox_prefix = self._sandbox_name(task)
        tmpdir = Path(tempfile.mkdtemp(prefix=f"{sandbox_prefix}-"))
        repo_dir_name = self._repo_dir_name(task)
        base_commit = task["base_commit"]

        # Clone repo
        if self.repos_root:
            src = self.repos_root / repo_dir_name
            result = await self._run(
                ["git", "clone", "--local", str(src), str(tmpdir)],
                timeout_s=300.0,
            )
        else:
            repo_url = f"https://github.com/{task['repo']}.git"
            result = await self._run(
                ["git", "clone", repo_url, str(tmpdir)],
                timeout_s=600.0,
            )

        if result.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(
                f"git clone failed for {task['repo']}: {result.stderr[:500]}"
            )

        # Checkout base_commit
        result = await self._run(
            ["git", "-C", str(tmpdir), "checkout", base_commit],
            timeout_s=60.0,
        )
        if result.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(
                f"git checkout {base_commit[:8]} failed: {result.stderr[:500]}"
            )
        logger.info("Cloned %s@%s → %s", task["repo"], base_commit[:8], tmpdir)

        # Install dependencies (best-effort)
        setup_files = ["setup.py", "pyproject.toml"]
        if any((tmpdir / f).exists() for f in setup_files):
            result = await self._run(
                ["python3", "-m", "pip", "install", "-e", ".", "--quiet"],
                cwd=tmpdir,
                timeout_s=600.0,
            )
            if result.returncode != 0:
                logger.warning(
                    "pip install failed in %s (non-fatal): %s",
                    tmpdir, result.stderr[-200:],
                )

        return str(tmpdir)

    async def exec_in_container(
        self,
        container_id: str,
        command: str,
        *,
        timeout_s: float = 120.0,
    ) -> tuple[int, str]:
        """Execute a bash command in the sandbox.

        With proot: binds the sandbox dir to /workspace/repo so all
        path references work without modification.

        Without proot: replaces /workspace/repo with the actual tmpdir
        path before executing.

        Args:
            container_id: Sandbox directory path.
            command: Shell command to execute.
            timeout_s: Maximum execution time in seconds.

        Returns:
            Tuple of (return_code, combined_output).
        """
        tmpdir = container_id

        if _has_proot():
            args: list[str] = [
                "proot",
                "-b", f"{tmpdir}:/workspace/repo",
                "-w", "/workspace/repo",
                "/bin/bash", "-c", command,
            ]
        else:
            translated = command.replace("/workspace/repo", tmpdir)
            args = ["/bin/bash", "-c", translated]

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                args,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Command timed out after {timeout_s}s: {' '.join(str(a) for a in args)}"
            ) from exc

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return result.returncode, output.strip()

    async def copy_to_container(
        self,
        container_id: str,
        src: str,
        dest: str,
    ) -> None:
        """Copy a file from the host into the sandbox.

        Args:
            container_id: Sandbox directory path.
            src: Host source file path.
            dest: Destination path inside the sandbox.
                  Paths under /workspace/repo are remapped to the sandbox dir.
        """
        tmpdir = Path(container_id)
        if dest.startswith("/workspace/repo"):
            rel = dest[len("/workspace/repo"):].lstrip("/")
            dst_path = tmpdir / rel
        else:
            dst_path = tmpdir / dest.lstrip("/")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_path)

    async def destroy_container(self, container_id: str) -> None:
        """Remove the sandbox directory.

        Args:
            container_id: Sandbox directory path.
        """
        try:
            shutil.rmtree(container_id, ignore_errors=True)
            logger.info("Destroyed sandbox %s", container_id)
        except Exception as exc:
            logger.warning("Failed to destroy sandbox %s: %s", container_id, exc)
