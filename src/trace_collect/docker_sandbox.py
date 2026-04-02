"""Docker container lifecycle management for per-task sandboxes.

Drop-in replacement for ContainerManager / LocalSandbox that uses Docker
instead of Podman or proot.  Designed for trace collection on machines
with Docker but without GPU or vLLM.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Manage Docker containers for SWE-bench task isolation.

    Each task gets its own container with:
    - A clone of the target repo at the correct base_commit
    - An independent pip environment for dependency installation
    - Isolated filesystem (no host pollution)

    Lifecycle::

        container_id = await sandbox.create_container(task)
        # ... agent loop: exec_in_container() for each tool call ...
        await sandbox.destroy_container(container_id)
    """

    def __init__(
        self,
        base_image: str = "python:3.11-slim",
        repos_root: Path | None = None,
    ) -> None:
        self.base_image = base_image
        self.repos_root = repos_root.resolve() if repos_root else None

    async def _run_docker(
        self,
        args: list[str],
        *,
        timeout_s: float = 300.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a docker command asynchronously."""
        cmd = ["docker"] + args
        try:
            return await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Docker command timed out after {timeout_s}s: {' '.join(cmd)}"
            ) from exc

    def _container_name(self, task: dict[str, Any]) -> str:
        """Derive a unique container name from the task."""
        instance_id = task["instance_id"]
        return f"swebench-{instance_id.replace('/', '-').replace('__', '-')}"

    def _repo_dir_name(self, task: dict[str, Any]) -> str:
        """Derive the local repo directory name: owner__name."""
        owner, name = task["repo"].split("/")
        return f"{owner}__{name}"

    async def cleanup_stale(self, prefix: str = "swebench-") -> int:
        """Remove any leftover containers from prior runs."""
        result = await self._run_docker(
            ["ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
            timeout_s=10.0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0
        names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
        for name in names:
            await self._run_docker(["rm", "-f", name], timeout_s=10.0)
            logger.info("Cleaned up stale container %s", name)
        return len(names)

    async def create_container(self, task: dict[str, Any]) -> str:
        """Create and start a Docker container for the given task.

        Returns:
            The container name (used as container_id).

        Raises:
            RuntimeError: If container creation or repo setup fails.
        """
        container_name = self._container_name(task)
        await self._run_docker(["rm", "-f", container_name], timeout_s=10.0)
        repo_dir = self._repo_dir_name(task)
        base_commit = task["base_commit"]

        run_args = ["run", "-d", "--name", container_name]
        if self.repos_root:
            run_args.extend(["-v", f"{self.repos_root}:/mnt/repos:ro"])
        run_args.extend([self.base_image, "sleep", "infinity"])

        result = await self._run_docker(run_args)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create container {container_name}: {result.stderr}"
            )
        logger.info("Created container %s", container_name)

        # Install git (base python:3.11-slim may not have it)
        await self.exec_in_container(
            container_name,
            "apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1",
            timeout_s=120.0,
        )

        # Clone repo at correct commit
        if self.repos_root:
            clone_cmd = (
                f"git clone /mnt/repos/{repo_dir} /workspace/repo"
                f" && cd /workspace/repo"
                f" && git checkout {base_commit}"
            )
        else:
            repo_url = f"https://github.com/{task['repo']}.git"
            clone_cmd = (
                f"git clone {repo_url} /workspace/repo"
                f" && cd /workspace/repo"
                f" && git checkout {base_commit}"
            )

        returncode, output = await self.exec_in_container(
            container_name, clone_cmd, timeout_s=300.0,
        )
        if returncode != 0:
            raise RuntimeError(
                f"Repo setup failed in {container_name}: {output[:500]}"
            )
        logger.info("Repo cloned at %s in %s", base_commit[:8], container_name)

        # Install dependencies (best effort)
        install_cmd = (
            "cd /workspace/repo"
            " && if [ -f setup.py ] || [ -f pyproject.toml ]; then"
            "   pip install -e . 2>&1 | tail -5;"
            " fi"
        )
        returncode, output = await self.exec_in_container(
            container_name, install_cmd, timeout_s=600.0,
        )
        if returncode != 0:
            logger.warning(
                "pip install failed in %s (non-fatal): %s",
                container_name, output[-200:],
            )

        return container_name

    async def exec_in_container(
        self,
        container_id: str,
        command: str,
        *,
        timeout_s: float = 120.0,
    ) -> tuple[int, str]:
        """Execute a bash command inside the container."""
        result = await self._run_docker(
            ["exec", container_id, "bash", "-c", command],
            timeout_s=timeout_s,
        )
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
        """Copy a file from host into the container."""
        result = await self._run_docker(
            ["cp", src, f"{container_id}:{dest}"],
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker cp failed: {result.stderr}")

    async def destroy_container(self, container_id: str) -> None:
        """Stop and remove the container."""
        result = await self._run_docker(
            ["rm", "-f", container_id],
            timeout_s=30.0,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to remove container %s: %s",
                container_id, result.stderr,
            )
        else:
            logger.info("Destroyed container %s", container_id)
