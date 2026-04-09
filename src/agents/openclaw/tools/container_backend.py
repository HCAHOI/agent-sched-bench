"""Container-backed openclaw tools routed through ``podman exec``/``podman cp``.

When an openclaw run participates in the attempt_pipeline (SWE-rebench path),
its five filesystem/shell tools execute inside the task's pre-built container
rather than on the host. This way (a) the resource sampler can observe the
pytest/pip work, (b) the host conda env is not polluted by ``pip install -e .``,
and (c) the run is structurally comparable with the Claude Code harness.

The module provides a ``ContainerWorkspace`` helper plus five ``Tool``
subclasses that override ``execute()``. All other tool metadata (name,
description, JSON schema, concurrency flags) is inherited verbatim from the
host versions so the rest of the openclaw agent loop is unchanged.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agents.openclaw.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from agents.openclaw.tools.shell import ExecTool


class ContainerWorkspace:
    """Bundles a podman container id with exec/cp helpers.

    All filesystem operations resolve paths as container-absolute: a
    caller-supplied ``path`` that does not start with ``/`` is joined onto
    ``self.cwd`` (default ``/testbed``).
    """

    def __init__(
        self,
        container_id: str,
        *,
        executable: str = "podman",
        cwd: str = "/testbed",
    ) -> None:
        self.container_id = container_id
        self.executable = executable
        self.cwd = cwd

    def resolve(self, path: str) -> str:
        if not path:
            return self.cwd
        if path.startswith("/"):
            return path
        return f"{self.cwd}/{path}"

    async def exec(
        self,
        command: str,
        *,
        timeout: float,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        cmd = [
            self.executable,
            "exec",
            "-w",
            cwd or self.cwd,
            self.container_id,
            "bash",
            "-lc",
            command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return -1, "", f"timed out after {timeout}s"
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )

    async def read_bytes(self, container_path: str) -> bytes:
        """Read file contents via ``podman cp``."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            host_path = tmp.name
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    self.executable,
                    "cp",
                    f"{self.container_id}:{container_path}",
                    host_path,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if result.returncode != 0:
                raise FileNotFoundError(
                    f"podman cp failed for {container_path}: {result.stderr.strip()}"
                )
            return Path(host_path).read_bytes()
        finally:
            Path(host_path).unlink(missing_ok=True)

    async def write_bytes(self, container_path: str, data: bytes) -> None:
        """Write file contents via ``podman cp`` (atomic, handles binary safely)."""
        parent = str(Path(container_path).parent) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}", timeout=30)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            host_path = tmp.name
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    self.executable,
                    "cp",
                    host_path,
                    f"{self.container_id}:{container_path}",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if result.returncode != 0:
                raise OSError(
                    f"podman cp failed for {container_path}: {result.stderr.strip()}"
                )
        finally:
            Path(host_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tool subclasses — each overrides execute() only.
# ---------------------------------------------------------------------------


class ContainerReadFileTool(ReadFileTool):
    def __init__(self, workspace: ContainerWorkspace) -> None:
        super().__init__(workspace=None, allowed_dir=None, extra_allowed_dirs=None)
        self._ws = workspace

    async def execute(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if not path:
            return "Error reading file: Unknown path"
        try:
            fp = self._ws.resolve(path)
            raw = await self._ws.read_bytes(fp)
            if not raw:
                return f"(Empty file: {path})"
            try:
                text_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                return f"Error: Cannot read binary file {path}"

            all_lines = text_content.splitlines()
            total = len(all_lines)
            if offset < 1:
                offset = 1
            if offset > total:
                return (
                    f"Error: offset {offset} is beyond end of file ({total} lines)"
                )
            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [
                f"{start + i + 1}| {line}"
                for i, line in enumerate(all_lines[start:end])
            ]
            result = "\n".join(numbered)
            if len(result) > self._MAX_CHARS:
                trimmed: list[str] = []
                chars = 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)
            if end < total:
                result += (
                    f"\n\n(Showing lines {offset}-{end} of {total}. "
                    f"Use offset={end + 1} to continue.)"
                )
            else:
                result += f"\n\n(End of file — {total} lines total)"
            return result
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as exc:
            return f"Error reading file: {exc}"


class ContainerWriteFileTool(WriteFileTool):
    def __init__(self, workspace: ContainerWorkspace) -> None:
        super().__init__(workspace=None, allowed_dir=None)
        self._ws = workspace

    async def execute(
        self,
        path: str | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not path:
            return "Error: Unknown path"
        if content is None:
            return "Error: Unknown content"
        try:
            fp = self._ws.resolve(path)
            await self._ws.write_bytes(fp, content.encode("utf-8"))
            return f"Wrote {len(content)} chars to {path}"
        except Exception as exc:
            return f"Error writing file: {exc}"


class ContainerEditFileTool(EditFileTool):
    """Str-replace edit implemented via read+write round-trip.

    Uses the same ``old_text`` / ``new_text`` parameter names as the host
    ``EditFileTool`` so the LLM's tool schema (pulled from the base class's
    ``parameters``) matches the execute() signature.
    """

    def __init__(self, workspace: ContainerWorkspace) -> None:
        super().__init__(workspace=None, allowed_dir=None)
        self._ws = workspace

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> str:
        if not path:
            return "Error: Unknown path"
        if old_text is None:
            return "Error: Unknown old_text"
        if new_text is None:
            return "Error: Unknown new_text"
        try:
            fp = self._ws.resolve(path)
            raw = await self._ws.read_bytes(fp)
            text = raw.decode("utf-8", errors="replace")
            if old_text not in text:
                return (
                    f"Error: old_text not found in {path} "
                    "(container_backend uses exact-match only; try a more specific "
                    "substring or split the edit)."
                )
            count = text.count(old_text)
            if count > 1 and not replace_all:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context to make it unique, or set replace_all=true."
                )
            new_content = (
                text.replace(old_text, new_text)
                if replace_all
                else text.replace(old_text, new_text, 1)
            )
            await self._ws.write_bytes(fp, new_content.encode("utf-8"))
            return f"Successfully edited {path}"
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as exc:
            return f"Error editing file: {exc}"


class ContainerListDirTool(ListDirTool):
    def __init__(self, workspace: ContainerWorkspace) -> None:
        super().__init__(workspace=None, allowed_dir=None)
        self._ws = workspace

    async def execute(
        self,
        path: str | None = None,
        recursive: bool = False,
        **kwargs: Any,
    ) -> str:
        fp = self._ws.resolve(path or ".")
        if recursive:
            cmd = (
                f"find {shlex.quote(fp)} -maxdepth 4 -printf '%y %p\\n' | head -500"
            )
        else:
            cmd = f"ls -la --color=never {shlex.quote(fp)}"
        rc, stdout, stderr = await self._ws.exec(cmd, timeout=30)
        if rc != 0:
            return f"Error listing {path}: {stderr.strip() or stdout.strip()}"
        return stdout or "(empty)"


class ContainerExecTool(ExecTool):
    """``ExecTool`` that routes commands through ``podman exec`` inside the task container."""

    def __init__(
        self,
        workspace: ContainerWorkspace,
        *,
        timeout: int = 60,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            working_dir=workspace.cwd,
            deny_patterns=deny_patterns,
            allow_patterns=allow_patterns,
            restrict_to_workspace=False,
        )
        self._ws = workspace

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        guard_error = self._guard_command(command, working_dir or self._ws.cwd)
        if guard_error:
            return guard_error
        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        rc, stdout, stderr = await self._ws.exec(
            command,
            timeout=effective_timeout,
            cwd=working_dir or self._ws.cwd,
        )

        if rc == -1 and stderr.startswith("timed out"):
            return f"Error: Command timed out after {effective_timeout} seconds"

        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr.strip():
            parts.append(f"STDERR:\n{stderr}")
        parts.append(f"\nExit code: {rc}")
        result = "\n".join(parts) if parts else "(no output)"

        max_len = self._MAX_OUTPUT
        if len(result) > max_len:
            half = max_len // 2
            result = (
                result[:half]
                + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                + result[-half:]
            )
        return result


__all__ = [
    "ContainerWorkspace",
    "ContainerReadFileTool",
    "ContainerWriteFileTool",
    "ContainerEditFileTool",
    "ContainerListDirTool",
    "ContainerExecTool",
]
