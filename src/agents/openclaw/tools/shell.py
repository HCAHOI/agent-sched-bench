import asyncio
import os
import re
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from agents.openclaw.tools.base import Tool


MAX_EXEC_TOOL_TIMEOUT_SEC = 600


class ExecTool(Tool):
    _DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        *,
        container_id: str | None = None,
        container_executable: str | None = None,
        container_default_cwd: str = "/testbed",
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.container_default_cwd = container_default_cwd
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"(?:^|[;&|]\s*)format\b",  # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self._container_id = container_id
        self._container_executable = container_executable

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = MAX_EXEC_TOOL_TIMEOUT_SEC
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 300, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        in_container = bool(self._container_id and self._container_executable)
        # Guard must evaluate paths in the namespace the command runs in.
        cwd = (
            (working_dir or self.container_default_cwd)
            if in_container
            else (working_dir or self.working_dir or os.getcwd())
        )
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        # Container mode: redirect through docker exec
        if in_container:
            return await self._execute_in_container(
                command=command,
                working_dir=working_dir,
                timeout=effective_timeout,
            )

        # Local execution via subprocess
        return await self._execute_locally(
            command=command,
            cwd=cwd,
            effective_timeout=effective_timeout,
        )

    async def _execute_in_container(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> str:
        cwd = working_dir or self.container_default_cwd
        effective_timeout = timeout or self._DEFAULT_TIMEOUT
        try:
            # Wrap with timeout(1) inside the container so the entire
            # process tree (descendants of sh -c) is killed on deadline,
            # not just the local docker exec process.  The asyncio
            # deadline has a generous margin (+60 s) so timeout(1) does
            # the real enforcement; the outer guard catches the rare
            # case where timeout(1) is missing in the container image.
            timeout_command = f"timeout {effective_timeout} /bin/sh -c {shlex.quote(command)}"
            proc = await asyncio.create_subprocess_exec(
                self._container_executable,
                "exec",
                "-i",
                "-w", cwd,
                self._container_id,
                "/bin/sh", "-c", timeout_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout + 60,
                )
            except asyncio.TimeoutError:
                # Last resort — timeout(1) inside the container may be
                # absent.  Kill the local docker exec; the orphaned
                # container-side process is a minor leak that the
                # container's PID 1 will reap on container stop.
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError, OSError):
                    pass
                return f"Error: Command timed out after {effective_timeout} seconds"

            # timeout(1) exits 124 when it kills the process tree
            if proc.returncode == 124:
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")
            output_parts.append(f"\nExit code: {proc.returncode}")
            result = "\n".join(output_parts) if output_parts else "(no output)"

            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )
            return result
        except Exception as e:
            return f"Error executing command: {str(e)}"

    async def _execute_locally(
        self,
        command: str,
        cwd: str,
        effective_timeout: int,
    ) -> str:
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=(sys.platform != "win32"),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # Kill the entire process group so descendants (e.g. python
                # train.py spawned by `sh -c`) don't survive as orphans.
                if sys.platform != "win32":
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError) as e:
                        logger.debug("killpg failed: {}", e)
                        process.kill()
                else:
                    process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if sys.platform != "win32":
                        try:
                            os.waitpid(process.pid, os.WNOHANG)
                        except (ProcessLookupError, ChildProcessError) as e:
                            logger.debug("Process already reaped or not found: {}", e)
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation to preserve both start and end of output
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        from agents.openclaw.security.network import contains_internal_url

        if contains_internal_url(cmd):
            return (
                "Error: Command blocked by safety guard (internal/private URL detected)"
            )

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return (
                    "Error: Command blocked by safety guard (path traversal detected)"
                )

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        # Windows: match drive-root paths like `C:\` as well as `C:\path\to\file`
        # NOTE: `*` is required so `C:\` (nothing after the slash) is still extracted.
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]*", command)
        posix_paths = re.findall(
            r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command
        )  # POSIX: /absolute only
        home_paths = re.findall(
            r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command
        )  # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths
