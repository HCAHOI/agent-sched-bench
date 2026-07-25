from __future__ import annotations
import shlex

from typing import Any, Iterable

from agents.openclaw.tools.base import (
    Tool,
    _list_dir_parameters_schema,
    _read_file_parameters_schema,
    _write_file_parameters_schema,
)
from agents.openclaw.tools.shell import ExecTool, MAX_EXEC_TOOL_TIMEOUT_SEC
from agents.openclaw.tools.registry import TOOL_ERROR_HINT
from trace_collect.openclaw_tools import (
    ContainerAgent,
    source_runtime_artifact_root_from_path,
)


class _ContainerTool(Tool):
    """Base class for OpenClaw tools executed through a task-container agent."""

    def __init__(self, agent: ContainerAgent) -> None:
        self._agent = agent

    async def _request(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        timeout_s: float | None = 600.0,
    ) -> dict[str, Any]:
        return await self._agent.execute(
            {"tool": tool, "args": args},
            timeout_s=timeout_s,
        )

    @staticmethod
    def _result_or_error(response: dict[str, Any]) -> str:
        result = str(response.get("result", ""))
        if response.get("ok", False):
            return result
        return result if result.startswith("Error") else f"Error: {result}"


class ContainerReadFileTool(_ContainerTool):
    _DEFAULT_LIMIT = 2000

    def __init__(
        self,
        agent: ContainerAgent,
        *,
        runtime_artifact_root_map: dict[str, str] | None = None,
    ) -> None:
        super().__init__(agent)
        self._runtime_artifact_root_map = runtime_artifact_root_map or {}

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from the task container. Returns numbered lines. "
            "Use offset and limit to paginate through large files."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return _read_file_parameters_schema()

    async def execute(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int | None = None,
        **_: Any,
    ) -> str:
        if not path:
            return "Error reading file: Unknown path"
        source_root = source_runtime_artifact_root_from_path(path)
        if source_root in self._runtime_artifact_root_map:
            path = (
                self._runtime_artifact_root_map[source_root] + path[len(source_root) :]
            )
        response = await self._request(
            "read_file",
            {
                "path": path,
                # The in-container replay shim uses zero-based offsets.
                "offset": max(0, int(offset or 1) - 1),
                "limit": int(limit or self._DEFAULT_LIMIT),
            },
        )
        return self._result_or_error(response)


class ContainerWriteFileTool(_ContainerTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file inside the task container. Creates parents if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return _write_file_parameters_schema()

    async def execute(
        self,
        path: str | None = None,
        content: str | None = None,
        **_: Any,
    ) -> str:
        if not path:
            return "Error writing file: Unknown path"
        if content is None:
            return "Error writing file: Unknown content"
        response = await self._request(
            "write_file",
            {"path": path, "content": content},
        )
        return self._result_or_error(response)


class ContainerEditFileTool(_ContainerTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file inside the task container by replacing old_text with new_text. "
            "Supports minor whitespace/line-ending differences."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "Text to replace"},
                "new_text": {"type": "string", "description": "Replacement text"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences instead of exactly one",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        **_: Any,
    ) -> str:
        if not path:
            return "Error editing file: Unknown path"
        if old_text is None or new_text is None:
            return "Error editing file: Unknown replacement text"
        response = await self._request(
            "edit_file",
            {
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
                "replace_all": replace_all,
            },
        )
        return self._result_or_error(response)


class ContainerListDirTool(_ContainerTool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List directory contents inside the task container. "
            "Set recursive=true to explore nested structure."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return _list_dir_parameters_schema()

    async def execute(
        self,
        path: str | None = ".",
        recursive: bool = False,
        max_entries: int | None = None,
        **_: Any,
    ) -> str:
        response = await self._request(
            "list_dir",
            {
                "path": path or ".",
                "recursive": bool(recursive),
                "max_entries": int(max_entries or 200),
            },
        )
        return self._result_or_error(response)


class ContainerExecTool(_ContainerTool):
    _MAX_TIMEOUT = MAX_EXEC_TOOL_TIMEOUT_SEC

    def __init__(
        self,
        agent: ContainerAgent,
        *,
        timeout: int = 300,
        path_append: str = "",
        restrict_to_workspace: bool = False,
        workspace: str = "/testbed",
        clause_telemetry: Any | None = None,
    ) -> None:
        super().__init__(agent)
        self.timeout = timeout
        self.path_append = path_append
        self.workspace = workspace or "/testbed"
        self._clause_telemetry = clause_telemetry
        self._tool_call_id: str | None = None
        self._pending_clause_token: Any | None = None
        self._pending_clause_response: dict[str, Any] | None = None
        self._guard = ExecTool(
            timeout=timeout,
            working_dir=self.workspace,
            restrict_to_workspace=restrict_to_workspace,
            path_append=path_append,
        )

    def set_tool_call_context(
        self,
        tool_call_id: str,
        _arguments: dict[str, Any],
    ) -> None:
        self._tool_call_id = tool_call_id

    def finish_clause_telemetry(self) -> None:
        token, self._pending_clause_token = self._pending_clause_token, None
        response, self._pending_clause_response = (
            self._pending_clause_response,
            None,
        )
        if token is not None:
            try:
                self._clause_telemetry.finish_tool_call(
                    token,
                    replay_response=response,
                )
            except BaseException as exc:
                self._record_telemetry_failure("finish", exc)

    def _record_telemetry_failure(self, phase: str, exc: BaseException) -> None:
        try:
            self._clause_telemetry.add_integrity_error(
                f"telemetry {phase} failed: {type(exc).__name__}: {exc}"
            )
        except BaseException:
            pass

    @property
    def clause_telemetry_enabled(self) -> bool:
        return self._clause_telemetry is not None

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command inside the task container and return its output."
        )

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return self._guard.parameters

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **_: Any,
    ) -> str:
        self._pending_clause_response = None
        workdir = working_dir or self.workspace
        guard_error = self._guard._guard_command(command, workdir)
        if guard_error:
            replay_result = guard_error + TOOL_ERROR_HINT
            if self._clause_telemetry is not None:
                try:
                    self._clause_telemetry.record_safety_guard_blocked(
                        self._tool_call_id or "",
                        command,
                        replay_result,
                    )
                except BaseException as exc:
                    self._record_telemetry_failure("safety_guard", exc)
            return replay_result
        effective_timeout = min(int(timeout or self.timeout), self._MAX_TIMEOUT)
        effective_command = command
        if workdir != self.workspace:
            effective_command = f"cd {shlex.quote(workdir)} && {command}"
        if self._clause_telemetry is not None:
            try:
                self._pending_clause_token = self._clause_telemetry.begin_tool_call(
                    self._tool_call_id or "",
                    command,
                )
            except BaseException as exc:
                self._record_telemetry_failure("begin", exc)
        response = await self._request(
            "exec",
            {"command": effective_command, "timeout": effective_timeout},
            timeout_s=float(effective_timeout),
        )
        self._pending_clause_response = response
        result = self._result_or_error(response)
        returncode = response.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool):
            return f"{result}\n\nExit code: {returncode}".strip()
        return result


class UnsupportedReplayTool(Tool):
    """Fail-closed placeholder for OpenClaw tools without container-backed replay."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            f"{self._name} is unavailable during OpenClaw host replay because "
            "it has no container-backed implementation."
        )

    def set_context(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        return (
            f"Error: Tool '{self._name}' is unsupported during OpenClaw host replay; "
            "only container-backed tools may execute."
        )


_UNSUPPORTED_REPLAY_TOOL_NAMES = (
    "web_search",
    "web_fetch",
    "message",
    "spawn",
    "sessions_yield",
)


def build_container_tool_overrides(
    agent: ContainerAgent,
    *,
    exec_timeout: int = 300,
    exec_path_append: str = "",
    restrict_to_workspace: bool = False,
    workspace: str = "/testbed",
    clause_telemetry: Any | None = None,
    runtime_artifact_root_map: dict[str, str] | None = None,
) -> list[Tool]:
    """Return OpenClaw tool replacements backed by a task-container agent."""

    return [
        ContainerReadFileTool(
            agent,
            runtime_artifact_root_map=runtime_artifact_root_map,
        ),
        ContainerWriteFileTool(agent),
        ContainerEditFileTool(agent),
        ContainerListDirTool(agent),
        ContainerExecTool(
            agent,
            timeout=exec_timeout,
            path_append=exec_path_append,
            restrict_to_workspace=restrict_to_workspace,
            workspace=workspace,
            clause_telemetry=clause_telemetry,
        ),
        *[UnsupportedReplayTool(name) for name in _UNSUPPORTED_REPLAY_TOOL_NAMES],
    ]


def register_tool_overrides(registry: Any, tools: Iterable[Tool]) -> None:
    for tool in tools:
        registry.register(tool)
