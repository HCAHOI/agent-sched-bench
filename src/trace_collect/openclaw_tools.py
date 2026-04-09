"""Execute imported OpenClaw/Nanobot tool calls inside benchmark replays."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from agents.openclaw.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)

_OBSERVATION_TEMPLATE = (
    "<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
)
_PATH_TOKEN_RE = re.compile(r"/[^\s\"'<>|;&]+")


def _unwrap_tool_args(
    *,
    tool_name: str | None,
    tool_args_json: str,
) -> tuple[str | None, dict[str, Any], bool]:
    """Return (resolved_tool_name, params, is_nested_openclaw_style)."""

    try:
        parsed = json.loads(tool_args_json or "{}")
    except json.JSONDecodeError:
        raise
    if not isinstance(parsed, dict):
        return tool_name, {}, False

    if tool_name and isinstance(parsed.get(tool_name), dict):
        return tool_name, parsed[tool_name], True

    if len(parsed) == 1:
        only_name, only_value = next(iter(parsed.items()))
        if isinstance(only_value, dict):
            return (tool_name or only_name), only_value, True

    return tool_name, parsed, False


def _map_workspace_path(raw_path: str, repo_dir: Path, agent_id: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return (repo_dir / path).resolve()

    parts = list(path.parts)
    if agent_id in parts:
        idx = parts.index(agent_id)
        rel_parts = parts[idx + 1 :]
        return repo_dir.joinpath(*rel_parts).resolve()
    return path.resolve()


def _rewrite_command_paths(command: str, repo_dir: Path, agent_id: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if agent_id not in token:
            return token
        return str(_map_workspace_path(token, repo_dir, agent_id))

    return _PATH_TOKEN_RE.sub(repl, command)


def _format_exec_output(
    output: str,
    returncode: int,
    *,
    command_output_style: str,
) -> str:
    if command_output_style == "replay_observation":
        return _OBSERVATION_TEMPLATE.format(returncode=returncode, output=output)
    return f"{output}\n\nExit code: {returncode}".strip()


async def _run_shell_commands(
    commands: list[str],
    *,
    repo_dir: Path,
    agent_id: str,
    command_timeout_s: float,
    command_output_style: str,
) -> tuple[str, bool]:
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}
    all_output: list[str] = []
    last_returncode = 0

    for raw_cmd in commands:
        cmd = _rewrite_command_paths(raw_cmd, repo_dir, agent_id)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                shell=True,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=command_timeout_s,
                env=env,
            )
            all_output.append(result.stdout + result.stderr)
            last_returncode = result.returncode
        except subprocess.TimeoutExpired:
            all_output.append("[timeout]")
            last_returncode = 124

    if len(commands) > 1:
        combined = "\n".join(f"[call {k}]\n{out}" for k, out in enumerate(all_output))
    else:
        combined = all_output[0] if all_output else ""

    return (
        _format_exec_output(
            combined,
            last_returncode,
            command_output_style=command_output_style,
        ),
        last_returncode == 0,
    )


def _filesystem_tools(repo_dir: Path) -> dict[str, Any]:
    return {
        "read_file": ReadFileTool(workspace=repo_dir),
        "write_file": WriteFileTool(workspace=repo_dir),
        "edit_file": EditFileTool(workspace=repo_dir),
        "list_dir": ListDirTool(workspace=repo_dir),
    }


def _tool_result_ok(result: Any) -> bool:
    if not isinstance(result, str):
        return True
    return not (
        result.startswith("Error")
        or result.startswith("Warning:")
    )


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


async def execute_trace_tool(
    *,
    agent_id: str,
    tool_name: str | None,
    tool_args_json: str,
    repo_dir: Path,
    command_timeout_s: float,
    command_output_style: str = "raw",
) -> tuple[str, bool]:
    """Execute one benchmark or imported OpenClaw tool call inside *repo_dir*."""

    resolved_name, params, nested_style = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
    )

    if "command" in params:
        return await _run_shell_commands(
            [params["command"]],
            repo_dir=repo_dir,
            agent_id=agent_id,
            command_timeout_s=command_timeout_s,
            command_output_style=command_output_style,
        )
    if "commands" in params:
        return await _run_shell_commands(
            list(params["commands"]),
            repo_dir=repo_dir,
            agent_id=agent_id,
            command_timeout_s=command_timeout_s,
            command_output_style=command_output_style,
        )

    if resolved_name == "exec":
        command = params.get("command")
        commands = params.get("commands")
        if command:
            return await _run_shell_commands(
                [command],
                repo_dir=repo_dir,
                agent_id=agent_id,
                command_timeout_s=command_timeout_s,
                command_output_style="raw" if nested_style else command_output_style,
            )
        if commands:
            return await _run_shell_commands(
                list(commands),
                repo_dir=repo_dir,
                agent_id=agent_id,
                command_timeout_s=command_timeout_s,
                command_output_style="raw" if nested_style else command_output_style,
            )
        return "Error: exec requires 'command' or 'commands'.", False

    fs_tool = _filesystem_tools(repo_dir).get(resolved_name or "")
    if fs_tool is not None:
        tool_params = dict(params)
        if "path" in tool_params:
            tool_params["path"] = str(
                _map_workspace_path(tool_params["path"], repo_dir, agent_id)
            )
        result = await fs_tool.execute(**tool_params)
        return _stringify_tool_result(result), _tool_result_ok(result)

    return f"Error: Unsupported replay tool {resolved_name!r}", False
