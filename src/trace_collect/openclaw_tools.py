"""Execute imported OpenClaw/Nanobot tool calls inside benchmark replays."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

_MAX_READ_CHARS = 128_000
_DEFAULT_READ_LIMIT = 2_000
_DEFAULT_LIST_LIMIT = 200
_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
}
_OBSERVATION_TEMPLATE = "<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
_PATH_TOKEN_RE = re.compile(r"/[^\s\"'<>|;&]+")


def _unwrap_tool_args(
    *,
    tool_name: str | None,
    tool_args_json: str,
    raw_response: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any], bool]:
    """Return (resolved_tool_name, params, is_nested_openclaw_style)."""

    try:
        parsed = json.loads(tool_args_json or "{}")
    except json.JSONDecodeError:
        fallback = _tool_args_from_raw_response(raw_response)
        if fallback is not None:
            fallback_name, fallback_args = fallback
            return (tool_name or fallback_name), fallback_args, True
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


def _tool_args_from_raw_response(
    raw_response: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    if not raw_response:
        return None
    message = (raw_response.get("choices") or [{}])[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    function = (tool_calls[0] or {}).get("function") or {}
    tool_name = function.get("name")
    arguments = function.get("arguments")
    if not tool_name or not arguments:
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return tool_name, parsed


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


def _read_file(path: Path, *, offset: int = 1, limit: int | None = None) -> str:
    if not path.exists():
        return f"Error: File not found: {path}"
    if not path.is_file():
        return f"Error: Not a file: {path}"

    raw = path.read_bytes()
    if not raw:
        return f"(Empty file: {path})"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: Cannot read binary file {path}"

    all_lines = text.splitlines()
    total = len(all_lines)
    if offset < 1:
        offset = 1
    if total and offset > total:
        return f"Error: offset {offset} is beyond end of file ({total} lines)"

    start = offset - 1
    end = min(start + (limit or _DEFAULT_READ_LIMIT), total)
    numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
    result = "\n".join(numbered)

    if len(result) > _MAX_READ_CHARS:
        trimmed: list[str] = []
        chars = 0
        for line in numbered:
            chars += len(line) + 1
            if chars > _MAX_READ_CHARS:
                break
            trimmed.append(line)
        end = start + len(trimmed)
        result = "\n".join(trimmed)

    if end < total:
        result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
    else:
        result += f"\n\n(End of file — {total} lines total)"
    return result


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    if old_text in content:
        return old_text, content.count(old_text)

    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0

    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates: list[str] = []
    for idx in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[idx : idx + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))

    if candidates:
        return candidates[0], len(candidates)
    return None, 0


def _edit_file(
    path: Path,
    *,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    if not path.exists():
        return f"Error: File not found: {path}"

    raw = path.read_bytes()
    uses_crlf = b"\r\n" in raw
    content = raw.decode("utf-8").replace("\r\n", "\n")
    match, count = _find_match(content, old_text.replace("\r\n", "\n"))
    if match is None:
        return _not_found_msg(old_text, content, str(path))
    if count > 1 and not replace_all:
        return (
            f"Warning: old_text appears {count} times. "
            "Provide more context to make it unique, or set replace_all=true."
        )

    normalized_new = new_text.replace("\r\n", "\n")
    if replace_all:
        updated = content.replace(match, normalized_new)
    else:
        updated = content.replace(match, normalized_new, 1)
    if uses_crlf:
        updated = updated.replace("\n", "\r\n")
    path.write_bytes(updated.encode("utf-8"))
    return f"Successfully edited {path}"


def _not_found_msg(old_text: str, content: str, path: str) -> str:
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)

    best_ratio, best_start = 0.0, 0
    for idx in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[idx : idx + window]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = idx

    if best_ratio > 0.5:
        diff = "\n".join(
            difflib.unified_diff(
                old_lines,
                lines[best_start : best_start + window],
                fromfile="old_text (provided)",
                tofile=f"{path} (actual, line {best_start + 1})",
                lineterm="",
            )
        )
        return (
            f"Error: old_text not found in {path}.\n"
            f"Best match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        )
    return f"Error: old_text not found in {path}. No similar text found. Verify the file content."


def _list_dir(path: Path, *, recursive: bool = False, max_entries: int | None = None) -> str:
    if not path.exists():
        return f"Error: Directory not found: {path}"
    if not path.is_dir():
        return f"Error: Not a directory: {path}"

    cap = max_entries or _DEFAULT_LIST_LIMIT
    items: list[str] = []
    total = 0

    if recursive:
        for item in sorted(path.rglob("*")):
            if any(part in _IGNORE_DIRS for part in item.parts):
                continue
            total += 1
            if len(items) < cap:
                rel = item.relative_to(path)
                items.append(f"{rel}/" if item.is_dir() else str(rel))
    else:
        for item in sorted(path.iterdir()):
            if item.name in _IGNORE_DIRS:
                continue
            total += 1
            if len(items) < cap:
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

    if not items and total == 0:
        return f"Directory {path} is empty"

    result = "\n".join(items)
    if total > cap:
        result += f"\n\n(truncated, showing first {cap} of {total} entries)"
    return result


async def execute_trace_tool(
    *,
    agent_id: str,
    tool_name: str | None,
    tool_args_json: str,
    repo_dir: Path,
    command_timeout_s: float,
    command_output_style: str = "raw",
    raw_response: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Execute one benchmark or imported OpenClaw tool call inside *repo_dir*."""

    resolved_name, params, nested_style = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        raw_response=raw_response,
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
        return "", True

    if resolved_name == "read_file":
        path = _map_workspace_path(params["path"], repo_dir, agent_id)
        return _read_file(path, offset=params.get("offset", 1), limit=params.get("limit")), True

    if resolved_name == "write_file":
        path = _map_workspace_path(params["path"], repo_dir, agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = params.get("content", "")
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}", True

    if resolved_name == "edit_file":
        path = _map_workspace_path(params["path"], repo_dir, agent_id)
        return (
            _edit_file(
                path,
                old_text=params.get("old_text", ""),
                new_text=params.get("new_text", ""),
                replace_all=bool(params.get("replace_all", False)),
            ),
            True,
        )

    if resolved_name == "list_dir":
        path = _map_workspace_path(params["path"], repo_dir, agent_id)
        return (
            _list_dir(
                path,
                recursive=bool(params.get("recursive", False)),
                max_entries=params.get("max_entries"),
            ),
            True,
        )

    return "", True
