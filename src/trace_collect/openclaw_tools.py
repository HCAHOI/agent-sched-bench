"""Execute tool calls inside Docker/Podman containers during trace replay."""

from __future__ import annotations

import asyncio
import json
import subprocess
import textwrap
from typing import Any


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


# Self-contained edit script piped into `docker exec -i {cid} python3`.
# Reads JSON request from stdin, performs find-match-replace, writes result.
_EDIT_SCRIPT = textwrap.dedent(r'''
import json, sys, difflib

def _find_match(content, old_text):
    if old_text in content:
        return old_text, content.count(old_text)
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))
    if candidates:
        return candidates[0], len(candidates)
    return None, 0

def _not_found_msg(old_text, content, path):
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)
    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i:i+window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
    if best_ratio > 0.5:
        diff = "\n".join(difflib.unified_diff(
            old_lines, lines[best_start:best_start+window],
            fromfile="old_text (provided)",
            tofile=f"{path} (actual, line {best_start+1})", lineterm=""))
        return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start+1}:\n{diff}"
    return f"Error: old_text not found in {path}. No similar text found."

req = json.loads(sys.stdin.read())
path, old_text, new_text = req["path"], req["old_text"], req["new_text"]
replace_all = req.get("replace_all", False)
try:
    raw = open(path, "rb").read()
    uses_crlf = b"\r\n" in raw
    content = raw.decode("utf-8").replace("\r\n", "\n")
    match, count = _find_match(content, old_text.replace("\r\n", "\n"))
    if match is None:
        print(json.dumps({"ok": False, "msg": _not_found_msg(old_text, content, path)}))
        sys.exit(0)
    if count > 1 and not replace_all:
        print(json.dumps({"ok": False, "msg": f"Warning: old_text appears {count} times. Provide more context or set replace_all=true."}))
        sys.exit(0)
    norm_new = new_text.replace("\r\n", "\n")
    new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
    if uses_crlf:
        new_content = new_content.replace("\n", "\r\n")
    open(path, "wb").write(new_content.encode("utf-8"))
    print(json.dumps({"ok": True, "msg": f"Successfully edited {path}"}))
except Exception as e:
    print(json.dumps({"ok": False, "msg": f"Error editing file: {e}"}))
''').strip()


async def _docker_exec(
    container_id: str,
    container_executable: str,
    cmd_args: list[str],
    *,
    timeout_s: float,
    stdin_data: str | None = None,
) -> tuple[str, int]:
    """Run a command inside the container via ``docker exec``.

    Returns (combined_output, returncode).
    """
    full_cmd = [container_executable, "exec"]
    if stdin_data is not None:
        full_cmd.append("-i")
    full_cmd.extend(cmd_args)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            input=stdin_data,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "[timeout]", 124


async def _run_container_command(
    container_id: str,
    container_executable: str,
    command: str,
    *,
    timeout_s: float,
) -> tuple[str, int]:
    """Execute a shell command inside the container."""
    return await _docker_exec(
        container_id,
        container_executable,
        ["-w", "/testbed", container_id, "bash", "-c", command],
        timeout_s=timeout_s,
    )


async def _run_shell_commands(
    commands: list[str],
    *,
    container_id: str,
    container_executable: str,
    command_timeout_s: float,
) -> tuple[str, bool]:
    all_output: list[str] = []
    last_returncode = 0

    for raw_cmd in commands:
        output, returncode = await _run_container_command(
            container_id, container_executable, raw_cmd, timeout_s=command_timeout_s,
        )
        all_output.append(output)
        last_returncode = returncode

    if len(commands) > 1:
        combined = "\n".join(f"[call {k}]\n{out}" for k, out in enumerate(all_output))
    else:
        combined = all_output[0] if all_output else ""

    return f"{combined}\n\nExit code: {last_returncode}".strip(), last_returncode == 0


async def _read_file(
    container_id: str,
    container_executable: str,
    path: str,
    *,
    timeout_s: float,
) -> tuple[str, bool]:
    output, rc = await _docker_exec(
        container_id, container_executable,
        ["-w", "/testbed", container_id, "cat", path],
        timeout_s=timeout_s,
    )
    if rc != 0:
        return f"Error: Failed to read {path}: {output.strip()}", False
    return output, True


async def _write_file(
    container_id: str,
    container_executable: str,
    path: str,
    content: str,
    *,
    timeout_s: float,
) -> tuple[str, bool]:
    quoted = json.dumps(path)
    output, rc = await _docker_exec(
        container_id, container_executable,
        [container_id, "bash", "-c", f'mkdir -p "$(dirname {quoted})" && cat > {quoted}'],
        timeout_s=timeout_s,
        stdin_data=content,
    )
    if rc != 0:
        return f"Error: Failed to write {path}: {output.strip()}", False
    return f"Successfully wrote {path}", True


async def _edit_file(
    container_id: str,
    container_executable: str,
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool,
    *,
    timeout_s: float,
) -> tuple[str, bool]:
    request_json = json.dumps({
        "path": path,
        "old_text": old_text,
        "new_text": new_text,
        "replace_all": replace_all,
    })
    output, rc = await _docker_exec(
        container_id, container_executable,
        [container_id, "python3", "-c", _EDIT_SCRIPT],
        timeout_s=timeout_s,
        stdin_data=request_json,
    )
    if rc != 0:
        return f"Error: edit_file failed: {output.strip()}", False
    try:
        result = json.loads(output.strip().splitlines()[-1])
        return result["msg"], result.get("ok", False)
    except (json.JSONDecodeError, KeyError, IndexError):
        return output.strip() or "Error: unexpected edit_file output", False


async def _list_dir(
    container_id: str,
    container_executable: str,
    path: str,
    *,
    timeout_s: float,
) -> tuple[str, bool]:
    output, rc = await _docker_exec(
        container_id, container_executable,
        ["-w", "/testbed", container_id, "ls", "-1a", path],
        timeout_s=timeout_s,
    )
    if rc != 0:
        return f"Error: Failed to list {path}: {output.strip()}", False
    return output, True


async def execute_trace_tool(
    *,
    container_id: str,
    container_executable: str,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
) -> tuple[str, bool]:
    """Execute one trace tool call inside a Docker/Podman container."""

    resolved_name, params, _nested = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
    )

    # Shell commands (exec / command / commands)
    if "command" in params:
        return await _run_shell_commands(
            [params["command"]],
            container_id=container_id,
            container_executable=container_executable,
            command_timeout_s=command_timeout_s,
        )
    if "commands" in params:
        return await _run_shell_commands(
            list(params["commands"]),
            container_id=container_id,
            container_executable=container_executable,
            command_timeout_s=command_timeout_s,
        )

    if resolved_name == "exec":
        command = params.get("command")
        commands = params.get("commands")
        if command:
            return await _run_shell_commands(
                [command],
                container_id=container_id,
                container_executable=container_executable,
                command_timeout_s=command_timeout_s,
            )
        if commands:
            return await _run_shell_commands(
                list(commands),
                container_id=container_id,
                container_executable=container_executable,
                command_timeout_s=command_timeout_s,
            )
        return "Error: exec requires 'command' or 'commands'.", False

    # Filesystem tools
    if resolved_name == "read_file":
        path = params.get("path", "")
        return await _read_file(
            container_id, container_executable, path,
            timeout_s=command_timeout_s,
        )

    if resolved_name == "write_file":
        path = params.get("path", "")
        content = params.get("content", "")
        return await _write_file(
            container_id, container_executable, path, content,
            timeout_s=command_timeout_s,
        )

    if resolved_name == "edit_file":
        path = params.get("path", "")
        old_text = params.get("old_text", "")
        new_text = params.get("new_text", "")
        replace_all = bool(params.get("replace_all", False))
        return await _edit_file(
            container_id, container_executable, path, old_text, new_text,
            replace_all, timeout_s=command_timeout_s,
        )

    if resolved_name == "list_dir":
        path = params.get("path", ".")
        return await _list_dir(
            container_id, container_executable, path,
            timeout_s=command_timeout_s,
        )

    return f"Error: Unsupported replay tool {resolved_name!r}", False
