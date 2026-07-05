"""Shared agent tool-handler code embedded in both FC and Docker replay agents.

This file is read as *source text* by ``fc_rootfs_builder.py`` and
``openclaw_tools.py`` and embedded verbatim into their respective agent
scripts.  It must be self-contained Python 3 stdlib only -- no imports
beyond ``json``, ``os``, ``subprocess``, ``difflib``, ``time``, ``sys``.

Each consumer appends a transport-specific main loop (vsock accept for FC,
stdin/stdout for the Docker replay agent) plus any overrides.

NOTE about ``_load_agent_env``: the Docker replay agent inherits its
environment from the container's ``docker exec``, which already carries the
image's ``ENV``.  The FC agent runs inside a Firecracker VM where
``docker export`` stripped the image config, so it relies on
``/etc/agent-env.json`` materialized at rootfs-build time.  Calling
``_load_agent_env()`` is safe in both cases -- the file simply won't exist
in the Docker container.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 10_000
_READ_MAX_CHARS = 128_000
_READ_DEFAULT_LIMIT = 2000
_LIST_IGNORE = frozenset({
    ".git", "node_modules", "__pycache__", ".venv",
    ".tox", ".mypy_cache", ".pytest_cache",
})
_LIST_MAX = 200

# ---------------------------------------------------------------------------
# Environment helper (FC-specific but harmless in Docker)
# ---------------------------------------------------------------------------


def _load_agent_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` merged with ``/etc/agent-env.json``.

    The JSON file is written at rootfs-build time from
    ``docker image inspect`` and carries the image's ``ENV`` and
    ``WORKDIR``.
    """
    env = {**os.environ}
    try:
        config = json.load(open("/etc/agent-env.json"))
        for entry in config.get("env", []):
            if "=" in entry:
                key, val = entry.split("=", 1)
                env[key] = val
    except Exception:
        pass
    return env

# ---------------------------------------------------------------------------
# Edit-file helpers
# ---------------------------------------------------------------------------


def _find_match(content: str, old_text: str):
    """Locate *old_text* in *content*, trying literal match then whitespace-
    insensitive search.  Returns ``(matched_text, count)`` or ``(None, 0)``.
    """
    if old_text in content:
        return old_text, content.count(old_text)
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates: list[str] = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))
    if candidates:
        return candidates[0], len(candidates)
    return None, 0


def _not_found_msg(old_text: str, content: str, path: str) -> str:
    """Build a diagnostic error message when *old_text* is not found.

    Includes the closest unified-diff match when similarity > 50 %.
    """
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)
    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(
            None, old_lines, lines[i : i + window],
        ).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
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
            f"Best match ({best_ratio:.0%}) at line {best_start + 1}:\n{diff}"
        )
    return f"Error: old_text not found in {path}. No similar text found."

# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------


def _truncate_output(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... ({len(text) - limit:,} chars truncated) ...\n\n"
        + text[-half:]
    )


def _format_exec_result(stdout: str, stderr: str, returncode: int) -> str:
    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout)
    if stderr and stderr.strip():
        output_parts.append(f"STDERR:\n{stderr}")
    output_parts.append(f"\nExit code: {returncode}")
    return "\n".join(output_parts)


def _format_command_timeout(timeout: object) -> str:
    return f"Error: Command timed out after {timeout} seconds"

# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def handle_exec(args: dict) -> dict:
    cmd = args.get("command", "")
    timeout = float(args.get("timeout", 600))
    env = _load_agent_env()
    cwd = args.get("cwd", "/testbed")
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = _format_exec_result(r.stdout or "", r.stderr or "", r.returncode)
        return {
            "ok": True,
            "result": _truncate_output(output),
            "returncode": r.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "result": _format_command_timeout(timeout),
            "returncode": 124,
            "timed_out": True,
        }


def handle_commands(args: dict) -> dict:
    cmds = args.get("commands", [])
    timeout = float(args.get("timeout", 600))
    env = _load_agent_env()
    cwd = args.get("cwd", "/testbed")
    all_output: list[str] = []
    last_rc = 0
    first_failed_rc = 0
    any_timeout = False
    for cmd in cmds:
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            all_output.append(
                _format_exec_result(r.stdout or "", r.stderr or "", r.returncode)
            )
            last_rc = r.returncode
            if r.returncode != 0 and first_failed_rc == 0:
                first_failed_rc = r.returncode
        except subprocess.TimeoutExpired:
            all_output.append(_format_command_timeout(timeout))
            last_rc = 124
            any_timeout = True
    if len(cmds) > 1:
        combined = "\n".join(
            f"[call {k}]\n{out}" for k, out in enumerate(all_output)
        )
    else:
        combined = all_output[0] if all_output else ""
    returncode = 124 if any_timeout else (first_failed_rc or last_rc)
    return {
        "ok": not any_timeout,
        "result": combined,
        "returncode": returncode,
        "timed_out": any_timeout,
    }


def handle_read_file(args: dict) -> dict:
    path = args.get("path", "")
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", _READ_DEFAULT_LIMIT))
    try:
        content = open(path).read()
        if not content:
            return {"ok": True, "result": f"(Empty file: {path})"}
        lines = content.splitlines()
        selected = lines[offset : offset + limit]
        numbered = "\n".join(
            f"{offset + i + 1}| {ln}" for i, ln in enumerate(selected)
        )
        if len(numbered) > _READ_MAX_CHARS:
            numbered = (
                numbered[:_READ_MAX_CHARS]
                + f"\n\n... (truncated at {_READ_MAX_CHARS} chars)"
            )
        return {"ok": True, "result": numbered}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}


def handle_write_file(args: dict) -> dict:
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return {"ok": True, "result": f"Successfully wrote {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}


def handle_edit_file(args: dict) -> dict:
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    replace_all = args.get("replace_all", False)
    try:
        raw = open(path, "rb").read()
        uses_crlf = b"\r\n" in raw
        content = raw.decode("utf-8").replace("\r\n", "\n")
        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if match is None:
            return {
                "ok": False,
                "result": _not_found_msg(old_text, content, path),
            }
        if count > 1 and not replace_all:
            return {
                "ok": False,
                "result": (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context or set replace_all=true."
                ),
            }
        norm_new = new_text.replace("\r\n", "\n")
        new_content = (
            content.replace(match, norm_new)
            if replace_all
            else content.replace(match, norm_new, 1)
        )
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        open(path, "wb").write(new_content.encode("utf-8"))
        return {"ok": True, "result": f"Successfully edited {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error editing file: {e}"}


def handle_list_dir(args: dict) -> dict:
    path = args.get("path", ".")
    try:
        entries = sorted(e for e in os.listdir(path) if e not in _LIST_IGNORE)
        if len(entries) > _LIST_MAX:
            entries = entries[:_LIST_MAX]
            entries.append(
                f"... ({len(os.listdir(path)) - _LIST_MAX} more entries)"
            )
        return {"ok": True, "result": "\n".join(entries)}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}


# NOTE: Each consumer appends its own HANDLERS dispatch dict in its
# transport-specific code block. This file provides only the handler
# function definitions.
