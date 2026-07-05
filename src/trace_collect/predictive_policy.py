from __future__ import annotations

import json
import re
import shlex
from enum import Enum
from typing import Any


class CommandFamily(Enum):
    READ_ONLY = "read_only"
    FLAKY_READ = "flaky_read"
    MUTATING = "mutating"
    UNKNOWN = "unknown"


class CheckpointSchedule(Enum):
    DEFERRED_SKIP = "deferred_skip"
    IMMEDIATE_CAPTURE = "immediate_capture"
    AWAIT_PREDICTION = "await_prediction"


_READ_ONLY_TOOL_NAMES = frozenset({"read_file"})
_FLAKY_READ_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
_MUTATING_TOOL_NAMES = frozenset({"write_file", "edit_file", "spawn"})

_READ_ONLY_COMMANDS = frozenset(
    {
        "ls",
        "cat",
        "grep",
        "find",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "echo",
        "which",
        "file",
        "stat",
        "du",
        "df",
        "date",
        "pwd",
        "env",
        "printenv",
        "type",
        "dirname",
        "basename",
        "realpath",
    }
)
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({"status", "diff", "log"})

_MUTATING_COMMANDS = frozenset(
    {
        "pip",
        "pip3",
        "npm",
        "cargo",
        "apt",
        "apt-get",
        "make",
        "cmake",
        "mv",
        "chmod",
        "chown",
        "mkdir",
        "rmdir",
        "touch",
    }
)
_MUTATING_GIT_SUBCOMMANDS = frozenset(
    {"clone", "apply", "commit", "push", "add", "merge"}
)

_TAR_OPERATIONS = frozenset({"x", "c", "t", "r", "u"})


def classify_tool_result(
    tool_name: str | None,
    tool_args_json: str | None,
) -> CommandFamily:
    """Classify based on tool_name and the first token of exec commands."""
    normalized_tool_name = (tool_name or "").strip()
    if normalized_tool_name in _READ_ONLY_TOOL_NAMES:
        return CommandFamily.READ_ONLY
    if normalized_tool_name in _FLAKY_READ_TOOL_NAMES:
        return CommandFamily.FLAKY_READ
    if normalized_tool_name in _MUTATING_TOOL_NAMES:
        return CommandFamily.MUTATING

    command = _command_from_tool_args(tool_args_json)
    if command is None:
        return CommandFamily.UNKNOWN
    return _classify_command(command)


def needs_checkpoint(family: CommandFamily) -> bool:
    """READ_ONLY -> False, MUTATING/UNKNOWN -> True, FLAKY_READ -> False."""
    return family in {CommandFamily.MUTATING, CommandFamily.UNKNOWN}


def get_checkpoint_schedule(
    tool_name: str | None,
    tool_args: str | dict[str, Any] | None,
    probe_status: str,
    backend_type: str,
) -> str:
    """Return the online checkpoint schedule from probe status and prediction."""
    if not backend_type:
        raise ValueError("backend_type must be non-empty")

    normalized_probe_status = probe_status.strip().lower()
    if normalized_probe_status in {"changed", "maybe_changed", "initial"}:
        return CheckpointSchedule.IMMEDIATE_CAPTURE.value
    if normalized_probe_status != "unchanged":
        raise ValueError(f"unsupported probe_status: {probe_status!r}")

    tool_args_json = (
        json.dumps(tool_args, ensure_ascii=False)
        if isinstance(tool_args, dict)
        else tool_args
    )
    family = classify_tool_result(tool_name, tool_args_json)
    if needs_checkpoint(family):
        return CheckpointSchedule.AWAIT_PREDICTION.value
    return CheckpointSchedule.DEFERRED_SKIP.value


def _command_from_tool_args(tool_args_json: str | None) -> str | None:
    if not tool_args_json:
        return None
    try:
        parsed = json.loads(tool_args_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    payload = _command_payload(parsed)
    command = payload.get("command")
    if isinstance(command, str) and command.strip():
        return command

    commands = payload.get("commands")
    if isinstance(commands, list) and commands:
        first = commands[0]
        if isinstance(first, str) and first.strip():
            return first
    return None


def _command_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    nested_exec = parsed.get("exec")
    if isinstance(nested_exec, dict):
        return nested_exec
    if len(parsed) == 1:
        only_value = next(iter(parsed.values()))
        if isinstance(only_value, dict):
            return only_value
    return parsed


def _classify_command(command: str) -> CommandFamily:
    tokens = _split_command(command)
    if not tokens:
        return CommandFamily.UNKNOWN

    first = tokens[0]
    family = _classify_tokens(first, tokens)

    # Output redirect check: commands that use shell output redirects
    # (>, >>, 2>, &>) are modifying files, so upgrade to MUTATING.
    # This applies to READ_ONLY and UNKNOWN classifications alike —
    # only MUTATING is already correct.
    # Input redirect (<) alone does NOT trigger this — reading from a
    # file doesn't modify anything.
    if family != CommandFamily.MUTATING and _has_output_redirect(command):
        return CommandFamily.MUTATING

    return family


def _classify_tokens(first: str, tokens: list[str]) -> CommandFamily:
    if first in _READ_ONLY_COMMANDS:
        return CommandFamily.READ_ONLY
    if first == "git":
        return _classify_git(tokens)
    if first in _MUTATING_COMMANDS:
        return CommandFamily.MUTATING
    if first == "rm" and _has_short_option(tokens[1:], "r"):
        return CommandFamily.MUTATING
    if first == "cp" and _has_short_option(tokens[1:], "r"):
        return CommandFamily.MUTATING
    if first == "tar" and _has_tar_extract_option(tokens[1:]):
        return CommandFamily.MUTATING
    if first == "ln" and _has_short_option(tokens[1:], "s"):
        return CommandFamily.MUTATING
    return CommandFamily.UNKNOWN


_OUTPUT_REDIRECT_RE = re.compile(r">>|[12&]?>")


def _has_output_redirect(command: str) -> bool:
    """Return True if *command* contains shell output-redirect operators.

    Only output redirects count: ``>``, ``>>``, ``2>``, ``&>``.
    Input redirect (``<``) alone does NOT make a command mutating.
    Quoted regions are stripped before matching so ``echo "a > b"``
    does not falsely trigger the redirect detector.
    """
    cleaned = _strip_shell_quoted(command)
    return bool(_OUTPUT_REDIRECT_RE.search(cleaned))


def _strip_shell_quoted(text: str) -> str:
    """Replace single- and double-quoted regions with spaces.

    This neutralises redirect-like substrings that appear inside string
    literals (e.g. ``echo "a > b"``) so the redirect regex only sees
    the unquoted metacharacters.
    """
    result: list[str] = []
    i = 0
    in_single = False
    in_double = False
    while i < len(text):
        ch = text[i]
        if in_single:
            if ch == "'":
                in_single = False
            else:
                result.append(" ")
            i += 1
        elif in_double:
            if ch == '"':
                in_double = False
            elif ch == "\\" and i + 1 < len(text):
                result.append("  ")
                i += 2
                continue
            else:
                result.append(" ")
            i += 1
        else:
            if ch == "'":
                in_single = True
                result.append(" ")
            elif ch == '"':
                in_double = True
                result.append(" ")
            else:
                result.append(ch)
            i += 1
    return "".join(result)


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        return []


def _classify_git(tokens: list[str]) -> CommandFamily:
    if len(tokens) < 2:
        return CommandFamily.UNKNOWN
    subcommand = tokens[1]
    if subcommand in _READ_ONLY_GIT_SUBCOMMANDS:
        return CommandFamily.READ_ONLY
    if subcommand in _MUTATING_GIT_SUBCOMMANDS:
        return CommandFamily.MUTATING
    return CommandFamily.UNKNOWN


def _has_short_option(tokens: list[str], option: str) -> bool:
    for token in tokens:
        normalized = token.lower()
        if token == f"-{option}":
            return True
        if (
            normalized.startswith("-")
            and not normalized.startswith("--")
            and option in normalized[1:]
        ):
            return True
    return False


def _has_tar_extract_option(tokens: list[str]) -> bool:
    for token in tokens:
        normalized = token.lower()
        if normalized == "--extract":
            return True
        if normalized.startswith("-") and "x" in normalized[1:]:
            return True
        # Old-style tar flags without a dash (e.g. "xf", "xzf").
        # Must be short, all-alpha, start with a tar operation letter,
        # and contain "x" to avoid false positives on filenames like
        # "matrix.tar" or unrelated tokens like "pxz".
        if (
            not normalized.startswith("-")
            and "x" in normalized
            and len(normalized) <= 5
            and normalized.isalpha()
            and normalized[0] in _TAR_OPERATIONS
        ):
            return True
    return False
