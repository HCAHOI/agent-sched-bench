"""Data-driven command grouping for tool latency prediction.

Latency for command-running tools (e.g. shell ``exec``) depends on *what*
command runs, so grouping by tool name alone collapses to the base rate.
This module derives a finer group key from the command string itself: the
sorted set of command heads across pipeline/list segments (``find .. |
xargs .. | head`` -> ``find+head+xargs``). Only generic shell semantics are
used - operator splitting, env-assignment prefixes, path basenames - and
the resulting groups are whatever the data contains; no command classes
are hardcoded.

Heads are treated as an unordered, deduplicated set because the programs a
command invokes proxy its cost regardless of pipeline order, and shell
comments (``# ...``) are dropped by tokenization. Command substitution
bodies (``$(date)``) contribute their inner head - a known over-splitting
limitation, harmless under the group -> tool -> global hierarchy. The
``:``/``+`` key delimiters could in principle appear inside a head; such a
collision only merges two groups' histories, never breaks causality.

Commands that cannot be tokenized (e.g. unbalanced quotes emitted by the
model) yield no key and group at the tool level - the designed fallback of
the group -> tool -> global hierarchy, not an error, because malformed
commands are expected input in agent traces.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Callable

# Tokens made purely of these characters separate commands (&&, ||, ;, |, &,
# subshell parens); redirection operators (>, >>, <, >&) and their target /
# fd-number words do not start a new command head.
_HEAD_SEPARATOR_CHARS = frozenset(";|&()")
_REDIRECTION_CHARS = frozenset("<>&")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def shell_command_heads(command: str) -> list[str]:
    """Command heads (first word of each pipeline/list segment) of a shell command.

    Heads are path-basenamed so ``/usr/bin/python`` and ``python`` group
    together. Untokenizable commands return an empty list.
    """

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    heads: list[str] = []
    expect_head = True
    skip_redirection_target = False
    for index, token in enumerate(tokens):
        if not token:
            continue
        if all(char in _HEAD_SEPARATOR_CHARS for char in token):
            expect_head = True
            skip_redirection_target = False
            continue
        if _is_redirection_operator(token):
            skip_redirection_target = True
            continue
        if skip_redirection_target:
            skip_redirection_target = False
            continue
        if not expect_head:
            continue
        if _ENV_ASSIGNMENT.match(token):
            continue
        if token.isdigit() and index + 1 < len(tokens) and _is_redirection_operator(
            tokens[index + 1]
        ):
            continue  # fd number of a redirection like 2>&1
        head = token.rsplit("/", 1)[-1]
        if head:
            heads.append(head)
            expect_head = False
    return heads


def _is_redirection_operator(token: str) -> bool:
    return bool(token) and all(char in _REDIRECTION_CHARS for char in token) and any(
        char in "<>" for char in token
    )


def command_group_key(tool_name: str, command: str) -> str | None:
    """Group key for one tool call: tool plus its distinct command heads."""

    heads = shell_command_heads(command)
    if not heads:
        return None
    return f"{tool_name}:{'+'.join(sorted(set(heads)))}"


def make_row_command_key(command_field: str) -> Callable[[dict[str, Any]], str | None]:
    """Row-level group key function reading the command from ``tool_args``.

    Rows without a parseable ``tool_args`` dict or without a string command
    under ``command_field`` yield ``None`` (tool-level grouping).
    """

    def row_key(row: dict[str, Any]) -> str | None:
        tool_args = row.get("tool_args")
        if not isinstance(tool_args, dict):
            return None
        command = tool_args.get(command_field)
        if not isinstance(command, str) or not command.strip():
            return None
        tool_name = row.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            return None
        return command_group_key(tool_name, command)

    return row_key


__all__ = ["command_group_key", "make_row_command_key", "shell_command_heads"]
