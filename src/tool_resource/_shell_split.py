"""Shell token normalization and degraded clause-parse fallback splitting.

Vendored so ``tool_resource`` stays a self-contained directory with no
repository imports. The normalized token stream is also reused by the
historical raw-prefix evaluation baseline.
"""

from __future__ import annotations

import functools
import re
import shlex

# Tokens made purely of these characters separate commands (&&, ||, ;, |, &,
# subshell parens); redirection operators (>, >>, <, >&) and their target /
# fd-number words are dropped from the normalized stream.
_HEAD_SEPARATOR_CHARS = frozenset(";|&()")
_REDIRECTION_CHARS = frozenset("<>&")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Separators whose sides execute one after another, so their times add.
# Pipes (|) and background (&) run concurrently and stay within one unit;
# `||` alternatives rarely both run - counting them as sequential
# over-approximates total time, documented.
_SEQUENTIAL_SEPARATOR_TOKENS = frozenset({"&&", ";", ";;", "||"})
_GROUPING_TOKENS = frozenset({"(", ")"})


def shell_command_segments(command: str) -> list[list[str]]:
    """Sequential execution units of a shell command, as token lists.

    Units are split at sequential separators (``&&``, ``;``, ``||``) whose
    sides run one after another; pipeline (``|``) and background (``&``)
    parts run concurrently and stay within one unit. Grouping parens are
    dropped. Untokenizable commands return an empty list. Escaped
    separators (e.g. find's ``\\;`` exec terminator) are indistinguishable
    from real ones after POSIX unescaping and over-split such commands -
    a rare, granularity-only ambiguity.
    """

    segments: list[list[str]] = []
    current: list[str] = []
    for token in shell_command_prefix_tokens(command):
        if token in _SEQUENTIAL_SEPARATOR_TOKENS:
            if current:
                segments.append(current)
                current = []
        elif token in _GROUPING_TOKENS:
            continue
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def shell_command_prefix_tokens(command: str) -> list[str]:
    """Normalized token stream of a shell command.

    Heads are path-basenamed; env assignments, redirection operators,
    redirection targets, and fd numbers are dropped; command separators
    (``&&``, ``|``, ...) are kept as structural tokens. Untokenizable
    commands return an empty list.
    """

    return [token for token, _ in _normalized_tokens(command)]


# Keep a few thousand-command batch hot without retaining unbounded trace text.
@functools.lru_cache(maxsize=4096)
def _normalized_tokens(command: str) -> tuple[tuple[str, bool], ...]:
    """Cached normalized ``(token, is_head)`` stream of a shell command."""

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return ()
    result: list[tuple[str, bool]] = []
    expect_head = True
    skip_redirection_target = False
    for index, token in enumerate(tokens):
        if not token:
            continue
        if all(char in _HEAD_SEPARATOR_CHARS for char in token):
            expect_head = True
            skip_redirection_target = False
            result.append((token, False))
            continue
        if _is_redirection_operator(token):
            skip_redirection_target = True
            continue
        if skip_redirection_target:
            skip_redirection_target = False
            continue
        if (
            token.isdigit()
            and index + 1 < len(tokens)
            and _is_redirection_operator(tokens[index + 1])
            and "&" in tokens[index + 1]
        ):
            continue  # fd number of a dup redirection like 2>&1
        if expect_head:
            if _ENV_ASSIGNMENT.match(token):
                continue
            head = token.rsplit("/", 1)[-1]
            if head:
                result.append((head, True))
                expect_head = False
            continue
        result.append((token, False))
    return tuple(result)


def _is_redirection_operator(token: str) -> bool:
    return (
        bool(token)
        and all(char in _REDIRECTION_CHARS for char in token)
        and any(char in "<>" for char in token)
    )


__all__ = ["shell_command_prefix_tokens", "shell_command_segments"]
