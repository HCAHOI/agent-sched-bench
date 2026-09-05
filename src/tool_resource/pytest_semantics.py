"""Deterministic, name-free work signatures for pytest clauses."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence

_PYTHON = re.compile(r"(?:python|python\d+(?:\.\d+)*|pypy\d*)", re.IGNORECASE)
_EXPRESSION_TOKEN = re.compile(r"\(|\)|\b(?:and|or|not)\b|[^\s()]+")
_IGNORED_FLAGS = {
    "-q",
    "--quiet",
    "-v",
    "--verbose",
    "--disable-warnings",
    "--no-header",
    "--no-summary",
}
_IGNORED_VALUE_OPTIONS = {
    "--tb",
    "--color",
    "--code-highlight",
    "-r",
    "--junitxml",
    "--junit-prefix",
    "--durations",
    "--durations-min",
}
_DIST_MODES = {"no", "load", "loadscope", "loadfile", "loadgroup", "worksteal", "each"}


@dataclass(frozen=True)
class PytestSignature:
    target_shapes: tuple[tuple[str, int], ...]
    maxfail: int | None
    workers: str | None
    dist: str | None
    collect_only: bool
    last_failed: bool
    failed_first: bool
    stepwise: str | None
    k_shape: tuple[str, ...] | None
    k_atoms: int
    m_shape: tuple[str, ...] | None
    m_atoms: int


def is_pytest_invocation(argv: Sequence[str]) -> bool:
    words = tuple(str(word) for word in argv)
    if not words:
        return False
    executable = PurePosixPath(words[0]).name.lower()
    return executable in {"pytest", "py.test"} or (
        len(words) >= 3
        and _PYTHON.fullmatch(executable) is not None
        and words[1:3] == ("-m", "pytest")
    )


def _selection_shape(expression: str) -> tuple[tuple[str, ...], int] | None:
    tokens = _EXPRESSION_TOKEN.findall(expression)
    if not tokens:
        return None
    shape: list[str] = []
    atoms = 0
    balance = 0
    expect_operand = True
    for token in tokens:
        if token == "(":
            if not expect_operand:
                return None
            balance += 1
            shape.append(token)
        elif token == ")":
            if expect_operand or balance == 0:
                return None
            balance -= 1
            shape.append(token)
        elif token == "not":
            if not expect_operand:
                return None
            shape.append(token)
        elif token in {"and", "or"}:
            if expect_operand:
                return None
            expect_operand = True
            shape.append(token)
        else:
            if not expect_operand:
                return None
            atoms += 1
            expect_operand = False
            shape.append("ATOM")
    if expect_operand or balance:
        return None
    return tuple(shape), atoms


def _target_shape(value: str) -> str:
    if "::" in value:
        return "nodeid"
    if value.removesuffix("/").lower().endswith(".py"):
        return "file"
    return "directory"


def parse_pytest(argv: Sequence[str]) -> PytestSignature | None:
    """Parse the frozen Phase-B pytest grammar, failing closed on unknown flags."""

    words = tuple(str(word) for word in argv)
    if not is_pytest_invocation(words):
        return None
    index = 3 if PurePosixPath(words[0]).name.lower() not in {"pytest", "py.test"} else 1
    target_shapes: Counter[str] = Counter()
    maxfail: int | None = None
    workers: str | None = None
    dist: str | None = None
    collect_only = False
    last_failed = False
    failed_first = False
    stepwise: str | None = None
    selections: dict[str, tuple[tuple[str, ...], int] | None] = {"k": None, "m": None}

    def take_value(option: str) -> str | None:
        nonlocal index
        word = words[index]
        if word.startswith(option + "="):
            value = word[len(option) + 1 :]
            return value or None
        index += 1
        if index == len(words) or words[index].startswith("-"):
            return None
        return words[index]

    while index < len(words):
        word = words[index]
        if word == "--":
            return None
        if word in {"-x", "--exitfirst"}:
            maxfail = 1
        elif word == "--maxfail" or word.startswith("--maxfail="):
            value = take_value("--maxfail")
            if value is None or not value.isdigit():
                return None
            maxfail = int(value)
        elif word == "-n" or word.startswith("-n"):
            value = take_value("-n") if word == "-n" else word[2:]
            if value is None or (value != "auto" and not value.isdigit()):
                return None
            workers = value
        elif word == "--numprocesses" or word.startswith("--numprocesses="):
            value = take_value("--numprocesses")
            if value is None or (value != "auto" and not value.isdigit()):
                return None
            workers = value
        elif word == "--dist" or word.startswith("--dist="):
            dist = take_value("--dist")
            if dist not in _DIST_MODES:
                return None
        elif word in {"--collect-only", "--co"}:
            collect_only = True
        elif word in {"--last-failed", "--lf"}:
            last_failed = True
        elif word in {"--failed-first", "--ff"}:
            failed_first = True
        elif word in {"--stepwise", "--sw"}:
            stepwise = "stepwise"
        elif word in {"--stepwise-skip", "--sw-skip"}:
            stepwise = "skip"
        elif len(word) >= 2 and word[:2] in {"-k", "-m"}:
            option = word[:2]
            value = take_value(option) if word == option else word[2:].removeprefix("=")
            shaped = None if value is None else _selection_shape(value)
            if shaped is None:
                return None
            selections[option[1]] = shaped
        elif word in _IGNORED_FLAGS or (
            word.startswith("-")
            and len(word) > 2
            and set(word[1:]) <= {"q", "v", "x"}
        ):
            if "x" in word[1:]:
                maxfail = 1
        elif word.startswith("-r") and len(word) > 2:
            pass
        elif any(
            word == option or word.startswith(option + "=")
            for option in _IGNORED_VALUE_OPTIONS
        ):
            option = next(
                option
                for option in _IGNORED_VALUE_OPTIONS
                if word == option or word.startswith(option + "=")
            )
            if take_value(option) is None:
                return None
        elif word.startswith("-"):
            return None
        else:
            target_shapes[_target_shape(word)] += 1
        index += 1

    k_shape, k_atoms = selections["k"] or (None, 0)
    m_shape, m_atoms = selections["m"] or (None, 0)
    return PytestSignature(
        target_shapes=tuple(sorted(target_shapes.items())),
        maxfail=maxfail,
        workers=workers,
        dist=dist,
        collect_only=collect_only,
        last_failed=last_failed,
        failed_first=failed_first,
        stepwise=stepwise,
        k_shape=k_shape,
        k_atoms=k_atoms,
        m_shape=m_shape,
        m_atoms=m_atoms,
    )


__all__ = ["PytestSignature", "is_pytest_invocation", "parse_pytest"]
