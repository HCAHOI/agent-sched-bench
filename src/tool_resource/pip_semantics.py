"""Deterministic semantics and causal task state for pip install clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, Sequence

from tool_resource.clause_parser import parse_command_clauses

_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_LOCAL_EDITABLE = re.compile(r"^\.(?:\[([A-Za-z0-9._,-]+)\])?$")
_EXIT_CODE = re.compile(r"^Exit code: (-?\d+)$")
_NAME_SEPARATORS = re.compile(r"[-_.]+")
_PYTHON_EXECUTABLE = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")


@dataclass(frozen=True)
class PipInstallSignature:
    interpreter: str
    invocation: str
    requirements: tuple[str, ...]
    package_names: tuple[str, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PipQueryState:
    availability: str
    remaining_packages: tuple[str, ...]


def _canonical_name(value: str) -> str:
    return _NAME_SEPARATORS.sub("-", value).lower()


def _requirement(value: str, *, editable: bool) -> tuple[str, str] | None:
    local = _LOCAL_EDITABLE.fullmatch(value)
    if local:
        extras = local.group(1)
        normalized = "editable:local"
        if extras:
            normalized += "[" + ",".join(sorted(extras.lower().split(","))) + "]"
        return normalized, "editable:local"
    match = _NAME.match(value)
    if match is None or match.end() == 0:
        return None
    name = _canonical_name(match.group())
    suffix = value[match.end() :]
    if suffix and not re.fullmatch(r"(?:\[[A-Za-z0-9._,-]+\])?(?:(?:===|==|~=|!=|<=|>=|<|>).+)?", suffix):
        return None
    normalized = name + suffix.lower().replace(" ", "")
    return (f"editable:{normalized}" if editable else normalized), name


def parse_pip_install(argv: Sequence[str]) -> PipInstallSignature | None:
    """Parse the frozen, intentionally narrow pip-install grammar."""

    words = tuple(str(word) for word in argv)
    if not words:
        return None
    executable = PurePosixPath(words[0]).name.lower()
    if (
        len(words) >= 4
        and _PYTHON_EXECUTABLE.fullmatch(executable)
        and words[1:4] == ("-m", "pip", "install")
    ):
        invocation = "python-module"
        start = 4
    elif len(words) >= 2 and executable in {"pip", "pip3"} and words[1] == "install":
        invocation = "direct"
        start = 2
    else:
        return None

    requirements: list[str] = []
    package_names: list[str] = []
    flags: list[str] = []
    index = start
    while index < len(words):
        word = words[index]
        editable = word in {"-e", "--editable"}
        if word == "--break-system-packages":
            flags.append(word)
            index += 1
            continue
        if editable:
            flags.append("--editable")
            index += 1
            if index == len(words):
                return None
            word = words[index]
        elif word.startswith("-"):
            return None
        parsed = _requirement(word, editable=editable)
        if parsed is None:
            return None
        requirement, package_name = parsed
        requirements.append(requirement)
        package_names.append(package_name)
        index += 1
    if not requirements:
        return None
    return PipInstallSignature(
        interpreter=executable,
        invocation=invocation,
        requirements=tuple(sorted(requirements)),
        package_names=tuple(sorted(package_names)),
        flags=tuple(sorted(flags)),
    )


def _exit_code(tool_result: str) -> int | None:
    lines = [line for line in tool_result.splitlines() if line]
    if not lines or (match := _EXIT_CODE.fullmatch(lines[-1])) is None:
        return None
    return int(match.group(1))


def _installs_system_pip(argv: Sequence[str]) -> bool:
    words = tuple(str(word) for word in argv)
    return (
        len(words) >= 3
        and PurePosixPath(words[0]).name in {"apt", "apt-get"}
        and words[1] == "install"
        and "python3-pip" in words[2:]
    )


class PipTaskState:
    """Facts established by earlier commands in one task container."""

    def __init__(self) -> None:
        self._availability: dict[str, str] = {}
        self._installed: dict[str, set[str]] = {}

    def query(
        self,
        signature: PipInstallSignature,
        *,
        conditional_present: bool = False,
    ) -> PipQueryState:
        availability = (
            "present"
            if conditional_present
            else self._availability.get(signature.interpreter, "unknown")
        )
        installed = self._installed.get(signature.interpreter, set())
        return PipQueryState(
            availability=availability,
            remaining_packages=tuple(
                package
                for package in signature.package_names
                if package not in installed
            ),
        )

    def observe(self, command: str, tool_result: str) -> None:
        exit_code = _exit_code(tool_result)
        if exit_code is None:
            return
        parsed = parse_command_clauses(command)
        clauses = parsed.get("clauses", ())
        signatures = [
            parse_pip_install(clause.get("argv", ()))
            for clause in clauses
            if isinstance(clause, Mapping)
        ]
        if exit_code != 0:
            lower = tool_result.lower()
            if "no module named pip" in lower or re.search(
                r"(?:^|\n).*\bpip3?\b.*(?:not found|no such file)", lower
            ):
                for signature in signatures:
                    if signature is not None:
                        self._availability[signature.interpreter] = "absent"
            return
        for clause, signature in zip(clauses, signatures, strict=True):
            argv = clause.get("argv", ())
            if _installs_system_pip(argv):
                self._availability["python3"] = "present"
            if signature is None:
                continue
            self._availability[signature.interpreter] = "present"
            self._installed.setdefault(signature.interpreter, set()).update(
                signature.package_names
            )


def apt_installs_system_pip(argv: Sequence[str]) -> bool:
    """Return whether a successful apt clause establishes system pip."""

    return _installs_system_pip(argv)


__all__ = [
    "PipInstallSignature",
    "PipQueryState",
    "PipTaskState",
    "apt_installs_system_pip",
    "parse_pip_install",
]
