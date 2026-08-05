#!/usr/bin/env python3
"""Build the bounded regular-file template for the physical-state experiment."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
from typing import Any, Iterable

MAX_PATHS = 4096
MAX_BYTES = 512 * 1024 * 1024
RAW_SCHEMA = "sqlglot-physical-state-open-paths-v1"
TEMPLATE_SCHEMA = "sqlglot-physical-state-file-template-v1"
_OPEN_START = re.compile(r"^(?:open|openat|openat2)\(")
_OPEN_RESUMED = re.compile(r"^<\.\.\. (?:open|openat|openat2) resumed>")
_EXEC = re.compile(r'^execve\(("(?:\\.|[^"\\])*").*\)\s+=\s+0$')
_FD_RESULT = re.compile(r"=\s+\d+<(.+)>\s*$")
_BRACKETED_PID = re.compile(r"^\[pid\s+\d+\]\s+(.*)$")
_EXCLUDED_ROOTS = ("/proc", "/sys", "/dev")


def _normalized_path(value: str) -> str | None:
    if (
        not value.startswith("/")
        or posixpath.normpath(value) != value
        or any(char in value for char in "\t\r\n")
    ):
        return None
    path = PurePosixPath(value)
    if any(
        path == PurePosixPath(root) or path.is_relative_to(root)
        for root in _EXCLUDED_ROOTS
    ):
        return None
    return value


def opened_paths(lines: Iterable[str]) -> list[str]:
    """Return resolved successful open/exec paths in first-access order."""
    paths: list[str] = []
    seen = set()
    for line in lines:
        body = line.rstrip("\n")
        bracketed = _BRACKETED_PID.match(body)
        if bracketed is not None:
            body = bracketed.group(1)
        else:
            parts = body.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                body = parts[1]
        value: str | None = None
        if _OPEN_START.match(body) or _OPEN_RESUMED.match(body):
            match = _FD_RESULT.search(body)
            if match is not None:
                value = match.group(1)
        else:
            match = _EXEC.match(body)
            if match is not None:
                decoded = ast.literal_eval(match.group(1))
                value = decoded if isinstance(decoded, str) else None
        value = _normalized_path(value) if value is not None else None
        if value is not None and value not in seen:
            seen.add(value)
            paths.append(value)
    return paths


def bounded_template(
    paths: Iterable[str],
    *,
    max_paths: int = MAX_PATHS,
    max_bytes: int = MAX_BYTES,
) -> dict[str, Any]:
    """Filter existing regular files and enforce the frozen first-access bounds."""
    files = []
    seen_inodes = set()
    total_bytes = 0
    rejected = {
        "missing": 0,
        "not_regular": 0,
        "empty": 0,
        "changed_path": 0,
        "duplicate_inode": 0,
    }
    truncated_by = None
    for value in paths:
        normalized = _normalized_path(value)
        if normalized != value:
            rejected["changed_path"] += 1
            continue
        try:
            info = os.stat(value)
        except OSError:
            rejected["missing"] += 1
            continue
        if not stat.S_ISREG(info.st_mode):
            rejected["not_regular"] += 1
            continue
        if info.st_size <= 0:
            rejected["empty"] += 1
            continue
        inode = (info.st_dev, info.st_ino)
        if inode in seen_inodes:
            rejected["duplicate_inode"] += 1
            continue
        if len(files) >= max_paths:
            truncated_by = "path_count"
            break
        if total_bytes + info.st_size > max_bytes:
            truncated_by = "total_bytes"
            break
        files.append({"path": value, "size_bytes": info.st_size})
        seen_inodes.add(inode)
        total_bytes += info.st_size
    return {
        "schema": TEMPLATE_SCHEMA,
        "limits": {"max_paths": max_paths, "max_bytes": max_bytes},
        "file_count": len(files),
        "total_bytes": total_bytes,
        "truncated_by": truncated_by,
        "rejected": rejected,
        "files": files,
    }


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    parse = subparsers.add_parser("parse-strace")
    parse.add_argument("--strace", type=Path, required=True)
    parse.add_argument("--out", type=Path, required=True)
    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--raw", type=Path, required=True)
    filter_parser.add_argument("--out", type=Path, required=True)
    filter_parser.add_argument("--probe-input", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "parse-strace":
        with args.strace.open(encoding="utf-8") as source:
            paths = opened_paths(source)
        if not paths:
            raise ValueError("strace contained no resolved successful file accesses")
        _write_new(
            args.out,
            json.dumps(
                {"schema": RAW_SCHEMA, "source": str(args.strace), "paths": paths},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    if raw.get("schema") != RAW_SCHEMA or not isinstance(raw.get("paths"), list):
        raise ValueError("raw path artifact schema changed")
    template = bounded_template(raw["paths"])
    if not template["files"]:
        raise ValueError("file discovery produced an empty regular-file template")
    _write_new(args.out, json.dumps(template, indent=2, sort_keys=True) + "\n")
    _write_new(
        args.probe_input,
        "".join(f"{row['size_bytes']}\t{row['path']}\n" for row in template["files"]),
    )


if __name__ == "__main__":
    main()
