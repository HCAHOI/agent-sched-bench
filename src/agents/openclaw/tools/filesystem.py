"""File system tools: read, write, edit, list."""

import asyncio
import base64
import difflib
import mimetypes
import os
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from agents.openclaw.tools.base import Tool
from agents.openclaw.utils.helpers import build_image_content_blocks, detect_image_mime


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        all_dirs = [allowed_dir] + (extra_allowed_dirs or [])
        if not any(_is_under(resolved, d) for d in all_dirs):
            raise PermissionError(
                f"Path {path} is outside allowed directory {allowed_dir}"
            )
    return resolved


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _is_under_pure(path: PurePosixPath, directory: PurePosixPath) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
        container_runtime: dict | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs
        self._container = container_runtime

    def _resolve(self, path: str) -> Path:
        """Local-mode path resolution (host filesystem)."""
        return _resolve_path(
            path, self._workspace, self._allowed_dir, self._extra_allowed_dirs
        )

    # ------------------------------------------------------------------
    # I/O primitives — local
    # ------------------------------------------------------------------

    async def _read_bytes_local(self, path: str) -> bytes:
        fp = self._resolve(path)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not fp.is_file():
            raise IsADirectoryError(f"Not a file: {path}")
        return fp.read_bytes()

    async def _write_bytes_local(self, path: str, data: bytes) -> None:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(data)

    def _list_entries_local(
        self, path: str, recursive: bool
    ) -> list[tuple[str, bool]]:
        dp = self._resolve(path)
        if not dp.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dp.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        entries: list[tuple[str, bool]] = []
        if recursive:
            for item in sorted(dp.rglob("*")):
                rel = str(item.relative_to(dp))
                entries.append((rel, item.is_dir()))
        else:
            for item in sorted(dp.iterdir()):
                entries.append((item.name, item.is_dir()))
        return entries

    # ------------------------------------------------------------------
    # I/O primitives — container
    # ------------------------------------------------------------------

    def _resolve_container_path(self, path_str: str) -> str:
        """Resolve path in container namespace (pure lexical, no host fs access).

        Relative paths are joined against the container workspace (e.g. /testbed).
        Absolute paths are used as-is. '..' and '.' are normalised via os.path.normpath.
        """
        ws = PurePosixPath(str(self._workspace))
        p = PurePosixPath(path_str)
        if not p.is_absolute():
            p = ws / p
        normalized_str = os.path.normpath(str(p))
        if self._allowed_dir:
            all_dirs = [PurePosixPath(str(self._allowed_dir))] + [
                PurePosixPath(str(d)) for d in (self._extra_allowed_dirs or [])
            ]
            if not any(
                _is_under_pure(PurePosixPath(normalized_str), d) for d in all_dirs
            ):
                raise PermissionError(
                    f"Path {path_str} is outside allowed directory {self._allowed_dir}"
                )
        return normalized_str

    async def _read_bytes_container(self, path: str) -> bytes:
        container_path = self._resolve_container_path(path)
        q = shlex.quote(container_path)
        # Sentinel pre-checks: classify errors via tokens we control, not
        # image/locale-dependent stderr text from base64.
        script = (
            f"if [ -d {q} ]; then echo __ISDIR__ >&2; exit 1; fi; "
            f"if [ ! -e {q} ]; then echo __NOENT__ >&2; exit 1; fi; "
            f"if [ ! -r {q} ]; then echo __EACCES__ >&2; exit 1; fi; "
            f"base64 < {q}"
        )
        proc = await asyncio.create_subprocess_exec(
            self._container["executable"],
            "exec",
            "-i",
            self._container["id"],
            "sh",
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if "__ISDIR__" in stderr_text:
                raise IsADirectoryError(f"Not a file: {path}")
            if "__NOENT__" in stderr_text:
                raise FileNotFoundError(f"File not found: {path}")
            if "__EACCES__" in stderr_text:
                raise PermissionError(f"Permission denied: {path}")
            raise IOError(f"Failed to read {path}: {stderr_text[:200]}")
        return base64.b64decode(stdout.strip())

    async def _write_bytes_container(self, path: str, data: bytes) -> None:
        container_path = self._resolve_container_path(path)
        dir_path = str(PurePosixPath(container_path).parent)
        proc = await asyncio.create_subprocess_exec(
            self._container["executable"],
            "exec",
            "-i",
            self._container["id"],
            "sh",
            "-c",
            f"mkdir -p {shlex.quote(dir_path)} && cat > {shlex.quote(container_path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=data)
        if proc.returncode != 0:
            raise IOError(
                f"Failed to write {container_path}: {stderr.decode()[:200]}"
            )

    async def _list_entries_container(
        self, path: str, recursive: bool
    ) -> list[tuple[str, bool]]:
        container_path = self._resolve_container_path(path)
        q = shlex.quote(container_path)
        maxdepth = "" if recursive else "-maxdepth 1 "
        # Sentinel pre-checks mirror _list_entries_local's exists/is_dir order.
        # {{}} renders as literal {} for find -exec.
        cmd = (
            f"if [ ! -e {q} ]; then echo __NOENT__ >&2; exit 1; fi; "
            f"if [ ! -d {q} ]; then echo __NOTDIR__ >&2; exit 1; fi; "
            f"find {q} -mindepth 1 {maxdepth}"
            f"-exec sh -c 'for p; do [ -d \"$p\" ] && echo \"d:$p\" || echo \"f:$p\"; done' _ {{}} +"
        )
        proc = await asyncio.create_subprocess_exec(
            self._container["executable"],
            "exec",
            "-i",
            self._container["id"],
            "sh",
            "-c",
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if "__NOENT__" in stderr_text:
                raise FileNotFoundError(f"Directory not found: {path}")
            if "__NOTDIR__" in stderr_text:
                raise NotADirectoryError(f"Not a directory: {path}")
            raise RuntimeError(stderr_text[:200])

        entries: list[tuple[str, bool]] = []
        prefix_len = len(container_path) + 1  # +1 for trailing /
        for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
            if not line:
                continue
            type_char = line[0]
            full_path = line[2:]
            rel = (
                full_path[prefix_len:]
                if full_path.startswith(container_path + "/")
                else full_path
            )
            entries.append((rel, type_char == "d"))
        return sorted(entries)

    # ------------------------------------------------------------------
    # I/O primitives — dispatch
    # ------------------------------------------------------------------

    async def _read_bytes(self, path: str) -> bytes:
        if self._container:
            return await self._read_bytes_container(path)
        return await self._read_bytes_local(path)

    async def _write_bytes(self, path: str, data: bytes) -> None:
        if self._container:
            await self._write_bytes_container(path, data)
        else:
            await self._write_bytes_local(path, data)

    async def _list_entries(
        self, path: str, recursive: bool
    ) -> list[tuple[str, bool]]:
        if self._container:
            return await self._list_entries_container(path, recursive)
        return self._list_entries_local(path, recursive)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class ReadFileTool(_FsTool):
    """Read file contents with optional line-based pagination."""

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Returns numbered lines. "
            "Use offset and limit to paginate through large files."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"},
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default 1)",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default 2000)",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    def _format_paginated_output(
        self,
        text_content: str,
        offset: int,
        limit: int | None,
        path: str,
    ) -> str:
        """Format paginated read output from text content."""
        all_lines = text_content.splitlines()
        total = len(all_lines)

        if offset < 1:
            offset = 1
        if offset > total:
            return f"Error: offset {offset} is beyond end of file ({total} lines)"

        start = offset - 1
        end = min(start + (limit or self._DEFAULT_LIMIT), total)
        numbered = [
            f"{start + i + 1}| {line}"
            for i, line in enumerate(all_lines[start:end])
        ]
        result = "\n".join(numbered)

        if len(result) > self._MAX_CHARS:
            trimmed, chars = [], 0
            for line in numbered:
                chars += len(line) + 1
                if chars > self._MAX_CHARS:
                    break
                trimmed.append(line)
            end = start + len(trimmed)
            result = "\n".join(trimmed)

        if end < total:
            result += (
                f"\n\n(Showing lines {offset}-{end} of {total}."
                f" Use offset={end + 1} to continue.)"
            )
        else:
            result += f"\n\n(End of file — {total} lines total)"
        return result

    async def execute(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not path:
                return "Error reading file: Unknown path"

            try:
                raw = await self._read_bytes(path)
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            except IsADirectoryError:
                return f"Error: Not a file: {path}"

            if not raw:
                return f"(Empty file: {path})"

            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if mime and mime.startswith("image/"):
                return build_image_content_blocks(
                    raw, mime, path, f"(Image file: {path})"
                )

            try:
                text_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                return f"Error: Cannot read binary file {path} (MIME: {mime or 'unknown'}). Only UTF-8 text and images are supported."

            return self._format_paginated_output(text_content, offset, limit, path)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class WriteFileTool(_FsTool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(
        self, path: str | None = None, content: str | None = None, **kwargs: Any
    ) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if content is None:
                raise ValueError("Unknown content")

            await self._write_bytes(path, content.encode("utf-8"))
            return f"Successfully wrote {len(content)} bytes to {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """Locate old_text in content: exact first, then line-trimmed sliding window.

    Both inputs should use LF line endings (caller normalises CRLF).
    Returns (matched_fragment, count) or (None, 0).
    """
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


class EditFileTool(_FsTool):
    """Edit a file by replacing text with fallback matching."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. "
            "Supports minor whitespace/line-ending differences. "
            "Set replace_all=true to replace every occurrence."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {
                    "type": "string",
                    "description": "The text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace with",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if old_text is None:
                raise ValueError("Unknown old_text")
            if new_text is None:
                raise ValueError("Unknown new_text")

            try:
                raw = await self._read_bytes(path)
            except FileNotFoundError:
                return f"Error: File not found: {path}"

            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            if match is None:
                return self._not_found_msg(old_text, content, path)
            if count > 1 and not replace_all:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context to make it unique, or set replace_all=true."
                )

            norm_new = new_text.replace("\r\n", "\n")
            new_content = (
                content.replace(match, norm_new)
                if replace_all
                else content.replace(match, norm_new, 1)
            )
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            await self._write_bytes(path, new_content.encode("utf-8"))
            return f"Successfully edited {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> str:
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(
                None, old_lines, lines[i : i + window]
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
            return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        return f"Error: old_text not found in {path}. No similar text found. Verify the file content."


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


class ListDirTool(_FsTool):
    """List directory contents with optional recursion."""

    _DEFAULT_MAX = 200
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

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. "
            "Set recursive=true to explore nested structure. "
            "Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"},
                "recursive": {
                    "type": "boolean",
                    "description": "Recursively list all files (default false)",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum entries to return (default 200)",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    @staticmethod
    def _is_ignored(rel: str, is_dir: bool, recursive: bool) -> bool:
        parts = rel.replace("\\", "/").split("/")
        if recursive:
            return any(p in ListDirTool._IGNORE_DIRS for p in parts)
        leaf = parts[-1] if parts else ""
        return leaf in ListDirTool._IGNORE_DIRS

    async def execute(
        self,
        path: str | None = None,
        recursive: bool = False,
        max_entries: int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if path is None:
                raise ValueError("Unknown path")

            try:
                entries = await self._list_entries(path, recursive)
            except FileNotFoundError:
                return f"Error: Directory not found: {path}"
            except NotADirectoryError:
                return f"Error: Not a directory: {path}"
            except RuntimeError as e:
                return f"Error: {e}"

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            for rel, is_dir in entries:
                if self._is_ignored(rel, is_dir, recursive):
                    continue
                total += 1
                if len(items) < cap:
                    if recursive:
                        items.append(f"{rel}/" if is_dir else rel)
                    else:
                        pfx = "📁 " if is_dir else "📄 "
                        items.append(f"{pfx}{rel}")

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"
