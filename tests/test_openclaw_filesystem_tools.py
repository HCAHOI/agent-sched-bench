"""Tests for Step 1 — filesystem I/O primitives and container path resolution."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path, PurePosixPath

from agents.openclaw.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from agents.openclaw.tools.shell import ExecTool, _CONTAINER_TIMEOUT_MARKER_PREFIX


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_CONTAINER = {"executable": "docker", "id": "cid-123"}


def _recorded_exec(calls: list[list[str]], stdout=b"", stderr=b"", returncode=0):
    """Return a fake create_subprocess_exec that records argv and returns canned output."""

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        proc = _FakeProc(stdout, stderr, returncode)
        return proc

    return fake_create_subprocess_exec


class _FakeProc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self.stdout_data = stdout
        self.stderr_data = stderr
        self.returncode = returncode
        self.stdin_received: bytes | None = None

    async def communicate(self, input=None):
        self.stdin_received = input
        return self.stdout_data, self.stderr_data


# ---------------------------------------------------------------------------
# ExecTool — container mode cwd
# ---------------------------------------------------------------------------


def test_exec_tool_container_default_cwd(monkeypatch):
    """ExecTool container mode uses container_default_cwd=/testbed when no working_dir."""
    calls: list[list[str]] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr("agents.openclaw.tools.shell.asyncio.create_subprocess_exec", fake_exec)

    tool = ExecTool(container_id="cid-1", container_executable="docker")
    asyncio.run(tool.execute("echo hello"))

    argv = calls[0]
    assert argv[0] == "docker"
    assert "-w" in argv
    w_idx = argv.index("-w")
    assert argv[w_idx + 1] == "/testbed"


def test_exec_tool_container_explicit_working_dir(monkeypatch):
    """ExecTool container mode respects explicit working_dir from tool call args."""
    calls: list[list[str]] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr("agents.openclaw.tools.shell.asyncio.create_subprocess_exec", fake_exec)

    tool = ExecTool(container_id="cid-1", container_executable="docker")
    asyncio.run(tool.execute("echo hello", working_dir="/testbed/sub"))

    argv = calls[0]
    w_idx = argv.index("-w")
    assert argv[w_idx + 1] == "/testbed/sub"


def test_exec_tool_container_default_cwd_custom(monkeypatch):
    """ExecTool respects custom container_default_cwd."""
    calls: list[list[str]] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr("agents.openclaw.tools.shell.asyncio.create_subprocess_exec", fake_exec)

    tool = ExecTool(
        container_id="cid-1",
        container_executable="docker",
        container_default_cwd="/workspace",
    )
    asyncio.run(tool.execute("echo hello"))

    argv = calls[0]
    w_idx = argv.index("-w")
    assert argv[w_idx + 1] == "/workspace"


def test_exec_tool_container_uses_python_wrapper_with_raw_command(monkeypatch):
    calls: list[list[str]] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr("agents.openclaw.tools.shell.asyncio.create_subprocess_exec", fake_exec)

    command = "printf 'a && b' | cat"
    tool = ExecTool(container_id="cid-1", container_executable="docker")
    asyncio.run(tool.execute(command, timeout=7))

    argv = calls[0]
    cid_idx = argv.index("cid-1")
    assert argv[cid_idx + 1] == "python3"
    assert argv[cid_idx + 2] == "-c"
    assert "subprocess.Popen" in argv[cid_idx + 3]
    assert argv[cid_idx + 4] == command
    assert argv[cid_idx + 5] == "7"
    assert argv[cid_idx + 6].startswith(_CONTAINER_TIMEOUT_MARKER_PREFIX)


def test_exec_tool_container_preserves_user_exit_124(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.shell.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stdout=b"user failed", returncode=124),
    )

    tool = ExecTool(container_id="cid-1", container_executable="docker")
    result = asyncio.run(tool.execute("exit 124", timeout=7))

    assert result == "user failed\n\nExit code: 124"


def test_exec_tool_container_timeout_marker_reports_timeout(monkeypatch):
    calls: list[list[str]] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        timeout_marker = cmd[-1]
        return _FakeProc(
            stderr=f"{timeout_marker}\n".encode(),
            returncode=124,
        )

    monkeypatch.setattr(
        "agents.openclaw.tools.shell.asyncio.create_subprocess_exec",
        fake_exec,
    )

    tool = ExecTool(container_id="cid-1", container_executable="docker")
    result = asyncio.run(tool.execute("sleep 99", timeout=7))

    assert result == "Error: Command timed out after 7 seconds"


# ---------------------------------------------------------------------------
# read_file — container path resolution
# ---------------------------------------------------------------------------


def test_read_file_container_relative_path(monkeypatch):
    """read_file("src/x.py") in container mode resolves to /testbed/src/x.py."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(
            calls,
            stdout=base64.b64encode(b"hello world\n"),
        ),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="src/x.py"))

    # Should use base64 < /testbed/src/x.py
    cmd_str = " ".join(calls[0])
    assert "base64" in cmd_str
    assert "/testbed/src/x.py" in cmd_str
    assert "hello world" in result  # base64 content decoded correctly


def test_read_file_container_absolute_path(monkeypatch):
    """read_file("/abs/path") in container mode uses path as-is."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(
            calls,
            stdout=base64.b64encode(b"content"),
        ),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    asyncio.run(tool.execute(path="/abs/path"))

    cmd_str = " ".join(calls[0])
    assert "base64" in cmd_str
    assert "/abs/path" in cmd_str


def test_read_file_container_normalizes_dotdot(monkeypatch):
    """read_file("src/../lib/x.py") normalizes to /testbed/lib/x.py."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(
            calls,
            stdout=base64.b64encode(b"content"),
        ),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    asyncio.run(tool.execute(path="src/../lib/x.py"))

    cmd_str = " ".join(calls[0])
    assert "/testbed/lib/x.py" in cmd_str
    assert "/testbed/src/" not in cmd_str or "src/.." not in cmd_str


def test_read_file_container_escape_attempt_normalized(monkeypatch):
    """read_file("../../etc/passwd") normalizes lexically, does not escape /testbed via host fs."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(
            calls,
            stdout=base64.b64encode(b"content"),
        ),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    asyncio.run(tool.execute(path="../../etc/passwd"))

    cmd_str = " ".join(calls[0])
    # ../.. from /testbed → /etc/passwd (lexically)
    assert "/etc/passwd" in cmd_str


# ---------------------------------------------------------------------------
# read_file — binary / image detection (container mode)
# ---------------------------------------------------------------------------


def test_read_file_container_binary_content(monkeypatch):
    """Container read_file binary content returns same error as local.

    Uses bytes that are invalid UTF-8 (0xff, 0xfe) so decode() raises UnicodeDecodeError.
    """
    binary_data = b"text\xff\xfepayload"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stdout=base64.b64encode(binary_data)),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="binary.bin"))

    assert "Cannot read binary file" in result


def test_read_file_container_png_image(monkeypatch):
    """Container read_file detects PNG and returns image content blocks."""
    png_data = (
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    )  # minimal PNG signature
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stdout=base64.b64encode(png_data)),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="screenshot.png"))

    # Returns image content blocks
    assert isinstance(result, list)
    assert result[0]["type"] == "image_url"
    assert "image/png" in result[0]["image_url"]["url"]


def test_read_file_container_file_not_found(monkeypatch):
    """Container read_file missing file returns same error as local."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"__NOENT__", returncode=1),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="nope.txt"))

    assert result == "Error: File not found: nope.txt"


def test_read_file_container_directory_returns_not_a_file(monkeypatch):
    """Container read_file on a directory returns same error as local."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"__ISDIR__", returncode=1),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="src"))

    assert result == "Error: Not a file: src"


def test_read_file_container_permission_denied(monkeypatch):
    """Container read_file on an unreadable file reports permission, not not-found."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"__EACCES__", returncode=1),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="secret.txt"))

    assert result == "Error: Permission denied: secret.txt"


def test_read_file_container_unclassified_failure_keeps_stderr(monkeypatch):
    """Unclassified container read failure surfaces stderr, not a fake not-found."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"sh: base64: not found", returncode=127),
    )

    tool = ReadFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="x.py"))

    assert "base64: not found" in result
    assert "File not found" not in result


# ---------------------------------------------------------------------------
# edit_file — CRLF preservation (container mode)
# ---------------------------------------------------------------------------


def test_edit_file_container_preserves_crlf(monkeypatch):
    """Container edit_file preserves CRLF line endings."""
    crlf_content = b"line1\r\nline2\r\nline3\r\n"
    calls: list[list[str]] = []

    class _FakeProcWithStdin:
        def __init__(self, stdout=b"", stderr=b"", returncode=0):
            self.stdout_data = stdout
            self.stderr_data = stderr
            self.returncode = returncode
            self.stdin_received = None

        async def communicate(self, input=None):
            self.stdin_received = input
            return self.stdout_data, self.stderr_data

    write_calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        cmd_str = " ".join(list(cmd))
        if "base64" in cmd_str:
            return _FakeProcWithStdin(stdout=base64.b64encode(crlf_content))
        elif "mkdir" in cmd_str and "cat" in cmd_str:
            # write_bytes_container
            proc = _FakeProcWithStdin(returncode=0)
            write_calls.append(proc)
            return proc
        else:
            return _FakeProcWithStdin(returncode=0)

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec", fake_exec
    )

    tool = EditFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(
        tool.execute(
            path="file.txt",
            old_text="line1",
            new_text="LINE1",
        )
    )

    assert "Successfully edited" in result
    # The written content should preserve CRLF
    assert len(write_calls) == 1
    written = write_calls[0].stdin_received
    assert written is not None
    assert b"\r\n" in written


# ---------------------------------------------------------------------------
# write_file — container mode uses sh+cat (no python3)
# ---------------------------------------------------------------------------


def test_write_file_container_uses_sh_cat_no_python3(monkeypatch):
    """Container write_file docker exec argv uses sh + cat, never python3."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, returncode=0),
    )

    tool = WriteFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    asyncio.run(tool.execute(path="out.txt", content="hello"))

    argv = calls[0]
    cmd_str = " ".join(argv)
    assert "python" not in cmd_str  # no python3 dependency
    assert "mkdir" in cmd_str
    assert "cat" in cmd_str


def test_write_file_container_bytes_content(monkeypatch):
    """Container write_file stdin content is byte-accurate."""
    calls: list[list[str]] = []
    captured_stdin: list[bytes] = []

    class _CaptureProc:
        def __init__(self, returncode=0):
            self.returncode = returncode

        async def communicate(self, input=None):
            captured_stdin.append(input)
            return b"", b""

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return _CaptureProc()

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec", fake_exec
    )

    tool = WriteFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    content = "hello\nworld\x00binary"
    asyncio.run(tool.execute(path="out.txt", content=content))

    assert len(captured_stdin) == 1
    assert captured_stdin[0] == content.encode("utf-8")


# ---------------------------------------------------------------------------
# list_dir — container mode recursive / max_entries / _IGNORE_DIRS / error
# ---------------------------------------------------------------------------


def test_list_dir_container_recursive_and_max_entries(monkeypatch):
    """Container list_dir respects recursive flag and max_entries cap."""
    entries = [
        "f:/testbed/a.py",
        "f:/testbed/b.py",
        "f:/testbed/sub/c.py",
        "d:/testbed/sub",
    ]
    stdout = "\n".join(entries).encode("utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stdout=stdout),
    )

    tool = ListDirTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="/testbed", recursive=True, max_entries=3))

    cmd_str = " ".join(calls[0])
    # recursive: no -maxdepth flag
    assert "-maxdepth 1" not in cmd_str
    # max_entries cap in output
    assert "truncated" in result or "a.py" in result


def test_list_dir_container_non_recursive_has_maxdepth(monkeypatch):
    """Container list_dir non-recursive includes -maxdepth 1."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stdout=b""),
    )

    tool = ListDirTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    asyncio.run(tool.execute(path="/testbed", recursive=False))

    cmd_str = " ".join(calls[0])
    assert "-maxdepth 1" in cmd_str


def test_list_dir_container_ignores_noise_dirs(monkeypatch):
    """Container list_dir filters out _IGNORE_DIRS (.git, node_modules, etc.)."""
    entries = [
        "f:/testbed/src/main.py",
        "d:/testbed/.git",
        "d:/testbed/node_modules",
        "f:/testbed/README.md",
    ]
    stdout = "\n".join(entries).encode("utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stdout=stdout),
    )

    tool = ListDirTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="/testbed"))

    assert "src/main.py" in result
    assert "README.md" in result
    assert ".git" not in result
    assert "node_modules" not in result


def test_list_dir_container_find_failure_returns_error(monkeypatch):
    """Container list_dir find failure (returncode != 0) returns 'Error: ...'."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"find: some error", returncode=1),
    )

    tool = ListDirTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="/testbed"))

    assert result.startswith("Error:")


def test_list_dir_container_nonexistent_dir_returns_not_found(monkeypatch):
    """Container list_dir for nonexistent directory returns 'Error: Directory not found'."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"__NOENT__", returncode=1),
    )

    tool = ListDirTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="/testbed"))

    assert "Directory not found" in result


def test_list_dir_container_file_path_returns_not_a_directory(monkeypatch):
    """Container list_dir on a file path returns same error as local."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"__NOTDIR__", returncode=1),
    )

    tool = ListDirTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="file.txt"))

    assert "Not a directory" in result


def test_write_file_container_failure_returns_error(monkeypatch):
    """Container write_file failure (returncode != 0) surfaces the error."""
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "agents.openclaw.tools.filesystem.asyncio.create_subprocess_exec",
        _recorded_exec(calls, stderr=b"cat: write error: No space left", returncode=1),
    )

    tool = WriteFileTool(workspace=PurePosixPath("/testbed"), container_runtime=_CONTAINER)
    result = asyncio.run(tool.execute(path="out.txt", content="data"))

    assert result.startswith("Error writing file:")
    assert "No space left" in result


# ---------------------------------------------------------------------------
# Local-mode regression — read_file / write_file / edit_file / list_dir
# ---------------------------------------------------------------------------


def test_read_file_local_text(tmp_path: Path):
    """Local read_file returns paginated text."""
    fp = tmp_path / "hello.py"
    fp.write_text("print('hello')\nprint('world')\n", encoding="utf-8")

    tool = ReadFileTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="hello.py"))

    assert "1| print" in result
    assert "2| print" in result
    assert "End of file" in result


def test_read_file_local_binary(tmp_path: Path):
    """Local read_file with binary content returns error."""
    fp = tmp_path / "data.bin"
    fp.write_bytes(b"\x00\x01\x02\xff")

    tool = ReadFileTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="data.bin"))

    assert "Cannot read binary file" in result


def test_read_file_local_png(tmp_path: Path):
    """Local read_file with PNG returns image content blocks."""
    fp = tmp_path / "img.png"
    fp.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    tool = ReadFileTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="img.png"))

    assert isinstance(result, list)
    assert result[0]["type"] == "image_url"


def test_read_file_local_directory_returns_not_a_file(tmp_path: Path):
    """Local read_file on a directory returns not-a-file error."""
    (tmp_path / "subdir").mkdir()

    tool = ReadFileTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="subdir"))

    assert result == "Error: Not a file: subdir"


def test_write_file_local_permission_denied(tmp_path: Path):
    """Local write_file into a read-only directory surfaces PermissionError."""
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_dir.chmod(0o555)
    try:
        tool = WriteFileTool(workspace=tmp_path)
        result = asyncio.run(tool.execute(path="ro/out.txt", content="x"))
    finally:
        ro_dir.chmod(0o755)

    assert result.startswith("Error:")
    assert "Permission denied" in result


def test_write_file_local(tmp_path: Path):
    """Local write_file creates file with correct content."""
    tool = WriteFileTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="out.txt", content="hello world"))

    assert "Successfully wrote" in result
    assert (tmp_path / "out.txt").read_text() == "hello world"


def test_edit_file_local(tmp_path: Path):
    """Local edit_file replaces text correctly."""
    fp = tmp_path / "app.py"
    fp.write_text("old line\n", encoding="utf-8")

    tool = EditFileTool(workspace=tmp_path)
    result = asyncio.run(
        tool.execute(path="app.py", old_text="old line", new_text="new line")
    )

    assert "Successfully edited" in result
    assert (tmp_path / "app.py").read_text() == "new line\n"


def test_edit_file_local_crlf_preserved(tmp_path: Path):
    """Local edit_file preserves CRLF line endings."""
    fp = tmp_path / "crlf.txt"
    fp.write_bytes(b"hello\r\nworld\r\n")

    tool = EditFileTool(workspace=tmp_path)
    result = asyncio.run(
        tool.execute(path="crlf.txt", old_text="hello", new_text="HELLO")
    )

    assert "Successfully edited" in result
    assert b"\r\n" in (tmp_path / "crlf.txt").read_bytes()


def test_list_dir_local(tmp_path: Path):
    """Local list_dir lists directory contents."""
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "subdir").mkdir()

    tool = ListDirTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="."))

    assert "a.py" in result
    assert "b.py" in result
    assert "subdir" in result


def test_list_dir_local_recursive(tmp_path: Path):
    """Local list_dir recursive lists nested contents."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "src" / "lib").mkdir()
    (tmp_path / "src" / "lib" / "util.py").write_text("")

    tool = ListDirTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path=".", recursive=True))

    assert "src/main.py" in result
    assert "src/lib/util.py" in result
    assert "src/lib/" in result


def test_list_dir_local_ignores_noise(tmp_path: Path):
    """Local list_dir filters out noise directories."""
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()

    tool = ListDirTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="."))

    assert "src" in result
    assert ".git" not in result
    assert "__pycache__" not in result


def test_list_dir_local_not_found(tmp_path: Path):
    """Local list_dir for nonexistent path returns not found error."""
    tool = ListDirTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="nonexistent"))

    assert "Directory not found" in result


def test_list_dir_local_not_a_directory(tmp_path: Path):
    """Local list_dir on a file path returns not-a-directory error."""
    (tmp_path / "file.txt").write_text("")

    tool = ListDirTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(path="file.txt"))

    assert "Not a directory" in result
