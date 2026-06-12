"""Pure-logic tests for tool-outcome segment metadata (exit_code / tool_error).

Covers the conservative text parser (parse_tool_exit_code / detect_tool_error)
and its plumbing into per-segment metadata via _message_segment_metadata.
No GPU / model load — runs on the CPU .venv.
"""

from __future__ import annotations

from serving.recording.backend_hf import _message_segment_metadata
from serving.recording.recording import detect_tool_error, parse_tool_exit_code


def test_parse_exit_code_success() -> None:
    assert parse_tool_exit_code("ok\n\nExit code: 0") == 0


def test_parse_exit_code_nonzero() -> None:
    assert parse_tool_exit_code("boom\nSTDERR:\nbad\n\nExit code: 127") == 127


def test_parse_exit_code_last_wins_on_multiple() -> None:
    # Head+tail truncation can leave more than one marker; trust the tail one.
    assert parse_tool_exit_code("Exit code: 0\n...trunc...\nExit code: 2") == 2


def test_parse_exit_code_absent_returns_none_not_zero() -> None:
    assert parse_tool_exit_code("just some file contents, no marker") is None


def test_parse_exit_code_negative() -> None:
    assert parse_tool_exit_code("killed\n\nExit code: -9") == -9


def test_parse_exit_code_non_string_returns_none() -> None:
    assert parse_tool_exit_code(None) is None
    assert parse_tool_exit_code(123) is None
    assert parse_tool_exit_code("") is None


def test_parse_exit_code_substring_not_matched() -> None:
    # "Exit code:" must be its own line; a mention inside prose is not a marker.
    assert parse_tool_exit_code("The program prints Exit code: when done.") is None


def test_detect_error_from_nonzero_exit() -> None:
    assert detect_tool_error("whatever", exit_code=1) is True


def test_detect_error_false_on_zero_exit() -> None:
    assert detect_tool_error("Error-looking but exit 0", exit_code=0) is False


def test_detect_error_marker_when_no_exit_code() -> None:
    assert detect_tool_error("Error: File not found: /x") is True
    assert detect_tool_error("Error executing command: boom") is True


def test_detect_error_false_for_clean_text() -> None:
    assert detect_tool_error("x=1\nline two") is False


def test_detect_error_none_without_signal() -> None:
    assert detect_tool_error(None) is None
    assert detect_tool_error("") is None


def test_detect_error_substring_not_a_failure() -> None:
    # "Error" mid-output (e.g. a grep hit) is not a tool failure.
    assert detect_tool_error("grep found: raise ValueError('Error')") is False


def test_segment_metadata_exec_failure() -> None:
    msg = {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "exec",
        "content": "STDERR:\nNo such file\n\nExit code: 2",
    }
    meta = _message_segment_metadata(msg)
    assert meta["exit_code"] == 2
    assert meta["tool_error"] is True
    assert meta["name"] == "exec"


def test_segment_metadata_exec_success() -> None:
    msg = {
        "role": "tool",
        "tool_call_id": "c2",
        "name": "exec",
        "content": "hello\n\nExit code: 0",
    }
    meta = _message_segment_metadata(msg)
    assert meta["exit_code"] == 0
    assert meta["tool_error"] is False


def test_segment_metadata_non_exec_error() -> None:
    msg = {
        "role": "tool",
        "tool_call_id": "c3",
        "name": "read_file",
        "content": "Error: File not found: /x",
    }
    meta = _message_segment_metadata(msg)
    assert meta["exit_code"] is None
    assert meta["tool_error"] is True


def test_segment_metadata_non_exec_clean() -> None:
    msg = {
        "role": "tool",
        "tool_call_id": "c4",
        "name": "read_file",
        "content": "line1\nline2",
    }
    meta = _message_segment_metadata(msg)
    assert meta["exit_code"] is None
    assert meta["tool_error"] is False


def test_segment_metadata_absent_on_non_tool_roles() -> None:
    # Only tool-result messages carry the new keys; old readers of other roles
    # see no schema change.
    for role in ("system", "user", "assistant"):
        meta = _message_segment_metadata({"role": role, "content": "hi"})
        assert "exit_code" not in meta
        assert "tool_error" not in meta


def test_persisted_preview_yields_false_not_none() -> None:
    # Over-budget tool results are replaced by a "[tool output persisted]" head
    # preview (helpers.maybe_persist_tool_result): no exit code, no leading
    # "Error" marker. Documented behavior: tool_error=False (no-error-signal),
    # NOT None — locked here per review.
    preview = "[tool output persisted]\nFull output saved to: /tmp/x.txt\nhead..."
    assert parse_tool_exit_code(preview) is None
    assert detect_tool_error(preview) is False
