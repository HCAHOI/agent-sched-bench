"""Regression tests for vendored Tongyi-DeepResearch runtime guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.tongyi_deepresearch.vendor.file_tools.file_parser import (
    FileParserError,
    SingleFileParser,
)
from agents.tongyi_deepresearch.vendor.tool_python import Timeout
from agents.tongyi_deepresearch.vendor.tool_python import PythonInterpreter


def test_python_interpreter_handles_missing_sandbox_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.tool_python.SANDBOX_FUSION_ENDPOINTS",
        [],
    )
    tool = PythonInterpreter()

    result = tool.call("print('hello')")

    assert result == "[Python Interpreter Error]: No sandbox fusion endpoints configured."


def test_python_interpreter_uses_consistent_five_attempt_retry_budget(
    monkeypatch,
) -> None:
    call_count = {"n": 0}

    def _always_timeout(*args, **kwargs):
        call_count["n"] += 1
        raise Timeout()

    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.tool_python.SANDBOX_FUSION_ENDPOINTS",
        ["endpoint-a"],
    )
    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.tool_python.run_code",
        _always_timeout,
    )

    tool = PythonInterpreter()
    result = tool.call("print('hello')")

    assert call_count["n"] == 5
    assert result == (
        "[Python Interpreter Error] TimeoutError: Execution timed out on endpoint endpoint-a."
    )


def test_single_file_parser_raises_clear_error_for_missing_fallback_parser(
    monkeypatch,
    tmp_path: Path,
) -> None:
    parser = SingleFileParser(cfg={"path": str(tmp_path)})
    image_path = tmp_path / "page.jpg"

    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.file_tools.file_parser.USE_IDP",
        True,
    )

    def _raise_idp_failure(*args, **kwargs):
        raise RuntimeError("idp failed")

    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.file_tools.file_parser.parse_file_by_idp",
        _raise_idp_failure,
    )

    with pytest.raises(FileParserError, match="No parser available for file type: jpg"):
        parser._process_new_file(str(image_path))
