"""Regression tests for vendored Tongyi-DeepResearch runtime guards."""

from __future__ import annotations

import builtins
import http.client
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.tongyi_deepresearch.vendor.file_tools.file_parser import (
    FileParserError,
    SingleFileParser,
)
from agents.tongyi_deepresearch.vendor.file_tools.idp import IDP
from agents.tongyi_deepresearch.vendor.file_tools.utils import save_url_to_local_work_dir
from agents.tongyi_deepresearch.vendor.tool_python import Timeout
from agents.tongyi_deepresearch.vendor.tool_python import PythonInterpreter
from agents.tongyi_deepresearch.vendor.tool_search import Search


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


def test_save_url_to_local_work_dir_sets_network_timeout(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class _Resp:
        status_code = 200
        content = b"payload"

    def _fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.file_tools.utils.requests.get",
        _fake_get,
    )

    out = save_url_to_local_work_dir("https://example.com/file.txt", str(tmp_path))

    assert seen["url"] == "https://example.com/file.txt"
    assert seen["timeout"] == 30
    assert Path(out).read_bytes() == b"payload"


def test_tool_search_sets_https_connection_timeout(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _Resp:
        def read(self):
            return b'{"organic":[{"title":"T","link":"https://example.com","snippet":"S"}]}'

    class _Conn:
        def __init__(self, host, timeout=None):
            seen["host"] = host
            seen["timeout"] = timeout

        def request(self, method, path, payload, headers):
            return None

        def getresponse(self):
            return _Resp()

    monkeypatch.setattr(http.client, "HTTPSConnection", _Conn)
    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.tool_search.SERPER_KEY",
        "test-key",
    )

    result = Search().google_search_with_serp("asyncio")

    assert seen["host"] == "google.serper.dev"
    assert seen["timeout"] == 30
    assert "A Google search for 'asyncio' found 1 results" in result


def test_idp_file_submit_with_path_closes_file_handle(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"pdf")

    closed = {"value": False}

    class _FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            closed["value"] = True

    class _FakeClient:
        def submit_doc_parser_job_advance(self, request, runtime):
            assert request.file_url_object is not None
            return SimpleNamespace(body=SimpleNamespace(data=SimpleNamespace(id="job-1")))

    class _Req:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: _FakeFile())
    monkeypatch.setattr(
        "agents.tongyi_deepresearch.vendor.file_tools.idp.docmind_api20220711_models.SubmitDocParserJobAdvanceRequest",
        _Req,
    )

    idp = object.__new__(IDP)
    idp.client = _FakeClient()

    assert idp.file_submit_with_path(str(file_path)) == "job-1"
    assert closed["value"] is True


def test_tongyi_extra_includes_vendor_runtime_dependencies() -> None:
    import tomllib

    pyproject = Path(os.getcwd()) / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    tongyi = data["project"]["optional-dependencies"]["tongyi"]

    required_prefixes = {
        "beautifulsoup4",
        "ffmpeg-python",
        "json5",
        "lxml",
        "pandas",
        "pdfminer.six",
        "pdfplumber",
        "Pillow",
        "python-docx",
        "python-pptx",
        "requests",
        "scenedetect",
        "tabulate",
        "transformers",
    }

    present = {dep.split(">=")[0].split("<")[0].strip() for dep in tongyi}
    missing = required_prefixes - present
    assert not missing, f"Missing Tongyi runtime deps: {sorted(missing)}"
