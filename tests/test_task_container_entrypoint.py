"""Tests for task-container request -> entrypoint mode mapping."""

from __future__ import annotations

import io
import json
from pathlib import Path

from trace_collect.runtime.task_container import (
    exec_task_container_entrypoint,
    write_task_container_request,
)


class _FakeStdin:
    def __init__(self) -> None:
        self.value = ""
        self.closed = False

    def write(self, data: str) -> int:
        self.value += data
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(self, cmd, **kwargs) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdin = _FakeStdin()
        self.stdout = io.StringIO(kwargs.pop("stdout_text", ""))
        self.stderr = io.StringIO(kwargs.pop("stderr_text", ""))
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def test_exec_task_container_entrypoint_uses_run_mode_for_scaffold_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"kind": "run_openclaw", "result_path": str(tmp_path / "r.json")}),
        encoding="utf-8",
    )
    seen: list[str] = []

    def fake_popen(cmd, **kwargs):
        seen.extend(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.subprocess.Popen", fake_popen
    )

    exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        runtime="/usr/bin/python3",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    assert seen[-1] == "run"


def test_exec_task_container_entrypoint_uses_preflight_mode_for_preflight_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"kind": "preflight", "result_path": str(tmp_path / "r.json")}),
        encoding="utf-8",
    )
    seen: list[str] = []

    def fake_run(cmd, **kwargs):
        seen.extend(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        runtime="/usr/bin/python3",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    assert seen[-1] == "preflight"


def test_write_task_container_request_redacts_api_keys(tmp_path: Path) -> None:
    path = write_task_container_request(
        attempt_dir=tmp_path,
        scaffold="openclaw",
        payload={
            "kind": "run_openclaw",
            "api_key": "secret-value",
            "nested": {"api_key": "nested-secret"},
            "result_path": str(tmp_path / "r.json"),
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == "***REDACTED***"
    assert payload["nested"]["api_key"] == "***REDACTED***"


def test_exec_task_container_entrypoint_uses_original_payload_for_stdin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = write_task_container_request(
        attempt_dir=tmp_path,
        scaffold="openclaw",
        payload={
            "kind": "run_openclaw",
            "api_key": "secret-value",
            "result_path": str(tmp_path / "r.json"),
        },
    )
    seen: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        proc = _FakePopen(cmd, **kwargs)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.subprocess.Popen", fake_popen
    )

    exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        request_payload={
            "kind": "run_openclaw",
            "api_key": "secret-value",
            "result_path": str(tmp_path / "r.json"),
        },
        runtime="/usr/bin/python3",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    proc = seen["proc"]
    assert isinstance(proc, _FakePopen)
    assert "***REDACTED***" not in proc.stdin.value
    assert "secret-value" in proc.stdin.value


def test_exec_task_container_entrypoint_streams_run_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"kind": "run_openclaw", "result_path": str(tmp_path / "r.json")}),
        encoding="utf-8",
    )

    def fake_popen(cmd, **kwargs):
        proc = _FakePopen(cmd, **kwargs)
        proc.stdout = io.StringIO("line 1\nline 2\n")
        return proc

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.subprocess.Popen", fake_popen
    )

    result = exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        runtime="/usr/bin/python3",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    assert result.stdout == "line 1\nline 2\n"
    assert "line 1\nline 2\n" in capsys.readouterr().out
