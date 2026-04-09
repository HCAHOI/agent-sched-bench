"""Tests for task-container request -> entrypoint mode mapping."""

from __future__ import annotations

import json
from pathlib import Path

from trace_collect.runtime.task_container import exec_task_container_entrypoint


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
        runtime="/repo/.venv/bin/python",
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
        runtime="/repo/.venv/bin/python",
        timeout=10,
    )

    assert seen[-1] == "preflight"
