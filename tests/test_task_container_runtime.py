"""Tests for task-container runtime helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trace_collect.runtime.task_container import (
    TaskContainerRunResult,
    current_container_python_runtime,
    preflight_task_container_runtime,
    project_mount_args,
    run_task_container_agent,
)


def test_project_mount_args_include_attempt_dir_and_repo(
    tmp_path: Path,
) -> None:
    args = project_mount_args(tmp_path / "attempt")
    joined = " ".join(args)

    assert str((tmp_path / "attempt").resolve()) in joined
    assert str((Path(__file__).resolve().parents[1]).resolve()) in joined


def test_current_container_python_runtime_keeps_unresolved_venv_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_python = tmp_path / "real-python"
    real_python.write_text("", encoding="utf-8")
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    symlink_python = bin_dir / "python"
    os.symlink(real_python, symlink_python)

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.REPO_VENV_PYTHON",
        symlink_python,
    )

    runtime = Path(current_container_python_runtime())
    assert runtime == symlink_python
    assert runtime.resolve() == real_python


def test_preflight_task_container_runtime_reads_runtime_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "preflight" / "result.json"

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "runtime_proof": {
                        "container_id": "cid-1",
                        "hostname": "host-a",
                        "cwd": "/testbed",
                        "python_executable": "/repo/.venv/bin/python",
                        "python_prefix": "/repo/.venv",
                        "project_root": "/repo",
                        "sys_path": ["/repo/src"],
                    },
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    proof = preflight_task_container_runtime(
        container_id="cid-1",
        attempt_dir=tmp_path,
    )

    assert proof.container_id == "cid-1"
    assert proof.python_executable == "/repo/.venv/bin/python"


def test_run_task_container_agent_reads_result_and_writes_raw_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "miniswe" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "miniswe" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "miniswe" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "miniswe" / "trace.jsonl"

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "n_iterations": 3,
                    "total_llm_ms": 1.0,
                    "total_tool_ms": 2.0,
                    "total_tokens": 4,
                    "runtime_proof": {"hostname": "container-a"},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "stdout text"
            stderr = "stderr text"

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    result = run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        request={
            "scaffold": "miniswe",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert isinstance(result, TaskContainerRunResult)
    assert result.success is True
    assert stdout_path.read_text(encoding="utf-8") == "stdout text"
    assert stderr_path.read_text(encoding="utf-8") == "stderr text"


def test_run_task_container_agent_preserves_existing_raw_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "miniswe" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "miniswe" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "miniswe" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "miniswe" / "trace.jsonl"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("container stdout", encoding="utf-8")
    stderr_path.write_text("container stderr", encoding="utf-8")

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "n_iterations": 3,
                    "total_llm_ms": 1.0,
                    "total_tool_ms": 2.0,
                    "total_tokens": 4,
                    "runtime_proof": {"hostname": "container-a"},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        request={
            "kind": "run_miniswe",
            "scaffold": "miniswe",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert stdout_path.read_text(encoding="utf-8") == "container stdout"
    assert stderr_path.read_text(encoding="utf-8") == "container stderr"


def test_run_task_container_agent_timeout_writes_partial_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "miniswe" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "miniswe" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "miniswe" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "miniswe" / "trace.jsonl"

    def fake_exec(**kwargs):
        raise __import__("subprocess").TimeoutExpired(
            cmd="podman exec ...",
            timeout=10,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    try:
        run_task_container_agent(
            container_id="cid-2",
            timeout=10,
            request={
                "kind": "run_miniswe",
                "scaffold": "miniswe",
                "result_path": str(result_path),
                "trace_file": str(trace_path),
                "raw_stdout_path": str(stdout_path),
                "raw_stderr_path": str(stderr_path),
            },
        )
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected timeout failure")

    assert stdout_path.read_text(encoding="utf-8") == "partial stdout"
    assert stderr_path.read_text(encoding="utf-8") == "partial stderr"
