"""Tests for SWEBenchRunner local in-container parity path."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

from agents.openclaw.eval.runner import SWEBenchRunner
from agents.openclaw.eval.types import EvalTask


def test_swebench_runner_extracts_patch_from_exec_working_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    target = repo / "main.py"
    target.write_text("print('before')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    target.write_text("print('after')\n", encoding="utf-8")

    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws")

    async def fake_run(**kwargs):
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
        )

    runner._session_runner.run = fake_run  # type: ignore[method-assign]

    task = EvalTask(
        instance_id="encode__httpx-2701",
        problem_statement="fix bug",
        workspace_dir=tmp_path / "runner-ws",
        repo="encode/httpx",
        base_commit="HEAD",
        image_name="swerebench/example",
    )

    result = asyncio.run(
        runner.run_task(
            task,
            container_workspace=None,
            prompt_template="cc_aligned",
            exec_working_dir=str(repo),
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert "diff --git" in result.model_patch
    assert "print('after')" in result.model_patch


def test_swebench_runner_local_patch_extraction_includes_untracked_files(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    untracked = repo / "new_file.py"
    untracked.write_text("print('new')\n", encoding="utf-8")

    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws")

    async def fake_run(**kwargs):
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
        )

    runner._session_runner.run = fake_run  # type: ignore[method-assign]

    task = EvalTask(
        instance_id="encode__httpx-2701",
        problem_statement="fix bug",
        workspace_dir=tmp_path / "runner-ws",
        repo="encode/httpx",
        base_commit="HEAD",
        image_name="swerebench/example",
    )

    result = asyncio.run(
        runner.run_task(
            task,
            container_workspace=None,
            prompt_template="cc_aligned",
            exec_working_dir=str(repo),
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert "diff --git" in result.model_patch
    assert "new_file.py" in result.model_patch
