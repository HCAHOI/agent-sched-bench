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
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws", benchmark_slug="swe-rebench")

    async def fake_run(**kwargs):
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="completed",
            error=None,
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
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws", benchmark_slug="swe-rebench")

    async def fake_run(**kwargs):
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="completed",
            error=None,
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
            prompt_template="cc_aligned",
            exec_working_dir=str(repo),
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert "diff --git" in result.model_patch
    assert "new_file.py" in result.model_patch


def test_swebench_runner_prefers_patch_txt_and_excludes_submission_artifacts(
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
    tracked = repo / "tracked.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    tracked.write_text("print('after')\n", encoding="utf-8")
    (repo / "patch.txt").write_text(
        "diff --git a/tracked.py b/tracked.py\n"
        "--- a/tracked.py\n"
        "+++ b/tracked.py\n"
        "@@ -1 +1 @@\n"
        "-print('before')\n"
        "+print('after')\n",
        encoding="utf-8",
    )
    (repo / "local.bin").write_text("chunk1chunk2chunk3", encoding="utf-8")

    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws", benchmark_slug="swe-rebench")

    async def fake_run(**kwargs):
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="completed",
            error=None,
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
            prompt_template="cc_aligned",
            exec_working_dir=str(repo),
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert "tracked.py" in result.model_patch
    assert "patch.txt" not in result.model_patch
    assert "local.bin" not in result.model_patch


def test_swebench_runner_propagates_noncompleted_stop_reason(
    tmp_path: Path,
) -> None:
    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws", benchmark_slug="swe-rebench")

    async def fake_run(**kwargs):
        return SimpleNamespace(
            content="I reached the maximum number of tool call iterations.",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="max_iterations",
            error="I reached the maximum number of tool call iterations.",
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
            prompt_template="cc_aligned",
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert result.stop_reason == "max_iterations"
    assert "maximum number of tool call iterations" in (result.error or "")


def test_swebench_runner_passes_tool_workspace_as_project_workspace(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    provider = SimpleNamespace(get_default_model=lambda: "qwen-plus-latest")
    runner = SWEBenchRunner(provider=provider, workspace_base=tmp_path / "ws", benchmark_slug="swe-rebench")

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content="done",
            elapsed_s=0.1,
            trace_file=kwargs["trace_file"],
            session_key=kwargs["session_key"],
            session_manager=None,
            stop_reason="completed",
            error=None,
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
    tool_workspace = tmp_path / "tool-ws"

    asyncio.run(
        runner.run_task(
            task,
            prompt_template="cc_aligned",
            tool_workspace=tool_workspace,
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert captured["tool_workspace"] == tool_workspace
    assert captured["project_workspace"] == tool_workspace


def test_extract_container_patch_via_fake_docker_runner_prefers_patch_txt(
    tmp_path: Path,
) -> None:
    """Fake docker exec runner: patch.txt exists → adopted without git commands."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "main.py").write_text("print('before')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True,
    )
    (repo / "main.py").write_text("print('after')\n", encoding="utf-8")
    (repo / "patch.txt").write_text(
        "diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-print('before')\n+print('after')\n",
        encoding="utf-8",
    )

    argv_log: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        argv_log.append(list(argv))
        export_cmd = [c for c in argv if c != "-w" and c != "/testbed" and c != "cid-1"]
        if argv[:1] == ["cat"]:
            path = argv[1]
            return subprocess.CompletedProcess(
                export_cmd, 0,
                stdout=(repo / path.rsplit("/", 1)[1]).read_text(encoding="utf-8"),
                stderr="",
            )
        return subprocess.CompletedProcess(export_cmd, 0, stdout="", stderr="")

    patch = SWEBenchRunner._extract_container_patch(
        "/testbed", base_commit="HEAD", run=fake_run,
    )

    assert patch is not None
    assert "print('after')" in patch
    assert all("git" not in a[0] for a in argv_log), \
        f"patch.txt present → no git commands; got {argv_log}"
    assert any("cat" in a[0] for a in argv_log)


def test_extract_container_patch_via_fake_docker_runner_no_patch_txt(
    tmp_path: Path,
) -> None:
    """Fake docker exec runner: no patch.txt → safe.directory + git_diff_excluding."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "main.py").write_text("print('before')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True,
    )
    (repo / "main.py").write_text("print('after')\n", encoding="utf-8")

    argv_log: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        argv_log.append(list(argv))
        export_cmd = [c for c in argv if c != "-w" and c != "/testbed" and c != "cid-1"]
        return subprocess.run(export_cmd, cwd=str(repo),
                              capture_output=True, text=True, timeout=30)

    patch = SWEBenchRunner._extract_container_patch(
        "/testbed", base_commit="HEAD", run=fake_run,
    )

    assert patch is not None
    assert "print('after')" in patch
    # Command sequence: cat patch.txt, git config safe.directory, git add -A, git diff
    commands = [[a for a in argv if a not in ("-w", "/testbed", "cid-1")] for argv in argv_log]
    cat_calls = [c for c in commands if c and c[0] == "cat"]
    git_config_calls = [c for c in commands if c and c[0] == "git" and "config" in c]
    git_add_calls = [c for c in commands if c and c[0] == "git" and "add" in c]
    git_diff_calls = [c for c in commands if c and c[0] == "git" and "diff" in c]
    assert len(cat_calls) == 1, f"expected 1 cat call, got {cat_calls}"
    assert "patch.txt" in cat_calls[0][1]
    assert len(git_config_calls) >= 1
    assert any("safe.directory" in c for c in git_config_calls[0])
    assert len(git_add_calls) >= 1
    assert len(git_diff_calls) >= 1
    # base_commit passed to git diff
    diff_argv = git_diff_calls[0]
    assert "HEAD" in diff_argv or diff_argv[2] == "HEAD"


def test_docker_git_diff_excluding_includes_untracked_files_via_stage(
    tmp_path: Path,
) -> None:
    """Untracked files enter diff via git add -A stage (tested via local runner)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True,
    )
    (repo / "new_file.py").write_text("print('new')\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    from agents.openclaw.eval.types import git_diff_excluding, EvalResult

    result = git_diff_excluding(
        str(repo), "HEAD", EvalResult.exclude_pathspecs(), add_excludes=True,
    )
    assert result.returncode == 0
    diff_text = result.stdout
    assert "new_file.py" in diff_text, f"untracked file missing from diff:\n{diff_text}"
    assert "tracked.txt" in diff_text, f"modified file missing from diff:\n{diff_text}"
    assert diff_text.lstrip().startswith("diff --git")
