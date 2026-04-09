"""Tests for src/trace_collect/attempt_pipeline.run_attempt orchestration."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_image_prep import clear_image_cache  # noqa: E402
from harness.disk_preflight import DiskSpaceError  # noqa: E402
from trace_collect.attempt_pipeline import (  # noqa: E402
    AttemptContext,
    AttemptResult,
    run_attempt,
)


def _make_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="mozilla__bleach-259",
        attempt=1,
        task={"instance_id": "mozilla__bleach-259", "repo": "mozilla/bleach"},
        model="qwen-plus-latest",
        requested_model="qwen-plus",
        scaffold="mini-swe-agent",
        source_image="swerebench/img:latest",
        prompt_template="default",
    )


@pytest.fixture(autouse=True)
def _reset_image_cache() -> None:
    clear_image_cache()
    yield
    clear_image_cache()


@pytest.fixture(autouse=True)
def _mock_image_exists() -> None:
    """Pretend the writable derivative image already exists so no build runs."""

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["podman", "image", "exists"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        yield


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"trace_metadata","scaffold":"mini-swe-agent","trace_format_version":5}\n'
        '{"type":"summary","agent_id":"mozilla__bleach-259","success":true}\n',
        encoding="utf-8",
    )


def test_run_attempt_success_writes_all_six_files(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        ctx.mark_container_ready("fake_container_id_xyz")
        ctx.container_stdout = "hello from container stdout"
        return AttemptResult(
            success=True,
            exit_status="Submitted",
            trace_path=trace_source,
            tool_calls=[
                {"tool": "Bash", "input": {"command": "ls"}, "duration_ms": 12.0}
            ],
            n_iterations=18,
            total_llm_ms=94000.0,
            total_tool_ms=12000.0,
            total_tokens=98088,
        )

    result = asyncio.run(
        run_attempt(ctx, inner=inner, min_free_disk_gb=0.001)
    )

    assert result.success is True
    assert ctx.attempt_dir.exists()
    for name in (
        "trace.jsonl",
        "run_manifest.json",
        "results.json",
        "resources.json",
        "tool_calls.json",
        "container_stdout.txt",
    ):
        assert (ctx.attempt_dir / name).exists(), f"{name} missing"

    manifest = json.loads((ctx.attempt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["task"]["instance_id"] == "mozilla__bleach-259"
    assert manifest["task"]["repo"] == "mozilla/bleach"
    assert manifest["attempt"] == "attempt_1"
    assert manifest["model"]["requested"] == "qwen-plus"
    assert manifest["result_summary"]["exit_code"] == 0
    assert manifest["result_summary"]["total_time"] >= 0.0
    assert manifest["scaffold"] == "mini-swe-agent"
    assert manifest["prompt_template"] == "default"

    results = json.loads((ctx.attempt_dir / "results.json").read_text())
    assert results["instance_id"] == "mozilla__bleach-259"
    assert results["success"] is True
    assert results["container_stdout"]["stdout"] == "hello from container stdout"
    assert results["container_stdout"]["exit_code"] == 0

    tool_calls = json.loads((ctx.attempt_dir / "tool_calls.json").read_text())
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "Bash"

    resources = json.loads((ctx.attempt_dir / "resources.json").read_text())
    assert "samples" in resources

    trace = (ctx.attempt_dir / "trace.jsonl").read_text()
    assert "trace_metadata" in trace


def test_run_attempt_inner_exception_writes_error_manifest(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run_attempt(ctx, inner=inner, min_free_disk_gb=0.001))

    # Even on failure, manifest + claude output files exist
    assert (ctx.attempt_dir / "run_manifest.json").exists()
    manifest = json.loads((ctx.attempt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["result_summary"]["exit_code"] == 1
    assert "boom" in (manifest["result_summary"]["error"] or "")


def test_run_attempt_disk_shortfall_aborts_early(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)

    called = {"inner": False}

    async def inner(ctx: AttemptContext) -> AttemptResult:
        called["inner"] = True
        return AttemptResult(
            success=True, exit_status="ok", trace_path=tmp_path / "nope.jsonl"
        )

    with pytest.raises(DiskSpaceError):
        asyncio.run(run_attempt(ctx, inner=inner, min_free_disk_gb=10**12))

    assert called["inner"] is False
    # attempt_dir should NOT have been created
    assert not ctx.attempt_dir.exists()
