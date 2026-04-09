"""Tests for benchmark-owned runtime selection in collector orchestration."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
import threading
from types import SimpleNamespace

from trace_collect.attempt_pipeline import AttemptResult
from trace_collect.collector import (
    _cleanup_task_images,
    _ensure_task_source_ready,
    _run_scaffold_tasks,
    _select_tasks,
)


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"trace_metadata","scaffold":"miniswe","trace_format_version":5}\n',
        encoding="utf-8",
    )


def test_run_scaffold_tasks_uses_benchmark_prompt_default_and_runtime_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, executable="podman": None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, executable="podman": False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda executable="podman": None,
    )

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["prompt_template"] = ctx.prompt_template
            seen["agent_runtime_mode"] = ctx.agent_runtime_mode
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="miniswe",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "prompt_template": "cc_aligned",
        "agent_runtime_mode": "task_container_agent",
    }


def test_run_scaffold_tasks_prompt_override_stays_independent_of_runtime_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, executable="podman": None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, executable="podman": False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda executable="podman": None,
    )

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["prompt_template"] = ctx.prompt_template
            seen["agent_runtime_mode"] = ctx.agent_runtime_mode
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="miniswe",
            prompt_template="default",
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "prompt_template": "default",
        "agent_runtime_mode": "task_container_agent",
    }


def test_select_tasks_preserves_explicit_instance_order() -> None:
    tasks = [
        {"instance_id": "mozilla__bleach-259"},
        {"instance_id": "encode__httpx-2701"},
        {"instance_id": "Kinto__kinto-http.py-384"},
    ]

    selected = _select_tasks(
        tasks,
        instance_ids=[
            "encode__httpx-2701",
            "Kinto__kinto-http.py-384",
        ],
        sample=None,
    )

    assert [task["instance_id"] for task in selected] == [
        "encode__httpx-2701",
        "Kinto__kinto-http.py-384",
    ]


def test_cleanup_task_images_keeps_next_source_image(monkeypatch) -> None:
    removed: list[str] = []
    cached: list[str] = []
    pruned: list[str] = []

    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, executable="podman": removed.append(image) or True,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: cached.append(source_image),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda executable="podman": pruned.append("pruned"),
    )

    _cleanup_task_images(
        instance_id="encode__httpx-2701",
        source_image="shared-image",
        fixed_image="fixed-shared-image",
        keep_source_image="shared-image",
    )

    assert removed == ["fixed-shared-image"]
    assert cached == ["shared-image"]
    assert pruned == ["pruned"]


def test_ensure_task_source_ready_falls_back_after_prefetch_failure(
    monkeypatch,
) -> None:
    seen: list[str] = []
    failed_prefetch: Future[None] = Future()
    failed_prefetch.set_exception(RuntimeError("prefetch boom"))

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, executable="podman": seen.append(source_image),
    )

    _ensure_task_source_ready(
        instance_id="encode__httpx-2701",
        source_image="docker.io/library/img-a",
        prefetched_source_image="docker.io/library/img-a",
        prefetch_future=failed_prefetch,
    )

    assert seen == ["docker.io/library/img-a"]


def test_run_scaffold_tasks_prefetches_next_image_and_cleans_after_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
    )
    events: list[tuple[str, str]] = []
    prefetch_started = threading.Event()
    allow_prefetch_finish = threading.Event()

    def fake_ensure_source_image(image: str, executable: str = "podman") -> None:
        events.append(("ensure_source", image))
        if image == "docker.io/library/img-b":
            prefetch_started.set()
            assert allow_prefetch_finish.wait(timeout=1.0)

    async def fake_run_attempt(ctx, *, inner, min_free_disk_gb, executable="podman"):
        events.append(("run_start", ctx.instance_id))
        if ctx.instance_id == "task-a":
            assert prefetch_started.wait(timeout=1.0)
            allow_prefetch_finish.set()
        ctx.fixed_image = f"fixed-{ctx.source_image}"
        events.append(("run_end", ctx.instance_id))
        return AttemptResult(
            success=True,
            exit_status="ok",
            trace_path=trace_path,
            model_patch=f"diff --git a/{ctx.instance_id} b/{ctx.instance_id}",
        )

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        fake_ensure_source_image,
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_attempt",
        fake_run_attempt,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, executable="podman": events.append(("remove_image", image)) or True,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: events.append(("drop_cache", source_image)),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda executable="podman": events.append(("prune", "done")),
    )

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[
                {"instance_id": "task-a", "image_name": "img-a"},
                {"instance_id": "task-b", "image_name": "img-b"},
                {"instance_id": "task-c", "image_name": "img-c"},
            ],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: (lambda ctx: None),
        )
    )

    assert [event for event in events if event[0] == "ensure_source"] == [
        ("ensure_source", "docker.io/library/img-a"),
        ("ensure_source", "docker.io/library/img-b"),
        ("ensure_source", "docker.io/library/img-c"),
    ]
    assert events.index(("ensure_source", "docker.io/library/img-b")) < events.index(("run_end", "task-a"))
    assert events.index(("run_end", "task-a")) < events.index(("remove_image", "fixed-docker.io/library/img-a"))
    assert events.index(("run_end", "task-b")) < events.index(("remove_image", "fixed-docker.io/library/img-b"))
    assert events.index(("run_end", "task-c")) < events.index(("remove_image", "fixed-docker.io/library/img-c"))


def test_run_scaffold_tasks_reuses_source_image_for_consecutive_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
    )
    events: list[tuple[str, str]] = []

    async def fake_run_attempt(ctx, *, inner, min_free_disk_gb, executable="podman"):
        ctx.fixed_image = f"fixed-{ctx.instance_id}"
        events.append(("run_end", ctx.instance_id))
        return AttemptResult(
            success=True,
            exit_status="ok",
            trace_path=trace_path,
            model_patch=f"diff --git a/{ctx.instance_id} b/{ctx.instance_id}",
        )

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, executable="podman": events.append(("ensure_source", source_image)),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_attempt",
        fake_run_attempt,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, executable="podman": events.append(("remove_image", image)) or True,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: events.append(("drop_cache", source_image)),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda executable="podman": events.append(("prune", "done")),
    )

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[
                {"instance_id": "task-a", "image_name": "shared-image"},
                {"instance_id": "task-b", "image_name": "shared-image"},
            ],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: (lambda ctx: None),
        )
    )

    assert [event for event in events if event == ("remove_image", "docker.io/library/shared-image")] == [
        ("remove_image", "docker.io/library/shared-image")
    ]
    assert events.index(("run_end", "task-a")) < events.index(("remove_image", "fixed-task-a"))
    assert events.index(("remove_image", "fixed-task-a")) < events.index(("run_end", "task-b"))
