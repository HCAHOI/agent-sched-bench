from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.baselines.native_priority_aging import (
    PATCHED_SCHEDULER_SHA256,
    patch_scheduler,
    record_admission,
)


def test_one_batch_of_return_admissions_promotes_initial_request(
    tmp_path: Path, monkeypatch
) -> None:
    log = tmp_path / "aging.jsonl"
    monkeypatch.setenv("NATIVE_PRIORITY_AGING_BYPASS_LIMIT", "8")
    monkeypatch.setenv("NATIVE_PRIORITY_AGING_EVENT_LOG", str(log))
    initial = SimpleNamespace(request_id="initial", priority=1)

    for index in range(7):
        assert not record_admission(
            [initial], SimpleNamespace(request_id=f"return-{index}", priority=0)
        )
    assert initial.priority == 1
    assert record_admission(
        [initial], SimpleNamespace(request_id="return-7", priority=0)
    )
    assert initial.priority == 0
    assert json.loads(log.read_text()) == {
        "schema": "native-priority-aging-v1",
        "request_id": "initial",
        "bypass_count": 8,
    }


def test_exact_stock_scheduler_patch(tmp_path: Path) -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        pytest.skip("serving-spike extra is not installed")
    source = Path(spec.origin).parent / "v1/core/sched/scheduler.py"
    scheduler = tmp_path / "scheduler.py"
    shutil.copyfile(source, scheduler)

    assert patch_scheduler(scheduler)
    assert not patch_scheduler(scheduler)
    assert hashlib.sha256(scheduler.read_bytes()).hexdigest() == (
        PATCHED_SCHEDULER_SHA256
    )
