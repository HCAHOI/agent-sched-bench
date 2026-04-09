"""Simulator trace validation tests.

The simulator only supports traces that can be prepared into a real workspace
for local tool replay. This module keeps the early-fail invariant covered
without tying the validation logic to any specific benchmark slug.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCTION_CALL_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "function_call_minimal_header.jsonl"
)
OPENCLAW_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "openclaw_minimal_v5.jsonl"

NEEDS_PREPARE_ERROR = (
    "Simulate mode requires a prepare-able workspace trace; "
    "metadata.needs_prepare was false."
)
TASK_SHAPE_ERROR = (
    "Simulate mode only supports repo-backed swe_patch traces; "
    "got task_shape='function_call'."
)


def _evict(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _isolate_modules():
    _evict(
        "trace_collect.scaffold_registry",
        "agents.miniswe",
        "agents.miniswe.agent",
        "agents.openclaw",
    )
    yield
    _evict(
        "trace_collect.scaffold_registry",
        "agents.miniswe",
        "agents.miniswe.agent",
        "agents.openclaw",
    )


def test_validate_trace_rejects_needs_prepare_false() -> None:
    from trace_collect.simulator import _validate_trace_for_simulation

    with pytest.raises(NotImplementedError, match=NEEDS_PREPARE_ERROR):
        _validate_trace_for_simulation({"needs_prepare": False})


def test_validate_trace_rejects_non_swe_patch_task_shape() -> None:
    from trace_collect.simulator import _validate_trace_for_simulation

    with pytest.raises(NotImplementedError, match="task_shape='function_call'"):
        _validate_trace_for_simulation({"task_shape": "function_call"})


def test_validate_trace_accepts_swe_patch_and_empty_metadata() -> None:
    from trace_collect.simulator import _validate_trace_for_simulation

    _validate_trace_for_simulation(None)
    _validate_trace_for_simulation({})
    _validate_trace_for_simulation({"scaffold": "miniswe", "task_shape": "swe_patch"})


def test_load_trace_metadata_extracts_function_call_header() -> None:
    from trace_collect.simulator import _load_trace_metadata

    metadata = _load_trace_metadata(FUNCTION_CALL_FIXTURE)
    assert metadata is not None
    assert metadata["scaffold"] == "openclaw"
    assert metadata["benchmark"] == "function-call-smoke"
    assert metadata["task_shape"] == "function_call"
    assert metadata["needs_prepare"] is False


def test_simulator_rejects_non_prepareable_trace_before_registry_lookup() -> None:
    from trace_collect.simulator import simulate

    coro = simulate(
        source_trace=FUNCTION_CALL_FIXTURE,
        task_source=Path("/tmp/dummy_tasks.json"),
        repos_root=Path("/tmp/dummy_repos"),
        output_dir=Path("/tmp/dummy_out"),
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        model="dummy",
    )

    with pytest.raises(NotImplementedError, match=NEEDS_PREPARE_ERROR):
        asyncio.run(coro)

    assert "agents.openclaw" not in sys.modules
    assert "agents.miniswe" not in sys.modules


def test_validate_trace_does_not_reject_openclaw_fixture() -> None:
    from trace_collect.simulator import _load_trace_metadata, _validate_trace_for_simulation

    metadata = _load_trace_metadata(OPENCLAW_FIXTURE)
    assert metadata is not None
    assert metadata["scaffold"] == "openclaw"
    _validate_trace_for_simulation(metadata)


def test_task_shape_error_is_used_when_needs_prepare_is_absent() -> None:
    from trace_collect.simulator import _validate_trace_for_simulation

    with pytest.raises(NotImplementedError, match=TASK_SHAPE_ERROR):
        _validate_trace_for_simulation({"task_shape": "function_call"})
