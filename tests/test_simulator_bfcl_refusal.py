"""Phase 6 unit tests: simulator refuses BFCL v4 traces with exact error.

Mirrors the existing refusal pattern at
``src/agents/benchmarks/bfcl_v4.py:294-301`` for the simulate path.

Verifies:
- Loading a BFCL v4 fixture into ``simulate()`` raises NotImplementedError
  with the EXACT message specified by the PRD.
- The refusal happens BEFORE scaffold registry lookup — no agents.*
  package gets imported as a side effect.
- The refusal triggers on any of: task_shape='function_call',
  benchmark='bfcl-v4', needs_prepare=False.
- Non-BFCL v5 traces (mini-swe, openclaw) do NOT trigger the refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BFCL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bfcl_v4_minimal_header.jsonl"
OPENCLAW_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "openclaw_minimal_v5.jsonl"


EXPECTED_ERROR_SUBSTRING = (
    "BFCL v4 traces have task_shape='function_call' with "
    "needs_prepare=False, which the simulator does not support. "
    "Simulate mode requires a prepare-able scaffold."
)


# ---------------------------------------------------------------------------
# Module isolation — evict registry + agents.* between tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _refuse_bfcl_v4_simulate helper — direct unit test
# ---------------------------------------------------------------------------


def test_refuse_bfcl_v4_via_task_shape() -> None:
    """task_shape='function_call' triggers the refusal."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    metadata = {
        "type": "trace_metadata",
        "scaffold": "openclaw",
        "task_shape": "function_call",
    }
    with pytest.raises(NotImplementedError) as exc_info:
        _refuse_bfcl_v4_simulate(metadata)
    assert EXPECTED_ERROR_SUBSTRING in str(exc_info.value)


def test_refuse_bfcl_v4_via_benchmark_slug() -> None:
    """benchmark='bfcl-v4' triggers the refusal."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    metadata = {
        "type": "trace_metadata",
        "scaffold": "openclaw",
        "benchmark": "bfcl-v4",
    }
    with pytest.raises(NotImplementedError) as exc_info:
        _refuse_bfcl_v4_simulate(metadata)
    assert EXPECTED_ERROR_SUBSTRING in str(exc_info.value)


def test_refuse_bfcl_v4_via_underscore_slug() -> None:
    """benchmark='bfcl_v4' (alternative spelling) also triggers."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    metadata = {"benchmark": "bfcl_v4"}
    with pytest.raises(NotImplementedError):
        _refuse_bfcl_v4_simulate(metadata)


def test_refuse_bfcl_v4_via_needs_prepare_false() -> None:
    """needs_prepare=False (forward-compat hook) triggers the refusal."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    metadata = {"scaffold": "openclaw", "needs_prepare": False}
    with pytest.raises(NotImplementedError):
        _refuse_bfcl_v4_simulate(metadata)


def test_refuse_bfcl_v4_does_not_trigger_for_swe_bench() -> None:
    """SWE-bench traces (with prepare) must NOT trigger the refusal."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    metadata = {
        "scaffold": "mini-swe-agent",
        "benchmark": "swe-bench-verified",
        "task_shape": "swe_patch",
    }
    # Should not raise
    _refuse_bfcl_v4_simulate(metadata)


def test_refuse_bfcl_v4_does_not_trigger_for_swe_rebench() -> None:
    """SWE-rebench traces must NOT trigger the refusal."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    metadata = {
        "scaffold": "openclaw",
        "benchmark": "swe-rebench",
        "task_shape": "swe_patch",
    }
    _refuse_bfcl_v4_simulate(metadata)


def test_refuse_bfcl_v4_handles_none_metadata() -> None:
    """If the trace has no metadata at all, the refusal does not trigger.

    A missing metadata record means we don't know what kind of trace
    it is — the downstream scaffold registry lookup will produce a
    descriptive error if the scaffold is unrecognized.
    """
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    _refuse_bfcl_v4_simulate(None)


def test_refuse_bfcl_v4_handles_empty_metadata() -> None:
    """An empty metadata dict doesn't trigger the refusal either."""
    from trace_collect.simulator import _refuse_bfcl_v4_simulate

    _refuse_bfcl_v4_simulate({})


# ---------------------------------------------------------------------------
# Fixture-based integration: load BFCL fixture, assert refusal
# ---------------------------------------------------------------------------


def test_bfcl_v4_fixture_exists() -> None:
    assert BFCL_FIXTURE.exists(), f"missing fixture: {BFCL_FIXTURE}"


def test_load_trace_metadata_extracts_bfcl_v4_header() -> None:
    """The metadata loader correctly reads the BFCL v4 fixture header."""
    from trace_collect.simulator import _load_trace_metadata

    metadata = _load_trace_metadata(BFCL_FIXTURE)
    assert metadata is not None
    assert metadata["scaffold"] == "openclaw"
    assert metadata["benchmark"] == "bfcl-v4"
    assert metadata["task_shape"] == "function_call"
    assert metadata["needs_prepare"] is False


def test_simulator_refuses_bfcl_fixture_before_scaffold_lookup() -> None:
    """End-to-end: feeding the BFCL fixture to simulate() raises with the
    exact message AND does not import any agents.* package as a side effect.

    The refusal happens at the top of simulate() — BEFORE _detect_agent_id,
    BEFORE scaffold registry lookup, BEFORE any prepare adapter callable.
    """
    import asyncio

    from trace_collect.simulator import simulate

    # Build minimal kwargs (most are unused because we expect the call
    # to short-circuit early)
    coro = simulate(
        source_trace=BFCL_FIXTURE,
        task_source=Path("/tmp/dummy_tasks.json"),
        repos_root=Path("/tmp/dummy_repos"),
        output_dir=Path("/tmp/dummy_out"),
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        model="dummy",
    )

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(coro)

    assert EXPECTED_ERROR_SUBSTRING in str(exc_info.value)

    # Refusal must short-circuit BEFORE the scaffold registry triggers
    # any agents.* lazy import. Note: agents.openclaw and agents.miniswe
    # may already be in sys.modules from a prior test in the same
    # process — we just check that the simulator's path didn't NEW-load
    # them. The fixture above evicts them; if they reappear here, the
    # refusal is happening too late.
    assert "agents.openclaw" not in sys.modules, (
        "BFCL refusal triggered after agents.openclaw was loaded — the "
        "refusal must happen BEFORE scaffold registry lookup"
    )
    assert "agents.miniswe" not in sys.modules, (
        "BFCL refusal triggered after agents.miniswe was loaded — the "
        "refusal must happen BEFORE scaffold registry lookup"
    )


def test_simulator_does_not_refuse_openclaw_fixture() -> None:
    """The openclaw v5 fixture (no BFCL markers) must NOT trigger refusal.

    We can't fully run simulate() against the fixture because it would
    need a real repo, but we can verify the refusal stage doesn't fire
    by calling _refuse_bfcl_v4_simulate on the fixture's metadata.
    """
    from trace_collect.simulator import _load_trace_metadata, _refuse_bfcl_v4_simulate

    metadata = _load_trace_metadata(OPENCLAW_FIXTURE)
    assert metadata is not None
    assert metadata["scaffold"] == "openclaw"
    # Should not raise
    _refuse_bfcl_v4_simulate(metadata)
