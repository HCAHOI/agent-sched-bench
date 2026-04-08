"""Unit tests for trace_collect.scaffold_registry.

Phase 1 of the trace-sim-vastai-pipeline plan. Asserts:
- The registry contains exactly 'miniswe' after the lazy-import side effect fires.
- get_prepare('openclaw') raises NotImplementedError naming Phase 1.5
  WITHOUT importing the agents.openclaw package as a side effect.
- get_prepare('bogus_scaffold') raises NotImplementedError listing the
  known scaffolds.
- Importing trace_collect.scaffold_registry alone does NOT load any
  agents.* package — the load happens only when get_prepare(name) is
  called for that name.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper: fully reset module-level state between tests so each test sees
# a clean import lifecycle. The lazy-import gate's behavior depends on
# whether agents.miniswe / agents.openclaw is already in sys.modules,
# so we evict them per test.
# ---------------------------------------------------------------------------


def _evict_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _isolate_scaffold_registry_modules():
    """Evict scaffold_registry + agents.miniswe + agents.openclaw before
    AND after each test so module state cannot leak across tests.
    """
    _evict_modules(
        "trace_collect.scaffold_registry",
        "agents.miniswe",
        "agents.miniswe.agent",
        "agents.openclaw",
    )
    yield
    _evict_modules(
        "trace_collect.scaffold_registry",
        "agents.miniswe",
        "agents.miniswe.agent",
        "agents.openclaw",
    )


# ---------------------------------------------------------------------------
# Lazy-import invariants (Pre-mortem B item 2)
# ---------------------------------------------------------------------------


def test_importing_scaffold_registry_alone_does_not_load_agents() -> None:
    """Importing scaffold_registry must not transitively pull agents.*."""
    importlib.import_module("trace_collect.scaffold_registry")

    assert "agents.miniswe" not in sys.modules, (
        "scaffold_registry transitively loaded agents.miniswe — the lazy "
        "import gate is broken"
    )
    assert "agents.openclaw" not in sys.modules, (
        "scaffold_registry transitively loaded agents.openclaw — the lazy "
        "import gate is broken"
    )


# ---------------------------------------------------------------------------
# Registry contents and lookup behavior
# ---------------------------------------------------------------------------


def test_miniswe_lookup_triggers_lazy_import_and_returns_callable() -> None:
    """get_prepare('miniswe') triggers agents.miniswe import + returns callable."""
    sr = importlib.import_module("trace_collect.scaffold_registry")

    assert "agents.miniswe" not in sys.modules

    callable_ = sr.get_prepare("miniswe")

    assert callable(callable_), "Returned object must be callable"
    assert "agents.miniswe" in sys.modules, (
        "get_prepare('miniswe') should have triggered the lazy import"
    )
    assert "miniswe" in sr.SCAFFOLD_PREPARE_REGISTRY


def test_openclaw_lookup_triggers_lazy_import_after_phase_1_5_1() -> None:
    """Phase 1.5.1: openclaw lookup must register + return a callable.

    Replaces the Phase-1-only test that asserted a NotImplementedError
    for openclaw. Now that `src/agents/openclaw/simulate_adapter.py`
    exists and registers itself on `import agents.openclaw`, the
    lookup must succeed and return the registered adapter.
    """
    sr = importlib.import_module("trace_collect.scaffold_registry")

    assert "agents.openclaw" not in sys.modules

    callable_ = sr.get_prepare("openclaw")

    assert callable(callable_), (
        "Phase 1.5.1: openclaw lookup must return a callable adapter, "
        "not raise NotImplementedError"
    )
    assert "agents.openclaw" in sys.modules, (
        "get_prepare('openclaw') should have triggered the lazy import "
        "of agents.openclaw, which fires _register_openclaw_prepare"
    )
    assert "openclaw" in sr.SCAFFOLD_PREPARE_REGISTRY


def test_unknown_scaffold_raises_with_known_list() -> None:
    """An unknown scaffold name must raise NotImplementedError listing knowns."""
    sr = importlib.import_module("trace_collect.scaffold_registry")

    # Trigger miniswe registration so we have at least one known scaffold
    sr.get_prepare("miniswe")

    with pytest.raises(NotImplementedError) as exc_info:
        sr.get_prepare("definitely_not_a_real_scaffold_xyz")

    msg = str(exc_info.value)
    assert "definitely_not_a_real_scaffold_xyz" in msg, (
        "Error message must name the missing scaffold"
    )
    assert "miniswe" in msg, (
        f"Error message must list known scaffolds (including miniswe); got: {msg}"
    )
    assert "src/agents/<name>/__init__.py" in msg, (
        "Error message must point to the registration site"
    )


# ---------------------------------------------------------------------------
# PreparedWorkspace + SimulatePrepareConfig dataclass shape
# ---------------------------------------------------------------------------


def test_simulate_prepare_config_required_fields() -> None:
    """SimulatePrepareConfig must accept the union of scaffold-required fields."""
    from pathlib import Path

    sr = importlib.import_module("trace_collect.scaffold_registry")

    # Construct should not raise
    cfg = sr.SimulatePrepareConfig(
        agent_id="test_agent",
        api_base="http://localhost:8000/v1",
        model="test-model",
        api_key="EMPTY",
        command_timeout_s=120.0,
        task_timeout_s=1200.0,
        repos_root=Path("/tmp/repos"),
        max_context_tokens=256_000,
    )
    assert cfg.agent_id == "test_agent"
    assert cfg.repos_root == Path("/tmp/repos")


def test_prepared_workspace_carries_repo_dir_and_cleanup() -> None:
    """PreparedWorkspace must expose repo_dir + cleanup callable."""
    from pathlib import Path

    sr = importlib.import_module("trace_collect.scaffold_registry")

    called = {"count": 0}

    def _cleanup() -> None:
        called["count"] += 1

    pw = sr.PreparedWorkspace(repo_dir=Path("/tmp/fake/repo"), cleanup=_cleanup)
    assert pw.repo_dir == Path("/tmp/fake/repo")

    pw.cleanup()
    pw.cleanup()
    assert called["count"] == 2, "cleanup callable must be invokable multiple times"
