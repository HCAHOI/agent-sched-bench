"""Scaffold registry for the trace simulator.

Maps scaffold names (read from v5 trace metadata) to async prepare callables
that construct and prepare per-task workspaces. Each registered adapter
returns a `PreparedWorkspace` carrying the prepared `repo_dir` plus an
opaque `cleanup()` callable.

Lazy import design (Pre-mortem B item 2 of trace-sim-vastai-pipeline plan):
importing this module does NOT load any `agents.*` package. The actual
`agents.<scaffold>` import happens only when `get_prepare(scaffold)` is
called for a registered scaffold. This prevents an import cycle between
`trace_collect.simulator` and `agents.miniswe` (which itself imports back
into `trace_collect.scaffold_registry` to register).

Currently registered scaffolds (post Phase 1.5.1):

- ``"miniswe"`` — registered by ``agents/miniswe/__init__.py`` on first
  ``get_prepare("miniswe")`` call (Phase 1).
- ``"openclaw"`` — registered by ``agents/openclaw/__init__.py`` on
  first ``get_prepare("openclaw")`` call (Phase 1.5.1).

New scaffolds register via ``register_scaffold_prepare(name, callable)``
as a side effect of ``import agents.<name>``. Unknown scaffolds raise
``NotImplementedError`` listing the known scaffolds.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass
class SimulatePrepareConfig:
    """Per-call configuration passed to scaffold prepare adapters.

    Adapters ignore fields they don't need; this is the union of
    parameters across all current scaffolds. New scaffolds can add their
    own fields here as the plan evolves.
    """

    agent_id: str
    api_base: str
    model: str
    api_key: str
    command_timeout_s: float
    task_timeout_s: float
    repos_root: Path | None
    max_context_tokens: int


@dataclass
class PreparedWorkspace:
    """Result of a scaffold prepare call.

    Carries the prepared repo path plus an opaque cleanup callable. The
    cleanup closure typically captures the scaffold-specific agent
    instance (e.g. `MiniSWECodeAgent._workdir`) and is responsible for
    `shutil.rmtree`'ing the workdir on exit.
    """

    repo_dir: Path
    cleanup: Callable[[], None]


# Adapter callable signature
ScaffoldPrepareCallable = Callable[
    [dict[str, Any], SimulatePrepareConfig],
    Awaitable[PreparedWorkspace],
]


SCAFFOLD_PREPARE_REGISTRY: dict[str, ScaffoldPrepareCallable] = {}


def register_scaffold_prepare(
    name: str, callable_: ScaffoldPrepareCallable
) -> None:
    """Register a scaffold prepare adapter under the given name.

    Idempotent on the same `(name, callable_)` pair: re-registering the
    same name simply replaces the existing entry. Called from
    `agents/<name>/__init__.py` as a side effect of import.
    """
    SCAFFOLD_PREPARE_REGISTRY[name] = callable_


def _ensure_loaded(scaffold: str) -> None:
    """Trigger the side-effect import of `agents.<scaffold>` if needed.

    This is the lazy-import gate that keeps `scaffold_registry` from
    transitively pulling `agents.*` at module load time. The first call
    to `get_prepare("miniswe")` is what makes `agents.miniswe.__init__`
    run its registration logic.

    Silently swallows `ImportError` so `get_prepare` can raise its own
    descriptive error for unknown scaffolds.
    """
    if scaffold in SCAFFOLD_PREPARE_REGISTRY:
        return
    try:
        importlib.import_module(f"agents.{scaffold}")
    except ImportError:
        pass


def get_prepare(scaffold: str) -> ScaffoldPrepareCallable:
    """Look up the prepare adapter for a scaffold.

    Triggers a lazy `import agents.<scaffold>` if the registry has no
    entry for the name yet — this fires the registration side effect
    in `agents/<name>/__init__.py`. Raises `NotImplementedError` with
    a descriptive message listing known scaffolds if the name is still
    unknown after the lazy import attempt.

    Phase 1.5.1 of the trace-sim-vastai-pipeline plan removed the
    openclaw-specific short-circuit because openclaw replay is now
    supported via `src/agents/openclaw/simulate_adapter.py`.
    """
    _ensure_loaded(scaffold)

    if scaffold not in SCAFFOLD_PREPARE_REGISTRY:
        known = sorted(SCAFFOLD_PREPARE_REGISTRY.keys())
        raise NotImplementedError(
            f"Scaffold '{scaffold}' is not registered in "
            f"SCAFFOLD_PREPARE_REGISTRY. Known scaffolds: {known}. Register "
            f"via src/agents/<name>/__init__.py."
        )
    return SCAFFOLD_PREPARE_REGISTRY[scaffold]
