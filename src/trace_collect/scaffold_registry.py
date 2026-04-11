"""Lazy scaffold-prepare registry for the trace simulator."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass
class SimulateLLMConfig:
    """LLM settings needed by prepare adapters that instantiate agents."""

    api_base: str
    model: str
    api_key: str


@dataclass
class SimulatePrepareConfig:
    """Inputs accepted by scaffold prepare adapters."""

    agent_id: str
    llm: SimulateLLMConfig | None
    command_timeout_s: float
    task_timeout_s: float
    repos_root: Path | None
    max_context_tokens: int


@dataclass
class PreparedWorkspace:
    """Prepared repo path plus scaffold-owned cleanup."""

    repo_dir: Path
    cleanup: Callable[[], None]


ScaffoldPrepareCallable = Callable[
    [dict[str, Any], SimulatePrepareConfig],
    Awaitable[PreparedWorkspace],
]


SCAFFOLD_PREPARE_REGISTRY: dict[str, ScaffoldPrepareCallable] = {}


def register_scaffold_prepare(
    name: str, callable_: ScaffoldPrepareCallable
) -> None:
    """Register or replace a scaffold prepare adapter."""
    SCAFFOLD_PREPARE_REGISTRY[name] = callable_


def _ensure_loaded(scaffold: str) -> None:
    """Import ``agents.<scaffold>`` on first lookup to trigger registration."""
    if scaffold in SCAFFOLD_PREPARE_REGISTRY:
        return
    try:
        importlib.import_module(f"agents.{scaffold}")
    except ImportError:
        pass


def get_prepare(scaffold: str) -> ScaffoldPrepareCallable:
    """Return the registered prepare adapter for ``scaffold``."""
    _ensure_loaded(scaffold)

    if scaffold not in SCAFFOLD_PREPARE_REGISTRY:
        known = sorted(SCAFFOLD_PREPARE_REGISTRY.keys())
        raise NotImplementedError(
            f"Scaffold '{scaffold}' is not registered in "
            f"SCAFFOLD_PREPARE_REGISTRY. Known scaffolds: {known}. Register "
            f"via src/agents/<name>/__init__.py."
        )
    return SCAFFOLD_PREPARE_REGISTRY[scaffold]
