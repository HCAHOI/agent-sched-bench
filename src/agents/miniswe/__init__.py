"""Register the MiniSWE scaffold adapter on import.

The adapter keeps scaffold discovery aligned with the MiniSWE agent package.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MiniSWECodeAgent", "ContextManagedAgent"]

def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from agents.miniswe.agent import ContextManagedAgent, MiniSWECodeAgent

    exports = {
        "MiniSWECodeAgent": MiniSWECodeAgent,
        "ContextManagedAgent": ContextManagedAgent,
    }
    return exports[name]

