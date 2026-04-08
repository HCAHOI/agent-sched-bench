"""OpenClaw agent core — self-contained port of nanobot's agent engine.

Side effect on import: registers the openclaw scaffold prepare adapter
into `trace_collect.scaffold_registry` for the trace simulator (Phase
1.5.1 of the trace-sim-vastai-pipeline plan). Importing the adapter
itself is light (the heavy openclaw imports happen lazily inside the
adapter callable's body), so this side effect does not violate the
existing `__getattr__` lazy-load pattern below.
"""


def __getattr__(name: str):  # noqa: ANN001
    """Lazy imports to avoid cascading dependency failures."""
    _imports = {
        "UnifiedProvider": ("agents.openclaw.unified_provider", "UnifiedProvider"),
        "SWEBenchRunner": ("agents.openclaw.eval.runner", "SWEBenchRunner"),
        "EvalTask": ("agents.openclaw.eval.types", "EvalTask"),
        "EvalResult": ("agents.openclaw.eval.types", "EvalResult"),
    }
    if name in _imports:
        module_path, attr = _imports[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_openclaw_prepare() -> None:
    """Register the openclaw scaffold prepare adapter.

    Called as a side effect of `import agents.openclaw`. Both imports
    inside this function are light (no transitive heavy openclaw module
    loads) so the existing lazy-load firewall is preserved.
    """
    from agents.openclaw.simulate_adapter import _openclaw_prepare
    from trace_collect.scaffold_registry import register_scaffold_prepare

    register_scaffold_prepare("openclaw", _openclaw_prepare)


_register_openclaw_prepare()


__all__ = [
    "UnifiedProvider",
    "SWEBenchRunner",
    "EvalTask",
    "EvalResult",
]
