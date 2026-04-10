"""Expose the OpenClaw package and register its scaffold adapter."""

def __getattr__(name: str):  # noqa: ANN001
    """Lazy imports to avoid cascading dependency failures."""
    _imports = {
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
    "SWEBenchRunner",
    "EvalTask",
    "EvalResult",
]
