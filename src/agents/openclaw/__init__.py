"""OpenClaw agent core — self-contained port of nanobot's agent engine."""


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


__all__ = [
    "UnifiedProvider",
    "SWEBenchRunner",
    "EvalTask",
    "EvalResult",
]
