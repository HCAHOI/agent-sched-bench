"""Trace field helpers for OpenClaw cloud-provider runs."""

from __future__ import annotations

from typing import Any, Mapping


def filter_provider_trace_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return public provider trace telemetry fields.

    Cloud provider-specific allow-lists live at the call site because fields are
    intentionally provider-scoped (for example OpenRouter metadata).
    """
    if not extra:
        return {}
    return {key: value for key, value in extra.items() if not key.startswith("_")}
