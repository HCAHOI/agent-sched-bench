"""BFCL-as-openclaw Tool wrappers + BFCL→JSON-Schema normalizer.

Wraps each BFCL task's function specs as :class:`~agents.openclaw.tools.base.Tool`
instances for use with the custom-registry extension point
(``SessionRunner.run(tools=...)``). Schema normalization rewrites
BFCL's non-standard types (``"dict"``, ``"tuple"``, ``"any"``) to
standard JSON Schema before registration. See
``docs/benchmark_plugin_spec.md §11`` for the full wiring pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── Schema normalization ───────────────────────────────────────────────


def _normalize_bfcl_schema(schema: Any) -> dict[str, Any]:
    """Recursively rewrite BFCL's JSON Schema dialect to the standard one.

    Mappings:
    - ``"type": "dict"``  → ``"type": "object"``
    - ``"type": "tuple"`` → ``"type": "array"``
    - ``"type": "any"``   → drops the ``type`` key entirely (validator
      treats an object with no ``type`` as permissive)
    - Walks ``properties`` (for object schemas) and ``items`` (for array
      schemas) recursively so nested schemas are normalized too.

    Unknown non-standard types are dropped with a WARN log. Non-dict
    input is returned as a minimal permissive object schema so callers
    can always pass the result through to openclaw without a pre-check.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    result: dict[str, Any] = dict(schema)
    raw_type = result.get("type")

    if raw_type == "dict":
        result["type"] = "object"
    elif raw_type == "tuple":
        result["type"] = "array"
    elif raw_type == "float":
        # BFCL uses "float" where JSON Schema has "number". Preserve the
        # type info rather than dropping it so prompt-level tool schemas
        # still carry a meaningful constraint.
        result["type"] = "number"
    elif raw_type == "any":
        result.pop("type", None)
    elif isinstance(raw_type, str) and raw_type not in (
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    ):
        logger.warning(
            "BFCL schema: dropping unknown type %r (no JSON Schema equivalent)",
            raw_type,
        )
        result.pop("type", None)

    # Recurse into properties for object-shaped schemas.
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["properties"] = {
            key: _normalize_bfcl_schema(val) for key, val in properties.items()
        }

    # Recurse into items for array-shaped schemas.
    items = result.get("items")
    if isinstance(items, dict):
        result["items"] = _normalize_bfcl_schema(items)

    return result


# ── BFCLNoOpTool ───────────────────────────────────────────────────────


class BFCLNoOpTool(Tool):
    """Openclaw Tool wrapper for a single BFCL function spec.

    Records each invocation in a caller-supplied ``recorder`` list and
    returns ``"OK"`` so openclaw's dispatch path completes cleanly.
    The BFCL runner reads ``recorder`` after the session ends and scores
    it against ground truth via ``_ast_match``. The recorder is shared
    by reference — :func:`build_bfcl_tool_registry` returns a fresh one
    per call to keep concurrent ``run_task`` invocations isolated.
    """

    def __init__(self, spec: dict[str, Any], recorder: list[dict[str, Any]]) -> None:
        self._name = str(spec.get("name", "unknown"))
        self._description = str(spec.get("description", ""))
        self._parameters = _normalize_bfcl_schema(spec.get("parameters", {}))
        self._recorder = recorder

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        # BFCL tools have no real side effects — they just record.
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Always return no errors.

        BFCL's parameter schemas use the non-standard dialect handled
        by ``_normalize_bfcl_schema``, but we still don't want openclaw's
        strict :meth:`Tool.validate_params` to reject the call — the
        scoring layer (``BFCLRunner._ast_match``) is the source of truth
        for whether a predicted call matches ground truth, and it
        checks argument values against the ground-truth alternatives
        directly. A registration-time type mismatch here would block
        the recorder from ever seeing the call.
        """
        return []

    async def execute(self, **kwargs: Any) -> str:
        self._recorder.append({"name": self._name, "arguments": dict(kwargs)})
        return "OK"


# ── Registry builder ───────────────────────────────────────────────────


def build_bfcl_tool_registry(
    task_tools: list[dict[str, Any]],
) -> tuple[ToolRegistry, list[dict[str, Any]]]:
    """Build a :class:`ToolRegistry` from a BFCL task's function specs.

    Returns ``(registry, recorder)`` where ``registry`` has one
    :class:`BFCLNoOpTool` per valid spec and ``recorder`` is the shared
    list that each tool appends to on execution. Non-dict entries are
    dropped with a WARN log.
    """
    registry = ToolRegistry()
    recorder: list[dict[str, Any]] = []
    for spec in task_tools:
        if not isinstance(spec, dict):
            logger.warning(
                "BFCLNoOpTool: dropping non-dict tool spec (got %s): %r",
                type(spec).__name__,
                spec,
            )
            continue
        registry.register(BFCLNoOpTool(spec, recorder))
    return registry, recorder
