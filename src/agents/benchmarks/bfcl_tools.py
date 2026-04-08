"""BFCL-as-openclaw Tool wrappers + BFCL→JSON-Schema normalizer.

Used by :mod:`agents.benchmarks.bfcl_runner` to wrap each BFCL task's
function specs into :class:`agents.openclaw.tools.base.Tool` instances
that can be registered with an :class:`agents.openclaw.tools.registry.ToolRegistry`
and passed to ``SessionRunner.run(tools=...)`` via the Phase 0
extension point.

BFCL ships parameter schemas using a non-standard JSON Schema dialect
(``type: "dict"`` instead of ``"object"``, ``"tuple"`` instead of
``"array"``, ``"any"`` for polymorphic args). :func:`_normalize_bfcl_schema`
recursively rewrites these to standard JSON Schema before wrapping so
openclaw's :meth:`Tool.validate_params` doesn't reject them.
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

    Single-turn BFCL categories ask the model to emit the right
    structured tool call — not to actually execute it. This wrapper
    records each invocation in a caller-supplied ``recorder`` list and
    returns a neutral acknowledgment string so openclaw's dispatch
    path (``AgentRunner._run_tool`` → ``await tool.execute()``)
    completes cleanly. After the session returns, the BFCL runner reads
    ``recorder`` directly and scores it against ground truth via
    ``_ast_match``.

    Non-reentrant: the ``recorder`` list is shared by reference, so
    callers must instantiate a fresh recorder per ``BFCLRunner.run_task``
    invocation. :func:`build_bfcl_tool_registry` enforces this by
    returning ``(registry, recorder)`` together.
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

    Args:
        task_tools: The ``tools`` field of an :class:`~agents.openclaw.eval.types.EvalTask`
            that was populated by ``BFCLv4Benchmark.normalize_task``.
            Each entry is a BFCL function spec dict
            (``{name, description, parameters}``).

    Returns:
        A pair ``(registry, recorder)``:
        - ``registry`` has one :class:`BFCLNoOpTool` per valid spec and is
          ready to pass as ``tools=`` to ``SessionRunner.run()``.
        - ``recorder`` is an initially-empty list that will be populated
          as the LLM (via openclaw's dispatch loop) invokes each tool.
          After the session ends the runner reads this list directly
          for scoring.

    Non-dict entries in ``task_tools`` are dropped with a WARN log
    (CLAUDE.md §4 "no silent failures").
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
