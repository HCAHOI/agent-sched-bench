"""Tests for :mod:`agents.benchmarks.bfcl_tools`."""

from __future__ import annotations

import asyncio
import logging

import pytest

from agents.benchmarks.bfcl_tools import (
    BFCLNoOpTool,
    _normalize_bfcl_schema,
    build_bfcl_tool_registry,
)
from agents.openclaw.tools.registry import ToolRegistry


# ── _normalize_bfcl_schema ─────────────────────────────────────────────


def test_normalize_dict_to_object() -> None:
    """BFCL's ``'type': 'dict'`` becomes standard ``'type': 'object'``."""
    raw = {"type": "dict", "properties": {"a": {"type": "integer"}}}
    result = _normalize_bfcl_schema(raw)
    assert result["type"] == "object"
    assert result["properties"]["a"]["type"] == "integer"


def test_normalize_recursive_nested_dict() -> None:
    """Nested ``dict`` types inside properties are rewritten recursively."""
    raw = {
        "type": "dict",
        "properties": {
            "outer": {
                "type": "dict",
                "properties": {
                    "inner": {"type": "dict", "properties": {}},
                },
            },
        },
    }
    result = _normalize_bfcl_schema(raw)
    assert result["type"] == "object"
    assert result["properties"]["outer"]["type"] == "object"
    assert result["properties"]["outer"]["properties"]["inner"]["type"] == "object"


def test_normalize_tuple_to_array() -> None:
    raw = {"type": "tuple", "items": {"type": "integer"}}
    result = _normalize_bfcl_schema(raw)
    assert result["type"] == "array"
    assert result["items"]["type"] == "integer"


def test_normalize_any_drops_type() -> None:
    """``'type': 'any'`` is BFCL's polymorphic marker; drop the type key
    so openclaw's validator treats the value as permissive."""
    raw = {"type": "any", "description": "polymorphic value"}
    result = _normalize_bfcl_schema(raw)
    assert "type" not in result
    assert result["description"] == "polymorphic value"


def test_normalize_nested_any_inside_object() -> None:
    raw = {
        "type": "dict",
        "properties": {
            "x": {"type": "any"},
            "y": {"type": "integer"},
        },
    }
    result = _normalize_bfcl_schema(raw)
    assert result["type"] == "object"
    assert "type" not in result["properties"]["x"]
    assert result["properties"]["y"]["type"] == "integer"


def test_normalize_unknown_type_warns_and_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A type name with no JSON Schema equivalent is dropped with a WARN."""
    raw = {"type": "bfcl_custom_blob", "description": "frob"}
    with caplog.at_level(logging.WARNING, logger="agents.benchmarks.bfcl_tools"):
        result = _normalize_bfcl_schema(raw)
    assert "type" not in result
    assert any("unknown type" in rec.message for rec in caplog.records)


def test_normalize_non_dict_input_returns_permissive_object() -> None:
    """Degenerate inputs (None, string, list) return an empty object schema
    so the caller can always pass the result straight to openclaw."""
    assert _normalize_bfcl_schema(None)["type"] == "object"
    assert _normalize_bfcl_schema("not a schema")["type"] == "object"
    assert _normalize_bfcl_schema([])["type"] == "object"


# ── BFCLNoOpTool ──────────────────────────────────────────────────────


def test_no_op_tool_records_call() -> None:
    """``BFCLNoOpTool.execute`` appends the call to the shared recorder."""
    recorder: list[dict] = []
    tool = BFCLNoOpTool(
        {
            "name": "add",
            "description": "Adds two numbers",
            "parameters": {"type": "dict", "properties": {"a": {"type": "integer"}}},
        },
        recorder,
    )
    asyncio.run(tool.execute(a=2, b=3))
    asyncio.run(tool.execute(a=5))
    assert recorder == [
        {"name": "add", "arguments": {"a": 2, "b": 3}},
        {"name": "add", "arguments": {"a": 5}},
    ]


def test_no_op_tool_returns_ok() -> None:
    tool = BFCLNoOpTool({"name": "f", "description": "", "parameters": {}}, [])
    assert asyncio.run(tool.execute()) == "OK"


def test_no_op_tool_normalizes_dict_schema_on_construction() -> None:
    tool = BFCLNoOpTool(
        {"name": "f", "description": "", "parameters": {"type": "dict"}},
        [],
    )
    # _normalize_bfcl_schema ran at __init__ → parameters.type is "object"
    assert tool.parameters["type"] == "object"


def test_no_op_tool_validate_params_always_permissive() -> None:
    """BFCL tools delegate argument correctness to ``_ast_match`` at
    scoring time, not to openclaw's strict type validator."""
    tool = BFCLNoOpTool(
        {
            "name": "add",
            "description": "",
            "parameters": {
                "type": "dict",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
        },
        [],
    )
    # Missing required arg → openclaw would normally fail this. BFCL
    # tool override returns no errors so the recorder still sees the call.
    assert tool.validate_params({}) == []
    assert tool.validate_params({"a": "not an integer"}) == []


# ── build_bfcl_tool_registry ───────────────────────────────────────────


def test_build_registry_from_task_tools() -> None:
    specs = [
        {"name": "add", "description": "", "parameters": {}},
        {"name": "sub", "description": "", "parameters": {}},
        {"name": "mul", "description": "", "parameters": {}},
    ]
    registry, recorder = build_bfcl_tool_registry(specs)
    assert isinstance(registry, ToolRegistry)
    assert set(registry.tool_names) == {"add", "sub", "mul"}
    assert recorder == []  # starts empty; populated at execute() time


def test_build_registry_shared_recorder_across_tools() -> None:
    """All BFCLNoOpTool instances in one registry must share the SAME
    recorder list so the runner can read a single chronological call
    log after the session ends."""
    specs = [
        {"name": "a", "description": "", "parameters": {}},
        {"name": "b", "description": "", "parameters": {}},
    ]
    registry, recorder = build_bfcl_tool_registry(specs)
    tool_a = registry.get("a")
    tool_b = registry.get("b")
    assert tool_a is not None and tool_b is not None
    asyncio.run(tool_a.execute(x=1))
    asyncio.run(tool_b.execute(y=2))
    assert recorder == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]


def test_build_registry_survives_malformed_entry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    specs = [
        {"name": "good", "description": "", "parameters": {}},
        "not a dict",
        None,
        {"name": "also_good", "description": "", "parameters": {}},
    ]
    with caplog.at_level(logging.WARNING, logger="agents.benchmarks.bfcl_tools"):
        registry, _ = build_bfcl_tool_registry(specs)  # type: ignore[arg-type]
    assert set(registry.tool_names) == {"good", "also_good"}
    warnings = "\n".join(rec.message for rec in caplog.records)
    assert "non-dict tool spec" in warnings
