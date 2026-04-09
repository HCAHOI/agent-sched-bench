from __future__ import annotations

import json
from typing import Any


def parsed_tool_args_from_raw_response(
    raw_response: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Return the first tool-call name and parsed JSON arguments."""
    if not raw_response:
        return None
    message = (raw_response.get("choices") or [{}])[0].get("message")
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function = (tool_calls[0] or {}).get("function") or {}
            tool_name = function.get("name")
            arguments = function.get("arguments")
            if not tool_name or not arguments:
                return None
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            return tool_name, parsed

    message = raw_response.get("message") or {}
    content = message.get("content")
    if not isinstance(content, list):
        return None
    first_tool_use = next(
        (
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ),
        None,
    )
    if first_tool_use is None:
        return None
    tool_name = first_tool_use.get("name")
    arguments = first_tool_use.get("input")
    if not tool_name or not isinstance(arguments, dict):
        return None
    return tool_name, arguments
