from __future__ import annotations

import json
from typing import Any


def parsed_tool_args_from_raw_response(
    raw_response: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Return the first tool-call name and parsed JSON arguments."""
    if not raw_response:
        return None
    message = (raw_response.get("choices") or [{}])[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None

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
