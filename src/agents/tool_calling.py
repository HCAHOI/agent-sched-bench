from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ToolCall:
    """Parsed tool invocation emitted by an agent response."""

    name: str
    args: str


def strip_code_fences(text: str) -> str:
    """Remove surrounding fenced code blocks when present."""
    stripped = text.strip()
    match = re.fullmatch(
        r"```[a-zA-Z0-9_-]*\n(?P<body>.*)\n```", stripped, flags=re.DOTALL
    )
    if match:
        return match.group("body").strip()
    return stripped


def extract_sql_block(text: str) -> str | None:
    """Extract a fenced SQL block from model output."""
    match = re.search(
        r"```sql\n(?P<body>.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not match:
        return None
    return match.group("body").strip()


def parse_tool_call(text: str, allowed_tools: set[str]) -> ToolCall | None:
    """Parse the first balanced `tool_name(...)` invocation in text."""
    best_match: tuple[int, ToolCall] | None = None
    for tool_name in allowed_tools:
        marker = f"{tool_name}("
        start = text.find(marker)
        if start == -1:
            continue
        cursor = start + len(marker)
        depth = 1
        in_single_quote = False
        in_double_quote = False
        escaped = False
        while cursor < len(text):
            char = text[cursor]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        args = text[start + len(marker) : cursor]
                        candidate = ToolCall(name=tool_name, args=args.strip())
                        if best_match is None or start < best_match[0]:
                            best_match = (start, candidate)
                        break
            cursor += 1
    return best_match[1] if best_match else None
