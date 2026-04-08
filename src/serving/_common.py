from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def resolve_python_sibling_executable(name: str) -> str:
    """Return the executable next to the active Python interpreter."""
    return str(Path(sys.executable).with_name(name))


def print_or_exec_command(
    *,
    config: Any,
    command: list[str],
    print_only: bool,
) -> None:
    """Emit the resolved command as JSON or replace the current process."""
    if print_only:
        print(json.dumps({"config": asdict(config), "command": command}, indent=2))
        return
    os.execvpe(command[0], command, os.environ)


def write_json_report(report: dict[str, Any], output_path: str | Path) -> None:
    """Write a JSON report to disk with parent directory creation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def first_choice_content(chat_response: dict[str, Any]) -> str:
    """Return the first choice's content, or an empty string when absent."""
    return (
        (((chat_response.get("choices") or [{}])[0].get("message") or {}).get("content"))
        or ""
    )


def append_followup_turn(
    messages: list[dict[str, str]],
    *,
    chat_response: dict[str, Any],
    followup_prompt: str,
) -> list[dict[str, str]]:
    """Append the assistant reply and follow-up user turn to a message list."""
    return [
        *messages,
        {"role": "assistant", "content": first_choice_content(chat_response)},
        {"role": "user", "content": followup_prompt},
    ]


def validate_chat_responses(chat_responses: list[dict[str, Any]]) -> list[str]:
    """Validate that each chat completion returned at least one non-empty choice."""
    errors: list[str] = []
    for index, chat_response in enumerate(chat_responses):
        choices = chat_response.get("choices") or []
        if not choices:
            errors.append(f"chat completion #{index} returned no choices")
            continue
        if not first_choice_content(chat_response):
            errors.append(f"chat completion #{index} returned empty content")
    return errors
