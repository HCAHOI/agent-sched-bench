from __future__ import annotations

import json

from trace_collect._raw_response import parsed_tool_args_from_raw_response
from trace_collect.openclaw_tools import _unwrap_tool_args


def _raw_response(arguments: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                }
            }
        ]
    }


def test_parsed_tool_args_from_raw_response_extracts_first_tool_call() -> None:
    raw_response = _raw_response(
        {
            "path": "/tmp/source/task-1/file.py",
            "old_text": "before\n",
            "new_text": "after\n",
        }
    )

    parsed = parsed_tool_args_from_raw_response(raw_response)

    assert parsed == (
        "edit_file",
        {
            "path": "/tmp/source/task-1/file.py",
            "old_text": "before\n",
            "new_text": "after\n",
        },
    )


def test_unwrap_tool_args_falls_back_to_raw_response_on_malformed_json() -> None:
    raw_response = _raw_response(
        {
            "path": "/tmp/source/task-1/file.py",
            "old_text": "before\n",
            "new_text": "after\n",
        }
    )

    resolved_tool_name, params, nested = _unwrap_tool_args(
        tool_name="edit_file",
        tool_args_json='{"edit_file": {"new_text": "unterminated}',
        raw_response=raw_response,
    )

    assert resolved_tool_name == "edit_file"
    assert params["path"] == "/tmp/source/task-1/file.py"
    assert params["new_text"] == "after\n"
    assert nested is True
