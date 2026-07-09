from __future__ import annotations

import pytest

from trace_collect.command_features import (
    command_group_key,
    make_row_command_key,
    shell_command_heads,
)


@pytest.mark.parametrize(
    ("command", "heads"),
    [
        ("ls -la /tmp", ["ls"]),
        ("find /testbed -type f | xargs grep -l X | head -20", ["find", "xargs", "head"]),
        ("cd /repo && FOO=1 python -m pytest tests/", ["cd", "python"]),
        ("pytest -x; echo done", ["pytest", "echo"]),
        ("/usr/bin/python3 script.py > out.txt 2>&1", ["python3"]),
        ("./run_tests.sh --fast || cat log.txt", ["run_tests.sh", "cat"]),
        ("(cd /tmp && make) | tee build.log", ["cd", "make", "tee"]),
        ("> out.txt cat foo", ["cat"]),
        ("2>&1 ls", ["ls"]),
        ("echo $(date)", ["echo", "date"]),
    ],
)
def test_shell_command_heads(command: str, heads: list[str]) -> None:
    assert shell_command_heads(command) == heads


def test_unparseable_and_empty_commands_yield_no_key() -> None:
    assert shell_command_heads("echo 'unbalanced") == []
    assert command_group_key("exec", "echo 'unbalanced") is None
    assert command_group_key("exec", "   ") is None


def test_command_group_key_is_order_invariant_and_deduped() -> None:
    key_a = command_group_key("exec", "find . | xargs grep X | head")
    key_b = command_group_key("exec", "head -1 f && xargs <x && find .")
    assert key_a == key_b == "exec:find+head+xargs"


def test_make_row_command_key_reads_tool_args() -> None:
    row_key = make_row_command_key("command")
    assert (
        row_key({"tool_name": "exec", "tool_args": {"command": "pytest -x"}})
        == "exec:pytest"
    )
    assert row_key({"tool_name": "exec", "tool_args": None}) is None
    assert row_key({"tool_name": "exec", "tool_args": {"path": "/x"}}) is None
    assert row_key({"tool_name": "exec", "tool_args": {"command": 3}}) is None
    assert row_key({"tool_args": {"command": "ls"}}) is None
