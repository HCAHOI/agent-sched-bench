from __future__ import annotations

import pytest

from trace_collect.command_features import (
    command_prefix_keys,
    make_row_command_prefix_keys,
    shell_command_heads,
    shell_command_prefix_tokens,
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


@pytest.mark.parametrize(
    ("command", "tokens"),
    [
        ("make -j2 all", ["make", "-j2", "all"]),
        ("cd /repo && FOO=1 python -m pytest", ["cd", "/repo", "&&", "python", "-m", "pytest"]),
        ("pytest -x > log.txt 2>&1", ["pytest", "-x"]),
        ("find . | head -5", ["find", ".", "|", "head", "-5"]),
        # Numeric argument before a plain redirect is kept (only 2>&1 drops fds).
        ("sleep 300 > log", ["sleep", "300"]),
        ("timeout 60 pytest 2>&1", ["timeout", "60", "pytest"]),
    ],
)
def test_shell_command_prefix_tokens(command: str, tokens: list[str]) -> None:
    assert shell_command_prefix_tokens(command) == tokens


def test_prefix_keys_are_nested_and_depth_capped() -> None:
    keys = command_prefix_keys("exec", "make -j12 all clean install", max_depth=3)
    assert keys == ("exec:make", "exec:make -j12", "exec:make -j12 all")
    assert command_prefix_keys("exec", "make", max_depth=4) == ("exec:make",)


def test_prefix_keys_distinguish_flag_values_and_token_order() -> None:
    j2 = command_prefix_keys("exec", "make -j2", max_depth=4)
    j12 = command_prefix_keys("exec", "make -j12", max_depth=4)
    assert j2[0] == j12[0] == "exec:make"
    assert j2[1] != j12[1]
    assert command_prefix_keys("exec", "a b", max_depth=2) != command_prefix_keys(
        "exec", "b a", max_depth=2
    )


def test_unparseable_and_empty_commands_yield_no_keys() -> None:
    assert shell_command_prefix_tokens("echo 'unbalanced") == []
    assert command_prefix_keys("exec", "echo 'unbalanced", max_depth=4) == ()
    assert command_prefix_keys("exec", "   ", max_depth=4) == ()


def test_prefix_keys_reject_invalid_depth() -> None:
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        command_prefix_keys("exec", "ls", max_depth=0)
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        make_row_command_prefix_keys("command", max_depth=0)


def test_make_row_command_prefix_keys_reads_tool_args() -> None:
    row_keys = make_row_command_prefix_keys("command", max_depth=4)
    assert row_keys(
        {"tool_name": "exec", "tool_args": {"command": "pytest -x"}}
    ) == ("exec:pytest", "exec:pytest -x")
    assert row_keys({"tool_name": "exec", "tool_args": None}) == ()
    assert row_keys({"tool_name": "exec", "tool_args": {"path": "/x"}}) == ()
    assert row_keys({"tool_name": "exec", "tool_args": {"command": 3}}) == ()
    assert row_keys({"tool_args": {"command": "ls"}}) == ()
