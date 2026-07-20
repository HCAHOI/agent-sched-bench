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


def test_skip_leading_cd_drops_directory_segments() -> None:
    assert shell_command_prefix_tokens(
        "cd /testbed && python3 -m pytest", skip_leading_cd=True
    ) == ["python3", "-m", "pytest"]
    assert shell_command_prefix_tokens(
        "cd /a && cd /b && make -j2", skip_leading_cd=True
    ) == ["make", "-j2"]
    # Trailing or mid-command cd segments are kept; only leading ones drop.
    assert shell_command_prefix_tokens(
        "make && cd /x", skip_leading_cd=True
    ) == ["make", "&&", "cd", "/x"]


def test_skip_leading_cd_keeps_cd_only_commands() -> None:
    assert shell_command_prefix_tokens("cd /a", skip_leading_cd=True) == ["cd", "/a"]
    assert shell_command_prefix_tokens(
        "cd /a && cd /b", skip_leading_cd=True
    ) == ["cd", "/a", "&&", "cd", "/b"]


def test_skip_leading_cd_changes_prefix_keys() -> None:
    keys = command_prefix_keys(
        "exec", "cd /x && make -j2", max_depth=2, skip_leading_cd=True
    )
    assert keys == ("exec:make", "exec:make -j2")
    # Default remains directory-first.
    assert command_prefix_keys("exec", "cd /x && make -j2", max_depth=2) == (
        "exec:cd",
        "exec:cd /x",
    )


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


# --------------------------------------------------------------------------- #
# Learned wrapper-transparency hook (transparent_wrappers).
# --------------------------------------------------------------------------- #
_HOOK_COMMANDS = [
    "cd /w && pip install -e . && python3 foo.py",
    "make -j2 all",
    "conda activate env && python run.py",
    "make && cd /w",
    "echo 'unbalanced",  # untokenizable
    "cd /a && python x",
]


@pytest.mark.parametrize("command", _HOOK_COMMANDS)
def test_empty_transparent_wrappers_is_byte_identical(command: str) -> None:
    # The empty-set default MUST reproduce the frozen keys byte-for-byte at
    # every layer, so the certified P_0 path is untouched.
    assert shell_command_prefix_tokens(
        command, transparent_wrappers=frozenset()
    ) == shell_command_prefix_tokens(command)
    assert command_prefix_keys(
        "exec", command, max_depth=4, transparent_wrappers=frozenset()
    ) == command_prefix_keys("exec", command, max_depth=4)
    row = {"tool_name": "exec", "tool_args": {"command": command}}
    empty = make_row_command_prefix_keys(
        "command", max_depth=4, transparent_wrappers=frozenset()
    )
    base = make_row_command_prefix_keys("command", max_depth=4)
    assert empty(row) == base(row)


def test_transparent_wrappers_drop_only_nonfinal_matches() -> None:
    # A transparent verb in a NON-FINAL segment is dropped; the final segment
    # is never a candidate even if its head matches.
    assert shell_command_prefix_tokens(
        "cd /w && make x", transparent_wrappers=frozenset({"cd"})
    ) == ["make", "x"]
    assert shell_command_prefix_tokens(
        "make && cd /w", transparent_wrappers=frozenset({"cd"})
    ) == ["make", "&&", "cd", "/w"]
    # Mid-chain wrapper beyond the leading segment is reachable (unlike cd-skip).
    assert shell_command_prefix_tokens(
        "cd /w && pip install && python x", transparent_wrappers=frozenset({"pip"})
    ) == ["cd", "/w", "&&", "python", "x"]


def test_transparent_wrappers_support_consolidation() -> None:
    # Dropping the task-specific cd dir collapses two commands onto one key.
    a = command_prefix_keys(
        "exec", "cd /a && python x", max_depth=4, transparent_wrappers=frozenset({"cd"})
    )
    b = command_prefix_keys(
        "exec", "cd /b && python x", max_depth=4, transparent_wrappers=frozenset({"cd"})
    )
    assert a == b == ("exec:python", "exec:python x")


def test_transparent_wrappers_reject_invalid_set() -> None:
    with pytest.raises(ValueError, match="transparent_wrappers must be non-empty"):
        make_row_command_prefix_keys(
            "command", max_depth=4, transparent_wrappers=frozenset({""})
        )
