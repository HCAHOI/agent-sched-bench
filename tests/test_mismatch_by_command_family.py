from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.mismatch_by_command_family import (
    _classify_family,
    _parse_command,
    main,
    _discover_traces,
    _load_tool_execs,
)


# ---------------------------------------------------------------------------
# _classify_family tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pip install foo", "pip_install"),
        ("pip3 install foo", "pip_install"),
        ("sudo pip install foo", "pip_install"),
        ("python -m pip install foo", "pip_install"),
        ("apt install foo", "apt"),
        ("apt-get update", "apt"),
        ("sudo apt-get install -y python3", "apt"),
        ("dpkg -i foo.deb", "apt"),
        ("conda install numpy", "package_other"),
        ("npm install lodash", "package_other"),
        ("yarn add react", "package_other"),
        ("cargo build", "other"),  # cargo install -> package_other, cargo build -> other
        ("cargo install foo", "package_other"),
        ("gem install bundler", "package_other"),
        ("brew install wget", "package_other"),
        ("curl -O https://example.com/file", "network_fetch"),
        ("wget https://example.com/file", "network_fetch"),
        ("git clone https://github.com/user/repo", "network_fetch"),
        ("git fetch origin", "network_fetch"),
        ("git pull upstream main", "network_fetch"),
        ("git submodule update --init", "network_fetch"),
        ("git status", "git_local"),
        ("git diff", "git_local"),
        ("git log", "git_local"),
        ("git checkout main", "git_local"),
        ("git branch", "git_local"),
        ("git commit -m 'msg'", "git_local"),
        ("git push origin main", "git_local"),
        ("pytest tests/", "test"),
        ("tox -e py312", "test"),
        ("python -m pytest tests/", "test"),
        ("make build", "build"),
        ("cmake -B build", "build"),
        ("gcc -o test test.c", "build"),
        ("g++ -std=c++17 main.cpp", "build"),
        ("python setup.py build", "build"),
        ("ls -la", "readonly"),
        ("cat file.txt", "readonly"),
        ("grep -r foo .", "readonly"),
        ("head -n 5 file", "readonly"),
        ("tail -f log", "readonly"),
        ("sed -n 's/foo/bar/p' file", "readonly"),
        ("awk '{print $1}' file", "readonly"),
        ("env FOO=bar ls", "readonly"),
        ("echo hello", "readonly"),
        ("pwd", "readonly"),
        ("ls > out.txt", "other"),  # redirect -> not readonly
        ("echo 'pip install foo'", "pip_install"),  # substring match by pip_install
        ("some_random_command", "other"),
        ("", "other"),
        ("python script.py", "other"),
        ("cd /some/dir", "other"),
    ],
)
def test_classify_family(command: str, expected: str) -> None:
    assert _classify_family(command) == expected


# ---------------------------------------------------------------------------
# _parse_command tests
# ---------------------------------------------------------------------------

def test_parse_command_single_exec() -> None:
    assert _parse_command('{"command": "pip install foo"}') == "pip install foo"


def test_parse_command_openclaw_nested_exec() -> None:
    assert (
        _parse_command('{"exec": {"command": "ls -la"}}')
        == "ls -la"
    )


def test_parse_command_multi_command_first() -> None:
    assert (
        _parse_command('{"commands": ["pip install foo", "python run.py"]}')
        == "pip install foo"
    )


def test_parse_command_non_exec_returns_none() -> None:
    assert _parse_command('{"path": "src/main.py"}') is None


def test_parse_command_invalid_json() -> None:
    assert _parse_command("not json") is None
    assert _parse_command("") is None


# ---------------------------------------------------------------------------
# Integration test with minimal fixture
# ---------------------------------------------------------------------------

def test_mismatch_by_command_family_reports_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the script on a minimal simulate trace and check output."""
    # Create a simulate output structure
    sim_dir = tmp_path / "simulate_run"
    sim_dir.mkdir()

    # Write a results.jsonl with a few tool_exec records
    records = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "pip install pytest"}),
                "mismatch_reason": "command_output_mismatch",
                "replay_outcome_match": False,
                "source_output_hash": "abc",
                "replay_output_hash": "def",
                "output_diff_snippet": "- foo\n+ bar",
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "apt-get update"}),
                "mismatch_reason": "timeout_mismatch",
                "replay_outcome_match": False,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "ls -la"}),
                "replay_outcome_match": True,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "git checkout main"}),
                "mismatch_reason": "command_exit_code_mismatch",
                "replay_outcome_match": False,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "curl https://example.com"}),
                "mismatch_reason": "tool_success_mismatch",
                "replay_outcome_match": False,
            },
        },
        {
            # Skipped/no-op tool: no replay comparison ran, must not be counted
            "type": "action",
            "action_type": "tool_exec",
            "agent_id": "agent-a",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "pip install numpy"}),
            },
        },
        {
            "type": "summary",
            "elapsed_s": 1.0,
            "unresolved_mismatches": 2,
        },
    ]
    result_path = sim_dir / "run_20260703.jsonl"
    result_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["mismatch_by_command_family.py", str(sim_dir), "--json", str(json_path)],
    )
    main()

    output = capsys.readouterr().out
    assert "pip_install" in output
    assert "apt" in output
    assert "readonly" in output
    assert "git_local" in output
    assert "network_fetch" in output
    assert "command_output_mismatch" in output
    assert "timeout_mismatch" in output
    assert "command_exit_code_mismatch" in output
    assert "tool_success_mismatch" in output

    stats = json.loads(json_path.read_text(encoding="utf-8"))
    # The record without replay_outcome_match is excluded from the denominator
    assert stats["pip_install"]["total"] == 1
    assert stats["pip_install"]["mismatches"] == 1
    assert stats["readonly"]["total"] == 1
    assert stats["readonly"]["mismatches"] == 0


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

def test_discover_traces_flat_jsonl_layout(tmp_path: Path) -> None:
    """Simulate writes flat <output_dir>/<run_id>.jsonl files."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "run_a.jsonl").write_text("", encoding="utf-8")
    (sim_dir / "nested").mkdir()
    (sim_dir / "nested" / "run_b.jsonl").write_text("", encoding="utf-8")
    (sim_dir / "notes.txt").write_text("", encoding="utf-8")

    assert len(_discover_traces(sim_dir)) == 2
    # A JSONL file passed directly resolves to itself
    assert _discover_traces(sim_dir / "run_a.jsonl") == [sim_dir / "run_a.jsonl"]


def test_load_tool_execs_filters_correctly(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    p.write_text(
        json.dumps({"type": "action", "action_type": "tool_exec", "data": {}}) + "\n"
        + json.dumps({"type": "action", "action_type": "llm_call", "data": {}}) + "\n"
        + json.dumps({"type": "summary", "elapsed_s": 1.0}) + "\n",
        encoding="utf-8",
    )
    tool_execs = _load_tool_execs(p)
    assert len(tool_execs) == 1
    assert tool_execs[0]["action_type"] == "tool_exec"
