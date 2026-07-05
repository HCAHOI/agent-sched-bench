from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.validate_predictive_policy import build_report, main
from trace_collect.predictive_policy import (
    CheckpointSchedule,
    CommandFamily,
    _has_tar_extract_option,
    classify_tool_result,
    get_checkpoint_schedule,
    needs_checkpoint,
)


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "expected"),
    [
        # --- tool_name-based classification ---
        ("read_file", {"path": "src/main.py"}, CommandFamily.READ_ONLY),
        ("web_search", {"query": "docs"}, CommandFamily.FLAKY_READ),
        ("web_fetch", {"url": "https://example.test"}, CommandFamily.FLAKY_READ),
        ("write_file", {"path": "x", "content": "y"}, CommandFamily.MUTATING),
        ("edit_file", {"path": "x"}, CommandFamily.MUTATING),
        ("spawn", {"cmd": "worker"}, CommandFamily.MUTATING),
        # --- READ_ONLY exec commands ---
        ("exec", {"command": "ls -la"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "cat /etc/hosts"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "grep -r pattern ."}, CommandFamily.READ_ONLY),
        ("exec", {"command": "find . -name '*.py'"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "head -n 10 file.txt"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "tail -f log.txt"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "wc -l file.txt"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "sort data.txt"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "uniq -c"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "echo hello"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "which python"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "file /bin/ls"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "stat /etc/passwd"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "du -sh ."}, CommandFamily.READ_ONLY),
        ("exec", {"command": "df -h"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "date"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "pwd"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "env"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "printenv PATH"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "type python3"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "dirname /a/b/c"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "basename /a/b/c"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "realpath ./file"}, CommandFamily.READ_ONLY),
        ("exec", {"exec": {"command": "git status --short"}}, CommandFamily.READ_ONLY),
        ("exec", {"command": "git diff HEAD"}, CommandFamily.READ_ONLY),
        ("exec", {"command": "git log --oneline"}, CommandFamily.READ_ONLY),
        # --- MUTATING exec commands ---
        ("exec", {"command": "pip install pytest"}, CommandFamily.MUTATING),
        ("exec", {"command": "pip3 install -r requirements.txt"}, CommandFamily.MUTATING),
        ("exec", {"command": "npm install"}, CommandFamily.MUTATING),
        ("exec", {"command": "npm ci"}, CommandFamily.MUTATING),
        ("exec", {"command": "cargo build"}, CommandFamily.MUTATING),
        ("exec", {"command": "apt-get update"}, CommandFamily.MUTATING),
        ("exec", {"command": "make test"}, CommandFamily.MUTATING),
        ("exec", {"command": "cmake .."}, CommandFamily.MUTATING),
        ("exec", {"command": "mv old new"}, CommandFamily.MUTATING),
        ("exec", {"command": "mkdir -p a/b/c"}, CommandFamily.MUTATING),
        ("exec", {"command": "rmdir empty_dir"}, CommandFamily.MUTATING),
        ("exec", {"command": "touch newfile.txt"}, CommandFamily.MUTATING),
        ("exec", {"command": "chmod +x script.sh"}, CommandFamily.MUTATING),
        ("exec", {"command": "chown user:group file"}, CommandFamily.MUTATING),
        ("exec", {"command": "git clone https://example.test/repo.git"}, CommandFamily.MUTATING),
        ("exec", {"command": "git apply patch.diff"}, CommandFamily.MUTATING),
        ("exec", {"command": "git commit -m 'msg'"}, CommandFamily.MUTATING),
        ("exec", {"command": "git push origin main"}, CommandFamily.MUTATING),
        ("exec", {"command": "git add file.py"}, CommandFamily.MUTATING),
        ("exec", {"command": "git merge feature"}, CommandFamily.MUTATING),
        ("exec", {"command": "rm -rf build"}, CommandFamily.MUTATING),
        ("exec", {"command": "rm -r /tmp/old"}, CommandFamily.MUTATING),
        ("exec", {"command": "cp -r src dst"}, CommandFamily.MUTATING),
        ("exec", {"command": "tar -xf archive.tar"}, CommandFamily.MUTATING),
        ("exec", {"command": "tar -xzf archive.tar.gz"}, CommandFamily.MUTATING),
        ("exec", {"command": "tar --extract -f archive.tar"}, CommandFamily.MUTATING),
        ("exec", {"command": "tar xf archive.tar"}, CommandFamily.MUTATING),
        ("exec", {"command": "ln -s target link"}, CommandFamily.MUTATING),
        # --- UNKNOWN exec commands ---
        ("exec", {"command": "python -m pytest tests"}, CommandFamily.UNKNOWN),
        ("exec", {"command": "gh pr list"}, CommandFamily.UNKNOWN),
        ("exec", {"command": "docker ps"}, CommandFamily.UNKNOWN),
    ],
)
def test_classify_tool_result(
    tool_name: str,
    tool_args: dict[str, object],
    expected: CommandFamily,
) -> None:
    assert classify_tool_result(tool_name, json.dumps(tool_args)) == expected


# ── classify_tool_result edge cases ──────────────────────────────────


def test_classify_tool_result_invalid_command_args_are_unknown() -> None:
    assert classify_tool_result("exec", "not json") == CommandFamily.UNKNOWN
    assert classify_tool_result("exec", json.dumps({"commands": ["ls", "touch x"]})) == (
        CommandFamily.READ_ONLY  # first subcommand "ls" determines classification
    )
    assert classify_tool_result(None, None) == CommandFamily.UNKNOWN


def test_classify_tool_result_none_tool_name_falls_back_to_command() -> None:
    """None tool_name should fall through to command-based classification."""
    assert classify_tool_result(None, json.dumps({"command": "ls"})) == CommandFamily.READ_ONLY
    assert classify_tool_result(None, json.dumps({"command": "pip install x"})) == (
        CommandFamily.MUTATING
    )
    assert classify_tool_result(None, json.dumps({"command": "python script.py"})) == (
        CommandFamily.UNKNOWN
    )


def test_classify_tool_result_empty_tool_name_falls_back_to_command() -> None:
    assert classify_tool_result("", json.dumps({"command": "ls"})) == CommandFamily.READ_ONLY
    assert classify_tool_result("  ", json.dumps({"command": "cat file"})) == CommandFamily.READ_ONLY


def test_classify_tool_result_empty_tool_args_is_unknown() -> None:
    assert classify_tool_result("exec", None) == CommandFamily.UNKNOWN
    assert classify_tool_result("exec", "") == CommandFamily.UNKNOWN
    assert classify_tool_result("exec", "{}") == CommandFamily.UNKNOWN


def test_classify_tool_result_non_dict_tool_args_is_unknown() -> None:
    assert classify_tool_result("exec", json.dumps(["not", "a", "dict"])) == CommandFamily.UNKNOWN
    assert classify_tool_result("exec", json.dumps(42)) == CommandFamily.UNKNOWN


def test_classify_tool_result_tar_without_extract_option_is_unknown() -> None:
    """tar -czf creates, not extracts — should be UNKNOWN."""
    assert classify_tool_result("exec", json.dumps({"command": "tar -czf out.tar.gz src/"})) == (
        CommandFamily.UNKNOWN
    )
    assert classify_tool_result("exec", json.dumps({"command": "tar -cf out.tar src/"})) == (
        CommandFamily.UNKNOWN
    )
    assert classify_tool_result("exec", json.dumps({"command": "tar --create -f out.tar src/"})) == (
        CommandFamily.UNKNOWN
    )


def test_classify_tool_result_git_without_subcommand_is_unknown() -> None:
    assert classify_tool_result("exec", json.dumps({"command": "git"})) == CommandFamily.UNKNOWN


def test_classify_tool_result_rm_without_r_is_unknown() -> None:
    """rm without -r flag should be UNKNOWN."""
    assert classify_tool_result("exec", json.dumps({"command": "rm single_file.txt"})) == (
        CommandFamily.UNKNOWN
    )


# ── needs_checkpoint ─────────────────────────────────────────────────


def test_needs_checkpoint_policy() -> None:
    assert not needs_checkpoint(CommandFamily.READ_ONLY)
    assert not needs_checkpoint(CommandFamily.FLAKY_READ)
    assert needs_checkpoint(CommandFamily.MUTATING)
    assert needs_checkpoint(CommandFamily.UNKNOWN)


# ── get_checkpoint_schedule ──────────────────────────────────────────


def test_get_checkpoint_schedule_combines_probe_and_prediction() -> None:
    assert (
        get_checkpoint_schedule("exec", {"command": "ls"}, "unchanged", "walk")
        == "deferred_skip"
    )
    assert (
        get_checkpoint_schedule(
            "exec",
            json.dumps({"command": "python -m pytest"}),
            "unchanged",
            "overlay",
        )
        == "await_prediction"
    )
    assert (
        get_checkpoint_schedule("write_file", {"path": "x"}, "changed", "walk")
        == "immediate_capture"
    )


@pytest.mark.parametrize(
    "probe_status",
    ["changed", "CHANGED", "Changed", " changed ", "maybe_changed", "initial"],
)
def test_get_checkpoint_schedule_non_unchanged_probe_is_immediate_capture(
    probe_status: str,
) -> None:
    """Any probe status that is not 'unchanged' triggers immediate capture."""
    assert (
        get_checkpoint_schedule("exec", {"command": "ls"}, probe_status, "overlay")
        == CheckpointSchedule.IMMEDIATE_CAPTURE.value
    )


def test_get_checkpoint_schedule_unchanged_read_only_is_deferred_skip() -> None:
    assert (
        get_checkpoint_schedule("exec", {"command": "ls"}, "unchanged", "overlay")
        == CheckpointSchedule.DEFERRED_SKIP.value
    )
    assert (
        get_checkpoint_schedule("read_file", {"path": "x"}, "unchanged", "overlay")
        == CheckpointSchedule.DEFERRED_SKIP.value
    )


def test_get_checkpoint_schedule_unchanged_flaky_read_is_deferred_skip() -> None:
    assert (
        get_checkpoint_schedule("web_search", {"query": "test"}, "unchanged", "overlay")
        == CheckpointSchedule.DEFERRED_SKIP.value
    )


def test_get_checkpoint_schedule_unchanged_mutating_is_await_prediction() -> None:
    assert (
        get_checkpoint_schedule("write_file", {"path": "x"}, "unchanged", "overlay")
        == CheckpointSchedule.AWAIT_PREDICTION.value
    )
    assert (
        get_checkpoint_schedule("exec", {"command": "pip install x"}, "unchanged", "overlay")
        == CheckpointSchedule.AWAIT_PREDICTION.value
    )


def test_get_checkpoint_schedule_unchanged_unknown_is_await_prediction() -> None:
    assert (
        get_checkpoint_schedule(
            "exec", {"command": "python script.py"}, "unchanged", "overlay"
        )
        == CheckpointSchedule.AWAIT_PREDICTION.value
    )


def test_get_checkpoint_schedule_invalid_probe_status_raises() -> None:
    with pytest.raises(ValueError, match="unsupported probe_status"):
        get_checkpoint_schedule("exec", {"command": "ls"}, "bogus", "overlay")


def test_get_checkpoint_schedule_empty_backend_type_raises() -> None:
    with pytest.raises(ValueError, match="backend_type must be non-empty"):
        get_checkpoint_schedule("exec", {"command": "ls"}, "unchanged", "")


# ── _has_tar_extract_option ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["-x", "archive.tar"], True),
        (["-xf", "archive.tar"], True),
        (["-xzf", "archive.tar.gz"], True),
        (["-xvzf", "archive.tar.gz"], True),
        (["--extract", "-f", "archive.tar"], True),
        (["--extract", "--file", "archive.tar"], True),
        # Old-style tar without dash prefix
        (["xf", "archive.tar"], True),
        (["xzf", "archive.tar.gz"], True),
        # Non-extract options
        (["-c", "archive.tar"], False),
        (["-czf", "archive.tar.gz"], False),
        (["--create", "-f", "archive.tar"], False),
        (["-t", "archive.tar"], False),
        # False positives: these must NOT match
        (["matrix.tar"], False),
        (["pxz"], False),
        (["fix.txt"], False),
        (["extract.sh"], False),
        (["context.py"], False),
        (["archive.tar"], False),
        # Empty / edge
        ([], False),
    ],
)
def test_has_tar_extract_option(tokens: list[str], expected: bool) -> None:
    assert _has_tar_extract_option(tokens) == expected


# ── classify_tool_result: tar false-positive guard ───────────────────


def test_classify_tool_result_tar_no_extract_false_positives() -> None:
    """Commands containing 'x' in a non-extract context should not be MUTATING."""
    assert classify_tool_result(
        "exec", json.dumps({"command": "tar -czf matrix.tar src/"})
    ) == CommandFamily.UNKNOWN
    assert classify_tool_result(
        "exec", json.dumps({"command": "tar -czf pxz.tar.gz src/"})
    ) == CommandFamily.UNKNOWN


# ── build_report tests ───────────────────────────────────────────────


def test_build_report_counts_confusion_by_family_and_benchmark(tmp_path: Path) -> None:
    trace_dir = tmp_path / "simulate"
    trace_path = trace_dir / "task-a" / "attempt_1" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True)
    _write_jsonl(
        trace_path,
        [
            {"type": "trace_metadata", "benchmark": "bench-a"},
            _tool("tn-read", "exec", {"command": "ls"}, {"skipped": "no changes"}),
            _tool("tp-write", "write_file", {"path": "x"}, {"path": "cp.json"}),
            _tool("fn-read", "read_file", {"path": "x"}, {"path": "cp2.json"}),
            _tool("fp-unknown", "exec", {"command": "python script.py"}, {"skipped": "no changes"}),
            _tool_error("fn-error", "exec", {"command": "git status"}),
            _tool_without_ground_truth("no-gt", "exec", {"command": "mkdir build"}),
        ],
    )

    report = build_report(trace_dir)

    assert report["trace_count"] == 1
    assert report["total_actions"] == 6
    assert report["predictions_made"] == 6
    assert report["ground_truth_available"] == 5
    assert report["confusion"] == {
        "true_negative": 1,
        "true_positive": 1,
        "false_negative": 2,
        "false_positive": 1,
    }
    assert report["misprediction_rate"] == pytest.approx(3 / 5)
    assert report["fn_rate"] == pytest.approx(2 / 3)
    assert report["families"]["read_only"]["false_negative"] == 2
    assert report["families"]["unknown"]["false_positive"] == 1
    assert report["families"]["mutating"]["true_positive"] == 1
    assert report["benchmarks"]["bench-a"]["ground_truth_available"] == 5


def test_build_report_follows_simulate_source_traces_without_ground_truth(
    tmp_path: Path,
) -> None:
    source_trace = tmp_path / "source" / "task-a" / "attempt_1" / "trace.jsonl"
    source_trace.parent.mkdir(parents=True)
    _write_jsonl(
        source_trace,
        [
            {"type": "trace_metadata", "benchmark": "bench-source"},
            _tool("tn-read", "exec", {"command": "pwd"}, {"skipped": "no changes"}),
        ],
    )

    replay_trace = tmp_path / "simulate" / "task-a" / "attempt_1" / "trace.jsonl"
    replay_trace.parent.mkdir(parents=True)
    _write_jsonl(
        replay_trace,
        [
            {
                "type": "trace_metadata",
                "source_trace_entries": [{"source_trace": str(source_trace)}],
            },
            _tool_without_ground_truth("replay-tool", "exec", {"command": "mkdir x"}),
        ],
    )

    report = build_report(tmp_path / "simulate")

    assert report["input_trace_count"] == 1
    assert report["trace_count"] == 1
    assert report["total_actions"] == 1
    assert report["ground_truth_available"] == 1
    assert report["confusion"]["true_negative"] == 1
    assert "bench-source" in report["benchmarks"]


def test_main_prints_table_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_dir = tmp_path / "simulate"
    trace_path = trace_dir / "task-a" / "attempt_1" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True)
    _write_jsonl(
        trace_path,
        [
            {"type": "trace_metadata", "benchmark": "bench-a"},
            _tool("tn-read", "exec", {"command": "pwd"}, {"skipped": "no changes"}),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_predictive_policy.py", "--trace-dir", str(trace_dir)],
    )

    main()

    output = capsys.readouterr().out
    assert "PREDICTIVE CHECKPOINT POLICY VALIDATION" in output
    assert "By command family" in output
    assert "By benchmark" in output
    assert '"misprediction_rate": 0.0' in output


# ── helpers ──────────────────────────────────────────────────────────


def _tool(
    action_id: str,
    tool_name: str,
    tool_args: dict[str, object],
    checkpoint_after: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "data": {
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args),
            "checkpoint_after": checkpoint_after,
        },
    }


def _tool_error(
    action_id: str,
    tool_name: str,
    tool_args: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "data": {
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args),
            "checkpoint_after_error": {"error": "checkpoint failed"},
        },
    }


def _tool_without_ground_truth(
    action_id: str,
    tool_name: str,
    tool_args: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "data": {
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args),
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
