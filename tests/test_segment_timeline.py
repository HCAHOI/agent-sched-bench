from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from trace_collect.openclaw_tools import _REPLAY_AGENT_SCRIPT
from trace_collect.tool_latency_dataset import (
    extract_many_segment_latency_samples,
    extract_segment_latency_samples,
)


def _script_namespace(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> dict[str, Any]:
    """Exec the in-container replay helpers (prefix before HANDLERS)."""
    if enabled:
        monkeypatch.setenv("OPENCLAW_SEGMENT_TIMELINE", "1")
    else:
        monkeypatch.delenv("OPENCLAW_SEGMENT_TIMELINE", raising=False)
    namespace: dict[str, Any] = {}
    exec(_REPLAY_AGENT_SCRIPT.split("\nHANDLERS = ", 1)[0], namespace)
    return namespace


# --- parser (harvest) ------------------------------------------------------


def test_segments_from_xtrace_keeps_top_level_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse: Callable[..., dict[str, Any]] = _script_namespace(monkeypatch, enabled=False)[
        "_segments_from_xtrace"
    ]
    text = "\n".join(
        [
            "+1000.000000 cd /tmp",
            "+1000.100000 make",
            "++1000.150000 echo nested-subshell",  # depth 2 -> skipped
            "+1000.500000 pytest",
        ]
    )

    timeline = parse(text, 1000.0, 1001.0)

    assert timeline["version"] == 2
    assert timeline["source"] == "bash_xtrace_epochrealtime"
    assert timeline["segment_count"] == 3
    assert timeline["raw_total_ms"] == pytest.approx(1000.0)
    assert [s["command_text"] for s in timeline["segments"]] == [
        "cd /tmp",
        "make",
        "pytest",
    ]
    seg0, seg1, seg2 = timeline["segments"]
    assert seg0["t_start_ms"] == pytest.approx(0.0)
    assert seg0["t_end_ms"] == pytest.approx(100.0)
    # Nested line is skipped, so make's boundary is the next top-level segment.
    assert seg1["t_end_ms"] == pytest.approx(500.0)
    # Last segment ends at the measured exec wall time (end_wall).
    assert seg2["t_end_ms"] == pytest.approx(1000.0)


def test_segments_from_xtrace_accepts_comma_radix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse = _script_namespace(monkeypatch, enabled=False)["_segments_from_xtrace"]

    timeline = parse("+1000,250000 ls", 1000.0, 1000.5)

    assert timeline["segments"][0]["t_start_ms"] == pytest.approx(250.0)


def test_segments_from_xtrace_absent_when_no_top_level_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse = _script_namespace(monkeypatch, enabled=False)["_segments_from_xtrace"]

    timeline = parse("++1000.0 only-nested\nnot a trace line", 1000.0, 1001.0)

    assert timeline == {
        "version": 2,
        "telemetry_absent": True,
        "reason": "no_segments_traced",
    }


def test_shell_launch_is_faithful_when_tracing_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _script_namespace(monkeypatch, enabled=False)["_shell_launch"]

    args, kwargs = launch("cd /x && make", {"A": "1"}, None)

    assert args == "cd /x && make"
    assert kwargs == {"shell": True, "env": {"A": "1"}}


def test_shell_launch_sets_ps4_in_script_body_not_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: PS4 set via subprocess env (not a bash-internal assignment)
    was proven live, in a real task container (bash 5.2.15), to be captured
    once at shell startup -- before EPOCHREALTIME is live -- and never
    re-expanded per xtrace line, silently freezing every segment timestamp
    empty (telemetry_absent: no_segments_traced for every exec). This bug
    passed on host bash (which apparently does re-expand env-inherited PS4),
    so a host-only functional test cannot reliably catch a regression to the
    env-based pattern -- this test pins the fix structurally instead: PS4
    must live in the script body (subject to bash's normal per-line
    re-expansion), and must never be injected via the subprocess environment.
    """
    launch = _script_namespace(monkeypatch, enabled=True)["_shell_launch"]

    args, kwargs = launch("cd /x && make", {"A": "1"}, (99, "/tmp/whatever"))

    assert "PS4" not in kwargs["env"], "PS4 must not be set via subprocess env"
    assert kwargs["env"]["BASH_XTRACEFD"] == "99"
    assert args[0].endswith("bash")
    assert args[1] == "-c"
    script_body = args[2]
    assert script_body.startswith("PS4='+$EPOCHREALTIME '\n")
    assert "\nset -x\n" in script_body
    # PS4 assignment must precede set -x, and the user's command must follow
    # both untouched (verbatim, no wrapping/escaping).
    ps4_idx = script_body.index("PS4=")
    set_x_idx = script_body.index("set -x")
    cmd_idx = script_body.index("cd /x && make")
    assert ps4_idx < set_x_idx < cmd_idx
    assert script_body.endswith("cd /x && make")


# --- real exec smoke (host bash; docker-independent) -----------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_handle_exec_records_real_segment_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # WORKDIR is read at script-exec time; point it at a real host dir since
    # the default /testbed only exists inside task containers.
    monkeypatch.setenv("OPENCLAW_CONTAINER_WORKDIR", "/tmp")
    namespace = _script_namespace(monkeypatch, enabled=True)
    handle_exec = namespace["handle_exec"]

    resp = handle_exec({"command": "cd /tmp && sleep 0.2 && echo ok"})

    assert resp["ok"] is True
    assert resp["returncode"] == 0
    assert "ok" in resp["result"]
    assert resp["stdout"] == "ok\n"
    assert resp["stderr"] == ""
    # Telemetry must never leak into the command's own stdout/stderr.
    assert "EPOCHREALTIME" not in resp["result"]
    assert "+ " not in resp["result"]
    timeline = resp["segment_timeline"]
    assert timeline["version"] == 2
    commands = [s["command_text"] for s in timeline["segments"]]
    assert any(c.startswith("sleep") for c in commands)
    sleep_seg = next(s for s in timeline["segments"] if s["command_text"].startswith("sleep"))
    # The sleep segment really took ~200ms.
    assert 150.0 <= (sleep_seg["t_end_ms"] - sleep_seg["t_start_ms"]) <= 900.0
    # Reconciliation: segments fit within the measured exec wall time.
    assert timeline["segments"][-1]["t_end_ms"] <= timeline["raw_total_ms"] + 1.0


# --- extractor -------------------------------------------------------------


def _write_trace(path: Path, timeline: Any, *, tool_args: str | None = "{}") -> Path:
    data: dict[str, Any] = {
        "tool_name": "exec",
        "success": True,
        "duration_ms": 512.0,
        "segment_timeline": timeline,
    }
    if tool_args is not None:
        data["tool_args"] = tool_args
    records = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "openclaw",
            "instance_id": "task-seg",
            "model": "fixture-model",
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool-1",
            "agent_id": "agent-a",
            "iteration": 1,
            "ts_start": 10.0,
            "ts_end": 10.512,
            "data": data,
        },
    ]
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


def _timeline(segments: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    payload = {
        "version": 2,
        "source": "bash_xtrace_epochrealtime",
        "segments": segments,
        "segment_count": len(segments),
        "raw_total_ms": 512.0,
    }
    payload.update(extra)
    return payload


def test_extract_segment_samples_yields_per_atom_rows(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path / "trace.jsonl",
        _timeline(
            [
                {"segment_index": 0, "command_text": "cd /tmp", "t_start_ms": 0.0, "t_end_ms": 2.0},
                {"segment_index": 1, "command_text": "make", "t_start_ms": 2.0, "t_end_ms": 500.0},
            ]
        ),
        tool_args=json.dumps({"exec": {"command": "cd /tmp && make"}}),
    )

    samples = extract_segment_latency_samples(trace)

    assert [s.segment_index for s in samples] == [0, 1]
    assert samples[0].task_id == "task-seg"
    assert samples[0].tool_name == "exec"
    assert samples[0].segment_command == "cd /tmp"
    assert samples[1].segment_ms == pytest.approx(498.0)
    assert samples[1].parent_chain_command == "cd /tmp && make"
    assert samples[1].parent_total_ms == pytest.approx(512.0)
    assert samples[1].parent_raw_total_ms == pytest.approx(512.0)
    assert samples[0].sample_id == f"{trace}:agent-a:tool-1:0"


def test_extract_segment_samples_skips_absent_telemetry(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path / "absent.jsonl",
        {"version": 2, "telemetry_absent": True, "reason": "bash_unavailable"},
    )

    assert extract_segment_latency_samples(trace) == []


def test_extract_segment_samples_rejects_bad_version(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path / "badver.jsonl",
        _timeline(
            [{"segment_index": 0, "command_text": "ls", "t_start_ms": 0.0, "t_end_ms": 1.0}],
            version=3,
        ),
    )

    with pytest.raises(ValueError, match="unsupported segment_timeline version"):
        extract_segment_latency_samples(trace)


def test_extract_segment_samples_rejects_reversed_bounds(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path / "reversed.jsonl",
        _timeline(
            [{"segment_index": 0, "command_text": "ls", "t_start_ms": 5.0, "t_end_ms": 1.0}]
        ),
    )

    with pytest.raises(ValueError, match="t_end_ms < t_start_ms"):
        extract_segment_latency_samples(trace)


def test_extract_many_segment_samples_requires_rows(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path / "absent.jsonl",
        {"version": 2, "telemetry_absent": True, "reason": "bash_unavailable"},
    )

    with pytest.raises(ValueError, match="no segment latency samples"):
        extract_many_segment_latency_samples([trace])
