from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.container import ContainerExecTool
from agents.openclaw.tools.registry import ToolRegistry
from llm_call.provider_base import LLMProvider, LLMResponse, ToolCallRequest
from trace_collect.clause_telemetry import (
    ClauseTelemetryCollector,
    ClauseTelemetryIntegrityError,
    ToolCallToken,
    shell_command_lookup_failure_evidence,
    validate_clause_telemetry_runtime,
)
from trace_collect.cli import parse_simulate_args
from trace_collect.openclaw_host_runtime import _attach_clause_telemetry


def _event(
    event_type: str,
    ts_ns: int,
    pid: int,
    *,
    seq: int = 2**64 - 1,
    tid: int | None = None,
    parent: int = 0,
    child: int = 0,
    child_tid: int = 0,
    arg_index: int = 0,
    arg: str = "",
    cpu_ns: int = 0,
    rss_pages: int = 0,
    mm_ptr: int = 0,
    exit_code: int = 0,
    io_read: int = 0,
    io_write: int = 0,
    io_cancelled: int = 0,
) -> dict[str, Any]:
    return {
        "type": event_type,
        "ts_ns": ts_ns,
        "cgroup_id": 7,
        "exec_seq": seq,
        "cpu_ns": cpu_ns,
        "rss_pages": rss_pages,
        "mm_ptr": mm_ptr,
        "hiwater_pages": 0,
        "io_read_bytes": io_read,
        "io_write_bytes": io_write,
        "io_cancelled_write_bytes": io_cancelled,
        "host_pid": pid,
        "host_tid": pid if tid is None else tid,
        "parent_host_pid": parent,
        "child_host_pid": child,
        "child_host_tid": child_tid or child,
        "arg_index": arg_index,
        "arg": arg,
        "exit_code": exit_code,
        "errno": exit_code if event_type == "failed_exec_attempt" else 0,
    }


def _collector_without_bpf() -> ClauseTelemetryCollector:
    collector = object.__new__(ClauseTelemetryCollector)
    collector.cgroup_id = 7
    collector.quota_cores = 4.0
    collector.repo = "repo"
    collector._epoch_offset_s = 1_000.0
    return collector


def _clean_events() -> list[dict[str, Any]]:
    return [
        _event("fork", 110, 50, child=100),
        _event("exec_arg", 120, 100, seq=0, arg_index=0, arg="/bin/echo"),
        _event("exec_arg", 121, 100, seq=0, arg_index=1, arg="hi"),
        _event(
            "exec_boundary",
            130,
            100,
            seq=0,
            parent=50,
            cpu_ns=1,
            rss_pages=1,
            mm_ptr=10,
        ),
        # The persistent container agent is in the cgroup but outside the
        # command subtree, so its sample is an explicit structural gap.
        _event("perf", 150, 50, cpu_ns=1, rss_pages=1, mm_ptr=5),
        _event(
            "perf",
            180,
            100,
            seq=0,
            cpu_ns=2,
            rss_pages=1,
            mm_ptr=10,
        ),
        _event(
            "exit_boundary",
            220,
            100,
            seq=0,
            parent=50,
            cpu_ns=3,
            rss_pages=1,
            mm_ptr=10,
        ),
    ]


def _failed_exec_events() -> list[dict[str, Any]]:
    return [
        _event("fork", 110, 50, child=100),
        _event("exec_arg", 120, 100, seq=0, arg_index=0, arg="/bin/sh"),
        _event(
            "exec_boundary",
            130,
            100,
            seq=0,
            parent=50,
            cpu_ns=1,
            rss_pages=1,
            mm_ptr=10,
        ),
        _event("exec_arg", 140, 100, seq=1, arg_index=0, arg="python"),
        _event("exec_arg", 141, 100, seq=1, arg_index=1, arg="-m"),
        _event("exec_arg", 142, 100, seq=1, arg_index=2, arg="pytest"),
        _event(
            "failed_exec_attempt",
            150,
            100,
            seq=1,
            parent=50,
            exit_code=2,
        ),
        _event(
            "exit_boundary",
            220,
            100,
            seq=0,
            parent=50,
            cpu_ns=2,
            rss_pages=1,
            mm_ptr=10,
        ),
    ]


def _control_events(exit_status: int) -> list[dict[str, Any]]:
    return [
        _event("fork", 110, 50, child=100),
        _event("exec_arg", 120, 100, seq=0, arg="/bin/left"),
        _event(
            "exec_boundary",
            130,
            100,
            seq=0,
            parent=50,
            cpu_ns=1,
            rss_pages=1,
            mm_ptr=10,
        ),
        _event(
            "exit_boundary",
            220,
            100,
            seq=0,
            parent=50,
            cpu_ns=2,
            rss_pages=1,
            mm_ptr=10,
            exit_code=exit_status << 8,
        ),
    ]


def _apt_fork_chain_events() -> list[dict[str, Any]]:
    return [
        _event("fork", 101, 10, child=20),
        _event("fork", 102, 20, child=30),
        _event("exec_arg", 110, 30, seq=0, arg="/bin/sh"),
        _event("exec_boundary", 120, 30, seq=0, parent=20),
        _event("fork", 130, 30, child=40),
        _event("fork", 140, 40, child=50),
        _event("exec_arg", 150, 50, seq=1, arg_index=0, arg="apt-get"),
        _event("exec_arg", 151, 50, seq=1, arg_index=1, arg="update"),
        _event("exec_boundary", 160, 50, seq=1, parent=40),
        _event("fork", 170, 50, child=60),
        _event("fork", 180, 60, child=70),
        _event("exec_arg", 190, 70, seq=2, arg="dpkg"),
        _event("exec_boundary", 200, 70, seq=2, parent=60),
        _event("exit_boundary", 220, 70, seq=2, parent=60),
        _event("exit_boundary", 230, 50, seq=1, parent=40),
        _event("exit_boundary", 240, 30, seq=0, parent=20),
    ]


def test_cli_defaults_to_command_and_accepts_clause() -> None:
    default = parse_simulate_args(["--manifest", "manifest.yaml"])
    clause = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--tool-resource-telemetry",
            "clause",
        ]
    )
    assert default.tool_resource_telemetry == "command"
    assert clause.tool_resource_telemetry == "clause"


def test_clause_runtime_rejects_configuration_before_bcc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("trace_collect.clause_telemetry.os.geteuid", lambda: 0)
    with pytest.raises(ValueError, match="container docker"):
        validate_clause_telemetry_runtime(
            container_executable="podman",
            concurrency=1,
            workers=1,
            pacct=False,
        )
    monkeypatch.setitem(sys.modules, "bcc", object())
    validate_clause_telemetry_runtime(
        container_executable="docker",
        concurrency=2,
        workers=1,
        pacct=False,
    )
    with pytest.raises(ValueError, match="workers 1"):
        validate_clause_telemetry_runtime(
            container_executable="docker",
            concurrency=2,
            workers=2,
            pacct=False,
        )
    with pytest.raises(ValueError, match="incompatible with --pacct"):
        validate_clause_telemetry_runtime(
            container_executable="docker",
            concurrency=1,
            workers=1,
            pacct=True,
        )


def test_summary_preserves_structural_gap_and_target_availability() -> None:
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-1", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=_clean_events(),
        loss=0,
        perf_samples=2,
    )
    assert violations == []
    assert summary["mapping"]["coverage"] == 1.0
    assert summary["coverage_gaps"]["relevant"]["count"] == 0
    assert summary["coverage_gaps"]["structural"]["count"] == 1
    assert summary["coverage_gaps"]["structural"]["events"] == [
        {
            "type": "perf",
            "ts_ns": 150,
            "host_pid": 50,
            "host_tid": 50,
            "exec_seq": 2**64 - 1,
            "entry_pid": 50,
            "entry_parent_relation": "entry_parent",
            "fork_parent_pid": None,
            "reason": (
                "sentinel_exec_seq_without_active_exec_image_or_owned_ancestor"
            ),
        }
    ]
    assert summary["target_availability"]["latency"]["available"] == 1
    assert summary["target_availability"]["cpu"]["reasons"] == {
        "unknown:clause_shorter_than_1s_ineligible_for_peak": 1
    }
    assert summary["ring_loss"]["reserve_failures"] == 0
    assert summary["clauses"][0]["disk_io"] == {
        "read_bytes_total": 0,
        "write_bytes_total": 0,
        "cancelled_write_bytes_total": 0,
        "read_write_bytes_total": 0,
        "availability": "ok",
    }
    assert summary["provenance"]["window_ns"] == 500_000_000
    assert summary["provenance"]["disk_io_semantics"] == (
        "linux_task_io_accounting_total_bytes"
    )
    assert summary["provenance"]["repo"] == "repo"
    assert summary["provenance"]["command_tree"] == {
        "status": "ok",
        "reason": None,
        "entry_pid": 50,
        "root_pids": [100],
        "exec_ancestry": [
            {
                "exec_pid": 100,
                "ancestor_chain": [50],
                "nearest_exec_ancestor_pid": None,
                "is_root": True,
            }
        ],
    }


def test_fork_only_processes_collapse_to_nearest_transitive_exec_ancestor() -> None:
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-apt", "apt-get update", 100, 0, 0),
        ended_ns=250,
        events=_apt_fork_chain_events(),
        loss=0,
        perf_samples=0,
    )

    assert violations == []
    assert summary["provenance"]["command_tree"] == {
        "status": "ok",
        "reason": None,
        "entry_pid": 10,
        "root_pids": [30],
        "exec_ancestry": [
            {
                "exec_pid": 30,
                "ancestor_chain": [20, 10],
                "nearest_exec_ancestor_pid": None,
                "is_root": True,
            },
            {
                "exec_pid": 50,
                "ancestor_chain": [40, 30, 20, 10],
                "nearest_exec_ancestor_pid": 30,
                "is_root": False,
            },
            {
                "exec_pid": 70,
                "ancestor_chain": [60, 50, 40, 30, 20, 10],
                "nearest_exec_ancestor_pid": 50,
                "is_root": False,
            },
        ],
    }
    assert summary["clauses"][0]["owned_exec_image_count"] == 2


def test_disconnected_command_trees_persist_provenance_in_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        _event("fork", 101, 10, child=20),
        _event("exec_arg", 110, 20, seq=0, arg="/bin/sh"),
        _event("exec_boundary", 120, 20, seq=0, parent=10),
        _event("exit_boundary", 150, 20, seq=0, parent=10),
        _event("fork", 102, 11, child=21),
        _event("exec_arg", 111, 21, seq=1, arg="/bin/sh"),
        _event("exec_boundary", 121, 21, seq=1, parent=11),
        _event("exit_boundary", 151, 21, seq=1, parent=11),
    ]
    collector = _collector_without_bpf()
    token = ToolCallToken("call-disconnected", "true", 100, 0, 0)
    collector._active = token
    collector._bpf = object()
    collector._events_lock = Lock()
    collector._events = events
    collector.calls = []
    collector._integrity_errors = []
    monkeypatch.setattr("trace_collect.clause_telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("trace_collect.clause_telemetry.time.sleep", lambda *_: None)

    with pytest.raises(ClauseTelemetryIntegrityError, match="disconnected"):
        collector.finish_tool_call(token, replay_response={"returncode": 0})

    tree = collector.calls[0]["provenance"]["command_tree"]
    assert tree["status"] == "failed"
    assert tree["reason"] == "disconnected_command_trees"
    assert tree["root_pids"] == [20, 21]
    assert tree["exec_ancestry"] == [
        {
            "exec_pid": 20,
            "ancestor_chain": [10],
            "nearest_exec_ancestor_pid": None,
            "is_root": True,
        },
        {
            "exec_pid": 21,
            "ancestor_chain": [11],
            "nearest_exec_ancestor_pid": None,
            "is_root": True,
        },
    ]


def test_command_tree_rejects_ambiguous_ancestry_above_active_exec() -> None:
    events = [
        _event("fork", 101, 10, child=20),
        _event("fork", 102, 11, child=20),
        _event("exec_arg", 110, 20, seq=0, arg="/bin/sh"),
        _event("exec_boundary", 120, 20, seq=0, parent=10),
        _event("exit_boundary", 150, 20, seq=0, parent=10),
    ]
    collector = _collector_without_bpf()

    with pytest.raises(
        ClauseTelemetryIntegrityError,
        match="ambiguous_fork_ancestry",
    ) as raised:
        collector._summarize_call(
            token=ToolCallToken("call-ambiguous", "true", 100, 0, 0),
            ended_ns=200,
            events=events,
            loss=0,
            perf_samples=0,
        )

    assert raised.value.artifact_payload["provenance"]["command_tree"] == {
        "status": "failed",
        "reason": "ambiguous_fork_ancestry",
        "entry_pid": None,
        "root_pids": [20],
        "exec_ancestry": [
            {
                "exec_pid": 20,
                "ancestor_chain": [],
                "nearest_exec_ancestor_pid": None,
                "is_root": True,
            }
        ],
        "fork_ambiguities": [
            {
                "child_pid": 20,
                "parent_candidates": [10, 11],
                "candidate_records": [
                    {"parent_pid": 10, "ts_ns": 101},
                    {"parent_pid": 11, "ts_ns": 102},
                ],
            }
        ],
    }


def test_short_circuit_source_replay_disagreement_stays_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _collector_without_bpf()
    token = ToolCallToken(
        "call-control-mismatch",
        "left && right",
        100,
        0,
        0,
        source_tool_call_id="source-control",
        source_command="left && right",
        source_tool_result="source success\n\nExit code: 0",
    )
    collector._active = token
    collector._bpf = object()
    collector._events_lock = Lock()
    collector._events = _control_events(1)
    collector.calls = []
    collector._integrity_errors = []
    monkeypatch.setattr("trace_collect.clause_telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("trace_collect.clause_telemetry.time.sleep", lambda *_: None)

    with pytest.raises(ClauseTelemetryIntegrityError, match="mapping gaps"):
        collector.finish_tool_call(
            token,
            replay_response={
                "ok": True,
                "result": "replay failure",
                "returncode": 1,
            },
        )

    call = collector.calls[0]
    assert call["no_runtime_exec"] == []
    assert call["provenance"]["source_replay_control_flow_fidelity"] == {
        "source_action_available": True,
        "source_command_matches": True,
        "source_exit_code": 0,
        "replay_exit_code": 1,
        "exit_code_matches": False,
        "tool_result_exact": False,
        "short_circuit_eligible": False,
    }
    assert call["mapping"]["gaps"][0]["kind"] == "unmatched_static_clause"


def test_relevant_gap_fails_integrity() -> None:
    collector = _collector_without_bpf()
    events = _clean_events()
    events.append(_event("perf", 220, 100, cpu_ns=4, rss_pages=1, mm_ptr=10))
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-2", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=3,
    )
    assert summary["coverage_gaps"]["relevant"]["count"] == 1
    assert summary["coverage_gaps"]["relevant"]["events"] == [
        {
            "type": "perf",
            "ts_ns": 220,
            "host_pid": 100,
            "host_tid": 100,
            "exec_seq": 2**64 - 1,
            "entry_pid": 50,
            "entry_parent_relation": "command_descendant",
            "fork_parent_pid": 50,
            "reason": "sentinel_after_successful_exec",
        }
    ]
    assert violations == ["call-2: relevant coverage gaps=1"]
    assert summary["integrity"]["status"] == "failed"


def test_entry_parent_thread_gap_is_structural() -> None:
    events = _clean_events()
    events.append(
        _event(
            "perf",
            160,
            50,
            tid=51,
            cpu_ns=1,
            rss_pages=1,
            mm_ptr=5,
        )
    )
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-parent-thread", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=3,
    )
    assert violations == []
    assert summary["coverage_gaps"]["relevant"]["count"] == 0
    assert [
        event["entry_parent_relation"]
        for event in summary["coverage_gaps"]["structural"]["events"]
    ] == ["entry_parent", "entry_parent_thread"]


def test_entry_fork_pre_exec_gap_is_structural_with_payload() -> None:
    events = _clean_events()
    events.append(
        _event(
            "perf",
            115,
            100,
            cpu_ns=1,
            rss_pages=1,
            mm_ptr=10,
        )
    )
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-entry-fork", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=3,
    )

    assert violations == []
    setup = [
        event
        for event in summary["coverage_gaps"]["structural"]["events"]
        if event["entry_parent_relation"]
        == "entry_fork_pre_exec_structural_setup"
    ]
    assert setup == [
        {
            "type": "perf",
            "ts_ns": 115,
            "host_pid": 100,
            "host_tid": 100,
            "exec_seq": 2**64 - 1,
            "entry_pid": 50,
            "entry_parent_relation": (
                "entry_fork_pre_exec_structural_setup"
            ),
            "fork_parent_pid": 50,
            "reason": "entry_fork_pre_exec_structural_setup",
            "fork_ancestry": [100, 50],
            "fork_chain_records": [
                {"child_id": 100, "parent_pid": 50, "ts_ns": 110}
            ],
            "fork_ts_ns": 110,
        }
    ]


def test_repeated_fork_generation_gap_keeps_command_relation_evidence() -> None:
    events = _clean_events()
    events.extend(
        [
            _event("fork", 160, 100, child=200),
            _event("fork", 161, 100, child=200),
            _event("perf", 170, 200, cpu_ns=1, rss_pages=1, mm_ptr=20),
        ]
    )
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-reused-pid", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=3,
    )

    assert violations == ["call-reused-pid: relevant coverage gaps=1"]
    assert summary["coverage_gaps"]["relevant"]["events"] == [
        {
            "type": "perf",
            "ts_ns": 170,
            "host_pid": 200,
            "host_tid": 200,
            "exec_seq": 2**64 - 1,
            "entry_pid": 50,
            "entry_parent_relation": "command_descendant",
            "fork_parent_pid": 100,
            "reason": "sentinel_pre_exec_ambiguous_fork_ancestry",
            "fork_ancestry": [200],
            "fork_chain_records": [],
            "fork_resolution_failure": {
                "failure_kind": "ambiguous_generation",
                "child_id": 200,
                "timestamp_bound_ns": 170,
                "eligible_records": [
                    {"parent_pid": 100, "ts_ns": 160},
                    {"parent_pid": 100, "ts_ns": 161},
                ],
                "rejected_records": [],
            },
        }
    ]


def test_failed_exec_is_target_unavailable_without_mapping_gap() -> None:
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken(
            "call-failed",
            "cd /testbed && python -m pytest",
            100,
            0,
            0,
        ),
        ended_ns=230,
        events=_failed_exec_events(),
        loss=0,
        perf_samples=0,
    )
    assert violations == []
    assert summary["mapping"] == {
        "static_clause_count": 2,
        "mappable_clause_count": 1,
        "mapped_clause_count": 1,
        "observation_clause_count": 0,
        "no_runtime_exec_count": 1,
        "coverage": 1.0,
        "gaps": [],
        "unobserved_builtins": ["cd"],
    }
    assert summary["clauses"] == []
    assert summary["no_runtime_exec"][0]["errno"] == [2]
    assert summary["target_availability"]["latency"]["reasons"] == {
        "unknown:no_runtime_exec": 1
    }


def test_unmatched_static_without_failed_exec_evidence_remains_fatal() -> None:
    collector = _collector_without_bpf()
    events = [
        event
        for event in _failed_exec_events()
        if event["type"] not in {"exec_arg", "failed_exec_attempt"}
        or event["exec_seq"] == 0
    ]
    summary, violations = collector._summarize_call(
        token=ToolCallToken(
            "call-unmatched",
            "cd /testbed && python -m pytest",
            100,
            0,
            0,
        ),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=0,
    )
    assert summary["no_runtime_exec"] == []
    assert violations == [
        "call-unmatched: mapping gaps=unmatched_static_clause"
    ]


def test_direct_command_not_found_is_separate_target_unavailable_evidence() -> None:
    command = "cd /testbed && python -m pytest"
    diagnostic = "/bin/sh: 1: python: not found"
    evidence = shell_command_lookup_failure_evidence(
        command=command,
        source_tool_call_id="source-1",
        replay_tool_call_id="replay-1",
        source_command=command,
        source_tool_result=f"{diagnostic}\n\nExit code: 127",
        replay_result=diagnostic,
        replay_stderr=diagnostic,
        replay_exit_code=127,
    )
    assert evidence is not None
    assert evidence.replay_channel == "raw_stderr"
    events = [
        event
        for event in _failed_exec_events()
        if event["type"] not in {"exec_arg", "failed_exec_attempt"}
        or event["exec_seq"] == 0
    ]
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("replay-1", command, 100, 0, 0),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=0,
        command_lookup_failure=evidence,
    )
    assert violations == []
    assert summary["clauses"] == []
    row = summary["no_runtime_exec"][0]
    assert row["attempt_count"] == 0
    assert "errno" not in row
    assert row["provenance"] == {
        "evidence_kind": "shell_command_lookup_failure",
        "parser": "anchored_shell_command_not_found_v1",
        "command": command,
        "executable_head": "python",
        "exit_code_semantics": "direct_command_not_found_127",
        "source": {
            "tool_call_id": "source-1",
            "exit_code": 127,
            "channel": "source_tool_result",
            "diagnostic": diagnostic,
        },
        "replay": {
            "tool_call_id": "replay-1",
            "exit_code": 127,
            "channel": "raw_stderr",
            "diagnostic": diagnostic,
        },
    }


def test_pipeline_masked_command_not_found_uses_anchored_tool_result() -> None:
    command = "python -m pytest 2>&1 | tail -40"
    diagnostic = "/bin/sh: 1: python: not found"
    evidence = shell_command_lookup_failure_evidence(
        command=command,
        source_tool_call_id="source-1",
        replay_tool_call_id="replay-1",
        source_command=command,
        source_tool_result=f"{diagnostic}\n\nExit code: 0",
        replay_result=diagnostic,
        replay_stderr="",
        replay_exit_code=0,
    )
    assert evidence is not None
    assert evidence.replay_channel == "tool_result"
    assert evidence.source_exit_code == evidence.replay_exit_code == 0


def test_command_not_found_text_that_is_not_a_shell_diagnostic_is_rejected() -> None:
    command = "printf '%s\\n' 'python: not found'"
    assert (
        shell_command_lookup_failure_evidence(
            command=command,
            source_tool_call_id="source-1",
            replay_tool_call_id="replay-1",
            source_command=command,
            source_tool_result="python: not found\n\nExit code: 0",
            replay_result="python: not found",
            replay_stderr="",
            replay_exit_code=0,
        )
        is None
    )


def test_raw_stderr_is_preferred_over_diagnostic_looking_stdout() -> None:
    command = "python -m pytest"
    diagnostic = "/bin/sh: 1: python: not found"
    assert (
        shell_command_lookup_failure_evidence(
            command=command,
            source_tool_call_id="source-1",
            replay_tool_call_id="replay-1",
            source_command=command,
            source_tool_result=f"{diagnostic}\n\nExit code: 127",
            replay_result=diagnostic,
            replay_stderr="warning from shell",
            replay_exit_code=127,
        )
        is None
    )


def test_command_not_found_source_replay_disagreement_is_rejected() -> None:
    command = "python -m pytest"
    source_result = "/bin/sh: 1: python: not found\n\nExit code: 127"
    assert (
        shell_command_lookup_failure_evidence(
            command=command,
            source_tool_call_id="source-1",
            replay_tool_call_id="replay-1",
            source_command=command,
            source_tool_result=source_result,
            replay_result="/bin/sh: 1: python3: not found",
            replay_stderr="/bin/sh: 1: python3: not found",
            replay_exit_code=127,
        )
        is None
    )
    assert (
        shell_command_lookup_failure_evidence(
            command=command,
            source_tool_call_id="source-1",
            replay_tool_call_id="replay-1",
            source_command=command,
            source_tool_result=source_result,
            replay_result="/bin/sh: 1: python: not found",
            replay_stderr="/bin/sh: 1: python: not found",
            replay_exit_code=0,
        )
        is None
    )


def test_command_not_found_path_heads_must_agree_exactly() -> None:
    command = "/opt/python -m pytest"
    assert (
        shell_command_lookup_failure_evidence(
            command=command,
            source_tool_call_id="source-1",
            replay_tool_call_id="replay-1",
            source_command=command,
            source_tool_result=(
                "/bin/sh: 1: /opt/python: not found\n\nExit code: 127"
            ),
            replay_result="/bin/sh: 1: /usr/bin/python: not found",
            replay_stderr="/bin/sh: 1: /usr/bin/python: not found",
            replay_exit_code=127,
        )
        is None
    )


@pytest.mark.parametrize(
    "command",
    ["python || true", "python; true", "python; echo ok"],
)
def test_exit_zero_command_not_found_requires_pipeline_masking(command: str) -> None:
    diagnostic = "/bin/sh: 1: python: not found"
    assert (
        shell_command_lookup_failure_evidence(
            command=command,
            source_tool_call_id="source-1",
            replay_tool_call_id="replay-1",
            source_command=command,
            source_tool_result=f"{diagnostic}\n\nExit code: 0",
            replay_result=diagnostic,
            replay_stderr=diagnostic,
            replay_exit_code=0,
        )
        is None
    )


def test_command_not_found_requires_both_tool_call_ids() -> None:
    diagnostic = "/bin/sh: 1: python: not found"
    assert (
        shell_command_lookup_failure_evidence(
            command="python",
            source_tool_call_id="",
            replay_tool_call_id="replay-1",
            source_command="python",
            source_tool_result=f"{diagnostic}\n\nExit code: 127",
            replay_result=diagnostic,
            replay_stderr=diagnostic,
            replay_exit_code=127,
        )
        is None
    )


def test_unrelated_failed_exec_evidence_does_not_resolve_static_clause() -> None:
    events = [
        event
        for event in _failed_exec_events()
        if event["type"] not in {"exec_arg", "failed_exec_attempt"}
        or event["exec_seq"] == 0
    ]
    events.extend(
        [
            _event(
                "exec_arg",
                140,
                999,
                seq=9,
                arg_index=0,
                arg="python",
            ),
            _event(
                "exec_arg",
                141,
                999,
                seq=9,
                arg_index=1,
                arg="-m",
            ),
            _event(
                "exec_arg",
                142,
                999,
                seq=9,
                arg_index=2,
                arg="pytest",
            ),
            _event(
                "failed_exec_attempt",
                150,
                999,
                seq=9,
                exit_code=2,
            ),
        ]
    )
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken(
            "call-unrelated",
            "cd /testbed && python -m pytest",
            100,
            0,
            0,
        ),
        ended_ns=230,
        events=events,
        loss=0,
        perf_samples=0,
    )
    assert summary["no_runtime_exec"] == []
    assert violations == [
        "call-unrelated: mapping gaps=unmatched_static_clause"
    ]


def test_exec_delimiter_uses_tool_call_id_and_original_command() -> None:
    class Agent:
        async def execute(
            self,
            request: dict[str, Any],
            *,
            timeout_s: float | None,
        ) -> dict[str, Any]:
            assert request["args"]["command"] == "cd /tmp && echo hi"
            assert timeout_s == 10.0
            return {"ok": True, "result": "hi", "returncode": 0}

    class Collector:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.finished = 0

        def begin_tool_call(self, tool_call_id: str, command: str) -> object:
            self.calls.append((tool_call_id, command))
            return object()

        def finish_tool_call(
            self,
            _token: object,
            *,
            replay_response: dict[str, Any] | None = None,
        ) -> None:
            self.finished += 1
            assert replay_response == {
                "ok": True,
                "result": "hi",
                "returncode": 0,
            }

    collector = Collector()
    tool = ContainerExecTool(
        Agent(),  # type: ignore[arg-type]
        timeout=10,
        workspace="/testbed",
        clause_telemetry=collector,
    )
    tool.set_tool_call_context("tc-123", {"command": "echo hi"})
    result = asyncio.run(
        tool.execute("echo hi", working_dir="/tmp", timeout=10)
    )
    tool.finish_clause_telemetry()
    assert result == "hi\n\nExit code: 0"
    assert collector.calls == [("tc-123", "echo hi")]
    assert collector.finished == 1


def test_attached_summary_is_keyed_by_tool_call_id(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "type": "action",
                "action_type": "tool_exec",
                "data": {
                    "tool_name": "exec",
                    "tool_call_id": "tc-1",
                    "tool_args": json.dumps({"command": "echo hi"}),
                    "tool_result": "hi\n\nExit code: 0",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "tool_call_id": "tc-1",
        "command": "echo hi",
        "integrity": {"status": "ok"},
    }
    source_actions = [
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_result": "hi\n\nExit code: 0",
            },
        }
    ]
    assert _attach_clause_telemetry(trace, [summary], source_actions) == []
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["data"]["clause_telemetry"] == summary
    assert record["data"]["exit_code_agreement"] == {
        "source": 0,
        "replay": 0,
        "available": True,
        "matches": True,
    }


def test_duplicate_tool_call_ids_fail_attachment(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    actions = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "duplicate",
                "tool_args": json.dumps({"command": command}),
            },
        }
        for command in ("echo one", "echo two")
    ]
    trace.write_text(
        "".join(json.dumps(action) + "\n" for action in actions),
        encoding="utf-8",
    )
    calls = [
        {"tool_call_id": "duplicate", "command": command}
        for command in ("echo one", "echo two")
    ]
    errors = _attach_clause_telemetry(trace, calls, [])
    assert "duplicate clause telemetry tool_call_id duplicate" in errors
    assert "duplicate exec action tool_call_id duplicate" in errors
    assert all(
        "clause_telemetry" not in json.loads(line)["data"]
        for line in trace.read_text(encoding="utf-8").splitlines()
    )


def test_attachment_rejects_command_mismatch(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "type": "action",
                "action_type": "tool_exec",
                "data": {
                    "tool_name": "exec",
                    "tool_call_id": "tc-1",
                    "tool_args": {"command": "echo actual"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    errors = _attach_clause_telemetry(
        trace,
        [{"tool_call_id": "tc-1", "command": "echo other"}],
        [],
    )
    assert errors == [
        "exec action tc-1 command does not match clause telemetry",
        "clause telemetry tc-1 has no matching exec action",
    ]
    assert "clause_telemetry" not in json.loads(trace.read_text())["data"]


def test_integrity_error_is_fatal_when_tool_errors_are_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tool:
        clause_telemetry_enabled = True

        async def execute(self, **_kwargs: Any) -> str:
            return "/bin/sh: 1: python: not found\n\n\nExit code: 127"

        def finish_clause_telemetry(self) -> None:
            raise ClauseTelemetryIntegrityError("mapping gap")

    class Tools:
        def prepare_call(
            self,
            _name: str,
            params: dict[str, Any],
        ) -> tuple[Tool, dict[str, Any], None]:
            return Tool(), params, None

    monkeypatch.setenv("OPENCLAW_TOOL_RESOURCE_TELEMETRY", "off")
    spec = AgentRunSpec(
        initial_messages=[],
        tools=Tools(),  # type: ignore[arg-type]
        model="unused",
        max_iterations=1,
        max_tool_result_chars=1,
    )
    result = asyncio.run(
        AgentRunner(None)._run_tool(  # type: ignore[arg-type]
            spec,
            ToolCallRequest("tc-failed", "exec", {"command": "true"}),
            {},
        )
    )
    assert result[0] == "/bin/sh: 1: python: not found\n\n\nExit code: 127"
    assert isinstance(result[2], ClauseTelemetryIntegrityError)


def test_integrity_finalization_preserves_tool_result_for_trace_hook() -> None:
    class Provider(LLMProvider):
        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> LLMResponse:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        "tc-failed",
                        "exec",
                        {"command": "python -m pytest"},
                    )
                ],
                finish_reason="tool_calls",
            )

        def get_default_model(self) -> str:
            return "fake"

    class ExecTool(Tool):
        clause_telemetry_enabled = True

        @property
        def name(self) -> str:
            return "exec"

        @property
        def description(self) -> str:
            return "execute"

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }

        async def execute(self, **_kwargs: Any) -> str:
            return "/bin/sh: 1: python: not found\n\n\nExit code: 127"

        def finish_clause_telemetry(self) -> None:
            raise ClauseTelemetryIntegrityError("mapping gap")

    class Hook(AgentHook):
        def __init__(self) -> None:
            self.tool_rows: list[dict[str, Any]] = []

        async def after_iteration(self, context: AgentHookContext) -> None:
            self.tool_rows = [
                message
                for message in context.messages
                if message.get("role") == "tool"
            ]

    registry = ToolRegistry()
    registry.register(ExecTool())
    hook = Hook()
    result = asyncio.run(
        AgentRunner(Provider()).run(
            AgentRunSpec(
                initial_messages=[],
                tools=registry,
                model="fake",
                max_iterations=1,
                max_tool_result_chars=10_000,
                hook=hook,
            )
        )
    )
    assert result.stop_reason == "tool_error"
    assert hook.tool_rows[0]["content"] == (
        "/bin/sh: 1: python: not found\n\n\nExit code: 127"
    )
