from __future__ import annotations

import asyncio
import json
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.container import ContainerExecTool
from agents.openclaw.tools.registry import ToolRegistry
from llm_call.provider_base import LLMProvider, LLMResponse, ToolCallRequest
from tool_resource import telemetry
from tool_resource.artifact_schema import CLAUSE_TELEMETRY_SCHEMA_VERSION
from tool_resource.telemetry import (
    ARG_FLAG_ARGV_CAPPED,
    ARG_FLAG_CONTINUED,
    ARG_FLAG_TRUNCATED,
    LOSS_COUNTER_NAMES,
    MAX_ARGS,
    ClauseTelemetryCollector,
    ClauseTelemetryIntegrityError,
    EventRow,
    ToolCallToken,
    _EventSpool,
    _captured_argv,
    _is_protocol_timeout,
    shell_command_lookup_failure_evidence,
    validate_clause_telemetry_runtime,
)
from trace_collect.cli import _run_simulate, parse_simulate_args
from trace_collect.openclaw_host_runtime import (
    _attach_resource_observations,
    _finalized_resource_status,
)


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
    arg_chunk_index: int = 0,
    arg: str = "",
    arg_flags: int = 0,
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
        "arg_chunk_index": arg_chunk_index,
        "arg": arg,
        "arg_flags": arg_flags,
        "exit_code": exit_code,
        "errno": exit_code if event_type == "failed_exec_attempt" else 0,
    }


def _collector_without_bpf() -> ClauseTelemetryCollector:
    collector = object.__new__(ClauseTelemetryCollector)
    collector.cgroup_id = 7
    collector.quota_cores = 4.0
    collector.repo = "repo"
    collector._spool_tmp = TemporaryDirectory(prefix="telemetry-test-")
    collector.artifact_path = Path(collector._spool_tmp.name) / "artifact.json"
    collector._epoch_offset_s = 1_000.0
    collector.state = "active"
    collector._disabled_reason = None
    collector._first_disabled_call = None
    collector._poll_error = None
    # Every real collector owns these from __init__, and the disabled one from
    # unavailable(); a fixture without them is not a reachable state.
    collector._events_lock = Lock()
    collector._poll_lock = Lock()
    collector._spool = _EventSpool(collector.artifact_path.parent)
    return collector


def _set_collector_events(
    collector: ClauseTelemetryCollector,
    events: list[dict[str, Any]],
) -> None:
    collector._spool.close()
    collector._spool = _EventSpool(collector.artifact_path.parent)
    for event in events:
        collector._spool.append(event)


def _active_collector() -> ClauseTelemetryCollector:
    collector = _collector_without_bpf()
    collector.container_id = "container"
    collector._bpf = SimpleNamespace(ring_buffer_consume=lambda: None)
    collector._events_lock = Lock()
    _set_collector_events(collector, _clean_events())
    collector._stop_poll = Event()
    collector._active = None
    collector._closed = False
    collector._cleanup_status = "not_started"
    collector._integrity_errors = []
    collector.calls = []
    collector._source_exec_actions = []
    collector._source_exec_index = 0
    return collector


def _clean_events() -> list[dict[str, Any]]:
    return [
        _event("fork", 110, 50, child=100),
        _event("exec_arg", 120, 100, seq=0, arg_index=0, arg="echo"),
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


def test_finalizer_error_cannot_reuse_stale_valid_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "clause-telemetry.json"
    artifact_path.write_text(
        json.dumps({"telemetry_quality": "ok", "collection_validity": "valid"}),
        encoding="utf-8",
    )
    collector = SimpleNamespace(
        finalize=lambda **_kwargs: "telemetry finalize failed: RuntimeError: boom",
        final_artifact=None,
    )

    status, errors = _finalized_resource_status(
        collector,
        replay_execution="completed",
    )

    assert status == {
        "telemetry_quality": "unavailable",
        "formal_completeness": "unavailable",
        "call_coverage": None,
        "collection_validity": "invalid",
    }
    assert errors == ["telemetry finalize failed: RuntimeError: boom"]


def test_finalized_status_comes_from_resource_artifact() -> None:
    collector = SimpleNamespace(
        finalize=lambda **_kwargs: None,
        final_artifact={
            "telemetry_quality": "ok",
            "formal_completeness": "partial",
            "call_coverage": {
                "total_call_count": 2,
                "eligible_call_count": 1,
                "withheld_call_count": 1,
                "eligible_fraction": 0.5,
            },
            "collection_validity": "valid",
        },
    )

    status, errors = _finalized_resource_status(
        collector,
        replay_execution="completed",
    )

    assert status["telemetry_quality"] == "ok"
    assert status["formal_completeness"] == "partial"
    assert status["call_coverage"]["eligible_fraction"] == 0.5
    assert status["collection_validity"] == "valid"
    assert errors == []


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


def test_cli_resource_profile_is_explicit() -> None:
    default = parse_simulate_args(["--manifest", "manifest.yaml"])
    configured = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--tool-resource-profile",
            "resource.yaml",
        ]
    )
    assert default.tool_resource_profile is None
    assert configured.tool_resource_profile == "resource.yaml"


def test_formal_resource_sweep_finishes_before_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[int] = []

    async def fake_simulate(**kwargs: Any) -> Path:
        concurrency = int(kwargs["concurrency"])
        seen.append(concurrency)
        trace = tmp_path / f"run-{concurrency}.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "type": "summary",
                    "collection_validity": ("invalid" if concurrency == 1 else "valid"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        trace.with_name(f"{trace.stem}.throughput_summary.json").write_text(
            json.dumps({"concurrency": concurrency, "run_id": str(concurrency)}) + "\n",
            encoding="utf-8",
        )
        return trace

    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--manifest",
            "manifest.yaml",
            "--output-dir",
            str(tmp_path),
            "--concurrency",
            "1,2",
            "--tool-resource-profile",
            str(tmp_path / "resource.yaml"),
        ]
    )

    with pytest.raises(SystemExit) as raised:
        _run_simulate(args)

    assert raised.value.code == 1
    assert seen == [1, 2]
    assert (tmp_path / "throughput_sweep.jsonl").exists()


def test_clause_runtime_rejects_configuration_before_bcc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tool_resource.telemetry.os.geteuid", lambda: 0)
    with pytest.raises(ValueError, match="container docker"):
        validate_clause_telemetry_runtime(
            container_executable="podman",
            concurrency=1,
            workers=1,
        )
    monkeypatch.setitem(sys.modules, "bcc", object())
    validate_clause_telemetry_runtime(
        container_executable="docker",
        concurrency=2,
        workers=1,
    )
    with pytest.raises(ValueError, match="workers 1"):
        validate_clause_telemetry_runtime(
            container_executable="docker",
            concurrency=2,
            workers=2,
        )


def test_collector_attach_failure_cleans_partial_bpf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeBPF:
        instance: "FakeBPF | None" = None

        def __init__(self, *, text: str) -> None:
            assert text
            self.cleaned = False
            FakeBPF.instance = self

        def attach_kprobe(self, **_kwargs: Any) -> None:
            raise RuntimeError("attach rejected")

        def cleanup(self) -> None:
            self.cleaned = True

    monkeypatch.setitem(
        sys.modules,
        "bcc",
        SimpleNamespace(BPF=FakeBPF, PerfSWConfig=object(), PerfType=object()),
    )
    monkeypatch.setattr(
        "tool_resource.telemetry._container_cgroup",
        lambda *_args: (tmp_path, 1),
    )
    monkeypatch.setattr(
        "tool_resource.telemetry.observed_quota_cores",
        lambda _path: 1.0,
    )

    with pytest.raises(RuntimeError, match="attach rejected"):
        ClauseTelemetryCollector(
            container_id="container",
            container_executable="docker",
            repo="repo",
            artifact_path=tmp_path / "clause.json",
        )

    assert FakeBPF.instance is not None
    assert FakeBPF.instance.cleaned


def test_summary_preserves_structural_gap_and_target_availability() -> None:
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-1", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=_clean_events(),
        loss_counts={},
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
            "reason": ("sentinel_exec_seq_without_active_exec_image_or_owned_ancestor"),
        }
    ]
    assert summary["target_availability"]["latency"]["available"] == 1
    assert summary["target_availability"]["cpu"]["reasons"] == {
        "unknown:clause_shorter_than_1s_ineligible_for_peak": 1
    }
    assert summary["telemetry_loss"] == {
        "ringbuf_reserve_failures": 0,
        "argv_read_failures": 0,
        "argv_boundary_read_failures": 0,
        "total": 0,
        "perf_sample_count": 2,
    }
    assert summary["ring_loss"] == {
        "reserve_failures": 0,
        "perf_sample_count": 2,
    }
    assert summary["clauses"][0]["ts_start"] < summary["clauses"][0]["ts_end"]
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


@pytest.mark.parametrize("cause", LOSS_COUNTER_NAMES)
def test_every_telemetry_loss_cause_fails_closed(cause: str) -> None:
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-loss", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=_clean_events(),
        loss_counts={cause: 1},
        perf_samples=2,
    )

    assert summary["integrity"]["status"] == "failed"
    assert summary["telemetry_loss"][cause] == 1
    assert summary["telemetry_loss"]["total"] == 1
    assert summary["ring_loss"]["reserve_failures"] == 1
    assert summary["clauses"] == []
    assert any("telemetry loss=1" in violation for violation in violations)


def test_capped_runtime_invocation_persists_exact_argc_and_zero_observations() -> None:
    argv = ("/bin/cmd", *[str(index) for index in range(16)])
    events = [
        _event("fork", 110, 50, child=100),
        _event("exec_meta", 115, 100, seq=0, arg="/bin/cmd"),
        _event("bprm_meta", 116, 100, seq=0, arg="/bin/cmd", exit_code=len(argv)),
        _event("interp_meta", 117, 100, seq=0, arg="/bin/cmd", exit_code=len(argv)),
        *[
            _event("exec_arg", 120 + index, 100, seq=0, arg_index=index, arg=word)
            for index, word in enumerate(argv[:MAX_ARGS])
        ],
        _event(
            "exec_arg",
            140,
            100,
            seq=0,
            arg_index=MAX_ARGS,
            arg_flags=ARG_FLAG_ARGV_CAPPED,
            exit_code=len(argv),
        ),
        _event("exec_boundary", 150, 100, seq=0, parent=50),
        _event("exit_boundary", 220, 100, seq=0, parent=50),
    ]
    collector = _collector_without_bpf()

    summary, violations = collector._summarize_call(
        token=ToolCallToken("capped", "time " + " ".join(argv), 100, 0, 0),
        ended_ns=230,
        events=events,
        loss_counts={},
        perf_samples=0,
    )

    assert len(violations) == 1
    assert "runtime_argv_incomplete" in violations[0]
    assert summary["mapping"]["observation_clause_count"] == 0
    assert summary["clauses"] == []
    assert summary["static_word_intent"][0]["structural_context"] == ["time"]
    assert summary["runtime_invocations"][0] == {
        "host_pid": 100,
        "exec_seq": 0,
        "requested_executable_path": "/bin/cmd",
        "requested_executable_path_truncated": False,
        "argv": list(argv[:MAX_ARGS]),
        "argc": len(argv),
        "argv_capped": True,
        "truncated_words": [],
        "bprm_filename": "/bin/cmd",
        "bprm_interp": "/bin/cmd",
        "bprm_evidence_truncated": False,
    }


def test_chunked_argv_requires_one_complete_contiguous_sequence() -> None:
    payload = ("a" * 510 + "雪").encode()
    first = _event(
        "exec_arg",
        100,
        10,
        seq=1,
        arg_index=2,
        arg_chunk_index=0,
        arg=payload[:511].decode("utf-8", "replace"),
        arg_flags=ARG_FLAG_CONTINUED,
    )
    first["arg_raw"] = payload[:511].hex()
    second = _event(
        "exec_arg",
        101,
        10,
        seq=1,
        arg_index=2,
        arg_chunk_index=1,
        arg=payload[511:].decode("utf-8", "replace"),
    )
    second["arg_raw"] = payload[511:].hex()
    events = [
        first,
        second,
        _event(
            "exec_arg",
            102,
            20,
            seq=2,
            arg_index=1,
            arg_chunk_index=1,
            arg="orphan",
        ),
        *[
            _event(
                "exec_arg",
                103 + chunk,
                30,
                seq=3,
                arg_index=1,
                arg_chunk_index=chunk,
                arg=str(chunk),
                arg_flags=ARG_FLAG_CONTINUED if chunk < 8 else 0,
            )
            for chunk in range(9)
        ],
        _event(
            "exec_arg",
            112,
            40,
            seq=4,
            arg_index=1,
            arg="malformed",
            arg_flags=ARG_FLAG_ARGV_CAPPED,
        ),
    ]

    words, flags = _captured_argv(events)

    assert words[(10, 1)][2] == "a" * 510 + "雪"
    assert flags.get((10, 1), 0) == 0
    assert words[(20, 2)][1] == "orphan"
    assert flags[(20, 2)] == 1 << 1
    assert flags[(30, 3)] == 1 << 1
    assert flags[(40, 4)] == 1 << 1


def test_event_row_keeps_one_raw_argv_payload() -> None:
    payload = b"\xffraw"
    row = EventRow(
        "exec_arg",
        1,
        7,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        10,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        arg_payload=payload,
    )

    assert row._arg_payload is payload
    assert row["arg"] == "\ufffdraw"
    assert row["arg_raw"] == payload.hex()


def test_event_row_preserves_absent_payload_across_process_pickle() -> None:
    row = EventRow(
        "perf",
        1,
        7,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        10,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    restored = pickle.loads(pickle.dumps(row))

    assert dict(restored) == dict(row)
    assert "arg" not in restored


def test_event_spool_stably_sorts_across_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        telemetry,
        "_EVENT_SPOOL_SEGMENT_BYTES",
        telemetry._EVENT_RECORD.size,
    )
    spool = _EventSpool(tmp_path)
    for ts_ns, arg in ((20, "first"), (10, "earlier"), (20, "second")):
        spool.append(_event("exec_arg", ts_ns, 10, seq=1, arg=arg))

    with telemetry._sorted_event_source(
        spool.snapshot(),
        started_ns=0,
        ended_ns=20,
        cgroup_id=7,
        directory=tmp_path,
    ) as source:
        assert [(event["ts_ns"], event["arg"]) for event in source] == [
            (10, "earlier"),
            (20, "first"),
            (20, "second"),
        ]
    spool.close()


def test_event_spool_preserves_call_summary() -> None:
    events = _clean_events()
    token = ToolCallToken("call-spool", "echo hi", 100, 0, 0)
    baseline, baseline_violations = _collector_without_bpf()._summarize_call(
        token=token,
        ended_ns=230,
        events=events,
        loss_counts={},
        perf_samples=2,
    )

    collector = _collector_without_bpf()
    for event in events:
        collector._spool.append(event)
    snapshot = collector._spool.snapshot()
    source = telemetry._sorted_event_source(
        snapshot,
        started_ns=100,
        ended_ns=230,
        cgroup_id=7,
        directory=collector.artifact_path.parent,
    )
    with source:
        spooled, spooled_violations = collector._summarize_call(
            token=token,
            ended_ns=230,
            events=source,
            raw_event_count=len(source),
            loss_counts={},
            perf_samples=2,
        )

    assert spooled == baseline
    assert spooled_violations == baseline_violations


def test_event_spool_preserves_pending_exec_start_evidence() -> None:
    events = [
        _event("exec_arg", 10, 100, seq=0, arg="sh"),
        _event("perf", 15, 100, cpu_ns=1, rss_pages=1, mm_ptr=10),
        _event("exec_boundary", 20, 100, seq=0),
        _event("exit_boundary", 30, 100, seq=0),
    ]
    baseline_clauses, baseline_fork_parent = telemetry._clauses_and_lineage(events)
    _, baseline_gaps = telemetry._attribute(
        events,
        baseline_clauses,
        baseline_fork_parent,
        entry_pid=50,
    )

    collector = _collector_without_bpf()
    for event in events:
        collector._spool.append(event)
    snapshot = collector._spool.snapshot()
    source = telemetry._sorted_event_source(
        snapshot,
        started_ns=0,
        ended_ns=30,
        cgroup_id=7,
        directory=collector.artifact_path.parent,
    )
    with source:
        spooled_clauses, spooled_fork_parent = telemetry._clauses_and_lineage(source)
        _, spooled_gaps = telemetry._attribute(
            source,
            spooled_clauses,
            spooled_fork_parent,
            entry_pid=50,
        )

    assert spooled_gaps == baseline_gaps
    assert spooled_gaps[0]["reason"] == (
        "initial_exec_pending_pre_boundary_structural_setup"
    )


def test_event_spool_split_by_call_boundary_fails_closed() -> None:
    collector = _collector_without_bpf()
    collector._spool.append(
        _event(
            "exec_arg",
            99,
            10,
            seq=1,
            arg="before",
            arg_flags=ARG_FLAG_CONTINUED,
        ),
    )
    collector._spool.append(
        _event(
            "exec_arg",
            101,
            10,
            seq=1,
            arg_chunk_index=1,
            arg="inside",
        ),
    )
    collector._spool.append(_event("exec_boundary", 105, 10, seq=1))
    snapshot = collector._spool.snapshot()
    source = telemetry._sorted_event_source(
        snapshot,
        started_ns=100,
        ended_ns=110,
        cgroup_id=7,
        directory=collector.artifact_path.parent,
    )
    with source:
        words, flags = _captured_argv(source)

    assert words == {(10, 1): {0: "inside"}}
    assert flags == {(10, 1): 1}


def test_truncated_requested_path_invalidates_bare_head_mapping() -> None:
    events = [
        _event(
            "exec_meta",
            119,
            100,
            seq=0,
            arg="/very/long/path",
            arg_flags=ARG_FLAG_TRUNCATED,
        ),
        *_clean_events(),
    ]
    collector = _collector_without_bpf()

    summary, violations = collector._summarize_call(
        token=ToolCallToken("path-truncated", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=events,
        loss_counts={},
        perf_samples=2,
    )

    assert len(violations) == 1
    assert "runtime_argv_incomplete" in violations[0]
    assert summary["mapping"]["observation_clause_count"] == 0
    assert summary["runtime_invocations"][0]["requested_executable_path_truncated"]


def test_mapping_failure_does_not_disable_later_valid_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _active_collector()
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)
    monkeypatch.setattr("tool_resource.telemetry.time.monotonic_ns", lambda: 230)

    bad = ToolCallToken("bad", "missing arg", 100, 0, 0)
    collector._active = bad
    first = collector.finish_tool_call(bad, replay_response={"returncode": 0})
    _set_collector_events(collector, _clean_events())
    good = ToolCallToken("good", "echo hi", 100, 0, 0)
    collector._active = good
    second = collector.finish_tool_call(good, replay_response={"returncode": 0})

    assert first["telemetry_quality"] == "invalid"
    assert first["clauses"] == []
    assert second["telemetry_quality"] == "ok"
    assert second["eligible_for_kb"]
    assert collector.state == "active"


def test_internal_analysis_failure_disables_later_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _active_collector()
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)
    monkeypatch.setattr("tool_resource.telemetry.time.monotonic_ns", lambda: 230)

    def fail_analysis(**_kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        raise RuntimeError("analyzer state corrupt")

    monkeypatch.setattr(collector, "_summarize_call", fail_analysis)
    token = ToolCallToken("broken", "echo hi", 100, 0, 0)
    collector._active = token
    failed = collector.finish_tool_call(token, replay_response={"returncode": 0})
    next_token = collector.begin_tool_call("next", "echo again")
    following = collector.finish_tool_call(
        next_token, replay_response={"returncode": 0}
    )

    assert failed["telemetry_quality"] == "invalid"
    assert collector.state == "disabled"
    assert collector._first_disabled_call == "broken"
    assert following["telemetry_quality"] == "unavailable"


def test_event_spool_read_failure_disables_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _active_collector()
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)

    def fail_spool(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("spool read failed")

    monkeypatch.setattr(
        "tool_resource.telemetry._sorted_event_source",
        fail_spool,
    )
    token = ToolCallToken("spool-broken", "echo hi", 100, 0, 0)
    collector._active = token

    failed = collector.finish_tool_call(
        token,
        replay_response={"returncode": 0},
        ended_ns=230,
    )

    assert failed["telemetry_quality"] == "unavailable"
    assert failed["eligible_for_kb"] is False
    assert collector.state == "disabled"
    assert collector._first_disabled_call == "spool-broken"


def test_event_spool_prepare_failure_does_not_block_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _active_collector()
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)

    def fail_spool(_started_ns: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(collector._spool, "drop_before", fail_spool)

    token = collector.begin_tool_call("spool-full", "echo hi", started_ns=100)

    assert token.tool_call_id == "spool-full"
    assert collector._active is token
    assert collector.state == "disabled"
    assert collector._first_disabled_call == "spool-full"


def test_event_spool_closes_when_bpf_cleanup_fails() -> None:
    collector = _collector_without_bpf()
    collector._closed = False
    collector._stop_poll = Event()
    collector._poller = SimpleNamespace(
        join=lambda **_kwargs: None,
        is_alive=lambda: False,
    )
    collector._perf_type = SimpleNamespace(SOFTWARE=1)
    collector._perf_config = SimpleNamespace(CPU_CLOCK=2)

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    collector._bpf = SimpleNamespace(
        detach_perf_event=lambda **_kwargs: None,
        cleanup=fail_cleanup,
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        collector._close_bpf()

    assert collector._spool._current.file.closed
    assert collector._closed is True


def test_per_call_loss_does_not_disable_later_valid_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _active_collector()
    loss_deltas = iter(
        [
            {
                "ringbuf_reserve_failures": 1,
                "argv_read_failures": 0,
                "argv_boundary_read_failures": 0,
            },
            dict.fromkeys(LOSS_COUNTER_NAMES, 0),
        ]
    )
    monkeypatch.setattr(
        "tool_resource.telemetry._loss_delta",
        lambda *_: next(loss_deltas),
    )
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)
    monkeypatch.setattr("tool_resource.telemetry.time.monotonic_ns", lambda: 230)

    first_token = ToolCallToken("loss", "echo hi", 100, 0, 0)
    collector._active = first_token
    first = collector.finish_tool_call(first_token, replay_response={"returncode": 0})
    _set_collector_events(collector, _clean_events())
    second_token = ToolCallToken("recovered", "echo hi", 100, 0, 0)
    collector._active = second_token
    second = collector.finish_tool_call(second_token, replay_response={"returncode": 0})

    assert first["telemetry_quality"] == "invalid"
    assert first["mapping"]["observation_clause_count"] == 0
    assert second["telemetry_quality"] == "ok"
    assert collector.state == "active"


def test_poller_failure_disables_session_and_marks_following_calls_unavailable() -> (
    None
):
    collector = _active_collector()
    collector._poll_error = RuntimeError("poll stopped")

    first_token = collector.begin_tool_call("first", "echo hi")
    first = collector.finish_tool_call(first_token, replay_response={"returncode": 0})
    second_token = collector.begin_tool_call("second", "echo again")
    second = collector.finish_tool_call(second_token, replay_response={"returncode": 0})

    assert collector.state == "disabled"
    assert collector._first_disabled_call == "first"
    assert collector._stop_poll.is_set()
    assert first["telemetry_quality"] == "unavailable"
    assert second["telemetry_quality"] == "unavailable"
    assert second["invalid_reasons"][0]["kind"] == "collector_disabled"


@pytest.mark.parametrize(
    "marker",
    ["[timeout]", "[resource_timeout]", "[resource_stall_timeout]"],
)
def test_protocol_timeout_requires_124_and_exact_marker_line(marker: str) -> None:
    assert _is_protocol_timeout(124, f"output\n{marker}\n")
    assert not _is_protocol_timeout(124, "ordinary failure")
    assert not _is_protocol_timeout(1, marker)
    assert not _is_protocol_timeout(124, f"prefix {marker} suffix")


def test_protocol_timeout_artifact_keeps_metrics_without_new_schema_fields() -> None:
    collector = _collector_without_bpf()
    events = [
        _event("fork", 1, 50, child=100),
        _event("exec_arg", 10_000_000, 100, seq=0, arg="slow"),
        _event(
            "exec_boundary",
            20_000_000,
            100,
            seq=0,
            parent=50,
            rss_pages=256,
            mm_ptr=10,
        ),
        _event(
            "perf",
            520_000_000,
            100,
            seq=0,
            cpu_ns=400_000_000,
            rss_pages=256,
            mm_ptr=10,
        ),
        _event(
            "perf",
            1_020_000_000,
            100,
            seq=0,
            cpu_ns=800_000_000,
            rss_pages=512,
            mm_ptr=10,
        ),
        _event(
            "exit_boundary",
            1_220_000_000,
            100,
            seq=0,
            parent=50,
            cpu_ns=900_000_000,
            rss_pages=512,
            mm_ptr=10,
            exit_code=9,
            io_read=10,
            io_write=20,
        ),
    ]

    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-timeout", "slow", 0, 0, 0),
        ended_ns=1_300_000_000,
        events=events,
        loss_counts={},
        perf_samples=2,
        protocol_timeout=True,
    )

    assert violations == []
    row = summary["clauses"][0]
    assert row["latency_ms"] is None
    assert row["peak_cpu_cores"] is not None
    assert row["sampled_peak_rss_mb"] is not None
    assert row["cpu_ns_cumulative"] == 900_000_000
    assert row["disk_io"]["read_bytes_total"] == 10
    assert row["disk_io"]["write_bytes_total"] == 20
    assert set(row["availability"].values()) == {"unknown:protocol_timeout"}
    assert summary["mapping"]["observation_clause_count"] == 0

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for child in value.values() for key in keys(child)}
        if isinstance(value, list):
            return {key for child in value for key in keys(child)}
        return set()

    assert keys(summary).isdisjoint(
        {
            "argv_capture_flags",
            "right_censored",
            "censored_wall_ms",
            "source_agreed",
            "timeout_evidence",
            "protocol_timeout",
        }
    )


def test_fork_only_processes_collapse_to_nearest_transitive_exec_ancestor() -> None:
    collector = _collector_without_bpf()
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-apt", "apt-get update", 100, 0, 0),
        ended_ns=250,
        events=_apt_fork_chain_events(),
        loss_counts={},
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
    collector._bpf = SimpleNamespace(ring_buffer_consume=lambda: None)
    collector._events_lock = Lock()
    _set_collector_events(collector, events)
    collector.calls = []
    collector._integrity_errors = []
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)

    summary = collector.finish_tool_call(token, replay_response={"returncode": 0})

    assert summary["telemetry_quality"] == "invalid"
    assert not summary["eligible_for_kb"]
    assert collector.state == "active"
    tree = summary["provenance"]["command_tree"]
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
            loss_counts={},
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


def test_short_circuit_source_replay_disagreement_invalidates_only_telemetry(
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
    collector._bpf = SimpleNamespace(ring_buffer_consume=lambda: None)
    collector._events_lock = Lock()
    _set_collector_events(collector, _control_events(1))
    collector.calls = []
    collector._integrity_errors = []
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)

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


def _mapped_control_events(exit_status: int) -> list[dict[str, Any]]:
    # Same shape as _control_events, but argv[0] matches the static head so the
    # controller clause actually maps and can supply a normal exit status.
    return [
        _event("fork", 110, 50, child=100),
        _event("exec_arg", 120, 100, seq=0, arg="left"),
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


def _finish_control_call(
    monkeypatch: pytest.MonkeyPatch, exit_status: int
) -> dict[str, Any]:
    collector = _collector_without_bpf()
    token = ToolCallToken(
        "call-control",
        "left && right",
        100,
        0,
        0,
        source_tool_call_id="source-control",
        source_command="left && right",
        source_tool_result="source success\n\nExit code: 0",
    )
    collector._active = token
    collector._bpf = SimpleNamespace(ring_buffer_consume=lambda: None)
    collector._events_lock = Lock()
    _set_collector_events(collector, _mapped_control_events(exit_status))
    collector.calls = []
    collector._integrity_errors = []
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)
    collector.finish_tool_call(
        token,
        # Replay output deliberately differs from the source, so
        # tool_result_exact and short_circuit_eligible are both False.
        replay_response={"ok": True, "result": "replay failure", "returncode": 1},
    )
    return collector.calls[0]


def test_short_circuit_resolves_without_byte_exact_replay_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Whether a clause executed in THIS replay is decided by replay-side kernel
    # evidence, not by the replay reproducing the source trace byte-for-byte.
    # Stateful workloads (Terminal-Bench) rarely reproduce output exactly.
    call = _finish_control_call(monkeypatch, exit_status=1)
    fidelity = call["provenance"]["source_replay_control_flow_fidelity"]
    assert fidelity["tool_result_exact"] is False
    assert fidelity["short_circuit_eligible"] is False
    assert [
        (item["bin"], item["provenance"]["evidence_kind"])
        for item in call["no_runtime_exec"]
    ] == [("right", "shell_control_short_circuit")]
    assert call["mapping"]["gaps"] == []
    assert call["eligible_for_kb"] is True


def test_consumed_events_are_pruned_after_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _collector_without_bpf()
    token = ToolCallToken("call-prune", "left && right", 100, 0, 0)
    collector._active = token
    collector._bpf = SimpleNamespace(ring_buffer_consume=lambda: None)
    collector._events_lock = Lock()
    stale = dict(_mapped_control_events(1)[0], ts_ns=10)
    future = dict(_mapped_control_events(1)[0], ts_ns=230)
    _set_collector_events(
        collector,
        [stale, *_mapped_control_events(1), future],
    )
    collector.calls = []
    collector._integrity_errors = []
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    monkeypatch.setattr("tool_resource.telemetry.time.sleep", lambda *_: None)

    collector.finish_tool_call(
        token,
        replay_response={"ok": True, "result": "ok", "returncode": 0},
        ended_ns=220,
    )

    snapshot = collector._spool.snapshot()
    with telemetry._sorted_event_source(
        snapshot,
        started_ns=221,
        ended_ns=230,
        cgroup_id=7,
        directory=collector.artifact_path.parent,
    ) as remaining:
        assert [dict(event) for event in remaining] == [future]


def test_finish_drains_ring_before_spool_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _active_collector()
    _set_collector_events(collector, _clean_events()[:-1])
    late_exit = _clean_events()[-1]

    def consume() -> None:
        with collector._events_lock:
            collector._spool.append(late_exit)

    collector._bpf = SimpleNamespace(ring_buffer_consume=consume)
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)
    token = ToolCallToken("call-drain", "echo hi", 100, 0, 0)
    collector._active = token

    summary = collector.finish_tool_call(
        token,
        replay_response={"returncode": 0},
        ended_ns=230,
    )

    assert summary["clauses"][0]["provenance"]["boundary_coverage"] == {
        "has_exec": True,
        "has_exit": True,
    }


def test_short_circuit_fails_closed_when_controller_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Controller exited 0, so the `&&` right operand SHOULD have run. Its
    # absence is unexplained and must still withhold the call.
    call = _finish_control_call(monkeypatch, exit_status=0)
    assert call["no_runtime_exec"] == []
    assert [gap["kind"] for gap in call["mapping"]["gaps"]] == [
        "unmatched_static_clause"
    ]
    assert call["eligible_for_kb"] is False


def test_relevant_gap_fails_integrity() -> None:
    collector = _collector_without_bpf()
    events = _clean_events()
    events.append(_event("perf", 220, 100, cpu_ns=4, rss_pages=1, mm_ptr=10))
    summary, violations = collector._summarize_call(
        token=ToolCallToken("call-2", "echo hi", 100, 0, 0),
        ended_ns=230,
        events=events,
        loss_counts={},
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
    assert summary["mapping"]["observation_clause_count"] == 0
    assert not summary["eligible_for_kb"]


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
        loss_counts={},
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
        loss_counts={},
        perf_samples=3,
    )

    assert violations == []
    setup = [
        event
        for event in summary["coverage_gaps"]["structural"]["events"]
        if event["entry_parent_relation"] == "entry_fork_pre_exec_structural_setup"
    ]
    assert setup == [
        {
            "type": "perf",
            "ts_ns": 115,
            "host_pid": 100,
            "host_tid": 100,
            "exec_seq": 2**64 - 1,
            "entry_pid": 50,
            "entry_parent_relation": ("entry_fork_pre_exec_structural_setup"),
            "fork_parent_pid": 50,
            "reason": "entry_fork_pre_exec_structural_setup",
            "fork_ancestry": [100, 50],
            "fork_chain_records": [{"child_id": 100, "parent_pid": 50, "ts_ns": 110}],
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
        loss_counts={},
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


def test_failed_exec_enoent_is_zero_without_mapping_gap() -> None:
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
        loss_counts={},
        perf_samples=0,
    )
    assert violations == []
    assert summary["mapping"] == {
        "static_clause_count": 2,
        "mappable_clause_count": 1,
        "mapped_clause_count": 1,
        "observation_clause_count": 1,
        "no_runtime_exec_count": 0,
        "coverage": 1.0,
        "gaps": [],
        "unobserved_builtins": ["cd"],
    }
    assert summary["no_runtime_exec"] == []
    assert summary["clauses"][0]["latency_ms"] == 0.0
    assert summary["clauses"][0]["mapping_evidence"] == "failed_exec_enoent_zero"
    assert summary["target_availability"]["latency"]["reasons"] == {"ok": 1}


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
        loss_counts={},
        perf_samples=0,
    )
    assert summary["no_runtime_exec"] == []
    assert violations == ["call-unmatched: mapping gaps=unmatched_static_clause"]


def test_incomplete_failed_exec_argv_does_not_resolve_static_clause() -> None:
    collector = _collector_without_bpf()
    events = _failed_exec_events()
    next(event for event in events if event["type"] == "failed_exec_attempt")[
        "arg_flags"
    ] = ARG_FLAG_TRUNCATED

    summary, violations = collector._summarize_call(
        token=ToolCallToken(
            "call-incomplete-failed",
            "cd /testbed && python -m pytest",
            100,
            0,
            0,
        ),
        ended_ns=230,
        events=events,
        loss_counts={},
        perf_samples=0,
    )

    assert summary["no_runtime_exec"] == []
    assert violations == [
        "call-incomplete-failed: mapping gaps=unmatched_static_clause"
    ]


def test_direct_command_not_found_is_zero_target_evidence() -> None:
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
        loss_counts={},
        perf_samples=0,
        command_lookup_failure=evidence,
    )
    assert violations == []
    assert summary["no_runtime_exec"] == []
    row = summary["clauses"][0]
    assert row["latency_ms"] == 0.0
    assert row["mapping_evidence"] == "shell_command_lookup_failure_zero"
    assert row["provenance"]["command_lookup_failure"]["executable_head"] == "python"
    assert (
        row["provenance"]["command_lookup_failure"]["exit_code_semantics"]
        == "direct_command_not_found_127"
    )


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
            source_tool_result=("/bin/sh: 1: /opt/python: not found\n\nExit code: 127"),
            replay_result="/bin/sh: 1: /usr/bin/python: not found",
            replay_stderr="/bin/sh: 1: /usr/bin/python: not found",
            replay_exit_code=127,
        )
        is None
    )


@pytest.mark.parametrize(
    "command",
    ["python; true", "python; echo ok"],
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


def test_exit_zero_command_not_found_accepts_explicit_or_true() -> None:
    command = "python || true"
    diagnostic = "/bin/sh: 1: python: not found"
    evidence = shell_command_lookup_failure_evidence(
        command=command,
        source_tool_call_id="source-1",
        replay_tool_call_id="replay-1",
        source_command=command,
        source_tool_result=f"{diagnostic}\n\nExit code: 0",
        replay_result=diagnostic,
        replay_stderr=diagnostic,
        replay_exit_code=0,
    )
    assert evidence is not None
    assert evidence.exit_code_semantics == "or_true_masked_0"


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
        loss_counts={},
        perf_samples=0,
    )
    assert summary["no_runtime_exec"] == []
    assert violations == ["call-unrelated: mapping gaps=unmatched_static_clause"]


def test_guard_blocked_exec_is_explicit_no_runtime_and_advances_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = "cd /testbed && rm -f scratch.py"
    result_text = (
        "Error: Command blocked by safety guard (dangerous pattern detected)\n\n"
        "[Analyze the error above and try a different approach.]"
    )
    collector = _collector_without_bpf()
    collector._closed = False
    collector._active = None
    collector._bpf = SimpleNamespace(ring_buffer_consume=lambda: None)
    collector.calls = []
    collector._integrity_errors = []
    collector._source_exec_actions = [
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "source-guard",
                "tool_args": json.dumps({"command": command}),
                "tool_result": result_text,
            },
        }
    ]
    collector._source_exec_index = 0
    monkeypatch.setattr("tool_resource.telemetry._counter", lambda *_: 0)

    summary = collector.record_safety_guard_blocked(
        "replay-guard",
        command,
        result_text,
    )

    assert collector._source_exec_index == 1
    assert collector._active is None
    assert collector.calls == [summary]
    assert summary["integrity"] == {"status": "ok", "errors": []}
    assert summary["mapping"]["no_runtime_exec_count"] == 1
    assert summary["mapping"]["unobserved_builtins"] == ["cd"]
    assert summary["no_runtime_exec"][0]["bin"] == "rm"
    assert summary["no_runtime_exec"][0]["provenance"]["evidence_kind"] == (
        "safety_guard_blocked_before_runtime"
    )


def test_container_guard_block_records_exact_final_replay_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Agent:
        async def execute(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("guard-blocked command must not enter container")

    class Collector:
        def __init__(self) -> None:
            self.records: list[tuple[str, str, str]] = []

        def record_safety_guard_blocked(
            self,
            tool_call_id: str,
            command: str,
            result: str,
        ) -> None:
            self.records.append((tool_call_id, command, result))

    collector = Collector()
    tool = ContainerExecTool(
        Agent(),  # type: ignore[arg-type]
        timeout=10,
        workspace="/testbed",
        resource_trace=collector,
    )
    registry = ToolRegistry()
    registry.register(tool)
    monkeypatch.setenv("OPENCLAW_RESOURCE_TIMELINE", "off")
    result = asyncio.run(
        AgentRunner(None)._run_tool(  # type: ignore[arg-type]
            AgentRunSpec(
                initial_messages=[],
                tools=registry,
                model="unused",
                max_iterations=1,
                max_tool_result_chars=1,
            ),
            ToolCallRequest(
                "guard-call",
                "exec",
                {"command": "rm -f scratch.py"},
            ),
            {},
        )
    )[0]

    assert result.startswith("Error: Command blocked by safety guard")
    assert result.count("[Analyze the error above and try a different approach.]") == 1
    assert collector.records == [("guard-call", "rm -f scratch.py", result)]


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
        resource_trace=collector,
    )
    tool.set_tool_call_context("tc-123", {"command": "echo hi"})
    result = asyncio.run(tool.execute("echo hi", working_dir="/tmp", timeout=10))
    tool.finish_resource_call()
    assert result == "hi\n\nExit code: 0"
    assert collector.calls == [("tc-123", "echo hi")]
    assert collector.finished == 1


def test_unavailable_collector_preserves_tool_output_and_exit_code(
    tmp_path: Path,
) -> None:
    class Agent:
        async def execute(
            self,
            _request: dict[str, Any],
            *,
            timeout_s: float | None,
        ) -> dict[str, Any]:
            assert timeout_s == 10.0
            return {"ok": False, "result": "original stderr", "returncode": 7}

    collector = ClauseTelemetryCollector.unavailable(
        repo="repo",
        artifact_path=tmp_path / "clause.json",
        reason="attach failed",
    )
    tool = ContainerExecTool(
        Agent(),  # type: ignore[arg-type]
        timeout=10,
        workspace="/testbed",
        resource_trace=collector,
    )
    tool.set_tool_call_context("call-1", {"command": "false"})

    result = asyncio.run(tool.execute("false"))
    tool.finish_resource_call()

    assert result == "Error: original stderr\n\nExit code: 7"
    assert collector.calls[0]["telemetry_quality"] == "unavailable"


def test_concurrent_tools_isolate_one_telemetry_failure() -> None:
    class Agent:
        def __init__(self, result: str) -> None:
            self.result = result

        async def execute(
            self,
            _request: dict[str, Any],
            *,
            timeout_s: float | None,
        ) -> dict[str, Any]:
            return {"ok": True, "result": self.result, "returncode": 0}

    class BrokenCollector:
        def begin_tool_call(self, *_args: Any) -> object:
            raise RuntimeError("attach stream failed")

        def add_integrity_error(self, _message: str) -> None:
            return None

    class HealthyCollector:
        def __init__(self) -> None:
            self.finished = False

        def begin_tool_call(self, *_args: Any) -> object:
            return object()

        def finish_tool_call(
            self,
            _token: object,
            *,
            replay_response: dict[str, Any] | None = None,
        ) -> None:
            assert replay_response is not None
            self.finished = True

    healthy = HealthyCollector()
    broken_tool = ContainerExecTool(
        Agent("broken-stream workload"),  # type: ignore[arg-type]
        timeout=10,
        workspace="/testbed",
        resource_trace=BrokenCollector(),
    )
    healthy_tool = ContainerExecTool(
        Agent("healthy-stream workload"),  # type: ignore[arg-type]
        timeout=10,
        workspace="/testbed",
        resource_trace=healthy,
    )
    broken_tool.set_tool_call_context("broken", {"command": "echo broken"})
    healthy_tool.set_tool_call_context("healthy", {"command": "echo healthy"})

    async def run_both() -> tuple[str, str]:
        first, second = await asyncio.gather(
            broken_tool.execute("echo broken"),
            healthy_tool.execute("echo healthy"),
        )
        broken_tool.finish_resource_call()
        healthy_tool.finish_resource_call()
        return first, second

    assert asyncio.run(run_both()) == (
        "broken-stream workload\n\nExit code: 0",
        "healthy-stream workload\n\nExit code: 0",
    )
    assert healthy.finished


def test_artifact_records_disabled_session_and_replay_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clause.json"
    collector = ClauseTelemetryCollector.unavailable(
        repo="repo",
        artifact_path=path,
        reason="collector attach failed",
    )
    token = collector.begin_tool_call("call-1", "echo hi")
    collector.finish_tool_call(token, replay_response={"returncode": 0})
    collector.finalize(replay_execution="failed")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["version"] == CLAUSE_TELEMETRY_SCHEMA_VERSION
    assert artifact["replay_execution"] == "failed"
    assert artifact["telemetry_quality"] == "unavailable"
    assert artifact["collection_validity"] == "invalid"
    assert artifact["collector"] == {
        "state": "closed",
        "state_before_close": "disabled",
        "health": "unavailable",
        "first_disabled_call": None,
        "disabled_reason": "collector attach failed",
        "valid_call_count": 0,
        "invalid_call_count": 0,
        "unavailable_call_count": 1,
        "eligible_call_count": 0,
    }
    assert artifact["formal_completeness"] == "unavailable"
    assert artifact["call_coverage"]["eligible_fraction"] == 0.0


def test_replay_failure_is_separate_from_healthy_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _active_collector()
    collector.artifact_path = tmp_path / "clause.json"
    trimmed: list[tuple[str, str, str]] = []

    def record_trim() -> None:
        artifact = json.loads(collector.artifact_path.read_text(encoding="utf-8"))
        trimmed.append(
            (collector.state, collector._cleanup_status, artifact["collection_validity"])
        )

    monkeypatch.setattr("tool_resource.telemetry._trim_process_heap", record_trim)
    monkeypatch.setattr(
        "tool_resource.telemetry._loss_counts",
        lambda _bpf: {
            "ringbuf_reserve_failures": 0,
            "argv_read_failures": 0,
            "argv_boundary_read_failures": 0,
        },
    )
    monkeypatch.setattr(
        collector,
        "_close_bpf",
        lambda: setattr(collector, "_cleanup_status", "ok"),
    )

    collector.finalize(replay_execution="failed")

    artifact = json.loads(collector.artifact_path.read_text(encoding="utf-8"))
    assert artifact["replay_execution"] == "failed"
    assert artifact["telemetry_quality"] == "ok"
    assert artifact["formal_completeness"] == "complete"
    assert artifact["collection_validity"] == "valid"
    assert trimmed == [("closed", "ok", "valid")]


def test_finalize_records_argv_failure_sites_without_double_counting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _active_collector()
    collector.artifact_path = tmp_path / "clause.json"
    monkeypatch.setattr(
        "tool_resource.telemetry._loss_counts",
        lambda _bpf: {
            "ringbuf_reserve_failures": 0,
            "argv_read_failures": 2,
            "argv_boundary_read_failures": 0,
        },
    )
    monkeypatch.setattr(
        "tool_resource.telemetry._argv_read_failure_sites",
        lambda _bpf: {"missing_bprm_capture": 2},
    )
    monkeypatch.setattr(
        collector,
        "_close_bpf",
        lambda: setattr(collector, "_cleanup_status", "ok"),
    )

    collector.finalize()

    artifact = json.loads(collector.artifact_path.read_text(encoding="utf-8"))
    assert artifact["telemetry_loss_total"]["total"] == 2
    assert artifact["argv_read_failure_sites"] == {"missing_bprm_capture": 2}
    assert "argv read failure sites: missing_bprm_capture=2" in artifact["integrity"][
        "errors"
    ]


def test_finalize_marks_mapping_gaps_partial_without_discarding_valid_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _active_collector()
    collector.artifact_path = tmp_path / "clause.json"
    mapping_error = "call-2: mapping gaps=unmatched_static_clause"
    collector.calls = [
        {
            "telemetry_quality": "ok",
            "eligible_for_kb": True,
            "integrity": {"status": "ok", "errors": []},
        },
        {
            "telemetry_quality": "invalid",
            "eligible_for_kb": False,
            "integrity": {"status": "failed", "errors": [mapping_error]},
        },
    ]
    collector._integrity_errors = [mapping_error]
    monkeypatch.setattr(
        "tool_resource.telemetry._loss_counts",
        lambda _bpf: {
            "ringbuf_reserve_failures": 0,
            "argv_read_failures": 0,
            "argv_boundary_read_failures": 0,
        },
    )
    monkeypatch.setattr(
        collector,
        "_close_bpf",
        lambda: setattr(collector, "_cleanup_status", "ok"),
    )

    collector.finalize()

    artifact = json.loads(collector.artifact_path.read_text(encoding="utf-8"))
    assert artifact["telemetry_quality"] == "ok"
    assert artifact["formal_completeness"] == "partial"
    assert artifact["collection_validity"] == "valid"
    assert artifact["collector"]["health"] == "healthy"
    assert artifact["call_coverage"] == {
        "total_call_count": 2,
        "eligible_call_count": 1,
        "withheld_call_count": 1,
        "eligible_fraction": 0.5,
    }
    assert artifact["integrity"] == {"status": "ok", "errors": []}


def test_finalize_records_disable_state_after_health_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _active_collector()
    collector.artifact_path = tmp_path / "clause.json"
    monkeypatch.setattr(
        "tool_resource.telemetry._loss_counts",
        lambda _bpf: (_ for _ in ()).throw(RuntimeError("counter failed")),
    )
    monkeypatch.setattr(collector, "_close_bpf", lambda: None)

    collector.finalize()

    artifact = json.loads(collector.artifact_path.read_text(encoding="utf-8"))
    assert artifact["collector"]["state_before_close"] == "disabled"
    assert "counter failed" in artifact["collector"]["disabled_reason"]


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
    assert _attach_resource_observations(trace, [summary], source_actions) == []
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["data"]["resource_observation"] == summary
    assert record["data"]["exit_code_agreement"] == {
        "source": 0,
        "replay": 0,
        "available": True,
        "matches": True,
    }


def test_attachment_replace_failure_preserves_authoritative_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    original = (
        json.dumps(
            {
                "type": "action",
                "action_type": "tool_exec",
                "data": {
                    "tool_name": "exec",
                    "tool_call_id": "tc-1",
                    "tool_args": {"command": "echo hi"},
                },
            }
        )
        + "\n"
    )
    trace.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        _attach_resource_observations(
            trace,
            [{"tool_call_id": "tc-1", "command": "echo hi"}],
            [],
        )

    assert trace.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".trace.jsonl.*")) == []


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
    errors = _attach_resource_observations(trace, calls, [])
    assert "duplicate resource observation tool_call_id duplicate" in errors
    assert "duplicate exec action tool_call_id duplicate" in errors
    assert all(
        "resource_observation" not in json.loads(line)["data"]
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
    errors = _attach_resource_observations(
        trace,
        [{"tool_call_id": "tc-1", "command": "echo other"}],
        [],
    )
    assert errors == [
        "exec action tc-1 command does not match resource observation",
        "resource observation tc-1 has no matching exec action",
    ]
    assert "resource_observation" not in json.loads(trace.read_text())["data"]


def test_integrity_error_does_not_replace_recoverable_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tool:
        resource_service_enabled = True

        async def execute(self, **_kwargs: Any) -> str:
            return "/bin/sh: 1: python: not found\n\n\nExit code: 127"

        def finish_resource_call(self) -> None:
            raise ClauseTelemetryIntegrityError("mapping gap")

    class Tools:
        def prepare_call(
            self,
            _name: str,
            params: dict[str, Any],
        ) -> tuple[Tool, dict[str, Any], None]:
            return Tool(), params, None

    monkeypatch.setenv("OPENCLAW_RESOURCE_TIMELINE", "off")
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
    assert result[2] is None


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
        resource_service_enabled = True

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

        def finish_resource_call(self) -> None:
            raise ClauseTelemetryIntegrityError("mapping gap")

    class Hook(AgentHook):
        def __init__(self) -> None:
            self.tool_rows: list[dict[str, Any]] = []

        async def after_iteration(self, context: AgentHookContext) -> None:
            self.tool_rows = [
                message for message in context.messages if message.get("role") == "tool"
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
    assert result.stop_reason == "max_iterations"
    assert hook.tool_rows[0]["content"] == (
        "/bin/sh: 1: python: not found\n\n\nExit code: 127"
    )
