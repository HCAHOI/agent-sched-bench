"""Bridge tests: real mvdan parsing + synthetic Stage-2 exec-image records."""

from __future__ import annotations

import pytest

from tool_resource.clause_bridge import (
    ExecImageRecord,
    FailedExecAttempt,
    ShellCommandLookupFailure,
    bridge_command,
)
from tool_resource.runtime_kb import CPU_HEAVY_TARGET, ClauseObservation, ClauseResourceKB

_MS = 1_000_000
_S = 1_000_000_000
_WINDOW_NS = 500_000_000
_ALIGN_BIN_NS = 20_000_000


def _cpu_windows(t_exec: int, t_end: int, cores: float | None) -> tuple | None:
    if cores is None:
        return ()  # profile present, no CPU (idle/short) -> insufficient, not missing
    out = []
    w0, w1 = t_exec // _WINDOW_NS, (t_end - 1) // _WINDOW_NS
    for widx in range(w0, w1 + 1):
        lo = max(widx * _WINDOW_NS, t_exec)
        hi = min((widx + 1) * _WINDOW_NS, t_end)
        out.append((widx, int(cores * (hi - lo))))
    return tuple(out)


def _rss_bins(t_exec: int, t_end: int, rss_mb: float | None, mm: int) -> tuple | None:
    if rss_mb is None:
        return ()
    b0, b1 = t_exec // _ALIGN_BIN_NS, (t_end - 1) // _ALIGN_BIN_NS
    return tuple((b, mm, rss_mb) for b in range(b0, b1 + 1))


def _img(
    pid: int,
    seq: int,
    bin_: str,
    t_exec: int,
    t_end: int,
    *,
    terminal: bool,
    cores: float | None = None,
    rss_mb: float | None = None,
    mm: int | None = None,
    cpu_ns: int = 0,
    signal: int | None = None,
    status: int | None = 0,
    argv: tuple[str, ...] | None = None,
    cpu_profile: object = "auto",
    rss_profile: object = "auto",
    quota: float = 8.0,
    has_causal_end: bool = True,
    disk_read: int | None = 0,
    disk_write: int | None = 0,
    disk_cancelled: int | None = 0,
    disk_io_reason: str = "ok",
) -> ExecImageRecord:
    cpu_windows = (
        _cpu_windows(t_exec, t_end, cores) if cpu_profile == "auto" else cpu_profile
    )
    rss_bins = (
        _rss_bins(t_exec, t_end, rss_mb, mm if mm is not None else pid)
        if rss_profile == "auto"
        else rss_profile
    )
    return ExecImageRecord(
        host_pid=pid,
        exec_seq=seq,
        t_exec_ns=t_exec,
        t_end_ns=t_end,
        bin=bin_,
        argv=argv if argv is not None else (bin_,),
        terminal=terminal,
        cpu_windows=cpu_windows,
        rss_bins=rss_bins,
        peak_cpu_cores=cores,
        sampled_peak_rss_mb=rss_mb,
        disk_read_bytes_total=disk_read,
        disk_write_bytes_total=disk_write,
        disk_cancelled_write_bytes_total=disk_cancelled,
        disk_io_reason=disk_io_reason,
        cpu_ns_cumulative=cpu_ns,
        exit_signal=signal,
        normal_exit_status=None if signal else status,
        has_causal_end=has_causal_end,
        provenance={
            "quota_cores": quota,
            "disk_io": {
                "source": "linux_task_io_accounting",
                "exact_endpoint_tids": [pid],
            },
        },
    )


def _fit(bin_: str, *, cpu: float | None = 0.5, rss: float | None = 10.0):
    return ClauseObservation(
        repo="pub", bin=bin_, argv=(bin_,), ts_start=0.0, ts_end=0.001,
        latency_ms=10.0, peak_cpu_cores=cpu, sampled_peak_rss_mb=rss,
        cpu_ns_cumulative=0,
    )


def _lookup_failure(
    exit_code: int = 127,
    command: str = "cd /testbed && python -m pytest",
) -> ShellCommandLookupFailure:
    diagnostic = "/bin/sh: 1: python: not found"
    return ShellCommandLookupFailure(
        executable_head="python",
        command=command,
        source_tool_call_id="source-1",
        replay_tool_call_id="replay-1",
        source_exit_code=exit_code,
        replay_exit_code=exit_code,
        source_diagnostic=diagnostic,
        replay_diagnostic=diagnostic,
        source_channel="source_tool_result",
        replay_channel="raw_stderr" if exit_code else "tool_result",
        parser="anchored_shell_command_not_found_v1",
        exit_code_semantics=(
            "direct_command_not_found_127"
            if exit_code == 127
            else "nonfinal_pipeline_masked_0"
        ),
    )


# --------------------------------------------------------------------------
# Identity boundary (W/P/M/K/B) with profile-based aggregation
# --------------------------------------------------------------------------


def test_W_exec_chain_becomes_one_clause_headed_by_env() -> None:
    images = [
        _img(101, 0, "env", 0, 1000, terminal=False, disk_read=10,
             argv=("env", "nice", "-n", "0", "workload")),
        _img(101, 1, "nice", 1000, 2000, terminal=False, disk_write=20,
             argv=("nice", "-n", "0", "workload")),
        _img(101, 2, "workload", 2000, 1300 * _MS, terminal=True, cores=1.0,
             rss_mb=40.0, cpu_ns=1300 * _MS, disk_read=30, disk_write=40,
             disk_cancelled=5, argv=("workload", "cpu-threads", "1")),
    ]
    result = bridge_command(
        "r1", "env nice -n 0 workload cpu-threads 1 1.3", images,
        entry_pid=100, fork_parent={101: 100},
    )
    assert len(result.bridged) == 1
    assert not result.coverage_gaps
    obs = result.observations[0]
    assert obs.bin == "env"
    assert "workload" in obs.argv
    assert obs.peak_cpu_cores == pytest.approx(1.0, abs=0.05)
    assert result.bridged[0].owned_exec_images == ((101, 0), (101, 1), (101, 2))
    assert (
        result.bridged[0].disk_read_bytes_total,
        result.bridged[0].disk_write_bytes_total,
        result.bridged[0].disk_cancelled_write_bytes_total,
    ) == (40, 60, 5)
    assert result.bridged[0].availability["disk_io"] == "ok"
    assert result.bridged[0].provenance["per_image_diagnostics"][0][
        "disk_io_provenance"
    ] == {
        "source": "linux_task_io_accounting",
        "exact_endpoint_tids": [101],
    }
    assert not hasattr(obs, "disk_read_bytes_total")


def test_P_pipeline_two_separate_clauses_no_cross_leak() -> None:
    images = [
        _img(101, 0, "workload", 0, 1500 * _MS, terminal=True, cores=2.0, rss_mb=50.0,
             argv=("workload", "cpu-threads", "2", "1.5")),
        _img(102, 0, "workload", 10, 1500 * _MS, terminal=True, cores=2.0, rss_mb=50.0,
             argv=("workload", "cpu-threads", "2", "1.5")),
    ]
    cmd = "workload cpu-threads 2 1.5 | workload cpu-threads 2 1.5"
    result = bridge_command(
        "r1", cmd, images, entry_pid=100, fork_parent={101: 100, 102: 100}
    )
    assert len(result.bridged) == 2
    peaks = sorted(o.peak_cpu_cores for o in result.observations)
    assert peaks == pytest.approx([2.0, 2.0], abs=0.05)  # no 4.0 cross-leak

    kb = ClauseResourceKB.fit_public([_fit("workload", cpu=0.5)])
    for o in result.observations:
        kb.observe_completed_clause(o)
    cpu = kb.predict_command("r1", cmd, 100.0).targets[CPU_HEAVY_TARGET]
    assert cpu.flag is False  # composer does not sum concurrent clauses
    assert "does not sum concurrent" in cpu.note


def test_M_memory_flag_from_sampled_peak_rss_not_hiwater() -> None:
    images = [
        _img(101, 0, "python3", 0, 1200 * _MS, terminal=True,
             cores=None, rss_mb=600.0, argv=("python3", "-c", "x")),
    ]
    result = bridge_command(
        "r1", "python3 -c 'x=1'", images, entry_pid=100, fork_parent={101: 100}
    )
    obs = result.observations[0]
    assert obs.sampled_peak_rss_mb == pytest.approx(600.0)
    assert not hasattr(obs, "hiwater_pages")
    assert result.bridged[0].availability["cpu"].startswith("unknown")
    assert result.bridged[0].availability["memory"] == "ok"


def test_disk_io_unavailable_does_not_change_kb_observation() -> None:
    result = bridge_command(
        "r1",
        "python3 -c 'x=1'",
        [
            _img(
                101,
                0,
                "python3",
                0,
                1200 * _MS,
                terminal=True,
                cores=0.5,
                rss_mb=10.0,
                argv=("python3", "-c", "x=1"),
                disk_read=None,
                disk_write=None,
                disk_cancelled=None,
                disk_io_reason="missing_exact_tid_io_endpoint",
            )
        ],
        entry_pid=100,
        fork_parent={101: 100},
    )
    clause = result.bridged[0]
    assert clause.disk_read_bytes_total is None
    assert clause.availability["disk_io"] == (
        "unknown:owned_image_unavailable:missing_exact_tid_io_endpoint"
    )
    assert clause.observation.peak_cpu_cores == pytest.approx(0.5)
    assert clause.observation.sampled_peak_rss_mb == pytest.approx(10.0)


def test_K_killed_descendant_preserves_peak_and_signal() -> None:
    images = [
        _img(101, 0, "timeout", 0, 1400 * _MS, terminal=True, cores=None,
             argv=("timeout", "-s", "KILL", "1.3", "workload")),
        _img(102, 0, "workload", 5 * _MS, 1300 * _MS, terminal=True, cores=1.0,
             rss_mb=40.0, signal=9, argv=("workload", "cpu-threads", "1", "5")),
    ]
    result = bridge_command(
        "r1", "timeout -s KILL 1.3 workload cpu-threads 1 5", images,
        entry_pid=100, fork_parent={101: 100, 102: 101},
    )
    assert len(result.bridged) == 1
    obs = result.observations[0]
    assert obs.bin == "timeout"
    assert obs.peak_cpu_cores == pytest.approx(1.0, abs=0.1)  # from owned descendant
    assert result.bridged[0].provenance["exit_signal"] == 9
    assert sorted(result.bridged[0].owned_pids) == [101, 102]


def test_B_background_descendant_maps_without_leakage() -> None:
    images = [
        _img(201, 0, "workload", 100, 1300 * _MS, terminal=True, cores=1.0, rss_mb=40.0,
             argv=("workload", "cpu-threads", "1", "1.3")),
        _img(202, 0, "sleep", 200, 100 * _MS, terminal=True, cores=0.0, rss_mb=1.0,
             argv=("sleep", "0.1")),
        _img(203, 0, "sleep", 300, 1300 * _MS, terminal=True, cores=0.0, rss_mb=1.0,
             argv=("sleep", "1.3")),
    ]
    cmd = "( workload cpu-threads 1 1.3 & sleep 0.1 ); sleep 1.3"
    result = bridge_command(
        "r1", cmd, images, entry_pid=100,
        fork_parent={200: 100, 201: 200, 202: 200, 203: 100},
    )
    assert len(result.bridged) == 3
    assert not result.coverage_gaps
    by_bin: dict = {}
    for o in result.observations:
        by_bin.setdefault(o.bin, []).append(o)
    assert by_bin["workload"][0].peak_cpu_cores == pytest.approx(1.0, abs=0.1)
    # sleep 0.1 is <1 s (CPU unavailable); sleep 1.3 is ~0 cores; neither carries
    # the burner's ~1 core -> no cross-clause leakage.
    assert all(
        o.peak_cpu_cores is None or o.peak_cpu_cores < 0.5 for o in by_bin["sleep"]
    )


def test_failed_exec_exactly_resolves_one_static_clause_without_observation() -> None:
    failed = FailedExecAttempt(
        host_pid=101,
        exec_seq=1,
        ts_ns=10,
        argv=("python", "-m", "pytest"),
        errno=2,
    )
    result = bridge_command(
        "r1",
        "cd /testbed && python -m pytest",
        [],
        failed_exec_attempts=[failed],
        entry_pid=100,
        fork_parent={101: 100},
    )
    assert result.observations == []
    assert result.coverage_gaps == []
    assert len(result.no_runtime_exec) == 1
    assert result.no_runtime_exec[0].attempts == (failed,)
    assert result.no_runtime_exec[0].availability == {
        "latency": "unknown:no_runtime_exec",
        "cpu": "unknown:no_runtime_exec",
        "memory": "unknown:no_runtime_exec",
        "disk_io": "unknown:no_runtime_exec",
    }


def test_failed_exec_does_not_resolve_ambiguous_repeated_static_clauses() -> None:
    result = bridge_command(
        "r1",
        "missing; missing",
        [],
        failed_exec_attempts=[
            FailedExecAttempt(101, 1, 10, ("missing",), 2)
        ],
        entry_pid=100,
        fork_parent={101: 100},
    )
    assert result.no_runtime_exec == []
    assert [gap.kind for gap in result.coverage_gaps] == [
        "unmatched_static_clause",
        "unmatched_static_clause",
    ]


def test_shell_lookup_failure_resolves_one_exact_static_head_without_observation() -> None:
    evidence = _lookup_failure()
    result = bridge_command(
        "r1",
        evidence.command,
        [],
        command_lookup_failure=evidence,
        entry_pid=100,
        fork_parent={},
    )
    assert result.observations == []
    assert result.coverage_gaps == []
    assert result.no_runtime_exec[0].attempts == ()
    assert result.no_runtime_exec[0].command_lookup_failure is evidence
    assert (
        result.no_runtime_exec[0].mapping_evidence
        == "shell_command_lookup_failure_exact_head"
    )


def test_pipeline_masked_lookup_failure_maps_only_python() -> None:
    command = "python -m pytest 2>&1 | tail -40"
    evidence = _lookup_failure(0, command)
    result = bridge_command(
        "r1",
        command,
        [
            _img(
                102,
                0,
                "tail",
                0,
                10 * _MS,
                terminal=True,
                argv=("tail", "-40"),
            )
        ],
        command_lookup_failure=evidence,
        entry_pid=100,
        fork_parent={102: 100},
    )
    assert [item.bin for item in result.no_runtime_exec] == ["python"]
    assert [item.observation.bin for item in result.bridged] == ["tail"]
    assert result.coverage_gaps == []


def test_shell_lookup_failure_does_not_choose_between_repeated_static_heads() -> None:
    result = bridge_command(
        "r1",
        "python -V; python --version",
        [],
        command_lookup_failure=_lookup_failure(),
        entry_pid=100,
        fork_parent={},
    )
    assert result.no_runtime_exec == []
    assert [gap.kind for gap in result.coverage_gaps] == [
        "unmatched_static_clause",
        "unmatched_static_clause",
    ]


def test_exit_zero_lookup_failure_requires_nonfinal_pipeline_clause() -> None:
    command = "python || true"
    result = bridge_command(
        "r1",
        command,
        [],
        command_lookup_failure=_lookup_failure(0, command),
        entry_pid=100,
        fork_parent={},
    )
    assert result.no_runtime_exec == []
    assert [gap.kind for gap in result.coverage_gaps] == [
        "unmatched_static_clause"
    ]


def test_bridge_rejects_internally_inconsistent_lookup_evidence() -> None:
    evidence = ShellCommandLookupFailure(
        **{
            **_lookup_failure().__dict__,
            "replay_exit_code": 0,
            "replay_diagnostic": "python not found",
            "replay_tool_call_id": "",
        }
    )
    result = bridge_command(
        "r1",
        evidence.command,
        [],
        command_lookup_failure=evidence,
        entry_pid=100,
        fork_parent={},
    )
    assert result.no_runtime_exec == []
    assert [gap.kind for gap in result.coverage_gaps] == [
        "unmatched_static_clause"
    ]


@pytest.mark.parametrize(
    ("command", "status", "operator"),
    [
        ("left && middle && right", 1, "&&"),
        ("left || middle || right", 0, "||"),
    ],
)
def test_shell_control_chain_resolves_only_proven_short_circuits(
    command: str,
    status: int,
    operator: str,
) -> None:
    result = bridge_command(
        "r1",
        command,
        [_img(101, 0, "left", 0, 10, terminal=True, status=status)],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert not result.coverage_gaps
    assert [item.bin for item in result.no_runtime_exec] == ["middle", "right"]
    assert all(
        item.mapping_evidence == "shell_control_short_circuit"
        for item in result.no_runtime_exec
    )
    assert {
        (
            item.control_short_circuit["controller_clause_index"],
            item.control_short_circuit["controller_normal_exit_status"],
            item.control_short_circuit["operator"],
        )
        for item in result.no_runtime_exec
    } == {(0, status, operator)}


@pytest.mark.parametrize(
    ("command", "image"),
    [
        (
            "left && right",
            _img(101, 0, "left", 0, 10, terminal=True, status=None),
        ),
        (
            "left && right",
            _img(101, 0, "left", 0, 10, terminal=True, signal=9),
        ),
        (
            "left; right",
            _img(101, 0, "left", 0, 10, terminal=True, status=1),
        ),
        (
            "left | middle && right",
            _img(101, 0, "left", 0, 10, terminal=True, status=1),
        ),
    ],
)
def test_shell_control_without_exact_normal_evidence_stays_fatal(
    command: str,
    image: ExecImageRecord,
) -> None:
    result = bridge_command(
        "r1",
        command,
        [image],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert result.no_runtime_exec == []
    assert any(gap.kind == "unmatched_static_clause" for gap in result.coverage_gaps)


@pytest.mark.parametrize(
    ("command", "status", "operator"),
    [
        ("false && (b | c)", 1, "&&"),
        ("true || (b | c)", 0, "||"),
    ],
)
def test_outer_control_skips_every_rhs_pipeline_member(
    command: str,
    status: int,
    operator: str,
) -> None:
    result = bridge_command(
        "r1",
        command,
        [_img(101, 0, command.split()[0], 0, 10, terminal=True, status=status)],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert not result.coverage_gaps
    assert [item.observation.bin for item in result.bridged] == [
        command.split()[0]
    ]
    assert [item.bin for item in result.no_runtime_exec] == ["b", "c"]
    for index, item in enumerate(result.no_runtime_exec, start=1):
        assert item.control_short_circuit == {
            "parser": "mvdan.cc/sh/v3",
            "control_edge_id": 0,
            "control_edge_path": [0],
            "operator": operator,
            "controller_clause_index": 0,
            "controller_bin": command.split()[0],
            "controller_pid": 101,
            "controller_exec_seq": 0,
            "controller_normal_exit_status": status,
            "controlled_clause_index": index,
            "controlled_rhs_clause_indices": [1, 2],
            "controlled_rhs_executable_clause_indices": [1, 2],
            "controlled_rhs_subtree": {
                "kind": "unsupported",
                "index": -1,
                "clause_indices": [1, 2],
                "span": (9, 16) if operator == "&&" else (8, 15),
                "negated": False,
                "contains_pipeline": True,
                "contains_subshell": True,
            },
            "source_replay_fidelity": "exact_tool_result_and_exit_code",
        }


def test_outer_control_skips_nested_rhs_subtree_deterministically() -> None:
    result = bridge_command(
        "r1",
        "false && (b && (c | d))",
        [_img(101, 0, "false", 0, 10, terminal=True, status=1)],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert not result.coverage_gaps
    assert [item.bin for item in result.no_runtime_exec] == ["b", "c", "d"]
    assert {
        tuple(item.control_short_circuit["controlled_rhs_clause_indices"])
        for item in result.no_runtime_exec
    } == {(1, 2, 3)}
    assert {
        item.control_short_circuit["controlled_clause_index"]
        for item in result.no_runtime_exec
    } == {1, 2, 3}


def test_outer_control_rejects_nonunique_controller_mapping() -> None:
    result = bridge_command(
        "r1",
        "left x && (b | c)",
        [
            _img(
                101,
                0,
                "left",
                0,
                10,
                terminal=True,
                status=1,
                argv=("left", "x"),
            ),
            _img(
                102,
                0,
                "left",
                0,
                10,
                terminal=True,
                status=1,
                argv=("left", "x"),
            ),
        ],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100, 102: 100},
    )

    assert result.no_runtime_exec == []
    assert any(gap.kind == "unmatched_static_clause" for gap in result.coverage_gaps)


@pytest.mark.parametrize(
    ("command", "images", "fork_parent"),
    [
        (
            "a | b && c",
            [
                _img(101, 0, "a", 0, 10, terminal=True, status=0),
                _img(102, 0, "b", 0, 10, terminal=True, status=1),
            ],
            {101: 100, 102: 100},
        ),
        (
            "(a) && c",
            [_img(101, 0, "a", 0, 10, terminal=True, status=1)],
            {101: 100},
        ),
    ],
)
def test_outer_control_rejects_lhs_pipeline_or_subshell_status(
    command: str,
    images: list[ExecImageRecord],
    fork_parent: dict[int, int],
) -> None:
    result = bridge_command(
        "r1",
        command,
        images,
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent=fork_parent,
    )

    assert result.no_runtime_exec == []
    assert any(gap.kind == "unmatched_static_clause" for gap in result.coverage_gaps)


def test_outer_control_rejects_partial_rhs_runtime_evidence() -> None:
    result = bridge_command(
        "r1",
        "false && (b | c)",
        [
            _img(101, 0, "false", 0, 10, terminal=True, status=1),
            _img(102, 0, "b", 0, 10, terminal=True, status=0),
        ],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100, 102: 100},
    )

    assert result.no_runtime_exec == []
    assert any(
        gap.kind == "control_flow_contradiction"
        for gap in result.coverage_gaps
    )


def test_outer_control_resolves_the_fixed_image_smoke_pipeline() -> None:
    command = (
        "cd /testbed && git stash && cd src/azure-cli-core && "
        "/opt/conda/envs/testbed/bin/python -m pytest "
        "azure/cli/core/tests/test_help.py::HelpTest::test_help_extra_missing_params "
        "azure/cli/core/tests/test_vcr_security.py::Test_vcr_security::"
        "test_deployment_name_scrub -v 2>&1 | tail -20"
    )
    result = bridge_command(
        "r1",
        command,
        [
            _img(
                101,
                0,
                "git",
                0,
                10,
                terminal=True,
                status=128,
                argv=("git", "stash"),
            )
        ],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert not result.coverage_gaps
    assert [item.bin for item in result.no_runtime_exec] == ["python", "tail"]
    assert {
        tuple(item.control_short_circuit["controlled_rhs_clause_indices"])
        for item in result.no_runtime_exec
    } == {(3, 4)}
    assert {
        tuple(item.control_short_circuit["control_edge_path"])
        for item in result.no_runtime_exec
    } == {(0, 1, 2)}


def test_shell_control_mapping_ambiguity_stays_fatal() -> None:
    result = bridge_command(
        "r1",
        "left x && left y && right",
        [
            _img(101, 0, "left", 0, 10, terminal=True, status=1),
            _img(102, 0, "left", 0, 10, terminal=True, status=1),
        ],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100, 102: 100},
    )

    assert result.no_runtime_exec == []
    assert any(gap.kind == "ambiguous" for gap in result.coverage_gaps)


@pytest.mark.parametrize(
    ("command", "status"),
    [("! left && right", 1), ("! left || right", 0)],
)
def test_negated_shell_control_stays_fatal(
    command: str,
    status: int,
) -> None:
    result = bridge_command(
        "r1",
        command,
        [_img(101, 0, "left", 0, 10, terminal=True, status=status)],
        allow_control_short_circuit=True,
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert result.no_runtime_exec == []
    assert any(gap.kind == "unmatched_static_clause" for gap in result.coverage_gaps)


def test_shell_control_requires_source_replay_fidelity_gate() -> None:
    result = bridge_command(
        "r1",
        "left && right",
        [_img(101, 0, "left", 0, 10, terminal=True, status=1)],
        entry_pid=100,
        fork_parent={101: 100},
    )

    assert result.no_runtime_exec == []
    assert any(gap.kind == "unmatched_static_clause" for gap in result.coverage_gaps)


# --------------------------------------------------------------------------
# BLOCKER 1 — time-aligned aggregation, never scalar max
# --------------------------------------------------------------------------


def test_concurrent_descendants_cpu_sums_to_three_cores() -> None:
    # one static clause (a wrapper) owns two concurrent 1.5-core descendants
    span_end = 2000 * _MS
    images = [
        _img(101, 0, "runner", 0, span_end, terminal=True, cores=None,
             argv=("runner", "two")),  # wrapper itself no CPU
        _img(102, 0, "worker", 0, span_end, terminal=True, cores=1.5, rss_mb=10.0),
        _img(103, 0, "worker", 0, span_end, terminal=True, cores=1.5, rss_mb=10.0),
    ]
    result = bridge_command(
        "r1", "runner two", images, entry_pid=100,
        fork_parent={101: 100, 102: 101, 103: 101},
    )
    assert len(result.bridged) == 1
    obs = result.observations[0]
    assert obs.peak_cpu_cores == pytest.approx(3.0, abs=0.1)  # summed, not max=1.5
    kb = ClauseResourceKB.fit_public([_fit("runner", cpu=0.5)])
    kb.observe_completed_clause(obs)
    # predict_command absorbs the causally-prior observation before predicting
    assert kb.predict_command("r1", "runner two", 100.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0].flag is True


def test_concurrent_descendants_rss_sums_distinct_mm() -> None:
    span_end = 1200 * _MS
    images = [
        _img(101, 0, "runner", 0, span_end, terminal=True, cores=0.1,
             rss_profile=()),  # wrapper trivial rss
        _img(102, 0, "worker", 0, span_end, terminal=True, cores=0.1, rss_mb=300.0, mm=1),
        _img(103, 0, "worker", 0, span_end, terminal=True, cores=0.1, rss_mb=300.0, mm=2),
    ]
    result = bridge_command(
        "r1", "runner two", images, entry_pid=100,
        fork_parent={101: 100, 102: 101, 103: 101},
    )
    obs = result.observations[0]
    assert obs.sampled_peak_rss_mb == pytest.approx(600.0, abs=1.0)  # 300 + 300


def test_sequential_images_rss_is_not_summed_across_time() -> None:
    # two owned images each peak 300 MB at DIFFERENT times -> clause peak ~300
    images = [
        _img(101, 0, "seq", 0, 1200 * _MS, terminal=False,
             rss_profile=((0, 1, 300.0), (1, 1, 300.0))),  # bins 0-1, mm=1
        _img(101, 1, "seq", 1200 * _MS, 2400 * _MS, terminal=True, cores=0.1,
             rss_profile=((100, 2, 300.0), (101, 2, 300.0))),  # bins 100-101, mm=2
    ]
    result = bridge_command(
        "r1", "seq", images, entry_pid=100, fork_parent={101: 100}
    )
    obs = result.observations[0]
    assert obs.sampled_peak_rss_mb == pytest.approx(300.0, abs=1.0)  # not 600


def test_missing_or_inconsistent_quota_makes_cpu_unavailable() -> None:
    # missing quota (<=0) -> CPU unavailable, no inf fallback
    missing = [
        _img(101, 0, "prog", 0, 2000 * _MS, terminal=True, cores=3.0, quota=0.0,
             argv=("prog",)),
    ]
    r1 = bridge_command("r1", "prog", missing, entry_pid=100, fork_parent={101: 100})
    assert r1.observations[0].peak_cpu_cores is None
    assert r1.bridged[0].availability["cpu"] == "unknown:missing_or_inconsistent_quota"

    # conflicting quotas across owned images -> unavailable
    conflict = [
        _img(101, 0, "runner", 0, 2000 * _MS, terminal=True, cores=None, quota=8.0,
             argv=("runner",)),
        _img(102, 0, "worker", 0, 2000 * _MS, terminal=True, cores=1.5, quota=4.0),
    ]
    r2 = bridge_command(
        "r1", "runner", conflict, entry_pid=100, fork_parent={101: 100, 102: 101}
    )
    assert r2.observations[0].peak_cpu_cores is None
    assert r2.bridged[0].availability["cpu"] == "unknown:missing_or_inconsistent_quota"


def test_missing_profile_yields_unavailable_not_scalar_max() -> None:
    images = [
        _img(101, 0, "runner", 0, 2000 * _MS, terminal=True, cores=None,
             cpu_profile=None, rss_profile=None),
        _img(102, 0, "worker", 0, 2000 * _MS, terminal=True, cores=1.5, rss_mb=300.0,
             cpu_profile=None, rss_profile=None),
    ]
    result = bridge_command(
        "r1", "runner", images, entry_pid=100,
        fork_parent={101: 100, 102: 101},
    )
    obs = result.observations[0]
    assert obs.peak_cpu_cores is None
    assert obs.sampled_peak_rss_mb is None
    assert result.bridged[0].availability["cpu"] == "unknown:missing_cpu_profile"
    assert result.bridged[0].availability["memory"] == "unknown:missing_rss_profile"


# --------------------------------------------------------------------------
# BLOCKER 2 — evidence-prioritized, ambiguity-preserving matching
# --------------------------------------------------------------------------


def test_pipeline_maps_by_argv_not_timestamp() -> None:
    # b.py chain has the EARLIER t_exec; exact-argv must still map each correctly
    images = [
        _img(201, 0, "python", 500 * _MS, 1500 * _MS, terminal=True, cores=1.0,
             argv=("python", "a.py")),
        _img(202, 0, "python", 0, 1500 * _MS, terminal=True, cores=3.0,
             argv=("python", "b.py")),  # earlier, heavier
    ]
    result = bridge_command(
        "r1", "python a.py | python b.py", images, entry_pid=100,
        fork_parent={201: 100, 202: 100},
    )
    by_argv = {o.argv: o for o in result.observations}
    assert by_argv[("python", "a.py")].peak_cpu_cores == pytest.approx(1.0, abs=0.1)
    assert by_argv[("python", "b.py")].peak_cpu_cores == pytest.approx(3.0, abs=0.1)


def test_repeated_same_bin_distinct_args_reversed_order() -> None:
    images = [
        _img(201, 0, "grep", 0, 1200 * _MS, terminal=True, cores=1.0,
             argv=("grep", "-r", "z")),  # earlier in runtime
        _img(202, 0, "grep", 400 * _MS, 1600 * _MS, terminal=True, cores=2.5,
             argv=("grep", "-r", "a")),
    ]
    result = bridge_command(
        "r1", "grep -r a && grep -r z", images, entry_pid=100,
        fork_parent={201: 100, 202: 100},
    )
    by_argv = {o.argv: o for o in result.observations}
    assert by_argv[("grep", "-r", "a")].peak_cpu_cores == pytest.approx(2.5, abs=0.1)
    assert by_argv[("grep", "-r", "z")].peak_cpu_cores == pytest.approx(1.0, abs=0.1)


def test_genuinely_ambiguous_pair_yields_gaps_and_no_observations() -> None:
    # two DISTINCT static clauses (foo --x / foo --y) but runtime execs carry no
    # distinguishing argv -> only bin-level ties -> ambiguous, no observations
    images = [
        _img(201, 0, "foo", 0, 1200 * _MS, terminal=True, cores=1.0, argv=("foo",)),
        _img(202, 0, "foo", 10, 1200 * _MS, terminal=True, cores=3.0, argv=("foo",)),
    ]
    result = bridge_command(
        "r1", "foo --x | foo --y", images, entry_pid=100,
        fork_parent={201: 100, 202: 100},
    )
    assert result.observations == []
    assert {g.kind for g in result.coverage_gaps} == {"ambiguous"}
    assert len(result.coverage_gaps) == 2


def test_identical_repeated_clauses_map_interchangeably() -> None:
    # identical static identities may map to either chain; both become observations
    images = [
        _img(201, 0, "make", 0, 1200 * _MS, terminal=True, cores=2.5, argv=("make",)),
        _img(202, 0, "make", 10, 1200 * _MS, terminal=True, cores=2.5, argv=("make",)),
    ]
    result = bridge_command(
        "r1", "make | make", images, entry_pid=100,
        fork_parent={201: 100, 202: 100},
    )
    assert len(result.bridged) == 2
    assert not result.coverage_gaps
    assert all(o.bin == "make" for o in result.observations)
    assert all(
        bc.mapping_evidence == "interchangeable_identical" for bc in result.bridged
    )


# --------------------------------------------------------------------------
# Coverage-gap behavior
# --------------------------------------------------------------------------


def test_coverage_gaps_builtins_and_unmatched() -> None:
    images = [
        _img(101, 0, "cd_is_never_execed", 0, 1, terminal=True, cores=0.1),
    ]
    result = bridge_command(
        "r1", "cd /x && realbin --flag", images,
        entry_pid=100, fork_parent={101: 100},
    )
    assert "cd" in result.unobserved_builtins
    kinds = {g.kind for g in result.coverage_gaps}
    assert "unmatched_static_clause" in kinds
    assert "unmatched_exec_image" in kinds
    assert not result.observations


def test_insufficient_coverage_isolated_per_target() -> None:
    images = [
        _img(101, 0, "prog", 0, 1200 * _MS, terminal=True,
             cores=None, rss_mb=None, argv=("prog",)),
    ]
    result = bridge_command(
        "r1", "prog", images, entry_pid=100, fork_parent={101: 100}
    )
    avail = result.bridged[0].availability
    assert avail["latency"] == "ok"
    assert avail["cpu"].startswith("unknown")
    assert avail["memory"].startswith("unknown")
    obs = result.observations[0]
    assert obs.latency_ms is not None
    assert obs.peak_cpu_cores is None
    assert obs.sampled_peak_rss_mb is None


# --------------------------------------------------------------------------
# Fail-closed regressions (B1..B6) and mapping/math regressions (C1, C3)
# --------------------------------------------------------------------------


def test_B1_no_causal_end_withholds_observation() -> None:
    images = [
        _img(101, 0, "prog", 0, 2000 * _MS, terminal=True, cores=3.0, rss_mb=600.0,
             argv=("prog",), has_causal_end=False),  # never really exited
    ]
    r = bridge_command("r1", "prog", images, entry_pid=100, fork_parent={101: 100})
    assert r.observations == []
    assert {g.kind for g in r.coverage_gaps} == {"no_causal_end"}


def test_B1b_any_owned_image_without_causal_end_withholds() -> None:
    # Terminal root image HAS a real exit (latest-ending, has_causal_end True),
    # but a forked descendant it owns never exited (has_causal_end False, earlier
    # t_end). Withhold on ANY owned image lacking a causal end, not just latest.
    images = [
        _img(101, 0, "prog", 0, 2000 * _MS, terminal=True, cores=3.0, rss_mb=600.0,
             argv=("prog",)),  # real exit, latest-ending
        _img(102, 0, "child", 100, 1000 * _MS, terminal=True, cores=1.0,
             rss_mb=50.0, argv=("child",), has_causal_end=False),  # never exited
    ]
    r = bridge_command("r1", "prog", images, entry_pid=100,
                       fork_parent={101: 100, 102: 101})
    assert r.observations == []
    assert {g.kind for g in r.coverage_gaps} == {"no_causal_end"}


def test_B2_parse_failed_withholds_all_observations() -> None:
    images = [_img(101, 0, "echo", 0, 1200 * _MS, terminal=True, cores=1.0,
                   argv=("echo", "ok"))]
    # unbalanced paren -> mvdan parse_failed
    r = bridge_command("r1", "echo ok )", images, entry_pid=100,
                       fork_parent={101: 100})
    assert r.observations == []
    assert any(g.kind == "parse_failed" for g in r.coverage_gaps)


def test_B3_nonzero_loss_withholds_all_observations() -> None:
    images = [_img(101, 0, "prog", 0, 2000 * _MS, terminal=True, cores=3.0,
                   rss_mb=600.0, argv=("prog",))]
    r = bridge_command("r1", "prog", images, entry_pid=100,
                       fork_parent={101: 100}, loss_count=1)
    assert r.observations == []
    assert any(g.kind == "nonzero_loss" for g in r.coverage_gaps)


def test_B4_B6_invalid_or_nonfinite_profile_is_unavailable() -> None:
    images = [
        _img(101, 0, "prog", 0, 2000 * _MS, terminal=True, argv=("prog",),
             cpu_profile=((0, -5),),  # negative cpu_ns -> invalid
             rss_profile=((0, 1, float("inf")), (1, 1, 300.0))),  # non-finite rss
    ]
    r = bridge_command("r1", "prog", images, entry_pid=100, fork_parent={101: 100})
    obs = r.observations[0]
    assert obs.peak_cpu_cores is None
    assert obs.sampled_peak_rss_mb is None
    assert r.bridged[0].availability["cpu"] == "unknown:invalid_cpu_profile"
    assert r.bridged[0].availability["memory"] == "unknown:invalid_rss_profile"


def test_B5_insufficient_rss_samples_is_unavailable() -> None:
    images = [
        _img(101, 0, "prog", 0, 1200 * _MS, terminal=True, argv=("prog",),
             cores=None, rss_profile=((5, 1, 600.0),)),  # a single rss sample
    ]
    r = bridge_command("r1", "prog", images, entry_pid=100, fork_parent={101: 100})
    assert r.observations[0].sampled_peak_rss_mb is None
    assert r.bridged[0].availability["memory"] == "unknown:insufficient_rss_samples"


def test_C1_path_valued_arguments_are_not_basenamed() -> None:
    # two distinct commands differing only by an argument PATH must not merge:
    # basenaming args would make both "cat log" and collide their identities.
    a = ExecImageRecord(
        host_pid=201, exec_seq=0, t_exec_ns=0, t_end_ns=1200 * _MS,
        bin="cat", argv=("/bin/cat", "a/log"), terminal=True,
        cpu_windows=_cpu_windows(0, 1200 * _MS, 1.0),
        rss_bins=_rss_bins(0, 1200 * _MS, 10.0, 201), provenance={"quota_cores": 8.0},
    )
    b = ExecImageRecord(
        host_pid=202, exec_seq=0, t_exec_ns=0, t_end_ns=1200 * _MS,
        bin="cat", argv=("/bin/cat", "b/log"), terminal=True,
        cpu_windows=_cpu_windows(0, 1200 * _MS, 3.0),
        rss_bins=_rss_bins(0, 1200 * _MS, 10.0, 202), provenance={"quota_cores": 8.0},
    )
    r = bridge_command("r1", "cat a/log | cat b/log", [a, b], entry_pid=100,
                       fork_parent={201: 100, 202: 100})
    by_argv = {o.argv: o for o in r.observations}
    # head basenamed to "cat"; path arguments preserved and used to disambiguate
    assert ("cat", "a/log") in by_argv and ("cat", "b/log") in by_argv
    assert by_argv[("cat", "a/log")].peak_cpu_cores == pytest.approx(1.0, abs=0.1)
    assert by_argv[("cat", "b/log")].peak_cpu_cores == pytest.approx(3.0, abs=0.1)


def test_C3_nonoverlapping_mm_lifetimes_not_summed() -> None:
    # two distinct mm whose observed lifetimes do not overlap (adjacent bin
    # ranges) must NOT be summed into one figure, even both ~300 MB.
    images = [
        _img(101, 0, "seq", 0, 2400 * _MS, terminal=True, cores=0.1, argv=("seq",),
             rss_profile=(
                 (0, 1, 300.0), (1, 1, 300.0),      # mm 1 alive bins 0-1
                 (100, 2, 300.0), (101, 2, 300.0),  # mm 2 alive bins 100-101
             )),
    ]
    r = bridge_command("r1", "seq", images, entry_pid=100, fork_parent={101: 100})
    assert r.observations[0].sampled_peak_rss_mb == pytest.approx(300.0, abs=1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
