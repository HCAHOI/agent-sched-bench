from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from trace_collect.openclaw_tools import (
    ContainerPacctSession,
)
from trace_collect.simulate_outputs import _split_combined_worker_trace_by_agent
from trace_collect.simulator import (
    LoadedTraceSession,
    PreparedContainer,
    PreparedTraceSession,
    WorkerReplayResult,
    _CONTAINER_PACCT_SESSIONS,
    _enable_replay_container_pacct,
    _exec_tool,
    _pacct_trace_metadata,
    _split_trace_by_agent_with_pacct,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "pacct_v3_sample.acct"


def _prepared_session(
    tmp_path: Path,
    *,
    container_id: str,
    scaffold: str = "cloud_model",
) -> PreparedTraceSession:
    task_output_dir = tmp_path / container_id
    task_output_dir.mkdir()
    loaded = LoadedTraceSession(
        source_trace=tmp_path / "source.jsonl",
        task_source=tmp_path / "tasks.json",
        task_instance_id="task-1",
        source_action_agent_id="source-agent",
        run_instance_id="run-1",
        manifest_index=0,
        scaffold=scaffold,
        metadata={"source_model": "model"},
        summary=None,
        task={},
        actions=[],
        iterations={},
    )
    return PreparedTraceSession(
        loaded=loaded,
        task_output_dir=task_output_dir,
        container=PreparedContainer(
            container_id=container_id,
            container_executable="docker",
            docker_image="image",
            agent=SimpleNamespace(),
        ),
    )


def _write_combined_trace(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "cloud_model",
                        "trace_format_version": 5,
                        "execution_environment": "container",
                        "replay_mode": "cloud_model",
                        "replay_speed": 20.0,
                    }
                ),
                json.dumps({"type": "summary", "agent_id": "run-1"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_exec_bracket_decodes_offset_delta(
    monkeypatch,
) -> None:
    encoded_delta = base64.b64encode(_FIXTURE.read_bytes()).decode("ascii")
    responses = iter(("128\n", encoded_delta))
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=next(responses), stderr="")

    monkeypatch.setattr("trace_collect.openclaw_tools.subprocess.run", fake_run)
    session = ContainerPacctSession("cid", "docker")

    class FakeAgent:
        _container_pacct_session = session

        async def execute(self, _request, *, timeout_s):
            assert timeout_s == 10.0
            return {
                "ok": True,
                "result": "ok",
                "returncode": 0,
                "inner_duration_ms": 1.0,
            }

    _result, _duration_ms, success, metadata = asyncio.run(
        _exec_tool(
            FakeAgent(),
            "exec",
            '{"command":"true","timeout":10}',
            10.0,
        )
    )

    rows = metadata["per_process"]
    assert success is True
    assert len(rows) == 28
    assert all(row["attribution"] == "offset_delta" for row in rows)
    assert all(
        {
            "comm",
            "pid",
            "ppid",
            "utime_s",
            "stime_s",
            "avg_mem_kb",
            "exitcode",
            "attribution",
        }
        == set(row)
        for row in rows
    )
    assert len(calls) == 2
    assert all(call[:7] == ["docker", "exec", "--user", "0", "cid", "python3", "-c"] for call in calls)
    assert "+64" in calls[0][7]
    assert calls[1][-1] == "128"


def test_missing_python3_is_fail_soft_and_structurally_recorded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run(cmd, **_kwargs):
        return subprocess.CompletedProcess(
            cmd,
            127,
            stdout="",
            stderr="exec: python3: not found",
        )

    monkeypatch.setattr("trace_collect.openclaw_tools.subprocess.run", fake_run)
    monkeypatch.setenv("OPENCLAW_PACCT", "1")
    prepared = _prepared_session(tmp_path, container_id="cid", scaffold="openclaw")
    try:
        asyncio.run(_enable_replay_container_pacct(prepared))
        session = _CONTAINER_PACCT_SESSIONS["cid"]
        assert session.unavailable is True
        assert session.unavailable_reason == "python3_unavailable"

        combined_path = tmp_path / "combined.jsonl"
        _write_combined_trace(combined_path)
        _split_trace_by_agent_with_pacct(combined_path, [prepared])
        metadata = json.loads(
            (prepared.task_output_dir / "trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert metadata["replay_mode"] == "cloud_model"
        assert metadata["replay_speed"] == 20.0
        assert metadata["pacct_unavailable"] is True
        assert metadata["pacct_attribution"] == "offset_delta"
        assert metadata["pacct_unavailable_reason"] == "python3_unavailable"
        assert not (prepared.task_output_dir / "run_manifest.json").exists()
    finally:
        _CONTAINER_PACCT_SESSIONS.pop("cid", None)


def test_available_pacct_metadata_is_written_by_real_trace_splitter(
    tmp_path: Path,
) -> None:
    session = ContainerPacctSession("cid-finalize", "docker", overlap_execs=2)
    prepared = _prepared_session(tmp_path, container_id="cid-finalize")
    _CONTAINER_PACCT_SESSIONS["cid-finalize"] = session
    combined_path = tmp_path / "combined.jsonl"
    _write_combined_trace(combined_path)

    try:
        _split_combined_worker_trace_by_agent(
            combined_path=combined_path,
            sessions=[prepared.loaded],
            worker_results=[
                WorkerReplayResult(
                    wave_index=0,
                    worker_index=0,
                    trace_file=str(combined_path),
                    task_stats=[],
                    task_output_dirs={
                        prepared.loaded.run_instance_id: str(
                            prepared.task_output_dir
                        )
                    },
                    pacct_metadata_by_agent={
                        prepared.loaded.run_instance_id: _pacct_trace_metadata(
                            prepared
                        )
                    },
                )
            ],
        )
    finally:
        _CONTAINER_PACCT_SESSIONS.pop("cid-finalize", None)

    metadata = json.loads(
        (prepared.task_output_dir / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert metadata["pacct_unavailable"] is False
    assert metadata["pacct_attribution"] == "offset_delta"
    assert metadata["pacct_overlap_execs"] == 2


def test_overlapping_execs_discard_ambiguous_attribution(
    monkeypatch,
) -> None:
    session = ContainerPacctSession("cid", "docker")
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    class OverlappingAgent:
        _container_pacct_session = session

        async def execute(self, _request, *, timeout_s):
            nonlocal started
            assert timeout_s == 10.0
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()
            return {
                "ok": True,
                "result": "ok",
                "returncode": 0,
                "inner_duration_ms": 1.0,
            }

    monkeypatch.setattr(
        "trace_collect.openclaw_tools.container_pacct_begin",
        lambda _session: 0,
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_tools.container_pacct_finish",
        lambda _session, _offset: [{"attribution": "offset_delta"}],
    )

    async def run_overlap():
        calls = [
            asyncio.create_task(
                _exec_tool(
                    OverlappingAgent(),
                    "exec",
                    '{"command":"true","timeout":10}',
                    10.0,
                )
            )
            for _ in range(2)
        ]
        await both_started.wait()
        release.set()
        return await asyncio.gather(*calls)

    results = asyncio.run(run_overlap())

    assert session.overlap_execs == 2
    for _result, _duration_ms, success, metadata in results:
        assert success is True
        assert "per_process" not in metadata
        assert metadata["per_process_unavailable_reason"] == "overlapping_exec"
