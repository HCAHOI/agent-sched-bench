from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tool_resource import (
    DockerExecutionContext,
    LatencyBuckets,
    SidecarUnavailableError,
    ToolResourceSDK,
)
from tool_resource.sidecar_protocol import SidecarTransport


def _clause(*, timestamps: bool) -> dict[str, Any]:
    row = {
        "bin": "echo",
        "argv": ["echo", "hi"],
        "latency_ms": 200.0,
        "peak_cpu_cores": None,
        "sampled_peak_rss_mb": None,
        "cpu_ns_cumulative": 1,
        "availability": {
            "latency": "ok",
            "cpu": "unknown:short",
            "memory": "unknown:short",
        },
    }
    if timestamps:
        row.update({"ts_start": 10.0, "ts_end": 12.0})
    return row


def _call(tool_call_id: str, *, timestamps: bool) -> dict[str, Any]:
    return {
        "version": 2,
        "tool_call_id": tool_call_id,
        "command": "echo hi",
        "telemetry_quality": "ok",
        "eligible_for_kb": True,
        "invalid_reasons": [],
        "clauses": [_clause(timestamps=timestamps)],
        "integrity": {"status": "ok", "errors": []},
    }


def _invalid_call(tool_call_id: str, *, timestamps: bool) -> dict[str, Any]:
    call = _call(tool_call_id, timestamps=timestamps)
    call.update(
        {
            "telemetry_quality": "invalid",
            "eligible_for_kb": False,
            "invalid_reasons": [
                {
                    "kind": "unmatched_static_clause",
                    "detail": "fixture mapping gap",
                }
            ],
            "integrity": {
                "status": "failed",
                "errors": ["fixture mapping gap"],
            },
        }
    )
    return call


def _write_artifact(
    path: Path,
    calls: list[dict[str, Any]],
    *,
    replay_execution: str = "completed",
    cleanup: str = "ok",
    container_id: str = "cold-container",
    repo: str = "public",
) -> None:
    valid_count = sum(call["telemetry_quality"] == "ok" for call in calls)
    invalid_count = sum(call["telemetry_quality"] == "invalid" for call in calls)
    unavailable_count = sum(
        call["telemetry_quality"] == "unavailable" for call in calls
    )
    eligible_count = sum(call["eligible_for_kb"] is True for call in calls)
    collector_healthy = cleanup == "ok" and unavailable_count == 0
    formal_completeness = (
        "unavailable"
        if not collector_healthy
        else ("complete" if eligible_count == len(calls) else "partial")
    )
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "mode": "clause",
                "status_model": "call_granular_v1",
                "container_id": container_id,
                "calls": calls,
                "telemetry_loss_total": {"total": 0},
                "collector": {
                    "state": "closed",
                    "state_before_close": (
                        "active" if collector_healthy else "disabled"
                    ),
                    "health": ("healthy" if collector_healthy else "unavailable"),
                    "valid_call_count": valid_count,
                    "invalid_call_count": invalid_count,
                    "unavailable_call_count": unavailable_count,
                    "eligible_call_count": eligible_count,
                },
                "cleanup": cleanup,
                "replay_execution": replay_execution,
                "telemetry_quality": ("ok" if collector_healthy else "unavailable"),
                "formal_completeness": formal_completeness,
                "collection_validity": ("valid" if collector_healthy else "invalid"),
                "integrity": {
                    "status": "ok" if collector_healthy else "failed",
                    "errors": (
                        [] if collector_healthy else ["fixture collector failure"]
                    ),
                },
                "provenance": {"repo": repo},
            }
        ),
        encoding="utf-8",
    )


def _cold_start_sdk(
    tmp_path: Path,
    *,
    transport_factory: Any | None = None,
) -> ToolResourceSDK:
    path = tmp_path / "cold-start.json"
    cold_call = _call("cold-1", timestamps=False)
    cold_call["clauses"][0]["latency_ms"] = 50.0
    _write_artifact(path, [cold_call])
    return ToolResourceSDK.from_traces(
        path,
        LatencyBuckets((100.0,)),
        transport_factory=transport_factory,
    )


class _Collector:
    latest: _Collector

    def __init__(self, **kwargs: Any) -> None:
        type(self).latest = self
        self.artifact_path = Path(kwargs["artifact_path"])
        self.attached = kwargs
        self.calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.integrity_errors: list[str] = []

    def begin_tool_call(self, tool_call_id: str, command: str) -> object:
        self.tool_call_id = tool_call_id
        self.events.append("observer_start")
        return object()

    def finish_tool_call(
        self,
        _token: object,
        *,
        replay_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.events.append("observer_finish")
        summary = _call(self.tool_call_id, timestamps=True)
        self.calls.append(summary)
        return summary

    def add_integrity_error(self, message: str) -> None:
        self.integrity_errors.append(message)

    def finalize(self, *, replay_execution: str) -> None:
        self.events.append(f"finalize:{replay_execution}")
        _write_artifact(
            self.artifact_path,
            self.calls,
            replay_execution=replay_execution,
            container_id=str(self.attached["container_id"]),
            repo=str(self.attached["repo"]),
        )


class _CollectorTransport:
    def __init__(self, context: DockerExecutionContext, collector_class: type) -> None:
        self.context = context
        self.collector_class = collector_class
        self.collector: Any | None = None
        self.tokens: dict[str, object] = {}

    def request(
        self,
        operation: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        if operation == "open":
            self.collector = self.collector_class(
                container_id=payload["container_id"],
                container_executable=self.context.container_executable,
                repo=payload["repo"],
                artifact_path=self.context.artifact_path.with_suffix(".sidecar.json"),
                source_actions=payload["source_actions"],
            )
            return {"session_id": "session-1"}
        assert self.collector is not None
        if operation == "begin":
            token = self.collector.begin_tool_call(
                payload["tool_call_id"],
                payload["command"],
            )
            self.tokens["token-1"] = token
            return {"token_id": "token-1"}
        if operation == "finish":
            return {
                "call": self.collector.finish_tool_call(
                    self.tokens.pop(payload["token_id"]),
                    replay_response=payload["replay_response"],
                )
            }
        if operation == "safety_guard":
            return {
                "call": self.collector.record_safety_guard_blocked(
                    payload["tool_call_id"],
                    payload["command"],
                    payload["replay_result"],
                )
            }
        if operation == "add_error":
            self.collector.add_integrity_error(payload["message"])
            return {}
        if operation == "finalize":
            self.collector.finalize(replay_execution=payload["replay_execution"])
            artifact_path = self.collector.artifact_path
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact_path.unlink()
            return {"artifact": artifact}
        raise AssertionError(operation)


def _observing_sdk(
    tmp_path: Path,
    collector_class: type = _Collector,
) -> ToolResourceSDK:
    def transport_factory(context: DockerExecutionContext) -> SidecarTransport:
        return _CollectorTransport(context, collector_class)

    return _cold_start_sdk(tmp_path, transport_factory=transport_factory)


def _context(tmp_path: Path, name: str) -> DockerExecutionContext:
    return DockerExecutionContext(
        container_id="container-1",
        container_executable="docker",
        repo="repo-1",
        artifact_path=tmp_path / f"{name}.json",
    )


def test_cold_start_command_transaction_and_causal_update(
    tmp_path: Path,
) -> None:
    sdk = _observing_sdk(tmp_path)
    run = sdk.start_command(
        _context(tmp_path, "first"),
        "call-1",
        "echo hi",
        ts_start=9.0,
    )
    collector = _Collector.latest

    collector.events.append("docker_execute")
    actual = {"returncode": 0, "result": "hi"}
    result = sdk.finish_command(run, actual)

    assert run.prediction is not None and run.prediction.prediction is not None
    assert run.prediction.prediction.scope == "public"
    assert run.prediction.prediction.bucket_id == 0
    assert collector.events == [
        "observer_start",
        "docker_execute",
        "observer_finish",
        "finalize:completed",
    ]
    assert collector.attached["container_id"] == "container-1"
    assert result.workload_result is actual
    assert result.telemetry_artifact is not None
    assert result.telemetry_artifact["collection_validity"] == "valid"
    assert result.kb_observations_added == 1
    assert result.kb_update_error is None
    with pytest.raises(ValueError, match="already been finished"):
        sdk.finish_command(run, actual)

    at_end = sdk.start_command(
        _context(tmp_path, "same-end"),
        "call-2",
        "echo hi",
        ts_start=12.0,
    )
    assert at_end.prediction is not None and at_end.prediction.prediction is not None
    assert at_end.prediction.prediction.scope == "public"
    sdk.finish_command(at_end, actual, replay_execution="failed")

    after_end = sdk.start_command(
        _context(tmp_path, "after-end"),
        "call-3",
        "echo hi",
        ts_start=12.1,
    )
    assert (
        after_end.prediction is not None and after_end.prediction.prediction is not None
    )
    assert after_end.prediction.prediction.scope == "repo"
    assert after_end.prediction.prediction.bucket_id == 1
    assert after_end.prediction.prediction.evidence_count == 2
    sdk.finish_command(after_end, actual, replay_execution="failed")


class _CleanupFailureCollector(_Collector):
    def finalize(self, *, replay_execution: str) -> None:
        _write_artifact(
            self.artifact_path,
            self.calls,
            replay_execution=replay_execution,
            cleanup="failed",
            container_id=str(self.attached["container_id"]),
            repo=str(self.attached["repo"]),
        )


def test_kb_update_waits_for_final_cleanup_and_collection_validity(
    tmp_path: Path,
) -> None:
    sdk = _observing_sdk(tmp_path, _CleanupFailureCollector)
    run = sdk.start_command(
        _context(tmp_path, "cleanup-failed"),
        "call-1",
        "echo hi",
        ts_start=9.0,
    )

    result = sdk.finish_command(run, {"returncode": 0, "result": "hi"})

    assert result.kb_observations_added == 0
    assert result.kb_update_error is not None
    assert result.telemetry_artifact is not None
    assert result.telemetry_artifact["collection_validity"] == "invalid"
    later = sdk.start_command(
        _context(tmp_path, "later"),
        "call-2",
        "echo hi",
        ts_start=20.0,
    )
    assert later.prediction is not None and later.prediction.prediction is not None
    assert later.prediction.prediction.scope == "public"
    sdk.finish_command(later, {"returncode": 0}, replay_execution="failed")


class _FailingCollector(_Collector):
    def finish_tool_call(
        self,
        _token: object,
        *,
        replay_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("ring reader failed")


def test_telemetry_failure_preserves_actual_result_and_does_not_update_kb(
    tmp_path: Path,
) -> None:
    sdk = _observing_sdk(tmp_path, _FailingCollector)
    run = sdk.start_command(
        _context(tmp_path, "telemetry-failed"),
        "call-1",
        "echo hi",
        ts_start=9.0,
    )
    actual = {"returncode": 7, "result": "workload failed"}

    result = sdk.finish_command(run, actual)

    assert result.workload_result is actual
    assert result.call_telemetry["telemetry_quality"] == "unavailable"
    assert result.kb_observations_added == 0
    assert result.kb_update_error is not None


def test_cold_start_rejects_invalid_trace(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    _write_artifact(path, [_call("cold-1", timestamps=False)], cleanup="failed")

    with pytest.raises(ValueError, match="no valid cold-start telemetry artifacts"):
        ToolResourceSDK.from_traces(path, LatencyBuckets((100.0,)))


def test_cold_start_uses_valid_calls_from_partial_artifact(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    partial_path = tmp_path / "partial.json"
    _write_artifact(valid_path, [_call("cold-1", timestamps=False)])
    _write_artifact(
        partial_path,
        [
            _call("cold-2", timestamps=False),
            _invalid_call("cold-3", timestamps=False),
        ],
    )

    sdk = ToolResourceSDK.from_traces(
        [valid_path, partial_path],
        LatencyBuckets((100.0,)),
    )

    assert sdk.cold_start_report is not None
    assert sdk.cold_start_report.artifacts_seen == 2
    assert sdk.cold_start_report.artifacts_accepted == 2
    assert sdk.cold_start_report.artifacts_rejected == 0
    assert sdk.cold_start_report.calls_seen == 3
    assert sdk.cold_start_report.eligible_calls_loaded == 2
    assert sdk.cold_start_report.calls_withheld == 1
    assert sdk.cold_start_report.observations_loaded == 2


def test_cold_start_isolates_unavailable_artifact(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    unavailable_path = tmp_path / "unavailable.json"
    _write_artifact(valid_path, [_call("cold-1", timestamps=False)])
    _write_artifact(
        unavailable_path,
        [_call("cold-2", timestamps=False)],
        cleanup="failed",
    )

    sdk = ToolResourceSDK.from_traces(
        [valid_path, unavailable_path],
        LatencyBuckets((100.0,)),
    )

    assert sdk.cold_start_report is not None
    assert sdk.cold_start_report.artifacts_accepted == 1
    assert sdk.cold_start_report.artifacts_rejected == 1
    assert sdk.cold_start_report.observations_loaded == 1
    assert str(unavailable_path) in sdk.cold_start_report.rejections[0]


def test_command_run_cannot_be_finished_by_another_sdk(
    tmp_path: Path,
) -> None:
    sdk = _observing_sdk(tmp_path)
    other = _observing_sdk(tmp_path)
    run = sdk.start_command(
        _context(tmp_path, "owned"),
        "call-1",
        "echo hi",
        ts_start=9.0,
    )

    with pytest.raises(ValueError, match="different SDK"):
        other.finish_command(run, {"returncode": 0})

    result = sdk.finish_command(run, {"returncode": 0})
    assert result.kb_observations_added == 1


class _FinalizeFailureCollector(_Collector):
    def finalize(self, *, replay_execution: str) -> None:
        raise RuntimeError("collector cleanup failed")


def test_stale_artifact_or_finalizer_failure_cannot_update_kb(
    tmp_path: Path,
) -> None:
    sdk = _observing_sdk(tmp_path, _FinalizeFailureCollector)
    stale_context = _context(tmp_path, "stale")
    _write_artifact(
        stale_context.artifact_path,
        [_call("call-1", timestamps=True)],
        container_id=stale_context.container_id,
        repo=stale_context.repo,
    )
    with pytest.raises(ValueError, match="already exists"):
        sdk.start_command(
            stale_context,
            "call-1",
            "echo hi",
            ts_start=9.0,
        )

    run = sdk.start_command(
        _context(tmp_path, "finalize-failed"),
        "call-2",
        "echo hi",
        ts_start=10.0,
    )
    result = sdk.finish_command(run, {"returncode": 0})

    assert result.telemetry_artifact is None
    assert result.kb_observations_added == 0
    assert "cleanup failed" in str(result.kb_update_error)


class _DisconnectingTransport(_CollectorTransport):
    def request(
        self,
        operation: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if operation == "finish":
            raise SidecarUnavailableError("socket disconnected")
        return super().request(operation, payload)


def test_sidecar_disconnect_preserves_workload_and_blocks_update(
    tmp_path: Path,
) -> None:
    sdk = _cold_start_sdk(
        tmp_path,
        transport_factory=lambda context: _DisconnectingTransport(
            context,
            _Collector,
        ),
    )
    run = sdk.start_command(
        _context(tmp_path, "disconnect"),
        "call-1",
        "echo hi",
        ts_start=9.0,
    )
    actual = {"returncode": 0, "result": "hi"}

    result = sdk.finish_command(run, actual)

    assert result.workload_result is actual
    assert result.call_telemetry["telemetry_quality"] == "unavailable"
    assert result.telemetry_artifact is None
    assert result.kb_observations_added == 0
    assert "disconnected" in str(result.kb_update_error)


def test_sidecar_unavailable_preserves_workload_and_blocks_update(
    tmp_path: Path,
) -> None:
    sdk = _cold_start_sdk(tmp_path)
    run = sdk.start_command(
        _context(tmp_path, "unavailable"),
        "call-1",
        "echo hi",
        ts_start=9.0,
    )
    actual = {"returncode": 9, "result": "workload failed"}

    result = sdk.finish_command(run, actual)

    assert result.workload_result is actual
    assert result.call_telemetry["telemetry_quality"] == "unavailable"
    assert result.telemetry_artifact is None
    assert result.kb_observations_added == 0
    assert "not configured" in str(result.kb_update_error)
