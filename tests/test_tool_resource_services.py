from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tool_resource._uds import receive_message, send_message
from tool_resource.client import ResourceRun
from tool_resource.profile import ResourceProfile
from tool_resource.resource_agentd import ResourceServer, ResourceService
from tool_resource.resource_protocol import (
    RESOURCE_PROTOCOL_VERSION,
    ResourceProtocolError,
    ResourceUnixTransport,
)
from tool_resource.store import ObservationStore
from tool_resource.telemetry_protocol import (
    TELEMETRY_PROTOCOL_VERSION,
    TelemetryProtocolError,
    TelemetryUnavailableError,
    TelemetryUnixTransport,
)
from tool_resource.telemetryd import TelemetryServer, TelemetryService
from trace_collect.openclaw_tools import ContainerAgent


class _DirectTransport:
    def __init__(self, service: Any) -> None:
        self.service = service
        self.fail_operation: str | None = None
        self.failure_detail = "disconnected"

    def request(
        self,
        operation: str,
        payload: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_id
        if operation == self.fail_operation:
            raise TelemetryUnavailableError(f"{operation} {self.failure_detail}")
        return self.service.dispatch(operation, payload or {})


class _FakeCollector:
    cleanup = "ok"

    def __init__(self, **kwargs: Any) -> None:
        self.state = "active"
        self.init_pid = 123
        self.cgroup_id = 456
        self.quota_cores = 2.0
        self.artifact_path = Path(kwargs["artifact_path"])
        self.calls: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def begin_tool_call(
        self,
        call_id: str,
        command: str,
        **plan: Any,
    ) -> dict[str, Any]:
        return {"call_id": call_id, "command": command, **plan}

    def finish_tool_call(
        self,
        token: dict[str, Any],
        *,
        replay_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del replay_response
        valid = "invalid" not in token["command"]
        clause = dict(token["static_plan"]["clauses"][0])
        now = time.time()
        summary = {
            "tool_call_id": token["call_id"],
            "command": token["command"],
            "telemetry_quality": "ok" if valid else "invalid",
            "eligible_for_kb": valid,
            "invalid_reasons": (
                [] if valid else [{"kind": "ambiguous", "detail": "fixture"}]
            ),
            "telemetry_loss": {"total": 0},
            "runtime_invocations": [{"raw_event": "must-not-cross-protocol"}],
            "clauses": [
                {
                    "bin": clause["bin"],
                    "argv": clause["argv"],
                    "ts_start": now - 0.25,
                    "ts_end": now,
                    "latency_ms": 250.0,
                    "peak_cpu_cores": 0.5,
                    "sampled_peak_rss_mb": 8.0,
                    "cpu_ns_cumulative": 10,
                    "in_loop": False,
                    "in_pipe": False,
                    "in_subst": False,
                    "pipeline_position": -1,
                    "availability": {
                        "latency": "ok",
                        "cpu": "ok",
                        "memory": "ok",
                        "disk_io": "ok",
                    },
                }
            ],
        }
        self.calls.append(summary)
        return summary

    def record_safety_guard_blocked(
        self,
        call_id: str,
        command: str,
        workload_result: str,
        **plan: Any,
    ) -> dict[str, Any]:
        del workload_result
        return self.finish_tool_call(self.begin_tool_call(call_id, command, **plan))

    def add_integrity_error(self, message: str) -> None:
        self.errors.append(message)

    def finalize(self, *, replay_execution: str) -> None:
        healthy = self.cleanup == "ok" and not self.errors
        eligible = sum(call["eligible_for_kb"] for call in self.calls)
        artifact = {
            "calls": self.calls,
            "replay_execution": replay_execution,
            "telemetry_quality": "ok" if healthy else "unavailable",
            "collection_validity": "valid" if healthy else "invalid",
            "formal_completeness": (
                "unavailable"
                if not healthy
                else ("complete" if eligible == len(self.calls) else "partial")
            ),
            "cleanup": self.cleanup,
            "collector": {"health": "healthy" if healthy else "unavailable"},
            "call_coverage": {
                "eligible_call_count": eligible,
                "withheld_call_count": len(self.calls) - eligible,
            },
            "telemetry_loss_total": {"total": 0},
            "integrity": {"errors": list(self.errors)},
        }
        self.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


class _CleanupFailureCollector(_FakeCollector):
    cleanup = "failed"


class _LossCollector(_FakeCollector):
    def finalize(self, *, replay_execution: str) -> None:
        super().finalize(replay_execution=replay_execution)
        artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        artifact["telemetry_quality"] = "unavailable"
        artifact["collection_validity"] = "invalid"
        artifact["formal_completeness"] = "unavailable"
        artifact["collector"]["health"] = "unavailable"
        artifact["telemetry_loss_total"] = {"total": 1}
        self.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


class _MismatchedClauseCollector(_FakeCollector):
    def finish_tool_call(
        self,
        token: dict[str, Any],
        *,
        replay_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = super().finish_tool_call(token, replay_response=replay_response)
        result["clauses"][0]["argv"] = ["different", "command"]
        return result


def _envelope(
    observation_id: str,
    *,
    run_id: str = "seed",
    scope: str = "repo",
    command: str = "printf old",
    end: float = 10.0,
    latency_ms: float = 50.0,
) -> dict[str, Any]:
    argv = command.split()
    return {
        "observation_id": observation_id,
        "run_id": run_id,
        "trace_id": "seed-trace",
        "call_id": observation_id,
        "workspace_scope": scope,
        "canonicalizer_version": "mvdan-sh-v1",
        "command_digest": "a" * 64,
        "observation_interval": {"start": end - latency_ms / 1000.0, "end": end},
        "normalized_measurements": [
            {
                "bin": argv[0],
                "argv": argv,
                "ts_start": end - latency_ms / 1000.0,
                "ts_end": end,
                "latency_ms": latency_ms,
                "peak_cpu_cores": None,
                "sampled_peak_rss_mb": None,
                "cpu_ns_cumulative": None,
                "in_loop": False,
                "in_pipe": False,
                "in_subst": False,
                "pipeline_position": -1,
                "availability": {"latency": "ok"},
            }
        ],
        "telemetry_eligible": True,
        "ingest_eligible": True,
        "rejection_reasons": [],
    }


def _open_run(
    service: ResourceService,
    *,
    behavior: str = "observe_predict_learn",
    update_policy: str = "causal",
    snapshot: str = "latest_at_run_start",
    run_id: str = "run",
    scope: str = "repo",
) -> dict[str, Any]:
    return service.dispatch(
        "OpenRun",
        {
            "run_id": run_id,
            "workspace_scope": scope,
            "snapshot": snapshot,
            "latency_bucket_edges_ms": [100.0],
            "update_policy": update_policy,
            "telemetry_requirement": "required_for_valid_evidence",
            "behavior": behavior,
        },
    )


def _open_trace(
    service: ResourceService,
    run_token: str,
    *,
    trace_id: str = "trace",
) -> dict[str, Any]:
    return service.dispatch(
        "OpenTrace",
        {
            "run_token": run_token,
            "trace_id": trace_id,
            "container_runtime": "docker",
            "container_id": f"container-{trace_id}",
            "repo_metadata": {},
            "source_actions": [],
        },
    )


def _run_call(
    service: ResourceService,
    trace_token: str,
    *,
    call_id: str,
    command: str,
    query_timestamp: float | None = None,
    workload_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    begin = service.dispatch(
        "BeginCall",
        {
            "trace_token": trace_token,
            "call_id": call_id,
            "command": command,
            "query_timestamp": query_timestamp or time.time(),
        },
    )
    actual = workload_result or {"returncode": 0, "result": "ok"}
    end = service.dispatch(
        "EndCall",
        {
            "call_token": begin["call_token"],
            "workload_result": actual,
            "end_timestamp": time.time(),
        },
    )
    assert end["workload_result"] is actual
    return begin, end


def _evict_response_cache(transport: Any, prefix: str) -> None:
    for index in range(1024):
        transport.request("Ping", request_id=f"{prefix}-{index}")


def test_sqlite_wal_idempotency_and_snapshot_scope(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    assert (
        store._connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    )
    first = _envelope("obs-1")
    assert store.insert_observation(first)[0] is True
    assert store.insert_observation(first)[0] is False
    with pytest.raises(ValueError, match="different normalized envelope"):
        store.insert_observation({**first, "call_id": "collision"})
    assert store.observation_count() == 1

    before = store.create_snapshot()
    assert store.observations_for_snapshot(before, "repo") == []
    assert store.promote_observations({"obs-1"}) == 1
    after = store.create_snapshot()
    assert len(store.observations_for_snapshot(after, "repo")) == 1
    assert store.observations_for_snapshot(after, "other") == []
    store.close()


def test_snapshot_membership_is_immutable_after_late_promotion(
    tmp_path: Path,
) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.insert_observation(_envelope("pending", run_id="pending"))
    store.insert_observation(_envelope("public", run_id="public"))
    store.promote_observations({"public"})
    snapshot = store.create_snapshot()

    store.promote_observations({"pending"})
    assert [
        row["observation_id"]
        for row in store.observations_for_snapshot(snapshot, "repo")
    ] == ["public"]
    next_snapshot = store.create_snapshot()
    assert {
        row["observation_id"]
        for row in store.observations_for_snapshot(next_snapshot, "repo")
    } == {"pending", "public"}
    store.close()


def test_snapshot_strict_causal_boundary_and_cross_scope(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.insert_observation(_envelope("repo", end=10.0))
    store.insert_observation(_envelope("other", scope="other", end=1.0))
    store.promote_observations({"repo", "other"})
    snapshot = store.create_snapshot()
    service = ResourceService(
        store,
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    run = _open_run(
        service,
        behavior="predict",
        snapshot=snapshot,
    )
    trace = _open_trace(service, run["run_token"])

    at_boundary = service.dispatch(
        "BeginCall",
        {
            "trace_token": trace["trace_token"],
            "call_id": "at-boundary",
            "command": "printf old",
            "query_timestamp": 10.0,
        },
    )
    assert at_boundary["prediction"] == {}
    after_boundary = service.dispatch(
        "BeginCall",
        {
            "trace_token": trace["trace_token"],
            "call_id": "after-boundary",
            "command": "printf old",
            "query_timestamp": 10.1,
        },
    )
    assert after_boundary["prediction"]["prediction"]["bucket_id"] == 0
    assert after_boundary["evidence_count"] == 1
    service.close()


@pytest.mark.parametrize(
    "update_policy, expected_bucket", [("frozen", 0), ("causal", 1)]
)
def test_frozen_and_causal_visibility(
    tmp_path: Path,
    update_policy: str,
    expected_bucket: int,
) -> None:
    store = ObservationStore(tmp_path / f"{update_policy}.sqlite3")
    store.insert_observation(_envelope("seed", command="printf old"))
    store.promote_observations({"seed"})
    telemetry = TelemetryService(
        collector_factory=_FakeCollector,
        state_dir=tmp_path / f"telemetry-{update_policy}",
    )
    service = ResourceService(store, _DirectTransport(telemetry))
    run = _open_run(service, update_policy=update_policy)
    first_trace = _open_trace(service, run["run_token"], trace_id="first")
    _run_call(
        service,
        first_trace["trace_token"],
        call_id="learn",
        command="echo new",
    )
    service.dispatch(
        "CloseTrace",
        {"trace_token": first_trace["trace_token"], "workload_status": "completed"},
    )

    second_trace = _open_trace(service, run["run_token"], trace_id="second")
    prediction = service.dispatch(
        "BeginCall",
        {
            "trace_token": second_trace["trace_token"],
            "call_id": "query",
            "command": "echo new",
            "query_timestamp": time.time() + 1.0,
        },
    )
    assert prediction["prediction"]["prediction"]["bucket_id"] == expected_bucket
    service.close()


def test_partial_trace_ingests_only_valid_calls(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    telemetry = TelemetryService(
        collector_factory=_FakeCollector,
        state_dir=tmp_path / "telemetry",
    )
    service = ResourceService(store, _DirectTransport(telemetry))
    run = _open_run(service)
    trace = _open_trace(service, run["run_token"])
    _run_call(service, trace["trace_token"], call_id="valid", command="echo ok")
    _run_call(
        service,
        trace["trace_token"],
        call_id="invalid",
        command="echo invalid",
    )

    closed = service.dispatch(
        "CloseTrace",
        {"trace_token": trace["trace_token"], "workload_status": "completed"},
    )
    assert closed["formal_completeness"] == "partial"
    assert closed["call_coverage"] == {
        "total_call_count": 2,
        "eligible_call_count": 1,
        "withheld_call_count": 1,
        "eligible_fraction": 0.5,
    }
    assert store.observation_count() == 2
    calls = {call["tool_call_id"]: call for call in closed["artifact"]["calls"]}
    assert calls["valid"]["eligible_for_kb"] is True
    assert calls["invalid"]["eligible_for_kb"] is False
    closed_run = service.dispatch(
        "CloseRun",
        {"run_token": run["run_token"], "workload_status": "completed"},
    )
    assert closed_run["promoted_observation_count"] == 1
    service.close()


def test_duplicate_external_run_ids_do_not_cross_promote(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    service = ResourceService(
        store,
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    first_run = _open_run(service, run_id="shared")
    second_run = _open_run(service, run_id="shared")
    first_trace = _open_trace(
        service,
        first_run["run_token"],
        trace_id="shared-trace",
    )
    second_trace = _open_trace(
        service,
        second_run["run_token"],
        trace_id="shared-trace",
    )
    _run_call(
        service,
        first_trace["trace_token"],
        call_id="call",
        command="echo one",
    )
    _run_call(
        service,
        second_trace["trace_token"],
        call_id="call",
        command="echo two",
    )
    first_closed = service.dispatch(
        "CloseTrace",
        {"trace_token": first_trace["trace_token"], "workload_status": "completed"},
    )
    service.dispatch(
        "CloseTrace",
        {"trace_token": second_trace["trace_token"], "workload_status": "completed"},
    )
    service.dispatch(
        "CloseRun",
        {"run_token": first_run["run_token"], "workload_status": "completed"},
    )
    with store._lock:
        visible_ids = {
            row[0]
            for row in store._connection.execute(
                "SELECT observation_id FROM observations WHERE visible=1"
            )
        }
    assert visible_ids == {first_closed["artifact"]["calls"][0]["observation_id"]}
    service.dispatch(
        "CloseRun",
        {"run_token": second_run["run_token"], "workload_status": "completed"},
    )
    with store._lock:
        assert (
            store._connection.execute(
                "SELECT COUNT(*) FROM observations WHERE visible=1"
            ).fetchone()[0]
            == 2
        )
    service.close()


def test_trace_integrity_failure_withholds_otherwise_valid_call(
    tmp_path: Path,
) -> None:
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    run = _open_run(service)
    trace = _open_trace(service, run["run_token"])
    _run_call(service, trace["trace_token"], call_id="call", command="echo ok")
    service.dispatch(
        "RecordTraceIntegrityFailure",
        {
            "trace_token": trace["trace_token"],
            "message": "action mapping failed",
        },
    )
    closed = service.dispatch(
        "CloseTrace",
        {"trace_token": trace["trace_token"], "workload_status": "completed"},
    )
    assert closed["formal_completeness"] == "partial"
    assert closed["artifact"]["calls"][0]["eligible_for_kb"] is False
    assert {
        reason["kind"] for reason in closed["artifact"]["calls"][0]["invalid_reasons"]
    } >= {"formal_mapping"}
    closed_run = service.dispatch(
        "CloseRun",
        {"run_token": run["run_token"], "workload_status": "completed"},
    )
    assert closed_run["promoted_observation_count"] == 0
    service.close()


def test_telemetry_clause_identity_mismatch_is_withheld(tmp_path: Path) -> None:
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_MismatchedClauseCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    run = _open_run(service)
    trace = _open_trace(service, run["run_token"])
    _run_call(service, trace["trace_token"], call_id="call", command="echo ok")
    closed = service.dispatch(
        "CloseTrace",
        {"trace_token": trace["trace_token"], "workload_status": "completed"},
    )
    assert closed["artifact"]["calls"][0]["eligible_for_kb"] is False
    assert closed["artifact"]["calls"][0]["invalid_reasons"][-1]["kind"] == (
        "canonicalization"
    )
    service.close()


@pytest.mark.parametrize(
    "failure",
    ["disconnect", "timeout", "ack_disconnect", "loss", "cleanup"],
)
def test_telemetry_failures_preserve_workload_and_block_ingestion(
    tmp_path: Path,
    failure: str,
) -> None:
    collector = {
        "cleanup": _CleanupFailureCollector,
        "loss": _LossCollector,
    }.get(failure, _FakeCollector)
    telemetry = TelemetryService(
        collector_factory=collector,
        state_dir=tmp_path / "telemetry",
    )
    transport = _DirectTransport(telemetry)
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        transport,
    )
    run = _open_run(service)
    trace = _open_trace(service, run["run_token"])
    begin = service.dispatch(
        "BeginCall",
        {
            "trace_token": trace["trace_token"],
            "call_id": "call",
            "command": "echo ok",
            "query_timestamp": time.time(),
        },
    )
    if failure in {"disconnect", "timeout"}:
        transport.fail_operation = "FinishCall"
        transport.failure_detail = failure
    actual = {"returncode": 7, "result": "workload result"}
    end = service.dispatch(
        "EndCall",
        {
            "call_token": begin["call_token"],
            "workload_result": actual,
            "end_timestamp": time.time(),
        },
    )
    assert end["workload_result"] is actual
    if failure == "ack_disconnect":
        transport.fail_operation = "AcknowledgeObservation"
    closed = service.dispatch(
        "CloseTrace",
        {"trace_token": trace["trace_token"], "workload_status": "completed"},
    )
    assert closed["telemetry_status"] == "unavailable"
    assert not any(
        call.get("eligible_for_kb") is True for call in closed["artifact"]["calls"]
    )
    closed_run = service.dispatch(
        "CloseRun",
        {"run_token": run["run_token"], "workload_status": "completed"},
    )
    assert (
        service.store.observations_for_snapshot(
            closed_run["run_manifest"]["resulting_snapshot_id"],
            "repo",
        )
        == []
    )
    service.close()


def test_telemetry_startup_failure_preserves_prediction_and_workload(
    tmp_path: Path,
) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.insert_observation(_envelope("seed"))
    store.promote_observations({"seed"})
    telemetry = TelemetryService(
        collector_factory=_FakeCollector,
        state_dir=tmp_path / "telemetry",
    )
    transport = _DirectTransport(telemetry)
    transport.fail_operation = "AttachTarget"
    service = ResourceService(store, transport)
    run = _open_run(service)
    trace = _open_trace(service, run["run_token"])
    begin, end = _run_call(
        service,
        trace["trace_token"],
        call_id="call",
        command="printf old",
    )
    assert begin["prediction"]["prediction"]["bucket_id"] == 0
    assert end["workload_result"] == {"returncode": 0, "result": "ok"}
    closed = service.dispatch(
        "CloseTrace",
        {"trace_token": trace["trace_token"], "workload_status": "completed"},
    )
    assert closed["telemetry_status"] == "unavailable"
    assert store.observation_count() == 1
    service.close()


def test_protocol_validation_and_two_client_isolation(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.insert_observation(_envelope("seed"))
    store.promote_observations({"seed"})
    telemetry = TelemetryService(
        collector_factory=_FakeCollector,
        state_dir=tmp_path / "telemetry",
    )
    service = ResourceService(store, _DirectTransport(telemetry))
    socket_path = tmp_path / "resource.sock"
    server = ResourceServer(
        socket_path,
        service=service,
        allowed_uids={os.getuid()},
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        first = ResourceUnixTransport(socket_path)
        second = ResourceUnixTransport(socket_path)
        assert first.ping() == {"protocol_version": RESOURCE_PROTOCOL_VERSION}
        with pytest.raises(ResourceProtocolError, match="unknown payload"):
            first.request("Ping", {"unexpected": True})

        first_payload = {
            "run_id": "one",
            "workspace_scope": "repo",
            "snapshot": "latest_at_run_start",
            "latency_bucket_edges_ms": [100.0],
            "update_policy": "frozen",
            "telemetry_requirement": "best_effort",
            "behavior": "predict",
        }
        second_payload = {
            "run_id": "two",
            "workspace_scope": "repo",
            "snapshot": "latest_at_run_start",
            "latency_bucket_edges_ms": [100.0],
            "update_policy": "causal",
            "telemetry_requirement": "best_effort",
            "behavior": "predict",
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            run1_future = pool.submit(first.request, "OpenRun", first_payload)
            run2_future = pool.submit(second.request, "OpenRun", second_payload)
            run1 = run1_future.result()
            run2 = run2_future.result()
        assert run1["run_token"] != run2["run_token"]
        assert (
            first.request("OpenRun", first_payload, request_id="retry")["run_token"]
            == first.request(
                "OpenRun",
                first_payload,
                request_id="retry",
            )["run_token"]
        )
        with pytest.raises(ResourceProtocolError, match="different request"):
            first.request(
                "OpenRun",
                {**first_payload, "behavior": "observe_predict"},
                request_id="retry",
            )
        trace1 = first.request(
            "OpenTrace",
            {
                "run_token": run1["run_token"],
                "trace_id": "same",
                "container_runtime": "docker",
                "container_id": "one",
                "repo_metadata": {},
                "source_actions": [],
            },
        )
        trace2 = second.request(
            "OpenTrace",
            {
                "run_token": run2["run_token"],
                "trace_id": "same",
                "container_runtime": "docker",
                "container_id": "two",
                "repo_metadata": {},
                "source_actions": [],
            },
        )
        assert trace1["trace_token"] != trace2["trace_token"]

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
            raw.connect(str(socket_path))
            send_message(
                raw,
                {
                    "protocol_version": RESOURCE_PROTOCOL_VERSION,
                    "request_id": "bad",
                    "operation": "Ping",
                    "payload": {},
                    "extra": True,
                },
            )
            response = receive_message(raw)
        assert response["payload"]["ok"] is False
        assert "envelope fields" in response["payload"]["error"]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
            raw.connect(str(socket_path))
            send_message(
                raw,
                {
                    "protocol_version": RESOURCE_PROTOCOL_VERSION + 1,
                    "request_id": "wrong-version",
                    "operation": "Ping",
                    "payload": {},
                },
            )
            response = receive_message(raw)
        assert response["payload"]["ok"] is False
        assert "unsupported protocol version" in response["payload"]["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_stateful_retries_survive_response_cache_eviction(tmp_path: Path) -> None:
    resource_service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "unused-telemetry",
            )
        ),
    )
    resource_socket = tmp_path / "resource.sock"
    resource_server = ResourceServer(
        resource_socket,
        service=resource_service,
        allowed_uids={os.getuid()},
    )
    resource_thread = threading.Thread(target=resource_server.serve_forever)
    resource_thread.start()
    try:
        transport = ResourceUnixTransport(resource_socket)
        run_payload = {
            "run_id": "stable-run",
            "workspace_scope": "repo",
            "snapshot": "latest_at_run_start",
            "latency_bucket_edges_ms": [100.0],
            "update_policy": "frozen",
            "telemetry_requirement": "best_effort",
            "behavior": "predict",
        }
        run = transport.request("OpenRun", run_payload, request_id="open-run")
        trace_payload = {
            "run_token": run["run_token"],
            "trace_id": "stable-trace",
            "container_runtime": "docker",
            "container_id": "container",
            "repo_metadata": {},
            "source_actions": [],
        }
        trace = transport.request(
            "OpenTrace",
            trace_payload,
            request_id="open-trace",
        )
        call_payload = {
            "trace_token": trace["trace_token"],
            "call_id": "stable-call",
            "command": "printf stable",
            "query_timestamp": 100.0,
        }
        call = transport.request("BeginCall", call_payload, request_id="begin-call")

        _evict_response_cache(transport, "resource-evict")

        assert (
            transport.request("OpenRun", run_payload, request_id="open-run") == run
        )
        assert (
            transport.request(
                "OpenTrace",
                trace_payload,
                request_id="open-trace",
            )
            == trace
        )
        assert (
            transport.request("BeginCall", call_payload, request_id="begin-call")
            == call
        )
        assert len(resource_service._runs) == 1
        assert len(resource_service._traces) == 1
        assert len(next(iter(resource_service._traces.values())).calls) == 1
        next(iter(resource_service._runs.values())).last_used = 0.0
        resource_service.expire()
        assert resource_service._runs == {}
        assert resource_service._traces == {}
        assert resource_service._operation_results == {}
    finally:
        resource_server.shutdown()
        resource_server.server_close()
        resource_thread.join()

    telemetry_socket = tmp_path / "telemetry.sock"
    telemetry_service = TelemetryService(
        collector_factory=_FakeCollector,
        state_dir=tmp_path / "telemetry",
    )
    telemetry_server = TelemetryServer(
        telemetry_socket,
        service=telemetry_service,
        allowed_uids={os.getuid()},
    )
    telemetry_thread = threading.Thread(target=telemetry_server.serve_forever)
    telemetry_thread.start()
    try:
        transport = TelemetryUnixTransport(
            telemetry_socket,
            expected_peer_uid=os.getuid(),
        )
        attach_payload = {
            "run_id": "stable-run",
            "trace_id": "stable-trace",
            "container_runtime": "docker",
            "container_id": "container",
            "workspace_scope": "repo",
        }
        session = transport.request(
            "AttachTarget",
            attach_payload,
            request_id="attach-target",
        )
        register_payload = {
            "telemetry_session_token": session["telemetry_session_token"],
            "call_id": "stable-call",
            "command_digest": "a" * 64,
            "static_call_plan": {
                "canonical_command": "printf stable",
                "parsed": {
                    "clauses": [],
                    "control_edges": [],
                    "parse_failed": False,
                },
                "source_tool_call_id": "",
                "source_command": "",
                "source_tool_result": "",
            },
        }
        call = transport.request(
            "RegisterCall",
            register_payload,
            request_id="register-call",
        )

        _evict_response_cache(transport, "telemetry-evict")

        assert (
            transport.request(
                "AttachTarget",
                attach_payload,
                request_id="attach-target",
            )
            == session
        )
        assert (
            transport.request(
                "RegisterCall",
                register_payload,
                request_id="register-call",
            )
            == call
        )
        assert len(telemetry_service._sessions) == 1
        assert len(next(iter(telemetry_service._sessions.values())).calls) == 1
        next(iter(telemetry_service._sessions.values())).last_used = 0.0
        telemetry_service.expire()
        assert telemetry_service._sessions == {}
        assert telemetry_service._operation_results == {}
    finally:
        telemetry_server.shutdown()
        telemetry_server.server_close()
        telemetry_thread.join()


def test_telemetry_protocol_is_strict_and_separate(tmp_path: Path) -> None:
    socket_path = tmp_path / "run" / "telemetry.sock"
    server = TelemetryServer(
        socket_path,
        service=TelemetryService(
            collector_factory=_FakeCollector,
            state_dir=tmp_path / "telemetry",
        ),
        allowed_uids={os.getuid()},
        socket_gid=os.getgid(),
    )
    assert socket_path.parent.stat().st_mode & 0o777 == 0o750
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        transport = TelemetryUnixTransport(
            socket_path,
            expected_peer_uid=os.getuid(),
        )
        assert transport.ping() == {"protocol_version": TELEMETRY_PROTOCOL_VERSION}
        with pytest.raises(TelemetryProtocolError, match="unknown payload"):
            transport.request("Ping", {"resource_field": True})
        with pytest.raises(TelemetryProtocolError, match="unsupported telemetry"):
            transport.request("OpenRun", {})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_profile_requires_explicit_canonical_edges(tmp_path: Path) -> None:
    profile_path = tmp_path / "resource.yaml"
    profile_path.write_text(
        """
tool_resource:
  endpoint: unix:///run/user/1000/resource.sock
  behavior: predict
  update_policy: frozen
  snapshot: latest_at_run_start
  telemetry_requirement: best_effort
  latency_bucket_edges_ms: [100, 1000]
""".lstrip(),
        encoding="utf-8",
    )
    profile = ResourceProfile.load(profile_path)
    assert profile.latency_bucket_edges_ms == (100, 1000)
    assert profile.open_run_payload(run_id="run", workspace_scope="repo")[
        "latency_bucket_edges_ms"
    ] == [100, 1000]

    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            "  latency_bucket_edges_ms: [100, 1000]\n",
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical schema"):
        ResourceProfile.load(profile_path)

    profile_path.write_text(
        """
tool_resource:
  endpoint: unix:///run/user/1000/resource.sock
  behavior: predict
  update_policy: frozen
  snapshot: null
  telemetry_requirement: best_effort
  latency_bucket_edges_ms: [100]
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot must be a string"):
        ResourceProfile.load(profile_path)


def test_thin_client_reaches_idempotent_ingestion_over_uds(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path / "observations.sqlite3")
    service = ResourceService(
        store,
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    socket_path = tmp_path / "resource.sock"
    server = ResourceServer(
        socket_path,
        service=service,
        allowed_uids={os.getuid()},
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    profile_path = tmp_path / "resource.yaml"
    profile_path.write_text(
        f"""
tool_resource:
  endpoint: unix://{socket_path}
  behavior: observe_predict_learn
  update_policy: causal
  snapshot: latest_at_run_start
  telemetry_requirement: required_for_valid_evidence
  latency_bucket_edges_ms: [100]
""".lstrip(),
        encoding="utf-8",
    )
    try:
        resource_run = ResourceRun.open(
            profile_path,
            run_id="client-run",
            workspace_scope="repo",
            manifest_path=tmp_path / "run.json",
        )
        trace = resource_run.open_trace(
            trace_id="client-trace",
            container_runtime="docker",
            container_id="container",
            artifact_path=tmp_path / "resource-observations.json",
        )
        token = trace.begin_tool_call("call", "echo ok")
        actual = {"returncode": 0, "result": "ok"}
        call = trace.finish_tool_call(token, replay_response=actual)
        assert call["telemetry_quality"] == "ok"
        assert trace.finalize(replay_execution="completed") is None
        assert trace.final_artifact is not None
        assert "runtime_invocations" not in json.dumps(trace.final_artifact)
        assert "raw_event" not in json.dumps(trace.final_artifact)
        assert trace.final_artifact["workload_execution"] == "completed"
        assert resource_run.finalize(workload_status="completed") is None
        run_artifact = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert run_artifact["run_manifest"]["pinned_snapshot_id"]
        assert run_artifact["run_manifest"]["latency_bucket_edges_ms"] == [100.0]
        assert store.observation_count() == 1
        assert (
            json.loads(
                (tmp_path / "resource-observations.json").read_text(encoding="utf-8")
            )["calls"][0]["eligible_for_kb"]
            is True
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_thin_client_run_spans_traces_for_causal_visibility(tmp_path: Path) -> None:
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    profile = ResourceProfile(
        endpoint="unix:///unused.sock",
        behavior="observe_predict_learn",
        update_policy="causal",
        snapshot="latest_at_run_start",
        telemetry_requirement="required_for_valid_evidence",
        latency_bucket_edges_ms=(100,),
    )
    resource_run = ResourceRun.open(
        profile,
        run_id="causal-run",
        workspace_scope="repo",
        manifest_path=tmp_path / "run.json",
        transport=_DirectTransport(service),
    )
    first = resource_run.open_trace(
        trace_id="first",
        container_runtime="docker",
        container_id="first",
        artifact_path=tmp_path / "first.json",
    )
    first_token = first.begin_tool_call("first-call", "echo learned")
    first.finish_tool_call(
        first_token,
        replay_response={"returncode": 0, "result": "ok"},
    )
    assert first.finalize(replay_execution="completed") is None

    second = resource_run.open_trace(
        trace_id="second",
        container_runtime="docker",
        container_id="second",
        artifact_path=tmp_path / "second.json",
    )
    second_token = second.begin_tool_call("second-call", "echo learned")
    assert second_token.prediction is not None
    assert second_token.prediction["prediction"]["bucket_id"] == 1
    second.finish_tool_call(
        second_token,
        replay_response={"returncode": 0, "result": "ok"},
    )
    assert second.finalize(replay_execution="completed") is None
    assert resource_run.finalize(workload_status="completed") is None
    assert resource_run.result is not None
    assert resource_run.result["promoted_observation_count"] == 2
    service.close()


def test_prediction_only_client_does_not_require_telemetry(tmp_path: Path) -> None:
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    profile = ResourceProfile(
        endpoint="unix:///unused.sock",
        behavior="predict",
        update_policy="frozen",
        snapshot="latest_at_run_start",
        telemetry_requirement="best_effort",
        latency_bucket_edges_ms=(100,),
    )
    resource_run = ResourceRun.open(
        profile,
        run_id="predict-run",
        workspace_scope="repo",
        manifest_path=tmp_path / "run.json",
        transport=_DirectTransport(service),
    )
    trace = resource_run.open_trace(
        trace_id="predict-trace",
        container_runtime="docker",
        container_id="container",
        artifact_path=tmp_path / "prediction.json",
    )
    token = trace.begin_tool_call("call", "echo prediction")
    trace.finish_tool_call(
        token,
        replay_response={"returncode": 0, "result": "ok"},
    )
    assert trace.finalize(replay_execution="completed") is None
    assert trace.final_artifact is not None
    assert trace.final_artifact["telemetry_quality"] == "not_requested"
    assert resource_run.finalize(workload_status="completed") is None
    assert resource_run.result is not None
    assert resource_run.result["evidence_valid"] is True
    assert service.store.observation_count() == 0
    service.close()


def test_best_effort_run_preserves_invalid_telemetry_as_valid_evidence_policy(
    tmp_path: Path,
) -> None:
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_CleanupFailureCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    profile = ResourceProfile(
        endpoint="unix:///unused.sock",
        behavior="observe_predict_learn",
        update_policy="frozen",
        snapshot="latest_at_run_start",
        telemetry_requirement="best_effort",
        latency_bucket_edges_ms=(100,),
    )
    resource_run = ResourceRun.open(
        profile,
        run_id="best-effort",
        workspace_scope="repo",
        manifest_path=tmp_path / "run.json",
        transport=_DirectTransport(service),
    )
    trace = resource_run.open_trace(
        trace_id="trace",
        container_runtime="docker",
        container_id="container",
        artifact_path=tmp_path / "trace.json",
    )
    token = trace.begin_tool_call("call", "echo ok")
    actual = {"returncode": 0, "result": "ok"}
    assert trace.finish_tool_call(token, replay_response=actual)["eligible_for_kb"]
    assert trace.finalize(replay_execution="completed") is not None
    assert resource_run.finalize(workload_status="completed") is None
    assert resource_run.result is not None
    assert resource_run.result["telemetry_valid"] is False
    assert resource_run.result["evidence_valid"] is True
    assert resource_run.result["promoted_observation_count"] == 0
    service.close()


@pytest.mark.parametrize(
    ("requirement", "evidence_valid"),
    [
        ("best_effort", True),
        ("required_for_valid_evidence", False),
    ],
)
def test_thin_client_records_resource_service_unavailability(
    tmp_path: Path,
    requirement: str,
    evidence_valid: bool,
) -> None:
    profile = ResourceProfile(
        endpoint="unix:///missing/resource.sock",
        behavior="observe_predict_learn",
        update_policy="frozen",
        snapshot="latest_at_run_start",
        telemetry_requirement=requirement,
        latency_bucket_edges_ms=(100,),
    )
    manifest_path = tmp_path / f"{requirement}.json"
    resource_run = ResourceRun.open(
        profile,
        run_id="unavailable-run",
        workspace_scope="repo",
        manifest_path=manifest_path,
    )

    assert resource_run.finalize(workload_status="completed") is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workload_status"] == "completed"
    assert manifest["telemetry_valid"] is False
    assert manifest["evidence_valid"] is evidence_valid
    assert manifest["run_manifest"]["resource_service_status"] == "unavailable"
    assert manifest["run_manifest"]["pinned_snapshot_id"] is None
    assert "run_token" not in json.dumps(manifest)


def test_observing_run_without_traces_is_not_telemetry_valid(
    tmp_path: Path,
) -> None:
    service = ResourceService(
        ObservationStore(tmp_path / "observations.sqlite3"),
        _DirectTransport(
            TelemetryService(
                collector_factory=_FakeCollector,
                state_dir=tmp_path / "telemetry",
            )
        ),
    )
    run = _open_run(service)
    closed = service.dispatch(
        "CloseRun",
        {"run_token": run["run_token"], "workload_status": "failed"},
    )
    assert closed["telemetry_valid"] is False
    assert closed["evidence_valid"] is False
    service.close()


@pytest.mark.skipif(
    os.environ.get("RUN_TOOL_RESOURCE_LIVE") != "1",
    reason="set RUN_TOOL_RESOURCE_LIVE=1 for the root/BCC/Docker service smoke",
)
def test_live_service_chain_produces_one_eligible_observation(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python = repo_root / ".venv" / "bin" / "python"
    telemetry_socket = tmp_path / "telemetry.sock"
    resource_socket = tmp_path / "resource.sock"
    database = tmp_path / "observations.sqlite3"
    container_name = f"resource-service-smoke-{os.getpid()}"
    telemetry_log = tmp_path / "telemetryd.log"
    resource_log = tmp_path / "resource-agentd.log"
    telemetry_proc: subprocess.Popen[str] | None = None
    resource_proc: subprocess.Popen[str] | None = None
    container_id = ""

    def wait_for_socket(path: Path, process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if path.exists():
                return
            if process.poll() is not None:
                raise AssertionError(
                    f"service exited {process.returncode} before creating {path}"
                )
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for service socket {path}")

    def stop_process(
        process: subprocess.Popen[str] | None,
        *,
        privileged: bool = False,
    ) -> None:
        if process is None:
            return
        if privileged:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "/bin/kill",
                    "-TERM",
                    "--",
                    f"-{process.pid}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if privileged:
                subprocess.run(
                    [
                        "sudo",
                        "-n",
                        "/bin/kill",
                        "-KILL",
                        "--",
                        f"-{process.pid}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            else:
                process.kill()
            process.wait(timeout=10)

    try:
        container_id = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "python:3.13-slim-bookworm",
                "sleep",
                "120",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["docker", "exec", container_id, "mkdir", "-p", "/testbed"],
            check=True,
        )
        pythonpath = f"{repo_root / 'src'}:{repo_root}:/usr/lib/python3/dist-packages"
        with telemetry_log.open("w", encoding="utf-8") as telemetry_output:
            telemetry_proc = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "env",
                    f"PYTHONPATH={pythonpath}",
                    str(python),
                    "-m",
                    "tool_resource.telemetryd",
                    "--socket",
                    str(telemetry_socket),
                    "--allowed-uid",
                    str(os.getuid()),
                    "--socket-gid",
                    str(os.getgid()),
                    "--container-runtime",
                    "docker",
                    "--state-dir",
                    str(tmp_path / "telemetry-state"),
                ],
                cwd=repo_root,
                stdout=telemetry_output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            wait_for_socket(telemetry_socket, telemetry_proc)
            with resource_log.open("w", encoding="utf-8") as resource_output:
                resource_proc = subprocess.Popen(
                    [
                        str(python),
                        "-m",
                        "tool_resource.resource_agentd",
                        "--socket",
                        str(resource_socket),
                        "--database",
                        str(database),
                        "--telemetry-socket",
                        str(telemetry_socket),
                    ],
                    cwd=repo_root,
                    env={
                        **os.environ,
                        "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
                    },
                    stdout=resource_output,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                wait_for_socket(resource_socket, resource_proc)

                async def execute() -> tuple[dict[str, Any], dict[str, Any]]:
                    transport = ResourceUnixTransport(resource_socket, timeout_s=30)
                    run = transport.request(
                        "OpenRun",
                        {
                            "run_id": container_name,
                            "workspace_scope": "live-service-smoke",
                            "snapshot": "latest_at_run_start",
                            "latency_bucket_edges_ms": [10, 100, 1000],
                            "update_policy": "causal",
                            "telemetry_requirement": ("required_for_valid_evidence"),
                            "behavior": "observe_predict_learn",
                        },
                    )
                    agent = ContainerAgent(container_id, "docker")
                    await agent.start()
                    try:
                        trace = transport.request(
                            "OpenTrace",
                            {
                                "run_token": run["run_token"],
                                "trace_id": "docker-ebpf",
                                "container_runtime": "docker",
                                "container_id": container_id,
                                "repo_metadata": {},
                                "source_actions": [],
                                "runner_pid": os.getpid(),
                            },
                        )
                        begun = transport.request(
                            "BeginCall",
                            {
                                "trace_token": trace["trace_token"],
                                "call_id": "live-call",
                                "command": (
                                    "python3 -c 'import time; "
                                    "time.sleep(0.05); print(42)'"
                                ),
                                "query_timestamp": time.time(),
                            },
                        )
                        workload_result = await agent.execute(
                            {
                                "tool": "exec",
                                "args": {
                                    "command": (
                                        "python3 -c 'import time; "
                                        "time.sleep(0.05); print(42)'"
                                    ),
                                    "timeout": 30,
                                },
                            },
                            timeout_s=30,
                        )
                        ended = transport.request(
                            "EndCall",
                            {
                                "call_token": begun["call_token"],
                                "workload_result": workload_result,
                                "end_timestamp": time.time(),
                            },
                        )
                        assert ended["workload_result"] == workload_result
                        closed = transport.request(
                            "CloseTrace",
                            {
                                "trace_token": trace["trace_token"],
                                "workload_status": "completed",
                            },
                        )
                        repeated = transport.request(
                            "CloseTrace",
                            {
                                "trace_token": trace["trace_token"],
                                "workload_status": "completed",
                            },
                        )
                        assert repeated == closed
                        closed_run = transport.request(
                            "CloseRun",
                            {
                                "run_token": run["run_token"],
                                "workload_status": "completed",
                            },
                        )
                        return closed, closed_run
                    finally:
                        await agent.stop()

                closed, closed_run = asyncio.run(execute())
                assert closed["telemetry_status"] == "ok"
                assert closed["collection_validity"] == "valid"
                assert closed["artifact"]["cleanup"] == "ok"
                assert len(closed["artifact"]["calls"]) == 1
                assert closed["artifact"]["calls"][0]["eligible_for_kb"] is True
                assert closed_run["promoted_observation_count"] == 1
                assert closed_run["evidence_valid"] is True

                stop_process(resource_proc)
                resource_proc = None
                with sqlite3.connect(database) as connection:
                    assert connection.execute(
                        "SELECT COUNT(*) FROM observations"
                    ).fetchone() == (1,)
                    assert connection.execute(
                        "SELECT COUNT(*) FROM observations WHERE visible=1"
                    ).fetchone() == (1,)
    finally:
        stop_process(resource_proc)
        stop_process(telemetry_proc, privileged=True)
        subprocess.run(
            ["sudo", "-n", "rmdir", str(tmp_path / "telemetry-state")],
            check=False,
            capture_output=True,
            text=True,
        )
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                capture_output=True,
                text=True,
            )
        assert not resource_socket.exists()
        assert not telemetry_socket.exists()
