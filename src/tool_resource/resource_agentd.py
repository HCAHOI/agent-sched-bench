"""Unprivileged resource service: parsing, prediction, SQLite, snapshots."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import queue
import signal
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_resource._uds import (
    DEFAULT_TIMEOUT_S,
    MAX_MESSAGE_BYTES,
    StrictUnixServer,
    process_identity,
    process_identity_is_alive,
    require_fields,
)
from tool_resource.clause_parser import parse_command_clauses
from tool_resource.resource_protocol import (
    RESOURCE_PROTOCOL_VERSION,
    ResourceProtocolError,
)
from tool_resource.runtime_kb import (
    CANONICAL_LATENCY_BUCKETS,
    RAW_ARGV_REPRESENTATION,
    ClauseObservation,
    ClauseResourceKB,
    LatencyBuckets,
)
from tool_resource.store import STORE_SCHEMA_VERSION, ObservationStore
from tool_resource.telemetry_protocol import (
    TelemetryTransport,
    TelemetryUnavailableError,
    TelemetryUnixTransport,
)

CANONICALIZER_VERSION = "mvdan-sh-v1"
DEFAULT_RESULT_TTL_S = 1800.0
_TELEMETRY_QUEUE_SIZE = 256

#: These daemons previously swallowed every failure into a status string, so
#: outages were invisible in the logs and had to be inferred from artifacts.
_LOG = logging.getLogger("tool_resource.resource_agentd")
_LEASE_IDEMPOTENT_OPERATIONS = frozenset({"OpenRun", "OpenTrace", "BeginCall"})


@dataclass
class _Call:
    call_id: str
    command: str
    command_digest: str
    parsed: dict[str, Any]
    query_timestamp: float
    started_monotonic_ns: int
    telemetry_call_token: str | None
    telemetry_status: str
    workload_result: Mapping[str, Any] | str | None = None
    end_timestamp: float | None = None
    end_result: dict[str, Any] | None = None


@dataclass
class _Trace:
    run_token: str
    trace_id: str
    owner_identity: tuple[int, int]
    telemetry_session_token: str | None
    telemetry_status: str
    expected_calls: list[dict[str, str]]
    calls: dict[str, _Call] = field(default_factory=dict)
    call_ids: set[str] = field(default_factory=set)
    source_index: int = 0
    integrity_errors: list[str] = field(default_factory=list)
    ingested_observation_ids: set[str] = field(default_factory=set)
    promotion_eligible: bool = False
    promotion_complete: bool = False
    promoted_observation_count: int = 0
    close_result: dict[str, Any] | None = None
    closing: bool = False
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: threading.Event = field(default_factory=threading.Event)
    telemetry_queue: queue.Queue[Callable[[], None] | None] = field(
        default_factory=lambda: queue.Queue(maxsize=_TELEMETRY_QUEUE_SIZE)
    )
    telemetry_thread: threading.Thread | None = None
    telemetry_ready: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Run:
    run_id: str
    workspace_scope: str
    pinned_snapshot_id: str
    buckets: LatencyBuckets
    update_policy: str
    telemetry_requirement: str
    behavior: str
    owner_identity: tuple[int, int]
    kb: ClauseResourceKB
    trace_tokens: set[str] = field(default_factory=set)
    trace_ids: set[str] = field(default_factory=set)
    aborted: bool = False
    close_result: dict[str, Any] | None = None
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ResourceService:
    """Sole parser, predictor, store writer, and telemetry orchestrator."""

    def __init__(
        self,
        store: ObservationStore,
        telemetry_transport: TelemetryTransport,
        *,
        result_ttl_s: float = DEFAULT_RESULT_TTL_S,
        kb_representation: str = RAW_ARGV_REPRESENTATION,
    ) -> None:
        if not math.isfinite(result_ttl_s) or result_ttl_s <= 0:
            raise ValueError("result_ttl_s must be finite and positive")
        self.store = store
        self.telemetry = telemetry_transport
        self.result_ttl_s = result_ttl_s
        # Validation belongs to the canonical KB; constructing an empty state
        # has no evidence or persistence side effect.
        self.kb_representation = ClauseResourceKB(
            representation=kb_representation
        ).representation
        self._runs: dict[str, _Run] = {}
        self._traces: dict[str, _Trace] = {}
        self._operation_results: dict[
            tuple[int, str], tuple[str, str, dict[str, Any], str]
        ] = {}
        #: (promotion watermark, decoded observations) shared by runs that open
        #: against an unchanged store; see :meth:`_snapshot_observations`.
        self._snapshot_cache: tuple[int, list[ClauseObservation]] | None = None
        self._lock = threading.Lock()

    def dispatch(
        self,
        operation: str,
        payload: Mapping[str, Any],
        request_identity: tuple[int, str] | None = None,
        peer_pid: int | None = None,
    ) -> dict[str, Any]:
        payload_identity = None
        if request_identity is not None:
            payload_identity = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            with self._lock:
                cached = self._operation_results.get(request_identity)
                if cached is not None:
                    cached_operation, cached_payload, result, run_token = cached
                    if (
                        cached_operation != operation
                        or cached_payload != payload_identity
                    ):
                        raise ResourceProtocolError(
                            "request_id was already used for a different request"
                        )
                    self._runs[run_token].last_used = time.monotonic()
                    return dict(result)
        handlers = {
            "Ping": self._ping,
            "Capabilities": self._capabilities,
            "AwaitTraceReady": self._await_trace_ready,
            "BeginCall": self._begin_call,
            "EndCall": self._end_call,
            "RecordSafetyGuardBlock": self._record_safety_guard_block,
            "RecordTraceIntegrityFailure": self._record_trace_integrity_failure,
            "CloseTrace": self._close_trace,
            "AbortTrace": self._abort_trace,
            "CloseRun": self._close_run,
            "AbortRun": self._abort_run,
        }
        if operation == "OpenRun":
            result = self._open_run(
                payload,
                owner_identity=process_identity(
                    os.getpid() if peer_pid is None else peer_pid
                ),
            )
        elif operation == "OpenTrace":
            result = self._open_trace(
                payload,
                owner_identity=process_identity(
                    os.getpid() if peer_pid is None else peer_pid
                ),
            )
        else:
            try:
                handler = handlers[operation]
            except KeyError as exc:
                raise ResourceProtocolError(
                    f"unsupported resource operation {operation!r}"
                ) from exc
            result = handler(payload)
        if (
            payload_identity is not None
            and request_identity is not None
            and operation in _LEASE_IDEMPOTENT_OPERATIONS
        ):
            if operation == "OpenRun":
                run_token = str(result["run_token"])
            elif operation == "OpenTrace":
                run_token = str(payload["run_token"])
            else:
                with self._lock:
                    run_token = self._traces[str(payload["trace_token"])].run_token
            with self._lock:
                self._operation_results[request_identity] = (
                    operation,
                    payload_identity,
                    dict(result),
                    run_token,
                )
        return result

    def _ping(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(payload, required=set(), error_type=ResourceProtocolError)
        return {"protocol_version": RESOURCE_PROTOCOL_VERSION}

    def _capabilities(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(payload, required=set(), error_type=ResourceProtocolError)
        return {
            "prediction_targets": [
                "latency_bucket",
                "peak_cpu_cores_heavy_light",
                "sampled_peak_rss_mb_heavy_light",
                "disk_read_write_bytes_total_heavy_light",
            ],
            "canonicalizer_version": CANONICALIZER_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "raw_events_exposed": False,
        }

    def _snapshot_observations(self, snapshot_id: str) -> list[ClauseObservation]:
        """Decoded observations for a snapshot, shared across runs that see the
        same store contents.

        A snapshot is identified by its promotion watermark, so two snapshots
        with the same watermark contain exactly the same observations. Decoding
        them per run cost a full JSON pass and a private copy of the corpus for
        every run opened against an unchanged store. ``ClauseObservation`` is
        frozen, so the decoded list is safe to share; only the KB built from it
        is per-run mutable state. One watermark is cached because runs march
        forward and an older watermark is not reopened.
        """

        watermark = self.store.require_snapshot(snapshot_id)
        with self._lock:
            cached = self._snapshot_cache
            if cached is not None and cached[0] == watermark:
                return cached[1]
        observations = [
            observation
            for envelope in self.store.observations_for_snapshot(snapshot_id)
            for observation in _clause_observations(envelope)
        ]
        with self._lock:
            self._snapshot_cache = (watermark, observations)
        return observations

    def _open_run(
        self,
        payload: Mapping[str, Any],
        *,
        owner_identity: tuple[int, int],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "run_id",
                "workspace_scope",
                "snapshot",
                "latency_bucket_edges_ms",
                "update_policy",
                "telemetry_requirement",
                "behavior",
            },
            error_type=ResourceProtocolError,
        )
        run_id = _string(payload, "run_id")
        scope = _string(payload, "workspace_scope")
        snapshot = _string(payload, "snapshot")
        edges = payload["latency_bucket_edges_ms"]
        if not isinstance(edges, list):
            raise ResourceProtocolError("latency_bucket_edges_ms must be a list")
        try:
            buckets = LatencyBuckets(tuple(edges))
        except (TypeError, ValueError) as exc:
            raise ResourceProtocolError(str(exc)) from exc
        if buckets != CANONICAL_LATENCY_BUCKETS:
            raise ResourceProtocolError(
                "latency_bucket_edges_ms must match the canonical boundaries"
            )
        update_policy = _choice(payload, "update_policy", {"frozen", "causal"})
        requirement = _choice(
            payload,
            "telemetry_requirement",
            {"best_effort", "required_for_valid_evidence"},
        )
        behavior = _choice(
            payload,
            "behavior",
            {"predict", "observe_predict", "observe_predict_learn"},
        )
        snapshot_id = (
            self.store.create_snapshot()
            if snapshot == "latest_at_run_start"
            else snapshot
        )
        # Building the KB is part of loading the snapshot: a stored observation
        # the KB refuses is a bad snapshot, and this boundary reports those as
        # protocol errors. Only the decode was inside the guard, so a refusal
        # from fit_public or observe_completed_clause escaped as a raw
        # ValueError while an identically-caused decode failure did not.
        try:
            observations = self._snapshot_observations(snapshot_id)
            public = [
                observation
                for observation in observations
                if observation.repo != scope and observation.latency_ms is not None
            ]
            kb = (
                ClauseResourceKB.fit_public(
                    public,
                    representation=self.kb_representation,
                )
                if public
                else ClauseResourceKB(representation=self.kb_representation)
            )
            for observation in observations:
                if observation.repo == scope:
                    kb.observe_completed_clause(observation)
        except ValueError as exc:
            raise ResourceProtocolError(str(exc)) from exc
        run_token = uuid.uuid4().hex
        with self._lock:
            self._runs[run_token] = _Run(
                run_id,
                scope,
                snapshot_id,
                buckets,
                update_policy,
                requirement,
                behavior,
                owner_identity,
                kb,
            )
        return {
            "run_token": run_token,
            "pinned_snapshot_id": snapshot_id,
            "canonicalizer_version": CANONICALIZER_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "capabilities": {
                "latency_bucket_prediction": True,
                "telemetry": "observe" in behavior,
                "causal_updates": update_policy == "causal"
                and behavior.endswith("learn"),
            },
        }

    def _open_trace(
        self,
        payload: Mapping[str, Any],
        *,
        owner_identity: tuple[int, int],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "run_token",
                "trace_id",
                "container_runtime",
                "container_id",
                "repo_metadata",
                "expected_calls",
            },
            error_type=ResourceProtocolError,
        )
        run_token = _string(payload, "run_token")
        run = self._run(run_token)
        trace_id = _string(payload, "trace_id")
        runtime = _string(payload, "container_runtime")
        container_id = _string(payload, "container_id")
        metadata = payload["repo_metadata"]
        expected_calls = payload["expected_calls"]
        if not isinstance(metadata, Mapping):
            raise ResourceProtocolError("repo_metadata must be an object")
        if not isinstance(expected_calls, list) or not all(
            isinstance(call, Mapping)
            and set(call)
            == {"source_tool_call_id", "source_command", "source_tool_result"}
            and all(isinstance(value, str) for value in call.values())
            for call in expected_calls
        ):
            raise ResourceProtocolError(
                "expected_calls must contain normalized call objects"
            )
        telemetry_status = (
            "pending" if "observe" in run.behavior else "not_requested"
        )
        trace_token = uuid.uuid4().hex
        normalized_expected_calls = [
            {str(key): str(value) for key, value in call.items()}
            for call in expected_calls
        ]
        with run.lock:
            if run.close_result is not None:
                raise ResourceProtocolError("run is already closed")
            if trace_id in run.trace_ids:
                raise ResourceProtocolError(f"duplicate trace_id {trace_id!r}")
            trace = _Trace(
                run_token=run_token,
                trace_id=trace_id,
                owner_identity=owner_identity,
                telemetry_session_token=None,
                telemetry_status=telemetry_status,
                expected_calls=normalized_expected_calls,
            )
            with self._lock:
                self._traces[trace_token] = trace
            run.trace_tokens.add(trace_token)
            run.trace_ids.add(trace_id)
            run.last_used = time.monotonic()
        if "observe" in run.behavior:
            self._start_telemetry_worker(trace)
            attach_payload = {
                "run_id": run.run_id,
                "trace_id": trace_id,
                "container_runtime": runtime,
                "container_id": container_id,
                "workspace_scope": run.workspace_scope,
            }

            def attach_target() -> None:
                try:
                    self._attach_target(trace, attach_payload)
                finally:
                    trace.telemetry_ready.set()

            if not self._enqueue_telemetry(
                trace,
                run,
                "trace open",
                attach_target,
            ):
                trace.telemetry_ready.set()
        else:
            trace.telemetry_ready.set()
        return {
            "trace_token": trace_token,
            "telemetry_status": telemetry_status,
        }

    def _await_trace_ready(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"trace_token"},
            error_type=ResourceProtocolError,
        )
        trace = self._trace(payload)
        trace.telemetry_ready.wait()
        with trace.lock:
            return {"telemetry_status": trace.telemetry_status}

    def _start_telemetry_worker(self, trace: _Trace) -> None:
        trace.telemetry_thread = threading.Thread(
            target=self._telemetry_worker,
            args=(trace,),
            name=f"resource-telemetry-{trace.trace_id}",
            daemon=True,
        )
        trace.telemetry_thread.start()

    @staticmethod
    def _telemetry_worker(trace: _Trace) -> None:
        while True:
            operation = trace.telemetry_queue.get()
            try:
                if operation is None:
                    return
                operation()
            finally:
                trace.telemetry_queue.task_done()

    def _enqueue_telemetry(
        self,
        trace: _Trace,
        run: _Run,
        phase: str,
        operation: Callable[[], None],
        *,
        call: _Call | None = None,
    ) -> bool:
        def run_operation() -> None:
            try:
                operation()
            except Exception as exc:  # noqa: BLE001 - workload must remain independent
                self._mark_telemetry_unavailable(
                    trace,
                    run,
                    phase,
                    f"{type(exc).__name__}:{exc}",
                    call=call,
                )

        try:
            trace.telemetry_queue.put_nowait(run_operation)
            return True
        except queue.Full:
            self._mark_telemetry_unavailable(
                trace,
                run,
                phase,
                "queue_full:bounded telemetry queue is full",
                call=call,
            )
            return False

    def _telemetry_request(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        try:
            return self.telemetry.request(
                operation,
                payload,
                request_id=request_id,
            )
        except TelemetryUnavailableError:
            return self.telemetry.request(
                operation,
                payload,
                request_id=request_id,
            )

    @staticmethod
    def _mark_telemetry_unavailable(
        trace: _Trace,
        run: _Run,
        phase: str,
        detail: str,
        *,
        call: _Call | None = None,
    ) -> None:
        status = f"unavailable:{detail}"
        with trace.lock:
            if call is None:
                trace.telemetry_status = status
            else:
                call.telemetry_status = status
        _LOG.warning(
            "telemetry %s failed run_id=%s trace_id=%s call_id=%s status=%s",
            phase,
            run.run_id,
            trace.trace_id,
            call.call_id if call is not None else "-",
            status,
        )

    def _attach_target(
        self,
        trace: _Trace,
        payload: Mapping[str, Any],
    ) -> None:
        result = self._telemetry_request("AttachTarget", payload)
        _require_result_fields(
            result,
            {"telemetry_session_token", "target_status", "resolved_target"},
            "AttachTarget",
        )
        _require_result_fields(
            result["resolved_target"],
            {"init_pid", "cgroup_id", "quota_cores"},
            "AttachTarget.resolved_target",
        )
        with trace.lock:
            trace.telemetry_session_token = _result_string(
                result,
                "telemetry_session_token",
            )
            trace.telemetry_status = str(
                result.get("target_status") or "unavailable"
            )

    def _register_call(
        self,
        trace: _Trace,
        call: _Call,
        source_plan: Mapping[str, Any],
    ) -> None:
        with trace.lock:
            session_token = trace.telemetry_session_token
            trace_status = trace.telemetry_status
        if session_token is None:
            with trace.lock:
                call.telemetry_status = trace_status
            return
        result = self._telemetry_request(
            "RegisterCall",
            {
                "telemetry_session_token": session_token,
                "call_id": call.call_id,
                "command_digest": call.command_digest,
                "call_started_monotonic_ns": call.started_monotonic_ns,
                "static_call_plan": {
                    "canonical_command": call.command,
                    "parsed": call.parsed,
                    **source_plan,
                },
            },
        )
        _require_result_fields(
            result,
            {"telemetry_call_token", "telemetry_status"},
            "RegisterCall",
        )
        with trace.lock:
            call.telemetry_call_token = _result_string(
                result,
                "telemetry_call_token",
            )
            if not call.telemetry_status.startswith("unavailable"):
                call.telemetry_status = str(
                    result.get("telemetry_status") or "registered"
                )

    def _finish_call_telemetry(
        self,
        trace: _Trace,
        call: _Call,
        workload_result: Mapping[str, Any] | None,
        end_timestamp: float,
        ended_monotonic_ns: int,
    ) -> None:
        with trace.lock:
            session_token = trace.telemetry_session_token
            call_token = call.telemetry_call_token
        if session_token is None or call_token is None:
            return
        result = self._telemetry_request(
            "FinishCall",
            {
                "telemetry_session_token": session_token,
                "telemetry_call_token": call_token,
                "workload_result": (
                    None if workload_result is None else dict(workload_result)
                ),
                "end_timestamp": end_timestamp,
                "call_ended_monotonic_ns": ended_monotonic_ns,
            },
        )
        _require_result_fields(
            result,
            {
                "observation_id",
                "finalized_call_observation",
                "telemetry_status",
            },
            "FinishCall",
        )
        if not isinstance(result.get("finalized_call_observation"), Mapping):
            raise ResourceProtocolError(
                "telemetryd returned no finalized observation"
            )
        with trace.lock:
            call.telemetry_status = str(
                result.get("telemetry_status") or "unavailable"
            )

    def _record_guard_telemetry(
        self,
        trace: _Trace,
        call: _Call,
        source_plan: Mapping[str, Any],
        workload_result: str,
        end_timestamp: float,
    ) -> None:
        with trace.lock:
            session_token = trace.telemetry_session_token
            trace_status = trace.telemetry_status
        if session_token is None:
            with trace.lock:
                call.telemetry_status = trace_status
            return
        result = self._telemetry_request(
            "RecordSafetyGuardBlock",
            {
                "telemetry_session_token": session_token,
                "call_id": call.call_id,
                "command_digest": call.command_digest,
                "static_call_plan": {
                    "canonical_command": call.command,
                    "parsed": call.parsed,
                    **source_plan,
                },
                "workload_result": workload_result,
                "end_timestamp": end_timestamp,
            },
        )
        _require_result_fields(
            result,
            {
                "observation_id",
                "finalized_call_observation",
                "telemetry_status",
            },
            "RecordSafetyGuardBlock",
        )
        if not isinstance(result.get("finalized_call_observation"), Mapping):
            raise ResourceProtocolError(
                "telemetryd returned no finalized observation"
            )
        with trace.lock:
            call.telemetry_status = str(
                result.get("telemetry_status") or "unavailable"
            )

    @staticmethod
    def _settle_telemetry(trace: _Trace) -> None:
        thread = trace.telemetry_thread
        if thread is None:
            return
        trace.telemetry_queue.join()
        trace.telemetry_queue.put(None)
        thread.join()
        trace.telemetry_thread = None

    def _begin_call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"trace_token", "call_id", "command", "query_timestamp"},
            error_type=ResourceProtocolError,
        )
        trace = self._trace(payload)
        run = self._run(trace.run_token)
        call_id = _string(payload, "call_id")
        command = _string(payload, "command")
        query_timestamp = _finite_number(payload, "query_timestamp")
        with trace.lock:
            if trace.close_result is not None or trace.closing:
                raise ResourceProtocolError("trace is closing or already closed")
            if call_id in trace.call_ids:
                raise ResourceProtocolError(f"duplicate call_id {call_id!r}")
            parsed = parse_command_clauses(command)
            digest = hashlib.sha256(command.encode()).hexdigest()
            source_plan = self._source_plan(trace)
            prediction, prediction_error = self._predict(
                run, command, parsed, query_timestamp
            )
            resource_predictions = self._predict_resources(
                run, command, parsed, query_timestamp
            )
            call_token = uuid.uuid4().hex
            call = _Call(
                call_id=call_id,
                command=command,
                command_digest=digest,
                parsed=parsed,
                query_timestamp=query_timestamp,
                started_monotonic_ns=time.monotonic_ns(),
                telemetry_call_token=None,
                telemetry_status=trace.telemetry_status,
            )
            trace.calls[call_token] = call
            trace.call_ids.add(call_id)
            trace.last_used = time.monotonic()
            observe = trace.telemetry_thread is not None
        if observe:
            self._enqueue_telemetry(
                trace,
                run,
                "call registration",
                lambda: self._register_call(trace, call, source_plan),
                call=call,
            )
        prediction_payload = _prediction_payload(prediction)
        selected = prediction_payload.get("prediction") or {}
        return {
            "call_token": call_token,
            "prediction": prediction_payload,
            "resource_classifications": resource_predictions,
            "probability_by_bucket": selected.get("probability_by_bucket"),
            "selected_scope": selected.get("scope"),
            "fallback_path": selected.get("fallback_path"),
            "fallback_reason": (
                prediction_error or prediction_payload.get("unavailable_reason")
            ),
            "evidence_count": selected.get("evidence_count", 0),
            "evidence_recency": None,
            "pinned_snapshot_id": run.pinned_snapshot_id,
            "canonicalizer_version": CANONICALIZER_VERSION,
            "telemetry_status": call.telemetry_status,
        }

    def _end_call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"call_token", "workload_result", "end_timestamp"},
            error_type=ResourceProtocolError,
        )
        trace, call = self._call(payload)
        run = self._run(trace.run_token)
        workload_result = payload["workload_result"]
        if workload_result is not None and not isinstance(workload_result, Mapping):
            raise ResourceProtocolError("workload_result must be an object or null")
        end_timestamp = _finite_number(payload, "end_timestamp")
        ended_monotonic_ns = time.monotonic_ns()
        with trace.lock:
            if call.end_result is not None:
                return call.end_result
            if trace.closing:
                raise ResourceProtocolError("trace is closing")
            call.workload_result = workload_result
            call.end_timestamp = end_timestamp
            observation = _provisional_observation(
                call.call_id,
                call.telemetry_status,
                observed=trace.telemetry_thread is not None,
            )
            call.end_result = {
                "workload_result": workload_result,
                "finalized_call_observation": observation,
                "telemetry_status": call.telemetry_status,
                "ingest_status": "pending_trace_finalization",
                "rejection_reasons": [],
            }
            trace.last_used = time.monotonic()
            observe = trace.telemetry_thread is not None
            end_result = call.end_result
        if observe:
            self._enqueue_telemetry(
                trace,
                run,
                "call finish",
                lambda: self._finish_call_telemetry(
                    trace,
                    call,
                    workload_result,
                    end_timestamp,
                    ended_monotonic_ns,
                ),
                call=call,
            )
        return end_result

    def _record_safety_guard_block(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "trace_token",
                "call_id",
                "command",
                "workload_result",
                "end_timestamp",
            },
            error_type=ResourceProtocolError,
        )
        trace = self._trace(payload)
        run = self._run(trace.run_token)
        call_id = _string(payload, "call_id")
        command = _string(payload, "command")
        workload_result = _string(payload, "workload_result", allow_empty=True)
        end_timestamp = _finite_number(payload, "end_timestamp")
        with trace.lock:
            if trace.close_result is not None or trace.closing:
                raise ResourceProtocolError("trace is closing or already closed")
            if call_id in trace.call_ids:
                raise ResourceProtocolError(f"duplicate call_id {call_id!r}")
            parsed = parse_command_clauses(command)
            digest = hashlib.sha256(command.encode()).hexdigest()
            prediction, prediction_error = self._predict(
                run,
                command,
                parsed,
                end_timestamp,
            )
            source_plan = self._source_plan(trace)
            call_token = uuid.uuid4().hex
            call = _Call(
                call_id=call_id,
                command=command,
                command_digest=digest,
                parsed=parsed,
                query_timestamp=end_timestamp,
                started_monotonic_ns=time.monotonic_ns(),
                telemetry_call_token=None,
                telemetry_status=trace.telemetry_status,
                workload_result=workload_result,
                end_timestamp=end_timestamp,
            )
            trace.calls[call_token] = call
            trace.call_ids.add(call_id)
            trace.last_used = time.monotonic()
            observe = trace.telemetry_thread is not None
        if observe:
            self._enqueue_telemetry(
                trace,
                run,
                "safety-guard record",
                lambda: self._record_guard_telemetry(
                    trace,
                    call,
                    source_plan,
                    workload_result,
                    end_timestamp,
                ),
                call=call,
            )
        return {
            "workload_result": workload_result,
            "prediction": _prediction_payload(prediction),
            "prediction_error": prediction_error,
            "finalized_call_observation": _provisional_observation(
                call_id,
                call.telemetry_status,
                observed=observe,
            ),
            "telemetry_status": call.telemetry_status,
            "ingest_status": "pending_trace_finalization",
        }

    def _record_trace_integrity_failure(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={"trace_token", "message"},
            error_type=ResourceProtocolError,
        )
        trace = self._trace(payload)
        message = _string(payload, "message")
        with trace.lock:
            if trace.close_result is not None:
                raise ResourceProtocolError("trace is already closed")
            if message not in trace.integrity_errors:
                trace.integrity_errors.append(message)
        return {"recorded": True}

    def _close_trace(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"trace_token", "workload_status"},
            error_type=ResourceProtocolError,
        )
        trace = self._trace(payload)
        workload_status = _choice(
            payload,
            "workload_status",
            {"completed", "failed", "incomplete"},
        )
        run = self._run(trace.run_token)
        with trace.lock:
            if trace.close_result is not None:
                self._promote_trace(trace)
                return trace.close_result
            owns_close = not trace.closing
            trace.closing = True
        if not owns_close:
            trace.closed.wait()
            with trace.lock:
                if trace.close_result is None:
                    raise ResourceProtocolError("trace settlement did not complete")
                self._promote_trace(trace)
                return trace.close_result
        try:
            self._settle_telemetry(trace)
            return self._close_trace_settled(trace, run, workload_status)
        finally:
            trace.closed.set()

    def _close_trace_settled(
        self,
        trace: _Trace,
        run: _Run,
        workload_status: str,
    ) -> dict[str, Any]:
        with trace.lock:
            if trace.close_result is not None:
                self._promote_trace(trace)
                return trace.close_result
            if trace.telemetry_session_token is None:
                trace.close_result = (
                    _not_requested_trace_result(trace, run, workload_status)
                    if "observe" not in run.behavior
                    else _unavailable_trace_result(
                        trace,
                        run,
                        "telemetryd was unavailable at trace open",
                        workload_status,
                    )
                )
                return trace.close_result
            calls: list[dict[str, Any]] = []
            ingested_envelopes: list[dict[str, Any]] = []
            try:
                finalized = self._telemetry_request(
                    "FinalizeSession",
                    {
                        "telemetry_session_token": trace.telemetry_session_token,
                        "workload_status": workload_status,
                        "integrity_errors": list(trace.integrity_errors),
                    },
                )
                summary = finalized.get("session_summary")
                observation_ids = finalized.get("observation_ids")
                _require_result_fields(
                    finalized,
                    {"session_summary", "observation_ids"},
                    "FinalizeSession",
                )
                if not isinstance(summary, Mapping) or not isinstance(
                    observation_ids, list
                ):
                    raise ResourceProtocolError(
                        "telemetryd returned an invalid session finalization"
                    )
                _require_result_fields(
                    summary,
                    {
                        "collector_health",
                        "formal_completeness",
                        "collection_validity",
                        "cleanup_status",
                        "eligible_call_count",
                        "withheld_call_count",
                        "loss_counters",
                        "errors",
                    },
                    "FinalizeSession.session_summary",
                )
                fetched_observations: list[Mapping[str, Any]] = []
                for observation_id in observation_ids:
                    if not isinstance(observation_id, str):
                        raise ResourceProtocolError("invalid observation_id")
                    fetched = self._telemetry_request(
                        "FetchFinalizedObservation",
                        {
                            "telemetry_session_token": trace.telemetry_session_token,
                            "observation_id": observation_id,
                        },
                    )
                    _require_result_fields(
                        fetched,
                        {"finalized_call_observation"},
                        "FetchFinalizedObservation",
                    )
                    raw_observation = fetched.get("finalized_call_observation")
                    if not isinstance(raw_observation, Mapping):
                        raise ResourceProtocolError(
                            "telemetryd returned an invalid observation"
                        )
                    fetched_observations.append(raw_observation)
                for raw_observation in fetched_observations:
                    call, envelope = self._ingest_observation(
                        run,
                        trace,
                        raw_observation,
                    )
                    calls.append(call)
                    ingested_envelopes.append(envelope)
                    if envelope["ingest_eligible"] is True:
                        trace.ingested_observation_ids.add(
                            str(envelope["observation_id"])
                        )
                settled_by_id = {
                    str(call.get("tool_call_id")): call
                    for call in calls
                    if call.get("tool_call_id")
                }
                local_call_ids = {call.call_id for call in trace.calls.values()}
                ordered_calls: list[dict[str, Any]] = []
                for local_call in trace.calls.values():
                    settled_call = settled_by_id.get(local_call.call_id)
                    if settled_call is not None:
                        ordered_calls.append(settled_call)
                        continue
                    if not local_call.telemetry_status.startswith("unavailable"):
                        local_call.telemetry_status = (
                            "unavailable:missing_finalized_observation"
                        )
                    ordered_calls.append(
                        {
                            "version": 1,
                            "tool_call_id": local_call.call_id,
                            "command": local_call.command,
                            "telemetry_status": local_call.telemetry_status,
                            "telemetry_quality": "unavailable",
                            "eligible_for_kb": False,
                            "invalid_reasons": [
                                {
                                    "kind": "service_unavailable",
                                    "detail": local_call.telemetry_status,
                                }
                            ],
                            "clauses": [],
                        }
                    )
                ordered_calls.extend(
                    call
                    for call in calls
                    if call.get("tool_call_id") not in local_call_ids
                )
                calls = ordered_calls
                for envelope in ingested_envelopes:
                    acknowledged = self._telemetry_request(
                        "AcknowledgeObservation",
                        {
                            "telemetry_session_token": trace.telemetry_session_token,
                            "observation_id": envelope["observation_id"],
                        },
                    )
                    _require_result_fields(
                        acknowledged,
                        {"acknowledged"},
                        "AcknowledgeObservation",
                    )
                    if acknowledged["acknowledged"] is not True:
                        raise ResourceProtocolError(
                            "telemetryd did not acknowledge the observation"
                        )
                trace.close_result = _trace_result(
                    run,
                    trace,
                    summary,
                    calls,
                    workload_status,
                )
            except Exception as exc:  # noqa: BLE001 - workload already completed
                trace.promotion_eligible = bool(trace.ingested_observation_ids)
                trace.close_result = _unavailable_trace_result(
                    trace,
                    run,
                    f"telemetry finalization failed: {type(exc).__name__}: {exc}",
                    workload_status,
                    settled_calls=calls,
                )
            if run.update_policy == "causal":
                with run.lock:
                    for envelope in ingested_envelopes:
                        if envelope["ingest_eligible"] is not True:
                            continue
                        for observation in _clause_observations(envelope):
                            run.kb.observe_completed_clause(observation)
            self._promote_trace(trace)
            trace.last_used = time.monotonic()
        with run.lock:
            run.last_used = time.monotonic()
        return trace.close_result

    def _abort_trace(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"trace_token", "reason"},
            error_type=ResourceProtocolError,
        )
        trace = self._trace(payload)
        reason = _string(payload, "reason")
        run = self._run(trace.run_token)
        with trace.lock:
            if trace.close_result is not None:
                self._promote_trace(trace)
                return {"aborted": True, **trace.close_result}
            owns_close = not trace.closing
            trace.closing = True
        if not owns_close:
            trace.closed.wait()
            with trace.lock:
                if trace.close_result is None:
                    raise ResourceProtocolError("trace settlement did not complete")
                self._promote_trace(trace)
                return {"aborted": True, **trace.close_result}
        try:
            self._settle_telemetry(trace)
            return self._abort_trace_settled(trace, run, reason)
        finally:
            trace.closed.set()

    def _abort_trace_settled(
        self,
        trace: _Trace,
        run: _Run,
        reason: str,
    ) -> dict[str, Any]:
        with trace.lock:
            if trace.close_result is None and trace.telemetry_session_token is not None:
                with suppress(Exception):
                    self._telemetry_request(
                        "AbortSession",
                        {
                            "telemetry_session_token": trace.telemetry_session_token,
                            "reason": reason,
                        },
                    )
            if trace.close_result is None:
                trace.close_result = _unavailable_trace_result(
                    trace,
                    run,
                    reason,
                    "incomplete",
                )
            self._promote_trace(trace)
        return {"aborted": True, **trace.close_result}

    def _promote_trace(self, trace: _Trace) -> None:
        if not trace.promotion_eligible or trace.promotion_complete:
            return
        expected_count = len(trace.ingested_observation_ids)
        promoted_count = self.store.promote_observations(
            set(trace.ingested_observation_ids)
        )
        if promoted_count != expected_count:
            raise ResourceProtocolError(
                f"store promoted {promoted_count} of {expected_count} observations"
            )
        trace.promoted_observation_count = promoted_count
        trace.promotion_complete = True

    def _close_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"run_token", "workload_status"},
            error_type=ResourceProtocolError,
        )
        run_token = _string(payload, "run_token")
        workload_status = _choice(
            payload,
            "workload_status",
            {"completed", "failed", "incomplete"},
        )
        run = self._run(run_token)
        for trace_token in list(run.trace_tokens):
            trace = self._traces.get(trace_token)
            if trace is not None:
                with trace.lock:
                    self._promote_trace(trace)
        with run.lock:
            if run.close_result is not None:
                return run.close_result
            open_traces = [
                token
                for token in run.trace_tokens
                if self._traces.get(token) is not None
                and self._traces[token].close_result is None
            ]
            if open_traces:
                raise ResourceProtocolError("cannot close a run with open traces")
            incomplete_promotions = [
                token
                for token in run.trace_tokens
                if token in self._traces
                and self._traces[token].promotion_eligible
                and not self._traces[token].promotion_complete
            ]
            if incomplete_promotions:
                raise ResourceProtocolError(
                    "cannot close a run with unsettled observation promotion"
                )
            trace_results = [
                self._traces[token].close_result
                for token in run.trace_tokens
                if token in self._traces
            ]
            promoted_observation_count = sum(
                self._traces[token].promoted_observation_count
                for token in run.trace_tokens
                if token in self._traces
                and self._traces[token].promotion_complete
            )
            resulting_snapshot = self.store.create_snapshot()
            telemetry_valid = "observe" not in run.behavior or (
                bool(trace_results)
                and all(
                    isinstance(result, Mapping)
                    and result.get("telemetry_status") == "ok"
                    for result in trace_results
                )
            )
            evidence_valid = not run.aborted and (
                telemetry_valid
                or run.telemetry_requirement == "best_effort"
                or "observe" not in run.behavior
            )
            run.close_result = {
                "workload_status": workload_status,
                "telemetry_valid": telemetry_valid,
                "evidence_valid": evidence_valid,
                "promoted_observation_count": promoted_observation_count,
                "run_manifest": {
                    "run_id": run.run_id,
                    "workspace_scope": run.workspace_scope,
                    "pinned_snapshot_id": run.pinned_snapshot_id,
                    "resulting_snapshot_id": resulting_snapshot,
                    "canonicalizer_version": CANONICALIZER_VERSION,
                    "kb_canonicalizer_version": run.kb.canonicalizer_version,
                    "store_schema_version": STORE_SCHEMA_VERSION,
                    "latency_bucket_edges_ms": list(run.buckets.edges_ms),
                    "update_policy": run.update_policy,
                    "telemetry_requirement": run.telemetry_requirement,
                    "behavior": run.behavior,
                },
            }
            run.last_used = time.monotonic()
            return run.close_result

    def _abort_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"run_token", "reason"},
            error_type=ResourceProtocolError,
        )
        run_token = _string(payload, "run_token")
        run = self._run(run_token)
        reason = _string(payload, "reason")
        with run.lock:
            run.aborted = True
        for trace_token in list(run.trace_tokens):
            trace = self._traces.get(trace_token)
            if trace is not None and trace.close_result is None:
                with suppress(Exception):
                    self._abort_trace({"trace_token": trace_token, "reason": reason})
        return self._close_run(
            {"run_token": run_token, "workload_status": "incomplete"}
        )

    def _predict(
        self,
        run: _Run,
        command: str,
        parsed: Mapping[str, Any],
        query_timestamp: float,
    ) -> tuple[Any | None, str | None]:
        try:
            with run.lock:
                prediction = run.kb.predict_command_latency_bucket_from_clauses(
                    run.workspace_scope,
                    parsed["clauses"],
                    query_timestamp,
                    run.buckets,
                    command=command,
                    parse_failed=bool(parsed["parse_failed"]),
                )
            return prediction, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _predict_resources(
        self,
        run: _Run,
        command: str,
        parsed: Mapping[str, Any],
        query_timestamp: float,
    ) -> dict[str, Any]:
        clauses = list(parsed["clauses"])
        reason = None
        if parsed["parse_failed"]:
            reason = "parse_failed"
        elif len(clauses) != 1:
            reason = "compound_command_uncomposed"
        if reason is not None:
            return {
                "command": command,
                "clause_bins": [str(clause["bin"]) for clause in clauses],
                "classifications": {},
                "unavailable_reason": reason,
            }
        clause = clauses[0]
        try:
            with run.lock:
                predictions = run.kb.predict_clause_resource_classes(
                    run.workspace_scope,
                    str(clause["bin"]),
                    tuple(clause["argv"]),
                    ts_start=query_timestamp,
                )
            return {
                "command": command,
                "clause_bins": [str(clause["bin"])],
                "classifications": {
                    resource: (
                        None if prediction is None else dataclasses.asdict(prediction)
                    )
                    for resource, prediction in predictions.items()
                },
                "unavailable_reason": None,
            }
        except Exception as exc:  # noqa: BLE001 - latency prediction remains usable
            return {
                "command": command,
                "clause_bins": [str(clause["bin"])],
                "classifications": {},
                "unavailable_reason": f"{type(exc).__name__}: {exc}",
            }

    def _ingest_observation(
        self,
        run: _Run,
        trace: _Trace,
        observation: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation_id = observation.get("observation_id")
        _require_result_fields(
            observation,
            {
                "observation_id",
                "run_id",
                "trace_id",
                "call_id",
                "command_digest",
                "observation_interval",
                "clauses",
                "attribution_valid",
                "telemetry_eligible",
                "invalid_reasons",
                "loss_counters",
                "collector_health",
                "cleanup_status",
                "formal_completeness",
            },
            "FinalizedCallObservation",
        )
        if not isinstance(observation_id, str) or not observation_id:
            raise ResourceProtocolError("telemetryd returned an invalid observation_id")
        call_id = observation.get("call_id")
        matching = [call for call in trace.calls.values() if call.call_id == call_id]
        reasons = list(observation.get("invalid_reasons") or [])
        telemetry_eligible = observation.get("telemetry_eligible") is True
        if len(matching) != 1:
            call = None
            reasons.append(
                {"kind": "call_identity", "detail": "call identity is not unique"}
            )
        else:
            call = matching[0]
        if (
            observation.get("run_id") != run.run_id
            or observation.get("trace_id") != trace.trace_id
        ):
            reasons.append(
                {"kind": "scope", "detail": "run or trace identity mismatch"}
            )
        if (
            call is not None
            and observation.get("command_digest") != call.command_digest
        ):
            reasons.append(
                {"kind": "command_identity", "detail": "command digest mismatch"}
            )
        if call is not None and call.telemetry_status.startswith("unavailable"):
            reasons.append(
                {
                    "kind": "service_unavailable",
                    "detail": "resource-agentd did not confirm the call telemetry RPC",
                }
            )
        if call is not None and call.workload_result is None:
            reasons.append(
                {"kind": "workload_incomplete", "detail": "call did not complete"}
            )
        if call is not None and call.parsed.get("parse_failed") is True:
            reasons.append(
                {"kind": "canonicalization", "detail": "command parse failed"}
            )
        if trace.integrity_errors:
            reasons.append(
                {
                    "kind": "formal_mapping",
                    "detail": "trace integrity is incomplete",
                }
            )
        clauses = observation.get("clauses")
        if not isinstance(clauses, list) or not any(
            isinstance(clause, Mapping)
            and isinstance(clause.get("availability"), Mapping)
            and clause["availability"].get("latency") == "ok"
            for clause in clauses
        ):
            reasons.append(
                {"kind": "target_unavailable", "detail": "latency is unavailable"}
            )
        if call is not None and isinstance(clauses, list):
            expected_clause_identities = Counter(
                (
                    str(clause.get("bin") or ""),
                    tuple(str(value) for value in clause.get("argv", [])),
                )
                for clause in call.parsed.get("clauses", [])
                if isinstance(clause, Mapping) and isinstance(clause.get("argv"), list)
            )
            observed_clause_identities = Counter(
                (
                    str(clause.get("bin") or ""),
                    tuple(str(value) for value in clause.get("argv", [])),
                )
                for clause in clauses
                if isinstance(clause, Mapping) and isinstance(clause.get("argv"), list)
            )
            # A loop body is one static clause that executes once per iteration,
            # so telemetry legitimately reports its identity several times.
            # Multiplicity is tolerated only for those clauses; every other
            # identity must still be present in the static plan.
            loop_identities = {
                (
                    str(clause.get("bin") or ""),
                    tuple(str(value) for value in clause.get("argv", [])),
                )
                for clause in call.parsed.get("clauses", [])
                if isinstance(clause, Mapping)
                and isinstance(clause.get("argv"), list)
                and clause.get("in_loop") is True
            }
            unexpected = set(
                observed_clause_identities - expected_clause_identities
            )
            if (
                sum(observed_clause_identities.values()) != len(clauses)
                or unexpected - loop_identities
            ):
                reasons.append(
                    {
                        "kind": "canonicalization",
                        "detail": "telemetry clauses differ from the static plan",
                    }
                )
        if not run.behavior.endswith("learn"):
            reasons.append(
                {"kind": "learning_disabled", "detail": "profile does not learn"}
            )
        ingest_eligible = telemetry_eligible and not reasons
        interval = observation.get("observation_interval")
        if not _valid_interval(interval):
            interval = {"start": None, "end": None}
            reasons.append(
                {"kind": "interval", "detail": "observation interval is invalid"}
            )
            ingest_eligible = False
        envelope = {
            "observation_id": observation_id,
            "run_id": run.run_id,
            "trace_id": trace.trace_id,
            "call_id": str(call_id or ""),
            "workspace_scope": run.workspace_scope,
            "canonicalizer_version": CANONICALIZER_VERSION,
            "command_digest": (
                call.command_digest
                if call is not None
                else str(observation.get("command_digest") or "")
            ),
            "canonical_command": {
                "digest": (
                    call.command_digest
                    if call is not None
                    else str(observation.get("command_digest") or "")
                ),
                "clauses": (call.parsed.get("clauses", []) if call is not None else []),
            },
            "observation_interval": {
                "start": interval.get("start"),
                "end": interval.get("end"),
            },
            "normalized_measurements": list(clauses or []),
            "telemetry_eligible": telemetry_eligible,
            "ingest_eligible": ingest_eligible,
            "rejection_reasons": reasons,
            "provenance": {
                "collector_health": observation.get("collector_health"),
                "cleanup_status": observation.get("cleanup_status"),
                "formal_completeness": observation.get("formal_completeness"),
                "loss_counters": observation.get("loss_counters"),
            },
        }
        inserted, sequence = self.store.insert_observation(envelope)
        call_payload = {
            "version": 1,
            "tool_call_id": str(call_id or ""),
            "command": call.command if call is not None else "",
            "telemetry_status": (
                call.telemetry_status if call is not None else "unavailable"
            ),
            "telemetry_quality": (
                "ok" if observation.get("attribution_valid") is True else "unavailable"
            ),
            "eligible_for_kb": ingest_eligible,
            "invalid_reasons": reasons,
            "clauses": list(clauses or []),
            "observation_id": envelope["observation_id"],
            "ingest_status": "inserted" if inserted else "duplicate",
            "ingestion_sequence": sequence,
        }
        return call_payload, envelope

    def expire(self) -> None:
        cutoff = time.monotonic() - self.result_ttl_s
        with self._lock:
            runs = list(self._runs.items())
            traces = list(self._traces.items())
        for token, trace in traces:
            if (
                trace.close_result is not None
                or process_identity_is_alive(trace.owner_identity)
            ):
                continue
            run = self._runs.get(trace.run_token)
            if run is None:
                continue
            _LOG.warning(
                "resource trace owner exited run_id=%s trace_id=%s "
                "owner_pid=%d",
                run.run_id,
                trace.trace_id,
                trace.owner_identity[0],
            )
            with suppress(Exception):
                self._abort_trace(
                    {
                        "trace_token": token,
                        "reason": "resource trace client exited",
                    }
                )
        for token, run in runs:
            owner_alive = process_identity_is_alive(run.owner_identity)
            if owner_alive and (
                run.close_result is None or run.last_used >= cutoff
            ):
                continue
            if owner_alive:
                self._remove_run(token)
                continue
            _LOG.warning(
                "resource run owner exited run_id=%s run_token=%s "
                "owner_pid=%d trace_count=%d",
                run.run_id,
                token,
                run.owner_identity[0],
                len(run.trace_tokens),
            )
            try:
                if run.close_result is None:
                    self._abort_run(
                        {
                            "run_token": token,
                            "reason": "resource client exited",
                        }
                    )
            except Exception:
                continue
            self._remove_run(token)

    def _remove_run(self, token: str) -> None:
        with self._lock:
            run = self._runs.pop(token, None)
            if run is None:
                return
            for trace_token in run.trace_tokens:
                self._traces.pop(trace_token, None)
            self._operation_results = {
                identity: record
                for identity, record in self._operation_results.items()
                if record[3] != token
            }

    def close(self) -> None:
        with self._lock:
            tokens = list(self._runs)
        for token in tokens:
            with suppress(Exception):
                self._abort_run(
                    {"run_token": token, "reason": "resource-agentd shutdown"}
                )
        with self._lock:
            self._operation_results.clear()
        self.store.close()

    def _run(self, token: str) -> _Run:
        with self._lock:
            try:
                return self._runs[token]
            except KeyError as exc:
                _LOG.warning(
                    "unknown resource run run_token=%s; it was closed or owner-exited",
                    token,
                )
                raise ResourceProtocolError("unknown resource run") from exc

    def _trace(self, payload: Mapping[str, Any]) -> _Trace:
        token = _string(payload, "trace_token")
        with self._lock:
            try:
                return self._traces[token]
            except KeyError as exc:
                raise ResourceProtocolError("unknown resource trace") from exc

    def _call(self, payload: Mapping[str, Any]) -> tuple[_Trace, _Call]:
        token = _string(payload, "call_token")
        with self._lock:
            traces = list(self._traces.values())
        matches = [
            (trace, trace.calls[token]) for trace in traces if token in trace.calls
        ]
        if len(matches) != 1:
            raise ResourceProtocolError("unknown resource call")
        return matches[0]

    @staticmethod
    def _source_plan(trace: _Trace) -> dict[str, str]:
        expected = (
            trace.expected_calls[trace.source_index]
            if trace.source_index < len(trace.expected_calls)
            else None
        )
        trace.source_index += 1
        if expected is None:
            return {
                "source_tool_call_id": "",
                "source_command": "",
                "source_tool_result": "",
            }
        return dict(expected)


def _string(
    payload: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ResourceProtocolError(f"{name} must be a string")
    return value


def _choice(payload: Mapping[str, Any], name: str, allowed: set[str]) -> str:
    value = _string(payload, name)
    if value not in allowed:
        raise ResourceProtocolError(f"invalid {name} {value!r}")
    return value


def _finite_number(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ResourceProtocolError(f"{name} must be finite")
    return float(value)


def _valid_interval(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    start = value.get("start")
    end = value.get("end")
    return (
        isinstance(start, (int, float))
        and not isinstance(start, bool)
        and math.isfinite(start)
        and isinstance(end, (int, float))
        and not isinstance(end, bool)
        and math.isfinite(end)
        and start <= end
    )


def _result_string(result: Mapping[str, Any], name: str) -> str:
    value = result.get(name)
    if not isinstance(value, str) or not value:
        raise ResourceProtocolError(f"response has invalid {name}")
    return value


def _require_result_fields(
    value: Any,
    expected: set[str],
    operation: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ResourceProtocolError(
            f"telemetryd {operation} response fields are invalid"
        )


def _prediction_payload(prediction: Any | None) -> dict[str, Any]:
    return {} if prediction is None else dataclasses.asdict(prediction)


def _provisional_observation(
    call_id: str,
    telemetry_status: str,
    *,
    observed: bool,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "telemetry_eligible": False,
        "telemetry_status": telemetry_status,
        "invalid_reasons": [
            {
                "kind": "telemetry_pending" if observed else "telemetry_not_requested",
                "detail": (
                    "telemetry settles after workload completion"
                    if observed
                    else "profile does not observe"
                ),
            }
        ],
        "clauses": [],
        "session_finalization": "pending" if observed else "not_requested",
    }


def _clause_observations(
    envelope: Mapping[str, Any],
) -> list[ClauseObservation]:
    workspace_scope = envelope.get("workspace_scope")
    interval = envelope.get("observation_interval")
    rows = envelope.get("normalized_measurements")
    if (
        not isinstance(workspace_scope, str)
        or not workspace_scope
        or not isinstance(interval, Mapping)
        or not isinstance(rows, list)
    ):
        return []
    observations: list[ClauseObservation] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        availability = row.get("availability")
        if not isinstance(availability, Mapping) or availability.get("latency") != "ok":
            continue
        # A downstream pipeline member blocks on its upstream, so its wall time
        # measures the upstream's work, not its own -- `tail -30` has been seen
        # credited with 557 s. A pipeline HEAD (position 0) is self-determined
        # and is kept. The artifact still records these rows; only KB evidence
        # skips them.
        try:
            pipeline_position = int(row.get("pipeline_position", -1))
        except (TypeError, ValueError):
            continue
        if pipeline_position > 0:
            continue
        try:
            disk_io = row.get("disk_io")
            disk_total = (
                disk_io.get("read_write_bytes_total")
                if isinstance(disk_io, Mapping)
                else None
            )
            observations.append(
                ClauseObservation(
                    repo=workspace_scope,
                    bin=str(row["bin"]),
                    argv=tuple(str(value) for value in row["argv"]),
                    ts_start=float(row["ts_start"]),
                    ts_end=float(row["ts_end"]),
                    latency_ms=float(row["latency_ms"]),
                    peak_cpu_cores=(
                        None
                        if row.get("peak_cpu_cores") is None
                        else float(row["peak_cpu_cores"])
                    ),
                    sampled_peak_rss_mb=(
                        None
                        if row.get("sampled_peak_rss_mb") is None
                        else float(row["sampled_peak_rss_mb"])
                    ),
                    disk_read_write_bytes_total=(
                        None if disk_total is None else float(disk_total)
                    ),
                    impute_short_null_resources_as_light=True,
                    cpu_ns_cumulative=(
                        None
                        if row.get("cpu_ns_cumulative") is None
                        else int(row["cpu_ns_cumulative"])
                    ),
                    in_loop=bool(row.get("in_loop", False)),
                    in_pipe=bool(row.get("in_pipe", False)),
                    in_subst=bool(row.get("in_subst", False)),
                    pipeline_position=int(row.get("pipeline_position", -1)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return observations


def _trace_result(
    run: _Run,
    trace: _Trace,
    summary: Mapping[str, Any],
    calls: list[dict[str, Any]],
    workload_status: str,
) -> dict[str, Any]:
    promotion_eligible = (
        summary.get("collector_health") == "healthy"
        and summary.get("collection_validity") == "valid"
        and summary.get("cleanup_status") == "ok"
    )
    telemetry_ok = promotion_eligible and all(
        not call.telemetry_status.startswith("unavailable")
        for call in trace.calls.values()
    )
    trace.promotion_eligible = promotion_eligible
    eligible = sum(call["eligible_for_kb"] is True for call in calls)
    formal_completeness = (
        "unavailable"
        if not telemetry_ok
        else (
            "partial"
            if summary.get("formal_completeness") == "partial" or eligible != len(calls)
            else "complete"
        )
    )
    artifact = {
        "version": 1,
        "mode": "resource",
        "status_model": "workload_telemetry_formal_v1",
        "run_id": run.run_id,
        "trace_id": trace.trace_id,
        "calls": calls,
        "workload_execution": workload_status,
        "telemetry_quality": "ok" if telemetry_ok else "unavailable",
        "formal_completeness": formal_completeness,
        "collection_validity": "valid" if telemetry_ok else "invalid",
        "cleanup": summary.get("cleanup_status"),
        "call_coverage": {
            "total_call_count": len(calls),
            "eligible_call_count": eligible,
            "withheld_call_count": len(calls) - eligible,
            "eligible_fraction": eligible / len(calls) if calls else 1.0,
        },
        "session_summary": dict(summary),
        "pinned_snapshot_id": run.pinned_snapshot_id,
        "canonicalizer_version": CANONICALIZER_VERSION,
    }
    return {
        "telemetry_status": "ok" if telemetry_ok else "unavailable",
        "formal_completeness": artifact["formal_completeness"],
        "collection_validity": artifact["collection_validity"],
        "call_coverage": artifact["call_coverage"],
        "artifact": artifact,
        "errors": list(summary.get("errors") or []),
    }


def _unavailable_trace_result(
    trace: _Trace,
    run: _Run,
    error: str,
    workload_status: str,
    *,
    settled_calls: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    call_ids = ",".join(sorted(call.call_id for call in trace.calls.values())) or "-"
    _LOG.warning(
        "telemetry trace unavailable run_id=%s trace_id=%s call_id=%s error=%s",
        run.run_id,
        trace.trace_id,
        call_ids,
        error,
    )
    settled_by_id = {
        str(call.get("tool_call_id")): dict(call)
        for call in settled_calls
        if call.get("tool_call_id")
    }
    calls = []
    for call in trace.calls.values():
        settled = settled_by_id.get(call.call_id)
        calls.append(
            settled
            if settled is not None
            else {
                "version": 1,
                "tool_call_id": call.call_id,
                "command": call.command,
                "telemetry_status": call.telemetry_status,
                "telemetry_quality": "unavailable",
                "eligible_for_kb": False,
                "invalid_reasons": [{"kind": "service_unavailable", "detail": error}],
                "clauses": [],
            }
        )
    eligible = sum(call["eligible_for_kb"] is True for call in calls)
    call_coverage = {
        "total_call_count": len(calls),
        "eligible_call_count": eligible,
        "withheld_call_count": len(calls) - eligible,
        "eligible_fraction": eligible / len(calls) if calls else 1.0,
    }
    return {
        "telemetry_status": "unavailable",
        "formal_completeness": "unavailable",
        "collection_validity": "invalid",
        "call_coverage": call_coverage,
        "artifact": {
            "version": 1,
            "mode": "resource",
            "status_model": "workload_telemetry_formal_v1",
            "run_id": run.run_id,
            "trace_id": trace.trace_id,
            "calls": calls,
            "workload_execution": workload_status,
            "telemetry_quality": "unavailable",
            "formal_completeness": "unavailable",
            "collection_validity": "invalid",
            "cleanup": "unknown",
            "call_coverage": call_coverage,
            "pinned_snapshot_id": run.pinned_snapshot_id,
            "canonicalizer_version": CANONICALIZER_VERSION,
        },
        "errors": [error],
    }


def _not_requested_trace_result(
    trace: _Trace,
    run: _Run,
    workload_status: str,
) -> dict[str, Any]:
    calls = [
        {
            "version": 1,
            "tool_call_id": call.call_id,
            "command": call.command,
            "telemetry_status": call.telemetry_status,
            "telemetry_quality": "not_requested",
            "eligible_for_kb": False,
            "invalid_reasons": [],
            "clauses": [],
        }
        for call in trace.calls.values()
    ]
    artifact = {
        "version": 1,
        "mode": "resource",
        "status_model": "workload_telemetry_formal_v1",
        "run_id": run.run_id,
        "trace_id": trace.trace_id,
        "calls": calls,
        "workload_execution": workload_status,
        "telemetry_quality": "not_requested",
        "formal_completeness": "not_requested",
        "collection_validity": "not_requested",
        "cleanup": "not_requested",
        "pinned_snapshot_id": run.pinned_snapshot_id,
        "canonicalizer_version": CANONICALIZER_VERSION,
    }
    return {
        "telemetry_status": "not_requested",
        "formal_completeness": "not_requested",
        "collection_validity": "not_requested",
        "call_coverage": {
            "total_call_count": len(calls),
            "eligible_call_count": 0,
            "withheld_call_count": len(calls),
            "eligible_fraction": 0.0 if calls else 1.0,
        },
        "artifact": artifact,
        "errors": [],
    }


class ResourceServer(StrictUnixServer):
    def __init__(
        self,
        socket_path: str | Path,
        *,
        service: ResourceService,
        allowed_uids: set[int],
        request_timeout_s: float = DEFAULT_TIMEOUT_S,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        socket_mode: int = 0o600,
        socket_gid: int | None = None,
    ) -> None:
        self.service = service
        super().__init__(
            socket_path,
            protocol_version=RESOURCE_PROTOCOL_VERSION,
            protocol_error=ResourceProtocolError,
            dispatch=service.dispatch,
            allowed_uids=allowed_uids,
            request_timeout_s=request_timeout_s,
            max_message_bytes=max_message_bytes,
            socket_mode=socket_mode,
            socket_gid=socket_gid,
        )

    def service_actions(self) -> None:
        self.service.expire()

    def server_close(self) -> None:
        try:
            self.service.close()
        finally:
            super().server_close()


def _parse_mode(value: str) -> int:
    mode = int(value, 8)
    if mode & ~0o660:
        raise argparse.ArgumentTypeError("socket mode may grant only user/group rw")
    return mode


def main(argv: Sequence[str] | None = None) -> int:
    if os.geteuid() == 0:
        raise PermissionError("resource-agentd must not run as root")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--telemetry-socket", type=Path, required=True)
    parser.add_argument("--telemetry-peer-uid", type=int, default=0)
    parser.add_argument("--allowed-uid", type=int, action="append")
    parser.add_argument("--socket-gid", type=int)
    parser.add_argument("--socket-mode", type=_parse_mode, default=0o600)
    parser.add_argument("--result-ttl", type=float, default=DEFAULT_RESULT_TTL_S)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    client_uids = set(args.allowed_uid or [os.geteuid()])
    if any(uid < 0 for uid in client_uids):
        raise ValueError("client UIDs must be non-negative")
    service = ResourceService(
        ObservationStore(args.database),
        TelemetryUnixTransport(
            args.telemetry_socket,
            expected_peer_uid=args.telemetry_peer_uid,
        ),
        result_ttl_s=args.result_ttl,
    )
    with ResourceServer(
        args.socket,
        service=service,
        allowed_uids=client_uids,
        socket_mode=args.socket_mode,
        socket_gid=args.socket_gid,
    ) as server:
        previous_sigterm = signal.signal(
            signal.SIGTERM,
            lambda *_args: setattr(server, "_BaseServer__shutdown_request", True),
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICALIZER_VERSION",
    "DEFAULT_RESULT_TTL_S",
    "ResourceServer",
    "ResourceService",
    "main",
]
