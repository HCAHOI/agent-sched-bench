"""Unprivileged resource service: parsing, prediction, SQLite, snapshots."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
import threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_resource._uds import (
    DEFAULT_TIMEOUT_S,
    MAX_MESSAGE_BYTES,
    StrictUnixServer,
    require_fields,
)
from tool_resource.features import parse_command_clauses
from tool_resource.resource_protocol import (
    RESOURCE_PROTOCOL_VERSION,
    ResourceProtocolError,
)
from tool_resource.runtime_kb import (
    CANONICAL_LATENCY_BUCKETS,
    ClauseObservation,
    ClauseResourceKB,
    LatencyBuckets,
)
from tool_resource.store import STORE_SCHEMA_VERSION, ObservationStore
from tool_resource.telemetry_protocol import (
    TelemetryTransport,
    TelemetryUnixTransport,
)

CANONICALIZER_VERSION = "mvdan-sh-v1"
DEFAULT_SESSION_TTL_S = 1800.0
_LEASE_IDEMPOTENT_OPERATIONS = frozenset({"OpenRun", "OpenTrace", "BeginCall"})


@dataclass
class _Call:
    call_id: str
    command: str
    command_digest: str
    parsed: dict[str, Any]
    query_timestamp: float
    telemetry_call_token: str | None
    telemetry_status: str
    workload_result: Mapping[str, Any] | str | None = None
    end_timestamp: float | None = None
    end_result: dict[str, Any] | None = None


@dataclass
class _Trace:
    run_token: str
    trace_id: str
    telemetry_session_token: str | None
    telemetry_status: str
    source_exec_actions: list[dict[str, Any]]
    calls: dict[str, _Call] = field(default_factory=dict)
    call_ids: set[str] = field(default_factory=set)
    source_index: int = 0
    integrity_errors: list[str] = field(default_factory=list)
    ingested_observation_ids: set[str] = field(default_factory=set)
    close_result: dict[str, Any] | None = None
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _Run:
    run_id: str
    workspace_scope: str
    pinned_snapshot_id: str
    buckets: LatencyBuckets
    update_policy: str
    telemetry_requirement: str
    behavior: str
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
        session_ttl_s: float = DEFAULT_SESSION_TTL_S,
    ) -> None:
        if not math.isfinite(session_ttl_s) or session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be finite and positive")
        self.store = store
        self.telemetry = telemetry_transport
        self.session_ttl_s = session_ttl_s
        self._runs: dict[str, _Run] = {}
        self._traces: dict[str, _Trace] = {}
        self._operation_results: dict[
            tuple[int, str], tuple[str, str, dict[str, Any], str]
        ] = {}
        self._lock = threading.Lock()

    def dispatch(
        self,
        operation: str,
        payload: Mapping[str, Any],
        request_identity: tuple[int, str] | None = None,
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
            "OpenRun": self._open_run,
            "OpenTrace": self._open_trace,
            "BeginCall": self._begin_call,
            "EndCall": self._end_call,
            "RecordSafetyGuardBlock": self._record_safety_guard_block,
            "RecordTraceIntegrityFailure": self._record_trace_integrity_failure,
            "CloseTrace": self._close_trace,
            "AbortTrace": self._abort_trace,
            "CloseRun": self._close_run,
            "AbortRun": self._abort_run,
        }
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
            "prediction_targets": ["latency_bucket"],
            "canonicalizer_version": CANONICALIZER_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "raw_events_exposed": False,
        }

    def _open_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
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
        try:
            envelopes = self.store.observations_for_snapshot(snapshot_id)
            observations = [
                observation
                for envelope in envelopes
                for observation in _clause_observations(envelope)
            ]
        except ValueError as exc:
            raise ResourceProtocolError(str(exc)) from exc
        public = [
            observation
            for observation in observations
            if observation.repo != scope and observation.latency_ms is not None
        ]
        kb = ClauseResourceKB.fit_public(public) if public else ClauseResourceKB()
        for observation in observations:
            if observation.repo == scope:
                kb.observe_completed_clause(observation)
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

    def _open_trace(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "run_token",
                "trace_id",
                "container_runtime",
                "container_id",
                "repo_metadata",
                "source_actions",
            },
            optional={"runner_pid"},
            error_type=ResourceProtocolError,
        )
        run_token = _string(payload, "run_token")
        run = self._run(run_token)
        trace_id = _string(payload, "trace_id")
        runtime = _string(payload, "container_runtime")
        container_id = _string(payload, "container_id")
        metadata = payload["repo_metadata"]
        actions = payload["source_actions"]
        if not isinstance(metadata, Mapping):
            raise ResourceProtocolError("repo_metadata must be an object")
        if not isinstance(actions, list) or not all(
            isinstance(action, Mapping) for action in actions
        ):
            raise ResourceProtocolError("source_actions must be a list of objects")
        runner_pid = payload.get("runner_pid")
        if runner_pid is not None and (
            not isinstance(runner_pid, int)
            or isinstance(runner_pid, bool)
            or runner_pid <= 0
        ):
            raise ResourceProtocolError("runner_pid must be a positive integer")
        telemetry_token = None
        telemetry_status = "not_requested"
        if "observe" in run.behavior:
            try:
                result = self.telemetry.request(
                    "AttachTarget",
                    {
                        "run_id": run.run_id,
                        "trace_id": trace_id,
                        "container_runtime": runtime,
                        "container_id": container_id,
                        "workspace_scope": run.workspace_scope,
                        **(
                            {"runner_pid": runner_pid} if runner_pid is not None else {}
                        ),
                    },
                )
                _require_result_fields(
                    result,
                    {
                        "telemetry_session_token",
                        "target_status",
                        "resolved_target",
                    },
                    "AttachTarget",
                )
                _require_result_fields(
                    result["resolved_target"],
                    {"init_pid", "cgroup_id", "quota_cores"},
                    "AttachTarget.resolved_target",
                )
                telemetry_token = _result_string(
                    result,
                    "telemetry_session_token",
                )
                telemetry_status = str(result.get("target_status") or "unavailable")
            except Exception as exc:  # noqa: BLE001 - workload must remain independent
                telemetry_status = f"unavailable:{type(exc).__name__}:{exc}"
        trace_token = uuid.uuid4().hex
        source_exec_actions = [
            dict(action)
            for action in actions
            if action.get("action_type") == "tool_exec"
            and isinstance(action.get("data"), Mapping)
            and action["data"].get("tool_name") == "exec"
        ]
        with run.lock:
            if run.close_result is not None:
                if telemetry_token is not None:
                    with suppress(Exception):
                        self.telemetry.request(
                            "AbortSession",
                            {
                                "telemetry_session_token": telemetry_token,
                                "reason": "resource run closed during trace open",
                            },
                        )
                raise ResourceProtocolError("run is already closed")
            if trace_id in run.trace_ids:
                if telemetry_token is not None:
                    with suppress(Exception):
                        self.telemetry.request(
                            "AbortSession",
                            {
                                "telemetry_session_token": telemetry_token,
                                "reason": "duplicate resource trace identity",
                            },
                        )
                raise ResourceProtocolError(f"duplicate trace_id {trace_id!r}")
            with self._lock:
                self._traces[trace_token] = _Trace(
                    run_token,
                    trace_id,
                    telemetry_token,
                    telemetry_status,
                    source_exec_actions,
                )
            run.trace_tokens.add(trace_token)
            run.trace_ids.add(trace_id)
            run.last_used = time.monotonic()
        return {
            "trace_token": trace_token,
            "telemetry_status": telemetry_status,
        }

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
            if trace.close_result is not None:
                raise ResourceProtocolError("trace is already closed")
            if call_id in trace.call_ids:
                raise ResourceProtocolError(f"duplicate call_id {call_id!r}")
            parsed = parse_command_clauses(command)
            digest = hashlib.sha256(command.encode()).hexdigest()
            source_plan = self._source_plan(trace)
            prediction, prediction_error = self._predict(
                run, command, parsed, query_timestamp
            )
            telemetry_call_token = None
            telemetry_status = trace.telemetry_status
            if trace.telemetry_session_token is not None:
                try:
                    result = self.telemetry.request(
                        "RegisterCall",
                        {
                            "telemetry_session_token": trace.telemetry_session_token,
                            "call_id": call_id,
                            "command_digest": digest,
                            "static_call_plan": {
                                "canonical_command": command,
                                "parsed": parsed,
                                **source_plan,
                            },
                        },
                    )
                    _require_result_fields(
                        result,
                        {"telemetry_call_token", "telemetry_status"},
                        "RegisterCall",
                    )
                    telemetry_call_token = _result_string(
                        result,
                        "telemetry_call_token",
                    )
                    telemetry_status = str(
                        result.get("telemetry_status") or "registered"
                    )
                except Exception as exc:  # noqa: BLE001 - fail isolated
                    telemetry_status = f"unavailable:{type(exc).__name__}:{exc}"
            call_token = uuid.uuid4().hex
            trace.calls[call_token] = _Call(
                call_id,
                command,
                digest,
                parsed,
                query_timestamp,
                telemetry_call_token,
                telemetry_status,
            )
            trace.call_ids.add(call_id)
            trace.last_used = time.monotonic()
        prediction_payload = _prediction_payload(prediction)
        selected = prediction_payload.get("prediction") or {}
        return {
            "call_token": call_token,
            "prediction": prediction_payload,
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
            "telemetry_status": telemetry_status,
        }

    def _end_call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"call_token", "workload_result", "end_timestamp"},
            error_type=ResourceProtocolError,
        )
        trace, call = self._call(payload)
        workload_result = payload["workload_result"]
        if workload_result is not None and not isinstance(workload_result, Mapping):
            raise ResourceProtocolError("workload_result must be an object or null")
        end_timestamp = _finite_number(payload, "end_timestamp")
        with trace.lock:
            if call.end_result is not None:
                return call.end_result
            telemetry_status = call.telemetry_status
            observation: dict[str, Any] | None = None
            if (
                trace.telemetry_session_token is not None
                and call.telemetry_call_token is not None
            ):
                try:
                    result = self.telemetry.request(
                        "FinishCall",
                        {
                            "telemetry_session_token": trace.telemetry_session_token,
                            "telemetry_call_token": call.telemetry_call_token,
                            "workload_result": (
                                None
                                if workload_result is None
                                else dict(workload_result)
                            ),
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
                        "FinishCall",
                    )
                    raw = result.get("finalized_call_observation")
                    if not isinstance(raw, Mapping):
                        raise ResourceProtocolError(
                            "telemetryd returned no finalized observation"
                        )
                    observation = dict(raw)
                    telemetry_status = str(
                        result.get("telemetry_status") or "unavailable"
                    )
                except Exception as exc:  # noqa: BLE001 - fail isolated
                    telemetry_status = f"unavailable:{type(exc).__name__}:{exc}"
            call.workload_result = workload_result
            call.end_timestamp = end_timestamp
            call.telemetry_status = telemetry_status
            if observation is None:
                observation = {
                    "call_id": call.call_id,
                    "telemetry_eligible": False,
                    "telemetry_status": telemetry_status,
                    "invalid_reasons": [
                        {
                            "kind": "telemetry_not_requested",
                            "detail": "profile does not observe",
                        }
                    ],
                    "clauses": [],
                    "session_finalization": "not_requested",
                }
            call.end_result = {
                "workload_result": workload_result,
                "finalized_call_observation": observation,
                "telemetry_status": telemetry_status,
                "ingest_status": "pending_trace_finalization",
                "rejection_reasons": [],
            }
            trace.last_used = time.monotonic()
            return call.end_result

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
            telemetry_status = trace.telemetry_status
            observation = None
            if trace.telemetry_session_token is not None:
                try:
                    result = self.telemetry.request(
                        "RecordSafetyGuardBlock",
                        {
                            "telemetry_session_token": trace.telemetry_session_token,
                            "call_id": call_id,
                            "command_digest": digest,
                            "static_call_plan": {
                                "canonical_command": command,
                                "parsed": parsed,
                                **self._source_plan(trace),
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
                    raw = result.get("finalized_call_observation")
                    observation = dict(raw) if isinstance(raw, Mapping) else None
                    telemetry_status = str(
                        result.get("telemetry_status") or "unavailable"
                    )
                except Exception as exc:  # noqa: BLE001 - fail isolated
                    telemetry_status = f"unavailable:{type(exc).__name__}:{exc}"
            call_token = uuid.uuid4().hex
            trace.calls[call_token] = _Call(
                call_id,
                command,
                digest,
                parsed,
                end_timestamp,
                None,
                telemetry_status,
                workload_result=workload_result,
                end_timestamp=end_timestamp,
            )
            trace.call_ids.add(call_id)
            trace.last_used = time.monotonic()
        return {
            "workload_result": workload_result,
            "prediction": _prediction_payload(prediction),
            "prediction_error": prediction_error,
            "finalized_call_observation": observation,
            "telemetry_status": telemetry_status,
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
            try:
                finalized = self.telemetry.request(
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
                completed_call_count = sum(
                    call.workload_result is not None for call in trace.calls.values()
                )
                if len(observation_ids) != completed_call_count:
                    raise ResourceProtocolError(
                        "telemetryd finalized incomplete call coverage"
                    )
                fetched_observations: list[Mapping[str, Any]] = []
                for observation_id in observation_ids:
                    if not isinstance(observation_id, str):
                        raise ResourceProtocolError("invalid observation_id")
                    fetched = self.telemetry.request(
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
                calls = []
                ingested_envelopes: list[dict[str, Any]] = []
                for raw_observation in fetched_observations:
                    call, envelope = self._ingest_observation(
                        run,
                        trace,
                        raw_observation,
                    )
                    calls.append(call)
                    ingested_envelopes.append(envelope)
                    acknowledged = self.telemetry.request(
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
                    if envelope["ingest_eligible"] is True:
                        trace.ingested_observation_ids.add(
                            str(envelope["observation_id"])
                        )
                if run.update_policy == "causal":
                    with run.lock:
                        for envelope in ingested_envelopes:
                            if envelope["ingest_eligible"] is not True:
                                continue
                            for observation in _clause_observations(envelope):
                                run.kb.observe_completed_clause(observation)
                trace.close_result = _trace_result(
                    run,
                    trace,
                    summary,
                    calls,
                    workload_status,
                )
            except Exception as exc:  # noqa: BLE001 - workload already completed
                trace.close_result = _unavailable_trace_result(
                    trace,
                    run,
                    f"telemetry finalization failed: {type(exc).__name__}: {exc}",
                    workload_status,
                )
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
            if trace.close_result is None and trace.telemetry_session_token is not None:
                with suppress(Exception):
                    self.telemetry.request(
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
        return {"aborted": True, **trace.close_result}

    def _close_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"run_token", "workload_status"},
            error_type=ResourceProtocolError,
        )
        run_token = _string(payload, "run_token")
        run = self._run(run_token)
        workload_status = _choice(
            payload,
            "workload_status",
            {"completed", "failed", "incomplete"},
        )
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
            trace_results = [
                self._traces[token].close_result
                for token in run.trace_tokens
                if token in self._traces
            ]
            promotable_observation_ids = {
                observation_id
                for token in run.trace_tokens
                if token in self._traces
                and isinstance(self._traces[token].close_result, Mapping)
                and self._traces[token].close_result.get("telemetry_status") == "ok"
                for observation_id in self._traces[token].ingested_observation_ids
            }
            promoted = self.store.promote_observations(promotable_observation_ids)
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
                "promoted_observation_count": promoted,
                "run_manifest": {
                    "run_id": run.run_id,
                    "workspace_scope": run.workspace_scope,
                    "pinned_snapshot_id": run.pinned_snapshot_id,
                    "resulting_snapshot_id": resulting_snapshot,
                    "canonicalizer_version": CANONICALIZER_VERSION,
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
            if (
                sum(observed_clause_identities.values()) != len(clauses)
                or observed_clause_identities - expected_clause_identities
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
        cutoff = time.monotonic() - self.session_ttl_s
        with self._lock:
            stale_runs = [
                token for token, run in self._runs.items() if run.last_used < cutoff
            ]
        for token in stale_runs:
            with suppress(Exception):
                self._abort_run(
                    {"run_token": token, "reason": "resource run lease expired"}
                )
            with self._lock:
                run = self._runs.pop(token, None)
                if run is not None:
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
        action = (
            trace.source_exec_actions[trace.source_index]
            if trace.source_index < len(trace.source_exec_actions)
            else None
        )
        trace.source_index += 1
        data = action.get("data") if isinstance(action, Mapping) else None
        if not isinstance(data, Mapping):
            return {
                "source_tool_call_id": "",
                "source_command": "",
                "source_tool_result": "",
            }
        raw_args = data.get("tool_args")
        command = ""
        if isinstance(raw_args, Mapping):
            command = str(raw_args.get("command") or "")
        elif isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = None
            if isinstance(parsed_args, Mapping):
                command = str(parsed_args.get("command") or "")
        return {
            "source_tool_call_id": str(data.get("tool_call_id") or ""),
            "source_command": command,
            "source_tool_result": str(
                data.get("tool_result", data.get("result", "")) or ""
            ),
        }


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
        try:
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
    telemetry_ok = (
        summary.get("collector_health") == "healthy"
        and summary.get("collection_validity") == "valid"
        and summary.get("cleanup_status") == "ok"
        and all(
            not call.telemetry_status.startswith("unavailable")
            for call in trace.calls.values()
        )
    )
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
) -> dict[str, Any]:
    calls = [
        {
            "version": 1,
            "tool_call_id": call.call_id,
            "command": call.command,
            "telemetry_quality": "unavailable",
            "eligible_for_kb": False,
            "invalid_reasons": [{"kind": "service_unavailable", "detail": error}],
            "clauses": [],
        }
        for call in trace.calls.values()
    ]
    return {
        "telemetry_status": "unavailable",
        "formal_completeness": "unavailable",
        "collection_validity": "invalid",
        "call_coverage": {
            "total_call_count": len(calls),
            "eligible_call_count": 0,
            "withheld_call_count": len(calls),
            "eligible_fraction": 0.0 if calls else 1.0,
        },
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
    parser.add_argument("--session-ttl", type=float, default=DEFAULT_SESSION_TTL_S)
    args = parser.parse_args(argv)
    client_uids = set(args.allowed_uid or [os.geteuid()])
    if any(uid < 0 for uid in client_uids):
        raise ValueError("client UIDs must be non-negative")
    service = ResourceService(
        ObservationStore(args.database),
        TelemetryUnixTransport(
            args.telemetry_socket,
            expected_peer_uid=args.telemetry_peer_uid,
        ),
        session_ttl_s=args.session_ttl,
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
    "DEFAULT_SESSION_TTL_S",
    "ResourceServer",
    "ResourceService",
    "main",
]
