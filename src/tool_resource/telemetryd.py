"""Privileged telemetry service: container resolution, BCC/eBPF, attribution."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import tempfile
import threading
import time
import uuid
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
from tool_resource.telemetry_protocol import (
    TELEMETRY_PROTOCOL_VERSION,
    TelemetryProtocolError,
)

DEFAULT_OBSERVATION_TTL_S = 900.0
_LOG = logging.getLogger("tool_resource.telemetryd")
_LEASE_IDEMPOTENT_OPERATIONS = frozenset({"AttachTarget", "RegisterCall"})
CollectorFactory = Callable[..., Any]


def _collector_factory(**kwargs: Any) -> Any:
    from tool_resource.telemetry import ClauseTelemetryCollector

    try:
        return ClauseTelemetryCollector(**kwargs)
    except Exception as exc:  # noqa: BLE001 - attach is explicitly fail-isolated
        return ClauseTelemetryCollector.unavailable(
            container_id=str(kwargs["container_id"]),
            repo=str(kwargs["repo"]),
            artifact_path=Path(kwargs["artifact_path"]),
            reason=f"collector attach failed: {type(exc).__name__}: {exc}",
        )


@dataclass
class _Call:
    call_id: str
    command_digest: str
    plan: dict[str, Any]
    collector_token: Any
    observation_id: str
    finished_summary: dict[str, Any] | None = None
    finish_result: dict[str, Any] | None = None
    collector_ready: bool = True


@dataclass
class _Session:
    run_id: str
    trace_id: str
    collector: Any
    artifact_path: Path
    owner_identity: tuple[int, int]
    attached_monotonic_ns: int
    calls: dict[str, _Call] = field(default_factory=dict)
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    acknowledged: set[str] = field(default_factory=set)
    integrity_errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None
    last_used: float = field(default_factory=time.monotonic)
    finalized_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class TelemetryService:
    """Own all privileged collector state behind the Telemetry Protocol."""

    def __init__(
        self,
        *,
        container_executable: str | None = None,
        collector_factory: CollectorFactory = _collector_factory,
        state_dir: str | Path | None = None,
        observation_ttl_s: float = DEFAULT_OBSERVATION_TTL_S,
    ) -> None:
        if not math.isfinite(observation_ttl_s) or observation_ttl_s <= 0:
            raise ValueError("observation_ttl_s must be finite and positive")
        self.container_executable = container_executable
        self.collector_factory = collector_factory
        self.observation_ttl_s = observation_ttl_s
        self._state_tmp = (
            tempfile.TemporaryDirectory(prefix="telemetryd-")
            if state_dir is None
            else None
        )
        self.state_dir = Path(
            self._state_tmp.name if self._state_tmp is not None else state_dir
        )
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._sessions: dict[str, _Session] = {}
        self._operation_results: dict[
            tuple[int, str], tuple[str, str, dict[str, Any], str]
        ] = {}
        self._lock = threading.Lock()
        self._attach_lock = threading.Lock()

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
                    cached_operation, cached_payload, result, session_token = cached
                    if (
                        cached_operation != operation
                        or cached_payload != payload_identity
                    ):
                        raise TelemetryProtocolError(
                            "request_id was already used for a different request"
                        )
                    self._sessions[session_token].last_used = time.monotonic()
                    return dict(result)
        handlers = {
            "Ping": self._ping,
            "Capabilities": self._capabilities,
            "RegisterCall": self._register_call,
            "FinishCall": self._finish_call,
            "RecordSafetyGuardBlock": self._record_safety_guard_block,
            "FinalizeSession": self._finalize_session,
            "AbortSession": self._abort_session,
            "FetchFinalizedObservation": self._fetch_observation,
            "AcknowledgeObservation": self._acknowledge_observation,
        }
        if operation == "AttachTarget":
            result = self._attach_target(
                payload,
                owner_identity=process_identity(
                    os.getpid() if peer_pid is None else peer_pid
                ),
            )
        else:
            try:
                handler = handlers[operation]
            except KeyError as exc:
                raise TelemetryProtocolError(
                    f"unsupported telemetry operation {operation!r}"
                ) from exc
            result = handler(payload)
        if (
            payload_identity is not None
            and request_identity is not None
            and operation in _LEASE_IDEMPOTENT_OPERATIONS
        ):
            session_token = (
                str(result["telemetry_session_token"])
                if operation == "AttachTarget"
                else str(payload["telemetry_session_token"])
            )
            with self._lock:
                self._operation_results[request_identity] = (
                    operation,
                    payload_identity,
                    dict(result),
                    session_token,
                )
        return result

    def _ping(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required=set(),
            error_type=TelemetryProtocolError,
        )
        return {"protocol_version": TELEMETRY_PROTOCOL_VERSION}

    def _capabilities(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required=set(),
            error_type=TelemetryProtocolError,
        )
        return {
            "operations": [
                "AttachTarget",
                "RegisterCall",
                "FinishCall",
                "RecordSafetyGuardBlock",
                "FinalizeSession",
                "AbortSession",
                "FetchFinalizedObservation",
                "AcknowledgeObservation",
            ],
            "raw_events_exposed": False,
        }

    def _attach_target(
        self,
        payload: Mapping[str, Any],
        *,
        owner_identity: tuple[int, int],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "run_id",
                "trace_id",
                "container_runtime",
                "container_id",
                "workspace_scope",
            },
            error_type=TelemetryProtocolError,
        )
        run_id = _string(payload, "run_id")
        trace_id = _string(payload, "trace_id")
        runtime = _string(payload, "container_runtime")
        container_id = _string(payload, "container_id")
        workspace_scope = _string(payload, "workspace_scope")
        if runtime not in {"docker", "podman"}:
            raise TelemetryProtocolError("container_runtime must be docker or podman")
        if (
            self.container_executable is not None
            and runtime != self.container_executable
        ):
            raise TelemetryProtocolError(
                "container runtime differs from telemetryd configuration"
            )
        token = uuid.uuid4().hex
        artifact_path = self.state_dir / f"{token}.json"
        # libbcc collector construction is not thread-safe within one process.
        with self._attach_lock:
            collector = self.collector_factory(
                container_id=container_id,
                container_executable=runtime,
                repo=workspace_scope,
                artifact_path=artifact_path,
                source_actions=(),
            )
        session = _Session(
            run_id,
            trace_id,
            collector,
            artifact_path,
            owner_identity,
            time.monotonic_ns(),
        )
        with self._lock:
            self._sessions[token] = session
        target_status = (
            "available"
            if getattr(collector, "state", None) == "active"
            else "unavailable"
        )
        _LOG.info(
            "attached target run_id=%s trace_id=%s call_id=- status=%s reason=%s",
            run_id,
            trace_id,
            target_status,
            getattr(collector, "_disabled_reason", None) or "-",
        )
        return {
            "telemetry_session_token": token,
            "target_status": target_status,
            "resolved_target": {
                "init_pid": int(getattr(collector, "init_pid", 0)),
                "cgroup_id": int(getattr(collector, "cgroup_id", 0)),
                "quota_cores": float(getattr(collector, "quota_cores", 0.0)),
            },
        }

    def _register_call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "telemetry_session_token",
                "call_id",
                "command_digest",
                "call_started_monotonic_ns",
                "static_call_plan",
            },
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        call_id = _string(payload, "call_id")
        command_digest = _sha256(payload, "command_digest")
        started_ns = _positive_integer(payload, "call_started_monotonic_ns")
        plan = _static_plan(payload["static_call_plan"])
        with session.lock:
            self._require_open(session)
            if any(call.call_id == call_id for call in session.calls.values()):
                raise TelemetryProtocolError(f"duplicate call_id {call_id!r}")
            collector_ready = started_ns >= session.attached_monotonic_ns
            collector_token = (
                session.collector.begin_tool_call(
                    call_id,
                    plan["canonical_command"],
                    static_plan=plan["parsed"],
                    source_tool_call_id=plan["source_tool_call_id"],
                    source_command=plan["source_command"],
                    source_tool_result=plan["source_tool_result"],
                    started_ns=started_ns,
                )
                if collector_ready
                else None
            )
            call_token = uuid.uuid4().hex
            session.calls[call_token] = _Call(
                call_id,
                command_digest,
                plan,
                collector_token,
                uuid.uuid4().hex,
                collector_ready=collector_ready,
            )
            session.last_used = time.monotonic()
        _LOG.info(
            "registered call run_id=%s trace_id=%s call_id=%s",
            session.run_id,
            session.trace_id,
            call_id,
        )
        return {
            "telemetry_call_token": call_token,
            "telemetry_status": (
                "registered"
                if collector_ready
                else "unavailable:collector_not_ready"
            ),
        }

    def _finish_call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "telemetry_session_token",
                "telemetry_call_token",
                "workload_result",
                "end_timestamp",
                "call_ended_monotonic_ns",
            },
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        call = self._call(session, payload)
        result = payload["workload_result"]
        if result is not None and not isinstance(result, Mapping):
            raise TelemetryProtocolError("workload_result must be an object or null")
        _finite_number(payload, "end_timestamp")
        ended_ns = _positive_integer(payload, "call_ended_monotonic_ns")
        with session.lock:
            self._require_open(session)
            if call.finish_result is not None:
                return call.finish_result
            summary = (
                session.collector.finish_tool_call(
                    call.collector_token,
                    replay_response=result,
                    ended_ns=ended_ns,
                )
                if call.collector_ready
                else {
                    "tool_call_id": call.call_id,
                    "command": call.plan["canonical_command"],
                    "telemetry_quality": "unavailable",
                    "eligible_for_kb": False,
                    "invalid_reasons": [
                        {
                            "kind": "collector_not_ready",
                            "detail": "target attachment completed after call start",
                        }
                    ],
                    "clauses": [],
                }
            )
            if not isinstance(summary, Mapping):
                raise TelemetryProtocolError("collector returned an invalid call")
            call.finished_summary = dict(summary)
            call.finish_result = {
                "observation_id": call.observation_id,
                "finalized_call_observation": _provisional_call(summary),
                "telemetry_status": str(
                    summary.get("telemetry_quality") or "unavailable"
                ),
            }
            session.last_used = time.monotonic()
            finish_result = call.finish_result
        _LOG.info(
            "finished call run_id=%s trace_id=%s call_id=%s status=%s",
            session.run_id,
            session.trace_id,
            call.call_id,
            finish_result["telemetry_status"],
        )
        return finish_result

    def _record_safety_guard_block(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "telemetry_session_token",
                "call_id",
                "command_digest",
                "static_call_plan",
                "workload_result",
                "end_timestamp",
            },
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        call_id = _string(payload, "call_id")
        command_digest = _sha256(payload, "command_digest")
        plan = _static_plan(payload["static_call_plan"])
        workload_result = payload["workload_result"]
        if not isinstance(workload_result, str):
            raise TelemetryProtocolError("workload_result must be a string")
        _finite_number(payload, "end_timestamp")
        with session.lock:
            self._require_open(session)
            summary = session.collector.record_safety_guard_blocked(
                call_id,
                plan["canonical_command"],
                workload_result,
                static_plan=plan["parsed"],
                source_tool_call_id=plan["source_tool_call_id"],
                source_command=plan["source_command"],
                source_tool_result=plan["source_tool_result"],
            )
            call_token = uuid.uuid4().hex
            observation_id = uuid.uuid4().hex
            session.calls[call_token] = _Call(
                call_id,
                command_digest,
                plan,
                None,
                observation_id,
                dict(summary),
            )
            result = {
                "observation_id": observation_id,
                "finalized_call_observation": _provisional_call(summary),
                "telemetry_status": str(
                    summary.get("telemetry_quality") or "unavailable"
                ),
            }
            session.calls[call_token].finish_result = result
            session.last_used = time.monotonic()
        return result

    def _finalize_session(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={
                "telemetry_session_token",
                "workload_status",
                "integrity_errors",
            },
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        workload_status = _string(payload, "workload_status")
        if workload_status not in {"completed", "failed", "incomplete"}:
            raise TelemetryProtocolError("invalid workload_status")
        errors = payload["integrity_errors"]
        if not isinstance(errors, list) or not all(
            isinstance(error, str) and error for error in errors
        ):
            raise TelemetryProtocolError("integrity_errors must be strings")
        with session.lock:
            if session.final_result is not None:
                return session.final_result
            for error in errors:
                if error not in session.integrity_errors:
                    session.integrity_errors.append(error)
            session.collector.finalize(replay_execution=workload_status)
            artifact = json.loads(session.artifact_path.read_text(encoding="utf-8"))
            session.artifact_path.unlink(missing_ok=True)
            calls = {
                str(call["tool_call_id"]): call
                for call in artifact.get("calls", [])
                if isinstance(call, Mapping) and call.get("tool_call_id")
            }
            for call in session.calls.values():
                summary = calls.get(call.call_id, call.finished_summary or {})
                observation = _normalized_observation(
                    session,
                    call,
                    summary,
                    artifact,
                )
                session.observations[call.observation_id] = observation
            session.summary = _session_summary(artifact)
            if session.integrity_errors:
                if session.summary["formal_completeness"] != "unavailable":
                    session.summary["formal_completeness"] = "partial"
                session.summary["errors"].extend(session.integrity_errors)
            session.finalized_at = time.monotonic()
            session.final_result = {
                "session_summary": session.summary,
                "observation_ids": list(session.observations),
            }
            session.last_used = session.finalized_at
            final_result = session.final_result
        _LOG.info(
            "finalized session run_id=%s trace_id=%s call_id=- "
            "collector_health=%s eligible_calls=%s withheld_calls=%s",
            session.run_id,
            session.trace_id,
            final_result["session_summary"]["collector_health"],
            final_result["session_summary"]["eligible_call_count"],
            final_result["session_summary"]["withheld_call_count"],
        )
        return final_result

    def _abort_session(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"telemetry_session_token", "reason"},
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        reason = _string(payload, "reason")
        with session.lock:
            if session.final_result is not None:
                return {"aborted": True, "session_summary": session.summary}
            with suppress(Exception):
                session.collector.add_integrity_error(f"session aborted: {reason}")
                session.collector.finalize(replay_execution="incomplete")
            session.artifact_path.unlink(missing_ok=True)
            session.finalized_at = time.monotonic()
            session.summary = {
                "collector_health": "unavailable",
                "formal_completeness": "unavailable",
                "collection_validity": "invalid",
                "cleanup_status": "unknown",
                "eligible_call_count": 0,
                "withheld_call_count": len(session.calls),
                "loss_counters": {},
                "errors": [reason],
            }
            session.final_result = {
                "session_summary": session.summary,
                "observation_ids": [],
            }
        _LOG.info(
            "aborted session run_id=%s trace_id=%s call_id=- reason=%s",
            session.run_id,
            session.trace_id,
            reason,
        )
        return {"aborted": True, "session_summary": session.summary}

    def _fetch_observation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require_fields(
            payload,
            required={"telemetry_session_token", "observation_id"},
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        observation_id = _string(payload, "observation_id")
        with session.lock:
            if session.final_result is None:
                raise TelemetryProtocolError("session is not finalized")
            try:
                observation = session.observations[observation_id]
            except KeyError as exc:
                raise TelemetryProtocolError("unknown finalized observation") from exc
            session.last_used = time.monotonic()
            return {"finalized_call_observation": observation}

    def _acknowledge_observation(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        require_fields(
            payload,
            required={"telemetry_session_token", "observation_id"},
            error_type=TelemetryProtocolError,
        )
        session = self._session(payload)
        observation_id = _string(payload, "observation_id")
        with session.lock:
            if observation_id not in session.observations:
                raise TelemetryProtocolError("unknown finalized observation")
            session.acknowledged.add(observation_id)
            session.last_used = time.monotonic()
        return {"acknowledged": True}

    def expire(self) -> None:
        now = time.monotonic()
        with self._lock:
            sessions = list(self._sessions.items())
        for token, session in sessions:
            if session.finalized_at is None:
                if process_identity_is_alive(session.owner_identity):
                    continue
                _LOG.warning(
                    "telemetry owner exited run_id=%s trace_id=%s call_id=- "
                    "owner_pid=%d",
                    session.run_id,
                    session.trace_id,
                    session.owner_identity[0],
                )
                try:
                    self._abort_session(
                        {
                            "telemetry_session_token": token,
                            "reason": "telemetry client exited",
                        }
                    )
                except Exception:
                    continue
            elif now < session.finalized_at + self.observation_ttl_s:
                continue
            with self._lock:
                self._sessions.pop(token, None)
                self._operation_results = {
                    identity: record
                    for identity, record in self._operation_results.items()
                    if record[3] != token
                }

    def close(self) -> None:
        with self._lock:
            tokens = list(self._sessions)
        for token in tokens:
            with suppress(Exception):
                self._abort_session(
                    {
                        "telemetry_session_token": token,
                        "reason": "telemetryd shutdown",
                    }
                )
        with self._lock:
            self._sessions.clear()
            self._operation_results.clear()
        if self._state_tmp is not None:
            self._state_tmp.cleanup()

    def _session(self, payload: Mapping[str, Any]) -> _Session:
        token = _string(payload, "telemetry_session_token")
        with self._lock:
            try:
                return self._sessions[token]
            except KeyError as exc:
                raise TelemetryProtocolError("unknown telemetry session") from exc

    @staticmethod
    def _call(session: _Session, payload: Mapping[str, Any]) -> _Call:
        token = _string(payload, "telemetry_call_token")
        try:
            return session.calls[token]
        except KeyError as exc:
            raise TelemetryProtocolError("unknown telemetry call") from exc

    @staticmethod
    def _require_open(session: _Session) -> None:
        if session.final_result is not None:
            raise TelemetryProtocolError("telemetry session is finalized")


def _string(
    payload: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TelemetryProtocolError(f"{name} must be a string")
    return value


def _sha256(payload: Mapping[str, Any], name: str) -> str:
    value = _string(payload, name)
    if len(value) != 64:
        raise TelemetryProtocolError(f"{name} must be a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise TelemetryProtocolError(f"{name} must be a SHA-256 digest") from exc
    return value


def _finite_number(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise TelemetryProtocolError(f"{name} must be finite")
    return float(value)


def _positive_integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TelemetryProtocolError(f"{name} must be a positive integer")
    return value


def _static_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryProtocolError("static_call_plan must be an object")
    expected = {
        "canonical_command",
        "parsed",
        "source_tool_call_id",
        "source_command",
        "source_tool_result",
    }
    if set(value) != expected:
        raise TelemetryProtocolError("static_call_plan fields are invalid")
    if not all(
        isinstance(value[name], str)
        for name in (
            "canonical_command",
            "source_tool_call_id",
            "source_command",
            "source_tool_result",
        )
    ):
        raise TelemetryProtocolError("static_call_plan string fields are invalid")
    parsed = value["parsed"]
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "clauses",
        "control_edges",
        "parse_failed",
    }:
        raise TelemetryProtocolError("parsed command fields are invalid")
    if not isinstance(parsed["clauses"], list) or not isinstance(
        parsed["control_edges"], list
    ):
        raise TelemetryProtocolError("parsed command lists are invalid")
    if not isinstance(parsed["parse_failed"], bool):
        raise TelemetryProtocolError("parsed command parse_failed is invalid")
    return {
        **dict(value),
        "parsed": {
            "clauses": [dict(clause) for clause in parsed["clauses"]],
            "control_edges": [dict(edge) for edge in parsed["control_edges"]],
            "parse_failed": parsed["parse_failed"],
        },
    }


def _provisional_call(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "call_id": str(summary.get("tool_call_id") or ""),
        "telemetry_eligible": summary.get("eligible_for_kb") is True,
        "telemetry_status": str(summary.get("telemetry_quality") or "unavailable"),
        "invalid_reasons": list(summary.get("invalid_reasons") or []),
        "clauses": _normalized_clauses(summary),
        "session_finalization": "pending",
    }


def _normalized_clauses(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = {
        "bin",
        "argv",
        "ts_start",
        "ts_end",
        "latency_ms",
        "peak_cpu_cores",
        "sampled_peak_rss_mb",
        "cpu_ns_cumulative",
        "in_loop",
        "in_pipe",
        "in_subst",
        "pipeline_position",
        "disk_io",
        "availability",
        "mapping_evidence",
        "telemetry_quality",
        "eligible_for_kb",
    }
    return [
        {key: value for key, value in clause.items() if key in allowed}
        for clause in summary.get("clauses", [])
        if isinstance(clause, Mapping)
    ]


def _normalized_observation(
    session: _Session,
    call: _Call,
    summary: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    clauses = _normalized_clauses(summary)
    starts = [
        float(row["ts_start"]) for row in clauses if row.get("ts_start") is not None
    ]
    ends = [float(row["ts_end"]) for row in clauses if row.get("ts_end") is not None]
    session_healthy = (
        artifact.get("telemetry_quality") == "ok"
        and artifact.get("collection_validity") == "valid"
        and artifact.get("cleanup") == "ok"
    )
    telemetry_eligible = session_healthy and summary.get("eligible_for_kb") is True
    reasons = list(summary.get("invalid_reasons") or [])
    if not session_healthy:
        reasons.append(
            {
                "kind": "session_unavailable",
                "detail": "collector loss, health, or cleanup gate failed",
            }
        )
    return {
        "observation_id": call.observation_id,
        "run_id": session.run_id,
        "trace_id": session.trace_id,
        "call_id": call.call_id,
        "command_digest": call.command_digest,
        "observation_interval": {
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        },
        "clauses": clauses,
        "attribution_valid": summary.get("telemetry_quality") == "ok",
        "telemetry_eligible": telemetry_eligible,
        "invalid_reasons": reasons,
        "loss_counters": dict(summary.get("telemetry_loss") or {}),
        "collector_health": (
            artifact.get("collector", {}).get("health")
            if isinstance(artifact.get("collector"), Mapping)
            else "unavailable"
        ),
        "cleanup_status": artifact.get("cleanup"),
        "formal_completeness": artifact.get("formal_completeness"),
    }


def _session_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    collector = artifact.get("collector")
    coverage = artifact.get("call_coverage")
    integrity = artifact.get("integrity")
    return {
        "collector_health": (
            collector.get("health") if isinstance(collector, Mapping) else "unavailable"
        ),
        "formal_completeness": artifact.get("formal_completeness"),
        "collection_validity": artifact.get("collection_validity"),
        "cleanup_status": artifact.get("cleanup"),
        "eligible_call_count": (
            int(coverage.get("eligible_call_count", 0))
            if isinstance(coverage, Mapping)
            else 0
        ),
        "withheld_call_count": (
            int(coverage.get("withheld_call_count", 0))
            if isinstance(coverage, Mapping)
            else 0
        ),
        "loss_counters": dict(artifact.get("telemetry_loss_total") or {}),
        "errors": (
            list(integrity.get("errors") or [])
            if isinstance(integrity, Mapping)
            else []
        ),
    }


class TelemetryServer(StrictUnixServer):
    def __init__(
        self,
        socket_path: str | Path,
        *,
        service: TelemetryService,
        allowed_uids: set[int],
        request_timeout_s: float = DEFAULT_TIMEOUT_S,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        socket_mode: int = 0o660,
        socket_gid: int | None = None,
    ) -> None:
        self.service = service
        super().__init__(
            socket_path,
            protocol_version=TELEMETRY_PROTOCOL_VERSION,
            protocol_error=TelemetryProtocolError,
            dispatch=service.dispatch,
            allowed_uids=allowed_uids,
            request_timeout_s=request_timeout_s,
            max_message_bytes=max_message_bytes,
            socket_mode=socket_mode,
            socket_gid=socket_gid,
            socket_dir_mode=0o750,
            socket_dir_gid=socket_gid,
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
    if os.geteuid() != 0:
        raise PermissionError("telemetryd must run as root")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--allowed-uid", type=int, required=True)
    parser.add_argument("--socket-gid", type=int)
    parser.add_argument("--socket-mode", type=_parse_mode, default=0o660)
    parser.add_argument("--container-runtime", choices=["docker", "podman"])
    parser.add_argument("--state-dir", type=Path)
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
    if args.allowed_uid < 0:
        raise ValueError("allowed UID must be non-negative")
    service = TelemetryService(
        container_executable=args.container_runtime,
        state_dir=args.state_dir,
    )
    with TelemetryServer(
        args.socket,
        service=service,
        allowed_uids={args.allowed_uid},
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
    "DEFAULT_OBSERVATION_TTL_S",
    "TelemetryServer",
    "TelemetryService",
    "main",
]
