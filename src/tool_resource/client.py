"""Thin unprivileged Resource Protocol client for workload runners."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from tool_resource.profile import ResourceProfile
from tool_resource.resource_protocol import ResourceTransport, ResourceUnixTransport


@dataclass(frozen=True)
class ResourceCallToken:
    call_id: str
    command: str
    prediction: Mapping[str, Any] | None
    _call_token: str | None = field(repr=False)
    _start_error: str | None = field(default=None, repr=False)


class ResourceRun:
    """One collection/simulation run pinned to one workspace scope."""

    def __init__(
        self,
        *,
        profile: ResourceProfile,
        run_id: str,
        workspace_scope: str,
        manifest_path: Path,
        transport: ResourceTransport | None,
        run_token: str | None,
        open_result: Mapping[str, Any] | None = None,
        open_error: str | None = None,
    ) -> None:
        self.profile = profile
        self.run_id = run_id
        self.workspace_scope = workspace_scope
        self.manifest_path = manifest_path
        self._transport = transport
        self._run_token = run_token
        self._open_result = dict(open_result or {})
        self._open_error = open_error
        self._closed = False
        self._result: dict[str, Any] | None = None
        self._traces: list[ResourceTrace] = []

    @classmethod
    def open(
        cls,
        profile: ResourceProfile | str | Path,
        *,
        run_id: str,
        workspace_scope: str,
        manifest_path: str | Path,
        transport: ResourceTransport | None = None,
    ) -> ResourceRun:
        resolved = (
            profile
            if isinstance(profile, ResourceProfile)
            else ResourceProfile.load(profile)
        )
        transport = transport or ResourceUnixTransport(resolved.socket_path())
        try:
            result = transport.request(
                "OpenRun",
                resolved.open_run_payload(
                    run_id=run_id,
                    workspace_scope=workspace_scope,
                ),
            )
            _require_response_fields(
                result,
                {
                    "run_token",
                    "pinned_snapshot_id",
                    "canonicalizer_version",
                    "store_schema_version",
                    "capabilities",
                },
                "OpenRun",
            )
            run_token = _response_token(result, "run_token")
        except Exception as exc:  # noqa: BLE001 - workload continues
            return cls(
                profile=resolved,
                run_id=run_id,
                workspace_scope=workspace_scope,
                manifest_path=Path(manifest_path),
                transport=None,
                run_token=None,
                open_error=f"resource run open failed: {type(exc).__name__}: {exc}",
            )
        return cls(
            profile=resolved,
            run_id=run_id,
            workspace_scope=workspace_scope,
            manifest_path=Path(manifest_path),
            transport=transport,
            run_token=run_token,
            open_result=result,
        )

    @property
    def run_token(self) -> str | None:
        return self._run_token

    @property
    def result(self) -> dict[str, Any] | None:
        return self._result

    def open_trace(
        self,
        *,
        trace_id: str,
        container_runtime: str,
        container_id: str,
        artifact_path: str | Path,
        expected_calls: Sequence[Mapping[str, str]] = (),
    ) -> ResourceTrace:
        if self._transport is None or self._run_token is None:
            trace = ResourceTrace(
                artifact_path=Path(artifact_path),
                transport=None,
                trace_token=None,
                open_error=self._open_error or "resource-agentd is unavailable",
            )
        else:
            trace = ResourceTrace.open(
                self.profile,
                run_token=self._run_token,
                trace_id=trace_id,
                container_runtime=container_runtime,
                container_id=container_id,
                artifact_path=artifact_path,
                expected_calls=expected_calls,
                transport=self._transport,
            )
        self._traces.append(trace)
        return trace

    def finalize(self, *, workload_status: str = "completed") -> str | None:
        if self._closed:
            if self._result is None:
                return self._open_error or "resource-agentd is unavailable"
            return (
                None
                if self._result["evidence_valid"] is True
                else "resource evidence invalid"
            )
        self._closed = True
        if self._transport is None or self._run_token is None:
            error = self._open_error or "resource-agentd is unavailable"
            manifest_error = self._record_unavailable(workload_status, error)
            return f"{error}; {manifest_error}" if manifest_error else error
        try:
            operation = (
                "AbortRun"
                if any(trace._open_error is not None for trace in self._traces)
                else "CloseRun"
            )
            result = self._transport.request(
                operation,
                {
                    "run_token": self._run_token,
                    **(
                        {"reason": "one or more resource traces failed to open"}
                        if operation == "AbortRun"
                        else {"workload_status": workload_status}
                    ),
                },
            )
            _require_response_fields(
                result,
                {
                    "workload_status",
                    "telemetry_valid",
                    "evidence_valid",
                    "promoted_observation_count",
                    "run_manifest",
                },
                operation,
            )
            if not isinstance(result["evidence_valid"], bool) or not isinstance(
                result["run_manifest"], Mapping
            ):
                raise RuntimeError("resource-agentd returned an invalid run result")
            self._result = dict(result)
            _write_json(
                self.manifest_path,
                {
                    "version": 1,
                    **self._result,
                },
            )
            return (
                None
                if result["evidence_valid"] is True
                else "resource evidence invalid"
            )
        except Exception as exc:  # noqa: BLE001 - workloads already completed
            error = f"resource run finalize failed: {type(exc).__name__}: {exc}"
            try:
                self._transport.request(
                    "AbortRun",
                    {"run_token": self._run_token, "reason": error},
                )
            except Exception:
                pass
            manifest_error = self._record_unavailable(workload_status, error)
            return f"{error}; {manifest_error}" if manifest_error else error

    def _record_unavailable(
        self,
        workload_status: str,
        error: str,
    ) -> str | None:
        evidence_valid = (
            self.profile.telemetry_requirement == "best_effort"
            or "observe" not in self.profile.behavior
        )
        self._result = {
            "workload_status": workload_status,
            "telemetry_valid": False,
            "evidence_valid": evidence_valid,
            "promoted_observation_count": 0,
            "run_manifest": {
                "run_id": self.run_id,
                "workspace_scope": self.workspace_scope,
                "pinned_snapshot_id": self._open_result.get("pinned_snapshot_id"),
                "resulting_snapshot_id": None,
                "canonicalizer_version": self._open_result.get("canonicalizer_version"),
                "store_schema_version": self._open_result.get("store_schema_version"),
                "latency_bucket_edges_ms": list(self.profile.latency_bucket_edges_ms),
                "update_policy": self.profile.update_policy,
                "telemetry_requirement": self.profile.telemetry_requirement,
                "behavior": self.profile.behavior,
                "resource_service_status": "unavailable",
                "error": error,
            },
        }
        try:
            _write_json(self.manifest_path, {"version": 1, **self._result})
        except Exception as exc:  # noqa: BLE001 - workload result must survive
            return f"run manifest write failed: {type(exc).__name__}: {exc}"
        return None


class ResourceTrace:
    """One fail-isolated run/trace session; never owns a service process."""

    def __init__(
        self,
        *,
        artifact_path: Path,
        transport: ResourceTransport | None,
        trace_token: str | None,
        open_error: str | None = None,
    ) -> None:
        self.artifact_path = artifact_path
        self._transport = transport
        self._trace_token = trace_token
        self._open_error = open_error
        self._errors: list[str] = []
        self._calls: list[dict[str, Any]] = []
        self._final_artifact: dict[str, Any] | None = None
        self._closed = False

    @classmethod
    def open(
        cls,
        profile: ResourceProfile | str | Path,
        *,
        run_token: str,
        trace_id: str,
        container_runtime: str,
        container_id: str,
        artifact_path: str | Path,
        expected_calls: Sequence[Mapping[str, str]] = (),
        transport: ResourceTransport | None = None,
    ) -> ResourceTrace:
        resolved = (
            profile
            if isinstance(profile, ResourceProfile)
            else ResourceProfile.load(profile)
        )
        transport = transport or ResourceUnixTransport(resolved.socket_path())
        canonical_runtime = Path(container_runtime).name
        try:
            result = transport.request(
                "OpenTrace",
                {
                    "run_token": run_token,
                    "trace_id": trace_id,
                    "container_runtime": canonical_runtime,
                    "container_id": container_id,
                    "repo_metadata": {},
                    "expected_calls": [dict(call) for call in expected_calls],
                },
            )
            _require_response_fields(
                result,
                {"trace_token", "telemetry_status"},
                "OpenTrace",
            )
            trace_token = _response_token(result, "trace_token")
        except Exception as exc:  # noqa: BLE001 - workload continues
            return cls(
                artifact_path=Path(artifact_path),
                transport=None,
                trace_token=None,
                open_error=f"resource trace open failed: {type(exc).__name__}: {exc}",
            )
        return cls(
            artifact_path=Path(artifact_path),
            transport=transport,
            trace_token=trace_token,
        )

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    @property
    def final_artifact(self) -> dict[str, Any] | None:
        return self._final_artifact

    def begin_tool_call(self, call_id: str, command: str) -> ResourceCallToken:
        if self._transport is None or self._trace_token is None:
            return ResourceCallToken(
                call_id,
                command,
                None,
                None,
                self._open_error or "resource-agentd is unavailable",
            )
        try:
            result = self._transport.request(
                "BeginCall",
                {
                    "trace_token": self._trace_token,
                    "call_id": call_id,
                    "command": command,
                    "query_timestamp": time.time(),
                },
            )
            _require_response_fields(
                result,
                {
                    "call_token",
                    "prediction",
                    "resource_classifications",
                    "probability_by_bucket",
                    "selected_scope",
                    "fallback_path",
                    "fallback_reason",
                    "evidence_count",
                    "evidence_recency",
                    "pinned_snapshot_id",
                    "canonicalizer_version",
                    "telemetry_status",
                },
                "BeginCall",
            )
            call_token = _response_token(result, "call_token")
            prediction = result.get("prediction")
            if not isinstance(prediction, Mapping):
                prediction = None
            elif isinstance(result.get("resource_classifications"), Mapping):
                prediction = {
                    **dict(prediction),
                    "resource_classifications": dict(
                        result["resource_classifications"]
                    ),
                }
            return ResourceCallToken(
                call_id,
                command,
                None if prediction is None else dict(prediction),
                call_token,
            )
        except Exception as exc:  # noqa: BLE001 - workload continues
            error = self._record_error("begin", exc)
            return ResourceCallToken(call_id, command, None, None, error)

    def wait_ready(self) -> str | None:
        """Wait before workload start until trace instrumentation is settled."""

        if self._transport is None or self._trace_token is None:
            return self._open_error or "resource-agentd is unavailable"
        try:
            result = self._transport.request(
                "AwaitTraceReady",
                {"trace_token": self._trace_token},
            )
            _require_response_fields(
                result,
                {"telemetry_status"},
                "AwaitTraceReady",
            )
            status = result.get("telemetry_status")
            if status not in {"available", "not_requested"}:
                raise RuntimeError(f"telemetry setup ended with status {status!r}")
            return None
        except Exception as exc:  # noqa: BLE001 - workload must still start
            return self._record_error("setup", exc)

    def finish_tool_call(
        self,
        token: ResourceCallToken,
        *,
        replay_response: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if token._start_error is not None:
            return self._failure_call(token, token._start_error)
        try:
            if (
                self._transport is None
                or self._trace_token is None
                or token._call_token is None
            ):
                raise RuntimeError("resource trace is unavailable")
            result = self._transport.request(
                "EndCall",
                {
                    "call_token": token._call_token,
                    "workload_result": (
                        None if replay_response is None else dict(replay_response)
                    ),
                    "end_timestamp": time.time(),
                },
            )
            _require_response_fields(
                result,
                {
                    "workload_result",
                    "finalized_call_observation",
                    "telemetry_status",
                    "ingest_status",
                    "rejection_reasons",
                },
                "EndCall",
            )
            observation = result.get("finalized_call_observation")
            if not isinstance(observation, Mapping):
                raise RuntimeError("resource-agentd returned no call observation")
            call = {
                "version": 1,
                "tool_call_id": token.call_id,
                "command": token.command,
                "telemetry_quality": str(
                    observation.get("telemetry_status")
                    or result.get("telemetry_status")
                    or "unavailable"
                ),
                "eligible_for_kb": observation.get("telemetry_eligible") is True,
                "invalid_reasons": list(observation.get("invalid_reasons") or []),
                "clauses": list(observation.get("clauses") or []),
                "prediction": token.prediction,
            }
            self._calls.append(call)
            return call
        except Exception as exc:  # noqa: BLE001 - workload continues
            return self._failure_call(token, self._record_error("finish", exc))

    def record_safety_guard_blocked(
        self,
        call_id: str,
        command: str,
        workload_result: str,
    ) -> dict[str, Any]:
        token = ResourceCallToken(call_id, command, None, None)
        try:
            if self._transport is None or self._trace_token is None:
                raise RuntimeError(self._open_error or "resource trace is unavailable")
            result = self._transport.request(
                "RecordSafetyGuardBlock",
                {
                    "trace_token": self._trace_token,
                    "call_id": call_id,
                    "command": command,
                    "workload_result": workload_result,
                    "end_timestamp": time.time(),
                },
            )
            _require_response_fields(
                result,
                {
                    "workload_result",
                    "prediction",
                    "prediction_error",
                    "finalized_call_observation",
                    "telemetry_status",
                    "ingest_status",
                },
                "RecordSafetyGuardBlock",
            )
            observation = result.get("finalized_call_observation")
            if not isinstance(observation, Mapping):
                raise RuntimeError("resource-agentd returned no guard observation")
            call = {
                "version": 1,
                "tool_call_id": call_id,
                "command": command,
                "telemetry_quality": str(
                    observation.get("telemetry_status")
                    or result.get("telemetry_status")
                    or "unavailable"
                ),
                "eligible_for_kb": observation.get("telemetry_eligible") is True,
                "invalid_reasons": list(observation.get("invalid_reasons") or []),
                "clauses": list(observation.get("clauses") or []),
                "prediction": result.get("prediction"),
            }
            self._calls.append(call)
            return call
        except Exception as exc:  # noqa: BLE001 - workload continues
            return self._failure_call(
                token,
                self._record_error("safety_guard", exc),
            )

    def add_integrity_error(self, message: str) -> None:
        if message not in self._errors:
            self._errors.append(message)
        if self._transport is None or self._trace_token is None:
            return
        try:
            self._transport.request(
                "RecordTraceIntegrityFailure",
                {"trace_token": self._trace_token, "message": message},
            )
        except Exception as exc:  # noqa: BLE001 - workload continues
            self._record_error("integrity", exc)

    def finalize(self, *, replay_execution: str = "completed") -> str | None:
        if self._closed:
            return self._errors[0] if self._errors else None
        self._closed = True
        if self._transport is None or self._trace_token is None:
            return self._open_error or "resource-agentd is unavailable"
        try:
            trace_result = self._transport.request(
                "CloseTrace",
                {
                    "trace_token": self._trace_token,
                    "workload_status": replay_execution,
                },
            )
            return self._consume_trace_result(trace_result, operation="CloseTrace")
        except Exception as exc:  # noqa: BLE001 - workload already completed
            error = f"resource finalize failed: {type(exc).__name__}: {exc}"
            try:
                trace_result = self._transport.request(
                    "AbortTrace",
                    {"trace_token": self._trace_token, "reason": error},
                )
                return self._consume_trace_result(
                    trace_result,
                    operation="AbortTrace",
                )
            except Exception:
                return self._record_error("finalize", exc)

    def _consume_trace_result(
        self,
        trace_result: Mapping[str, Any],
        *,
        operation: str,
    ) -> str | None:
        expected = {
            "telemetry_status",
            "formal_completeness",
            "collection_validity",
            "call_coverage",
            "artifact",
            "errors",
        }
        if operation == "AbortTrace":
            expected.add("aborted")
        _require_response_fields(trace_result, expected, operation)
        if operation == "AbortTrace" and trace_result.get("aborted") is not True:
            raise RuntimeError("resource-agentd did not abort the trace")
        artifact = trace_result.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RuntimeError("resource-agentd returned no trace artifact")
        self._final_artifact = dict(artifact)
        calls = artifact.get("calls")
        if isinstance(calls, list) and all(isinstance(call, Mapping) for call in calls):
            self._calls = [dict(call) for call in calls]
        _write_json(self.artifact_path, self._final_artifact)
        if trace_result.get("telemetry_status") not in {"ok", "not_requested"}:
            errors = trace_result.get("errors")
            if isinstance(errors, list):
                self._errors.extend(str(error) for error in errors)
            return self._errors[0] if self._errors else "telemetry unavailable"
        return None

    def _failure_call(
        self,
        token: ResourceCallToken,
        message: str,
    ) -> dict[str, Any]:
        call = {
            "version": 1,
            "tool_call_id": token.call_id,
            "command": token.command,
            "telemetry_quality": "unavailable",
            "eligible_for_kb": False,
            "invalid_reasons": [
                {"kind": "resource_service_failure", "detail": message}
            ],
            "clauses": [],
            "prediction": token.prediction,
        }
        if not self._calls or self._calls[-1] != call:
            self._calls.append(call)
        return call

    def _record_error(self, phase: str, exc: BaseException) -> str:
        message = f"resource {phase} failed: {type(exc).__name__}: {exc}"
        if message not in self._errors:
            self._errors.append(message)
        return message


def _response_token(result: Mapping[str, Any], name: str) -> str:
    value = result.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"resource-agentd returned invalid {name}")
    return value


def _require_response_fields(
    value: Any,
    expected: set[str],
    operation: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"resource-agentd {operation} response fields are invalid")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                dict(value),
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


__all__ = ["ResourceCallToken", "ResourceRun", "ResourceTrace"]
