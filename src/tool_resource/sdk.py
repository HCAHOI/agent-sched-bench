"""Cold-start, prediction, observation, and causal update facade."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from tool_resource.runtime_kb import (
    ClauseObservation,
    ClauseResourceKB,
    CommandLatencyBucketPrediction,
    LatencyBuckets,
)
from tool_resource.sidecar_protocol import (
    SidecarProtocolError,
    SidecarTransport,
    UnixSocketTransport,
)


@dataclass(frozen=True)
class DockerExecutionContext:
    """Docker lifecycle context supplied by the command executor."""

    container_id: str
    container_executable: str
    repo: str
    artifact_path: Path
    source_actions: Sequence[Mapping[str, Any]] = ()
    sidecar_socket: Path | None = None
    sidecar_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.container_id:
            raise ValueError("container_id is required")
        if not self.container_executable:
            raise ValueError("container_executable is required")
        if not self.repo:
            raise ValueError("repo is required")
        if not math.isfinite(self.sidecar_timeout_s) or self.sidecar_timeout_s <= 0:
            raise ValueError("sidecar_timeout_s must be finite and positive")


@dataclass(frozen=True)
class CommandObservationToken:
    """Opaque collector delimiter returned before Docker execution."""

    tool_call_id: str
    command: str
    _collector_token: Any = field(repr=False)
    _start_error: str | None = field(default=None, repr=False)


class DockerCommandObserver:
    """Fail-isolated client for a privileged Stage-2 sidecar session."""

    def __init__(
        self,
        context: DockerExecutionContext,
        transport: SidecarTransport | None,
        session_id: str | None,
        attach_error: str | None = None,
    ) -> None:
        self.context = context
        self._transport = transport
        self._session_id = session_id
        self._attach_error = attach_error
        self._transport_error: str | None = None
        self._calls: list[dict[str, Any]] = []
        self._final_artifact: dict[str, Any] | None = None

    @classmethod
    def attach(
        cls,
        context: DockerExecutionContext,
        transport: SidecarTransport | None = None,
    ) -> DockerCommandObserver:
        """Open a remote collector session; connection failure stays local."""

        if transport is None:
            if context.sidecar_socket is None:
                return cls.unavailable(context, "sidecar socket is not configured")
            transport = UnixSocketTransport(
                context.sidecar_socket,
                timeout_s=context.sidecar_timeout_s,
            )
        try:
            result = transport.request(
                "open",
                {
                    "container_id": context.container_id,
                    "repo": context.repo,
                    "source_actions": list(context.source_actions),
                },
            )
            session_id = result.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise SidecarProtocolError("sidecar returned no session id")
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
            return cls.unavailable(
                context,
                f"sidecar attach failed: {type(exc).__name__}: {exc}",
            )
        return cls(context, transport, session_id)

    @classmethod
    def unavailable(
        cls,
        context: DockerExecutionContext,
        reason: str,
    ) -> DockerCommandObserver:
        return cls(context, None, None, reason)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    @property
    def final_artifact(self) -> dict[str, Any] | None:
        return self._final_artifact

    def start(self, tool_call_id: str, command: str) -> CommandObservationToken:
        """Start observation immediately before the Docker runner executes."""

        if self._session_id is None or self._transport is None:
            return CommandObservationToken(
                tool_call_id,
                command,
                None,
                self._attach_error or "sidecar is unavailable",
            )
        try:
            result = self._transport.request(
                "begin",
                {
                    "session_id": self._session_id,
                    "tool_call_id": tool_call_id,
                    "command": command,
                },
            )
            token = result.get("token_id")
            if not isinstance(token, str) or not token:
                raise SidecarProtocolError("sidecar returned no token id")
            error = None
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
            token = None
            error = self._record_error("start", exc)
        return CommandObservationToken(tool_call_id, command, token, error)

    begin_tool_call = start

    def finish(
        self,
        token: CommandObservationToken,
        *,
        replay_response: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish observation after Docker returns without raising into it."""

        if token._start_error is not None:
            return self._failure_summary(token, token._start_error)
        try:
            if self._session_id is None or self._transport is None:
                raise RuntimeError("sidecar session is unavailable")
            result = self._transport.request(
                "finish",
                {
                    "session_id": self._session_id,
                    "token_id": token._collector_token,
                    "replay_response": (
                        None if replay_response is None else dict(replay_response)
                    ),
                },
            )
            summary = self._result_mapping(result, "call")
            self._calls.append(summary)
            return summary
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
            return self._failure_summary(token, self._record_error("finish", exc))

    finish_tool_call = finish

    def record_safety_guard_blocked(
        self,
        tool_call_id: str,
        command: str,
        replay_result: str,
    ) -> dict[str, Any]:
        try:
            if self._session_id is None or self._transport is None:
                raise RuntimeError(self._attach_error or "sidecar is unavailable")
            result = self._transport.request(
                "safety_guard",
                {
                    "session_id": self._session_id,
                    "tool_call_id": tool_call_id,
                    "command": command,
                    "replay_result": replay_result,
                },
            )
            summary = self._result_mapping(result, "call")
            self._calls.append(summary)
            return summary
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
            token = CommandObservationToken(tool_call_id, command, None)
            return self._failure_summary(
                token,
                self._record_error("safety_guard", exc),
            )

    def add_integrity_error(self, message: str) -> None:
        if self._session_id is None or self._transport is None:
            self._transport_error = self._transport_error or message
            return
        try:
            self._transport.request(
                "add_error",
                {"session_id": self._session_id, "message": message},
            )
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
            self._transport_error = self._transport_error or self._error_message(
                "add_error", exc
            )

    def finalize(self, *, replay_execution: str = "completed") -> str | None:
        if self._session_id is None or self._transport is None:
            return self._attach_error or "sidecar is unavailable"
        prior_error = self._transport_error
        session_id, self._session_id = self._session_id, None
        try:
            result = self._transport.request(
                "finalize",
                {
                    "session_id": session_id,
                    "replay_execution": replay_execution,
                },
            )
            artifact = self._result_mapping(result, "artifact")
            if prior_error is not None:
                return prior_error
            _write_artifact(self.context.artifact_path, artifact)
            self._final_artifact = artifact
            return None
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
            return prior_error or self._error_message("finalize", exc)

    def _record_error(self, phase: str, exc: BaseException) -> str:
        message = self._error_message(phase, exc)
        self._transport_error = self._transport_error or message
        self.add_integrity_error(message)
        return message

    @staticmethod
    def _error_message(phase: str, exc: BaseException) -> str:
        return f"telemetry {phase} failed: {type(exc).__name__}: {exc}"

    @staticmethod
    def _result_mapping(result: Mapping[str, Any], name: str) -> dict[str, Any]:
        value = result.get(name)
        if not isinstance(value, Mapping):
            raise SidecarProtocolError(f"sidecar returned invalid {name}")
        return dict(value)

    def _failure_summary(
        self,
        token: CommandObservationToken,
        message: str,
    ) -> dict[str, Any]:
        summary = {
            "version": 2,
            "tool_call_id": token.tool_call_id,
            "tool_trace_ref": token.tool_call_id,
            "command": token.command,
            "telemetry_quality": "unavailable",
            "eligible_for_kb": False,
            "invalid_reasons": [{"kind": "observer_failure", "detail": message}],
            "clauses": [],
            "integrity": {"status": "failed", "errors": [message]},
        }
        if not self._calls or self._calls[-1] != summary:
            self._calls.append(summary)
        return summary


@dataclass(frozen=True)
class CommandRun:
    """Prediction and observer state returned before Docker execution."""

    tool_call_id: str
    command: str
    ts_start: float
    prediction: CommandLatencyBucketPrediction | None
    prediction_error: str | None
    _owner: object = field(repr=False)
    _run_id: int = field(repr=False)
    _observer: DockerCommandObserver = field(repr=False)
    _observation_token: CommandObservationToken = field(repr=False)


@dataclass(frozen=True)
class CommandResult:
    """Workload result plus finalized telemetry and KB update status."""

    run: CommandRun
    workload_result: Mapping[str, Any] | None
    call_telemetry: Mapping[str, Any]
    telemetry_artifact: Mapping[str, Any] | None
    kb_observations_added: int
    kb_update_error: str | None


@dataclass(frozen=True)
class ColdStartReport:
    """Accepted and rejected Stage-2 inputs used to initialize the SDK."""

    artifacts_seen: int
    artifacts_accepted: int
    calls_seen: int
    eligible_calls_loaded: int
    calls_withheld: int
    observations_loaded: int
    rejections: tuple[str, ...]

    @property
    def artifacts_rejected(self) -> int:
        return len(self.rejections)


class ToolResourceSDK:
    """Cold-start knowledge plus one complete Docker command transaction."""

    def __init__(
        self,
        kb: ClauseResourceKB,
        latency_buckets: LatencyBuckets,
        cold_start_report: ColdStartReport | None = None,
        *,
        transport_factory: Callable[[DockerExecutionContext], SidecarTransport]
        | None = None,
    ) -> None:
        self._kb = kb
        self.latency_buckets = latency_buckets
        self.cold_start_report = cold_start_report
        self._transport_factory = transport_factory
        self._owner = object()
        self._next_run_id = 0
        self._pending_run_ids: set[int] = set()

    @classmethod
    def from_traces(
        cls,
        trace_paths: str | Path | Iterable[str | Path],
        latency_buckets: LatencyBuckets,
        *,
        transport_factory: Callable[[DockerExecutionContext], SidecarTransport]
        | None = None,
    ) -> ToolResourceSDK:
        """Fit frozen public knowledge from valid Stage-2 telemetry artifacts."""

        paths = (
            [Path(trace_paths)]
            if isinstance(trace_paths, (str, Path))
            else [Path(path) for path in trace_paths]
        )
        if not paths:
            raise ValueError("at least one cold-start trace is required")
        observations: list[ClauseObservation] = []
        rejections: list[str] = []
        accepted = 0
        calls_seen = 0
        eligible_calls_loaded = 0
        for path in paths:
            try:
                artifact = _load_valid_artifact(path)
                repo = str(artifact.get("provenance", {}).get("repo") or "public")
                artifact_observations: list[ClauseObservation] = []
                artifact_eligible_calls = 0
                for call in artifact["calls"]:
                    if call.get("eligible_for_kb") is not True:
                        continue
                    artifact_eligible_calls += 1
                    artifact_observations.extend(
                        _observations_from_call(
                            repo,
                            call,
                            require_timestamps=False,
                        )
                    )
            except (TypeError, ValueError) as exc:
                detail = str(exc)
                rejections.append(
                    detail if detail.startswith(str(path)) else f"{path}: {detail}"
                )
                continue
            observations.extend(artifact_observations)
            calls_seen += len(artifact["calls"])
            eligible_calls_loaded += artifact_eligible_calls
            accepted += 1
        if accepted == 0:
            detail = rejections[0] if rejections else "no inputs"
            raise ValueError(f"no valid cold-start telemetry artifacts: {detail}")
        return cls(
            ClauseResourceKB.fit_public(observations),
            latency_buckets,
            ColdStartReport(
                artifacts_seen=len(paths),
                artifacts_accepted=accepted,
                calls_seen=calls_seen,
                eligible_calls_loaded=eligible_calls_loaded,
                calls_withheld=calls_seen - eligible_calls_loaded,
                observations_loaded=len(observations),
                rejections=tuple(rejections),
            ),
            transport_factory=transport_factory,
        )

    def start_command(
        self,
        context: DockerExecutionContext,
        tool_call_id: str,
        command: str,
        *,
        ts_start: float | None = None,
    ) -> CommandRun:
        """Parse, query, predict, and start telemetry before Docker execution."""

        if context.artifact_path.exists():
            raise ValueError(
                f"command telemetry artifact already exists: {context.artifact_path}"
            )
        query_ts = time.time() if ts_start is None else float(ts_start)
        if not math.isfinite(query_ts):
            raise ValueError("ts_start must be finite")
        try:
            prediction = self._kb.predict_command_latency_bucket(
                context.repo,
                command,
                query_ts,
                self.latency_buckets,
            )
            prediction_error = None
        except Exception as exc:
            prediction = None
            prediction_error = f"{type(exc).__name__}: {exc}"
        if self._transport_factory is None:
            observer = DockerCommandObserver.attach(context)
        else:
            try:
                observer = DockerCommandObserver.attach(
                    context,
                    self._transport_factory(context),
                )
            except Exception as exc:  # noqa: BLE001 - telemetry is fail-isolated
                observer = DockerCommandObserver.unavailable(
                    context,
                    f"sidecar transport failed: {type(exc).__name__}: {exc}",
                )
        token = observer.start(tool_call_id, command)
        run_id = self._next_run_id
        self._next_run_id += 1
        self._pending_run_ids.add(run_id)
        return CommandRun(
            tool_call_id=tool_call_id,
            command=command,
            ts_start=query_ts,
            prediction=prediction,
            prediction_error=prediction_error,
            _owner=self._owner,
            _run_id=run_id,
            _observer=observer,
            _observation_token=token,
        )

    def finish_command(
        self,
        run: CommandRun,
        workload_result: Mapping[str, Any] | None,
        *,
        replay_execution: str = "completed",
    ) -> CommandResult:
        """Finalize telemetry, then update the KB only from a valid artifact."""

        if run._owner is not self._owner:
            raise ValueError("command run belongs to a different SDK")
        if run._run_id not in self._pending_run_ids:
            raise ValueError("command run has already been finished")
        self._pending_run_ids.remove(run._run_id)
        call_telemetry = run._observer.finish(
            run._observation_token,
            replay_response=workload_result,
        )
        finalize_error = run._observer.finalize(replay_execution=replay_execution)
        artifact: dict[str, Any] | None = None
        try:
            if finalize_error is not None:
                raise ValueError(finalize_error)
            artifact = run._observer.final_artifact
            if artifact is None:
                raise ValueError("sidecar returned no finalized artifact")
            _validate_artifact(
                run._observer.context.artifact_path,
                artifact,
                expected_container_id=run._observer.context.container_id,
                expected_repo=run._observer.context.repo,
            )
            calls = [
                call
                for call in artifact["calls"]
                if call.get("tool_call_id") == run.tool_call_id
            ]
            if len(calls) != 1:
                raise ValueError(
                    f"final artifact has {len(calls)} calls for {run.tool_call_id!r}"
                )
            call = calls[0]
            if call.get("command") != run.command:
                raise ValueError("final artifact command does not match the request")
            if call.get("eligible_for_kb") is not True:
                raise ValueError("final command telemetry is not eligible for KB")
            observations = _observations_from_call(
                run._observer.context.repo,
                call,
                require_timestamps=True,
            )
            for observation in observations:
                self._kb.observe_completed_clause(observation)
            update_error = None
        except Exception as exc:
            observations = []
            update_error = f"{type(exc).__name__}: {exc}"
        return CommandResult(
            run=run,
            workload_result=workload_result,
            call_telemetry=call_telemetry,
            telemetry_artifact=artifact,
            kb_observations_added=len(observations),
            kb_update_error=update_error,
        )


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
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
                dict(artifact),
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


def _load_valid_artifact(path: Path) -> dict[str, Any]:
    artifact = _read_artifact(path)
    _validate_artifact(path, artifact)
    return artifact


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read telemetry artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("version") != 2:
        raise ValueError(f"{path}: expected Stage-2 telemetry artifact version 2")
    return artifact


def _validate_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    expected_container_id: str | None = None,
    expected_repo: str | None = None,
) -> None:
    if artifact.get("mode") != "clause":
        raise ValueError(f"{path}: expected clause telemetry mode")
    if artifact.get("replay_execution") not in {"completed", "failed"}:
        raise ValueError(f"{path}: replay execution is incomplete")
    if artifact.get("cleanup") != "ok":
        raise ValueError(f"{path}: collector cleanup is not ok")
    calls = artifact.get("calls")
    if not isinstance(calls, list) or not all(
        isinstance(call, Mapping) for call in calls
    ):
        raise ValueError(f"{path}: calls must be a list of mappings")
    status_model = artifact.get("status_model")
    if status_model == "call_granular_v1":
        if artifact.get("telemetry_quality") != "ok":
            raise ValueError(f"{path}: collector telemetry is unavailable")
        if artifact.get("collection_validity") != "valid":
            raise ValueError(f"{path}: collection is not valid")
        if artifact.get("formal_completeness") not in {"complete", "partial"}:
            raise ValueError(f"{path}: formal completeness is unavailable")
        integrity = artifact.get("integrity")
        if not isinstance(integrity, Mapping) or integrity.get("status") != "ok":
            raise ValueError(f"{path}: collector integrity is not ok")
    elif status_model is not None:
        raise ValueError(f"{path}: unsupported status model {status_model!r}")
    elif not _legacy_collector_healthy(artifact):
        raise ValueError(f"{path}: legacy collector health is not usable")
    if (
        expected_container_id is not None
        and artifact.get("container_id") != expected_container_id
    ):
        raise ValueError(f"{path}: container identity does not match the request")
    provenance = artifact.get("provenance")
    if expected_repo is not None and (
        not isinstance(provenance, Mapping) or provenance.get("repo") != expected_repo
    ):
        raise ValueError(f"{path}: repository identity does not match the request")


def _legacy_collector_healthy(artifact: Mapping[str, Any]) -> bool:
    collector = artifact.get("collector")
    loss = artifact.get("telemetry_loss_total")
    return (
        artifact.get("telemetry_quality") in {"ok", "invalid"}
        and artifact.get("cleanup") == "ok"
        and isinstance(collector, Mapping)
        and collector.get("state_before_close") == "active"
        and collector.get("unavailable_call_count") == 0
        and isinstance(loss, Mapping)
        and loss.get("total") == 0
    )


def _observations_from_call(
    repo: str,
    call: Mapping[str, Any],
    *,
    require_timestamps: bool,
) -> list[ClauseObservation]:
    rows = call.get("clauses")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("eligible telemetry has invalid clauses")
    observations: list[ClauseObservation] = []
    for row in rows:
        availability = row.get("availability")
        if not isinstance(availability, Mapping):
            raise ValueError("eligible clause has invalid availability")
        if availability.get("latency") == "ok":
            observations.append(
                _observation_from_clause(
                    repo,
                    row,
                    require_timestamps=require_timestamps,
                )
            )
    return observations


def _observation_from_clause(
    repo: str,
    row: Mapping[str, Any],
    *,
    require_timestamps: bool,
) -> ClauseObservation:
    latency_ms = _required_nonnegative_float(row.get("latency_ms"), "latency_ms")
    raw_start = row.get("ts_start")
    raw_end = row.get("ts_end")
    if raw_start is None and raw_end is None and not require_timestamps:
        ts_start = 0.0
        ts_end = latency_ms / 1000.0
    else:
        ts_start = _required_finite_float(raw_start, "ts_start")
        ts_end = _required_finite_float(raw_end, "ts_end")
    argv = row.get("argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("eligible clause has invalid argv")
    bin_ = row.get("bin")
    if not isinstance(bin_, str) or not bin_:
        raise ValueError("eligible clause has invalid bin")
    return ClauseObservation(
        repo=repo,
        bin=bin_,
        argv=tuple(argv),
        ts_start=ts_start,
        ts_end=ts_end,
        latency_ms=latency_ms,
        peak_cpu_cores=_optional_finite_float(row.get("peak_cpu_cores")),
        sampled_peak_rss_mb=_optional_finite_float(row.get("sampled_peak_rss_mb")),
        cpu_ns_cumulative=_optional_nonnegative_int(row.get("cpu_ns_cumulative")),
        in_loop=bool(row.get("in_loop", False)),
        in_pipe=bool(row.get("in_pipe", False)),
        in_subst=bool(row.get("in_subst", False)),
        pipeline_position=int(row.get("pipeline_position", -1)),
    )


def _required_finite_float(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"eligible clause has invalid {name}")
    return float(value)


def _required_nonnegative_float(value: Any, name: str) -> float:
    result = _required_finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"eligible clause has negative {name}")
    return result


def _optional_finite_float(value: Any) -> float | None:
    return None if value is None else _required_finite_float(value, "measurement")


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("eligible clause has invalid cumulative CPU")
    return value


__all__ = [
    "CommandObservationToken",
    "CommandResult",
    "CommandRun",
    "DockerCommandObserver",
    "DockerExecutionContext",
    "ToolResourceSDK",
]
