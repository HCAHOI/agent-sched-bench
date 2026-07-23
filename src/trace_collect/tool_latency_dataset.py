"""Extract observed tool latencies from canonical trace JSONL files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from tool_time.command import command_has_concurrent_segments
from trace_collect.tool_gap_extractor import (
    _float_field,
    _int_field,
    _tool_name,
    discover_trace_files,
)
from trace_collect.trace_data import TraceData


MISSING_TOOL_NAME = "__missing_tool_name__"

_FIXED_CORPUS_CONFIG = {
    "fold_count": 5,
    "inner_folds": 4,
    "costs_ms": [
        500.0,
        1000.0,
        1500.0,
        2000.0,
        2500.0,
        3000.0,
        3500.0,
        4000.0,
        4500.0,
        5000.0,
    ],
    "guard_ms": 0.0,
    "min_tool_history": 1,
    "min_profile_tasks": 1,
    "command_field": "command",
    "max_prefix_depth": 4,
    "skip_leading_cd": False,
}


@dataclass(frozen=True)
class ToolLatencySample:
    """One observed tool execution latency sample."""

    sample_id: str
    source_trace: str
    task_id: str
    agent_id: str
    instance_id: str
    iteration: int
    action_id: str
    tool_name: str
    tool_call_id: str
    tool_ts_start: float
    tool_ts_end: float
    latency_ms: float
    success: bool | None
    reported_duration_ms: float | None
    tool_args: dict[str, Any] | None = None
    tool_name_missing: bool = False

    def to_json_obj(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "source_trace": self.source_trace,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "instance_id": self.instance_id,
            "iteration": self.iteration,
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_ts_start": self.tool_ts_start,
            "tool_ts_end": self.tool_ts_end,
            "latency_ms": self.latency_ms,
            "success": self.success,
        }
        if self.reported_duration_ms is not None:
            payload["reported_duration_ms"] = self.reported_duration_ms
        if self.tool_args is not None:
            payload["tool_args"] = self.tool_args
        if self.tool_name_missing:
            payload["tool_name_missing"] = True
        return payload


def extract_tool_latency_samples(
    trace_path: Path,
    *,
    agent_filter: str | None = None,
) -> list[ToolLatencySample]:
    """Extract tool latency labels from one trace.

    The output is an offline evaluation dataset. ``latency_ms`` is the observed
    label and must not be fed back as an online prediction feature.
    """

    trace = TraceData.load(trace_path, agent_filter=agent_filter)
    metadata_task_id = str(trace.metadata.get("instance_id") or "").strip()
    task_id = metadata_task_id or str(trace_path)
    samples: list[ToolLatencySample] = []
    for action in trace.actions:
        if action.get("action_type") != "tool_exec":
            continue
        data = action.get("data") or {}
        action_id = str(action.get("action_id") or "")
        agent_id = str(action.get("agent_id") or "")
        instance_id = str(
            action.get("instance_id") or trace.metadata.get("instance_id") or ""
        )
        iteration = _int_field(action, "iteration")
        ts_start = _float_field(action, "ts_start")
        ts_end = _float_field(action, "ts_end")
        if not math.isfinite(ts_start) or not math.isfinite(ts_end):
            raise ValueError(
                f"{trace_path}: tool action {action_id!r} has non-finite timestamp"
            )
        if ts_end < ts_start:
            raise ValueError(
                f"{trace_path}: tool action {action_id!r} has ts_end < ts_start"
            )
        latency_ms = (ts_end - ts_start) * 1000.0
        tool_name = _tool_name(action)
        tool_name_missing = not tool_name
        if tool_name_missing:
            tool_name = MISSING_TOOL_NAME
        tool_call_id = str(data.get("tool_call_id") or action_id)
        sample_id = f"{trace_path}:{agent_id}:{iteration}:{action_id}"
        samples.append(
            ToolLatencySample(
                sample_id=sample_id,
                source_trace=str(trace_path),
                task_id=task_id,
                tool_ts_start=ts_start,
                tool_ts_end=ts_end,
                agent_id=agent_id,
                instance_id=instance_id,
                iteration=iteration,
                action_id=action_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                latency_ms=latency_ms,
                success=_optional_bool(data.get("success")),
                reported_duration_ms=_optional_float(data.get("duration_ms")),
                tool_args=_parse_tool_args(data.get("tool_args")),
                tool_name_missing=tool_name_missing,
            )
        )
    return samples


@dataclass(frozen=True)
class SegmentLatencySample:
    """One per-atom (top-level command segment) latency sample from a replay.

    Segments come from ``tool_exec.data.segment_timeline`` (v2) emitted by the
    container replay path. ``segment_ms`` is the observed label and, like the
    parent-chain latency, must not be fed back as an online prediction feature.
    """

    sample_id: str
    source_trace: str
    task_id: str
    agent_id: str
    action_id: str
    tool_name: str
    segment_index: int
    segment_command: str
    segment_ms: float
    t_start_ms: float
    t_end_ms: float
    parent_chain_command: str | None
    parent_total_ms: float
    parent_raw_total_ms: float | None

    def to_json_obj(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "source_trace": self.source_trace,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "segment_index": self.segment_index,
            "segment_command": self.segment_command,
            "segment_ms": self.segment_ms,
            "t_start_ms": self.t_start_ms,
            "t_end_ms": self.t_end_ms,
            "parent_total_ms": self.parent_total_ms,
        }
        if self.parent_chain_command is not None:
            payload["parent_chain_command"] = self.parent_chain_command
        if self.parent_raw_total_ms is not None:
            payload["parent_raw_total_ms"] = self.parent_raw_total_ms
        return payload


def _segment_exec_command(tool_args: Any) -> str | None:
    params = _parse_tool_args(tool_args)
    if params is None:
        return None
    inner = params.get("exec")
    if isinstance(inner, dict):
        params = inner
    command = params.get("command")
    return command if isinstance(command, str) else None


def _require_number(entry: dict[str, Any], field: str, *, source: str) -> float:
    value = entry.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{source}: segment field {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}: segment field {field!r} must be finite")
    return number


def extract_segment_latency_samples(
    trace_path: Path,
    *,
    agent_filter: str | None = None,
    skip_concurrent: bool = False,
) -> list[SegmentLatencySample]:
    """Extract per-atom segment latency samples from one replayed trace.

    Malformed telemetry fails fast here (never during replay). Entries recorded
    as ``telemetry_absent`` are skipped, not errors: some execs run under a
    non-bash shell where per-segment timing is unavailable by design.

    ``skip_concurrent`` drops actions whose parent command runs its segments
    concurrently (pipelines, background jobs, loops): their segment_timeline
    bounds are fictitious by the duration-validity ceiling and can even be
    reversed (``t_end_ms < t_start_ms``), which otherwise fails fast here.
    Off by default so the strict per-segment validation is preserved for the
    canonical dataset; duration analyses that must exclude these parents
    anyway (and count them) pass ``skip_concurrent=True``.
    """

    trace = TraceData.load(trace_path, agent_filter=agent_filter)
    metadata_task_id = str(trace.metadata.get("instance_id") or "").strip()
    task_id = metadata_task_id or str(trace_path)
    samples: list[SegmentLatencySample] = []
    for action in trace.actions:
        if action.get("action_type") != "tool_exec":
            continue
        data = action.get("data") or {}
        timeline = data.get("segment_timeline")
        if timeline is None:
            continue
        action_id = str(action.get("action_id") or "")
        source = f"{trace_path}: action {action_id!r}"
        if not isinstance(timeline, dict):
            raise ValueError(f"{source}: segment_timeline must be an object")
        if timeline.get("telemetry_absent"):
            continue
        if timeline.get("version") != 2:
            raise ValueError(
                f"{source}: unsupported segment_timeline version "
                f"{timeline.get('version')!r}"
            )
        segments = timeline.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{source}: segment_timeline has no segments")
        agent_id = str(action.get("agent_id") or "")
        tool_name = _tool_name(action) or MISSING_TOOL_NAME
        parent_total_ms = _optional_float(data.get("duration_ms"))
        if parent_total_ms is None:
            parent_total_ms = (
                _float_field(action, "ts_end") - _float_field(action, "ts_start")
            ) * 1000.0
        parent_chain_command = _segment_exec_command(data.get("tool_args"))
        if (
            skip_concurrent
            and parent_chain_command is not None
            and command_has_concurrent_segments(parent_chain_command)
        ):
            continue
        raw_total = _optional_float(timeline.get("raw_total_ms"))
        for entry in segments:
            if not isinstance(entry, dict):
                raise ValueError(f"{source}: segment entry must be an object")
            segment_index = entry.get("segment_index")
            if not isinstance(segment_index, int) or isinstance(segment_index, bool):
                raise ValueError(f"{source}: segment_index must be an int")
            command_text = entry.get("command_text")
            if not isinstance(command_text, str):
                raise ValueError(f"{source}: command_text must be a string")
            t_start_ms = _require_number(entry, "t_start_ms", source=source)
            t_end_ms = _require_number(entry, "t_end_ms", source=source)
            if t_end_ms < t_start_ms:
                raise ValueError(
                    f"{source}: segment {segment_index} has t_end_ms < t_start_ms"
                )
            samples.append(
                SegmentLatencySample(
                    sample_id=f"{trace_path}:{agent_id}:{action_id}:{segment_index}",
                    source_trace=str(trace_path),
                    task_id=task_id,
                    agent_id=agent_id,
                    action_id=action_id,
                    tool_name=tool_name,
                    segment_index=segment_index,
                    segment_command=command_text,
                    segment_ms=t_end_ms - t_start_ms,
                    t_start_ms=t_start_ms,
                    t_end_ms=t_end_ms,
                    parent_chain_command=parent_chain_command,
                    parent_total_ms=parent_total_ms,
                    parent_raw_total_ms=raw_total,
                )
            )
    return samples


def extract_many_segment_latency_samples(
    trace_paths: Iterable[Path],
    *,
    agent_filter: str | None = None,
    skip_concurrent: bool = False,
) -> list[SegmentLatencySample]:
    samples: list[SegmentLatencySample] = []
    for trace_path in trace_paths:
        samples.extend(
            extract_segment_latency_samples(
                trace_path,
                agent_filter=agent_filter,
                skip_concurrent=skip_concurrent,
            )
        )
    samples.sort(
        key=lambda sample: (
            sample.source_trace,
            sample.action_id,
            sample.segment_index,
        )
    )
    if not samples:
        raise ValueError("no segment latency samples found")
    return samples


def _parse_tool_args(value: Any) -> dict[str, Any] | None:
    """Parse a tool_args payload (dict or JSON-encoded dict string).

    Model-emitted arguments can be malformed; such rows return ``None`` and
    downstream feature grouping falls back to the tool level by design.
    """

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_many_tool_latency_samples(
    trace_paths: Iterable[Path],
    *,
    agent_filter: str | None = None,
) -> list[ToolLatencySample]:
    samples: list[ToolLatencySample] = []
    for trace_path in trace_paths:
        samples.extend(
            extract_tool_latency_samples(trace_path, agent_filter=agent_filter)
        )
    samples.sort(
        key=lambda sample: (sample.source_trace, sample.tool_ts_start, sample.action_id)
    )
    if not samples:
        raise ValueError("no tool latency samples found")
    return samples


def write_tool_latency_jsonl(samples: Iterable[ToolLatencySample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(
                json.dumps(sample.to_json_obj(), ensure_ascii=False, sort_keys=True)
            )
            fh.write("\n")
            count += 1
    return count


def read_tool_latency_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            _required_text(row, "sample_id", source=f"{path}:{line_no}")
            _required_text(row, "tool_name", source=f"{path}:{line_no}")
            if "tool_name_missing" in row and not isinstance(
                row["tool_name_missing"], bool
            ):
                raise ValueError(
                    f"{path}:{line_no}: field 'tool_name_missing' must be a bool"
                )
            source_trace = _required_text(
                row,
                "source_trace",
                source=f"{path}:{line_no}",
            )
            if "task_id" in row:
                row["task_id"] = _required_text(
                    row,
                    "task_id",
                    source=f"{path}:{line_no}",
                )
            else:
                # Compatibility with latency JSONLs extracted before task_id
                # was made explicit. One trace is the safest available cluster.
                row["task_id"] = source_trace
            latency_ms = _required_nonnegative_float(
                row,
                "latency_ms",
                source=f"{path}:{line_no}",
            )
            tool_ts_start = _required_nonnegative_float(
                row,
                "tool_ts_start",
                source=f"{path}:{line_no}",
            )
            tool_ts_end = _required_nonnegative_float(
                row,
                "tool_ts_end",
                source=f"{path}:{line_no}",
            )
            if tool_ts_end < tool_ts_start:
                raise ValueError(f"{path}:{line_no}: tool_ts_end < tool_ts_start")
            row["latency_ms"] = latency_ms
            row["tool_ts_start"] = tool_ts_start
            row["tool_ts_end"] = tool_ts_end
            rows.append(row)
    if not rows:
        raise ValueError(f"empty tool latency file: {path}")
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"expected numeric optional value, got {value!r}")
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"expected bool optional value, got {value!r}")
    return value


def _required_text(row: dict[str, Any], field: str, *, source: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"{source}: missing required field {field!r}")
    if not isinstance(value, str):
        raise ValueError(f"{source}: field {field!r} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{source}: empty required field {field!r}")
    return text


def _required_nonnegative_float(
    row: dict[str, Any],
    field: str,
    *,
    source: str,
) -> float:
    value = row.get(field)
    if value is None:
        raise ValueError(f"{source}: missing required field {field!r}")
    if not isinstance(value, int | float):
        raise ValueError(f"{source}: field {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}: field {field!r} must be finite")
    if number < 0.0:
        raise ValueError(f"{source}: field {field!r} must be non-negative")
    return number


def read_task_ids(path: Path) -> list[str]:
    """Read a sorted, unique list of non-empty logical task IDs."""

    task_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not task_ids or any(not task_id for task_id in task_ids):
        raise ValueError("task_ids_file must contain non-empty task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids_file contains duplicate task IDs")
    return sorted(task_ids)


def read_tool_latency_corpus_manifest(
    path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Load shared corpus metadata and resolve its trace path."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tool-latency corpus manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("tool-latency corpus manifest schema_version must be 1")

    collection_id = payload.get("collection_id")
    if not isinstance(collection_id, str) or not collection_id.strip():
        raise ValueError("collection_id must be non-empty")
    integer_fields = (
        "expected_task_count",
        "fold_count",
        "inner_folds",
        "min_tool_history",
        "min_profile_tasks",
        "max_prefix_depth",
    )
    for field in integer_fields:
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"manifest field {field!r} must be a positive integer")
    if payload["expected_task_count"] < payload["fold_count"]:
        raise ValueError("expected_task_count must be >= fold_count")

    costs = payload.get("costs_ms")
    if not isinstance(costs, list) or not costs or any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in costs
    ):
        raise ValueError("manifest field 'costs_ms' must contain positive numbers")
    guard = payload.get("guard_ms")
    if (
        not isinstance(guard, int | float)
        or isinstance(guard, bool)
        or not math.isfinite(float(guard))
    ):
        raise ValueError("manifest field 'guard_ms' must be a finite number")
    if not isinstance(payload.get("command_field"), str):
        raise ValueError("manifest field 'command_field' must be a string")
    if not isinstance(payload.get("skip_leading_cd"), bool):
        raise ValueError("manifest field 'skip_leading_cd' must be a bool")

    normalized_config = {
        **{field: payload[field] for field in _FIXED_CORPUS_CONFIG},
        "costs_ms": [float(value) for value in costs],
        "guard_ms": float(guard),
    }
    if normalized_config != _FIXED_CORPUS_CONFIG:
        raise ValueError("manifest analysis config differs from the fixed corpus protocol")

    resolved = dict(payload)
    trace_root_value = payload.get("trace_root")
    if not isinstance(trace_root_value, str) or not trace_root_value.strip():
        raise ValueError("manifest field 'trace_root' must be a non-empty path")
    trace_root_input = Path(trace_root_value).expanduser()
    trace_root = (
        (repo_root / trace_root_input).resolve()
        if not trace_root_input.is_absolute()
        else trace_root_input.resolve()
    )
    if not trace_root.is_dir():
        raise ValueError(f"trace_root is not a directory: {trace_root}")
    resolved["trace_root"] = str(trace_root)

    task_ids = payload.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or any(
        not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids
    ):
        raise ValueError("manifest field 'task_ids' must contain non-empty strings")
    normalized_task_ids = sorted(task_id.strip() for task_id in task_ids)
    if len(normalized_task_ids) != len(set(normalized_task_ids)):
        raise ValueError("manifest field 'task_ids' contains duplicates")
    resolved["task_ids"] = normalized_task_ids
    resolved["costs_ms"] = [float(value) for value in costs]
    return resolved


def require_explicit_trace_task_ids(trace_paths: Iterable[Path]) -> dict[str, str]:
    """Map traces to explicit logical task IDs from canonical metadata."""

    task_by_trace: dict[str, str] = {}
    for trace_path in trace_paths:
        trace = TraceData.load(trace_path)
        task_id = str(trace.metadata.get("instance_id") or "").strip()
        if not task_id:
            raise ValueError(f"trace lacks explicit metadata instance_id: {trace_path}")
        task_by_trace[str(trace_path.resolve())] = task_id
    return task_by_trace


def load_tool_latency_corpus(
    manifest_path: Path,
    *,
    limit_tasks: int | None = None,
    final: bool = False,
) -> tuple[dict[str, list[ToolLatencySample]], list[str], dict[str, Any]]:
    """Load one declared trace corpus and verify its task membership."""

    repo_root = Path(__file__).resolve().parents[2]
    manifest = read_tool_latency_corpus_manifest(
        manifest_path.resolve(), repo_root=repo_root
    )
    trace_paths = discover_trace_files([Path(manifest["trace_root"])])
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {manifest['trace_root']}")

    task_by_trace = require_explicit_trace_task_ids(trace_paths)
    samples_by_task: dict[str, list[ToolLatencySample]] = defaultdict(list)
    for sample in extract_many_tool_latency_samples(trace_paths):
        expected = task_by_trace.get(str(Path(sample.source_trace).resolve()))
        if expected is None or sample.task_id != expected:
            raise ValueError(
                "extracted sample task_id differs from explicit trace metadata: "
                f"{sample.source_trace}: {sample.task_id!r} != {expected!r}"
            )
        samples_by_task[sample.task_id].append(sample)

    task_ids = list(manifest["task_ids"])
    if len(task_ids) != manifest["expected_task_count"]:
        raise ValueError(
            "manifest expected_task_count differs from pinned task_ids: "
            f"{manifest['expected_task_count']} != {len(task_ids)}"
        )
    if set(samples_by_task) != set(task_ids):
        raise ValueError(
            "extracted logical tasks differ from pinned task_ids: "
            f"missing={sorted(set(task_ids) - set(samples_by_task))}, "
            f"unexpected={sorted(set(samples_by_task) - set(task_ids))}"
        )
    if limit_tasks is not None:
        if final:
            raise ValueError("--limit-tasks is a smoke knob; not allowed with --final")
        if limit_tasks < manifest["fold_count"]:
            raise ValueError(
                f"--limit-tasks must be >= fold_count ({manifest['fold_count']})"
            )
        task_ids = task_ids[:limit_tasks]
    return dict(samples_by_task), task_ids, manifest


__all__ = [
    "SegmentLatencySample",
    "ToolLatencySample",
    "discover_trace_files",
    "extract_many_segment_latency_samples",
    "extract_many_tool_latency_samples",
    "extract_segment_latency_samples",
    "extract_tool_latency_samples",
    "load_tool_latency_corpus",
    "read_task_ids",
    "read_tool_latency_corpus_manifest",
    "read_tool_latency_jsonl",
    "require_explicit_trace_task_ids",
    "write_tool_latency_jsonl",
]
