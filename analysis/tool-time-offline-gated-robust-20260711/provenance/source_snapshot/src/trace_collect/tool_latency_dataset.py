"""Extract observed tool latencies from canonical trace JSONL files."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from trace_collect.tool_gap_extractor import (
    _float_field,
    _int_field,
    _tool_name,
    discover_trace_files,
)
from trace_collect.trace_data import TraceData


MISSING_TOOL_NAME = "__missing_tool_name__"


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


__all__ = [
    "ToolLatencySample",
    "discover_trace_files",
    "extract_many_tool_latency_samples",
    "extract_tool_latency_samples",
    "read_tool_latency_jsonl",
    "write_tool_latency_jsonl",
]
