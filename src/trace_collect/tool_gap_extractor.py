"""Extract observed tool-gap windows from canonical trace JSONL files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from trace_collect.trace_data import TraceData


@dataclass(frozen=True)
class ToolGapWindow:
    """Observed scheduling window after one LLM call's tool batch."""

    sample_id: str
    source_trace: str
    agent_id: str
    instance_id: str
    iteration: int
    llm_action_id: str
    next_llm_action_id: str
    available_gap_ms: float
    tool_count: int
    tool_names: tuple[str, ...]
    tool_call_ids: tuple[str, ...]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_trace": self.source_trace,
            "agent_id": self.agent_id,
            "instance_id": self.instance_id,
            "iteration": self.iteration,
            "llm_action_id": self.llm_action_id,
            "next_llm_action_id": self.next_llm_action_id,
            "available_gap_ms": self.available_gap_ms,
            "tool_count": self.tool_count,
            "tool_names": list(self.tool_names),
            "tool_call_ids": list(self.tool_call_ids),
        }


def discover_trace_files(paths: Iterable[Path]) -> list[Path]:
    """Return explicit trace files and recursive ``trace.jsonl`` files under dirs."""

    traces: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("trace.jsonl"))
        else:
            raise FileNotFoundError(f"trace path does not exist: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            traces.append(resolved)
    if not traces:
        raise ValueError("no trace files found")
    return traces


def extract_tool_gap_windows(
    trace_path: Path,
    *,
    agent_filter: str | None = None,
) -> list[ToolGapWindow]:
    """Extract per-agent tool-batch windows from one canonical trace.

    The output is an observed-label dataset. It intentionally keeps only the
    observed gap label plus pre-execution batch identity fields; future online
    predictors must not feed observed current-iteration timing back as features.
    """

    trace = TraceData.load(trace_path, agent_filter=agent_filter)
    actions_by_agent: dict[str, list[dict[str, Any]]] = {}
    for action in trace.actions:
        agent_id = str(action.get("agent_id") or "")
        actions_by_agent.setdefault(agent_id, []).append(action)

    windows: list[ToolGapWindow] = []
    for agent_id, actions in sorted(actions_by_agent.items()):
        llm_actions = [a for a in actions if a.get("action_type") == "llm_call"]
        tool_actions = [a for a in actions if a.get("action_type") == "tool_exec"]
        for index, llm_action in enumerate(llm_actions):
            iteration = _int_field(llm_action, "iteration")
            llm_end = _float_field(llm_action, "ts_end")
            next_llm = _next_llm_for_iteration(
                llm_actions,
                current=llm_action,
                current_index=index,
                source_trace=trace_path,
            )
            if next_llm is None:
                continue
            next_llm_start = _float_field(next_llm, "ts_start")
            candidate_batch = [
                tool
                for tool in tool_actions
                if _int_field(tool, "iteration") == iteration
                and _float_field(tool, "ts_start") >= llm_end
                and _float_field(tool, "ts_start") < next_llm_start
            ]
            if not candidate_batch:
                continue
            batch_end = max(_float_field(tool, "ts_end") for tool in candidate_batch)
            if batch_end > next_llm_start:
                raise ValueError(
                    f"{trace_path}: tool batch crosses next LLM start "
                    f"for agent {agent_id!r} iteration {iteration}"
                )
            batch = candidate_batch
            available_gap_ms = (next_llm_start - llm_end) * 1000.0
            tool_names = tuple(_tool_name(tool) for tool in batch)
            tool_call_ids = tuple(
                str(
                    (tool.get("data") or {}).get("tool_call_id")
                    or tool.get("action_id")
                    or ""
                )
                for tool in batch
            )
            instance_id = str(
                llm_action.get("instance_id")
                or trace.metadata.get("instance_id")
                or ""
            )
            llm_action_id = str(llm_action.get("action_id") or f"llm_{iteration}")
            next_llm_action_id = str(next_llm.get("action_id") or "")
            sample_id = f"{trace_path}:{agent_id}:{iteration}:{llm_action_id}"
            windows.append(
                ToolGapWindow(
                    sample_id=sample_id,
                    source_trace=str(trace_path),
                    agent_id=agent_id,
                    instance_id=instance_id,
                    iteration=iteration,
                    llm_action_id=llm_action_id,
                    next_llm_action_id=next_llm_action_id,
                    available_gap_ms=available_gap_ms,
                    tool_count=len(batch),
                    tool_names=tool_names,
                    tool_call_ids=tool_call_ids,
                )
            )
    return windows


def extract_many_tool_gap_windows(
    trace_paths: Iterable[Path],
    *,
    agent_filter: str | None = None,
) -> list[ToolGapWindow]:
    windows: list[ToolGapWindow] = []
    for trace_path in trace_paths:
        windows.extend(extract_tool_gap_windows(trace_path, agent_filter=agent_filter))
    if not windows:
        raise ValueError("no tool-gap windows found")
    return windows


def write_tool_gap_jsonl(windows: Iterable[ToolGapWindow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for window in windows:
            fh.write(json.dumps(window.to_json_obj(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            count += 1
    return count


def _next_llm_for_iteration(
    llm_actions: list[dict[str, Any]],
    *,
    current: dict[str, Any],
    current_index: int,
    source_trace: Path,
) -> dict[str, Any] | None:
    """Return the next plausible same-lane LLM after ``current``.

    Real traces can contain concatenated or interleaved sessions that reuse an
    ``agent_id`` and reset ``iteration``. Do not key tool batches by iteration
    alone across the whole trace: pair a window only with a later LLM whose
    iteration advances and whose timestamp starts after the current LLM ends.
    """

    current_iteration = _int_field(current, "iteration")
    current_end = _float_field(current, "ts_end")
    candidates = [
        (iteration, start, index, action)
        for index, action in enumerate(llm_actions)
        if index != current_index
        for iteration in [_int_field(action, "iteration")]
        for start in [_float_field(action, "ts_start")]
        if iteration > current_iteration and start >= current_end
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    next_iteration, _, _, next_action = candidates[0]
    if next_iteration != current_iteration + 1:
        raise ValueError(
            f"{source_trace}: missing next LLM iteration after "
            f"{current.get('action_id')!r}"
        )
    return next_action


def _tool_name(action: dict[str, Any]) -> str:
    data = action.get("data") or {}
    return str(data.get("tool_name") or "")


def _int_field(action: dict[str, Any], field: str) -> int:
    value = action.get(field)
    if not isinstance(value, int):
        raise ValueError(f"action {action.get('action_id')!r} has non-int {field}")
    return value


def _float_field(action: dict[str, Any], field: str) -> float:
    value = action.get(field)
    if not isinstance(value, int | float):
        raise ValueError(f"action {action.get('action_id')!r} has non-numeric {field}")
    return float(value)
