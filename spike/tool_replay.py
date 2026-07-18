"""Real-trace tool-call replayer for the certified scenario.

Reads REAL tool-call events (tool name, command, observed duration) from
fresh-corpus trace JSONLs through the production extractor
(``trace_collect.tool_latency_dataset.extract_tool_latency_samples``) and builds
a deterministic replay schedule. Durations are the observed
``latency_ms`` labels from the traces -- never synthetic (CLAUDE.md forbids
fabricated durations). Only calls that carry a shell command under
``command_field`` are replayable, because the certified trigger is keyed on the
command prefix; command-less calls have no group node and would only ever hit
the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Sequence

from trace_collect.tool_latency_dataset import (
    ToolLatencySample,
    extract_tool_latency_samples,
)


@dataclass(frozen=True)
class ReplayCall:
    """One real tool call to replay: name, command, and observed duration."""

    sample_id: str
    task_id: str
    tool_name: str
    command: str
    duration_ms: float


def select_replay_calls(
    samples: Sequence[ToolLatencySample],
    *,
    command_field: str = "command",
    task_ids: Sequence[str] | None = None,
    limit: int = 10,
    seed: int = 0,
) -> list[ReplayCall]:
    """Select a deterministic replay set from extracted latency samples.

    Keeps only samples carrying a non-empty string command under
    ``command_field``; optionally restricts to ``task_ids``. When more samples
    remain than ``limit``, a seeded random subset is chosen (reproducible for a
    given seed + task selection), then re-sorted by ``(task_id, tool_ts_start,
    sample_id)`` so the replay order is the calls' real temporal order. Fails
    fast if nothing is replayable.
    """

    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    wanted = set(task_ids) if task_ids is not None else None
    calls: list[tuple[str, float, ReplayCall]] = []
    for sample in samples:
        if wanted is not None and sample.task_id not in wanted:
            continue
        args = sample.tool_args
        if not isinstance(args, dict):
            continue
        command = args.get(command_field)
        if not isinstance(command, str) or not command.strip():
            continue
        calls.append(
            (
                sample.task_id,
                sample.tool_ts_start,
                ReplayCall(
                    sample_id=sample.sample_id,
                    task_id=sample.task_id,
                    tool_name=sample.tool_name,
                    command=command,
                    duration_ms=sample.latency_ms,
                ),
            )
        )
    if not calls:
        raise ValueError(
            "no replayable tool calls found (need a string command under "
            f"{command_field!r} in tool_args)"
        )
    calls.sort(key=lambda item: (item[0], item[1], item[2].sample_id))
    if len(calls) > limit:
        rng = Random(seed)
        chosen = rng.sample(calls, limit)
        chosen.sort(key=lambda item: (item[0], item[1], item[2].sample_id))
    else:
        chosen = calls
    return [call for _, _, call in chosen]


def build_replay_schedule(
    trace_root: str | Path,
    *,
    command_field: str = "command",
    task_ids: Sequence[str] | None = None,
    limit: int = 10,
    seed: int = 0,
    agent_filter: str | None = None,
) -> list[ReplayCall]:
    """Extract real tool calls under ``trace_root`` and build a replay schedule.

    Globs ``**/trace.jsonl`` under ``trace_root`` and runs the production
    extractor on each; selection/determinism is handled by
    :func:`select_replay_calls`.
    """

    root = Path(trace_root)
    trace_paths = sorted(root.glob("**/trace.jsonl"))
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {root}")
    samples: list[ToolLatencySample] = []
    for trace_path in trace_paths:
        samples.extend(
            extract_tool_latency_samples(trace_path, agent_filter=agent_filter)
        )
    return select_replay_calls(
        samples,
        command_field=command_field,
        task_ids=task_ids,
        limit=limit,
        seed=seed,
    )


__all__ = ["ReplayCall", "build_replay_schedule", "select_replay_calls"]
