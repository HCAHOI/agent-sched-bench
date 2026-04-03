from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import ActionRecord, StepRecord


def build_run_id(system: str, workload: str, concurrency: int, started_at: datetime | None = None) -> str:
    """Construct the canonical run id: {system}_{workload}_{N}_{timestamp}."""
    ts = started_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    utc_ts = ts.astimezone(timezone.utc)
    timestamp = utc_ts.strftime("%Y%m%dT%H%M%S") + f"{int(utc_ts.microsecond / 1000):03d}Z"
    return f"{system}_{workload}_{concurrency}_{timestamp}"


class TraceLogger:
    """Append-only JSONL trace logger for step and summary events."""

    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.path = Path(output_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode supports resume: existing records are preserved and
        # collector.py deduplicates via load_completed_ids().
        self._handle = self.path.open("a", encoding="utf-8")

    def log_action(self, agent_id: str, record: ActionRecord) -> None:
        entry = {"type": "action", "agent_id": agent_id, **asdict(record)}
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()

    def log_step(self, agent_id: str, record: StepRecord) -> None:
        entry = {"type": "step", "agent_id": agent_id, **asdict(record)}
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()

    def log_summary(self, agent_id: str, summary: dict[str, Any]) -> None:
        entry = {"type": "summary", "agent_id": agent_id, **summary}
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()

    def log_event(self, agent_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Write a fine-grained event record immediately."""
        entry = {"type": event_type, "agent_id": agent_id, **data}
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
