from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import quote


LATENCY_BUCKET_LOWER_EDGES_S = (0.0, 0.5, 2.0, 8.0, 30.0)


@dataclass(frozen=True)
class ToolGapPrediction:
    sample_id: str
    command: str
    probability_by_bucket: tuple[float, ...]
    hard_bucket: int
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        probabilities = self.probability_by_bucket
        if len(probabilities) != len(LATENCY_BUCKET_LOWER_EDGES_S):
            raise ValueError("tool-gap prediction must have five latency buckets")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in probabilities
        ):
            raise ValueError("tool-gap prediction probabilities must be numbers")
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("tool-gap prediction probabilities must be finite and nonnegative")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("tool-gap prediction probabilities must sum to one")
        if self.hard_bucket not in range(len(probabilities)):
            raise ValueError("tool-gap prediction hard bucket is out of range")
        expected_hard_bucket = max(
            range(len(probabilities)), key=probabilities.__getitem__
        )
        if self.hard_bucket != expected_hard_bucket:
            raise ValueError("tool-gap hard bucket must match the PMF argmax")


@dataclass(frozen=True)
class ToolGapLoanConfig:
    arm: Literal["fixed", "feedback", "predictor"]
    state_dir: str
    task_id: str
    foreground_task_ids: tuple[str, ...]
    can_lend: bool
    predictions: tuple[ToolGapPrediction, ...] = ()
    borrower_priority: int | None = None

    def __post_init__(self) -> None:
        if self.arm not in {"fixed", "feedback", "predictor"}:
            raise ValueError(f"unknown tool-gap loan arm: {self.arm}")
        if not self.task_id or not self.foreground_task_ids:
            raise ValueError("tool-gap loan task IDs must be nonempty")
        if len(set(self.foreground_task_ids)) != len(self.foreground_task_ids):
            raise ValueError("tool-gap foreground task IDs must be unique")
        if self.arm != "predictor" and self.predictions:
            raise ValueError("tool-gap predictions are only valid for the predictor arm")
        if self.borrower_priority is not None and (
            not isinstance(self.borrower_priority, int)
            or isinstance(self.borrower_priority, bool)
            or self.borrower_priority < 1
        ):
            raise ValueError("tool-gap borrower priority must be a positive integer")


@dataclass
class ActiveToolGap:
    call_id: str
    command: str
    wall_start_s: float
    budget_s: float | None
    prediction: ToolGapPrediction | None
    lower_edge_s: float | None
    timer: asyncio.Task[None] | None = None
    finished: bool = False


def _key(value: str) -> str:
    return quote(value, safe="")


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class ToolGapLoanRuntime:
    def __init__(self, config: ToolGapLoanConfig) -> None:
        self.config = config
        self.state_dir = Path(config.state_dir)
        self._prediction_index = 0
        self._loan_claimed = False

    def record_llm_response(
        self,
        action_id: str,
        latency_ms: float,
        wall_end_s: float,
    ) -> None:
        if self.config.task_id not in self.config.foreground_task_ids:
            return
        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            raise ValueError("LLM response latency must be finite and nonnegative")
        path = self.state_dir / "responses" / f"{_key(self.config.task_id)}.json"
        responses = _read_json(path, [])
        if not isinstance(responses, list):
            raise ValueError(f"invalid tool-gap response state: {path}")
        responses.append(
            {
                "action_id": action_id,
                "latency_ms": latency_ms,
                "wall_end_s": wall_end_s,
            }
        )
        _atomic_write_json(path, responses)

    def start_tool(
        self,
        call_id: str,
        command: str,
        wall_start_s: float,
    ) -> ActiveToolGap:
        prediction = self._match_prediction(command)
        budget_s = self._budget_before(wall_start_s)
        lower_edge_s = (
            LATENCY_BUCKET_LOWER_EDGES_S[prediction.hard_bucket]
            if prediction is not None
            else None
        )
        handle = ActiveToolGap(
            call_id=call_id,
            command=command,
            wall_start_s=wall_start_s,
            budget_s=budget_s,
            prediction=prediction,
            lower_edge_s=lower_edge_s,
        )
        self._append_event(
            {
                "event": "tool_start",
                "call_id": call_id,
                "command": command,
                "wall_time_s": wall_start_s,
                "budget_s": budget_s,
                "prediction": self._prediction_payload(prediction),
                "lower_edge_s": lower_edge_s,
            }
        )
        if not self.config.can_lend or self.config.arm == "fixed" or budget_s is None:
            return handle
        if (
            self.config.arm == "predictor"
            and lower_edge_s is not None
            and lower_edge_s > budget_s
        ):
            self._emit_loan(handle, trigger="predictor")
        else:
            handle.timer = asyncio.create_task(self._feedback_after(handle))
        return handle

    async def finish_tool(self, handle: ActiveToolGap, wall_end_s: float) -> None:
        handle.finished = True
        if handle.timer is not None and not handle.timer.done():
            handle.timer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.timer
        self._append_event(
            {
                "event": "tool_end",
                "call_id": handle.call_id,
                "command": handle.command,
                "wall_time_s": wall_end_s,
                "elapsed_s": max(0.0, wall_end_s - handle.wall_start_s),
            }
        )

    def _budget_before(self, wall_start_s: float) -> float | None:
        latencies_ms: list[float] = []
        for task_id in self.config.foreground_task_ids:
            path = self.state_dir / "responses" / f"{_key(task_id)}.json"
            responses = _read_json(path, [])
            if not isinstance(responses, list):
                raise ValueError(f"invalid tool-gap response state: {path}")
            visible = [
                float(response["latency_ms"])
                for response in responses
                if isinstance(response, dict)
                and float(response.get("wall_end_s", math.inf)) < wall_start_s
            ]
            if not visible:
                return None
            latencies_ms.extend(visible)
        return max(latencies_ms) / 1000.0

    def _match_prediction(self, command: str) -> ToolGapPrediction | None:
        if self.config.arm != "predictor":
            return None
        if self._prediction_index >= len(self.config.predictions):
            return None
        prediction = self.config.predictions[self._prediction_index]
        if prediction.command != command:
            return None
        self._prediction_index += 1
        return prediction

    async def _feedback_after(self, handle: ActiveToolGap) -> None:
        assert handle.budget_s is not None
        await asyncio.sleep(handle.budget_s)
        if not handle.finished:
            self._emit_loan(handle, trigger="feedback")

    def _emit_loan(self, handle: ActiveToolGap, *, trigger: str) -> None:
        if self._loan_claimed:
            return
        self._loan_claimed = True
        record = {
            "task_id": self.config.task_id,
            "call_id": handle.call_id,
            "command": handle.command,
            "trigger": trigger,
            "decision_wall_time_s": time.time(),
            "tool_start_wall_time_s": handle.wall_start_s,
            "budget_s": handle.budget_s,
            "hard_bucket": (
                handle.prediction.hard_bucket
                if handle.prediction is not None
                else None
            ),
            "lower_edge_s": handle.lower_edge_s,
            "prediction": self._prediction_payload(handle.prediction),
        }
        _atomic_write_json(
            self.state_dir / "loans" / f"{_key(self.config.task_id)}.json",
            record,
        )
        self._append_event({"event": "loan", **record})

    def _append_event(self, event: dict[str, object]) -> None:
        path = self.state_dir / "events" / f"{_key(self.config.task_id)}.json"
        events = _read_json(path, [])
        if not isinstance(events, list):
            raise ValueError(f"invalid tool-gap event state: {path}")
        events.append(event)
        _atomic_write_json(path, events)

    @staticmethod
    def _prediction_payload(
        prediction: ToolGapPrediction | None,
    ) -> dict[str, object] | None:
        if prediction is None:
            return None
        return {
            "sample_id": prediction.sample_id,
            "command": prediction.command,
            "probability_by_bucket": list(prediction.probability_by_bucket),
            "hard_bucket": prediction.hard_bucket,
            "provenance": dict(prediction.provenance),
        }
