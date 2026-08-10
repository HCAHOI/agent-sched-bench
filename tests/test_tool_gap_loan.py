from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace_collect.tool_gap_loan import (
    ToolGapLoanConfig,
    ToolGapLoanRuntime,
    ToolGapPrediction,
)


FOREGROUND = ("task-0", "task-1", "task-2", "task-3")


def _prediction(command: str = "pytest -q") -> ToolGapPrediction:
    return ToolGapPrediction(
        sample_id="sample-0",
        command=command,
        probability_by_bucket=(0.0, 0.0, 0.0, 1.0, 0.0),
        hard_bucket=3,
        provenance={"head": "task-aware"},
    )


def _runtime(
    tmp_path: Path,
    *,
    arm: str = "predictor",
    task_id: str = "task-0",
    can_lend: bool = True,
    predictions: tuple[ToolGapPrediction, ...] = (),
) -> ToolGapLoanRuntime:
    return ToolGapLoanRuntime(
        ToolGapLoanConfig(
            arm=arm,
            state_dir=str(tmp_path),
            task_id=task_id,
            foreground_task_ids=FOREGROUND,
            can_lend=can_lend,
            predictions=predictions,
        )
    )


def _seed_responses(runtime: ToolGapLoanRuntime, *, latency_ms: float = 2.0) -> None:
    for index, task_id in enumerate(FOREGROUND):
        peer = ToolGapLoanRuntime(
            ToolGapLoanConfig(
                arm="fixed",
                state_dir=str(runtime.state_dir),
                task_id=task_id,
                foreground_task_ids=FOREGROUND,
                can_lend=True,
            )
        )
        peer.record_llm_response(
            f"llm-{index}", latency_ms + index, wall_end_s=10.0 + index
        )


def _loan_records(tmp_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "loans").glob("*.json"))
    ]


def test_budget_requires_every_foreground_task(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = _runtime(tmp_path, predictions=(_prediction(),))
        for index, task_id in enumerate(FOREGROUND[:-1]):
            peer = _runtime(tmp_path, arm="fixed", task_id=task_id)
            peer.record_llm_response(f"llm-{index}", 2.0, wall_end_s=10.0)

        handle = runtime.start_tool("call-0", "pytest -q", wall_start_s=20.0)
        assert handle.budget_s is None
        await runtime.finish_tool(handle, wall_end_s=21.0)

    asyncio.run(exercise())
    assert _loan_records(tmp_path) == []


def test_predictor_emits_early_loan_when_bucket_lower_edge_exceeds_budget(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        runtime = _runtime(tmp_path, predictions=(_prediction(),))
        _seed_responses(runtime)
        handle = runtime.start_tool("call-0", "pytest -q", wall_start_s=20.0)
        assert handle.budget_s == 0.005
        await runtime.finish_tool(handle, wall_end_s=20.1)

    asyncio.run(exercise())
    [loan] = _loan_records(tmp_path)
    assert loan["trigger"] == "predictor"
    assert loan["call_id"] == "call-0"
    assert loan["hard_bucket"] == 3
    assert loan["lower_edge_s"] == 8.0
    assert loan["budget_s"] == 0.005


def test_feedback_emits_only_after_frozen_budget(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = _runtime(tmp_path, arm="feedback")
        _seed_responses(runtime, latency_ms=10.0)
        handle = runtime.start_tool("call-0", "pytest -q", wall_start_s=20.0)
        assert _loan_records(tmp_path) == []
        await asyncio.sleep(0.02)
        assert _loan_records(tmp_path)[0]["trigger"] == "feedback"
        await runtime.finish_tool(handle, wall_end_s=20.1)

    asyncio.run(exercise())


def test_one_lender_cannot_emit_two_loans(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = _runtime(
            tmp_path,
            predictions=(_prediction("pytest -q"), _prediction("pytest -x")),
        )
        _seed_responses(runtime)
        first = runtime.start_tool("call-0", "pytest -q", wall_start_s=20.0)
        await runtime.finish_tool(first, wall_end_s=20.1)
        second = runtime.start_tool("call-1", "pytest -x", wall_start_s=21.0)
        await runtime.finish_tool(second, wall_end_s=21.1)

    asyncio.run(exercise())
    assert len(_loan_records(tmp_path)) == 1


def test_waiting_task_cannot_lend(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = _runtime(
            tmp_path,
            task_id="task-4",
            can_lend=False,
            predictions=(_prediction(),),
        )
        _seed_responses(runtime)
        handle = runtime.start_tool("call-0", "pytest -q", wall_start_s=20.0)
        await runtime.finish_tool(handle, wall_end_s=20.1)

    asyncio.run(exercise())
    assert _loan_records(tmp_path) == []


def test_unmatched_prediction_falls_back_to_feedback(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = _runtime(tmp_path, predictions=(_prediction("pytest -q"),))
        _seed_responses(runtime, latency_ms=10.0)
        unmatched = runtime.start_tool("call-0", "echo hi", wall_start_s=20.0)
        assert unmatched.prediction is None
        await asyncio.sleep(0.02)
        assert _loan_records(tmp_path)[0]["trigger"] == "feedback"
        await runtime.finish_tool(unmatched, wall_end_s=20.1)

    asyncio.run(exercise())


def test_prediction_rejects_hard_bucket_inconsistent_with_pmf() -> None:
    with pytest.raises(ValueError, match="hard bucket must match"):
        ToolGapPrediction(
            sample_id="sample-0",
            command="pytest -q",
            probability_by_bucket=(1.0, 0.0, 0.0, 0.0, 0.0),
            hard_bucket=3,
            provenance={},
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        (True, False, False, False, False),
        ("1", 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_prediction_rejects_non_numeric_pmf_elements(
    probabilities: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="numbers"):
        ToolGapPrediction(
            sample_id="sample-0",
            command="pytest -q",
            probability_by_bucket=probabilities,  # type: ignore[arg-type]
            hard_bucket=0,
            provenance={},
        )
