"""Synthetic-fixture checks for corpus opportunity-mass arithmetic."""

from __future__ import annotations

from pathlib import Path


from scripts.exploration.analyze_corpus_opportunity_mass import (
    Call,
    analyze_corpus,
    extrapolate,
    swappable_mass_per_task,
)

# Two tasks, hand-computable latencies.
#   t1: 1000, 4000, 6000  -> over 3500: (500)+(2500)=3000 ; over 5000: 1000
#   t2: 3000, 3000        -> over 3500: 0                 ; over 5000: 0
CALLS = [
    Call("t1", 1000.0, "python"),
    Call("t1", 4000.0, "pytest"),
    Call("t1", 6000.0, "pytest"),
    Call("t2", 3000.0, "grep"),
    Call("t2", 3000.0, "grep"),
]


def test_swappable_mass_per_task() -> None:
    mass = swappable_mass_per_task(CALLS, 3500.0)
    assert mass["t1"] == 3000.0
    assert mass["t2"] == 0.0  # task present but no over-anchor call
    assert set(mass) == {"t1", "t2"}

    mass5k = swappable_mass_per_task(CALLS, 5000.0)
    assert mass5k["t1"] == 1000.0
    assert mass5k["t2"] == 0.0


def test_analyze_corpus_thresholds_and_mass() -> None:
    stats = analyze_corpus(
        "synthetic",
        Path("/dev/null"),
        CALLS,
        trace_files=2,
        zero_exec=0,
        thresholds=(3500.0, 5000.0),
        heavy_anchor=3500.0,
        mass_anchors=(3500.0, 5000.0),
    )
    assert stats.call_count == 5
    assert stats.task_count == 2
    assert stats.total_tool_time_ms == 17000.0

    # 2 of 5 calls exceed 3500; 1 exceeds 5000.
    assert stats.threshold_call_frac["3500"] == 2 / 5
    assert stats.threshold_call_frac["5000"] == 1 / 5
    # Time over 3500 = 4000+6000 = 10000 of 17000.
    assert stats.threshold_time_frac["3500"] == 10000.0 / 17000.0

    # Swap mass @3500: total 3000, mean over 2 tasks = 1500.
    assert stats.mass["3500"]["total_ms"] == 3000.0
    assert stats.mass["3500"]["mean_per_task_ms"] == 1500.0
    assert stats.mass["3500"]["tasks_with_mass"] == 1
    assert stats.heavy_call_counts["3500"] == 2
    assert stats.heavy_call_counts["5000"] == 1

    # Heavy-verb mix: only calls > 3500 -> two pytest calls (4000+6000).
    verbs = {v["verb"]: v for v in stats.heavy_verbs}
    assert verbs["pytest"]["total_time_ms"] == 10000.0
    assert verbs["pytest"]["n_calls"] == 2
    assert "grep" not in verbs  # grep calls are below the anchor


def test_extrapolate_rate() -> None:
    target = analyze_corpus(
        "tgt", Path("/dev/null"), CALLS, 2, 0,
        thresholds=(3500.0,), heavy_anchor=3500.0, mass_anchors=(3500.0,),
    )
    # Source: 1 task with 1 heavy call -> rate 1.0/task; target has 2 heavy calls.
    source_calls = [Call("s1", 9000.0, "make"), Call("s1", 100.0, "ls")]
    source = analyze_corpus(
        "src", Path("/dev/null"), source_calls, 1, 0,
        thresholds=(3500.0,), heavy_anchor=3500.0, mass_anchors=(3500.0,),
    )
    e = extrapolate(target, source, "3500")
    assert e["target_heavy_calls"] == 2
    assert e["source_heavy_calls_per_task"] == 1.0
    assert e["source_tasks_needed"] == 2.0
