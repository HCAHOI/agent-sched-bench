"""Unit tests for the Candidate B stable-atom overlap diagnostic.

Fixtures are hand-built study Chains with known atom durations and commands so
the numeric screen, fold-stability, overlap counting, and kill arithmetic are
checkable without any replay corpus on disk.
"""

from __future__ import annotations

import pytest

from scripts.analyze_segment_variance import Chain, Segment
from scripts.analyze_stable_atom_overlap import (
    OverlapConfig,
    ScreenConfig,
    _mean_pairwise_jaccard,
    run_overlap,
    stable_atoms,
)


def _chain(task: str, idx: int, command: str, seg_atom: str, dur: float) -> Chain:
    return Chain(
        task_id=task,
        source_trace=f"tr-{task}-{idx}",
        action_id=f"a-{task}-{idx}",
        tool_name="exec",
        parent_command=command,
        parent_total_ms=dur,
        parent_raw_total_ms=dur,
        segments=(Segment(atom=seg_atom, duration_ms=dur, token_count=1, tokens=(seg_atom,)),),
    )


def test_screen_selects_by_numbers_not_by_name() -> None:
    # hv: heavy (~1000ms) and stable across 3 tasks -> qualifies.
    # cd: below the floor -> rejected regardless of stability.
    # nz: heavy but wildly variable across tasks -> CV cap rejects it.
    chains: list[Chain] = []
    for t in ("t0", "t1", "t2"):
        chains.append(_chain(t, 0, "hv", "hv", 1000.0))
        chains.append(_chain(t, 1, "cd /x", "cd", 1.0))
    chains.append(_chain("t0", 2, "nz", "nz", 10.0))
    chains.append(_chain("t1", 2, "nz", "nz", 1000.0))
    chains.append(_chain("t2", 2, "nz", "nz", 5.0))
    screen = ScreenConfig(action_relevance_floor_ms=5.0, cv_cap=1.0, min_task_support=2)
    assert stable_atoms(chains, screen) == {"hv"}


def test_mean_pairwise_jaccard() -> None:
    assert _mean_pairwise_jaccard([{"a"}, {"a"}]) == pytest.approx(1.0)
    assert _mean_pairwise_jaccard([{"a"}, {"b"}]) == pytest.approx(0.0)
    assert _mean_pairwise_jaccard([{"a", "b"}, {"a"}]) == pytest.approx(0.5)
    assert _mean_pairwise_jaccard([{"a"}]) == pytest.approx(1.0)  # single fold


def _corpus(command_of) -> list[Chain]:
    # 6 tasks, 3 hv-calls each; a stable heavy "hv" atom in every task so the
    # screen qualifies it on every fold. command_of(task, idx) controls the
    # parent command (hence the chain-prefix trie node) independently of the
    # segment atom.
    chains: list[Chain] = []
    for i in range(6):
        task = f"task{i}"
        for j in range(3):
            chains.append(
                _chain(task, j, command_of(task, j), "hv", 1000.0 + j)
            )
    return chains


def _cfg(**overrides) -> OverlapConfig:
    base = dict(
        fold_count=2,
        screen=ScreenConfig(action_relevance_floor_ms=5.0, cv_cap=1.0, min_task_support=2),
        thin_support_cap=5,
        kill_overlap_frac=0.05,
        fold_jaccard_floor=1.0,
    )
    base.update(overrides)
    return OverlapConfig(**base)


def test_overlap_zero_when_trie_serves_a_strong_node_kills() -> None:
    # Every call's command is the same "hv": the fit trie holds a well-supported
    # "exec:hv" prefix node, so no firing call lands on fallback/thin support.
    results = run_overlap(_corpus(lambda task, j: "hv"), _cfg())
    ov = results["overlap"]
    assert ov["divergent_calls"] == ov["total_calls"]  # hv fires everywhere
    assert ov["overlap_calls"] == 0
    assert ov["overlap_fraction_of_divergent"] == pytest.approx(0.0)
    kill = results["kill_readout"]
    assert kill["fold_stable"] is True
    assert kill["overlap_below_bar"] is True
    assert kill["verdict"] == "KILL"


def test_overlap_full_when_trie_falls_back_survives() -> None:
    # Every call's command uses a unique verb, so the fit trie has no matching
    # prefix node and falls back to the tool level for every held-out call.
    results = run_overlap(
        _corpus(lambda task, j: f"vrb{task}x{j} arg"), _cfg()
    )
    ov = results["overlap"]
    assert ov["divergent_calls"] == ov["total_calls"]
    assert ov["overlap_calls"] == ov["divergent_calls"]  # all on fallback
    assert ov["strict_fallback_overlap_calls"] == ov["divergent_calls"]
    assert ov["overlap_fraction_of_divergent"] == pytest.approx(1.0)
    kill = results["kill_readout"]
    assert kill["fold_stable"] is True
    assert kill["overlap_below_bar"] is False
    assert kill["verdict"] == "SURVIVE"


def test_fold_instability_forces_kill_even_with_high_overlap() -> None:
    # An unreachable Jaccard floor makes the selection count as fold-unstable,
    # which must kill regardless of a healthy overlap fraction.
    results = run_overlap(
        _corpus(lambda task, j: f"vrb{task}x{j} arg"),
        _cfg(fold_jaccard_floor=2.0),
    )
    assert results["overlap"]["overlap_fraction_of_divergent"] == pytest.approx(1.0)
    kill = results["kill_readout"]
    assert kill["fold_stable"] is False
    assert kill["verdict"] == "KILL"


def test_thin_support_cap_reclassifies_small_prefix_nodes() -> None:
    # Common command "hv" (strong node) but a thin_support_cap above the fit
    # support reclassifies it as thin -> every firing call becomes overlap.
    results = run_overlap(_corpus(lambda task, j: "hv"), _cfg(thin_support_cap=1000))
    ov = results["overlap"]
    assert ov["overlap_calls"] == ov["divergent_calls"]
    assert ov["strict_fallback_overlap_calls"] == 0  # not fallback, just thin
    assert results["kill_readout"]["verdict"] == "SURVIVE"
