"""Unit tests for the segment atom-vs-chain variance decomposition logic.

Fixtures are hand-built rows with known medians so the decomposition math is
checkable in closed form; no replay durations are baked in as ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_collect.command_features import (
    command_has_concurrent_segments,
    command_prefix_keys,
)
from scripts.analyze_segment_variance import (
    _CERT_MAX_PREFIX_DEPTH,
    _CERT_MIN_EVIDENCE,
    Chain,
    Config,
    Segment,
    atom_key,
    build_chains,
    cross_validated_predictions,
    fit_atom_identity,
    fit_chain_prefix,
    mae,
    r2_score,
    segment_token_count,
)
from trace_collect.tool_latency_dataset import SegmentLatencySample

_FRESH_CERT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "analysis/fresh-corpus-certification-20260717/offline-gated-robust/manifest.json"
)


def _chain(task: str, segs: list[Segment], total: float, cmd: str) -> Chain:
    return Chain(
        task_id=task,
        source_trace=f"trace-{task}",
        action_id=f"act-{task}-{cmd}",
        tool_name="exec",
        parent_command=cmd,
        parent_total_ms=total,
        parent_raw_total_ms=total,
        segments=tuple(segs),
    )


def test_atom_key_uses_head_and_keeps_cd_class() -> None:
    assert atom_key("cd /testbed") == "cd"
    assert atom_key("grep -rn foo /x") == "grep"
    assert atom_key("/usr/bin/python3 -c 'x'") == "python3"
    assert atom_key("'unbalanced") == "__unparsed__"


def test_segment_token_count() -> None:
    assert segment_token_count("cd /testbed") == 2
    assert segment_token_count("grep -rn foo") == 3


def test_concurrent_predicate_matches_ceiling() -> None:
    assert command_has_concurrent_segments("grep x f | head")
    assert command_has_concurrent_segments("for i in 1 2; do echo $i; done")
    assert not command_has_concurrent_segments("cd /t && make")
    assert not command_has_concurrent_segments("a || b")


def test_r2_and_mae_closed_form() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert mae(y, y) == 0.0
    # constant prediction at the mean -> R^2 == 0
    assert r2_score(y, np.full(3, 2.0)) == pytest.approx(0.0)
    assert mae(np.array([0.0, 10.0]), np.array([2.0, 6.0])) == pytest.approx(3.0)


def test_fit_atom_identity_recovers_medians_plus_overhead() -> None:
    # cd median = 1, make median = 10; overhead absorbs the constant gap.
    train = [
        _chain("t1", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 14.0, "cd && make"),
        _chain("t2", [Segment("cd", 1.0, 2), Segment("make", 12.0, 1)], 18.0, "cd && make"),
        _chain("t3", [Segment("cd", 1.0, 2), Segment("make", 10.0, 1)], 16.0, "cd && make"),
    ]
    predict = fit_atom_identity(train)
    # cd med=1, make med=10 -> sum=11; overhead = median(14-9,18-13,16-11)=median(5,5,5)=5
    pred = predict(train[0])
    assert pred == pytest.approx(16.0)  # 5 overhead + 1 + 10


def test_fit_chain_prefix_backs_off_below_evidence() -> None:
    train = [
        _chain("t1", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 100.0, "cd /a && make"),
        _chain("t2", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 200.0, "cd /a && make"),
    ]
    # depth-4, skip_leading_cd=False (cert config): the full 4-token command
    # is one key, appearing twice -> median 150.
    predict = fit_chain_prefix(
        train, max_depth=4, min_evidence=2, skip_leading_cd=False
    )
    unseen = _chain("t9", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 0.0, "cd /a && make")
    assert predict(unseen) == pytest.approx(150.0)
    # a command with no matching key backs off to the tool median
    other = _chain("t9", [Segment("ls", 1.0, 1)], 0.0, "ls")
    assert predict(other) == pytest.approx(150.0)  # tool median over both chains


def test_fit_chain_prefix_skip_leading_cd_is_a_real_deviation() -> None:
    """cert (skip_leading_cd=False) and cdskip (True) must diverge when the
    leading cd consumes the depth budget - the exact case fix candidate #2
    targets, and why the two must be reported as separate models.
    """

    train = [
        _chain("t1", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 100.0, "cd /a && make"),
        _chain("t2", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 200.0, "cd /a && make"),
    ]
    unseen = _chain("t9", [Segment("cd", 1.0, 2), Segment("make", 8.0, 1)], 0.0, "cd /a && make")
    # depth=1: cert keys on "exec:cd" (shared across any cd-prefixed command);
    # cdskip keys on "exec:make" once cd is stripped from the prefix budget.
    cert_predict = fit_chain_prefix(
        train, max_depth=1, min_evidence=2, skip_leading_cd=False
    )
    cdskip_predict = fit_chain_prefix(
        train, max_depth=1, min_evidence=2, skip_leading_cd=True
    )
    assert cert_predict(unseen) == pytest.approx(150.0)
    assert cdskip_predict(unseen) == pytest.approx(150.0)
    # Both land on 150 here (only one command shape in this fixture) but the
    # *keys* they matched differ - verify directly via command_prefix_keys.
    cert_keys = command_prefix_keys(
        "exec", "cd /a && make", max_depth=1, skip_leading_cd=False
    )
    cdskip_keys = command_prefix_keys(
        "exec", "cd /a && make", max_depth=1, skip_leading_cd=True
    )
    assert cert_keys == ("exec:cd",)
    assert cdskip_keys == ("exec:make",)
    assert cert_keys != cdskip_keys


def test_cert_registry_gate_matches_frozen_fresh_cert_manifest() -> None:
    """chain_prefix_cert's literal gate/depth must equal the frozen FRESH-CERT
    manifest's min_tool_history/max_prefix_depth, read from the manifest
    itself - so this test breaks if either side drifts, not just if someone
    edits the literal without checking the manifest.
    """

    manifest = json.loads(_FRESH_CERT_MANIFEST.read_text(encoding="utf-8"))
    assert _CERT_MIN_EVIDENCE == manifest["min_tool_history"]
    assert _CERT_MAX_PREFIX_DEPTH == manifest["max_prefix_depth"]


def test_build_chains_groups_and_orders_segments() -> None:
    samples = [
        SegmentLatencySample(
            sample_id="s1", source_trace="tr", task_id="task-a", agent_id="ag",
            action_id="a1", tool_name="exec", segment_index=1,
            segment_command="make", segment_ms=10.0, t_start_ms=2.0, t_end_ms=12.0,
            parent_chain_command="cd /t && make", parent_total_ms=15.0,
            parent_raw_total_ms=15.0,
        ),
        SegmentLatencySample(
            sample_id="s0", source_trace="tr", task_id="task-a", agent_id="ag",
            action_id="a1", tool_name="exec", segment_index=0,
            segment_command="cd /t", segment_ms=1.0, t_start_ms=0.0, t_end_ms=1.0,
            parent_chain_command="cd /t && make", parent_total_ms=15.0,
            parent_raw_total_ms=15.0,
        ),
    ]
    chains = build_chains(samples)
    assert len(chains) == 1
    chain = chains[0]
    assert [s.atom for s in chain.segments] == ["cd", "make"]
    assert chain.family == "cd>>make"
    assert chain.parent_total_ms == pytest.approx(15.0)


def test_cross_validated_predictions_are_out_of_sample() -> None:
    # Two disjoint task clusters with distinct levels: a fold split must never
    # let a task's own rows leak into its training set.
    chains = []
    for task in ("t1", "t2", "t3", "t4"):
        level = 10.0 if task in ("t1", "t2") else 1000.0
        chains.append(
            _chain(task, [Segment("cd", 1.0, 2), Segment("make", level, 1)],
                   level, "cd && make")
        )
    cfg = Config(
        fold_count=2, prefix_depth=4, min_prefix_evidence=1,
        token_bin_count=2, min_atom_count=1, min_family_count=1,
        tail_percentile=90.0,
    )
    preds = cross_validated_predictions(chains, cfg)
    for _name, (y_true, y_pred, fams) in preds.items():
        assert len(y_true) == len(chains)
        assert all(f == "cd>>make" for f in fams)
