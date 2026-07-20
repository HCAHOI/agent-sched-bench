"""Unit tests for the Candidate A Stage-1 session-state CV kill test.

Fixtures are hand-built ``Call`` / ``StateRecord`` objects with known
durations and histories so history-ordering (no-leakage), emergent vocabulary,
state-hash bit arithmetic, the conditional-CV support floors, the kill
arithmetic (including a one-source-only signal), and bootstrap determinism are
checkable without any replay corpus on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_session_state_cv import (
    BootstrapConfig,
    Call,
    ClassCV,
    SelectionConfig,
    StateRecord,
    StateVocab,
    _count_bucket,
    _primary_verb,
    aggregate_reduction,
    assert_state_budget,
    build_class_cvs,
    call_changes_cwd,
    compute_state_records,
    evaluate_source,
    paired_task_cv_bootstrap,
    select_heavy_classes,
)

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_session_state_cv.py"


def _sel_cfg(**overrides) -> SelectionConfig:
    base = dict(
        top_k=3,
        floor_ms=1000.0,
        fold_count=2,
        count_bucket_edges=(1.0, 2.0, 4.0),
        max_state_bits=8,
        min_class_tasks=10,
        min_cell_tasks=5,
    )
    base.update(overrides)
    return SelectionConfig(**base)


# --------------------------------------------------------------------------- #
# State-hash bit arithmetic.
# --------------------------------------------------------------------------- #
def test_state_vocab_bits_and_budget():
    vocab = StateVocab(selected=("a", "b", "c"), top1="a", count_bucket_edges=(1.0, 2.0, 4.0))
    # 3 class bits + 4 buckets -> 2 bits + 1 cwd bit = 6
    assert vocab.bits() == 6
    assert_state_budget(vocab, 8)  # fits
    with pytest.raises(ValueError):
        assert_state_budget(vocab, 5)  # over budget fails fast


def test_count_bucket_edges():
    edges = (1.0, 2.0, 4.0)
    assert _count_bucket(0, edges) == 0
    assert _count_bucket(1, edges) == 1
    assert _count_bucket(2, edges) == 2
    assert _count_bucket(3, edges) == 2
    assert _count_bucket(4, edges) == 3
    assert _count_bucket(99, edges) == 3


def test_call_changes_cwd_delegates_to_production_stripper():
    assert call_changes_cwd("cd sub && pytest")  # leading cd stripped -> changed
    assert not call_changes_cwd("pytest -q")  # no leading cd
    assert not call_changes_cwd("cd sub")  # pure cd kept by production -> ceiling
    assert _primary_verb("cd sub && make && pytest") == "pytest"  # last segment head


# --------------------------------------------------------------------------- #
# History ordering / no-leakage.
# --------------------------------------------------------------------------- #
def test_state_uses_only_strictly_prior_calls_in_temporal_order():
    vocab = StateVocab(selected=("X",), top1="X", count_bucket_edges=(1.0, 2.0))
    early = Call(
        task_id="t",
        order_key=(0.0, "a"),
        cwd_changed=True,
        history_verbs=frozenset({"X"}),
        observations=(("X", 5000.0),),
    )
    late = Call(
        task_id="t",
        order_key=(1.0, "b"),
        cwd_changed=False,
        history_verbs=frozenset({"X"}),
        observations=(("X", 100.0),),  # smaller duration must NOT reorder history
    )
    # Feed reversed so only order_key (not list order, not duration) can be used.
    records = compute_state_records([late, early], vocab)
    by_dur = {rec.duration_ms: rec.state for rec in records}
    # early call: nothing seen before, count 0, no preceding call -> cwd bit False
    assert by_dur[5000.0] == ((False,), 0, False)
    # late call: X seen once before, 1 prior X completion -> bucket 1, prev changed cwd
    assert by_dur[100.0] == ((True,), 1, True)


def test_current_call_verbs_excluded_from_its_own_state():
    vocab = StateVocab(selected=("X",), top1="X", count_bucket_edges=(1.0,))
    only = Call(
        task_id="t",
        order_key=(0.0, "a"),
        cwd_changed=False,
        history_verbs=frozenset({"X"}),
        observations=(("X", 3000.0),),
    )
    (record,) = compute_state_records([only], vocab)
    # X is in this call, but its own occurrence must not set the "seen" bit.
    assert record.state == ((False,), 0, False)


# --------------------------------------------------------------------------- #
# Emergent vocabulary (no verb named in method logic).
# --------------------------------------------------------------------------- #
def test_heavy_class_emerges_from_data_never_named_in_code():
    verb = "florbnak"  # never appears in the module source
    assert verb not in _MODULE_PATH.read_text(encoding="utf-8")
    calls = []
    for i in range(12):
        calls.append(
            Call(
                task_id=f"task{i}",
                order_key=(0.0, "a"),
                cwd_changed=False,
                history_verbs=frozenset({verb, "trivial"}),
                observations=((verb, 4000.0 + i), ("trivial", 5.0)),
            )
        )
    selection = select_heavy_classes(calls, _sel_cfg())
    assert verb in selection["selected"]  # heavy (>= floor) -> emerges
    assert "trivial" not in selection["selected"]  # below floor -> excluded
    assert selection["top1"] == verb


# --------------------------------------------------------------------------- #
# Conditional-CV support floors.
# --------------------------------------------------------------------------- #
def test_build_class_cvs_support_floors():
    cfg = _sel_cfg(min_class_tasks=10, min_cell_tasks=5)
    records = []
    # class "big": 12 tasks, one well-supported cell + one thin cell.
    for i in range(12):
        state = ((True,), 0, False) if i < 8 else ((False,), 0, False)
        records.append(StateRecord(f"t{i}", "big", 1000.0 + i, state))
    # class "small": only 4 tasks -> dropped entirely.
    for i in range(4):
        records.append(StateRecord(f"s{i}", "small", 2000.0, ((False,), 0, False)))
    cvs = build_class_cvs(records, cfg)
    assert "small" not in cvs  # below min_class_tasks
    assert "big" in cvs
    # the 8-task cell qualifies, the 4-task cell is below min_cell_tasks.
    assert cvs["big"].cell_weight == {((True,), 0, False): 8}


# --------------------------------------------------------------------------- #
# CV reduction arithmetic.
# --------------------------------------------------------------------------- #
def _informative_records() -> list[StateRecord]:
    """Cross-task signal: cwd-cell membership separates fast vs slow tasks."""

    records = []
    for i in range(8):  # fast group, cwd bit True
        records.append(StateRecord(f"fast{i}", "X", 1500.0 + i, ((False,), 0, True)))
    for i in range(8):  # slow group, cwd bit False
        records.append(StateRecord(f"slow{i}", "X", 5000.0 + i, ((False,), 0, False)))
    return records


def test_informative_split_reduces_cv():
    cfg = _sel_cfg()
    cvs = build_class_cvs(_informative_records(), cfg)
    tasks = sorted({r.task_id for r in _informative_records()})
    reduction = aggregate_reduction(cvs, tasks)
    assert reduction > 0.3  # big cross-task CV collapses within each cell


def test_single_cell_gives_zero_reduction():
    cfg = _sel_cfg()
    # All observations share one state -> conditional == unconditional.
    records = [
        StateRecord(f"t{i}", "X", 1000.0 + 300 * i, ((False,), 0, False))
        for i in range(12)
    ]
    cvs = build_class_cvs(records, cfg)
    tasks = sorted({r.task_id for r in records})
    assert aggregate_reduction(cvs, tasks) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Bootstrap determinism + survive semantics.
# --------------------------------------------------------------------------- #
def test_bootstrap_is_deterministic_for_a_seed():
    cfg = _sel_cfg()
    cvs = build_class_cvs(_informative_records(), cfg)
    tasks = sorted({r.task_id for r in _informative_records()})
    a = paired_task_cv_bootstrap(cvs, tasks, replicates=500, confidence=0.95, seed=7)
    b = paired_task_cv_bootstrap(cvs, tasks, replicates=500, confidence=0.95, seed=7)
    assert a == b
    assert a["ci_low"] > 0.0  # informative signal -> CI excludes zero


# --------------------------------------------------------------------------- #
# Kill arithmetic incl. one-source-only signal -> KILL.
# --------------------------------------------------------------------------- #
def _calls_from_records(records: list[StateRecord]) -> list[Call]:
    """Wrap pre-stated records as single-call tasks whose emitted state matches.

    Each task has a leading state-setter call (sets the cwd bit) then the class
    call, so ``compute_state_records`` reproduces the intended cell.
    """

    calls: list[Call] = []
    for rec in records:
        _, _, cwd_bit = rec.state
        calls.append(
            Call(
                task_id=rec.task_id,
                order_key=(0.0, "a"),
                cwd_changed=cwd_bit,  # sets the FOLLOWING call's cwd bit
                history_verbs=frozenset(),
                observations=(),
            )
        )
        calls.append(
            Call(
                task_id=rec.task_id,
                order_key=(1.0, "b"),
                cwd_changed=False,
                history_verbs=frozenset({rec.verb_class}),
                observations=((rec.verb_class, rec.duration_ms),),
            )
        )
    return calls


def test_evaluate_source_survive_and_one_source_kills():
    sel = _sel_cfg()
    boot = BootstrapConfig(replicates=800, confidence=0.95, seed=0)

    informative = evaluate_source(
        _calls_from_records(_informative_records()), sel, boot, source="a"
    )
    assert informative["survive"] is True

    flat_records = [
        StateRecord(f"t{i}", "X", 1000.0 + 300 * i, ((False,), 0, False))
        for i in range(12)
    ]
    flat = evaluate_source(_calls_from_records(flat_records), sel, boot, source="b")
    assert flat["survive"] is False  # single cell -> zero reduction

    # Combined verdict is the AND: one-source-only signal must KILL.
    assert (informative["survive"] and flat["survive"]) is False
    assert (informative["survive"] and informative["survive"]) is True


def test_evaluate_source_degenerate_when_no_heavy_class():
    sel = _sel_cfg(floor_ms=10_000.0)  # nothing clears the floor
    boot = BootstrapConfig(replicates=100, confidence=0.95, seed=0)
    calls = [
        Call(
            task_id=f"t{i}",
            order_key=(0.0, "a"),
            cwd_changed=False,
            history_verbs=frozenset({"x"}),
            observations=(("x", 50.0),),
        )
        for i in range(12)
    ]
    block = evaluate_source(calls, sel, boot, source="a")
    assert block["degenerate"] is True
    assert block["survive"] is False
