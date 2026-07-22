"""Unit tests for the WTN Stage-1 wrapper-transparency harness.

Fixtures are hand-built Chains with known commands and totals so candidate
extraction, key normalization, nested-fold screen isolation, the min-support
default, the K1/K2/K3 arithmetic (including CI-sign logic), and grid
knob-matching are checkable without any replay corpus on disk.
"""

from __future__ import annotations

import pytest

from scripts.exploration.analyze_segment_variance import (
    Chain,
    Segment,
)
from scripts.exploration.analyze_wrapper_transparency import (
    GridConfig,
    ScreenConfig,
    ScreenResult,
    _k1,
    _k2,
    _k3,
    _skip_predict,
    beyond_leading_candidates,
    candidate_classes,
    census,
    make_wtn_key_fn,
    normalized_key_tokens,
    paired_task_bootstrap,
    run_grid,
    screen_transparency,
    wtn_prefix_keys,
)
from trace_collect.command_features import command_prefix_keys, shell_command_prefix_tokens


def _seg(cmd: str = "x") -> Segment:
    toks = tuple(shell_command_prefix_tokens(cmd))
    return Segment(
        atom=(toks[0] if toks else "__unparsed__"),
        duration_ms=1.0,
        token_count=len(toks),
        tokens=toks,
    )


def _chain(task: str, command: str, total: float, action: str = "a") -> Chain:
    return Chain(
        task_id=task,
        source_trace="tr",
        action_id=f"{task}-{action}",
        tool_name="exec",
        parent_command=command,
        parent_total_ms=total,
        parent_raw_total_ms=None,
        segments=(_seg(), _seg()),
    )


# --------------------------------------------------------------------------- #
# Candidate extraction.
# --------------------------------------------------------------------------- #
def test_candidate_classes_are_non_final_verbs() -> None:
    assert candidate_classes("cd /w && pip install -e . && python3 foo.py") == [
        "cd",
        "pip",
    ]
    # Final segment is never a candidate.
    assert candidate_classes("make -j2 all") == []
    assert candidate_classes("conda activate env && python run.py") == ["conda"]


def test_beyond_leading_skips_index_zero() -> None:
    # pip sits after the leading cd -> a leading-only stripper cannot reach it.
    assert beyond_leading_candidates("cd /w && pip install && python x") == ["pip"]
    # A single leading cd is reachable by a leading-only stripper -> empty.
    assert beyond_leading_candidates("cd /w && python x") == []


def test_paren_group_is_not_a_candidate_class() -> None:
    # Grouping parens are dropped by shell_command_segments, so "(" must never
    # surface as a spurious verb class in the candidate universe.
    verbs = candidate_classes("(cd /w && make) && python x")
    assert "(" not in verbs
    assert ")" not in verbs
    assert verbs == ["cd", "make"]


# --------------------------------------------------------------------------- #
# Key normalization: parity, drop semantics, support consolidation.
# --------------------------------------------------------------------------- #
def test_empty_transparent_set_matches_production_keys() -> None:
    # Degenerate path MUST equal the unmodified production key exactly.
    cmd = "cd /w && pip install -e . && python3 foo.py"
    assert wtn_prefix_keys("exec", cmd, max_depth=4, transparent=frozenset()) == (
        command_prefix_keys("exec", cmd, max_depth=4, skip_leading_cd=False)
    )
    assert normalized_key_tokens(cmd, frozenset()) == shell_command_prefix_tokens(cmd)


def test_normalization_drops_nonfinal_keeps_final() -> None:
    cmd = "cd /w && make x"
    assert normalized_key_tokens(cmd, frozenset({"cd"})) == ["make", "x"]
    # A transparent verb in the FINAL position is kept (not a candidate).
    tail = "make && cd /w"
    assert normalized_key_tokens(tail, frozenset({"cd"})) == ["make", "&&", "cd", "/w"]


def test_support_consolidation_collapses_to_same_key() -> None:
    a = wtn_prefix_keys("exec", "cd /a && python x", max_depth=4, transparent=frozenset({"cd"}))
    b = wtn_prefix_keys("exec", "cd /b && python x", max_depth=4, transparent=frozenset({"cd"}))
    assert a == b == ("exec:python", "exec:python x")


def test_wtn_key_fn_degenerate_equals_off_predictions() -> None:
    train = [
        _chain("t0", "cd /a && python x", 1000.0),
        _chain("t1", "cd /b && python y", 2000.0),
        _chain("t2", "make z", 3000.0),
    ]
    empty = make_wtn_key_fn(frozenset(), max_depth=4)
    off = _skip_predict(train, min_evidence=1, skip="off", wtn_key_fn=empty)
    wtn = _skip_predict(train, min_evidence=1, skip="WTN", wtn_key_fn=empty)
    for chain in train:
        assert off(chain) == wtn(chain)


# --------------------------------------------------------------------------- #
# Synthetic corpus where a leading uninformative wrapper is transparent.
# --------------------------------------------------------------------------- #
def _uninformative_wrapper_corpus() -> list[Chain]:
    """cd is a constant per-task-dir wrapper; the workload total depends only
    on the final module group, so dropping cd (task-specific dir) generalizes
    across tasks and cannot hurt -> cd must emerge transparent. `timeout` is a
    rare wrapper present in a single task, below support -> non-transparent.
    """

    group_total = {"A": 1000.0, "B": 5000.0}
    chains: list[Chain] = []
    for t in range(6):
        task = f"task{t}"
        for group, total in group_total.items():
            for k in range(3):
                cmd = f"cd /home/{task} && python mod_{group}.py"
                chains.append(_chain(task, cmd, total + k, action=f"{group}{k}"))
    # Rare heavy wrapper in one task only (support: 1 task).
    for k in range(2):
        chains.append(
            _chain("task0", "cd /home/task0 && timeout && python mod_A.py", 1000.0 + k,
                   action=f"to{k}")
        )
    return chains


def test_screen_emerges_cd_transparent_and_min_support_blocks_rare() -> None:
    corpus = _uninformative_wrapper_corpus()
    cfg = ScreenConfig(tolerance=0.05, min_tasks=2, min_chains=4, inner_folds=2)
    result = screen_transparency(corpus, cfg)
    assert not result.degenerate
    assert "cd" in result.marginal
    # timeout is below min-support -> defaults NON-transparent regardless of stat.
    assert "timeout" not in result.marginal
    timeout_row = next(r for r in result.per_class if r["verb_class"] == "timeout")
    assert timeout_row["support_ok"] is False
    # Joint check on the passing set should not degrade -> applied non-empty.
    assert result.applied == result.marginal


def test_screen_support_counts_use_train_only() -> None:
    # Screen support is computed from the passed split only (nested-fold
    # isolation): a class present in N train tasks reports exactly N.
    corpus = _uninformative_wrapper_corpus()
    cfg = ScreenConfig(tolerance=0.05, min_tasks=2, min_chains=4, inner_folds=2)
    result = screen_transparency(corpus, cfg)
    cd_row = next(r for r in result.per_class if r["verb_class"] == "cd")
    train_tasks = {c.task_id for c in corpus if "cd" in candidate_classes(c.parent_command)}
    assert cd_row["support_tasks"] == len(train_tasks)


def test_screen_degenerate_when_too_few_tasks() -> None:
    # Fewer tasks than inner folds -> screen degrades to empty (off) set.
    corpus = [_chain("only", "cd /a && python x", 1000.0)]
    result = screen_transparency(corpus, ScreenConfig(inner_folds=5))
    assert result.degenerate
    assert result.applied == frozenset()


# --------------------------------------------------------------------------- #
# Census.
# --------------------------------------------------------------------------- #
def test_census_counts_candidates_and_mass() -> None:
    chains = [
        _chain("t0", "cd /a && python x", 1000.0),
        _chain("t1", "cd /b && pip install && python y", 2000.0),
        _chain("t2", "make z", 3000.0),  # no candidate
    ]
    cen = census(chains)
    assert cen["multi_segment_chains"] == 3
    # Headline framing quantity: only t1 has a wrapper beyond the leading
    # segment (pip), carrying 2000 of the 6000ms total mass.
    assert cen["chains_beyond_leading_segment"] == 1
    assert cen["beyond_leading_mass_fraction"] == pytest.approx(2000.0 / 6000.0)
    # Upper bound incl. leading cd: t0 and t1 both have a non-final candidate.
    assert cen["chains_with_candidate_upper_bound"] == 2
    # t2 ("make z") is a single token-level segment though the chain has 2
    # xtrace segments -> counted in the population-mismatch row.
    assert cen["newline_only_chains"] == 1
    classes = {row["verb_class"]: row["chains"] for row in cen["per_class"]}
    assert classes == {"cd": 2, "pip": 1}


# --------------------------------------------------------------------------- #
# Kill-criterion arithmetic (CI-sign logic).
# --------------------------------------------------------------------------- #
def _pair(low: float, high: float) -> dict[str, float]:
    return {"mean_delta_ms": (low + high) / 2, "ci_low": low, "ci_high": high,
            "confidence": 0.95, "tasks": 5}


def test_k1_survives_when_beats_off_and_not_worse_than_cd() -> None:
    pairwise = {
        "WTN_vs_off@me1": _pair(-10.0, -2.0),  # CI strictly below 0 -> beats off
        "WTN_vs_cd-only@me1": _pair(-3.0, 1.0),  # CI covers 0 -> not worse
    }
    k1 = _k1(pairwise, 1)
    assert k1["beats_off_ci_excludes_zero"] is True
    assert k1["not_worse_than_cd_only"] is True
    assert k1["killed"] is False


def test_k1_kills_when_off_ci_covers_zero() -> None:
    pairwise = {
        "WTN_vs_off@me1": _pair(-4.0, 1.0),  # covers 0 -> does not beat off
        "WTN_vs_cd-only@me1": _pair(-3.0, 1.0),
    }
    assert _k1(pairwise, 1)["killed"] is True


def test_k1_kills_when_worse_than_cd_only() -> None:
    pairwise = {
        "WTN_vs_off@me1": _pair(-10.0, -2.0),
        "WTN_vs_cd-only@me1": _pair(2.0, 6.0),  # CI strictly above 0 -> worse
    }
    k1 = _k1(pairwise, 1)
    assert k1["not_worse_than_cd_only"] is False
    assert k1["killed"] is True


def _screen(marginal: set[str], universe: list[str], degenerate: bool = False) -> ScreenResult:
    return ScreenResult(
        marginal=frozenset(marginal),
        applied=frozenset(marginal),
        joint_ok=True,
        joint_stat=0.0,
        fit_side_mae=100.0,
        universe=universe,
        per_class=[],
        degenerate=degenerate,
    )


def test_k2_passes_when_cd_transparent_every_fold() -> None:
    folds = [_screen({"cd"}, ["cd", "pip"]), _screen({"cd", "pip"}, ["cd", "pip"])]
    k2 = _k2(folds)
    assert k2["emerges_every_fold"] is True
    assert k2["killed"] is False


def test_k2_kills_when_cd_missing_in_a_fold() -> None:
    folds = [_screen({"cd"}, ["cd"]), _screen(set(), ["cd"])]
    assert _k2(folds)["killed"] is True


def test_k2_kills_when_cd_never_a_candidate() -> None:
    folds = [_screen(set(), ["make"]), _screen(set(), ["make"])]
    k2 = _k2(folds)
    assert k2["candidate_present"] is False
    assert k2["killed"] is True


def test_k3_passes_on_unanimous_decisions() -> None:
    folds = [_screen({"cd"}, ["cd"]), _screen({"cd"}, ["cd"])]
    chains = [_chain("t0", "cd /a && python x", 1000.0)]
    k3 = _k3(chains, folds, floor=0.9)
    assert k3["mass_agreement_fraction"] == 1.0
    assert k3["killed"] is False


def test_k3_kills_when_class_decision_splits_across_folds() -> None:
    folds = [_screen({"cd"}, ["cd"]), _screen(set(), ["cd"])]
    chains = [_chain("t0", "cd /a && python x", 1000.0)]
    k3 = _k3(chains, folds, floor=0.9)
    assert k3["mass_agreement_fraction"] == 0.0
    assert k3["killed"] is True


def test_k3_weights_agreement_by_parent_total_ms() -> None:
    # cd is unanimous (transparent both folds); pip is not (only fold 0). The
    # heavy chain (cd only, mass 900) is stable; the light chain (cd+pip, mass
    # 100) is not. Count fraction = 0.5 but mass fraction = 900/1000 = 0.9.
    folds = [_screen({"cd", "pip"}, ["cd", "pip"]), _screen({"cd"}, ["cd", "pip"])]
    chains = [
        _chain("t0", "cd /a && python x", 900.0),
        _chain("t1", "cd /a && pip install && python y", 100.0),
    ]
    k3 = _k3(chains, folds, floor=0.9)
    assert k3["mass_agreement_fraction"] == pytest.approx(0.9)
    assert k3["count_agreement_fraction"] == pytest.approx(0.5)
    assert k3["killed"] is False  # 0.9 mass clears the 0.9 floor


# --------------------------------------------------------------------------- #
# Paired bootstrap sign.
# --------------------------------------------------------------------------- #
def test_paired_task_bootstrap_ci_below_zero_for_negative_deltas() -> None:
    deltas = {"t1": [-1.0, -1.0], "t2": [-2.0], "t3": [-1.5]}
    point, lo, hi = paired_task_bootstrap(
        deltas, replicates=2000, confidence=0.95, seed=0
    )
    assert point < 0.0
    assert hi < 0.0
    assert lo <= point


# --------------------------------------------------------------------------- #
# Grid knob-matching + end-to-end structure.
# --------------------------------------------------------------------------- #
def test_run_grid_produces_six_cells_and_kill_readout() -> None:
    corpus = _uninformative_wrapper_corpus()
    cfg = GridConfig(
        fold_count=3,
        screen=ScreenConfig(tolerance=0.05, min_tasks=2, min_chains=4, inner_folds=2),
        primary_min_evidence=1,
        k3_mass_agreement_floor=0.9,
        bootstrap_replicates=200,
        bootstrap_confidence=0.95,
        bootstrap_seed=0,
    )
    results = run_grid(corpus, cfg)
    cells = results["grid"]["cell_metrics"]
    assert len(cells) == 6  # 3 skip modes x 2 min_evidence
    assert set(results["grid"]["pairwise_deltas"]) == {
        "WTN_vs_off@me1",
        "WTN_vs_cd-only@me1",
        "WTN_vs_off@me5",
        "WTN_vs_cd-only@me5",
    }
    assert results["kill_readout"]["verdict"] in {"KILL", "SURVIVE"}
    # Dropping the task-specific cd dir generalizes across tasks: WTN should not
    # be worse than off on MAE here (behavioral sanity, not a knob).
    assert cells["WTN@me1"]["mae_ms"] <= cells["off@me1"]["mae_ms"] + 1e-9


def test_grid_cells_differ_only_in_declared_knobs() -> None:
    # off and WTN(empty transparent) must be identical; cd-only differs only by
    # the leading-cd strip. Same train, same depth, same min_evidence.
    train = _uninformative_wrapper_corpus()
    empty = make_wtn_key_fn(frozenset(), max_depth=4)
    off = _skip_predict(train, min_evidence=1, skip="off", wtn_key_fn=empty)
    wtn = _skip_predict(train, min_evidence=1, skip="WTN", wtn_key_fn=empty)
    assert all(off(c) == wtn(c) for c in train)
