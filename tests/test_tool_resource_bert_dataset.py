"""Causality and split guards for the BERT resource dataset (no torch needed)."""

from __future__ import annotations

import math
from types import SimpleNamespace

from tool_resource.bert_dataset import (
    SCALAR_FEATURE_NAMES,
    assign_repo_folds,
    examples_for_sequence,
    repo_of,
)

_TOOL_VOCAB = ("exec", "read_file")
_SCALAR = {name: i for i, name in enumerate(SCALAR_FEATURE_NAMES)}


def _call(
    ts_start,
    ts_end,
    *,
    command=None,
    tool_name="exec",
    core_s=None,
    peak_mem=None,
    peak_cpu=None,
    ambient_before_mb=None,
):
    kind = "exec_timeline" if (tool_name == "exec" and core_s is not None) else (
        "non_exec_zero" if tool_name != "exec" else "missing_exec_timeline"
    )
    ns = SimpleNamespace(
        tool_name=tool_name,
        tool_args={"command": command} if command is not None else None,
        tool_ts_start=ts_start,
        tool_ts_end=ts_end,
        cpu_core_seconds=core_s if tool_name == "exec" else 0.0,
        cpu_core_seconds_kind=kind,
        peak_cpu_cores=peak_cpu,
        peak_cpu_cores_eligible=peak_cpu is not None,
        peak_memory_mb=peak_mem,
    )
    if ambient_before_mb is not None:
        ns.ambient_before_mb = ambient_before_mb
        ns.ambient_before_age_s = 0.5
    return ns


def _feature(example, name):
    return example.numeric[_SCALAR[name]]


def test_prev_command_features_use_only_earlier_calls():
    calls = [
        _call(0.0, 1.0, command="first", core_s=2.0),
        _call(1.0, 3.0, command="second", core_s=4.0),
        _call(3.0, 4.0, command="third", core_s=1.0),
    ]
    examples = examples_for_sequence("repo__x-1", "trace", calls, "PROMPT", _TOOL_VOCAB, 5)

    def prev_block(example):
        return example.text.split("[PREV]")[1].split("[CUR]")[0]

    # First call has no history; later calls' PREV block holds strictly earlier
    # commands and never the current or a later one.
    assert "none" in prev_block(examples[0])
    assert "first" in prev_block(examples[1]) and "second" not in prev_block(examples[1])
    assert "first" in prev_block(examples[2]) and "second" in prev_block(examples[2])
    assert "third" not in prev_block(examples[2])


def test_running_max_memory_excludes_current_window():
    calls = [
        _call(0.0, 1.0, command="a", core_s=1.0, peak_mem=100.0),
        _call(1.0, 2.0, command="b", core_s=1.0, peak_mem=300.0),
        _call(2.0, 3.0, command="c", core_s=1.0, peak_mem=200.0),
    ]
    examples = examples_for_sequence("repo__x-1", "trace", calls, "P", _TOOL_VOCAB, 5)

    # No prior window for the first call.
    assert _feature(examples[0], "running_max_memory_mb") == 0.0
    # Second call sees only the first window's 100, never its own 300.
    assert _feature(examples[1], "running_max_memory_mb") == 100.0
    # Third call sees max(100, 300) from priors, never its own 200.
    assert _feature(examples[2], "running_max_memory_mb") == 300.0


def test_cumulative_exec_core_seconds_excludes_current():
    calls = [
        _call(0.0, 1.0, command="a", core_s=2.0),
        _call(1.0, 2.0, command="b", core_s=5.0),
        _call(2.0, 3.0, command="c", core_s=9.0),
    ]
    examples = examples_for_sequence("repo__x-1", "trace", calls, "P", _TOOL_VOCAB, 5)

    assert _feature(examples[0], "cum_prior_exec_core_s") == 0.0
    assert _feature(examples[1], "cum_prior_exec_core_s") == 2.0
    assert _feature(examples[2], "cum_prior_exec_core_s") == 7.0


def test_call_index_and_elapsed_are_causal():
    calls = [
        _call(10.0, 11.0, command="a", core_s=1.0),
        _call(15.0, 16.0, command="b", core_s=1.0),
    ]
    examples = examples_for_sequence("repo__x-1", "trace", calls, "P", _TOOL_VOCAB, 5)
    assert _feature(examples[0], "call_index") == 0.0
    assert _feature(examples[0], "elapsed_task_s") == 0.0
    assert _feature(examples[1], "call_index") == 1.0
    assert _feature(examples[1], "elapsed_task_s") == 5.0


def test_log_space_targets_and_masks():
    calls = [
        _call(0.0, 2.0, command="a", core_s=1.0, peak_mem=500.0, peak_cpu=2.0),
        _call(2.0, 4.0, command="b", core_s=1.0),  # no memory / peak-cpu window
    ]
    examples = examples_for_sequence("repo__x-1", "trace", calls, "P", _TOOL_VOCAB, 5)

    # All targets trained in log space; memory is log (not log1p), anchoring
    # moved to the input feature so the target is no longer a residual.
    t0 = examples[0]
    assert t0.target_mask["log_cpu_core_seconds"] is True
    assert math.isclose(t0.targets["log_cpu_core_seconds"], math.log1p(1.0))
    assert t0.target_mask["log_peak_cpu_cores"] is True
    assert math.isclose(t0.targets["log_peak_cpu_cores"], math.log1p(2.0))
    assert t0.target_mask["log_peak_memory_mb"] is True
    assert math.isclose(t0.targets["log_peak_memory_mb"], math.log(500.0))
    # Memory/peak-cpu masked out when their window samples are absent.
    assert examples[1].target_mask["log_peak_memory_mb"] is False
    assert examples[1].target_mask["log_peak_cpu_cores"] is False
    assert math.isnan(examples[1].targets["log_peak_memory_mb"])


def test_repo_of_strips_instance_suffix():
    assert repo_of("google__flax-4681") == "google__flax"
    assert repo_of("ArkEcosystem__python-crypto-116") == "ArkEcosystem__python-crypto"


def test_repo_folds_keep_a_repo_out_of_both_sides():
    task_ids = [f"owner__proj{p}-{i}" for p in range(6) for i in range(3)]
    repos = [repo_of(t) for t in task_ids]
    folds = assign_repo_folds(repos, n_folds=3, seed=7)

    # Every fold's val repos are disjoint from the other folds' train repos by
    # construction: a repo belongs to exactly one fold.
    for fold in range(3):
        val_repos = {r for r, f in folds.items() if f == fold}
        train_repos = {r for r, f in folds.items() if f != fold}
        assert val_repos.isdisjoint(train_repos)
    assert set(folds.values()) == {0, 1, 2}


def test_merged_corpora_share_one_fold_per_repo():
    # A repo present in two corpora (different instance ids) must get ONE fold,
    # so the merge never puts it in both train and val across corpora.
    corpus_a = ["shared__repo-1", "onlya__repo-9"]
    corpus_b = ["shared__repo-2", "onlyb__repo-4"]  # shared__repo also here
    union_repos = [repo_of(t) for t in corpus_a + corpus_b]
    folds = assign_repo_folds(union_repos, n_folds=3, seed=1)
    assert "shared__repo" in folds  # deduped to a single entry -> single fold
