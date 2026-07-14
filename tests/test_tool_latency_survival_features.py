from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from trace_collect.tool_latency_survival_features import (
    CausalRowFeatures,
    SurvivalFeatureSpec,
    fit_feature_encoder,
    iter_causal_row_features,
)


def test_running_call_is_excluded_from_every_history_feature() -> None:
    # `long` starts at 0 and does not finish until t=5.0. `during` starts at
    # t=1.0 while `long` is still running, so `long`'s latency is unobservable
    # and must not appear in any of `during`'s history features.
    rows = [
        _row("long", ts_start=0.0, ts_end=5.0, latency_ms=5000.0),
        _row("during", ts_start=1.0, ts_end=1.08, latency_ms=80.0),
        _row("after", ts_start=6.0, ts_end=6.08, latency_ms=80.0),
    ]

    feats = {f.sample_id: f for f in _extract(rows)}

    during = feats["during"]
    assert during.within_task_prefix_last_ms is None
    assert during.within_task_prefix_median_ms is None
    assert during.within_task_prefix_count == 0
    assert during.within_task_tool_last_ms is None
    assert during.task_running_mean_ms is None
    assert during.call_index_in_task == 1  # `long` started before it
    assert during.latency_ms == 80.0

    # By t=6.0 both `during` (ends 1.08) and `long` (ends 5.0) have completed.
    # They enter tool history in completion (tool_ts_end) order, so the last
    # completed call is `long` and the running mean averages both.
    after = feats["after"]
    assert after.within_task_tool_last_ms == 5000.0
    assert after.call_index_in_task == 2
    assert after.task_running_mean_ms == (80.0 + 5000.0) / 2.0


def test_prefix_features_use_deepest_key_with_history() -> None:
    rows = [
        _row("install", ts_start=0.0, ts_end=5.0, latency_ms=5000.0,
             command="pip install x"),
        _row("test-a", ts_start=1.0, ts_end=1.01, latency_ms=10.0,
             command="pytest a"),
        _row("test-b", ts_start=2.0, ts_end=2.02, latency_ms=20.0,
             command="pytest b"),
        _row("list", ts_start=3.0, ts_end=3.01, latency_ms=30.0,
             command="ls -la"),
    ]

    feats = {f.sample_id: f for f in _extract(rows, command_field="command")}

    # test-b shares "exec:pytest" with the completed test-a but not
    # "exec:pytest b"; the deepest key WITH history is "exec:pytest" -> [10].
    # The slow install (ends t=5.0) has not completed by t=2.0 and stays out.
    test_b = feats["test-b"]
    assert test_b.group_keys == ("exec:pytest", "exec:pytest b")
    assert test_b.within_task_prefix_count == 1
    assert test_b.within_task_prefix_last_ms == 10.0
    assert test_b.within_task_prefix_median_ms == 10.0
    assert test_b.within_task_tool_last_ms == 10.0

    # `ls` shares no command prefix with anything completed, so it has no
    # prefix history, but the tool-level history holds the two completed
    # short calls (test-a ends 1.01, test-b ends 2.02).
    ls = feats["list"]
    assert ls.group_keys == ("exec:ls", "exec:ls -la")
    assert ls.within_task_prefix_count == 0
    assert ls.within_task_prefix_last_ms is None
    assert ls.within_task_tool_last_ms == 20.0  # test-b completed last


def test_prefix_median_uses_all_deepest_key_samples() -> None:
    rows = [
        _row("a", ts_start=0.0, ts_end=0.1, latency_ms=10.0, command="make all"),
        _row("b", ts_start=1.0, ts_end=1.1, latency_ms=90.0, command="make all"),
        _row("c", ts_start=2.0, ts_end=2.1, latency_ms=50.0, command="make all"),
    ]

    feats = {f.sample_id: f for f in _extract(rows, command_field="command")}

    # `c` sees the two completed "exec:make all" samples [10, 90]; median 50,
    # last is the latest-completing sample (90), count 2.
    c = feats["c"]
    assert c.within_task_prefix_count == 2
    assert c.within_task_prefix_last_ms == 90.0
    assert c.within_task_prefix_median_ms == 50.0


def test_own_latency_never_influences_own_features() -> None:
    # Two rows identical except the scored row's own latency (and only its
    # own latency; tool_ts_end held fixed so ordering is untouched). The
    # feature vector must be identical: latency_ms is a label, not a feature.
    def build(second_latency: float) -> CausalRowFeatures:
        rows = [
            _row("first", ts_start=0.0, ts_end=0.1, latency_ms=10.0),
            _row("second", ts_start=1.0, ts_end=1.5, latency_ms=second_latency),
        ]
        return {f.sample_id: f for f in _extract(rows)}["second"]

    short = build(20.0)
    long = build(9000.0)

    spec = SurvivalFeatureSpec(command_field=None)
    encoder = fit_feature_encoder(
        [_row("first", ts_start=0.0, ts_end=0.1, latency_ms=10.0),
         _row("second", ts_start=1.0, ts_end=1.5, latency_ms=20.0)],
        spec=spec,
    )
    # Only the label differs.
    assert short.latency_ms == 20.0
    assert long.latency_ms == 9000.0
    assert replace(short, latency_ms=0.0) == replace(long, latency_ms=0.0)
    np.testing.assert_array_equal(encoder.transform(short), encoder.transform(long))


def test_encoder_vocab_is_train_only_and_oov_is_zero_block() -> None:
    train = [
        _row("t1", ts_start=0.0, ts_end=0.1, latency_ms=10.0, command="pytest a"),
        _row("t2", ts_start=1.0, ts_end=1.1, latency_ms=20.0, command="pytest a"),
    ]
    spec = SurvivalFeatureSpec(command_field="command")
    encoder = fit_feature_encoder(train, spec=spec)

    # Only the "exec" tool and its "pytest" prefixes are in vocab.
    assert set(encoder.tool_vocab) == {"exec"}
    assert set(encoder.prefix_vocab) == {"exec:pytest", "exec:pytest a"}

    # An unseen tool and unseen prefix at transform time -> all-zero blocks,
    # never a KeyError.
    unseen = CausalRowFeatures(
        sample_id="x",
        tool_name="browser",  # not in tool_vocab
        group_keys=("browser:open",),  # not in prefix_vocab
        within_task_prefix_last_ms=None,
        within_task_prefix_median_ms=None,
        within_task_prefix_count=0,
        within_task_tool_last_ms=None,
        call_index_in_task=0,
        task_running_mean_ms=None,
        latency_ms=1.0,
    )
    vector = encoder.transform(unseen)
    n_tool = len(encoder.tool_vocab)
    n_prefix = len(encoder.prefix_vocab)
    assert np.all(vector[: n_tool + n_prefix] == 0.0)


def test_encoder_missing_history_indicators_are_explicit() -> None:
    spec = SurvivalFeatureSpec(
        command_field=None,
        use_tool_identity=False,
        use_command_prefix=False,
    )
    encoder = fit_feature_encoder(
        [_row("t1", ts_start=0.0, ts_end=0.1, latency_ms=10.0)], spec=spec
    )
    # Both categorical blocks disabled: the vector is just the within-task
    # block (8) + task-aggregate block (2).
    assert encoder.n_columns == 10

    first_call = CausalRowFeatures(
        sample_id="c",
        tool_name="exec",
        group_keys=(),
        within_task_prefix_last_ms=None,
        within_task_prefix_median_ms=None,
        within_task_prefix_count=0,
        within_task_tool_last_ms=None,
        call_index_in_task=0,
        task_running_mean_ms=None,
        latency_ms=42.0,
    )
    vector = encoder.transform(first_call)
    # Layout: prefix_last[val,ind] prefix_median[val,ind] count tool_last[val,ind]
    #         call_index  running_mean[val,ind]
    expected = np.array(
        [
            0.0, 0.0,  # prefix_last missing -> value 0, indicator 0
            0.0, 0.0,  # prefix_median missing
            math.log1p(0),  # prefix_count == 0
            0.0, 0.0,  # tool_last missing
            math.log1p(0),  # call_index == 0
            0.0, 0.0,  # running_mean missing
        ]
    )
    np.testing.assert_array_equal(vector, expected)

    present = CausalRowFeatures(
        sample_id="d",
        tool_name="exec",
        group_keys=(),
        within_task_prefix_last_ms=100.0,
        within_task_prefix_median_ms=150.0,
        within_task_prefix_count=2,
        within_task_tool_last_ms=200.0,
        call_index_in_task=3,
        task_running_mean_ms=125.0,
        latency_ms=42.0,
    )
    got = encoder.transform(present)
    expected_present = np.array(
        [
            math.log1p(100.0), 1.0,
            math.log1p(150.0), 1.0,
            math.log1p(2),
            math.log1p(200.0), 1.0,
            math.log1p(3),
            math.log1p(125.0), 1.0,
        ]
    )
    np.testing.assert_allclose(got, expected_present)


def test_encoder_column_count_matches_output_width() -> None:
    train = [
        _row("t1", ts_start=0.0, ts_end=0.1, latency_ms=10.0, command="pytest a"),
        _row("t2", ts_start=1.0, ts_end=1.1, latency_ms=20.0, command="make all"),
    ]
    base = SurvivalFeatureSpec(command_field="command")
    # tools={exec}, prefixes={exec:pytest, exec:pytest a, exec:make, exec:make all}
    encoder = fit_feature_encoder(train, spec=base)
    assert encoder.n_columns == 1 + 4 + 8 + 2
    for feats in iter_causal_row_features(train, spec=base):
        assert encoder.transform(feats).shape == (encoder.n_columns,)

    # Every ablation toggle drops exactly its block width.
    for spec, expected in [
        (replace(base, use_tool_identity=False), 4 + 8 + 2),
        (replace(base, use_command_prefix=False), 1 + 8 + 2),
        (replace(base, use_within_task_history=False), 1 + 4 + 2),
        (replace(base, use_task_aggregates=False), 1 + 4 + 8),
    ]:
        enc = fit_feature_encoder(train, spec=spec)
        assert enc.n_columns == expected
        for feats in iter_causal_row_features(train, spec=spec):
            assert enc.transform(feats).shape == (expected,)


def test_fit_is_deterministic() -> None:
    train = [
        _row("t2", ts_start=1.0, ts_end=1.1, latency_ms=20.0, command="make all"),
        _row("t1", ts_start=0.0, ts_end=0.1, latency_ms=10.0, command="pytest a"),
    ]
    spec = SurvivalFeatureSpec(command_field="command")
    a = fit_feature_encoder(train, spec=spec)
    b = fit_feature_encoder(train, spec=spec)
    assert a.tool_vocab == b.tool_vocab
    assert a.prefix_vocab == b.prefix_vocab
    assert a.n_columns == b.n_columns
    # Vocab indices are assigned in sorted order.
    assert list(a.prefix_vocab) == sorted(a.prefix_vocab)


def test_rejects_duplicate_sample_ids() -> None:
    rows = [
        _row("dup", ts_start=0.0, ts_end=0.1, latency_ms=10.0),
        _row("dup", ts_start=1.0, ts_end=1.1, latency_ms=10.0),
    ]
    with pytest.raises(ValueError, match="duplicate survival sample_id"):
        _extract(rows)


def test_rejects_task_spanning_multiple_traces() -> None:
    rows = [
        _row("a", ts_start=0.0, ts_end=0.1, latency_ms=10.0),
        _row("b", ts_start=1.0, ts_end=1.1, latency_ms=10.0),
    ]
    rows[1]["source_trace"] = "other-trace"
    with pytest.raises(ValueError, match="spans multiple source traces"):
        _extract(rows)


def _extract(
    rows: list[dict[str, Any]],
    *,
    command_field: str | None = None,
) -> list[CausalRowFeatures]:
    spec = SurvivalFeatureSpec(command_field=command_field)
    return list(iter_causal_row_features(rows, spec=spec))


def _row(
    sample_id: str,
    *,
    ts_start: float,
    ts_end: float,
    latency_ms: float,
    task_id: str = "task-a",
    command: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "source_trace": f"trace-{task_id}",
        "task_id": task_id,
        "tool_name": "exec",
        "tool_ts_start": ts_start,
        "tool_ts_end": ts_end,
        "latency_ms": latency_ms,
    }
    if command is not None:
        row["tool_args"] = {"command": command}
    return row


__all__: list[str] = []
