"""CPU tests for metadata KV-residency policy and analysis contracts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import serving.kv_policies.metadata as metadata_mod
from serving.kv_policies.analysis import (
    assert_logits_byte_identical,
    assert_teacher_forced_ids_equal,
    mean_contiguity_displacement,
    resolve_affected_positions,
)
from serving.kv_policies.base import EvictionPolicyConfig, EvictionDecision
from serving.kv_policies.config import load_eviction_config
from serving.kv_policies.metadata import (
    MetadataResidencyCache,
    MetadataResidencySelector,
    NullEvictionCache,
    PositionControlCache,
    TokenMetadata,
)
from serving.sparse_attention.base import SparseAttentionContext
from serving.sparse_attention.metadata import MetadataResidencySparseAttention


def _segments() -> list[dict]:
    return [
        {"role": "system", "token_start": 0, "token_end": 1, "first_seen_call": 0},
        {"role": "user", "token_start": 1, "token_end": 3, "first_seen_call": 0},
        {
            "role": "tool_result",
            "token_start": 3,
            "token_end": 5,
            "first_seen_call": 0,
            "exit_code": 0,
            "tool_error": False,
        },
        {
            "role": "tool_result",
            "token_start": 5,
            "token_end": 7,
            "first_seen_call": 0,
            "exit_code": 0,
            "tool_error": False,
        },
        {
            "role": "tool_result",
            "token_start": 7,
            "token_end": 9,
            "first_seen_call": 0,
            "exit_code": 0,
            "tool_error": False,
        },
        {
            "role": "tool_result",
            "token_start": 9,
            "token_end": 11,
            "first_seen_call": 0,
            "exit_code": 2,
            "tool_error": True,
        },
        {
            "role": "tool_result",
            "token_start": 11,
            "token_end": 12,
            "first_seen_call": 0,
            "exit_code": 0,
            "tool_error": False,
        },
        {
            "role": "assistant_message",
            "token_start": 12,
            "token_end": 14,
            "first_seen_call": 0,
        },
    ]


def _metadata_cache(rung: str, *, budget: int = 6) -> MetadataResidencyCache:
    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=budget,
        sink_size=1,
        recent_window=2,
        metadata_rung=rung,  # type: ignore[arg-type]
        reserve_system_prompt=False,
    )
    cache = MetadataResidencyCache(cfg, num_layers=2)
    cache.notify_new_call(0, segments=_segments(), input_token_count=14)
    return cache


def _write_per_layer_table(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "scores:",
                "  - {layer: 0, role: user, age: 0, score: 10.0}",
                "  - {layer: 1, role: user, age: 0, score: 1.0}",
                "  - {layer: 0, role: assistant_message, age: 0, score: 1.0}",
                "  - {layer: 1, role: assistant_message, age: 0, score: 10.0}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_metadata_rungs_have_exact_keep_sets() -> None:
    expected = {
        "rung1": [0, 3, 5, 7, 9, 11],
        "rung2": [0, 3, 5, 7, 12, 13],
        "rung3": [0, 3, 5, 11, 12, 13],
        "rung4": [0, 9, 10, 11, 12, 13],
    }
    for rung, keep in expected.items():
        decision = _metadata_cache(rung)._decide_evict(layer_idx=0, key_len=14)
        assert decision.keep_indices == keep
        assert decision.evict_indices == sorted(set(range(14)) - set(keep))
        assert decision.policy_state is not None
        assert decision.policy_state["original_kept_indices"] == keep


def test_system_prompt_is_reserved_outside_metadata_budget() -> None:
    segments = [
        {"role": "system", "token_start": 0, "token_end": 4, "first_seen_call": 0},
        {"role": "user", "token_start": 4, "token_end": 10, "first_seen_call": 0},
        {
            "role": "assistant_message",
            "token_start": 10,
            "token_end": 12,
            "first_seen_call": 0,
        },
    ]
    cache = MetadataResidencyCache(
        EvictionPolicyConfig(
            name="metadata",
            budget=3,
            sink_size=0,
            recent_window=0,
            metadata_rung="rung1",
        ),
        num_layers=1,
    )
    cache.notify_new_call(0, segments=segments, input_token_count=12)

    decision = cache._decide_evict(layer_idx=0, key_len=12)

    # Four system/tool-schema tokens are forced resident and do not consume the
    # three-token conversation budget, so the physical keep set may exceed
    # config.budget by the system span size.
    assert decision.keep_indices == [0, 1, 2, 3, 4, 5, 6]
    assert decision.policy_state is not None
    assert decision.policy_state["original_kept_indices"] == [0, 1, 2, 3, 4, 5, 6]

    keys = torch.arange(12 * 2, dtype=torch.float32).reshape(1, 1, 12, 2)
    values = keys + 100
    out_keys, _out_values = cache.update(keys, values, layer_idx=0)
    assert int(out_keys.shape[-2]) == 7
    assert int(cache.get_seq_length(0)) == 7


def test_system_prompt_reserve_matches_vectorized_selector() -> None:
    original_indices = list(range(12))
    table: dict[int, TokenMetadata] = {}
    for original in range(4):
        table[original] = TokenMetadata(original, 0, "system", 0, original, 0, 4)
    for offset, original in enumerate(range(4, 10)):
        table[original] = TokenMetadata(original, 1, "user", 0, offset, 4, 10)
    for offset, original in enumerate(range(10, 12)):
        table[original] = TokenMetadata(
            original, 2, "assistant_message", 0, offset, 10, 12
        )
    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=3,
        sink_size=0,
        recent_window=0,
        metadata_rung="rung1",
    )
    selector = MetadataResidencySelector(cfg)

    expected = selector._select_python(
        layer_idx=0,
        key_len=len(original_indices),
        original_indices=original_indices,
        metadata_table=table,
    )
    arrays = metadata_mod._metadata_arrays_from_table(
        original_indices=original_indices,
        metadata_table=table,
    )
    actual = selector.select_from_arrays(
        layer_idx=0,
        key_len=len(original_indices),
        original_indices=arrays[0],
        role_rank=arrays[1],
        age=arrays[2],
        offset=arrays[3],
        segment_id=arrays[4],
        is_tool_result=arrays[5],
        is_error=arrays[6],
        is_system=arrays[7],
    )

    assert actual.keep_indices == expected.keep_indices == [0, 1, 2, 3, 4, 5, 6]
    assert actual.evict_indices == expected.evict_indices
    assert actual.original_kept_indices == expected.original_kept_indices


def test_metadata_cache_crop_to_logical_prefix_remaps_original_indices() -> None:
    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=16,
        sink_size=0,
        recent_window=0,
        metadata_rung="rung1",
    )
    cache = MetadataResidencyCache(cfg, num_layers=1)
    segments = [
        {"role": "system", "token_start": 0, "token_end": 1, "first_seen_call": 0},
        {"role": "user", "token_start": 1, "token_end": 8, "first_seen_call": 0},
    ]
    cache.notify_new_call(0, segments=segments, input_token_count=8)
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
    values = keys + 100
    cache.update(keys, values, layer_idx=0)

    cache.crop_to_logical_length(5)

    assert int(cache.get_seq_length(0)) == 5
    assert cache.original_indices_for_layer(0) == [0, 1, 2, 3, 4]

    cache.notify_new_call(1, segments=segments, input_token_count=8)
    delta_keys = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1)
    delta_values = delta_keys + 200
    cache.update(delta_keys, delta_values, layer_idx=0)

    assert cache.original_indices_for_layer(0) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_bridge_remap_is_per_layer_and_original_index_pure_after_eviction() -> None:
    cache = _metadata_cache("rung4")
    d0 = cache._decide_evict(layer_idx=0, key_len=14)
    cache._post_evict_hook(0, d0)

    # Simulate a different layer-local eviction to prove there is no shared
    # original->current fallback map.
    d1 = EvictionDecision(
        keep_indices=[0, 1, 2, 3, 4, 5],
        evict_indices=list(range(6, 14)),
        reason="test",
    )
    cache._ensure_layer_state(1, 14)  # test-local setup of layer 1 origins
    cache._post_evict_hook(1, d1)

    d0_next = cache._decide_evict(layer_idx=0, key_len=7)
    assert d0_next.policy_state is not None
    assert max(d0_next.policy_state["original_kept_indices"]) <= 14


def test_recent_tool_reservation_uses_current_physical_recency() -> None:
    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=5,
        sink_size=1,
        recent_window=2,
        metadata_rung="rung3",
        reserve_system_prompt=False,
    )
    selector = MetadataResidencySelector(cfg)
    original_indices = [0, 100, 101, 102, 200, 201, 202, 300, 301, 302]
    table: dict[int, TokenMetadata] = {
        0: TokenMetadata(0, 0, "system", 0, 0, 0, 1),
        300: TokenMetadata(300, 2, "tool_result", 0, 0, 300, 301),
    }
    for offset, original in enumerate([100, 101, 102]):
        table[original] = TokenMetadata(
            original,
            1,
            "tool_result",
            0,
            offset,
            100,
            999,
        )
    for original in [200, 201, 202, 301, 302]:
        table[original] = TokenMetadata(original, 3, "generation", 0, 0, original, original + 1)

    selection = selector.select(
        layer_idx=0,
        key_len=len(original_indices),
        original_indices=original_indices,
        metadata_table=table,
    )

    assert selection.keep_indices == [0, 1, 7, 8, 9]
    assert 7 in selection.keep_indices
    assert 2 not in selection.keep_indices


def test_vectorized_metadata_selector_matches_python_keep_sets() -> None:
    original_indices = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]
    table = {
        0: TokenMetadata(0, 0, "system", 0, 0, 0, 1),
        1: TokenMetadata(1, 1, "user", 0, 0, 1, 2),
        4: TokenMetadata(4, 2, "tool_result", 0, 0, 4, 6),
        5: TokenMetadata(5, 2, "tool_result", 0, 1, 4, 6),
        8: TokenMetadata(8, 3, "tool_result", 1, 0, 8, 10, exit_code=1),
        9: TokenMetadata(9, 3, "tool_result", 1, 1, 8, 10, tool_error=True),
        12: TokenMetadata(12, 4, "assistant_message", 1, 0, 12, 14),
        13: TokenMetadata(13, 4, "assistant_message", 1, 1, 12, 14),
        16: TokenMetadata(16, 5, "generation", 1, 0, 16, 18),
        17: TokenMetadata(17, 5, "generation", 1, 1, 16, 18),
    }
    for rung in ["rung1", "rung2", "rung3", "rung4"]:
        cfg = EvictionPolicyConfig(
            name="metadata",
            budget=5,
            sink_size=1,
            recent_window=2,
            metadata_rung=rung,
            reserve_system_prompt=False,
        )
        selector = MetadataResidencySelector(cfg)
        expected = selector._select_python(
            layer_idx=0,
            key_len=len(original_indices),
            original_indices=original_indices,
            metadata_table=table,
        )
        arrays = metadata_mod._metadata_arrays_from_table(
            original_indices=original_indices,
            metadata_table=table,
        )
        actual = selector.select_from_arrays(
            layer_idx=0,
            key_len=len(original_indices),
            original_indices=arrays[0],
            role_rank=arrays[1],
            age=arrays[2],
            offset=arrays[3],
            segment_id=arrays[4],
            is_tool_result=arrays[5],
            is_error=arrays[6],
            is_system=arrays[7],
        )

        assert actual.keep_indices == expected.keep_indices
        assert actual.evict_indices == expected.evict_indices
        assert actual.original_kept_indices == expected.original_kept_indices
        assert actual.original_evicted_indices == expected.original_evicted_indices
        assert actual.reason == expected.reason


def test_per_layer_table_changes_layer_scores_and_remaps(tmp_path: Path) -> None:
    table_path = tmp_path / "per_layer_scores.yaml"
    _write_per_layer_table(table_path)
    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=4,
        sink_size=0,
        recent_window=0,
        metadata_rung="rung1",
        reserve_system_prompt=False,
        per_layer_table=True,
        per_layer_table_path=str(table_path),
    )
    cache = MetadataResidencyCache(cfg, num_layers=2)
    cache.notify_new_call(
        0,
        segments=[
            {"role": "user", "token_start": 0, "token_end": 4, "first_seen_call": 0},
            {
                "role": "assistant_message",
                "token_start": 4,
                "token_end": 8,
                "first_seen_call": 0,
            },
        ],
        input_token_count=8,
    )

    d0 = cache._decide_evict(layer_idx=0, key_len=8)
    d1 = cache._decide_evict(layer_idx=1, key_len=8)
    assert d0.keep_indices == [0, 1, 2, 3]
    assert d1.keep_indices == [4, 5, 6, 7]

    cache._post_evict_hook(0, d0)
    cache._post_evict_hook(1, d1)


def test_metadata_sidecar_uses_kv_original_remap_after_compaction() -> None:
    cache = _metadata_cache("rung4", budget=6)
    first = cache._decide_evict(layer_idx=0, key_len=14)
    cache._post_evict_hook(0, first)

    sidecar = MetadataResidencySparseAttention(
        budget=6,
        sink_size=1,
        recent_window=2,
        metadata_rung="rung4",
        reserve_system_prompt=False,
    )
    sidecar.notify_new_call(call_idx=0, segments=_segments(), input_token_count=14)
    context = SparseAttentionContext(
        module=None,
        hidden_states=None,
        position_embeddings=None,
        past_key_values=cache,
        attention_mask=None,
    )
    sidecar.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=8,
        phase="decode",
        decode_step=0,
        device=None,
        dtype=None,
        context=context,
    )
    sidecar_meta = sidecar.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    decision = cache._decide_evict(layer_idx=0, key_len=8)

    assert sidecar_meta["selected_indices"] == decision.keep_indices
    assert decision.policy_state is not None
    assert sidecar_meta["original_selected_indices"] == decision.policy_state[
        "original_kept_indices"
    ]


def test_null_eviction_emptiness_and_logit_identity_assertion() -> None:
    cache = NullEvictionCache(
        EvictionPolicyConfig(name="null_eviction", budget=1),
        num_layers=1,
    )
    decision = cache._decide_evict(layer_idx=0, key_len=8)
    assert decision.keep_indices == list(range(8))
    assert decision.evict_indices == []
    assert decision.policy_state == {
        "original_kept_indices": list(range(8)),
        "original_evicted_indices": [],
    }

    logits = np.arange(12, dtype=np.float32).reshape(2, 6)
    assert_logits_byte_identical(logits, logits.copy())
    with pytest.raises(AssertionError, match="logits"):
        assert_logits_byte_identical(logits, logits + np.float32(1.0))


def test_affected_position_resolver_returns_set_b_not_set_a() -> None:
    result = resolve_affected_positions(
        evicted_original_by_row=[[1], [2], [3]],
        topk_indices=np.asarray([[0, 2], [4, 5], [3, 1]], dtype=np.int32),
        topk_weights=np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32),
    )
    assert result.set_a_mask.tolist() == [True, True, True]
    assert result.set_b_mask.tolist() == [False, False, True]
    assert result.set_b_indices.tolist() == [2]


def test_teacher_forced_identity_hard_fails_on_token_mismatch() -> None:
    assert_teacher_forced_ids_equal([1, 2, 3], np.asarray([1, 2, 3], dtype=np.int64))
    with pytest.raises(AssertionError, match="teacher-forced"):
        assert_teacher_forced_ids_equal([1, 2, 3], [1, 9, 3])


def test_delta_pos_golden_and_control_ordering() -> None:
    assert mean_contiguity_displacement([0, 3, 4, 7]) == pytest.approx(2.0)
    pc_random = mean_contiguity_displacement([2, 5, 9])
    pc_structured = mean_contiguity_displacement([0, 3, 4, 7])
    rung = mean_contiguity_displacement([0, 3, 4, 7])
    pc_middle = mean_contiguity_displacement([0, 1, 2, 9])
    assert pc_random > pc_structured
    assert pc_structured == pytest.approx(rung)
    assert rung > pc_middle


def _ns(**kwargs) -> argparse.Namespace:
    base = {
        "kv_policy": "none",
        "kv_budget": None,
        "kv_sink_size": 4,
        "kv_recent_window": 256,
        "kv_aggregate": "sum",
        "kv_record": "on",
        "kv_config": None,
        "kv_metadata_rung": "rung4",
        "kv_position_control": "random",
        "kv_per_layer_table": False,
        "kv_per_layer_table_path": None,
        "kv_per_layer_budget": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_metadata_and_control_config_parse_and_invalids_raise(tmp_path: Path) -> None:
    table_path = tmp_path / "per_layer_scores.yaml"
    _write_per_layer_table(table_path)
    cfg = load_eviction_config(
        _ns(
            kv_policy="metadata",
            kv_budget=1024,
            kv_metadata_rung="rung3",
            kv_per_layer_table=True,
            kv_per_layer_table_path=str(table_path),
        )
    )
    assert cfg is not None
    assert cfg.name == "metadata"
    assert cfg.metadata_rung == "rung3"
    assert cfg.reserve_system_prompt is True
    assert cfg.per_layer_table is True
    assert cfg.per_layer_table_path == str(table_path)

    pc = load_eviction_config(
        _ns(
            kv_policy="position_control",
            kv_budget=1024,
            kv_position_control="middle",
        )
    )
    assert pc is not None
    assert pc.name == "position_control"
    assert pc.position_control == "middle"

    null = load_eviction_config(_ns(kv_policy="null_eviction", kv_budget=1))
    assert null is not None
    assert null.name == "null_eviction"

    with pytest.raises(argparse.ArgumentTypeError, match="budget > 0"):
        load_eviction_config(_ns(kv_policy="null_eviction", kv_budget=0))
    with pytest.raises(argparse.ArgumentTypeError, match="metadata_rung"):
        load_eviction_config(
            _ns(kv_policy="metadata", kv_budget=1024, kv_metadata_rung="bad")
        )
    with pytest.raises(argparse.ArgumentTypeError, match="per_layer_budget"):
        load_eviction_config(
            _ns(kv_policy="metadata", kv_budget=1024, kv_per_layer_budget=True)
        )
    with pytest.raises(argparse.ArgumentTypeError, match="per_layer_table"):
        load_eviction_config(
            _ns(kv_policy="position_control", kv_budget=1024, kv_per_layer_table=True)
        )
    with pytest.raises(argparse.ArgumentTypeError, match="per_layer_table_path"):
        load_eviction_config(
            _ns(kv_policy="metadata", kv_budget=1024, kv_per_layer_table=True)
        )


def test_metadata_cache_update_physically_drops_and_resets_call_labels() -> None:
    cache = _metadata_cache("rung4", budget=6)
    keys = torch.arange(14 * 2, dtype=torch.float32).reshape(1, 1, 14, 2)
    values = keys + 100
    out_keys, _out_values = cache.update(keys, values, layer_idx=0)
    assert int(out_keys.shape[-2]) == 6
    assert int(cache.get_seq_length(0)) == 6

    cache.notify_new_call(1, segments=_segments(), input_token_count=14)
    one = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    cache.update(one, one, layer_idx=0)
    phase, step = cache._advance_step(0, query_len=1)
    assert (phase, step) == ("decode", 0)


def test_long_context_global_selection_is_cached_across_layers(monkeypatch) -> None:
    calls = 0
    original_default = metadata_mod.default_token_metadata

    def counting_default(original_index: int) -> metadata_mod.TokenMetadata:
        nonlocal calls
        calls += 1
        return original_default(original_index)

    monkeypatch.setattr(metadata_mod, "default_token_metadata", counting_default)
    input_tokens = 1600
    segments: list[dict] = []
    pos = 0
    role_cycle = ["system", "user", "assistant_message", "tool_result"]
    segment_idx = 0
    while pos < input_tokens:
        end = min(input_tokens, pos + 20)
        role = role_cycle[segment_idx % len(role_cycle)]
        segment = {
            "role": role,
            "token_start": pos,
            "token_end": end,
            "first_seen_call": 0,
        }
        if role == "tool_result":
            segment.update({"exit_code": segment_idx % 3, "tool_error": False})
        segments.append(segment)
        pos = end
        segment_idx += 1

    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=256,
        sink_size=4,
        recent_window=32,
        metadata_rung="rung4",
        reserve_system_prompt=False,
    )
    n_layers = 6
    n_decode_steps = 5
    cache = MetadataResidencyCache(cfg, num_layers=n_layers)
    cache.notify_new_call(0, segments=segments, input_token_count=input_tokens)

    for layer in range(n_layers):
        decision = cache._decide_evict(layer_idx=layer, key_len=input_tokens)
        cache._post_evict_hook(layer, decision)
    for _step in range(n_decode_steps):
        expected_keep: list[int] | None = None
        for layer in range(n_layers):
            decision = cache._decide_evict(layer_idx=layer, key_len=257)
            if expected_keep is None:
                expected_keep = decision.keep_indices
            else:
                assert decision.keep_indices == expected_keep
            cache._post_evict_hook(layer, decision)

    assert cache._selection_compute_count == 1 + n_decode_steps
    assert cache._selection_cache_hits == (n_layers - 1) * (1 + n_decode_steps)
    assert calls == 0


def test_layer_independent_cache_key_uses_full_original_map() -> None:
    cfg = EvictionPolicyConfig(
        name="metadata",
        budget=2,
        sink_size=0,
        recent_window=0,
        metadata_rung="rung1",
    )
    cache = MetadataResidencyCache(cfg, num_layers=1)
    cache.notify_new_call(
        0,
        segments=[
            {"role": "user", "token_start": 0, "token_end": 6, "first_seen_call": 0}
        ],
        input_token_count=6,
    )

    first = cache._cached_layer_independent_selection(
        key_len=4,
        originals=[0, 1, 2, 5],
    )
    second = cache._cached_layer_independent_selection(
        key_len=4,
        originals=[0, 3, 4, 5],
    )

    assert first.original_kept_indices == [0, 1]
    assert second.original_kept_indices == [0, 3]
    assert cache._selection_compute_count == 2
    assert cache._selection_cache_hits == 0


def test_position_controls_shape_and_structured_order() -> None:
    cfg = EvictionPolicyConfig(
        name="position_control",
        budget=6,
        sink_size=1,
        recent_window=2,
        position_control="structured",
        position_control_stride=3,
        position_control_cluster_size=2,
    )
    decision = PositionControlCache(cfg, num_layers=1)._decide_evict(0, 14)
    assert len(decision.keep_indices) == 6
    assert decision.keep_indices[:3] == [0, 1, 2]
    assert decision.keep_indices[-2:] == [12, 13]


def test_p0_segment_sim_skeleton_is_synthetic_unit_testable() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/analysis/gpu_causal_eviction_plan_20260616/segment_sim.py"
    )
    spec = importlib.util.spec_from_file_location("segment_sim_p0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mass = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    selection = module.select_top_segments_by_mass(mass, beta=1 / 3)
    assert selection.keep_mask.sum() == 1
    assert selection.ammr == pytest.approx(0.5)
    with pytest.raises(ValueError, match="complete token-id stream"):
        module.require_complete_token_id_stream({})
