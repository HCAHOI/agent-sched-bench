from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer
from serving.recording.recording import segment_bucket
from scripts.recoding_figures.recording_loader import decode_attention_topk


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 4
        self.num_key_value_groups = 1
        self.scaling = 0.5
        self.q_proj = nn.Linear(8, 4, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
    ):
        del position_embeddings, attention_mask, past_key_values
        return hidden_states, None


class _FakeCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.key_states = key_states

    def __getitem__(self, layer_idx: int):
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return self.key_states, None


class _LayerOnlyCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.layers = [SimpleNamespace(keys=key_states)]


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = nn.Linear(8, 3, bias=False)


def test_layer_capturer_writes_attention_routing_and_segments(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2, max_prefill_queries=4),
        model_summary={
            "name": "toy",
            "num_experts_per_tok": 2,
        },
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)

    segments = [
        {
            "role": "system",
            "message_index": 0,
            "token_start": 0,
            "token_end": 2,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        },
        {
            "role": "user",
            "message_index": 1,
            "token_start": 2,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        },
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        cos4 = torch.ones(1, 4, 4, dtype=torch.float32)
        sin4 = torch.zeros(1, 4, 4, dtype=torch.float32)
        non_contiguous_hidden = torch.zeros(1, 8, 4).transpose(1, 2)
        attn(
            non_contiguous_hidden,
            position_embeddings=(cos4, sin4),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 5, 4)),
        )
        router_logits = (
            torch.tensor(
                [
                    [4.0, 1.0, 0.0],
                    [1.0, 4.0, 0.0],
                    [0.0, 1.0, 4.0],
                    [4.0, 0.0, 1.0],
                    [1.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            ),
        )
        capturer.record_router_logits(
            SimpleNamespace(router_logits=router_logits),
            total_tokens=5,
        )
        capturer.flush(output_token_ids=[42])
    capturer.set_attempt_extra_meta(
        {
            "session_history": [
                {
                    "call_idx": 0,
                    "used_session_cache": False,
                    "lcp": 0,
                    "cached_len_before": 0,
                    "new_len": 4,
                    "delta_len": 4,
                    "diverged": False,
                }
            ]
        }
    )
    capturer.finish_attempt()

    call_dir = recordings_dir / "iter_0000"
    assert (call_dir / "attention.npz").exists()
    assert (call_dir / "routing.npz").exists()
    assert (call_dir / "segments.json").exists()

    segments_payload = json.loads((call_dir / "segments.json").read_text())
    assert segments_payload["call_idx"] == 0
    assert segments_payload["total_tokens"] == 5
    assert len(segments_payload["token_segment_id"]) == 5
    assert segments_payload["segments"][-1]["role"] == "generation"
    assert [segment["first_seen_call"] for segment in segments_payload["segments"]] == [
        0,
        0,
        0,
    ]
    assert [
        segment["first_seen_call_inferred"]
        for segment in segments_payload["segments"]
    ] == [False, False, False]

    with np.load(call_dir / "attention.npz") as attention:
        assert int(attention["call_idx"]) == 0
        assert attention["segment_mass"].shape[1] == 3
        assert attention["query_heads"].shape == attention["query_positions"].shape
        assert np.allclose(attention["segment_mass"].sum(axis=1), 1.0, atol=1e-6)
        assert attention["topk_indices"].shape[1] == 2
        decoded_indices, decoded_weights = decode_attention_topk(attention)
        np.testing.assert_array_equal(decoded_indices, attention["topk_indices"])
        np.testing.assert_array_equal(decoded_weights, attention["topk_weights"])
        assert attention["topk_csr_offsets"].shape[0] == int(attention["n_query_rows"]) + 1
        span_span = attention["span_span_matrix"]
        span_counts = attention["span_span_query_counts"]
        assert span_span.shape == (2, 3, 3)
        assert span_counts.shape == (2, 3)
        active_rows = span_counts > 0
        assert np.allclose(span_span.sum(axis=-1)[active_rows], 1.0, atol=1e-6)

    with np.load(call_dir / "routing.npz") as routing:
        assert int(routing["n_experts"]) == 3
        assert routing["expert_choice"].shape == (5, 2)
        assert routing["expert_load"].shape == (1, 3, 3)
        assert routing["expert_token_count"].shape == (1, 3, 3)
        assert int(routing["expert_token_count"].sum()) == 5
        assert routing["expert_capacity"].tolist() == [2]
        assert routing["drop_signal_mode"].tolist() == ["expected_uniform_capacity"]
        assert int(routing["expected_dropped_token_count"][0]) == int(
            routing["expert_expected_overflow_count"][0].sum()
        )

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["iters"][0]["call_idx"] == 0
    assert meta["iters"][0]["attention_records"] == 2
    assert meta["session_history"][0]["call_idx"] == 0


def test_layer_capturer_rejects_model_without_attention_modules() -> None:
    model = nn.Linear(2, 2)

    import pytest

    with pytest.raises(ValueError, match="no attention modules"):
        LayerCapturer(
            model,
            config=RecordingConfig(),
            model_summary={"name": "no-attn"},
        )


def test_layer_capturer_records_router_gate_hook_without_second_forward(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={
            "name": "toy",
            "num_experts_per_tok": 2,
        },
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(
                torch.ones(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        model.model.layers[0].mlp.gate(torch.zeros(1, 4, 8))
        model.model.layers[0].mlp.gate(torch.zeros(1, 1, 8))
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    segments_payload = json.loads(
        (recordings_dir / "iter_0000" / "segments.json").read_text()
    )
    assert [
        segment["first_seen_call_inferred"]
        for segment in segments_payload["segments"]
    ] == [True, False]

    with np.load(recordings_dir / "iter_0000" / "routing.npz") as routing:
        assert routing["record_path"].tolist() == ["gate", "gate"]
        assert routing["record_layer"].tolist() == [0, 0]
        assert routing["expert_choice"].shape == (5, 2)
        assert routing["expert_load"].shape == (2, 2, 3)
        assert routing["expert_token_count"].shape == (2, 2, 3)
        assert routing["expert_token_count"].sum(axis=(1, 2)).tolist() == [4, 1]
        assert routing["drop_signal_mode"].tolist() == [
            "expected_uniform_capacity",
            "expected_uniform_capacity",
        ]

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["model"]["router_capture_mode"] == "gate_forward_hook"


def test_layer_capturer_clears_records_after_failed_session(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={"name": "toy"},
    )
    capturer.start_attempt(tmp_path / "recordings")
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        with capturer.recording_session(
            call_idx=0,
            segments=segments,
            input_token_count=4,
        ):
            model.model.layers[0].self_attn(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            assert capturer._prefill_records
            raise RuntimeError("boom")

    assert capturer._prefill_records == []
    assert len(capturer._decode_records) == 0
    assert capturer._routing_records == []


def test_layer_capturer_aligns_meta_iters_to_trace_actions(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]
    attn = model.model.layers[0].self_attn

    for call_idx in range(2):
        with capturer.recording_session(
            call_idx=call_idx,
            segments=segments,
            input_token_count=4,
        ):
            attn(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            capturer.flush(output_token_ids=[42])

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"type": "trace_metadata"}) + "\n"
        + json.dumps(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm_1",
                "iteration": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    capturer.finish_attempt(trace_path=trace_path)

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["alignment"] == {
        "trace_path": str(trace_path),
        "trace_llm_actions": 1,
        "recording_iters": 2,
        "aligned_iters": 1,
        "orphan_iters": 1,
        "alignment_source": "iteration",
    }
    assert [item["call_idx"] for item in meta["iters"]] == [0]
    assert meta["iters"][0]["trace_action_id"] == "llm_1"
    assert [item["call_idx"] for item in meta["orphan_iters"]] == [1]
    assert meta["orphan_iters"][0]["trace_alignment_status"] == "orphan_no_llm_action"
    assert (recordings_dir / "iter_0001" / "attention.npz").exists()


def test_layer_capturer_uses_trace_indices_for_non_tail_gaps(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]
    attn = model.model.layers[0].self_attn

    for call_idx in range(3):
        with capturer.recording_session(
            call_idx=call_idx,
            segments=segments,
            input_token_count=4,
        ):
            attn(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            capturer.flush(output_token_ids=[42])

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_metadata"}),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "iteration": 0,
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_2",
                        "iteration": 2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    capturer.finish_attempt(trace_path=trace_path)

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert [item["call_idx"] for item in meta["iters"]] == [0, 2]
    assert [item["trace_action_id"] for item in meta["iters"]] == ["llm_0", "llm_2"]
    assert [item["call_idx"] for item in meta["orphan_iters"]] == [1]
    assert meta["alignment"]["aligned_iters"] == 2
    assert meta["alignment"]["orphan_iters"] == 1


def test_segment_bucket_normalizes_low_precision_rows() -> None:
    attn_rows = torch.tensor([[0.2500, 0.2490, 0.2500, 0.2490]], dtype=torch.bfloat16)
    token_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    segment_mass = segment_bucket(attn_rows, token_ids, n_segments=2)

    assert torch.allclose(segment_mass.sum(dim=1), torch.ones(1), atol=1e-6)


def test_layer_capturer_reads_cache_layer_keys_for_decode(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_LayerOnlyCache(torch.zeros(1, 1, 6, 4)),
        )
        capturer.flush(output_token_ids=[42, 43])
    capturer.finish_attempt()

    with np.load(recordings_dir / "iter_0000" / "attention.npz") as attention:
        assert attention["record_phase"].tolist() == ["decode"]
        assert attention["query_positions"].tolist() == [5]
        assert int(attention["n_query_rows"]) == 1
