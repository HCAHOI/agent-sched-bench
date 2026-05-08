from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer
from serving.recording.recording import segment_bucket


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
        },
        {
            "role": "user",
            "message_index": 1,
            "token_start": 2,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
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

    with np.load(call_dir / "attention.npz") as attention:
        assert int(attention["call_idx"]) == 0
        assert attention["segment_mass"].shape[1] == 3
        assert attention["query_heads"].shape == attention["query_positions"].shape
        assert np.allclose(attention["segment_mass"].sum(axis=1), 1.0, atol=1e-6)
        assert attention["topk_indices"].shape[1] == 2

    with np.load(call_dir / "routing.npz") as routing:
        assert int(routing["n_experts"]) == 3
        assert routing["expert_choice"].shape == (5, 2)
        assert routing["expert_load"].shape == (1, 3, 3)

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["iters"][0]["call_idx"] == 0
    assert meta["iters"][0]["attention_records"] == 2


def test_layer_capturer_rejects_model_without_attention_modules() -> None:
    model = nn.Linear(2, 2)

    import pytest

    with pytest.raises(ValueError, match="no attention modules"):
        LayerCapturer(
            model,
            config=RecordingConfig(),
            model_summary={"name": "no-attn"},
        )


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
