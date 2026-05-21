"""Pre-hook combine semantics: sparse mask must NOT additively overflow fp16.

When HF supplies an upstream causal mask whose blocked cells are already at
`finfo(fp16).min` (≈ -65504), a naive `existing + sparse_for_add` doubles to
-131008 which overflows fp16 to `-inf`. Some SDPA backends turn an all-`-inf`
attention row into NaN. The correct semantics is "force-mask" — sparse-masked
positions become `finfo.min` regardless of `existing`, kept positions
preserve `existing`.

Path is dormant on Qwen3-0.6B (prefill calls `self_attn(attention_mask=None)`,
so the `existing is None` branch fires), but activates on padded-batch inputs
and on future per-query methods (Quest, MInference) that mask aggressively.
"""

from __future__ import annotations

import torch
from torch import nn

from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer
from serving.sparse_attention.sliding import SlidingWindowSparseAttention


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

    def forward(self, hidden_states, **_kwargs):  # pragma: no cover - pre-hook only
        return hidden_states, None


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()


def _make_capturer() -> LayerCapturer:
    return LayerCapturer(
        _ToyModel(),
        config=RecordingConfig(attention_top_k=2, decode_window=2, max_prefill_queries=4),
        model_summary={"name": "toy"},
        sparse_attention=SlidingWindowSparseAttention(sink_size=2, recent_window=4),
    )


def test_sparse_attn_pre_hook_handles_fp16_additive_overflow() -> None:
    capturer = _make_capturer()
    pre_hook = capturer._sparse_pre_hook(layer=0)

    Q = K = 8
    dtype = torch.float16
    neg_inf_fp16 = torch.finfo(dtype).min

    # Upstream causal mask: above the diagonal = finfo.min, on/below = 0.
    causal = torch.zeros((1, 1, Q, K), dtype=dtype)
    for q in range(Q):
        for k in range(K):
            if k > q:
                causal[0, 0, q, k] = neg_inf_fp16

    hidden_states = torch.zeros((1, Q, 8), dtype=dtype)

    kwargs = {"attention_mask": causal.clone(), "past_key_values": None}
    _args, new_kwargs = pre_hook(
        module=capturer,  # unused inside pre_hook (deleted via `del module`)
        args=(hidden_states,),
        kwargs=kwargs,
    )

    out = new_kwargs["attention_mask"]
    assert out.dtype == dtype, f"dtype changed: {out.dtype}"
    assert not torch.isnan(out).any(), "NaN cells in combined mask"
    assert not torch.isinf(out).any(), "inf cells (fp16 overflow) in combined mask"

    # sink_size=2, recent_window=4, key_len=8 → kept keys = {0,1, 4,5,6,7}.
    # Sparse-masked keys = {2, 3}.
    sparse_masked = {2, 3}

    for q in range(Q):
        for k in range(K):
            cell = float(out[0, 0, q, k])
            causal_blocks = k > q
            sparse_blocks = k in sparse_masked
            if sparse_blocks:
                # Sparse force-masks regardless of upstream.
                assert cell == neg_inf_fp16, (
                    f"(q={q},k={k}) sparse-masked must be finfo.min, got {cell}"
                )
            elif causal_blocks:
                # Sparse keeps it but causal blocks — existing finfo.min preserved.
                assert cell == neg_inf_fp16, (
                    f"(q={q},k={k}) causal-only mask must stay finfo.min, got {cell}"
                )
            else:
                # Both keep — preserve existing 0.
                assert cell == 0.0, (
                    f"(q={q},k={k}) both-keep must be 0, got {cell}"
                )

    # Bonus: explicit witness for the cell that WOULD have overflowed under
    # the buggy `existing + sparse_for_add`. (q=1, k=3): causal blocks (k>q=1)
    # AND sparse blocks (k=3 middle). Result must be finfo.min, NOT -inf and
    # NOT -2 * finfo.min.
    witness = float(out[0, 0, 1, 3])
    assert witness == neg_inf_fp16
    assert witness != float("-inf")
