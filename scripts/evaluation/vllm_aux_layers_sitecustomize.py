# Copy of /workspace/outlen/aux-patch/sitecustomize.py on the GPU host (lane 4, 2026-09-13).
"""Multi-layer last-token features from a vLLM pooling server (OUTLETS-style fused layers).

Active only when OUTLEN_AUX_LAYERS is set (e.g. "2,24,45"): Qwen3-MoE forward collects the residual stream after
those layers (vLLM's EAGLE-3 aux-hidden-state hook) and returns [aux_2 | aux_24 | aux_45 | final-normed] per token,
so the LAST pooler yields a 4 x hidden vector. Loaded via PYTHONPATH=/workspace/outlen/aux-patch in the server env.
"""
import os

_layers = os.environ.get("OUTLEN_AUX_LAYERS")
if _layers:
    import torch
    from vllm.model_executor.models import qwen3_moe as _m

    _LAYERS = tuple(int(x) for x in _layers.split(","))
    _orig = _m.Qwen3MoeModel.forward

    def _forward(self, *args, **kwargs):
        self.aux_hidden_state_layers = _LAYERS
        out = _orig(self, *args, **kwargs)
        if isinstance(out, tuple):
            hidden, aux = out
            return torch.cat([*aux, hidden], dim=-1)
        return out

    _m.Qwen3MoeModel.forward = _forward
    print(f"[aux-patch] Qwen3MoeModel.forward returns layers {_LAYERS} + final", flush=True)
