# Copy of /workspace/outlen/aux-patch/sitecustomize.py on the GPU host (lane 4, 2026-09-13).
"""Selected-layer token features from a vLLM pooling server.

Active only when OUTLEN_AUX_LAYERS is set (e.g. "2,24,45"): Qwen3-MoE forward collects the residual stream after
those layers (vLLM's EAGLE-3 aux-hidden-state hook) and returns [aux_2 | aux_24 | aux_45 | final-normed] per token,
so the LAST pooler yields a 4 x hidden vector. Loaded via PYTHONPATH=/workspace/outlen/aux-patch in the server env.
OUTLEN_AUX_ONLY=1 omits the final state for single-layer comparisons without duplicating the existing cache.
OUTLEN_LAST_FFN=1 instead exports [MLP input | pre-MLP residual | MLP update | final normalized state].
This diagnostic requires --enforce-eager and LAST sequence pooling for prefix-end observations.
"""
import os
from functools import wraps

_layers = os.environ.get("OUTLEN_AUX_LAYERS")
_last_ffn = os.environ.get("OUTLEN_LAST_FFN") == "1"
if _layers and _last_ffn:
    raise ValueError("Select auxiliary layers or last-FFN diagnostics, not both")
if _layers or _last_ffn:
    import torch
    from vllm.model_executor.models import qwen3_moe as _m

    _LAYERS = tuple(int(x) for x in _layers.split(",")) if _layers else ()
    _orig = _m.Qwen3MoeModel.forward

    if _last_ffn:
        _orig_init = _m.Qwen3MoeModel.__init__

        @wraps(_orig_init)
        def _init(self, *args, **kwargs):
            config = kwargs["vllm_config"]
            if not config.model_config.enforce_eager or config.parallel_config.pipeline_parallel_size != 1:
                raise ValueError("Last-FFN diagnostics require eager execution and pipeline parallel size 1")
            _orig_init(self, *args, **kwargs)

        _m.Qwen3MoeModel.__init__ = _init

    @wraps(_orig)  # vLLM binds this signature to mark token dimensions dynamic.
    def _forward(self, *args, **kwargs):
        if _last_ffn:
            captured = {}

            def after_norm(module, inputs, output):
                # Fused final RMSNorm mutates its inputs; retain the pre-update values.
                captured["input"], captured["residual"] = (x.detach().clone() for x in output)

            def after_mlp(module, inputs, output):
                captured["update"] = output.detach().clone()

            layer = self.layers[-1]
            norm_hook = layer.post_attention_layernorm.register_forward_hook(after_norm)
            mlp_hook = layer.mlp.register_forward_hook(after_mlp)
            try:
                hidden = _orig(self, *args, **kwargs)
            finally:
                norm_hook.remove()
                mlp_hook.remove()
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError("Last-FFN diagnostic expected a single final-state tensor")
            return torch.cat([captured["input"], captured["residual"], captured["update"], hidden], dim=-1)
        self.aux_hidden_state_layers = _LAYERS
        out = _orig(self, *args, **kwargs)
        if isinstance(out, tuple):
            hidden, aux = out
            if os.environ.get("OUTLEN_AUX_ONLY") == "1":
                if len(aux) != len(_LAYERS):
                    raise RuntimeError("Requested auxiliary layers were not all returned")
                return torch.cat(aux, dim=-1)
            return torch.cat([*aux, hidden], dim=-1)
        if os.environ.get("OUTLEN_AUX_ONLY") == "1":
            raise RuntimeError("No auxiliary states returned")
        return out

    _m.Qwen3MoeModel.forward = _forward
    print(f"[aux-patch] Qwen3MoeModel.forward layers {_LAYERS}, aux_only={os.environ.get('OUTLEN_AUX_ONLY') == '1'}, last_ffn={_last_ffn}", flush=True)
