"""Full-sequence downsampled attention probe for selected trace stages.

Unlike `probe_attention_maps.py`, this script does not ask transformers to
materialize full `[seq, seq]` attention tensors. It runs the normal model forward
with SDPA, and pre-hooks selected layers to compute an exact causal attention
distribution in query blocks, aggregating directly into a fixed 2D heatmap.

This makes long prompts feasible for visualization. The output is still an
attention heatmap over the complete sampled prompt sequence, downsampled to
`--downsample` bins.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from probe_attention_maps import _flatten_to_chat, build_chat_with_segments


def load_llm_calls(trace_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") == "action" and row.get("action_type") == "llm_call":
                calls.append(row)
    if not calls:
        raise ValueError(f"no llm_call records found in {trace_path}")
    return calls


def parse_indices(value: str | None, n_calls: int) -> list[int]:
    if value:
        indices = [int(item) for item in value.split(",")]
    else:
        indices = [0, n_calls // 2, n_calls - 1]
    out: list[int] = []
    for index in indices:
        if index < 0:
            index = n_calls + index
        if index < 0 or index >= n_calls:
            raise ValueError(f"sample index out of range for {n_calls} calls: {index}")
        if index not in out:
            out.append(index)
    return out


def sample_label(order: int, n_samples: int) -> str:
    if n_samples == 1:
        return "sample"
    if order == 0:
        return "begin"
    if order == n_samples - 1:
        return "near_end"
    return "middle"


def token_bin_bounds(n_tokens: int, n_bins: int) -> list[tuple[int, int]]:
    return [
        (
            int(i * n_tokens // n_bins),
            int((i + 1) * n_tokens // n_bins),
        )
        for i in range(n_bins)
    ]


def normalize_segments_for_bins(
    segments: list[dict[str, Any]], n_tokens: int, n_bins: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment in segments:
        out.append(
            {
                **segment,
                "start_bin": int(segment["start"] * n_bins // max(n_tokens, 1)),
                "end_bin": int(segment["end"] * n_bins // max(n_tokens, 1)),
            }
        )
    return out


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    bad = int((~torch.isfinite(tensor)).sum().item())
    if bad:
        raise FloatingPointError(f"{name} contains {bad} non-finite values")


def downsample_layer_attention(
    attn_module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    segments: list[dict[str, Any]],
    n_bins: int,
    block_queries: int,
) -> tuple[torch.Tensor, list[list[float]]]:
    """Return downsampled attention and exact segment-to-segment mass."""
    if hidden_states.shape[0] != 1:
        raise ValueError("only batch size 1 is supported")

    with torch.no_grad():
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)
        query_states = (
            attn_module.q_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )
        key_states = (
            attn_module.k_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        key_states = repeat_kv(key_states, attn_module.num_key_value_groups)

        query_states = query_states[0]
        key_states = key_states[0]
        n_tokens = int(query_states.shape[1])
        device = query_states.device
        key_positions = torch.arange(n_tokens, device=device)
        bin_bounds = token_bin_bounds(n_tokens, n_bins)
        row_sums = torch.zeros((n_bins, n_bins), dtype=torch.float64, device="cpu")
        row_counts = torch.zeros(n_bins, dtype=torch.float64, device="cpu")
        n_segments = len(segments)
        seg_sums = torch.zeros(
            (n_segments, n_segments), dtype=torch.float64, device="cpu"
        )
        seg_counts = torch.zeros(n_segments, dtype=torch.float64, device="cpu")
        token_segments = torch.full((n_tokens,), -1, dtype=torch.long, device=device)
        for seg_idx, segment in enumerate(segments):
            start = int(segment["start"])
            end = int(segment["end"])
            if start < end:
                token_segments[start:end] = seg_idx

        key_t = key_states.transpose(-2, -1).to(torch.float32)
        sliding_window = getattr(attn_module, "sliding_window", None)
        for q_start in range(0, n_tokens, block_queries):
            q_end = min(q_start + block_queries, n_tokens)
            q_block = query_states[:, q_start:q_end, :].to(torch.float32)
            scores = torch.matmul(q_block, key_t) * attn_module.scaling
            q_positions = torch.arange(q_start, q_end, device=device)
            mask = key_positions.unsqueeze(0) > q_positions.unsqueeze(1)
            if sliding_window is not None:
                mask |= key_positions.unsqueeze(0) <= (
                    q_positions.unsqueeze(1) - int(sliding_window)
                )
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=0)
            require_finite_tensor("head-averaged attention probabilities", probs)

            pooled_cols = []
            for k_start, k_end in bin_bounds:
                if k_start >= k_end:
                    pooled_cols.append(
                        torch.zeros(q_end - q_start, dtype=torch.float32, device=device)
                    )
                else:
                    pooled_cols.append(probs[:, k_start:k_end].mean(dim=-1))
            pooled = torch.stack(pooled_cols, dim=1).cpu().to(torch.float64)

            segment_cols = []
            for segment in segments:
                k_start = int(segment["start"])
                k_end = int(segment["end"])
                if k_start >= k_end:
                    segment_cols.append(
                        torch.zeros(q_end - q_start, dtype=torch.float32, device=device)
                    )
                else:
                    segment_cols.append(probs[:, k_start:k_end].sum(dim=-1))
            segment_mass = torch.stack(segment_cols, dim=1).cpu().to(torch.float64)
            query_segment_ids = token_segments[q_start:q_end].cpu()

            for local_row, q_pos in enumerate(range(q_start, q_end)):
                q_bin = int(q_pos * n_bins // n_tokens)
                row_sums[q_bin] += pooled[local_row]
                row_counts[q_bin] += 1.0

            for seg_idx in range(n_segments):
                query_mask = query_segment_ids == seg_idx
                query_count = int(query_mask.sum().item())
                if query_count:
                    seg_sums[seg_idx] += segment_mass[query_mask].sum(dim=0)
                    seg_counts[seg_idx] += query_count

        row_counts = row_counts.clamp_min(1.0).unsqueeze(1)
        map_downsampled = (row_sums / row_counts).to(torch.float32)
        require_finite_tensor("downsampled prefill attention map", map_downsampled)
        seg_counts = seg_counts.clamp_min(1.0).unsqueeze(1)
        seg_to_seg = (seg_sums / seg_counts).tolist()
        return map_downsampled, seg_to_seg


def decode_layer_attention(
    attn_module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cached_keys: torch.Tensor,
) -> torch.Tensor:
    """Return head-averaged next-token attention over prompt plus current token."""
    if hidden_states.shape[:2] != (1, 1):
        raise ValueError("decode attention expects batch size 1 and query length 1")

    with torch.no_grad():
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)
        query_states = (
            attn_module.q_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )
        key_states = (
            attn_module.k_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        key_states = torch.cat([cached_keys, key_states], dim=-2)
        key_states = repeat_kv(key_states, attn_module.num_key_value_groups)

        query_states = query_states[0].to(torch.float32)
        key_states = key_states[0].to(torch.float32)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        scores = scores * attn_module.scaling

        sliding_window = getattr(attn_module, "sliding_window", None)
        if sliding_window is not None:
            n_keys = int(key_states.shape[-2])
            query_position = n_keys - 1
            key_positions = torch.arange(n_keys, device=key_states.device)
            mask = key_positions <= (query_position - int(sliding_window))
            scores = scores.masked_fill(mask.view(1, 1, -1), torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=0)[0]
        require_finite_tensor("decode attention probabilities", probs)
        return probs


def downsample_decode_attention(attn_vector: torch.Tensor, n_bins: int) -> list[float]:
    require_finite_tensor("decode attention vector", attn_vector)
    x = attn_vector.reshape(1, 1, -1)
    return F.adaptive_avg_pool1d(x, n_bins).reshape(-1).cpu().tolist()


def decode_segment_mass(
    attn_vector: torch.Tensor, segments: list[dict[str, Any]]
) -> list[float]:
    out: list[float] = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        if start >= end:
            out.append(0.0)
        else:
            out.append(float(attn_vector[start:end].sum().item()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="27")
    parser.add_argument(
        "--sample-call-indices",
        help="Comma-separated 0-based llm_call indices. Default: begin,middle,last.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--downsample", type=int, default=256)
    parser.add_argument("--block-queries", type=int, default=32)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    args = parser.parse_args()

    if args.downsample <= 0:
        raise SystemExit("--downsample must be positive")
    if args.block_queries <= 0:
        raise SystemExit("--block-queries must be positive")

    target_layers = sorted(int(item) for item in args.layers.split(","))
    calls = load_llm_calls(args.trace)
    sample_indices = parse_indices(args.sample_call_indices, len(calls))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    n_layers_total = model.config.num_hidden_layers
    bad_layers = [
        layer for layer in target_layers if layer < 0 or layer >= n_layers_total
    ]
    if bad_layers:
        raise SystemExit(
            f"layer indices out of range (model has {n_layers_total}): {bad_layers}"
        )

    samples: list[dict[str, Any]] = []
    for order, call_index in enumerate(sample_indices):
        row = calls[call_index]
        messages = row["data"].get("messages_in") or []
        chat = _flatten_to_chat(messages)
        input_ids, segments = build_chat_with_segments(
            tokenizer, chat, args.max_input_tokens
        )
        if input_ids is None or segments is None:
            prompt_tokens = row.get("data", {}).get("prompt_tokens", "unknown")
            raise SystemExit(
                f"sample call {call_index} exceeds --max-input-tokens "
                f"({prompt_tokens} recorded prompt tokens)"
            )

        layer_maps: dict[int, torch.Tensor] = {}
        layer_seg_to_seg: dict[int, list[list[float]]] = {}
        handles = []
        for layer_idx in target_layers:
            attn_module = model.model.layers[layer_idx].self_attn

            def _hook(module, hook_args, hook_kwargs, layer=layer_idx):
                hidden_states = hook_kwargs.get("hidden_states")
                if hidden_states is None:
                    hidden_states = hook_args[0]
                position_embeddings = hook_kwargs.get("position_embeddings")
                if position_embeddings is None:
                    position_embeddings = hook_args[1]
                map_downsampled, seg_to_seg = downsample_layer_attention(
                    module,
                    hidden_states.detach(),
                    position_embeddings,
                    segments,
                    args.downsample,
                    args.block_queries,
                )
                layer_maps[layer] = map_downsampled
                layer_seg_to_seg[layer] = seg_to_seg

            handles.append(attn_module.register_forward_pre_hook(_hook, with_kwargs=True))

        input_ids = input_ids.to("cuda:0")
        t0 = time.time()
        with torch.no_grad():
            prefill = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.time() - t0) * 1000
        for handle in handles:
            handle.remove()

        missing = [layer for layer in target_layers if layer not in layer_maps]
        if missing:
            raise RuntimeError(f"missing captured layers: {missing}")

        next_token = prefill.logits[:, -1:, :].argmax(dim=-1)
        layer_decode_attn: dict[int, torch.Tensor] = {}
        decode_handles = []
        for layer_idx in target_layers:
            attn_module = model.model.layers[layer_idx].self_attn

            def _decode_hook(module, hook_args, hook_kwargs, layer=layer_idx):
                hidden_states = hook_kwargs.get("hidden_states")
                if hidden_states is None:
                    hidden_states = hook_args[0]
                position_embeddings = hook_kwargs.get("position_embeddings")
                if position_embeddings is None:
                    position_embeddings = hook_args[1]
                cached_keys = prefill.past_key_values.layers[layer].keys
                layer_decode_attn[layer] = decode_layer_attention(
                    module,
                    hidden_states.detach(),
                    position_embeddings,
                    cached_keys,
                )

            decode_handles.append(
                attn_module.register_forward_pre_hook(_decode_hook, with_kwargs=True)
            )

        t1 = time.time()
        try:
            with torch.no_grad():
                decode = model(
                    input_ids=next_token,
                    past_key_values=prefill.past_key_values,
                    use_cache=False,
                )
            torch.cuda.synchronize()
            decode_ms = (time.time() - t1) * 1000
        finally:
            for handle in decode_handles:
                handle.remove()

        missing_decode = [layer for layer in target_layers if layer not in layer_decode_attn]
        if missing_decode:
            raise RuntimeError(f"missing captured decode layers: {missing_decode}")
        del decode

        layer_records: list[dict[str, Any]] = []
        prompt_len = int(input_ids.shape[1])
        for layer in target_layers:
            decode_attn_full = layer_decode_attn[layer]
            if decode_attn_full.shape[0] < prompt_len:
                raise RuntimeError(
                    f"decode attention shorter than prompt for layer {layer}: "
                    f"{decode_attn_full.shape[0]} < {prompt_len}"
                )
            decode_attn = decode_attn_full[:prompt_len]
            decode_self_mass = float(decode_attn_full[prompt_len:].sum().item())
            layer_records.append(
                {
                    "layer": layer,
                    "prefill_map_downsampled": layer_maps[layer].tolist(),
                    "prefill_seg_to_seg": layer_seg_to_seg[layer],
                    "decode_attn_downsampled": downsample_decode_attention(
                        decode_attn, args.downsample
                    ),
                    "decode_to_seg": decode_segment_mass(decode_attn, segments),
                    "decode_self_mass": decode_self_mass,
                }
            )

        sample = {
            "label": sample_label(order, len(sample_indices)),
            "call_index": call_index,
            "trace_iteration": int(row.get("iteration", call_index)),
            "input_tokens": int(input_ids.shape[1]),
            "recorded_prompt_tokens": row.get("data", {}).get("prompt_tokens"),
            "segments": normalize_segments_for_bins(
                segments, int(input_ids.shape[1]), args.downsample
            ),
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "next_token_id": int(next_token.item()),
            "next_token_text": tokenizer.decode(next_token[0].tolist()),
            "layers": layer_records,
        }
        samples.append(sample)
        print(
            f"{sample['label']}: call {call_index}, iter {sample['trace_iteration']}, "
            f"{sample['input_tokens']} tokens",
            file=sys.stderr,
        )
        del prefill
        torch.cuda.empty_cache()

    payload = {
        "model": args.model,
        "trace": str(args.trace),
        "target_layers": target_layers,
        "sample_call_indices": sample_indices,
        "downsample": args.downsample,
        "block_queries": args.block_queries,
        "dtype": args.dtype,
        "semantics": {
            "prefill_map_downsampled": "full prompt sequence, causal attention, blockwise aggregated into bins",
            "prefill_seg_to_seg": "per query segment, mean total attention mass assigned to each key segment",
            "decode_attn_downsampled": "next greedy token query attention to prompt tokens only; generated-token self mass is excluded",
            "decode_to_seg": "next greedy token query attention mass assigned to each prompt segment",
            "decode_self_mass": "attention mass assigned by the next greedy token to itself",
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
