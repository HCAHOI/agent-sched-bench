"""Small tensor utilities for per-call internal recordings."""

from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordingConfig:
    """Runtime knobs for bounded internal recording."""

    record_artifacts: bool = True
    attention_top_k: int = 32
    decode_window: int = 64
    max_prefill_queries: int = 80
    model_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    generation_seed: int = 0
    # Layers for which per-head per-span attention stats are captured.
    # Empty tuple disables the feature. Must all be < num_hidden_layers.
    # Recommended for real models: (0, 6, 12, 18, 24, 30, 36, 47)
    # Layer 47 is the last attention layer in Qwen3-Coder-30B (retrieval/copy heads concentrate there).
    per_head_stats_layers: tuple[int, ...] = ()
    # When True, additionally capture per-selected-block within-block attention
    # mean/std at decode (bucket axis = [sink, rank1..R_max, recent]). Only valid
    # with an active block_topk sparse method and non-empty per_head_stats_layers;
    # the CLI gate enforces both. block_size/sink/recent are read at runtime from
    # the sparse method instance (single source of truth), not duplicated here.
    per_head_block_stats: bool = False
    # When True, record block_topk's per-head independent top-`per_head_topk_rank`
    # block selections at each decode step (the counterfactual "what would each
    # head pick alone" set, uncensored by pooling). Reuses per_head_stats_layers
    # for the layer set (empty = nothing recorded). Only meaningful with an active
    # block_topk method; the CLI gate enforces it. The sparse method computes the
    # per-head block scores regardless of score_reduction when this is on; off =
    # zero extra compute / GPU memory (no [H, n_blocks] materialization).
    record_per_head_topk: bool = False
    # Per-head rank cap R_ph for record_per_head_topk. Default 64 sits ABOVE the
    # typical middle-block budget (≈48 at budget=1024/block_size=16) so the export
    # cap is never the binding constraint; per-step rows are capped at the
    # candidate-block count nb anyway. Lower it in YAML to cut storage
    # (≈ layers × decode_steps × H × R_ph × 6 bytes per call, see hooks).
    per_head_topk_rank: int = 64


def segment_role(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    if role == "assistant" and message.get("tool_calls"):
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role == "tool":
        return "tool_result"
    return role


# OpenClaw's exec tool (implemented in agents/openclaw/tools/shell.py) appends a
# literal "Exit code: <N>" line to every command result; this is the only
# structured exit signal in the tool-result string. The recorder sees the same
# string the serving system receives at inference time, so this parse is
# inference-time-legitimate.
_EXIT_CODE_PATTERN = re.compile(r"(?m)^\s*Exit code:\s*(-?\d+)\s*$")
# Tool failure marker shared across all OpenClaw tools: a result whose first
# non-blank line starts with "Error" (e.g. "Error:", "Error executing command:",
# "Error reading file:"). Conservative — matched only at the start so an "Error"
# substring mid-output does not trigger a false positive.
_TOOL_ERROR_PATTERN = re.compile(r"^\s*Error\b")


def parse_tool_exit_code(content: Any) -> int | None:
    """Extract an exec-style exit code from a tool-result string.

    Returns the integer following the last ``Exit code: <N>`` line, or None when
    no such line exists (non-exec tools, truncated-away tail, unparseable). Never
    guesses 0 — absence stays None so downstream cannot mistake "unknown" for
    "success". The serving system receives this same result text as the tool runs,
    so reading it is inference-time-legitimate (no oracle/hindsight signal).
    """
    if not isinstance(content, str) or not content:
        return None
    matches = _EXIT_CODE_PATTERN.findall(content)
    if not matches:
        return None
    return int(matches[-1])


def detect_tool_error(content: Any, *, exit_code: int | None = None) -> bool | None:
    """Best-effort failure flag for a tool result.

    True when ``exit_code`` is nonzero, or when the result string's first non-blank
    line starts with ``Error`` (the convention every OpenClaw tool follows for
    failures). False when an exit code of 0 is present or the result is a non-empty
    string with no error marker. None when there is no string content and no exit
    code to judge from. Over-budget results persisted to disk (content replaced by
    a "[tool output persisted]" head preview) carry no error marker and therefore
    yield False (no-error-signal), not None. Observe-only: derived from the result
    text already in the message, never altering it.
    """
    if exit_code is not None:
        if exit_code != 0:
            return True
        # exit_code == 0 is an explicit success signal from an exec tool.
        return False
    if not isinstance(content, str) or not content:
        return None
    return bool(_TOOL_ERROR_PATTERN.match(content))


def query_sampling_seed(base_seed: int, call_idx: int) -> str:
    """Stable per-call seed for bounded prefill query-row sampling."""
    return f"{int(base_seed)}:{int(call_idx)}"


def select_query_positions(
    query_len: int,
    max_queries: int,
    *,
    seed: int | str = 0,
) -> list[int]:
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")
    if query_len <= max_queries:
        return list(range(query_len))
    rng = random.Random(seed)
    positions: list[int] = []
    for idx in range(max_queries):
        start = (idx * query_len) // max_queries
        stop = ((idx + 1) * query_len) // max_queries
        positions.append(rng.randrange(start, stop))
    return positions


def token_segment_ids(
    total_tokens: int,
    segments: list[dict[str, Any]],
    *,
    generated_segment_id: int | None = None,
):
    import torch

    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    fill = -1 if generated_segment_id is None else generated_segment_id
    ids = torch.full((total_tokens,), fill, dtype=torch.long)
    for idx, segment in enumerate(segments):
        start = int(segment.get("token_start", 0))
        end = int(segment.get("token_end", 0))
        if start < 0 or end < start:
            raise ValueError(f"invalid segment bounds: {segment}")
        if start >= total_tokens:
            continue
        ids[start : min(end, total_tokens)] = idx
    return ids


def segment_bucket(attn_rows, token_ids, n_segments: int):
    import torch

    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if attn_rows.ndim == 4:
        attn_rows = attn_rows[0].mean(dim=0)
    elif attn_rows.ndim == 3:
        attn_rows = attn_rows.mean(dim=0)
    if attn_rows.ndim != 2:
        raise ValueError(f"attention rows must be rank 2 after head mean: {attn_rows.shape}")

    rows = attn_rows.to(dtype=torch.float32)
    key_ids = token_ids[: rows.shape[1]].to(device=rows.device)
    out = torch.zeros(
        (rows.shape[0], n_segments), dtype=torch.float32, device=rows.device
    )
    valid = (key_ids >= 0) & (key_ids < n_segments)
    if bool(valid.any()):
        index = key_ids[valid].view(1, -1).expand(rows.shape[0], -1)
        out.scatter_add_(1, index, rows[:, valid])
    row_sums = out.sum(dim=1, keepdim=True)
    nonzero = row_sums > 0
    out = torch.where(nonzero, out / row_sums.clamp_min(torch.finfo(out.dtype).tiny), out)
    return out


def padded_top_k(rows, k: int):
    import torch

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank 2, got {rows.shape}")
    effective_k = min(k, int(rows.shape[1]))
    weights, indices = torch.topk(rows, k=effective_k, dim=-1)
    weights = weights.to(dtype=torch.float32)
    if effective_k == k:
        return indices, weights
    pad_cols = k - effective_k
    index_pad = torch.full(
        (rows.shape[0], pad_cols), -1, dtype=indices.dtype, device=indices.device
    )
    weight_pad = torch.zeros(
        (rows.shape[0], pad_cols), dtype=weights.dtype, device=weights.device
    )
    return torch.cat([indices, index_pad], dim=1), torch.cat([weights, weight_pad], dim=1)


def heavy_hitter(rows, k: int):
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank 2, got {rows.shape}")
    return padded_top_k(rows.mean(dim=0, keepdim=True, dtype=rows.dtype), k)


def expert_load_per_segment(
    router_logits,
    token_ids,
    *,
    n_segments: int,
    top_k_experts: int,
):
    import torch

    if router_logits.ndim < 2:
        raise ValueError(f"router_logits must end with expert dimension: {router_logits.shape}")
    logits = router_logits.reshape(-1, router_logits.shape[-1]).to(dtype=torch.float32)
    n_tokens, n_experts = int(logits.shape[0]), int(logits.shape[1])
    k = min(top_k_experts, n_experts)
    weights, choices = torch.topk(torch.softmax(logits, dim=-1), k=k, dim=-1)
    load = torch.zeros((n_segments, n_experts), dtype=torch.float32, device=logits.device)

    segment_ids = token_ids[:n_tokens].to(device=logits.device)
    valid = (segment_ids >= 0) & (segment_ids < n_segments)
    # `index_put_` with empty index tensors is a safe no-op, so we skip the
    # `valid.any()` host sync; the accumulation result is identical when no
    # rows are valid (same pattern as the routing-count summary in hooks.py).
    valid_segments = segment_ids[valid]
    for rank in range(k):
        load.index_put_(
            (valid_segments, choices[valid, rank]),
            weights[valid, rank],
            accumulate=True,
        )
    return choices, weights, load


class DecodeRingBuffer:
    """Keep only the most recent decode records."""

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._dropped = 0

    def append(self, item: dict[str, Any]) -> None:
        if len(self._items) == self._items.maxlen:
            self._dropped += 1
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()
        self._dropped = 0

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def dropped_count(self) -> int:
        return int(self._dropped)

    def __len__(self) -> int:
        return len(self._items)
