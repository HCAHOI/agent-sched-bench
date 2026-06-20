"""Allow-lists for provider-specific fields copied into OpenClaw traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


HF_TRACE_EXTRA_KEYS: tuple[str, ...] = (
    "hf_call_idx",
    "hf_input_token_count",
    "hf_delta_input_token_count",
    "hf_used_session_cache",
    "hf_session_cache_type",
    "hf_session",
    "hf_cache_lcp",
    "hf_cache_cached_len_before",
    "hf_cache_new_len",
    "hf_cache_delta_len",
    "hf_cache_resume_len",
    "hf_cache_diverged",
    "hf_cache_replayed_last_token",
    "hf_generation",
    "hf_generate_wall_ms",
    "hf_output_token_count",
    "hf_hit_max_new_tokens",
    "hf_tool_call_count",
    "hf_malformed_tool_output",
    "hf_finish_reason_inferred",
)


def filter_hf_trace_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only allow-listed local-HF trace telemetry fields."""
    if not extra:
        return {}
    return {key: extra[key] for key in HF_TRACE_EXTRA_KEYS if key in extra}
