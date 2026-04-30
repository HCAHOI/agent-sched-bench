from __future__ import annotations

import logging
import re
from pathlib import Path

from harness.scheduler_hooks import GpuBaseline

# Tolerant patterns covering vLLM 0.5–0.7 stderr formats.
_WEIGHTS_PATTERNS = [
    re.compile(r"(?i)model\s+weights\s+(?:take|took)\s+([\d.]+)\s*Gi?B"),
    re.compile(r"(?i)Loading\s+model\s+weights\s+took\s+([\d.]+)\s*Gi?B"),
]
# vLLM 0.5+ logs "GPU KV cache size: <tokens> tokens, <X> GiB"; we extract X
# directly. The token count alone is not convertible to MiB without
# per-token-bytes (which we don't always have), so requiring the GiB suffix
# is the no-fabrication path.
_KV_GIB_PATTERN = re.compile(r"(?i)GPU\s+KV\s+cache\s+size:[^,]*,\s*([\d.]+)\s*Gi?B")
_DTYPE_PATTERN = re.compile(r"(?i)(?:dtype|torch_dtype)\s*[:=]\s*['\"]?(?:torch\.)?(\w+)")
_MODEL_PATTERN = re.compile(r"(?i)model\s*[:=]\s*['\"]?([^\s'\",]+)")
_TP_PATTERN = re.compile(r"(?i)tensor[_-]parallel[_-]size\s*[:=]\s*(\d+)")


def parse_startup_log(text: str) -> GpuBaseline | None:
    """Parse vLLM startup stderr and extract baseline GPU memory facts.

    Returns None when either the weights or KV-cache GiB line is absent
    (research-integrity: never fabricate a baseline). Optional fields
    (model, dtype, tp) fall back to sensible defaults when absent.
    """
    weights_gib = _first_match(_WEIGHTS_PATTERNS, text)
    if weights_gib is None:
        logging.warning("vllm_startup_parser: model-weights line not found in log")
        return None
    kv_gib_match = _KV_GIB_PATTERN.search(text)
    if kv_gib_match is None:
        logging.warning("vllm_startup_parser: GPU KV cache GiB line not found in log")
        return None
    return GpuBaseline(
        weights_mib=float(weights_gib) * 1024.0,
        kv_cache_total_mib=float(kv_gib_match.group(1)) * 1024.0,
        model=_first_match([_MODEL_PATTERN], text) or "unknown",
        dtype=_first_match([_DTYPE_PATTERN], text) or "unknown",
        tensor_parallel_size=int(_first_match([_TP_PATTERN], text) or "1"),
    )


def parse_startup_log_file(path: Path) -> GpuBaseline | None:
    return parse_startup_log(Path(path).read_text(encoding="utf-8", errors="replace"))


def _first_match(patterns: list[re.Pattern], text: str) -> str | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1)
    return None
