"""Measure the context-dependent prefill cost used by the W5 scheduler.

Runs prefix-cache-free vLLM requests over a context-length grid and writes the
quadratic fit consumed by ``spike.multitenant.load_prefill_cost_profile``.
Heavy GPU dependencies are imported lazily so ``--help`` and
``--model-config-only`` work on non-GPU hosts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Reuse the KV-layout helpers from the rho script so the prefill curve and the
# swap curve are reported against one KV byte layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.serving.measure_kv_swap_cost import (  # noqa: E402
    _device_info,
    derive_kv_layout,
    load_model_config,
)
from spike.multitenant import PREFILL_COST_SCHEMA_VERSION  # noqa: E402

# A near-zero-context baseline (8) measures the wall-clock overhead floor that
# the CUDA-event reload number does not carry; the rest span the deployment
# context range for the marginal-slope fit.
DEFAULT_CONTEXT_SWEEP = (8, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _parse_int_list(text: str, *, label: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{label} is empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} has duplicates")
    return sorted(values)


def _optional_string(value: str) -> str | None:
    return None if value.lower() == "none" else value


def measure_prefill(
    model: str,
    *,
    context_lengths: list[int],
    repeats: int,
    warmup: int,
    max_model_len: int,
    quantization: str | None,
    kv_cache_dtype: str,
    seed: int,
) -> dict[str, Any]:
    """Time vLLM prefill (TTFT) per context length; return per-length stats."""
    import numpy as np
    import torch  # noqa: F401  (ensures CUDA is present before the heavy load)
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available for prefill measurement")
    if len(context_lengths) < 3:
        raise ValueError(
            "quadratic prefill fit requires at least three context lengths"
        )

    max_ctx = max(context_lengths)
    if max_ctx + 1 > max_model_len:  # +1 for the single generated token
        raise ValueError(
            f"max context {max_ctx} (+1 gen token) must be <= max_model_len "
            f"{max_model_len}"
        )

    llm = LLM(
        model=model,
        dtype="auto",
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=False,  # force a real prefill every request
        enable_chunked_prefill=True,  # allow long single-request prefills
        max_num_batched_tokens=max_ctx,  # admit the largest sweep prefill
        enforce_eager=True,  # steady, comparable timings (no cudagraph capture)
        gpu_memory_utilization=0.9,
        seed=seed,
    )
    vocab = llm.get_tokenizer().vocab_size
    rng = np.random.default_rng(seed)
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    # Distinct token ids per (length, rep) so prefix caching can never elide a
    # prefill. ids drawn from [256, vocab) to skip byte-fallback ids 0..255;
    # a stray special/EOS token in the PROMPT does not halt prefill and output
    # is bounded by max_tokens=1 regardless, so timing is unaffected.
    def fresh_prompt(length: int) -> dict:
        ids = rng.integers(256, vocab, size=length, dtype=np.int64).tolist()
        return {"prompt_token_ids": ids}

    def time_once(length: int) -> float:
        prompt = fresh_prompt(length)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompt, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3  # ms

    points: list[dict[str, Any]] = []
    for length in context_lengths:
        for _ in range(warmup):
            time_once(length)
        samples = [time_once(length) for _ in range(repeats)]
        samples.sort()
        median = float(np.median(samples))
        points.append(
            {
                "context_tokens": length,
                "prefill_ms_median": median,
                "prefill_ms_min": float(samples[0]),
                "prefill_ms_max": float(samples[-1]),
                "ms_per_token": median / length,
                "repeats": repeats,
            }
        )
        print(
            f"  ctx={length:>6}  prefill={median:8.2f} ms  "
            f"(secant {median / length:.5f} ms/tok)",
            flush=True,
        )

    # Marginal-rate model for the linear-through-origin consumer: fit
    # prefill_ms ~= intercept + slope*context over the medians. The slope is the
    # per-token rate to feed P2; the intercept is the fixed overhead floor
    # (~= the near-zero-context baseline) that the CUDA-event reload side omits.
    ctx = np.array([p["context_tokens"] for p in points], dtype=float)
    pref = np.array([p["prefill_ms_median"] for p in points], dtype=float)
    slope, intercept = np.polyfit(ctx, pref, 1)
    residuals = pref - (slope * ctx + intercept)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((pref - pref.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    qa, qb, qc = np.polyfit(ctx, pref, 2)
    quadratic_residuals = pref - (qa * ctx**2 + qb * ctx + qc)
    quadratic_ss_res = float(np.sum(quadratic_residuals**2))
    quadratic_r_squared = 1.0 - quadratic_ss_res / ss_tot if ss_tot > 0 else None
    overhead_floor_ms = float(min(pref))  # prefill at the smallest context
    return {
        "points": points,
        "vocab_size": int(vocab),
        "linear_fit": {
            "slope_ms_per_token": float(slope),
            "intercept_ms": float(intercept),
            "r_squared": r_squared,
            "note": (
                "adopt slope_ms_per_token as the P2 recompute rate; prefill is "
                "super-linear so a linear rate under-charges long context"
            ),
        },
        "quadratic_fit": {
            "coefficients": [float(qa), float(qb), float(qc)],
            "coefficient_order": ["context_tokens^2", "context_tokens", "intercept"],
            "r_squared": quadratic_r_squared,
            "note": "Continuum online prefill-reload estimator",
        },
        "overhead_floor_ms": overhead_floor_ms,
        "min_context_tokens": int(min(ctx)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    p.add_argument(
        "--kv-cache-dtype",
        default="fp8",
        help="vLLM kv_cache_dtype (match the rho measurement, e.g. fp8/auto).",
    )
    p.add_argument(
        "--kv-layout-dtype",
        default="float8_e4m3fn",
        help="Dtype name for the reported per-token KV byte layout "
        "(e.g. float8_e4m3fn, bfloat16).",
    )
    p.add_argument(
        "--quantization",
        type=_optional_string,
        default="fp8",
        help="vLLM quantization mode, or 'none' for an unquantized model.",
    )
    p.add_argument(
        "--context-sweep",
        default=",".join(str(c) for c in DEFAULT_CONTEXT_SWEEP),
        help="Comma-separated context lengths (tokens) to prefill.",
    )
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model-config-only",
        action="store_true",
        help="Print the KV layout and exit (no GPU needed).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    context_lengths = _parse_int_list(args.context_sweep, label="context-sweep")

    config = load_model_config(args.model)
    kv_layout = derive_kv_layout(config, args.kv_layout_dtype)
    layout_dict = {
        "num_hidden_layers": kv_layout.num_hidden_layers,
        "num_key_value_heads": kv_layout.num_key_value_heads,
        "head_dim": kv_layout.head_dim,
        "dtype": kv_layout.dtype,
        "dtype_size": kv_layout.dtype_size,
        "bytes_per_token": kv_layout.bytes_per_token,
    }
    if args.model_config_only:
        print(json.dumps({"model": args.model, "kv_layout": layout_dict}, indent=2))
        return

    print(f"Measuring prefill cost for {args.model} ...", flush=True)
    result = measure_prefill(
        args.model,
        context_lengths=context_lengths,
        repeats=args.repeats,
        warmup=args.warmup,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        seed=args.seed,
    )

    payload = {
        "schema_version": PREFILL_COST_SCHEMA_VERSION,
        "measurement": "prefill_recompute_cost",
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "quantization": args.quantization,
        "max_model_len": args.max_model_len,
        "kv_layout": layout_dict,
        "device": _device_info(),
        "points": result["points"],
        "linear_fit": result["linear_fit"],
        "quadratic_fit": result["quadratic_fit"],
        "overhead_floor_ms": result["overhead_floor_ms"],
        "min_context_tokens": result["min_context_tokens"],
        "vocab_size": result["vocab_size"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
