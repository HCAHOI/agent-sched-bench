#!/usr/bin/env python
"""Measure the real KV-cache swap cost (rho) on GPU hardware.

Background
----------
When a tool call runs long, the scheduler swaps a request's KV cache off the
GPU (swap-out, GPU->host, off the critical path) and later swaps it back
(swap-in, host->GPU, ON the critical path when the call returns). The restore
cost campaign parameterizes

    rho = swap_in_cost / swap_out_cost

and sweeps it as a fraction in {0, 0.25, 0.5, 1.0}. This script MEASURES the
real value: vLLM's block swap is a ``cudaMemcpyAsync`` of KV blocks between GPU
and pinned host memory, so the faithful measurement is to derive the realistic
per-token KV byte layout from a real model config, then time D2H (swap-out) and
H2D (swap-in) transfers of KV-block-sized pinned buffers with CUDA events.

This is a standalone measurement utility. It does not touch the trace_collect
pipeline. Heavy deps (torch, transformers, vllm) are imported lazily so that
``--help`` and ``--model-config-only`` work without a GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from dataclasses import dataclass

# Bytes per scalar for the dtypes vLLM uses for KV cache.
_DTYPE_SIZE_BYTES: dict[str, int] = {
    "float32": 4,
    "float": 4,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
}

# Campaign cost grid: swap-out cost operating points, in milliseconds.
DEFAULT_COST_GRID_MS: tuple[int, ...] = tuple(range(500, 5001, 500))

# Default per-size token sweep (geometric).
DEFAULT_TOKEN_SWEEP: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
)


def dtype_size_bytes(dtype: str) -> int:
    """Return the byte width of a KV-cache scalar dtype.

    Raises:
        ValueError: If the dtype name is not recognized.
    """
    key = dtype.lower().replace("torch.", "").strip()
    if key not in _DTYPE_SIZE_BYTES:
        raise ValueError(
            f"Unknown dtype {dtype!r}; known: {sorted(_DTYPE_SIZE_BYTES)}"
        )
    return _DTYPE_SIZE_BYTES[key]


@dataclass
class KVLayout:
    """Per-token KV-cache byte layout derived from a model config."""

    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: str
    dtype_size: int
    bytes_per_token: int
    # Extra fields kept for the JSON record / auditability.
    num_attention_heads: int
    hidden_size: int | None = None


def bytes_per_token(
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype_size: int,
) -> int:
    """KV-cache bytes stored per token.

    Factor of 2 accounts for both the K and the V cache. GQA is handled by
    passing ``num_key_value_heads`` (which may be < the number of query heads).
    """
    if min(num_hidden_layers, num_key_value_heads, head_dim, dtype_size) <= 0:
        raise ValueError(
            "All KV-layout dimensions must be positive: "
            f"layers={num_hidden_layers}, kv_heads={num_key_value_heads}, "
            f"head_dim={head_dim}, dtype_size={dtype_size}"
        )
    return 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_size


def derive_kv_layout(config: dict, dtype: str) -> KVLayout:
    """Derive the per-token KV byte layout from an HF-style config dict.

    Handles GQA (``num_key_value_heads`` may differ from
    ``num_attention_heads``; defaults to MHA when absent) and derives
    ``head_dim`` from ``hidden_size / num_attention_heads`` when not explicit.

    Raises:
        ValueError: If required config fields are missing (fail fast).
    """
    num_hidden_layers = config.get("num_hidden_layers")
    num_attention_heads = config.get("num_attention_heads")
    if num_hidden_layers is None:
        raise ValueError("config missing required field 'num_hidden_layers'")
    if num_attention_heads is None:
        raise ValueError("config missing required field 'num_attention_heads'")

    # GQA: kv heads default to attention heads (MHA) when unspecified.
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)

    head_dim = config.get("head_dim")
    hidden_size = config.get("hidden_size")
    if head_dim is None:
        if hidden_size is None:
            raise ValueError(
                "config has neither 'head_dim' nor 'hidden_size' to derive it"
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by "
                f"num_attention_heads {num_attention_heads}; cannot derive head_dim"
            )
        head_dim = hidden_size // num_attention_heads

    dsize = dtype_size_bytes(dtype)
    bpt = bytes_per_token(num_hidden_layers, num_key_value_heads, head_dim, dsize)
    return KVLayout(
        num_hidden_layers=int(num_hidden_layers),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        dtype=dtype,
        dtype_size=dsize,
        bytes_per_token=int(bpt),
        num_attention_heads=int(num_attention_heads),
        hidden_size=int(hidden_size) if hidden_size is not None else None,
    )


def interpolate_cost_grid(
    tokens: list[int],
    swap_out_ms: list[float],
    swap_in_ms: list[float],
    target_ms_grid: tuple[int, ...],
) -> list[dict]:
    """Map measured points onto the campaign's swap-out cost grid.

    Given measured ``(tokens, swap_out_ms, swap_in_ms)`` rows, interpolate the
    token count and rho at each target swap-out cost. Targets outside the
    measured swap-out range are reported with null values (no extrapolation).

    Returns:
        One dict per target ms with keys: target_swap_out_ms, tokens,
        swap_in_ms, rho, in_range.
    """
    import numpy as np

    if len(tokens) != len(swap_out_ms) or len(tokens) != len(swap_in_ms):
        raise ValueError("tokens/swap_out_ms/swap_in_ms length mismatch")
    if len(tokens) < 2:
        raise ValueError("need at least 2 measured points to interpolate")

    # Sort by swap-out cost so it forms a monotonic x-axis for interpolation.
    order = sorted(range(len(swap_out_ms)), key=lambda i: swap_out_ms[i])
    xs = np.array([swap_out_ms[i] for i in order], dtype=float)
    toks = np.array([tokens[i] for i in order], dtype=float)
    ins = np.array([swap_in_ms[i] for i in order], dtype=float)

    lo, hi = float(xs.min()), float(xs.max())
    rows: list[dict] = []
    for target in target_ms_grid:
        in_range = lo <= target <= hi
        if in_range:
            tok = float(np.interp(target, xs, toks))
            sin = float(np.interp(target, xs, ins))
            rho = sin / target if target > 0 else None
        else:
            tok = sin = rho = None
        rows.append(
            {
                "target_swap_out_ms": int(target),
                "tokens": tok,
                "swap_in_ms": sin,
                "rho": rho,
                "in_range": in_range,
            }
        )
    return rows


def query_pcie_link() -> dict:
    """Record PCIe link width/gen via nvidia-smi. Returns nulls if unavailable.

    The rho denominator (swap-out bandwidth) depends on the PCIe link, so we
    record it alongside the measurement for reproducibility.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"pcie_link_width": None, "pcie_link_gen": None}
    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    width = parts[0] if len(parts) > 0 and parts[0] else None
    gen = parts[1] if len(parts) > 1 and parts[1] else None
    return {"pcie_link_width": width, "pcie_link_gen": gen}


@dataclass
class SizeResult:
    tokens: int
    bytes: int
    swap_out_ms: float
    swap_in_ms: float
    swap_out_GBps: float
    swap_in_GBps: float
    rho: float


def measure_transfers(
    layout: KVLayout,
    token_sweep: list[int],
    repeats: int,
    pinned: bool,
    warmup: int = 3,
) -> list[SizeResult]:
    """Time D2H (swap-out) and H2D (swap-in) transfers with CUDA events.

    For each token count, allocate a GPU buffer and a matching host buffer of
    ``tokens * bytes_per_token`` bytes (uint8 -- we measure bytes moved) and
    time ``repeats`` reps of each direction separately.
    """
    import torch

    results: list[SizeResult] = []
    for tokens in token_sweep:
        nbytes = tokens * layout.bytes_per_token
        gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        host = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)

        for _ in range(warmup):
            host.copy_(gpu, non_blocking=True)
            torch.cuda.synchronize()
            gpu.copy_(host, non_blocking=True)
            torch.cuda.synchronize()

        out_ms: list[float] = []
        in_ms: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            torch.cuda.synchronize()
            start.record()
            host.copy_(gpu, non_blocking=True)  # swap-out (D2H)
            end.record()
            torch.cuda.synchronize()
            out_ms.append(start.elapsed_time(end))

            torch.cuda.synchronize()
            start.record()
            gpu.copy_(host, non_blocking=True)  # swap-in (H2D)
            end.record()
            torch.cuda.synchronize()
            in_ms.append(start.elapsed_time(end))

        del gpu, host
        torch.cuda.empty_cache()

        so = statistics.median(out_ms)
        si = statistics.median(in_ms)
        results.append(
            SizeResult(
                tokens=tokens,
                bytes=nbytes,
                swap_out_ms=so,
                swap_in_ms=si,
                swap_out_GBps=nbytes / (so / 1000.0) / 1e9,
                swap_in_GBps=nbytes / (si / 1000.0) / 1e9,
                rho=si / so,
            )
        )
    return results


def load_model_config(model: str) -> dict:
    """Load an HF model config as a plain dict. Uses transformers (CPU-safe)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return cfg.to_dict()


def _device_info() -> dict:
    info = {"host": platform.node(), "device": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["torch_version"] = torch.__version__
    except Exception:
        pass
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure real KV-cache swap cost (rho) on GPU hardware.",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-32B",
        help="HF model id used to derive the per-token KV byte layout.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        help="KV-cache dtype (drives bytes-per-scalar).",
    )
    p.add_argument(
        "--kv-token-sweep",
        default=",".join(str(t) for t in DEFAULT_TOKEN_SWEEP),
        help="Comma-separated KV sizes in TOKENS to transfer.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed reps per size after warmup.",
    )
    p.add_argument(
        "--pinned",
        dest="pinned",
        action="store_true",
        default=True,
        help="Use pinned host memory (default; matches vLLM).",
    )
    p.add_argument(
        "--no-pinned",
        dest="pinned",
        action="store_false",
        help="Use pageable host memory instead of pinned.",
    )
    p.add_argument(
        "--cost-grid-ms",
        default=",".join(str(m) for m in DEFAULT_COST_GRID_MS),
        help="Campaign swap-out cost operating points, in ms.",
    )
    p.add_argument("--output", default=None, help="Path to write full JSON.")
    p.add_argument(
        "--model-config-only",
        action="store_true",
        help="Derive + print byte layout and exit (CPU-safe smoke path).",
    )
    return p


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token_sweep = _parse_int_list(args.kv_token_sweep)
    cost_grid = tuple(_parse_int_list(args.cost_grid_ms))

    config = load_model_config(args.model)
    layout = derive_kv_layout(config, args.dtype)

    print(f"model: {args.model}")
    print(f"dtype: {args.dtype} ({layout.dtype_size} bytes/scalar)")
    print(
        f"layers={layout.num_hidden_layers} "
        f"kv_heads={layout.num_key_value_heads} "
        f"(attn_heads={layout.num_attention_heads}) head_dim={layout.head_dim}"
    )
    print(f"bytes_per_token = {layout.bytes_per_token} "
          f"({layout.bytes_per_token / 1024:.1f} KiB/token)")

    record: dict = {
        "model": args.model,
        "dtype": args.dtype,
        "bytes_per_token": layout.bytes_per_token,
        "kv_layout": layout.__dict__,
        "token_sweep": token_sweep,
        "pinned": args.pinned,
        "repeats": args.repeats,
        "device_info": _device_info(),
        "pcie": query_pcie_link(),
        "note": (
            "rho = swap_in_ms / swap_out_ms measured by timing pinned-buffer "
            "D2H/H2D copies of KV-block-sized transfers with CUDA events."
        ),
    }

    if args.model_config_only:
        record["measurements"] = None
        if args.output:
            with open(args.output, "w") as f:
                json.dump(record, f, indent=2)
            print(f"wrote {args.output}")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available; run on the GPU host, or use "
            "--model-config-only for a CPU smoke test."
        )

    results = measure_transfers(layout, token_sweep, args.repeats, args.pinned)
    record["measurements"] = [r.__dict__ for r in results]
    record["cost_grid"] = interpolate_cost_grid(
        [r.tokens for r in results],
        [r.swap_out_ms for r in results],
        [r.swap_in_ms for r in results],
        cost_grid,
    )

    print()
    print(
        f"{'tokens':>8} {'MiB':>9} {'out_ms':>9} {'in_ms':>9} "
        f"{'out_GBps':>9} {'in_GBps':>9} {'rho':>6}"
    )
    for r in results:
        print(
            f"{r.tokens:>8} {r.bytes / 2**20:>9.1f} {r.swap_out_ms:>9.3f} "
            f"{r.swap_in_ms:>9.3f} {r.swap_out_GBps:>9.2f} "
            f"{r.swap_in_GBps:>9.2f} {r.rho:>6.3f}"
        )

    print("\ncost-grid operating points (swap-out ms -> tokens, rho):")
    for row in record["cost_grid"]:
        if row["in_range"]:
            print(
                f"  {row['target_swap_out_ms']:>5} ms -> "
                f"{row['tokens']:.0f} tokens, rho={row['rho']:.3f}"
            )
        else:
            print(f"  {row['target_swap_out_ms']:>5} ms -> out of measured range")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
