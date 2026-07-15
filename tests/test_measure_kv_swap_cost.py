"""CPU-safe unit tests for the pure functions in measure_kv_swap_cost.

These cover the byte-layout math, GQA / head_dim derivation, dtype mapping,
and the cost-grid interpolation helper. None of them touch torch/CUDA.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_kv_swap_cost",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_kv_swap_cost.py",
)
mks = importlib.util.module_from_spec(_SPEC)
# Register before exec so @dataclass can resolve the module via sys.modules.
sys.modules[_SPEC.name] = mks
_SPEC.loader.exec_module(mks)


def test_dtype_size_bytes():
    assert mks.dtype_size_bytes("bfloat16") == 2
    assert mks.dtype_size_bytes("float16") == 2
    assert mks.dtype_size_bytes("float32") == 4
    assert mks.dtype_size_bytes("fp8") == 1
    assert mks.dtype_size_bytes("torch.bfloat16") == 2
    with pytest.raises(ValueError):
        mks.dtype_size_bytes("float13")


def test_bytes_per_token_hand_computed():
    # layers=64, kv_heads=8, head_dim=128, bf16 -> 2*64*8*128*2 = 262144
    assert mks.bytes_per_token(64, 8, 128, 2) == 262144


def test_bytes_per_token_rejects_nonpositive():
    with pytest.raises(ValueError):
        mks.bytes_per_token(64, 0, 128, 2)


def test_derive_kv_layout_gqa_explicit_head_dim():
    # GQA: kv_heads (8) differs from attention heads (64).
    config = {
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    layout = mks.derive_kv_layout(config, "bfloat16")
    assert layout.num_key_value_heads == 8
    assert layout.head_dim == 128
    assert layout.bytes_per_token == 262144


def test_derive_kv_layout_derives_head_dim_from_hidden_size():
    # No head_dim: derive as hidden_size / num_attention_heads = 8192/64 = 128.
    config = {
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
    }
    layout = mks.derive_kv_layout(config, "bfloat16")
    assert layout.head_dim == 128
    assert layout.bytes_per_token == 262144


def test_derive_kv_layout_defaults_to_mha():
    # No num_key_value_heads -> MHA, kv_heads == attention_heads.
    config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "head_dim": 128,
    }
    layout = mks.derive_kv_layout(config, "float16")
    assert layout.num_key_value_heads == 32
    assert layout.bytes_per_token == 2 * 32 * 32 * 128 * 2


def test_derive_kv_layout_fails_fast_on_missing_fields():
    with pytest.raises(ValueError):
        mks.derive_kv_layout({"num_attention_heads": 32}, "bfloat16")
    with pytest.raises(ValueError):
        mks.derive_kv_layout({"num_hidden_layers": 32}, "bfloat16")
    # Neither head_dim nor hidden_size present.
    with pytest.raises(ValueError):
        mks.derive_kv_layout(
            {"num_hidden_layers": 32, "num_attention_heads": 32}, "bfloat16"
        )


def test_derive_kv_layout_rejects_indivisible_hidden_size():
    with pytest.raises(ValueError):
        mks.derive_kv_layout(
            {"num_hidden_layers": 32, "num_attention_heads": 7, "hidden_size": 100},
            "bfloat16",
        )


def test_interpolate_cost_grid_basic():
    # Synthetic linear points: swap_out_ms = tokens / 1000, swap_in = 0.5*out.
    tokens = [1000, 2000, 4000, 6000]
    swap_out_ms = [1000.0, 2000.0, 4000.0, 6000.0]
    swap_in_ms = [500.0, 1000.0, 2000.0, 3000.0]
    rows = mks.interpolate_cost_grid(
        tokens, swap_out_ms, swap_in_ms, (500, 3000, 10000)
    )
    by_target = {r["target_swap_out_ms"]: r for r in rows}

    # 500 ms is below the measured minimum (1000 ms) -> out of range.
    assert by_target[500]["in_range"] is False
    assert by_target[500]["tokens"] is None

    # 3000 ms interpolates linearly between 2000 and 4000 tokens -> 3000 tokens.
    mid = by_target[3000]
    assert mid["in_range"] is True
    assert mid["tokens"] == pytest.approx(3000.0)
    # rho = swap_in_ms / target; swap_in at 3000ms interpolates to 1500 -> 0.5.
    assert mid["rho"] == pytest.approx(0.5)

    # 10000 ms is above measured max -> out of range.
    assert by_target[10000]["in_range"] is False


def test_interpolate_cost_grid_requires_two_points():
    with pytest.raises(ValueError):
        mks.interpolate_cost_grid([1000], [1.0], [0.5], (500,))
