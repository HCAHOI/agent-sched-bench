"""Tests for the within-segment attention mean/std grid (head_span) plotting.

Covers the pure aggregation reducer, decode-step masking, the CLI config knob
parsing/validation, and an end-to-end smoke that builds a real (tiny) head_span
attempt directory via LayerCapturer and renders the grid headless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import matplotlib

matplotlib.use("Agg")

from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer

from scripts.recoding_figures.plot_segment_head_span_grid import (
    _fill_matrices,
    _parse_layer_arg,
    _phase_observations,
    build_head_span_segment_grids,
    reduce_head_span_cell,
)
from trace_collect.cli import _parse_per_head_stats_layers, _run_collect, parse_collect_args


# ---------------------------------------------------------------------------
# reduce_head_span_cell
# ---------------------------------------------------------------------------


def test_reduce_happy_path() -> None:
    means = np.array([0.10, 0.20, 0.30])
    variances = np.array([0.01, 0.02, 0.03])
    cell = reduce_head_span_cell(means, variances)
    assert cell["mean"] == pytest.approx(0.2)
    assert cell["var_pooled"] == pytest.approx(0.02)
    assert cell["std"] == pytest.approx(np.sqrt(0.02))
    assert cell["cross_head_std"] == pytest.approx(np.std(means))
    assert cell["n_contributors"] == 3
    assert cell["n_nan_contributors"] == 0


def test_reduce_all_nan_segment() -> None:
    cell = reduce_head_span_cell(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert cell["mean"] is None
    assert cell["std"] is None
    assert cell["var_pooled"] is None
    assert cell["cross_head_std"] is None
    assert cell["n_contributors"] == 0
    assert cell["n_nan_contributors"] == 2


def test_reduce_partial_nan_drops_missing() -> None:
    means = np.array([np.nan, 0.4, 0.6])
    variances = np.array([np.nan, 1.0, 1.0])
    cell = reduce_head_span_cell(means, variances)
    assert cell["mean"] == pytest.approx(0.5)
    assert cell["std"] == pytest.approx(1.0)
    assert cell["n_contributors"] == 2
    assert cell["n_nan_contributors"] == 1


def test_reduce_sqrt_order_guard() -> None:
    """std must be sqrt(mean(var)), never mean(sqrt(var))."""
    cell = reduce_head_span_cell(np.array([0.1, 0.2]), np.array([0.0, 1.0]))
    assert cell["std"] == pytest.approx(np.sqrt(0.5))
    assert cell["std"] != pytest.approx(0.5)  # the wrong (sqrt-then-mean) answer


def test_reduce_single_contributor_cross_std_zero() -> None:
    cell = reduce_head_span_cell(np.array([np.nan, 0.3]), np.array([np.nan, 0.04]))
    assert cell["cross_head_std"] == 0.0
    assert cell["std"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _phase_observations decode masking
# ---------------------------------------------------------------------------


def _decode_sel(mean_d, var_d, step_d, kept_d) -> dict[str, np.ndarray]:
    """Minimal sel dict for the decode branch of _phase_observations."""
    return {
        "mean_d": np.asarray(mean_d, dtype=np.float64),
        "var_d": np.asarray(var_d, dtype=np.float64),
        "step_d": np.asarray(step_d, dtype=np.int64),
        "kept_d": np.asarray(kept_d, dtype=np.int64),
    }


def test_decode_pool_steps_masks_padding() -> None:
    # 1 layer, 2 decode steps (second is padding step=-1), 1 head, 1 segment.
    sel = _decode_sel(
        mean_d=[[[[0.5]], [[9.9]]]],  # [L=1, T=2, H=1, S=1]
        var_d=[[[[0.04]], [[9.9]]]],
        step_d=[[0, -1]],
        kept_d=[[[3], [0]]],  # [L=1, T=2, S=1]
    )
    means, variances, kept = _phase_observations(
        sel, segment_idx=0, phase="decode", decode_reduce="pool_steps"
    )
    finite = means[np.isfinite(means)]
    assert finite.tolist() == [0.5]  # padding step dropped
    assert kept == 3
    cell = reduce_head_span_cell(means, variances)
    assert cell["mean"] == pytest.approx(0.5)
    assert cell["n_contributors"] == 1


def test_decode_last_step_picks_latest_valid() -> None:
    # 1 layer, 3 steps with steps [0, 2, -1]; last valid is step index 1 (step value 2).
    sel = _decode_sel(
        mean_d=[[[[0.1]], [[0.7]], [[9.9]]]],
        var_d=[[[[0.01]], [[0.02]], [[9.9]]]],
        step_d=[[0, 2, -1]],
        kept_d=[[[3], [3], [0]]],
    )
    means, variances, kept = _phase_observations(
        sel, segment_idx=0, phase="decode", decode_reduce="last_step"
    )
    cell = reduce_head_span_cell(means, variances)
    assert cell["mean"] == pytest.approx(0.7)


def test_decode_last_step_ragged_across_layers() -> None:
    # 2 layers, 2 steps, 1 head, 1 segment. Layer 0 valid steps [0,1] (last=1),
    # layer 1 valid steps [0,-1] (last=0, second row padded). Each layer must
    # contribute exactly its own latest valid step.
    sel = _decode_sel(
        mean_d=[[[[0.1]], [[0.5]]], [[[0.9]], [[9.9]]]],  # [L=2, T=2, H=1, S=1]
        var_d=[[[[0.01]], [[0.01]]], [[[0.01]], [[9.9]]]],
        step_d=[[0, 1], [0, -1]],
        kept_d=[[[3], [3]], [[3], [0]]],
    )
    means, variances, kept = _phase_observations(
        sel, segment_idx=0, phase="decode", decode_reduce="last_step"
    )
    finite = np.sort(means[np.isfinite(means)])
    assert finite.tolist() == [0.5, 0.9]  # layer0 step1, layer1 step0
    cell = reduce_head_span_cell(means, variances)
    assert cell["mean"] == pytest.approx(0.7)
    assert cell["n_contributors"] == 2


# ---------------------------------------------------------------------------
# CLI config knob
# ---------------------------------------------------------------------------


def test_parse_per_head_stats_layers_presets_and_lists() -> None:
    assert _parse_per_head_stats_layers("qwen3-coder-30b") == (0, 6, 12, 18, 24, 30, 36, 47)
    assert _parse_per_head_stats_layers("default") == (0, 6, 12, 18, 24, 30, 36, 47)
    assert _parse_per_head_stats_layers(None) == ()
    assert _parse_per_head_stats_layers("") == ()
    assert _parse_per_head_stats_layers("12, 0, 6, 6") == (0, 6, 12)
    with pytest.raises(ValueError):
        _parse_per_head_stats_layers("-1,2")
    with pytest.raises(ValueError):
        _parse_per_head_stats_layers("abc")


def test_per_head_flag_requires_record_internals() -> None:
    args = parse_collect_args(
        [
            "--provider", "dashscope",
            "--model", "m",
            "--scaffold", "openclaw",
            "--mcp-config", "none",
            "--per-head-stats-layers", "0,6",
        ]
    )
    assert args.per_head_stats_layers == "0,6"
    with pytest.raises(SystemExit) as excinfo:
        _run_collect(args)
    assert excinfo.value.code == 2


def test_recording_config_empty_tuple_is_default() -> None:
    assert RecordingConfig(per_head_stats_layers=()) == RecordingConfig()


def test_parse_layer_arg() -> None:
    assert _parse_layer_arg(None) is None
    assert _parse_layer_arg("") is None
    assert _parse_layer_arg("0,24,47") == [0, 24, 47]


# ---------------------------------------------------------------------------
# matrix scatter (guards against segment/iter transpose or mis-indexing)
# ---------------------------------------------------------------------------


def test_fill_matrices_places_values_in_expected_cells() -> None:
    segment_to_row = {"segA": 0, "segB": 1}
    iter_to_col = {3: 0, 4: 1, 5: 2}  # observed_call_idx -> column
    rows = [
        {"segment_id": "segA", "observed_call_idx": 5, "phase": "prefill",
         "within_segment_attention_mean": 0.11, "within_segment_attention_std": 0.02},
        {"segment_id": "segB", "observed_call_idx": 3, "phase": "decode",
         "within_segment_attention_mean": 0.33, "within_segment_attention_std": 0.04},
    ]
    means = _fill_matrices(
        rows, segment_to_row=segment_to_row, iter_to_col=iter_to_col,
        phases=("prefill", "decode"), key="within_segment_attention_mean",
    )
    # segA prefill at (row 0, col 2); rest of prefill NaN
    assert means["prefill"][0, 2] == pytest.approx(0.11)
    assert np.isnan(means["prefill"][1, 0])
    # segB decode at (row 1, col 0)
    assert means["decode"][1, 0] == pytest.approx(0.33)
    assert np.isnan(means["decode"][0, 2])
    stds = _fill_matrices(
        rows, segment_to_row=segment_to_row, iter_to_col=iter_to_col,
        phases=("prefill", "decode"), key="within_segment_attention_std",
    )
    assert stds["prefill"][0, 2] == pytest.approx(0.02)
    assert stds["decode"][1, 0] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# End-to-end smoke (real head_span via LayerCapturer)
# ---------------------------------------------------------------------------


class _ToyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 4
        self.num_key_value_groups = 1
        self.scaling = 0.5
        self.q_proj = torch.nn.Linear(8, 4, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        del position_embeddings, attention_mask, past_key_values
        return hidden_states, None


class _FakeCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.key_states = key_states

    def __getitem__(self, layer_idx: int):
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return self.key_states, None


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()
        self.model.layers[0].mlp = torch.nn.Module()
        self.model.layers[0].mlp.gate = torch.nn.Linear(8, 3, bias=False)


_SEGMENTS = [
    {
        "role": "system",
        "message_index": 0,
        "token_start": 0,
        "token_end": 2,
        "has_content": True,
        "has_tool_calls": False,
        "first_seen_call": 0,
    },
    {
        "role": "user",
        "message_index": 1,
        "token_start": 2,
        "token_end": 4,
        "has_content": True,
        "has_tool_calls": False,
        "first_seen_call": 0,
    },
]


def _make_attempt(tmp_path: Path, *, per_head_stats_layers: tuple[int, ...]) -> Path:
    """Record one toy iter and write the loader-required sidecar files."""
    attempt_dir = tmp_path / "0b01001001__spectree-64" / "attempt_1"
    recordings_dir = attempt_dir / "recordings"
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
            per_head_stats_layers=per_head_stats_layers,
        ),
        model_summary={"name": "toy"},
    )
    capturer.start_attempt(recordings_dir)
    with capturer.recording_session(call_idx=0, segments=_SEGMENTS, input_token_count=4):
        attn = model.model.layers[0].self_attn
        attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(torch.ones(1, 4, 4), torch.zeros(1, 4, 4)),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(torch.ones(1, 1, 4), torch.zeros(1, 1, 4)),
            past_key_values=_FakeCache(torch.zeros(1, 1, 5, 4)),
        )
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    iter_dir = recordings_dir / "iter_0000"
    (iter_dir / "segments.json").write_text(
        json.dumps(
            {
                "call_idx": 0,
                "input_tokens": 4,
                "output_tokens": 1,
                "total_tokens": 5,
                "complete": True,
                "segments": _SEGMENTS,
            }
        ),
        encoding="utf-8",
    )
    if not (iter_dir / "routing.npz").is_file():
        np.savez(iter_dir / "routing.npz", placeholder=np.zeros(1, dtype=np.int32))
    (iter_dir / ".done").write_text("", encoding="utf-8")
    return attempt_dir


def test_build_head_span_grid_end_to_end(tmp_path: Path) -> None:
    attempt_dir = _make_attempt(tmp_path, per_head_stats_layers=(0,))
    out_dir = tmp_path / "out"
    summary = build_head_span_segment_grids(
        inputs=[attempt_dir],
        output_dir=out_dir,
        split_by_task=True,
    )
    assert summary["layers_used"] == [0]
    assert summary["metric"] == "within_segment_attention_mean_std"
    assert len(summary["groups"]) == 1
    group = summary["groups"][0]
    assert group["n_segments"] == 2
    grid_png = out_dir / group["plot"]["grid_png"]
    assert grid_png.is_file()
    # trajectory CSV carries the mean/std columns for both phases
    traj = (out_dir / group["output_dir"] / "segment_head_span_trajectory.csv").read_text()
    assert "within_segment_attention_mean" in traj
    assert "within_segment_attention_std" in traj


def test_build_head_span_grid_raises_without_stats(tmp_path: Path) -> None:
    attempt_dir = _make_attempt(tmp_path, per_head_stats_layers=())
    with pytest.raises(ValueError, match="no per-head span stats"):
        build_head_span_segment_grids(
            inputs=[attempt_dir],
            output_dir=tmp_path / "out",
            split_by_task=True,
        )


def test_out_of_range_layer_raises_at_record_time(tmp_path: Path) -> None:
    """A layer index >= num_hidden_layers must fail loud at capturer build."""
    with pytest.raises(ValueError, match="num_hidden_layers"):
        LayerCapturer(
            _ToyModel(),  # toy model exposes a single attention layer (index 0)
            config=RecordingConfig(per_head_stats_layers=(5,)),
            model_summary={"name": "toy"},
        )
