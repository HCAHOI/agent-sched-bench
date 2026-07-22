from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_collect.kv_profile_sweep import evaluate_profile_sweep, load_profile, load_tool_gaps
from scripts.exploration.sweep_kv_profile import build_parser, run


_PROFILE_ROW = {
    "engine": "fixture-engine",
    "mechanism": "swap",
    "device": "fixture-gpu",
    "kv_size_mb": 128.0,
    "direction": "gpu_to_cpu",
    "memory_path": "pinned_host",
    "concurrency": "1",
    "samples": 7,
    "p50_ms": 10.0,
    "p90_ms": 12.0,
    "p95_ms": 15.0,
    "p99_ms": 20.0,
}

_TOOL_GAP_ROWS = [
    {"sample_id": "short", "available_gap_ms": 8.0, "tool_names": ["bash"]},
    {"sample_id": "exact", "available_gap_ms": 15.0, "tool_names": ["read"]},
    {"sample_id": "wide", "available_gap_ms": 25.0, "tool_names": ["edit"]},
]


def test_evaluate_profile_sweep_counts_safe_gaps_and_exposed_ms(tmp_path: Path) -> None:
    profile_path = _write_jsonl(tmp_path / "kv_profile.jsonl", [_PROFILE_ROW])
    gaps_path = _write_jsonl(tmp_path / "tool_gaps.jsonl", _TOOL_GAP_ROWS)

    results = evaluate_profile_sweep(
        load_profile(profile_path),
        load_tool_gaps(gaps_path),
        quantile="p95",
        guard_ms_values=[0.0, 5.0],
    )

    assert results == [
        {
            "profile": {
                "engine": "fixture-engine",
                "mechanism": "swap",
                "device": "fixture-gpu",
                "kv_size_mb": 128.0,
                "direction": "gpu_to_cpu",
                "memory_path": "pinned_host",
                "concurrency": "1",
                "samples": 7,
            },
            "quantile": "p95",
            "kv_cost_ms": 15.0,
            "guard_ms": 0.0,
            "threshold_ms": 15.0,
            "total_count": 3,
            "safe_count": 2,
            "unsafe_count": 1,
            "safe_rate": 2 / 3,
            "min_safe_slack_ms": 0.0,
            "max_unsafe_deficit_ms": 7.0,
            "absorbed_if_oracle_ms": 30.0,
            "exposed_if_always_swap_ms": 7.0,
        },
        {
            "profile": {
                "engine": "fixture-engine",
                "mechanism": "swap",
                "device": "fixture-gpu",
                "kv_size_mb": 128.0,
                "direction": "gpu_to_cpu",
                "memory_path": "pinned_host",
                "concurrency": "1",
                "samples": 7,
            },
            "quantile": "p95",
            "kv_cost_ms": 15.0,
            "guard_ms": 5.0,
            "threshold_ms": 20.0,
            "total_count": 3,
            "safe_count": 1,
            "unsafe_count": 2,
            "safe_rate": 1 / 3,
            "min_safe_slack_ms": 5.0,
            "max_unsafe_deficit_ms": 12.0,
            "absorbed_if_oracle_ms": 15.0,
            "exposed_if_always_swap_ms": 7.0,
        },
    ]


@pytest.mark.parametrize(
    ("bad_row", "message"),
    [
        ({**_PROFILE_ROW, "p90_ms": 18.0}, "profile quantiles must be monotonic"),
        ({key: value for key, value in _PROFILE_ROW.items() if key != "engine"}, "missing required field 'engine'"),
    ],
)
def test_load_profile_rejects_invalid_profile_rows(tmp_path: Path, bad_row: dict[str, object], message: str) -> None:
    profile_path = _write_jsonl(tmp_path / "invalid_profile.jsonl", [bad_row])

    with pytest.raises(ValueError, match=message):
        load_profile(profile_path)


def test_sweep_script_parser_and_run_evaluate_filtered_profile(tmp_path: Path) -> None:
    profile_path = _write_jsonl(
        tmp_path / "profiles.jsonl",
        [
            _PROFILE_ROW,
            {
                **_PROFILE_ROW,
                "engine": "other-engine",
                "p50_ms": 1.0,
                "p90_ms": 2.0,
                "p95_ms": 3.0,
                "p99_ms": 4.0,
            },
        ],
    )
    gaps_path = _write_jsonl(tmp_path / "tool_gaps.jsonl", _TOOL_GAP_ROWS)
    parser = build_parser()

    args = parser.parse_args(
        [
            "--profile",
            str(profile_path),
            "--gaps",
            str(gaps_path),
            "--quantile",
            "p95",
            "--guards-ms",
            "0,5",
            "--engine",
            "fixture-engine",
            "--direction",
            "gpu_to_cpu",
            "--kv-size-mb",
            "128",
        ]
    )

    results = run(args)

    assert [row["guard_ms"] for row in results] == [0.0, 5.0]
    assert [row["safe_count"] for row in results] == [2, 1]
    assert [row["exposed_if_always_swap_ms"] for row in results] == [7.0, 7.0]
    assert {row["profile"]["engine"] for row in results} == {"fixture-engine"}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
