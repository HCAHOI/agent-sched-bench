from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.analyze_utility_threshold_sweep import main as threshold_sweep_main
from trace_collect.tool_latency_threshold_sweep import analyze_threshold_sweep


def test_threshold_sweep_decomposes_band_gain_and_short_penalty() -> None:
    rows = _panel(costs=(100.0,), robust_triggers={"short": 0.0, "band": 0.0})

    result = analyze_threshold_sweep(
        rows,
        expected_costs_ms=[100.0],
        case_thresholds_ms=[100.0],
        bootstrap_replicates=100,
        bootstrap_seed=3,
        confidence_level=0.95,
    )

    decomposition = result["points"]["100.0"]["decomposition"]["robust_clock"]
    assert decomposition["deadline_headroom_ms"] == 100.0
    assert decomposition["band_gain_ms"] == 100.0
    assert decomposition["short_exposure_penalty_ms"] == 50.0
    assert decomposition["far_tail_delta_ms"] == 0.0
    assert decomposition["delta_vs_deadline_ms"] == 50.0
    assert decomposition["captured_fraction_of_headroom"] == 0.5
    case = result["case_studies"]["100.0"]
    assert len(case["robust_clock_groupings"]["task_id"]) == 3
    assert case["overall"]["robust_clock"] == decomposition


def test_threshold_sweep_manifest_cli_writes_json_and_real_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("matplotlib")
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _panel()),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_costs_ms": [100.0, 200.0, 300.0],
                "case_thresholds_ms": [200.0, 300.0],
                "bootstrap": {
                    "replicates": 100,
                    "seed": 7,
                    "confidence_level": 0.95,
                },
                "corpora": {"test-corpus": [decisions_path.name]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    figures_dir = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_utility_threshold_sweep.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--figures-dir",
            str(figures_dir),
        ],
    )

    threshold_sweep_main()

    assert "Analyzed 1 corpora and wrote 4 figures" in capsys.readouterr().out
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(result["corpora"]["test-corpus"]["points"]) == {
        "100.0",
        "200.0",
        "300.0",
    }
    for figure_path in result["figure_paths"]:
        path = Path(figure_path)
        assert path.is_file()
        assert path.stat().st_size > 0


def test_threshold_sweep_requires_case_study_fields() -> None:
    rows = _panel(costs=(100.0,))
    del rows[0]["robust_task_count"]

    with pytest.raises(ValueError, match="missing fields"):
        analyze_threshold_sweep(
            rows,
            expected_costs_ms=[100.0],
            case_thresholds_ms=[100.0],
            bootstrap_replicates=10,
            bootstrap_seed=0,
            confidence_level=0.95,
        )


def test_threshold_sweep_uses_exact_source_aware_robust_nodes() -> None:
    rows = _panel(costs=(100.0,))
    rows[0].update(
        tool_name="read_file",
        robust_source="prior_global",
        robust_group_key=None,
    )
    rows[1].update(
        tool_name="exec",
        robust_source="prior_global",
        robust_group_key=None,
    )
    rows[2].update(
        tool_name="exec",
        robust_source="prior_tool",
        robust_group_key=None,
    )
    rows.append(
        {
            **rows[2],
            "sample_id": "group-sample",
            "task_id": "task-group",
            "latency_ms": 175.0,
            "robust_source": "prior_group",
            "robust_group_key": "exec:pytest",
        }
    )

    result = analyze_threshold_sweep(
        rows,
        expected_costs_ms=[100.0],
        case_thresholds_ms=[100.0],
        bootstrap_replicates=100,
        bootstrap_seed=3,
        confidence_level=0.95,
    )

    nodes = result["case_studies"]["100.0"]["robust_clock_groupings"]["robust_node"]
    calls_by_node = {row["group"]: row["call_count"] for row in nodes}
    assert calls_by_node == {
        "prior_global:*": 2,
        "prior_group:exec:pytest": 1,
        "prior_tool:exec": 1,
    }


@pytest.mark.parametrize(
    ("source", "group_key", "message"),
    [
        ("prior_global", "exec:bad", "prior_global"),
        ("prior_tool", "exec:bad", "prior_tool"),
        ("prior_group", None, "prior_group"),
        ("unknown", None, "unsupported robust_source"),
    ],
)
def test_threshold_sweep_rejects_inconsistent_robust_node_identity(
    source: str,
    group_key: str | None,
    message: str,
) -> None:
    rows = _panel(costs=(100.0,))
    rows[0]["robust_source"] = source
    rows[0]["robust_group_key"] = group_key

    with pytest.raises(ValueError, match=message):
        analyze_threshold_sweep(
            rows,
            expected_costs_ms=[100.0],
            case_thresholds_ms=[100.0],
            bootstrap_replicates=10,
            bootstrap_seed=0,
            confidence_level=0.95,
        )


def _panel(
    *,
    costs: tuple[float, ...] = (100.0, 200.0, 300.0),
    robust_triggers: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    latencies = {"short": 50.0, "band": 150.0, "tail": 250.0}
    rows: list[dict[str, object]] = []
    for cost in costs:
        for sample_id, latency_ms in latencies.items():
            trigger = (
                robust_triggers.get(sample_id, cost)
                if robust_triggers is not None
                else cost
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "task_id": f"task-{sample_id}",
                    "tool_name": "exec",
                    "latency_ms": latency_ms,
                    "kv_cost_ms": cost,
                    "threshold_ms": cost,
                    "label_exceeds_threshold": latency_ms > cost,
                    "deadline_trigger_ms": cost,
                    "mean_hazard_trigger_ms": cost,
                    "robust_trigger_ms": trigger,
                    "robust_source": "prior_tool",
                    "robust_task_count": 2,
                    "robust_group_key": None,
                }
            )
    return rows
