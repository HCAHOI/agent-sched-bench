from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts.baselines.murakkab_reproduction import (
    PLAN_SCHEMA,
    _parse_problem,
    inference_manifest,
    solve_problem,
)

SCRIPT = Path(__file__).parents[1] / "scripts/baselines/murakkab_reproduction.sh"


def _profile(
    profile_id: str,
    *,
    resource: str = "A100",
    throughput: float = 20.0,
    ttft_s: float = 0.0,
    time_per_unit_s: float = 0.1,
    cost: float = 1.0,
    energy: float = 1.0,
) -> dict[str, object]:
    return {
        "id": profile_id,
        "resource": resource,
        "resource_units": 1,
        "throughput_work_units_per_s": throughput,
        "ttft_s": ttft_s,
        "time_per_work_unit_s": time_per_unit_s,
        "cost_per_resource_hour": cost,
        "energy_kwh_per_resource_hour": energy,
        "provenance": {
            "measurement_source": "profile-run.json",
            "hardware_binding": f"{resource}-80GB",
            "software_binding": "vllm-0.9/model/dtype",
            "operating_point": "batch=1,tp=1",
        },
    }


def _config(
    config_id: str, profile: str, *, accuracy: float = 0.9
) -> dict[str, object]:
    return {
        "id": config_id,
        "accuracy": accuracy,
        "nodes": {"call": {"profile": profile, "work_units_per_request": 5.0}},
    }


def _workflow(
    workflow_id: str, configurations: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "id": workflow_id,
        "nodes": ["call"],
        "edges": [],
        "configurations": configurations,
        "demands": [
            {
                "id": "good",
                "peak_requests_per_s": 1.0,
                "min_accuracy": 0.8,
                "max_latency_s": 10.0,
            }
        ],
    }


def _payload() -> dict[str, object]:
    return {
        "schema": "murakkab-profiled-deployment-v1",
        "objective": "min_cost",
        "epoch_hours": 1.0,
        "buffer_factor": 1.15,
        "resources": {"A100": 1},
        "profiles": [_profile("shared", throughput=10.5)],
        "workflows": [
            _workflow("a", [_config("shared-a", "shared")]),
            _workflow("b", [_config("shared-b", "shared")]),
        ],
    }


def test_joint_provisioning_multiplexes_profiles_across_workflows() -> None:
    result = solve_problem(_parse_problem(_payload()))

    assert result["schema"] == PLAN_SCHEMA
    assert [row["selected_configuration"] for row in result["deployments"]] == [
        "shared-a",
        "shared-b",
    ]
    provisioned = result["provisioned_profiles"][0]
    assert provisioned["instances"] == 1
    assert 10.0 <= provisioned["allocated_peak_work_units_per_s"] <= 10.5
    assert provisioned["provisioned_work_units_per_s"] == 10.5
    assert provisioned["provenance"]["measurement_source"] == "profile-run.json"
    for deployment in result["deployments"]:
        assert (
            deployment["declared_peak_requests_per_s"]
            <= deployment["allocated_peak_requests_per_s"]
        )
        assert (
            deployment["allocated_peak_requests_per_s"]
            <= deployment["max_allocatable_peak_requests_per_s"]
        )


def test_slo_filter_and_dag_critical_path_select_feasible_plan() -> None:
    payload = _payload()
    payload["profiles"] = [
        _profile("slow", ttft_s=0.1, time_per_unit_s=1.0, cost=1.0),
        _profile("fast", ttft_s=0.1, time_per_unit_s=0.1, cost=5.0),
    ]
    workflow = _workflow(
        "dag",
        [
            {
                "id": "slow-cheap",
                "accuracy": 0.9,
                "nodes": {
                    "left": {"profile": "slow", "work_units_per_request": 1.0},
                    "right": {"profile": "fast", "work_units_per_request": 1.0},
                    "join": {"profile": "fast", "work_units_per_request": 1.0},
                },
            },
            {
                "id": "fast",
                "accuracy": 0.9,
                "nodes": {
                    node: {"profile": "fast", "work_units_per_request": 1.0}
                    for node in ("left", "right", "join")
                },
            },
        ],
    )
    workflow["nodes"] = ["left", "right", "join"]
    workflow["edges"] = [["left", "join"], ["right", "join"]]
    workflow["demands"][0]["max_latency_s"] = 0.5
    payload["workflows"] = [workflow]

    result = solve_problem(_parse_problem(payload))

    deployment = result["deployments"][0]
    assert deployment["selected_configuration"] == "fast"
    assert "dispatch_waves" not in deployment
    assert deployment["dag"] == {
        "enabled_nodes": ["left", "right", "join"],
        "disabled_nodes": [],
        "enabled_edges": [["left", "join"], ["right", "join"]],
        "dependency_ready_rule": (
            "dispatch a node when all enabled predecessors have completed"
        ),
    }
    assert deployment["profiled_critical_path_latency_s"] == pytest.approx(0.4)


def test_workflow_configurations_are_shared_across_slos() -> None:
    payload = _payload()
    workflow = payload["workflows"][0]
    workflow["configurations"] = [
        _config("basic", "shared", accuracy=0.8),
        _config("best", "shared", accuracy=0.95),
    ]
    workflow["demands"] = [
        {
            "id": "basic-slo",
            "peak_requests_per_s": 0.25,
            "min_accuracy": 0.8,
            "max_latency_s": 10.0,
        },
        {
            "id": "best-slo",
            "peak_requests_per_s": 0.25,
            "min_accuracy": 0.9,
            "max_latency_s": 10.0,
        },
    ]
    payload["workflows"] = [workflow]

    result = solve_problem(_parse_problem(payload))

    selected = {
        row["demand_id"]: row["selected_configuration"] for row in result["deployments"]
    }
    assert selected["best-slo"] == "best"
    assert selected["basic-slo"] in {"basic", "best"}


def test_configuration_can_disable_optional_nodes() -> None:
    payload = _payload()
    payload["profiles"] = [
        _profile("core", ttft_s=0.1, time_per_unit_s=0.1),
        _profile("slow-optional", ttft_s=5.0, time_per_unit_s=1.0),
    ]
    workflow = _workflow("optional", [])
    workflow["nodes"] = ["core", "optional"]
    workflow["edges"] = [["core", "optional"]]
    workflow["configurations"] = [
        {
            "id": "with-optional",
            "accuracy": 0.9,
            "nodes": {
                "core": {"profile": "core", "work_units_per_request": 1.0},
                "optional": {
                    "profile": "slow-optional",
                    "work_units_per_request": 1.0,
                },
            },
        },
        {
            "id": "without-optional",
            "accuracy": 0.85,
            "nodes": {"core": {"profile": "core", "work_units_per_request": 1.0}},
        },
    ]
    workflow["demands"][0]["max_latency_s"] = 1.0
    payload["workflows"] = [workflow]

    deployment = solve_problem(_parse_problem(payload))["deployments"][0]

    assert deployment["selected_configuration"] == "without-optional"
    assert deployment["dag"]["enabled_nodes"] == ["core"]
    assert deployment["dag"]["disabled_nodes"] == ["optional"]
    assert deployment["dag"]["enabled_edges"] == []


def test_objective_switches_between_cost_and_energy_profiles() -> None:
    payload = _payload()
    payload["resources"] = {"A100": 1, "H100": 1}
    payload["profiles"] = [
        _profile("cheap", cost=1.0, energy=10.0),
        _profile("green", resource="H100", cost=4.0, energy=1.0),
    ]
    payload["workflows"] = [
        _workflow(
            "a",
            [_config("cheap-plan", "cheap"), _config("green-plan", "green")],
        )
    ]

    cost = solve_problem(_parse_problem(payload))
    payload["objective"] = "min_energy"
    energy = solve_problem(_parse_problem(payload))

    assert cost["deployments"][0]["selected_configuration"] == "cheap-plan"
    assert energy["deployments"][0]["selected_configuration"] == "green-plan"


def test_schema_fails_closed_for_black_box_or_invalid_workflows() -> None:
    payload = _payload()
    del payload["workflows"]
    with pytest.raises(ValueError, match="workflows"):
        _parse_problem(payload)

    payload = _payload()
    payload["workflows"][0]["edges"] = [["call", "call"]]
    with pytest.raises(ValueError, match="self edge"):
        _parse_problem(payload)

    payload = _payload()
    payload["workflows"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        _parse_problem(payload)

    payload = _payload()
    broken = copy.deepcopy(payload["workflows"][0]["configurations"][0])
    broken["nodes"] = {}
    payload["workflows"][0]["configurations"] = [broken]
    with pytest.raises(ValueError, match="non-empty subset"):
        _parse_problem(payload)

    payload = _payload()
    del payload["profiles"][0]["provenance"]["operating_point"]
    with pytest.raises(ValueError, match="operating_point"):
        _parse_problem(payload)


def test_manifest_and_shell_entrypoint(tmp_path: Path) -> None:
    manifest = inference_manifest()
    assert manifest["published"]["default_buffer_factor"] == 1.15
    assert "configuration_integrality" in manifest["inferred_not_tuned"]
    assert "private frontend" in " ".join(manifest["not_reproduced"])

    subprocess.run(["bash", "-n", SCRIPT], check=True)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_payload()))
    validated = subprocess.run(
        ["bash", SCRIPT, "validate", input_path],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated.stdout) == {
        "schema": "murakkab-profiled-deployment-v1",
        "valid": True,
    }
