#!/usr/bin/env python3
"""Paper-derived static-epoch deployment core for Murakkab (OSDI '26).

Murakkab's implementation is not public. This implements the strongest
published decision core that does not require its private frontend, profiles,
multi-engine runtime, or cloud manager: select one complete profiled
configuration per declared workflow/SLO and jointly provision shared executor
profiles. It is an offline deployment optimizer, not a vLLM request scheduler.
Black-box traces without a declared DAG and profiles are rejected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)

PAPER_URL = "https://www.usenix.org/system/files/osdi26-chaudhry.pdf"
SCHEMA = "murakkab-profiled-deployment-v1"
PLAN_SCHEMA = "murakkab-profiled-deployment-plan-v1"
DEFAULT_BUFFER_FACTOR = 1.15
SOLVER_TIME_LIMIT_S = 300.0


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


Identifier = Annotated[str, AfterValidator(_nonblank)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProfileProvenance(_StrictModel):
    measurement_source: Identifier
    hardware_binding: Identifier
    software_binding: Identifier
    operating_point: Identifier


class Profile(_StrictModel):
    id: Identifier
    resource: Identifier
    resource_units: PositiveInt
    throughput_work_units_per_s: Positive
    ttft_s: NonNegative
    time_per_work_unit_s: NonNegative
    cost_per_resource_hour: NonNegative
    energy_kwh_per_resource_hour: NonNegative
    provenance: ProfileProvenance


class NodeChoice(_StrictModel):
    profile: Identifier
    work_units_per_request: Positive


class Configuration(_StrictModel):
    id: Identifier
    accuracy: Probability
    nodes: dict[Identifier, NodeChoice]


class Demand(_StrictModel):
    id: Identifier
    peak_requests_per_s: Positive
    min_accuracy: Probability
    max_latency_s: Positive


class Workflow(_StrictModel):
    id: Identifier
    nodes: Annotated[list[Identifier], Field(min_length=1)]
    edges: list[list[Identifier]]
    configurations: Annotated[list[Configuration], Field(min_length=1)]
    demands: Annotated[list[Demand], Field(min_length=1)]

    @model_validator(mode="after")
    def valid_graph(self) -> Workflow:
        _unique(self.nodes, f"workflow {self.id!r} node IDs")
        _unique(
            [config.id for config in self.configurations],
            f"workflow {self.id!r} configuration IDs",
        )
        _unique([demand.id for demand in self.demands], "demand IDs")
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError("each edge must contain exactly two node IDs")
        edge_pairs = [tuple(edge) for edge in self.edges]
        if len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("edges contain duplicates")
        _topological_waves(self.nodes, edge_pairs, self.id)
        return self

    @property
    def edge_pairs(self) -> list[tuple[str, str]]:
        return [(edge[0], edge[1]) for edge in self.edges]

    @property
    def waves(self) -> list[list[str]]:
        return _topological_waves(self.nodes, self.edge_pairs, self.id)


class Problem(_StrictModel):
    schema_version: Literal[SCHEMA] = Field(alias="schema")
    objective: Literal["min_cost", "min_energy"]
    epoch_hours: Positive
    buffer_factor: Annotated[float, Field(ge=1, allow_inf_nan=False)] = (
        DEFAULT_BUFFER_FACTOR
    )
    resources: dict[Identifier, PositiveInt]
    profiles: Annotated[list[Profile], Field(min_length=1)]
    workflows: Annotated[list[Workflow], Field(min_length=1)]

    @model_validator(mode="after")
    def linked_declarations(self) -> Problem:
        if not self.resources:
            raise ValueError("resources must not be empty")
        _unique([profile.id for profile in self.profiles], "profile IDs")
        _unique([workflow.id for workflow in self.workflows], "workflow IDs")
        profiles = {profile.id: profile for profile in self.profiles}
        for profile in self.profiles:
            if profile.resource not in self.resources:
                raise ValueError(
                    f"profile {profile.id!r} references unknown resource "
                    f"{profile.resource!r}"
                )
            coefficient = (
                profile.cost_per_resource_hour
                if self.objective == "min_cost"
                else profile.energy_kwh_per_resource_hour
            )
            if coefficient == 0:
                raise ValueError(
                    f"profile {profile.id!r} has zero {self.objective} coefficient"
                )
        for workflow in self.workflows:
            declared_nodes = set(workflow.nodes)
            for config in workflow.configurations:
                enabled_nodes = set(config.nodes)
                if not enabled_nodes or not enabled_nodes <= declared_nodes:
                    raise ValueError(
                        f"configuration {config.id!r} must enable a non-empty "
                        "subset of declared DAG nodes"
                    )
                for choice in config.nodes.values():
                    if choice.profile not in profiles:
                        raise ValueError(
                            f"configuration {config.id!r} references unknown "
                            f"profile {choice.profile!r}"
                        )
        return self

    @property
    def profile_map(self) -> dict[str, Profile]:
        return {profile.id: profile for profile in self.profiles}


@dataclass(frozen=True)
class Candidate:
    workflow: Workflow
    demand: Demand
    configuration: Configuration
    latency_s: float
    work_by_profile: dict[str, float]


def _unique(values: list[str], where: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{where} contain duplicates")


def _topological_waves(
    nodes: list[str], edges: list[tuple[str, str]], where: str
) -> list[list[str]]:
    incoming = {node: 0 for node in nodes}
    children = {node: [] for node in nodes}
    for source, target in edges:
        if source not in incoming or target not in incoming:
            raise ValueError(f"workflow {where!r} edge references an unknown node")
        if source == target:
            raise ValueError(f"workflow {where!r} contains a self edge")
        incoming[target] += 1
        children[source].append(target)
    waves: list[list[str]] = []
    ready = sorted(node for node, degree in incoming.items() if degree == 0)
    while ready:
        waves.append(ready)
        following: list[str] = []
        for source in ready:
            for target in children[source]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    following.append(target)
        ready = sorted(following)
    if sum(map(len, waves)) != len(nodes):
        raise ValueError(f"workflow {where!r} must be acyclic")
    return waves


def _parse_problem(payload: Any) -> Problem:
    return Problem.model_validate(payload)


def load_problem(path: str | Path) -> Problem:
    return Problem.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _candidate(
    problem: Problem,
    workflow: Workflow,
    demand: Demand,
    config: Configuration,
) -> Candidate:
    profiles = problem.profile_map
    enabled = set(config.nodes)
    enabled_edges = [
        (source, target)
        for source, target in workflow.edge_pairs
        if source in enabled and target in enabled
    ]
    waves = _topological_waves(list(config.nodes), enabled_edges, workflow.id)
    node_latency = {
        node: profiles[choice.profile].ttft_s
        + choice.work_units_per_request * profiles[choice.profile].time_per_work_unit_s
        for node, choice in config.nodes.items()
    }
    parents = {node: [] for node in config.nodes}
    for source, target in enabled_edges:
        parents[target].append(source)
    finish: dict[str, float] = {}
    for wave in waves:
        for node in wave:
            finish[node] = node_latency[node] + max(
                (finish[parent] for parent in parents[node]), default=0.0
            )
    work: dict[str, float] = {}
    for choice in config.nodes.values():
        work[choice.profile] = (
            work.get(choice.profile, 0.0) + choice.work_units_per_request
        )
    return Candidate(workflow, demand, config, max(finish.values()), work)


def solve_problem(problem: Problem) -> dict[str, Any]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as error:
        raise RuntimeError(
            "Murakkab reproduction requires scipy.optimize.milp"
        ) from error

    groups: list[list[Candidate]] = []
    candidates: list[Candidate] = []
    for workflow in problem.workflows:
        for demand in workflow.demands:
            eligible = []
            for config in workflow.configurations:
                candidate = _candidate(problem, workflow, demand, config)
                if (
                    config.accuracy >= demand.min_accuracy
                    and candidate.latency_s <= demand.max_latency_s
                ):
                    eligible.append(candidate)
            if not eligible:
                raise RuntimeError(
                    f"no SLO-feasible configuration for {workflow.id}/{demand.id}"
                )
            groups.append(eligible)
            candidates.extend(eligible)

    profiles = problem.profile_map
    profile_ids = list(profiles)
    candidate_index = {
        id(candidate): index for index, candidate in enumerate(candidates)
    }
    allocation_offset = len(candidates)
    profile_offset = 2 * len(candidates)
    variable_count = profile_offset + len(profile_ids)
    coefficients = np.zeros(variable_count)
    for index, profile_id in enumerate(profile_ids):
        profile = profiles[profile_id]
        per_resource_hour = (
            profile.cost_per_resource_hour
            if problem.objective == "min_cost"
            else profile.energy_kwh_per_resource_hour
        )
        coefficients[profile_offset + index] = (
            problem.epoch_hours * profile.resource_units * per_resource_hour
        )

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for eligible in groups:
        demand = eligible[0].demand
        row = np.zeros(variable_count)
        for candidate in eligible:
            row[candidate_index[id(candidate)]] = 1
        rows.append(row)
        lower.append(1)
        upper.append(1)
        row = np.zeros(variable_count)
        for candidate in eligible:
            row[allocation_offset + candidate_index[id(candidate)]] = 1
        rows.append(row)
        lower.append(demand.peak_requests_per_s)
        upper.append(problem.buffer_factor * demand.peak_requests_per_s)
        for candidate in eligible:
            row = np.zeros(variable_count)
            index = candidate_index[id(candidate)]
            row[allocation_offset + index] = 1
            row[index] = -problem.buffer_factor * demand.peak_requests_per_s
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0)
    for profile_index, profile_id in enumerate(profile_ids):
        row = np.zeros(variable_count)
        for candidate in candidates:
            row[allocation_offset + candidate_index[id(candidate)]] = (
                candidate.work_by_profile.get(profile_id, 0)
            )
        row[profile_offset + profile_index] = -profiles[
            profile_id
        ].throughput_work_units_per_s
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0)
    for resource, capacity in problem.resources.items():
        row = np.zeros(variable_count)
        for profile_index, profile_id in enumerate(profile_ids):
            profile = profiles[profile_id]
            if profile.resource == resource:
                row[profile_offset + profile_index] = profile.resource_units
        rows.append(row)
        lower.append(-np.inf)
        upper.append(capacity)

    upper_bounds = np.full(variable_count, np.inf)
    upper_bounds[:allocation_offset] = 1
    integrality = np.zeros(variable_count)
    integrality[:allocation_offset] = 1
    integrality[profile_offset:] = 1
    result = milp(
        coefficients,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), lower, upper),
        options={"time_limit": SOLVER_TIME_LIMIT_S},
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"Murakkab deployment MILP did not prove optimality: {result.message}"
        )

    selected = [
        candidate for index, candidate in enumerate(candidates) if result.x[index] > 0.5
    ]
    allocated_peak = {
        id(candidate): float(result.x[allocation_offset + index])
        for index, candidate in enumerate(candidates)
    }
    instances = {
        profile_id: int(round(result.x[profile_offset + index]))
        for index, profile_id in enumerate(profile_ids)
    }
    profile_loads = {profile_id: 0.0 for profile_id in profile_ids}
    for candidate in selected:
        for profile_id, work in candidate.work_by_profile.items():
            profile_loads[profile_id] += allocated_peak[id(candidate)] * work

    deployments = [
        {
            "workflow_id": candidate.workflow.id,
            "demand_id": candidate.demand.id,
            "selected_configuration": candidate.configuration.id,
            "accuracy": candidate.configuration.accuracy,
            "profiled_critical_path_latency_s": candidate.latency_s,
            "declared_peak_requests_per_s": candidate.demand.peak_requests_per_s,
            "allocated_peak_requests_per_s": allocated_peak[id(candidate)],
            "max_allocatable_peak_requests_per_s": problem.buffer_factor
            * candidate.demand.peak_requests_per_s,
            "dag": {
                "enabled_nodes": list(candidate.configuration.nodes),
                "disabled_nodes": [
                    node
                    for node in candidate.workflow.nodes
                    if node not in candidate.configuration.nodes
                ],
                "enabled_edges": [
                    [source, target]
                    for source, target in candidate.workflow.edge_pairs
                    if source in candidate.configuration.nodes
                    and target in candidate.configuration.nodes
                ],
                "dependency_ready_rule": (
                    "dispatch a node when all enabled predecessors have completed"
                ),
            },
            "nodes": {
                node: choice.model_dump()
                for node, choice in candidate.configuration.nodes.items()
            },
        }
        for candidate in selected
    ]
    deployments.sort(key=lambda row: (row["workflow_id"], row["demand_id"]))
    provisioned = [
        {
            "profile": profile_id,
            "instances": instances[profile_id],
            "resource": profiles[profile_id].resource,
            "resource_units_per_instance": profiles[profile_id].resource_units,
            "allocated_peak_work_units_per_s": profile_loads[profile_id],
            "provisioned_work_units_per_s": instances[profile_id]
            * profiles[profile_id].throughput_work_units_per_s,
            "provenance": profiles[profile_id].provenance.model_dump(),
        }
        for profile_id in profile_ids
        if instances[profile_id]
    ]
    return {
        "schema": PLAN_SCHEMA,
        "fidelity": "paper-derived-static-epoch-subset",
        "objective": problem.objective,
        "objective_value": float(result.fun),
        "objective_unit": (
            "currency_units" if problem.objective == "min_cost" else "kWh"
        ),
        "epoch_hours": problem.epoch_hours,
        "buffer_factor": problem.buffer_factor,
        "solver": {
            "name": "scipy-highs-milp",
            "status": "optimal",
            "time_limit_s": SOLVER_TIME_LIMIT_S,
        },
        "deployments": deployments,
        "provisioned_profiles": provisioned,
    }


def inference_manifest() -> dict[str, Any]:
    return {
        "paper": PAPER_URL,
        "published": {
            "input": "declarative DAG, workflow/model profiles, demand, SLOs, resources",
            "epoch_decisions": "workflow knobs, executor, hardware/parallelism, instances, routing",
            "capacity": "projected peak load with a unified buffer factor",
            "default_buffer_factor": DEFAULT_BUFFER_FACTOR,
            "objectives": ["min_cost", "min_energy", "max_accuracy_under_cost"],
            "optimizer": "MILP with a 300 second Gurobi limit",
            "runtime_dispatch": "deterministic after executable workflow selection",
        },
        "implemented": {
            "input": SCHEMA,
            "objectives": ["min_cost", "min_energy"],
            "decision": "one SLO-filtered choice from the workflow's shared configuration space",
            "topology_knobs": "a configuration enables an induced non-empty subgraph",
            "latency": "enabled-subgraph critical path from profile TTFT and time per work unit",
            "demand": "paper bounds lambda <= allocated peak <= alpha*lambda",
            "capacity": "shared instances serve allocated peak work",
            "output": "configuration, enabled DAG edges, dependency readiness rule, provisioning",
            "profile_provenance": "measurement source plus hardware/software/operating-point binding",
        },
        "inferred_not_tuned": {
            "solver": "SciPy HiGHS replaces the unpublished Gurobi implementation",
            "dag_latency": "critical path over node profiles; Appendix A.5 gives only a scalar formula",
            "multiplexing_factor": "one; the paper names model-specific mu but does not define values",
            "configuration_integrality": "one configuration per workflow/SLO; Appendix A.5 permits x across configurations but omits selection variables",
            "topology": "enabled nodes use the induced subgraph of the declared DAG",
            "routing": "all allocated demand uses its chosen configuration; no fractional routing",
            "units": "cost and energy are per resource-hour times epoch length",
        },
        "not_reproduced": [
            "LLM workflow generation and executor matching",
            "profile collection, quality measurement, and demand forecasting",
            "dynamic per-request workflows",
            "max-accuracy-under-cost objective",
            "online autoscaling and early re-optimization",
            "multi-engine provisioning, routing, batching, and transition overhead",
            "private frontend, executor library, cloud manager, and vLLM changes",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    validate = subparsers.add_parser("validate")
    validate.add_argument("input", type=Path)
    solve = subparsers.add_parser("solve")
    solve.add_argument("input", type=Path)
    solve.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        print(json.dumps(inference_manifest(), indent=2, sort_keys=True))
        return
    problem = load_problem(args.input)
    if args.command == "validate":
        print(json.dumps({"schema": SCHEMA, "valid": True}, sort_keys=True))
        return
    output = json.dumps(solve_problem(problem), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
