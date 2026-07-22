#!/usr/bin/env python3
"""Run the configured W5 multi-tenant policy/load/workload matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/serving/w5_multitenant.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workloads", nargs="*")
    parser.add_argument("--policies", nargs="*")
    parser.add_argument("--loads", nargs="*", type=int)
    parser.add_argument("--limit-programs", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Permit heldout_eval workloads; forbids development subset flags.",
    )
    return parser


def _paths(value: str | list[str]) -> list[Path]:
    values = [value] if isinstance(value, str) else value
    if not values:
        raise ValueError("task_ids_file must be a path or non-empty path list")
    return [Path(path) for path in values]


def validate_matrix_inputs(
    config: dict[str, object],
    workloads: list[dict[str, object]],
    policies: list[str],
) -> None:
    """Fail before GPU work when a selected W5 input is missing or inconsistent."""

    trigger_path = Path(str(config["trigger_table"]))
    if not trigger_path.is_file():
        raise ValueError(f"trigger table is not a file: {trigger_path}")
    trigger_payload = json.loads(trigger_path.read_text(encoding="utf-8"))
    table_fraction = (trigger_payload.get("metadata") or {}).get(
        "restore_cost_fraction"
    )
    runtime_fraction = float(config["restore_cost_fraction"])
    if not isinstance(table_fraction, (int, float)) or not math.isclose(
        float(table_fraction), runtime_fraction, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "trigger-table restore_cost_fraction does not match runtime: "
            f"{table_fraction!r} != {runtime_fraction}"
        )

    if "continuum" in policies:
        prefill_path = Path(str(config["continuum_prefill_profile"]))
        if not prefill_path.is_file():
            raise ValueError(
                "Continuum requires its same-model, same-GPU prefill profile: "
                f"{prefill_path}"
            )

    for workload in workloads:
        name = str(workload["name"])
        for field in ("replay_trace_root", "profile_trace_root"):
            root = Path(str(workload[field]))
            if not root.is_dir():
                raise ValueError(f"{name} {field} is not a directory: {root}")
        for path in _paths(workload["task_ids_file"]):
            if not path.is_file():
                raise ValueError(f"{name} task IDs are not a file: {path}")


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    workload_rows = config["workloads"]
    selected_names = args.workloads or [
        row["name"] for row in workload_rows if row["corpus_role"] == "development"
    ]
    by_name = {row["name"]: row for row in workload_rows}
    unknown = sorted(set(selected_names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown workloads: {unknown}")
    selected = [by_name[name] for name in selected_names]
    selected_has_heldout = any(
        row["corpus_role"] == "heldout_eval" for row in selected
    )
    if selected_has_heldout and not args.final:
        raise ValueError("heldout_eval workloads require --final")
    if args.final and not selected_has_heldout:
        raise ValueError("--final requires a configured heldout_eval workload")
    if args.final and (args.limit_programs is not None or args.max_turns is not None):
        raise ValueError("--final forbids --limit-programs and --max-turns")
    if args.limit_programs is not None and args.limit_programs <= 0:
        raise ValueError("--limit-programs must be > 0")
    if args.max_turns is not None and args.max_turns <= 0:
        raise ValueError("--max-turns must be > 0")

    policies = args.policies or config["policies"]
    loads = args.loads or config["load_levels"]
    if not set(policies) <= set(config["policies"]):
        raise ValueError("requested policies are not a subset of the frozen config")
    if not set(loads) <= set(config["load_levels"]):
        raise ValueError("requested loads are not a subset of the frozen config")
    if "ours" in policies and any(
        workload["corpus_role"] == "heldout_eval" for workload in selected
    ):
        raise ValueError(
            "held-out ours is blocked: the configured static trigger table is an "
            "uncertified deployment-demo approximation"
        )
    validate_matrix_inputs(config, selected, policies)

    args.output_root.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    for workload in selected:
        for load in loads:
            for policy in policies:
                output = (
                    args.output_root / f"{workload['name']}--load{load}--{policy}.json"
                )
                command = [
                    sys.executable,
                    "spike/run_multitenant.py",
                    "--config",
                    str(args.config),
                    "--workload",
                    workload["name"],
                    "--policy",
                    policy,
                    "--load",
                    str(load),
                    "--output",
                    str(output),
                ]
                if workload["corpus_role"] == "heldout_eval":
                    command.append("--final")
                if args.limit_programs is not None:
                    command += ["--limit-programs", str(args.limit_programs)]
                if args.max_turns is not None:
                    command += ["--max-turns", str(args.max_turns)]
                print("+", " ".join(command), flush=True)
                completed = subprocess.run(command, env=env)
                cell = {
                    "workload": workload["name"],
                    "corpus_role": workload["corpus_role"],
                    "load": load,
                    "policy": policy,
                    "output": str(output),
                    "returncode": completed.returncode,
                }
                cells.append(cell)
                (args.output_root / "matrix.json").write_text(
                    json.dumps({"config": str(args.config), "cells": cells}, indent=2),
                    encoding="utf-8",
                )
                if completed.returncode != 0:
                    return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
