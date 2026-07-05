#!/usr/bin/env python3
"""Sample simulate tool executions for semantic divergence labeling."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from trace_collect.mismatch import MismatchOracle


FAMILIES = ("exec", "write", "edit", "spawn", "mcp", "other")
CSV_COLUMNS = (
    "source_action_id",
    "source_exit_code",
    "replay_exit_code",
    "normalized_output_match",
    "cas_manifest_match",
    "tier_1_verdict",
    "tier_2_verdict",
    "tier_3_verdict",
    "oracle_verdict",
    "human_label_different",
    "human_label_cause",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified CSV sample for divergence hand-labeling."
    )
    parser.add_argument("simulate_trace", type=Path, help="Simulate trace JSONL")
    parser.add_argument("output_csv", type=Path, help="CSV path to write")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Maximum number of tool_exec records to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed for sampling",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if not args.simulate_trace.is_file():
        raise SystemExit(f"not a file: {args.simulate_trace}")

    records = _load_compared_tool_execs(args.simulate_trace)
    sampled = _stratified_sample(records, sample_size=args.sample_size, seed=args.seed)
    _write_csv(args.output_csv, sampled)
    print(f"Wrote {len(sampled)} labeled rows to {args.output_csv}")


def _load_compared_tool_execs(trace_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("type") != "action" or record.get("action_type") != "tool_exec":
            continue
        data = _record_data(record)
        if isinstance(data.get("replay_outcome_match"), bool):
            records.append(record)
    return records


def _stratified_sample(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        data = _record_data(record)
        raw_match = data.get("replay_outcome_match")
        if not isinstance(raw_match, bool):
            raise ValueError(f"replay_outcome_match must be boolean: {raw_match!r}")
        strata[(_command_family(data), raw_match)].append(record)

    for bucket in strata.values():
        rng.shuffle(bucket)

    target = min(sample_size, len(records))
    present_families = [
        family
        for family in FAMILIES
        if strata.get((family, True)) or strata.get((family, False))
    ]
    if not present_families:
        return []

    quotas = _family_quotas(target, present_families)
    selected_ids: set[int] = set()
    selected: list[dict[str, Any]] = []
    for family in present_families:
        quota = quotas[family]
        matched_quota = quota // 2
        mismatched_quota = quota - matched_quota
        selected.extend(
            _take_from_bucket(
                strata[(family, True)],
                matched_quota,
                selected_ids=selected_ids,
            )
        )
        selected.extend(
            _take_from_bucket(
                strata[(family, False)],
                mismatched_quota,
                selected_ids=selected_ids,
            )
        )

    if len(selected) < target:
        remainder = [
            record for record in records if id(record) not in selected_ids
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: target - len(selected)])

    selected.sort(
        key=lambda record: (
            _command_family(_record_data(record)),
            _record_data(record).get("replay_outcome_match") is True,
            str(_source_action_id(record)),
        )
    )
    return selected


def _family_quotas(target: int, families: list[str]) -> dict[str, int]:
    base = target // len(families)
    remainder = target % len(families)
    return {
        family: base + (1 if index < remainder else 0)
        for index, family in enumerate(families)
    }


def _take_from_bucket(
    bucket: list[dict[str, Any]],
    count: int,
    *,
    selected_ids: set[int],
) -> list[dict[str, Any]]:
    taken: list[dict[str, Any]] = []
    for record in bucket:
        if len(taken) == count:
            break
        if id(record) in selected_ids:
            continue
        selected_ids.add(id(record))
        taken.append(record)
    return taken


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    oracle = MismatchOracle()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            data = _record_data(record)
            verdict = oracle.verdict_from_action_data(data)
            tiers = oracle.tier_verdicts_from_action_data(data)
            writer.writerow(
                {
                    "source_action_id": _source_action_id(record),
                    "source_exit_code": _csv_value(data.get("source_returncode")),
                    "replay_exit_code": _csv_value(_replay_exit_code(data)),
                    "normalized_output_match": _csv_value(
                        data.get("normalized_output_match")
                    ),
                    "cas_manifest_match": _csv_value(data.get("cas_manifest_match")),
                    "tier_1_verdict": tiers.tier_1,
                    "tier_2_verdict": tiers.tier_2,
                    "tier_3_verdict": tiers.tier_3,
                    "oracle_verdict": verdict.category if verdict is not None else "",
                    "human_label_different": "",
                    "human_label_cause": "",
                }
            )


def _record_data(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"tool_exec record data must be a dict: {record!r}")
    return data


def _source_action_id(record: dict[str, Any]) -> str:
    data = _record_data(record)
    value = data.get("source_action_id", record.get("action_id", ""))
    return str(value)


def _replay_exit_code(data: dict[str, Any]) -> Any:
    if "replay_returncode" in data:
        return data.get("replay_returncode")
    if "returncode" in data:
        return data.get("returncode")
    return data.get("command_exit_code")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _command_family(data: dict[str, Any]) -> str:
    raw_tool_name = data.get("tool_name")
    tool_name = raw_tool_name.lower() if isinstance(raw_tool_name, str) else ""
    if tool_name == "exec":
        return "exec"
    if tool_name.startswith("mcp"):
        return "mcp"
    if tool_name.startswith("write") or tool_name in {"write_file", "create_file"}:
        return "write"
    if (
        tool_name.startswith("edit")
        or tool_name in {"apply_patch", "replace", "update_file"}
    ):
        return "edit"
    if tool_name.startswith("spawn") or "spawn" in tool_name:
        return "spawn"
    return "other"


if __name__ == "__main__":
    main()
