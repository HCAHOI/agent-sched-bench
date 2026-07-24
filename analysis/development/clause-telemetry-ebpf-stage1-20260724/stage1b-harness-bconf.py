#!/usr/bin/env python3
"""Stage-1b B-only confirmation runner (amendment B-7)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePath
import random
import re
import statistics
import subprocess
import time
from typing import Any

from mapper import map_events, self_check as mapper_self_check


ARTIFACT_DIR = Path(__file__).resolve().parent
COLLECTOR = ARTIFACT_DIR / "collector.py"
FIXTURE = ARTIFACT_DIR / "stage1b-fixture-bconf.json"
RESULTS = ARTIFACT_DIR / "stage1b-results-bconf.json"
SUMMARY = Path("/tmp/fable-stage1b-scratch/bconf-auto-summary.md")
PRIVILEGED_LOG = ARTIFACT_DIR / "stage1b-privileged_commands-bconf.jsonl"
SCRATCH = Path("/tmp/fable-stage1b-scratch")
SENTINEL_SEQ = 2**64 - 1
MECHANISM_FAILURE_REASONS = {
    "collector_failed",
    "loss_zero",
    "sequence_complete",
    "exec_process_balanced",
    "argv_not_truncated",
    "attached_before_launch",
    "cgroup_configured",
}
ORACLE_MOUNTS = [
    "--mount",
    "type=bind,src=/usr/bin/strace,dst=/tmp/fable-strace,readonly",
    "--mount",
    "type=bind,src=/usr/lib/x86_64-linux-gnu/libunwind-ptrace.so.0.0.0,dst=/tmp/libunwind-ptrace.so.0,readonly",
    "--mount",
    "type=bind,src=/usr/lib/x86_64-linux-gnu/libunwind-x86_64.so.8.0.1,dst=/tmp/libunwind-x86_64.so.8,readonly",
    "--mount",
    "type=bind,src=/usr/lib/x86_64-linux-gnu/libunwind.so.8.0.1,dst=/tmp/libunwind.so.8,readonly",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_privileged(
    argv: list[str], start: str, end: str, exit_code: int
) -> None:
    row = {
        "argv": argv,
        "command": " ".join(argv),
        "start_utc": start,
        "end_utc": end,
        "exit_code": exit_code,
    }
    with PRIVILEGED_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def normalized_case(
    fixture: dict[str, Any], case_id: str, variant: str | None = None
) -> dict[str, Any]:
    source = fixture["mechanism"]["cases"][case_id]
    if case_id != "M":
        return source
    selected = source["variants"][variant]
    return {
        "entrypoint": source["entrypoint"],
        "args": selected["args"],
        "static_clauses": selected["static_clauses"],
        "live_marker": source["live_marker"],
    }


def mechanism_payload(
    fixture: dict[str, Any], case_id: str, variant: str | None
) -> tuple[Path, dict[str, Any]]:
    payload = {
        "image": fixture["mechanism"]["image"],
        "image_id": fixture["mechanism"]["image_id"],
        "collector": fixture["mechanism"]["collector"],
        "case": normalized_case(fixture, case_id, variant),
    }
    suffix = case_id if variant is None else f"{case_id}-{variant}"
    path = SCRATCH / f"mechanism-{suffix}.json"
    write_json(path, payload)
    return path, payload


def docker_command(
    payload: dict[str, Any], name: str, cidfile: Path
) -> list[str]:
    case = payload["case"]
    args = case.get("args")
    if args is None:
        args = ["-c", case["command"]]
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--cidfile",
        str(cidfile),
        "--entrypoint",
        case["entrypoint"],
        payload["image"],
        *args,
    ]


def run_control(payload: dict[str, Any], name: str, cidfile: Path) -> dict[str, Any]:
    cidfile.unlink(missing_ok=True)
    command = docker_command(payload, name, cidfile)
    launch_ns = time.monotonic_ns()
    run = subprocess.run(command, capture_output=True)
    exit_ns = time.monotonic_ns()
    cidfile.unlink(missing_ok=True)
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "docker_command": command,
        "stdout_hex": run.stdout.hex(),
        "stderr_hex": run.stderr.hex(),
        "exit_code": run.returncode,
        "elapsed_ns": exit_ns - launch_ns,
        "launch_monotonic_ns": launch_ns,
        "exit_monotonic_ns": exit_ns,
    }


def run_candidate(
    payload_path: Path,
    payload: dict[str, Any],
    case_id: str,
    rep: int,
    name: str,
    cidfile: Path,
) -> dict[str, Any]:
    argv = [
        "sudo",
        "/usr/bin/python3",
        str(COLLECTOR),
        "--payload",
        str(payload_path),
        "--case",
        case_id,
        "--rep",
        str(rep),
        "--name",
        name,
        "--cidfile",
        str(cidfile),
    ]
    start = utc_now()
    run = subprocess.run(argv, capture_output=True, text=True)
    end = utc_now()
    append_privileged(argv, start, end, run.returncode)
    if run.returncode != 0:
        return {
            "collector_exit_code": run.returncode,
            "collector_stdout": run.stdout,
            "collector_stderr": run.stderr,
            "error": "collector_failed",
        }
    candidate = json.loads(run.stdout)
    candidate["collector_stderr"] = run.stderr
    mapping = map_events(
        {
            "static_clauses": payload["case"]["static_clauses"],
            "exec_events": candidate["exec_events"],
            "process_records": candidate["process_records"],
            "timing": candidate["timing"],
        }
    )
    candidate["mapping"] = mapping
    return candidate


def expected_count(fixture: dict[str, Any], case_id: str) -> int:
    if case_id == "P":
        expected = fixture["evaluator"]["P"]
        return len(expected["expected_resolved"]) + expected["expected_unresolved"]["count"]
    if case_id == "W":
        return fixture["evaluator"]["gates"]["GW"]["exact_exec_events"]
    return fixture["evaluator"][case_id]["expected_exec_events"]


def assess_integrity(
    fixture: dict[str, Any], case_id: str, candidate: dict[str, Any]
) -> dict[str, Any]:
    if candidate.get("error"):
        return {"pass": False, "reasons": [candidate["error"]]}
    mapping = candidate["mapping"]
    observed = len(mapping["resolved"]) + len(mapping["unresolved"])
    expected = expected_count(fixture, case_id)
    checks = {
        "loss_zero": candidate["integrity"]["reserve_failures"] == 0,
        "sequence_complete": candidate["integrity"]["sequence_failures"] == 0,
        "exec_process_balanced": candidate["integrity"]["exec_process_balanced"],
        "argv_not_truncated": candidate["integrity"]["argv_truncation_count"] == 0,
        "attached_before_launch": candidate["integrity"]["attached_before_launch"],
        "cgroup_configured": candidate["integrity"]["cgroup_setup_error"] is None,
        "exact_count": observed == expected,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "observed_mapped_execs": observed,
        "expected_mapped_execs": expected,
        "reasons": [key for key, value in checks.items() if not value],
    }


def run_oracle(
    fixture: dict[str, Any], case_id: str, variant: str | None
) -> dict[str, Any]:
    payload = {
        "image": fixture["mechanism"]["image"],
        "case": normalized_case(fixture, case_id, variant),
    }
    case = payload["case"]
    target_args = case.get("args")
    if target_args is None:
        target_args = ["-c", case["command"]]
    name = f"fable-stage1-oracle-{case_id.lower()}"
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        *ORACLE_MOUNTS,
        "--env",
        "LD_LIBRARY_PATH=/tmp",
        "--entrypoint",
        "/tmp/fable-strace",
        payload["image"],
        "-f",
        "-qq",
        "-s",
        "4096",
        "-e",
        "trace=execve",
        "/usr/bin/env",
        "-u",
        "LD_LIBRARY_PATH",
        case["entrypoint"],
        *target_args,
    ]
    run = subprocess.run(argv, capture_output=True)
    trace = run.stderr.decode("utf-8", errors="replace")
    paths = re.findall(r'execve\("([^"]+)"', trace)
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "command": argv,
        "exit_code": run.returncode,
        "stdout_hex": run.stdout.hex(),
        "trace": trace,
        "exec_presence": [PurePath(path).name for path in paths],
        "use": "presence_absence_only",
    }


def compare_oracle(
    candidate: dict[str, Any], oracle: dict[str, Any]
) -> dict[str, Any]:
    oracle_names = set(oracle["exec_presence"])
    lane_e_events = [
        {
            "host_pid": event["host_pid"],
            "exec_seq": event["exec_seq"],
            "argv0": PurePath(event["argv"][0]).name,
            "present_in_oracle": PurePath(event["argv"][0]).name in oracle_names,
        }
        for event in candidate["exec_events"]
        if event["argv"]
    ]
    lane_e_names = {event["argv0"] for event in lane_e_events}
    return {
        "use": "comparison_output_only",
        "lane_e_events": lane_e_events,
        "all_lane_e_argv0_present": all(
            event["present_in_oracle"] for event in lane_e_events
        ),
        "lane_e_only_argv0": sorted(lane_e_names - oracle_names),
        "oracle_only_argv0": sorted(oracle_names - lane_e_names),
        "mapping_unchanged": True,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def constructed_truth(
    fixture: dict[str, Any], candidate: dict[str, Any]
) -> dict[tuple[int, int], tuple[str, int]]:
    clauses = fixture["mechanism"]["cases"]["P"]["static_clauses"]
    occurrences: dict[str, int] = {}
    truth = {}
    for event in sorted(candidate["exec_events"], key=lambda row: row["t_exec_ns"]):
        if not event["argv"]:
            continue
        normalized = [PurePath(event["argv"][0]).name, *event["argv"][1:]]
        matches = [clause for clause in clauses if clause["argv"] == normalized]
        if len(matches) != 1:
            continue
        clause_id = matches[0]["clause_id"]
        occurrence = occurrences.get(clause_id, 0)
        occurrences[clause_id] = occurrence + 1
        truth[(event["host_pid"], event["exec_seq"])] = (
            clause_id,
            occurrence,
        )
    return truth


def cpu_accounting(
    fixture: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    cpu_truth = {
        (row["clause_id"], row["occurrence_index"]): row["cpu_s"]
        for row in fixture["evaluator"]["P"]["expected_resolved"]
        if "cpu_s" in row
    }
    truth = constructed_truth(fixture, candidate)
    target_ns = 0
    non_target_ns = 0
    wrong_ns = 0
    correct_target_ns: dict[tuple[str, int], int] = {}
    for row in candidate["mapping"]["resolved"]:
        record = row["process_record"]
        if record is None:
            continue
        record_key = (record["host_pid"], record["exec_seq"])
        assigned = (row["clause_id"], row["occurrence_index"])
        cpu_ns = record["cpu_ns"]
        if assigned in cpu_truth:
            target_ns += cpu_ns
        else:
            non_target_ns += cpu_ns
        if truth.get(record_key) != assigned:
            wrong_ns += cpu_ns
        elif assigned in cpu_truth:
            correct_target_ns[assigned] = (
                correct_target_ns.get(assigned, 0) + cpu_ns
            )

    unattributed_ns = sum(
        row["process_record"]["cpu_ns"]
        for row in candidate["mapping"]["unresolved"]
        if row["process_record"] is not None
    )
    raw_exit_ns = sum(
        row["utime_ns"] + row["stime_ns"]
        for row in candidate["raw_events"]
        if row["type"] == "exit"
        and row["cgroup_id"] == candidate["cgroup"]["cgroup_id"]
    )
    housekeeping_ns = sum(
        row["utime_ns"] + row["stime_ns"]
        for row in candidate["raw_events"]
        if row["type"] == "exit"
        and row["cgroup_id"] == candidate["cgroup"]["cgroup_id"]
        and row["exec_seq"] == SENTINEL_SEQ
    )
    accounted_ns = target_ns + non_target_ns + unattributed_ns
    attributed_s = (target_ns + non_target_ns) / 1e9
    cgroup_cpu_s = candidate["cgroup"]["cpu_usage_usec"] / 1e6
    return {
        "correct_target_cpu_s": {
            key: value / 1e9 for key, value in correct_target_ns.items()
        },
        "target_clause_attributed_cpu_s": target_ns / 1e9,
        "non_target_clause_attributed_cpu_s": non_target_ns / 1e9,
        "unattributed_record_cpu_s": unattributed_ns / 1e9,
        "attributed_cpu_s": attributed_s,
        "attributed_to_cgroup_ratio": (
            attributed_s / cgroup_cpu_s if cgroup_cpu_s else math.inf
        ),
        "cgroup_cpu_s": cgroup_cpu_s,
        "accounted_cpu_s": accounted_ns / 1e9,
        "raw_exit_cpu_s": raw_exit_ns / 1e9,
        "runtime_housekeeping_cpu_s": housekeeping_ns / 1e9,
        "raw_exit_equals_accounted": raw_exit_ns == accounted_ns,
        "raw_exit_equals_accounted_plus_housekeeping": (
            raw_exit_ns == accounted_ns + housekeeping_ns
        ),
        "wrong_clause_cpu_s": wrong_ns / 1e9,
    }


def evaluate_g5_case(
    fixture: dict[str, Any], case_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    required = (
        fixture["mechanism"]["cases"][case_id]["repetitions_per_variant"] * 2
        if case_id == "M"
        else fixture["mechanism"]["cases"][case_id]["repetitions"]
    )
    partial = [
        {
            "repetition": row["repetition"],
            "variant": row["variant"],
            "reasons": row["integrity"]["reasons"],
        }
        for row in rows
        if not row["integrity"]["pass"]
    ]
    return {
        "pass": len(rows) == required and not partial,
        "completed_repetitions": len(rows),
        "required_repetitions": required,
        "partial_repetitions": len(partial),
        "partial_rows": partial,
    }


def update_g5(
    results: dict[str, Any],
    fixture: dict[str, Any],
    case_id: str,
    rows: list[dict[str, Any]],
) -> None:
    g5 = results["gates"].setdefault("G5", {"pass": False, "cases": {}})
    g5["cases"][case_id] = evaluate_g5_case(fixture, case_id, rows)
    g5["pass"] = (
        set(g5["cases"]) == set(fixture["mechanism"]["case_order"])
        and all(report["pass"] for report in g5["cases"].values())
    )


def compare_semantics(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    def normalize(stdout_hex: str) -> tuple[bytes, dict[str, str] | None]:
        lines = bytes.fromhex(stdout_hex).splitlines(keepends=True)
        env_indexes = [
            index for index, line in enumerate(lines) if line.startswith(b"ENV=")
        ]
        if len(env_indexes) != 1:
            return b"".join(lines), None
        index = env_indexes[0]
        line = lines[index]
        try:
            environment = json.loads(line.removeprefix(b"ENV="))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return b"".join(lines), None
        if not isinstance(environment, dict) or "HOSTNAME" not in environment:
            return b"".join(lines), None
        normalized, replacements = re.subn(
            rb'("HOSTNAME"\s*:\s*)"(?:[^"\\]|\\.)*"',
            rb'\1"<HOSTNAME excluded>"',
            line,
        )
        if replacements != 1:
            return b"".join(lines), None
        lines[index] = normalized
        return b"".join(lines), {
            str(key): str(value) for key, value in environment.items()
        }

    control_stdout, control_env = normalize(control["stdout_hex"])
    candidate_stdout, candidate_env = normalize(candidate["stdout_hex"])
    hostname_present = control_env is not None and candidate_env is not None
    other_keys = sorted(
        (set(control_env or {}) | set(candidate_env or {})) - {"HOSTNAME"}
    )
    differing_environment = [
        {
            "key": key,
            "control": (control_env or {}).get(key),
            "candidate": (candidate_env or {}).get(key),
        }
        for key in other_keys
        if (control_env or {}).get(key) != (candidate_env or {}).get(key)
    ]
    stdout_equal = hostname_present and control_stdout == candidate_stdout
    stderr_equal = control["stderr_hex"] == candidate["stderr_hex"]
    exit_equal = control["exit_code"] == candidate["exit_code"]
    return {
        "pass": (
            stdout_equal
            and stderr_equal
            and exit_equal
            and not differing_environment
        ),
        "stdout_equal_excluding_hostname_value": stdout_equal,
        "stderr_byte_exact": stderr_equal,
        "exit_code_exact": exit_equal,
        "child_environment": {
            "hostname_present_both": hostname_present,
            "hostname_value_excluded_from_equality": True,
            "equal_excluding_hostname_value": (
                hostname_present and not differing_environment
            ),
            "differing_keys_excluding_hostname": differing_environment,
        },
    }


def evaluate_p(fixture: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [pair for pair in pairs if not pair["candidate"].get("error")]
    expected = fixture["evaluator"]["P"]
    expected_keys = {
        (row["clause_id"], row["occurrence_index"])
        for row in expected["expected_resolved"]
    }
    cpu_truth = {
        (row["clause_id"], row["occurrence_index"]): row["cpu_s"]
        for row in expected["expected_resolved"]
        if "cpu_s" in row
    }
    expected_unresolved = expected["expected_unresolved"]
    semantic = [
        {
            "repetition": pair["repetition"],
            **compare_semantics(pair["control"], pair["candidate"]),
        }
        for pair in pairs
    ]
    identity_rows = []
    correct_cpu = 0.0
    excess_cpu = 0.0
    missing_cpu = 0.0
    sequential_errors = []
    pipeline_errors = []
    target_attributed = 0.0
    non_target_attributed = 0.0
    unattributed = 0.0
    accounted = 0.0
    raw_exit = 0.0
    wrong_clause = 0.0
    housekeeping = 0.0
    reconciliation_rows = []
    ratios = []
    pipeline_structure_rows = []
    startup_ms_all = []
    for pair in pairs:
        candidate = pair["candidate"]
        mapping = candidate["mapping"]
        keys = {
            (row["clause_id"], row["occurrence_index"])
            for row in mapping["resolved"]
        }
        multi_group, entry_group = expected_unresolved["groups"]
        multi_rows = [
            row
            for row in mapping["unresolved"]
            if row["reason"] == multi_group["reason"]
        ]
        entry_rows = [
            row
            for row in mapping["unresolved"]
            if row["reason"] == entry_group["reason"]
        ]
        unresolved_ok = (
            len(mapping["unresolved"]) == expected_unresolved["count"]
            and len(multi_rows) == multi_group["count"]
            and all(
                row["candidate_clause_ids"] == multi_group["candidate_clause_ids"]
                for row in multi_rows
            )
            and len(entry_rows) == entry_group["count"]
            and all(
                row["event"]["argv"]
                and PurePath(row["event"]["argv"][0]).name
                == entry_group["entrypoint_argv0"]
                for row in entry_rows
            )
        )
        identity_rows.append(
            {
                "resolved_exact": keys == expected_keys,
                "unresolved_exact": unresolved_ok,
                "resolved_count": len(mapping["resolved"]),
                "unresolved_count": len(mapping["unresolved"]),
            }
        )
        accounting = cpu_accounting(fixture, candidate)
        target_attributed += accounting["target_clause_attributed_cpu_s"]
        non_target_attributed += accounting["non_target_clause_attributed_cpu_s"]
        unattributed += accounting["unattributed_record_cpu_s"]
        accounted += accounting["accounted_cpu_s"]
        raw_exit += accounting["raw_exit_cpu_s"]
        wrong_clause += accounting["wrong_clause_cpu_s"]
        housekeeping += accounting["runtime_housekeeping_cpu_s"]
        reconciliation_rows.append(
            {
                "repetition": candidate["repetition"],
                "target_clause_attributed_cpu_s": accounting[
                    "target_clause_attributed_cpu_s"
                ],
                "non_target_clause_attributed_cpu_s": accounting[
                    "non_target_clause_attributed_cpu_s"
                ],
                "unattributed_record_cpu_s": accounting[
                    "unattributed_record_cpu_s"
                ],
                "accounted_cpu_s": accounting["accounted_cpu_s"],
                "raw_exit_cpu_s": accounting["raw_exit_cpu_s"],
                "raw_exit_equals_accounted": accounting[
                    "raw_exit_equals_accounted"
                ],
                "runtime_housekeeping_cpu_s": accounting[
                    "runtime_housekeeping_cpu_s"
                ],
                "raw_exit_equals_accounted_plus_housekeeping": accounting[
                    "raw_exit_equals_accounted_plus_housekeeping"
                ],
                "cgroup_cpu_s": accounting["cgroup_cpu_s"],
                "attributed_to_cgroup_ratio": accounting[
                    "attributed_to_cgroup_ratio"
                ],
            }
        )
        structure = fixture["evaluator"]["gates"]["G4"]["pipeline_structure"]
        stage_records = {
            row["clause_id"]: row["process_record"]
            for row in mapping["resolved"]
            if row["clause_id"] in structure["stages"]
            and row["process_record"] is not None
        }
        bound_records = {
            clause_id: next(
                (
                    row["process_record"]
                    for row in mapping["resolved"]
                    if row["clause_id"] == clause_id
                    and row["process_record"] is not None
                ),
                None,
            )
            for clause_id in (
                structure["after_exit_of"],
                structure["before_exec_of"],
            )
        }
        after_record = bound_records[structure["after_exit_of"]]
        before_record = bound_records[structure["before_exec_of"]]
        structure_complete = (
            len(stage_records) == len(structure["stages"])
            and after_record is not None
            and before_record is not None
            and after_record["t_exit_ns"] is not None
            and all(
                record["t_exit_ns"] is not None
                for record in stage_records.values()
            )
        )
        if structure_complete:
            stage_execs = [
                stage_records[stage]["t_exec_ns"] for stage in structure["stages"]
            ]
            stage_exits = [
                stage_records[stage]["t_exit_ns"] for stage in structure["stages"]
            ]
            startups_ms = [
                (stage_records[stage]["wall_ns"] / 1e9 - cpu_truth[(stage, 0)])
                * 1000
                for stage in structure["stages"]
            ]
            structure_row = {
                "repetition": candidate["repetition"],
                "stages_resolved": True,
                "startup_non_negative": min(startups_ms) >= 0.0,
                "parallel_overlap": max(stage_execs) < min(stage_exits),
                "ordering": (
                    min(stage_execs) > after_record["t_exit_ns"]
                    and max(stage_exits) < before_record["t_exec_ns"]
                ),
                "interpreter_startup_ms": startups_ms,
            }
            startup_ms_all.extend(startups_ms)
        else:
            structure_row = {
                "repetition": candidate["repetition"],
                "stages_resolved": False,
                "startup_non_negative": False,
                "parallel_overlap": False,
                "ordering": False,
                "interpreter_startup_ms": None,
            }
        structure_row["pass"] = (
            structure_row["stages_resolved"]
            and structure_row["startup_non_negative"]
            and structure_row["parallel_overlap"]
            and structure_row["ordering"]
        )
        pipeline_structure_rows.append(structure_row)
        for row in mapping["resolved"]:
            record = row["process_record"]
            if record is None:
                continue
            expected_row = next(
                (
                    item
                    for item in expected["expected_resolved"]
                    if item["clause_id"] == row["clause_id"]
                    and item["occurrence_index"] == row["occurrence_index"]
                ),
                None,
            )
            if expected_row and record["wall_ns"] is not None:
                wall_s = record["wall_ns"] / 1e9
                if "sequential_wall_s" in expected_row:
                    sequential_errors.append(
                        abs(wall_s - expected_row["sequential_wall_s"])
                    )
                if "pipeline_wall_s" in expected_row:
                    pipeline_errors.append(
                        abs(wall_s - expected_row["pipeline_wall_s"])
                    )
        for key, target in cpu_truth.items():
            observed = accounting["correct_target_cpu_s"].get(key, 0.0)
            correct_cpu += min(observed, target)
            excess_cpu += max(0.0, observed - target)
            missing_cpu += max(0.0, target - observed)
        ratios.append(accounting["attributed_to_cgroup_ratio"])

    overhead_ms = [
        (pair["candidate"]["elapsed_ns"] - pair["control"]["elapsed_ns"]) / 1e6
        for pair in pairs
    ]
    relative = [
        (pair["candidate"]["elapsed_ns"] - pair["control"]["elapsed_ns"])
        / pair["control"]["elapsed_ns"]
        for pair in pairs
    ]
    rng = random.Random(fixture["evaluator"]["bootstrap_seed"])
    boot = (
        [
            statistics.median(rng.choices(relative, k=len(relative)))
            for _ in range(fixture["evaluator"]["bootstrap_draws"])
        ]
        if relative
        else []
    )
    gates = fixture["evaluator"]["gates"]
    constructed = expected["constructed_cpu_s_per_repetition"] * len(pairs)
    return {
        "G1": {
            "pass": len(pairs) == 30 and all(row["pass"] for row in semantic),
            "matching_pairs": sum(row["pass"] for row in semantic),
            "required_pairs": 30,
            "hostname_value_excluded_under_amendment": "A-3",
            "rows": semantic,
        },
        "G2": {
            "pass": len(pairs) == 30
            and all(
                row["resolved_exact"] and row["unresolved_exact"]
                for row in identity_rows
            ),
            "rows": identity_rows,
        },
        "G3": {
            "pass": (
                len(pairs) == 30
                and constructed > 0
                and correct_cpu / constructed
                >= gates["G3"]["minimum_clipped_target_credit"]
                and max(ratios, default=math.inf)
                <= gates["G3"]["attributed_to_cgroup_cpu_ratio_max"]
                and wrong_clause <= gates["G3"]["wrong_clause_cpu_s_max"]
                and all(
                    row["raw_exit_equals_accounted_plus_housekeeping"]
                    for row in reconciliation_rows
                )
            ),
            "clipped_target_credit_s": correct_cpu,
            "constructed_target_s": constructed,
            "credit_fraction": correct_cpu / constructed if constructed else None,
            "excess_cpu_s": excess_cpu,
            "missing_cpu_s": missing_cpu,
            "wrong_clause_cpu_s": wrong_clause,
            "target_clause_attributed_cpu_s": target_attributed,
            "non_target_clause_attributed_cpu_s": non_target_attributed,
            "unattributed_record_cpu_s": unattributed,
            "accounted_cpu_s": accounted,
            "raw_exit_cpu_s": raw_exit,
            "runtime_housekeeping_cpu_s": housekeeping,
            "reconciliation_per_repetition": reconciliation_rows,
            "attributed_to_cgroup_ratios": ratios,
        },
        "G4": {
            "pass": (
                len(pairs) == 30
                and bool(sequential_errors)
                and statistics.mean(sequential_errors) * 1000
                <= gates["G4"]["sequential_wall_mae_ms_max"]
                and len(pipeline_structure_rows) == 30
                and all(row["pass"] for row in pipeline_structure_rows)
            ),
            "sequential_mae_ms": (
                statistics.mean(sequential_errors) * 1000
                if sequential_errors
                else None
            ),
            "pipeline_process_mae_ms_reported_only": (
                statistics.mean(pipeline_errors) * 1000
                if pipeline_errors
                else None
            ),
            "interpreter_startup_ms_mean": (
                statistics.mean(startup_ms_all) if startup_ms_all else None
            ),
            "interpreter_startup_ms_median": (
                statistics.median(startup_ms_all) if startup_ms_all else None
            ),
            "pipeline_structure_rows": pipeline_structure_rows,
        },
        "G6": {
            "pass": (
                len(pairs) == 30
                and abs(statistics.median(overhead_ms))
                < gates["G6"]["absolute_median_overhead_ms_max"]
            )
            if overhead_ms
            else False,
            "relative_ci_note": (
                "relative median bootstrap CI95 upper is REPORTED ONLY under "
                "amendment B-5; docker-per-rep protocol cannot establish <1% "
                "at n=30"
            ),
            "median_ms": statistics.median(overhead_ms) if overhead_ms else None,
            "p95_ms": percentile(overhead_ms, 0.95) if overhead_ms else None,
            "relative_median": statistics.median(relative) if relative else None,
            "relative_median_bootstrap_ci95": (
                [percentile(boot, 0.025), percentile(boot, 0.975)]
                if boot
                else None
            ),
        },
    }


def evaluate_other(
    fixture: dict[str, Any], case_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates = [
        row["candidate"] for row in rows if not row["candidate"].get("error")
    ]
    if case_id == "W":
        expected = fixture["evaluator"]["W"]
        exact_exec_events = fixture["evaluator"]["gates"]["GW"][
            "exact_exec_events"
        ]
        checks = []
        for candidate in candidates:
            records = sorted(
                candidate["process_records"], key=lambda row: row["t_exec_ns"]
            )
            complete = len(records) == exact_exec_events
            check = {
                "exact_exec_count": complete,
                "images_in_order": [
                    PurePath(row["argv"][0]).name for row in records
                ]
                == expected["expected_exec_argv0"],
                "chain_single_pid": complete
                and len({row["host_pid"] for row in records[1:]}) == 1,
                "chain_exec_indexes_consecutive": complete
                and [row["host_exec_index"] for row in records[1:]]
                in ([0, 1, 2], [1, 2, 3]),
                "intermediate_cpu_zero": complete
                and records[1]["cpu_ns"] == 0
                and records[2]["cpu_ns"] == 0,
                "terminal_cpu_positive": complete
                and records[-1]["cpu_ns"] > 0,
            }
            check["pass"] = all(check.values())
            check["entrypoint_shell_cpu_ns"] = (
                records[0]["cpu_ns"] if complete else None
            )
            checks.append(check)
        return {
            "GW": {
                "pass": len(rows) == 10
                and all(check["pass"] for check in checks),
                "checks": checks,
            }
        }
    if case_id == "M":
        lower, upper = fixture["evaluator"]["M"]["peak_rss_band_kb"]
        checks = []
        for candidate in candidates:
            peak = max(
                (
                    row["peak_rss_kb"] or 0
                    for row in candidate["process_records"]
                ),
                default=0,
            )
            live = candidate["cgroup"]["live_vmhwm_kb"]
            checks.append(
                {
                    "peak_rss_kb": peak,
                    "live_vmhwm_kb": live,
                    "in_band": lower <= peak <= upper,
                    "relative_error": (
                        abs(peak - live) / live if live else math.inf
                    ),
                    "marker_present": candidate["marker_present"],
                }
            )
        return {
            "GM": {
                "pass": len(rows) == 20 and all(
                    row["in_band"]
                    and row["relative_error"] <= 0.05
                    and row["marker_present"]
                    for row in checks
                ),
                "band_kb": [lower, upper],
                "checks": checks,
            }
        }
    if case_id == "K":
        checks = []
        for candidate in candidates:
            children = [
                row
                for row in candidate["process_records"]
                if row["argv"] and PurePath(row["argv"][0]).name == "python3"
            ]
            checks.append(
                len(children) == 1
                and children[0]["signal"] == 9
                and children[0]["cpu_ns"] > 0
                and children[0]["lineage_parent_host_pid"] is not None
            )
        return {"GK": {"pass": len(rows) == 10 and all(checks), "checks": checks}}
    checks = []
    for candidate in candidates:
        children = [
            row
            for row in candidate["process_records"]
            if row["argv"] and PurePath(row["argv"][0]).name == "python3"
        ]
        exits = {
            event["host_pid"]: event["timestamp_ns"]
            for event in candidate["raw_events"]
            if event["type"] == "exit"
        }
        checks.append(
            len(children) == 1
            and children[0]["t_exit_ns"] is not None
            and children[0]["lineage_parent_host_pid"] in exits
            and exits[children[0]["lineage_parent_host_pid"]]
            < children[0]["t_exit_ns"]
        )
    return {"GB": {"pass": len(rows) == 10 and all(checks), "checks": checks}}


def summary_markdown(results: dict[str, Any], fixture: dict[str, Any]) -> str:
    stopped = results.get("stop")
    lines = [
        "# Stage 1b eBPF clause telemetry",
        "",
        f"Lane E ran with **ringbuf** and the BPF reserve-failure counter. Outcome: **{results['outcome']}**.",
        "",
    ]
    if stopped:
        lines.extend(
            [
                f"The frozen sequence stopped in Case {stopped['case']} repetition {stopped['repetition']} because G5 marked the repetition partial: {', '.join(stopped['reasons'])}. No top-up or later case ran.",
                "",
            ]
        )
    lines.extend(["## Gates", ""])
    for gate, value in results.get("gates", {}).items():
        lines.append(f"- {gate}: **{'PASS' if value.get('pass') else 'FAIL'}** — `{json.dumps(value, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            f"- GM was frozen as `[65536, 90668]` KiB from 65536 KiB allocation + measured B={fixture['evaluator']['M']['baseline_peak_rss_kb']} KiB + 16384 KiB slack.",
            "- Oracle output was used only for presence/absence comparison and never joined into mappings.",
            "- This result covers only the frozen synthetic workloads. It makes no Terminal-Bench or per-builtin CPU/timing claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    image_id = subprocess.check_output(
        [
            "docker",
            "image",
            "inspect",
            fixture["mechanism"]["image"],
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    if image_id != fixture["mechanism"]["image_id"]:
        raise RuntimeError(f"image id mismatch: {image_id}")
    results: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "fixture_sha256": sha256(FIXTURE),
        "collector_sha256": sha256(COLLECTOR),
        "harness_sha256": sha256(Path(__file__)),
        "mapper_sha256": sha256(ARTIFACT_DIR / "mapper.py"),
        "image_id": image_id,
        "case_order": fixture["mechanism"]["case_order"],
        "cases": {},
        "gates": {},
        "outcome": "RUNNING",
    }
    write_json(RESULTS, results)

    for case_id in fixture["mechanism"]["case_order"]:
        variants = ["retained", "freed_sync"] if case_id == "M" else [None]
        repetitions = (
            fixture["mechanism"]["cases"][case_id]["repetitions_per_variant"]
            if case_id == "M"
            else fixture["mechanism"]["cases"][case_id]["repetitions"]
        )
        def execute_case() -> list[dict[str, Any]]:
            case_rows: list[dict[str, Any]] = []
            results["cases"][case_id] = {"rows": case_rows}
            for variant in variants:
                payload_path, payload = mechanism_payload(
                    fixture, case_id, variant
                )
                for rep in range(repetitions):
                    name = f"fable-stage1bconf-{case_id.lower()}-{rep:02d}"
                    cidfile = SCRATCH / f"{case_id}-{rep}.cid"
                    row: dict[str, Any] = {"repetition": rep, "variant": variant}
                    lane_order = (
                        ["control", "candidate"]
                        if case_id == "P" and rep % 2 == 0
                        else ["candidate", "control"]
                        if case_id == "P"
                        else ["candidate"]
                    )
                    row["lane_order"] = lane_order
                    for lane in lane_order:
                        if lane == "control":
                            row["control"] = run_control(payload, name, cidfile)
                        else:
                            row["candidate"] = run_candidate(
                                payload_path,
                                payload,
                                case_id,
                                rep,
                                name,
                                cidfile,
                            )
                            row["integrity"] = assess_integrity(
                                fixture, case_id, row["candidate"]
                            )
                    case_rows.append(row)
                    write_json(RESULTS, results)
            return case_rows

        first_attempt = None
        for attempt in range(2):
            case_rows = execute_case()
            if first_attempt is not None:
                results["cases"][case_id]["first_attempt_mechanism_failure"] = (
                    first_attempt
                )
            failures = sorted(
                {
                    reason
                    for row in case_rows
                    for reason in row["integrity"]["reasons"]
                }
                & MECHANISM_FAILURE_REASONS
            )
            if failures and first_attempt is None:
                first_attempt = {"reasons": failures, "rows": case_rows}
                continue
            if failures:
                results["cases"][case_id]["mechanism_failure_after_retry"] = (
                    failures
                )
            break
        update_g5(results, fixture, case_id, case_rows)
        oracle = run_oracle(fixture, case_id, variants[0])
        lane_e_row = next(
            row
            for row in case_rows
            if row["variant"] == variants[0]
            and not row["candidate"].get("error")
        )
        oracle["lane_e_comparison"] = compare_oracle(
            lane_e_row["candidate"], oracle
        )
        oracle["lane_e_repetition"] = lane_e_row["repetition"]
        oracle["lane_e_variant"] = lane_e_row["variant"]
        results["cases"][case_id]["oracle"] = oracle
        results["gates"].update(
            evaluate_p(fixture, case_rows)
            if case_id == "P"
            else evaluate_other(fixture, case_id, case_rows)
        )
        write_json(RESULTS, results)

    results["outcome"] = (
        "PASS" if all(gate["pass"] for gate in results["gates"].values()) else "FAIL"
    )
    results["completed_at_utc"] = utc_now()
    write_json(RESULTS, results)
    SUMMARY.write_text(summary_markdown(results, fixture), encoding="utf-8")
    return 0 if results["outcome"] == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        mapper_self_check()
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mechanism, evaluator = fixture["mechanism"], fixture["evaluator"]
        assert "evaluator" not in mechanism
        assert evaluator["bootstrap_seed"] == 20260725
        assert evaluator["bootstrap_draws"] == 2000
        assert evaluator["M"]["peak_rss_band_kb"] == [65536, 90668]
        p_expected = evaluator["P"]
        assert len(p_expected["expected_resolved"]) == 9
        assert p_expected["expected_unresolved"]["count"] == 3
        assert [
            group["reason"]
            for group in p_expected["expected_unresolved"]["groups"]
        ] == ["multiple_static_candidates", "no_eligible_static_candidate"]
        assert p_expected["constructed_cpu_s_per_repetition"] == 0.34
        assert evaluator["gates"]["GW"]["exact_exec_events"] == 4
        assert [
            evaluator[case]["expected_exec_events"] for case in ("M", "K", "B")
        ] == [1, 3, 4]
        assert fixture["mechanism"]["case_order"] == ["B"]
        assert fixture["mechanism"]["cases"]["B"]["repetitions"] == 10
        assert fixture["stage1b_amendments"][-1]["id"] == "B-7"
        amended_match = compare_semantics(
            {
                "stdout_hex": b'X\nENV={"A":"1","HOSTNAME":"control"}\nY\n'.hex(),
                "stderr_hex": b"".hex(),
                "exit_code": 0,
            },
            {
                "stdout_hex": b'X\nENV={"A":"1","HOSTNAME":"candidate"}\nY\n'.hex(),
                "stderr_hex": b"".hex(),
                "exit_code": 0,
            },
        )
        assert amended_match["pass"]
        assert amended_match["child_environment"]["hostname_present_both"]
        assert not compare_semantics(
            {
                "stdout_hex": b'X\nENV={"A":"1","HOSTNAME":"control"}\nY\n'.hex(),
                "stderr_hex": b"".hex(),
                "exit_code": 0,
            },
            {
                "stdout_hex": b'X\nENV={"A":"2","HOSTNAME":"candidate"}\nY\n'.hex(),
                "stderr_hex": b"".hex(),
                "exit_code": 0,
            },
        )["pass"]
        w_forked = [
            {
                "host_pid": 41,
                "exec_seq": 14,
                "host_exec_index": 0,
                "t_exec_ns": 0,
                "argv": ["/bin/sh"],
                "cpu_ns": 16_000_000,
            }
        ] + [
            {
                "host_pid": 42,
                "exec_seq": 15 + index,
                "host_exec_index": index,
                "t_exec_ns": 1 + index,
                "argv": [name],
                "cpu_ns": 0 if index < 2 else 1,
            }
            for index, name in enumerate(["env", "nice", "python3"])
        ]
        w_gate = evaluate_other(
            fixture,
            "W",
            [{"candidate": {"process_records": w_forked}}] * 10,
        )["GW"]
        assert w_gate["pass"]
        assert w_gate["checks"][0]["entrypoint_shell_cpu_ns"] == 16_000_000
        w_tail = [
            {
                "host_pid": 42,
                "exec_seq": 14 + index,
                "host_exec_index": index,
                "t_exec_ns": index,
                "argv": [name],
                "cpu_ns": 0 if index < 3 else 1,
            }
            for index, name in enumerate(
                ["/bin/sh", "env", "nice", "python3"]
            )
        ]
        assert evaluate_other(
            fixture, "W", [{"candidate": {"process_records": w_tail}}] * 10
        )["GW"]["pass"]
        w_bad = [dict(row) for row in w_forked]
        w_bad[2]["cpu_ns"] = 5
        assert not evaluate_other(
            fixture, "W", [{"candidate": {"process_records": w_bad}}] * 10
        )["GW"]["pass"]
        cpu_candidate = {
            "exec_events": [
                {
                    "host_pid": 1,
                    "exec_seq": 15,
                    "t_exec_ns": 1,
                    "argv": mechanism["cases"]["P"]["static_clauses"][2]["argv"],
                },
                {
                    "host_pid": 2,
                    "exec_seq": 16,
                    "t_exec_ns": 2,
                    "argv": mechanism["cases"]["P"]["static_clauses"][1]["argv"],
                },
                {
                    "host_pid": 3,
                    "exec_seq": 17,
                    "t_exec_ns": 3,
                    "argv": mechanism["cases"]["P"]["static_clauses"][6]["argv"],
                },
            ],
            "process_records": [
                {"host_pid": 1, "exec_seq": 15, "cpu_ns": 130_000_000},
                {"host_pid": 2, "exec_seq": 16, "cpu_ns": 1_000_000},
                {"host_pid": 3, "exec_seq": 17, "cpu_ns": 1_000_000},
            ],
            "raw_events": [
                {
                    "type": "exit",
                    "cgroup_id": 7,
                    "exec_seq": seq,
                    "utime_ns": cpu_ns,
                    "stime_ns": 0,
                }
                for seq, cpu_ns in [
                    (15, 130_000_000),
                    (16, 1_000_000),
                    (17, 1_000_000),
                    (SENTINEL_SEQ, 2_000_000),
                ]
            ],
            "cgroup": {"cgroup_id": 7, "cpu_usage_usec": 200_000},
        }
        cpu_candidate["mapping"] = {
            "resolved": [
                {
                    "clause_id": "c2:49-144",
                    "occurrence_index": 0,
                    "process_record": cpu_candidate["process_records"][0],
                },
                {
                    "clause_id": "c1:37-47",
                    "occurrence_index": 0,
                    "process_record": cpu_candidate["process_records"][1],
                },
            ],
            "unresolved": [
                {"process_record": cpu_candidate["process_records"][2]}
            ],
        }
        accounting = cpu_accounting(fixture, cpu_candidate)
        assert accounting["runtime_housekeeping_cpu_s"] == 0.002
        assert not accounting["raw_exit_equals_accounted"]
        assert accounting["raw_exit_equals_accounted_plus_housekeeping"]
        assert accounting["attributed_cpu_s"] == 0.131
        assert accounting["attributed_to_cgroup_ratio"] == 0.655
        assert accounting["unattributed_record_cpu_s"] == 0.001
        assert accounting["wrong_clause_cpu_s"] == 0
        dropped = cpu_candidate["mapping"]["unresolved"].pop()
        assert not cpu_accounting(fixture, cpu_candidate)[
            "raw_exit_equals_accounted_plus_housekeeping"
        ]
        cpu_candidate["mapping"]["unresolved"].append(dropped)
        cpu_candidate["mapping"]["resolved"].append(
            cpu_candidate["mapping"]["resolved"][0]
        )
        assert not cpu_accounting(fixture, cpu_candidate)[
            "raw_exit_equals_accounted_plus_housekeeping"
        ]
        cpu_candidate["mapping"]["resolved"].pop()
        cpu_candidate["mapping"]["resolved"][0]["clause_id"] = "c3:147-242"
        assert cpu_accounting(fixture, cpu_candidate)["wrong_clause_cpu_s"] == 0.13
        oracle_comparison = compare_oracle(
            {
                "exec_events": [
                    {
                        "host_pid": 1,
                        "exec_seq": 15,
                        "argv": ["/usr/bin/python3"],
                    }
                ]
            },
            {"exec_presence": ["env", "python3"]},
        )
        assert oracle_comparison["all_lane_e_argv0_present"]
        assert oracle_comparison["mapping_unchanged"]
        g5_results: dict[str, Any] = {"gates": {}}
        for case_id in mechanism["case_order"]:
            repetitions = (
                mechanism["cases"][case_id]["repetitions_per_variant"] * 2
                if case_id == "M"
                else mechanism["cases"][case_id]["repetitions"]
            )
            rows = [
                {
                    "repetition": index % 10,
                    "variant": (
                        "retained"
                        if case_id == "M" and index < 10
                        else "freed_sync"
                        if case_id == "M"
                        else None
                    ),
                    "integrity": {"pass": True, "reasons": []},
                }
                for index in range(repetitions)
            ]
            update_g5(g5_results, fixture, case_id, rows)
        assert g5_results["gates"]["G5"]["pass"]
        assert set(g5_results["gates"]["G5"]["cases"]) == set(
            mechanism["case_order"]
        )
        print("harness self-check passed")
        return
    raise SystemExit(run())


if __name__ == "__main__":
    main()
