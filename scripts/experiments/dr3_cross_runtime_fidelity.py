#!/usr/bin/env python3
"""DR3: Cross-runtime fidelity floor experiment.

Hypothesis: docker-collected traces replayed inside a Firecracker microVM
produce mismatch rates within 1.5x of docker replay.

The experiment reads a simulate-trace JSONL (output of ``trace_collect.cli
simulate``), extracts tool_exec commands with their expected source behavior,
and replays each command in two environments: Docker (podman exec) and a
Firecracker microVM (SSH).  Per-action comparisons yield mismatch rates,
a cross-runtime inflation ratio (FC mismatches / docker mismatches), and
per-cause breakdowns (timeout, exit code, CAS).

Output: CSV with columns trace_file, action_id, tool_name, command, docker_mismatch_reason,
fc_mismatch_reason, fc_mismatch_is_extra (bool), mismatch_cause, …
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trace_collect.output_normalize import normalize_tool_output
from trace_collect.simulator import _exec_semantics_payload
from trace_collect.simulator import _command_exit_code
from trace_collect.simulator import _tool_uses_exec_semantics
from trace_collect.simulator import _tool_result_indicates_wrapper_timeout


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SSH_TIMEOUT = 60
_DOCKER_EXEC_TIMEOUT = 60
_DRY_RUN_COMMAND_OUTPUT = "[DRY-RUN] command output"


# ---------------------------------------------------------------------------
# SSH helpers (mirrors DR2 pattern)
# ---------------------------------------------------------------------------


def ssh(
    host: str,
    command: str,
    *,
    port: int = 22,
    user: str = "root",
    identity_file: str | None = None,
    timeout: float = _SSH_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a remote host via the system ``ssh`` binary."""
    cmd: list[str] = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout={int(timeout)}",
        "-o", "ServerAliveInterval=5",
        "-p", str(port),
    ]
    if identity_file:
        cmd.extend(["-i", identity_file])
    cmd.extend([f"{user}@{host}", command])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 5,
        check=False,
    )


# ---------------------------------------------------------------------------
# Docker replay helpers
# ---------------------------------------------------------------------------


def docker_exec(
    container_id: str,
    command: str,
    *,
    container_executable: str = "podman",
    cwd: str = "/testbed",
    timeout: float = _DOCKER_EXEC_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Execute a command inside a Docker/Podman container."""
    cmd = [
        container_executable,
        "exec",
        "-i",
        "-w", cwd,
        container_id,
        "/bin/sh", "-c", command,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Trace reading helpers
# ---------------------------------------------------------------------------


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, returning all records."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _simulate_tool_execs(
    simulate_trace: Path,
) -> list[dict[str, Any]]:
    """Extract tool_exec records from a simulate-trace JSONL."""
    records = _parse_jsonl(simulate_trace)
    return [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]


def _source_trace_map(simulate_trace: Path) -> dict[str, Path]:
    """Build a map of source_action_id → source_trace_path from simulate metadata."""
    records = _parse_jsonl(simulate_trace)
    for record in records:
        if record.get("type") != "trace_metadata":
            continue
        entries = record.get("source_trace_entries", [])
        if entries:
            return {
                e["source_agent_id"]: Path(e["source_trace"])
                for e in entries
                if "source_trace" in e and "source_agent_id" in e
            }
        raw = record.get("source_traces", [])
        if raw:
            return {Path(p).name: Path(p) for p in raw if isinstance(p, str)}
    return {}


def _source_trace_for_record(
    record: dict[str, Any],
    source_trace_map: dict[str, Path],
) -> Path | None:
    """Find the source trace path for a given simulate tool_exec record."""
    agent_id = record.get("agent_id", "")
    if agent_id in source_trace_map:
        return source_trace_map[agent_id]
    return None


def _find_source_action(
    source_trace: Path,
    source_action_id: str,
) -> dict[str, Any] | None:
    """Find the source tool_exec record by source_action_id."""
    if not source_trace.is_file():
        return None
    for line in source_trace.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("action_id") == source_action_id:
            return rec
    return None


# ---------------------------------------------------------------------------
# Command extraction
# ---------------------------------------------------------------------------


def _extract_command(
    tool_name: str | None,
    tool_args_json: str,
) -> str | None:
    """Extract the shell command string from tool_args."""
    payload = _exec_semantics_payload(tool_name, tool_args_json)
    if payload is None:
        return None
    command = payload.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()
    commands = payload.get("commands")
    if isinstance(commands, list) and commands:
        return " && ".join(c for c in commands if isinstance(c, str) and c.strip())
    return None


def _tool_result_exit_code(tool_result_text: str, returncode: Any = None) -> int | None:
    """Parse the exit code from a tool result string."""
    return _command_exit_code(tool_result_text, returncode)


def _tool_result_timed_out(tool_result_text: str, timed_out: Any = None) -> bool:
    """Check if the tool result indicates a timeout."""
    return _tool_result_indicates_wrapper_timeout(tool_result_text, timed_out)


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ReplayResult:
    """Result of replaying a command in a specific runtime."""

    runtime: str  # "docker" or "firecracker"
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    elapsed_ms: float
    error: str | None = None


@dataclass(slots=True)
class ActionComparison:
    """Comparison of a single action across source, docker, and FC."""

    action_id: str
    tool_name: str
    command: str
    source_trace: str
    source_result: str
    source_returncode: int | None
    source_timed_out: bool
    docker_mismatch_reason: str | None  # From simulate trace (docker vs source)
    docker_result: ReplayResult | None = None  # Fresh docker replay
    fc_result: ReplayResult | None = None  # FC replay
    fc_mismatch_reason: str | None = None
    fc_mismatch_is_extra: bool | None = None  # FC mismatch when docker matched
    mismatch_cause: str | None = None  # Primary cause category


def _classify_mismatch(
    source_result: str,
    replay_result: ReplayResult,
    source_returncode: int | None,
    source_timed_out: bool,
) -> str | None:
    """Classify mismatch between source and replay result.

    Returns a mismatch reason string or None if matched.
    """
    if replay_result.error is not None:
        return f"replay_error:{replay_result.error}"

    # Timeout comparison.
    if source_timed_out != replay_result.timed_out:
        return "timeout_mismatch"

    # Exit code comparison.
    source_exit = _command_exit_code(source_result, source_returncode)
    replay_exit = replay_result.returncode

    if source_exit is not None and source_exit != replay_exit:
        return "command_exit_code_mismatch"

    # Normalized output comparison.
    source_normalized = normalize_tool_output(source_result)
    replay_normalized = normalize_tool_output(replay_result.stdout)
    if source_normalized != replay_normalized:
        return "normalized_output_mismatch"

    return None


def _compare_action(
    *,
    sim_record: dict[str, Any],
    source_record: dict[str, Any] | None,
    docker_result: ReplayResult | None,
    fc_result: ReplayResult | None,
) -> ActionComparison:
    """Build a per-action comparison."""
    data = sim_record.get("data") or {}
    tool_name = str(data.get("tool_name", ""))
    tool_args = str(data.get("tool_args", "{}"))
    action_id = str(data.get("source_action_id", sim_record.get("action_id", "")))
    source_trace = str(data.get("simulate_source", ""))

    command = _extract_command(tool_name, tool_args) or ""

    # Source expected behavior.
    if source_record is not None:
        src_data = source_record.get("data") or {}
        source_result = str(src_data.get("tool_result", src_data.get("result", "")))
        source_returncode_raw = src_data.get("returncode")
        source_returncode = (
            int(source_returncode_raw)
            if isinstance(source_returncode_raw, int)
            and not isinstance(source_returncode_raw, bool)
            else None
        )
        source_timed_out_raw = src_data.get("timed_out")
        source_timed_out = (
            bool(source_timed_out_raw)
            if isinstance(source_timed_out_raw, bool)
            else False
        )
    else:
        source_result = ""
        source_returncode = data.get("source_returncode")
        source_timed_out = data.get("source_timed_out", False)

    # Docker mismatch from simulate trace.
    sim_mismatch_reason = data.get("mismatch_reason")

    # FC mismatch classification.
    fc_mismatch_reason: str | None = None
    if fc_result is not None and fc_result.error is None:
        fc_mismatch_reason = _classify_mismatch(
            source_result=source_result,
            replay_result=fc_result,
            source_returncode=source_returncode,
            source_timed_out=source_timed_out,
        )
    elif fc_result is not None and fc_result.error is not None:
        fc_mismatch_reason = f"replay_error:{fc_result.error}"

    # Determine if FC mismatch is "extra" (docker matched but FC didn't).
    docker_matched = sim_mismatch_reason is None
    fc_matched = fc_mismatch_reason is None
    fc_mismatch_is_extra = docker_matched and not fc_matched

    # Primary cause.
    if fc_mismatch_reason is not None:
        mismatch_cause = fc_mismatch_reason.split(":")[0]
    elif sim_mismatch_reason is not None:
        mismatch_cause = sim_mismatch_reason
    else:
        mismatch_cause = None

    return ActionComparison(
        action_id=action_id,
        tool_name=tool_name,
        command=command,
        source_trace=source_trace,
        source_result=source_result,
        source_returncode=source_returncode,
        source_timed_out=source_timed_out,
        docker_mismatch_reason=sim_mismatch_reason,
        docker_result=docker_result,
        fc_result=fc_result,
        fc_mismatch_reason=fc_mismatch_reason,
        fc_mismatch_is_extra=fc_mismatch_is_extra,
        mismatch_cause=mismatch_cause,
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


CSV_FIELDNAMES = [
    "source_trace",
    "action_id",
    "tool_name",
    "command",
    "source_returncode",
    "source_timed_out",
    "docker_mismatch_reason",
    "docker_returncode",
    "docker_elapsed_ms",
    "fc_mismatch_reason",
    "fc_returncode",
    "fc_elapsed_ms",
    "fc_mismatch_is_extra",
    "mismatch_cause",
]


@dataclass(slots=True)
class ExperimentConfig:
    """All tunables for a DR3 run."""

    # Input.
    simulate_trace: Path

    # Docker configuration.
    container_id: str = ""
    container_executable: str = "podman"
    container_cwd: str = "/testbed"

    # Firecracker / SSH configuration.
    fc_host: str = ""
    fc_port: int = 22
    fc_user: str = "root"
    fc_identity_file: str | None = None

    # Experiment control.
    dry_run: bool = False
    max_actions: int | None = None
    skip_docker_replay: bool = False
    command_timeout: float = 60.0
    output: Path | None = None

    # Internal state.
    _comparisons: list[ActionComparison] = field(default_factory=list)


def _run_experiment(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Run the full DR3 experiment and return rows for the CSV."""
    if config.dry_run:
        return _run_dry(config)

    sim_trace = config.simulate_trace
    if not sim_trace.is_file():
        raise SystemExit(f"simulate-trace not found: {sim_trace}")

    print(f"Reading simulate trace: {sim_trace}", file=sys.stderr)
    tool_execs = _simulate_tool_execs(sim_trace)
    total = len(tool_execs)
    print(f"  Found {total} tool_exec records", file=sys.stderr)

    print("Building source trace map...", file=sys.stderr)
    src_map = _source_trace_map(sim_trace)
    print(f"  Mapped {len(src_map)} source traces", file=sys.stderr)

    comparisons: list[ActionComparison] = []
    rows: list[dict[str, Any]] = []

    limit = config.max_actions or total
    for i, sim_rec in enumerate(tool_execs[:limit]):
        data = sim_rec.get("data") or {}
        tool_name = str(data.get("tool_name", ""))
        tool_args = str(data.get("tool_args", "{}"))

        # Skip non-exec actions.
        if not _tool_uses_exec_semantics(tool_name, tool_args):
            continue

        command = _extract_command(tool_name, tool_args)
        if not command:
            continue

        source_action_id = str(data.get("source_action_id", ""))
        source_trace_path = _source_trace_for_record(sim_rec, src_map)
        source_record = None
        if source_trace_path is not None and source_action_id:
            source_record = _find_source_action(source_trace_path, source_action_id)

        # --- Docker replay ---
        docker_result: ReplayResult | None = None
        if not config.skip_docker_replay and config.container_id:
            docker_result = _replay_in_docker(
                config=config,
                command=command,
            )
        else:
            # Use simulate trace's existing docker mismatch reason as proxy.
            pass

        # --- Firecracker replay ---
        fc_result: ReplayResult | None = None
        if config.fc_host:
            fc_result = _replay_in_firecracker(
                config=config,
                command=command,
            )

        comparison = _compare_action(
            sim_record=sim_rec,
            source_record=source_record,
            docker_result=docker_result,
            fc_result=fc_result,
        )
        comparisons.append(comparison)

        row = _comparison_to_row(comparison)
        rows.append(row)

        # Progress.
        if (i + 1) % 10 == 0:
            _print_progress(i + 1, limit, comparisons)

    config._comparisons = comparisons
    return rows


def _replay_in_docker(
    config: ExperimentConfig,
    command: str,
) -> ReplayResult:
    """Replay a command inside a Docker container."""
    t_start = time.monotonic()
    try:
        result = docker_exec(
            container_id=config.container_id,
            command=command,
            container_executable=config.container_executable,
            cwd=config.container_cwd,
            timeout=config.command_timeout,
        )
        elapsed_ms = (time.monotonic() - t_start) * 1000
        timed_out = _tool_result_indicates_wrapper_timeout(result.stdout)
        return ReplayResult(
            runtime="docker",
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.monotonic() - t_start) * 1000
        return ReplayResult(
            runtime="docker",
            stdout=exc.output if isinstance(exc.output, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            returncode=-1,
            timed_out=True,
            elapsed_ms=elapsed_ms,
            error="timeout",
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t_start) * 1000
        return ReplayResult(
            runtime="docker",
            stdout="",
            stderr="",
            returncode=-1,
            timed_out=False,
            elapsed_ms=elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def _replay_in_firecracker(
    config: ExperimentConfig,
    command: str,
) -> ReplayResult:
    """Replay a command inside a Firecracker microVM via SSH."""
    t_start = time.monotonic()
    try:
        result = ssh(
            host=config.fc_host,
            command=command,
            port=config.fc_port,
            user=config.fc_user,
            identity_file=config.fc_identity_file,
            timeout=config.command_timeout,
        )
        elapsed_ms = (time.monotonic() - t_start) * 1000
        timed_out = _tool_result_indicates_wrapper_timeout(result.stdout)
        return ReplayResult(
            runtime="firecracker",
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.monotonic() - t_start) * 1000
        return ReplayResult(
            runtime="firecracker",
            stdout=exc.output if isinstance(exc.output, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            returncode=-1,
            timed_out=True,
            elapsed_ms=elapsed_ms,
            error="timeout",
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t_start) * 1000
        return ReplayResult(
            runtime="firecracker",
            stdout="",
            stderr="",
            returncode=-1,
            timed_out=False,
            elapsed_ms=elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def _comparison_to_row(comparison: ActionComparison) -> dict[str, Any]:
    """Convert an ActionComparison to a CSV row dict."""
    docker_rc = comparison.docker_result.returncode if comparison.docker_result else None
    docker_elapsed = (
        comparison.docker_result.elapsed_ms if comparison.docker_result else None
    )
    fc_rc = comparison.fc_result.returncode if comparison.fc_result else None
    fc_elapsed = (
        comparison.fc_result.elapsed_ms if comparison.fc_result else None
    )

    return {
        "source_trace": comparison.source_trace,
        "action_id": comparison.action_id,
        "tool_name": comparison.tool_name,
        "command": comparison.command,
        "source_returncode": comparison.source_returncode,
        "source_timed_out": comparison.source_timed_out,
        "docker_mismatch_reason": comparison.docker_mismatch_reason or "",
        "docker_returncode": docker_rc,
        "docker_elapsed_ms": docker_elapsed,
        "fc_mismatch_reason": comparison.fc_mismatch_reason or "",
        "fc_returncode": fc_rc,
        "fc_elapsed_ms": fc_elapsed,
        "fc_mismatch_is_extra": comparison.fc_mismatch_is_extra,
        "mismatch_cause": comparison.mismatch_cause or "",
    }


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


def _run_dry(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Simulate the experiment flow without executing real system commands."""
    print("=== DR3 DRY RUN ===", file=sys.stderr)
    print(f"simulate_trace: {config.simulate_trace}", file=sys.stderr)
    print(f"container_id: {config.container_id}", file=sys.stderr)
    print(f"container_executable: {config.container_executable}", file=sys.stderr)
    print(f"container_cwd: {config.container_cwd}", file=sys.stderr)
    print(f"fc_host: {config.fc_host}", file=sys.stderr)
    print(f"fc_port: {config.fc_port}", file=sys.stderr)
    print(f"fc_user: {config.fc_user}", file=sys.stderr)
    print(f"fc_identity_file: {config.fc_identity_file}", file=sys.stderr)
    print(f"max_actions: {config.max_actions}", file=sys.stderr)
    print(f"skip_docker_replay: {config.skip_docker_replay}", file=sys.stderr)
    print(f"command_timeout: {config.command_timeout}s", file=sys.stderr)

    sim_trace = config.simulate_trace
    rows: list[dict[str, Any]] = []

    if not sim_trace.is_file():
        print(f"  simulate-trace file not found (dry-run): {sim_trace}", file=sys.stderr)
        return rows

    tool_execs = _simulate_tool_execs(sim_trace)
    limit = config.max_actions or len(tool_execs)
    count = 0
    for sim_rec in tool_execs[:limit]:
        data = sim_rec.get("data") or {}
        tool_name = str(data.get("tool_name", ""))
        tool_args = str(data.get("tool_args", "{}"))
        if not _tool_uses_exec_semantics(tool_name, tool_args):
            continue
        command = _extract_command(tool_name, tool_args)
        if not command:
            continue
        count += 1
        if count <= 3:
            print(
                f"  [DRY-RUN] action={data.get('source_action_id', '?')} "
                f"tool={tool_name} cmd={command[:80]}...",
                file=sys.stderr,
            )
        rows.append(
            {
                "source_trace": str(data.get("simulate_source", "")),
                "action_id": str(data.get("source_action_id", "")),
                "tool_name": tool_name,
                "command": command,
                "source_returncode": data.get("source_returncode"),
                "source_timed_out": data.get("source_timed_out", False),
                "docker_mismatch_reason": data.get("mismatch_reason", ""),
                "docker_returncode": data.get("replay_returncode", data.get("returncode")),
                "docker_elapsed_ms": data.get("duration_ms"),
                "fc_mismatch_reason": "-",
                "fc_returncode": None,
                "fc_elapsed_ms": None,
                "fc_mismatch_is_extra": None,
                "mismatch_cause": data.get("mismatch_reason", ""),
            }
        )

    print(f"  Eligible exec actions: {count}", file=sys.stderr)
    print(f"=== DRY-RUN complete: {len(rows)} actions ===", file=sys.stderr)
    return rows


def _print_progress(
    completed: int,
    total: int,
    comparisons: list[ActionComparison],
) -> None:
    """Print progress to stderr."""
    docker_mismatches = sum(
        1 for c in comparisons if c.docker_mismatch_reason is not None
    )
    fc_mismatches = sum(
        1 for c in comparisons if c.fc_mismatch_reason is not None
    )
    extra = sum(1 for c in comparisons if c.fc_mismatch_is_extra)
    print(
        f"  [{completed}/{total}] docker_mm={docker_mismatches} "
        f"fc_mm={fc_mismatches} extra={extra}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(
    comparisons: list[ActionComparison],
    rows: list[dict[str, Any]],
) -> None:
    """Print the DR3 experiment report to stderr."""
    total = len(comparisons)
    if total == 0:
        print("No comparisons to report.", file=sys.stderr)
        return

    docker_mismatched = [
        c for c in comparisons if c.docker_mismatch_reason is not None
    ]
    fc_mismatched = [
        c for c in comparisons if c.fc_mismatch_reason is not None
    ]
    extra_fc = [
        c for c in comparisons if c.fc_mismatch_is_extra
    ]

    docker_mm_count = len(docker_mismatched)
    fc_mm_count = len(fc_mismatched)

    docker_mm_rate = docker_mm_count / total if total else 0
    fc_mm_rate = fc_mm_count / total if total else 0

    # Inflation ratio: FC mismatches / docker mismatches.
    # Avoid division by zero: if docker has zero mismatches, inflation is
    # undefined (or infinite if FC has any).
    if docker_mm_count > 0:
        inflation = fc_mm_count / docker_mm_count
    elif fc_mm_count > 0:
        inflation = float("inf")
    else:
        inflation = 1.0

    # Per-cause breakdown for FC mismatches.
    fc_cause_counts: Counter[str] = Counter()
    for c in fc_mismatched:
        cause = c.fc_mismatch_reason or "unknown"
        fc_cause_counts[cause] += 1

    # Per-cause breakdown for docker mismatches.
    docker_cause_counts: Counter[str] = Counter()
    for c in docker_mismatched:
        cause = c.docker_mismatch_reason or "unknown"
        docker_cause_counts[cause] += 1

    # Structural causes for extra FC mismatches.
    extra_cause_counts: Counter[str] = Counter()
    for c in extra_fc:
        cause = c.fc_mismatch_reason or "unknown"
        extra_cause_counts[cause] += 1

    print("=" * 70, file=sys.stderr)
    print("DR3: CROSS-RUNTIME FIDELITY FLOOR REPORT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total exec actions compared: {total}", file=sys.stderr)
    print(file=sys.stderr)

    print("--- Mismatch Rates ---", file=sys.stderr)
    print(
        f"  Docker mismatches:  {docker_mm_count}/{total} = {docker_mm_rate:.3f} "
        f"({docker_mm_rate*100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  FC mismatches:      {fc_mm_count}/{total} = {fc_mm_rate:.3f} "
        f"({fc_mm_rate*100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  Inflation ratio:    {inflation:.3f} (FC / docker)",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    print("--- Docker Mismatch Causes ---", file=sys.stderr)
    if docker_cause_counts:
        for cause, count in docker_cause_counts.most_common():
            print(f"  {cause}: {count}", file=sys.stderr)
    else:
        print("  (none)", file=sys.stderr)
    print(file=sys.stderr)

    print("--- FC Mismatch Causes ---", file=sys.stderr)
    if fc_cause_counts:
        for cause, count in fc_cause_counts.most_common():
            print(f"  {cause}: {count}", file=sys.stderr)
    else:
        print("  (none)", file=sys.stderr)
    print(file=sys.stderr)

    if extra_cause_counts:
        print("--- Extra FC Mismatch Causes (docker matched, FC did not) ---", file=sys.stderr)
        for cause, count in extra_cause_counts.most_common():
            print(f"  {cause}: {count}", file=sys.stderr)
        print(file=sys.stderr)

    # Structural analysis of extra mismatches.
    if extra_fc:
        print("--- Structural Cause List for Extra FC Mismatches ---", file=sys.stderr)
        for i, c in enumerate(extra_fc[:10]):
            cmd_preview = c.command[:100] if c.command else "(no command)"
            print(
                f"  [{i+1}] action={c.action_id} tool={c.tool_name} "
                f"cause={c.fc_mismatch_reason} cmd={cmd_preview}",
                file=sys.stderr,
            )
        if len(extra_fc) > 10:
            print(f"  ... and {len(extra_fc) - 10} more", file=sys.stderr)
        print(file=sys.stderr)

    # Hypothesis check.
    print("--- Hypothesis Check ---", file=sys.stderr)
    target = 1.5
    if inflation <= target:
        print(
            f"  PASS: inflation {inflation:.3f} <= {target}",
            file=sys.stderr,
        )
    else:
        print(
            f"  FAIL: inflation {inflation:.3f} > {target}",
            file=sys.stderr,
        )
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DR3: Cross-runtime fidelity floor experiment"
    )
    parser.add_argument(
        "--simulate-trace",
        type=Path,
        required=True,
        help="Path to simulate-trace JSONL file",
    )
    # Docker configuration.
    parser.add_argument(
        "--container-id",
        default="",
        help="Docker/Podman container ID for replay (required for real run)",
    )
    parser.add_argument(
        "--container-executable",
        default="podman",
        help="Container runtime executable (default: podman)",
    )
    parser.add_argument(
        "--container-cwd",
        default="/testbed",
        help="Working directory inside container (default: /testbed)",
    )
    # Firecracker / SSH configuration.
    parser.add_argument(
        "--fc-host",
        default="",
        help="Firecracker microVM host for SSH-based replay (required for real run)",
    )
    parser.add_argument(
        "--fc-port",
        type=int,
        default=22,
        help="SSH port on Firecracker guest (default: 22)",
    )
    parser.add_argument(
        "--fc-user",
        default="root",
        help="SSH user for Firecracker guest (default: root)",
    )
    parser.add_argument(
        "--fc-identity-file",
        default=None,
        help="SSH private key for Firecracker access",
    )
    # Experiment control.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without executing system commands",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Limit number of actions to replay (default: all)",
    )
    parser.add_argument(
        "--skip-docker-replay",
        action="store_true",
        help="Skip fresh docker replay; use simulate-trace docker results only",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=60.0,
        help="Timeout per command execution in seconds (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write CSV to file instead of stdout",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress the summary report on stderr",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        simulate_trace=args.simulate_trace,
        container_id=args.container_id,
        container_executable=args.container_executable,
        container_cwd=args.container_cwd,
        fc_host=args.fc_host,
        fc_port=args.fc_port,
        fc_user=args.fc_user,
        fc_identity_file=args.fc_identity_file,
        dry_run=args.dry_run,
        max_actions=args.max_actions,
        skip_docker_replay=args.skip_docker_replay,
        command_timeout=args.command_timeout,
        output=args.output,
    )

    if not config.dry_run:
        if not config.container_id and not config.skip_docker_replay:
            print(
                "WARNING: --container-id not set and --skip-docker-replay not specified; "
                "docker replay will be skipped.",
                file=sys.stderr,
            )
        if not config.fc_host:
            print(
                "WARNING: --fc-host not set; Firecracker replay will be skipped.",
                file=sys.stderr,
            )

    rows = _run_experiment(config)

    # Write CSV.
    out_fh = config.output.open("w", encoding="utf-8") if config.output else sys.stdout
    writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    if config.output:
        out_fh.close()
        print(f"Wrote {len(rows)} rows -> {config.output}", file=sys.stderr)
    else:
        print(
            f"# Wrote {len(rows)} exec action comparisons",
            file=sys.stderr,
        )

    # Print report.
    if not args.no_report and config._comparisons:
        _print_report(config._comparisons, rows)


if __name__ == "__main__":
    main()
