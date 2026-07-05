#!/usr/bin/env python3
"""Replay parity gate: compare DockerBackend vs FCBackend mismatch rates.

Loads one trace JSONL file, replays each tool_exec action through both
the Docker sandbox backend and the Firecracker sandbox backend, then
compares the per-tool outputs against the source using the tiered
mismatch oracle.

Exit criterion: the FC mismatch ratio (fc_any / docker_any) must be
<= 1.5, i.e. FC produces at most 50 % more mismatches than Docker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_FC_KERNEL_PATH = Path(
    os.environ.get("FC_KERNEL_PATH", "/tmp/fc-cache/vmlinux-5.10.225"),
)

# Must import after PYTHONPATH is set up.
_IMPORT_GUARD: dict[str, Exception | None] = {}


def _import_deps() -> tuple[
    Any,  # DockerBackend
    Any,  # FCBackend
    Any,  # FakeBackend
    Any,  # AgentTransportRequest
    Any,  # MismatchOracle
    Any,  # transport_response_from_agent_dict
    Any,  # normalize_image_reference
    Any,  # ensure_fixed_image
    Any,  # fixed_image_name_for
    Any,  # remove_image
    Any,  # start_task_container
    Any,  # stop_task_container
    Any,  # configure_task_container_apt_mirror
]:
    """Lazily import project modules (requires ``PYTHONPATH=src``)."""
    from agents.sandbox_runtime import (
        AgentTransportRequest,
        DockerBackend,
        FCBackend,
        FakeBackend,
        transport_response_from_agent_dict,
    )
    from trace_collect.mismatch import MismatchOracle
    from harness.container_image_prep import (
        ensure_fixed_image,
        fixed_image_name_for,
        normalize_image_reference,
        remove_image,
    )
    from trace_collect.attempt_pipeline import (
        configure_task_container_apt_mirror,
        start_task_container,
        stop_task_container,
    )

    return (
        DockerBackend,
        FCBackend,
        FakeBackend,
        AgentTransportRequest,
        MismatchOracle,
        transport_response_from_agent_dict,
        normalize_image_reference,
        ensure_fixed_image,
        fixed_image_name_for,
        remove_image,
        start_task_container,
        stop_task_container,
        configure_task_container_apt_mirror,
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ParityRow:
    tool_index: int
    tool_name: str
    source_result: str | None
    docker_result: str | None
    docker_ok: bool | None
    docker_mismatch: bool | None
    fc_result: str | None
    fc_ok: bool | None
    fc_mismatch: bool | None
    error: str | None


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_trace_actions(trace_path: Path) -> list[dict[str, Any]]:
    """Return every ``tool_exec`` action record from a trace JSONL file."""
    actions: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("type") == "action" and record.get("action_type") == "tool_exec":
                actions.append(record)
    return actions


# ---------------------------------------------------------------------------
# Docker replay
# ---------------------------------------------------------------------------


async def replay_with_docker(
    actions: list[dict[str, Any]],
    source_image: str,
    task_output_dir: Path,
    container_executable: str,
    *,
    agent_id: str = "parity-docker",
) -> list[ParityRow]:
    """Replay every action through DockerBackend, return per-tool rows."""
    (
        DockerBackend,
        _FC,
        _Fake,
        AgentTransportRequest,
        MismatchOracle,
        transport_response_from_agent_dict,
        normalize_image_reference,
        ensure_fixed_image,
        fixed_image_name_for,
        remove_image,
        start_task_container,
        stop_task_container,
        configure_task_container_apt_mirror,
    ) = _import_deps()

    normalized = normalize_image_reference(source_image)
    fixed_name = fixed_image_name_for(
        source_image=normalized,
        agent_id=agent_id,
        task_output_dir=task_output_dir,
    )

    backend = DockerBackend(
        source_image=normalized,
        fixed_image_name=fixed_name,
        agent_id=agent_id,
        source_agent_id="parity-source",
        manifest_index=0,
        task_output_dir=task_output_dir,
        container_executable=container_executable,
        network_mode="host",
        fixed_images_by_source={},
        bootstrap_mount_args=(),
        agent_env_kwargs={},
        startup_recorder=None,
        ensure_fixed_image_fn=ensure_fixed_image,
        start_task_container_fn=start_task_container,
        configure_apt_mirror_fn=configure_task_container_apt_mirror,
        stop_task_container_fn=stop_task_container,
        remove_image_fn=remove_image,
    )

    await backend.start()

    rows: list[ParityRow] = []
    try:
        for idx, action in enumerate(actions):
            data = action.get("data") or {}
            tool_name = str(data.get("tool_name", ""))
            tool_args = data.get("tool_args") or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            # Only replay exec / commands / read_file / write_file / edit_file /
            # list_dir — skip spawn / message / web_search / web_fetch
            if tool_name in {"spawn", "message", "web_search", "web_fetch"}:
                rows.append(
                    ParityRow(
                        tool_index=idx,
                        tool_name=tool_name,
                        source_result=_source_output(action),
                        docker_result=None,
                        docker_ok=None,
                        docker_mismatch=None,
                        fc_result=None,
                        fc_ok=None,
                        fc_mismatch=None,
                        error=None,
                    )
                )
                continue

            tool_timeout_s = _resolve_command_timeout(data)

            # Build request args from the tool call params.
            request_args: dict[str, Any] = {}
            if tool_name in ("exec", "commands"):
                request_args = dict(tool_args)
            elif tool_name == "read_file":
                request_args = {"path": tool_args.get("path", "")}
            elif tool_name == "write_file":
                request_args = {
                    "path": tool_args.get("path", ""),
                    "content": tool_args.get("content", ""),
                }
            elif tool_name == "edit_file":
                request_args = {
                    "path": tool_args.get("path", ""),
                    "old_text": tool_args.get("old_text", ""),
                    "new_text": tool_args.get("new_text", ""),
                    "replace_all": bool(tool_args.get("replace_all", False)),
                }
            elif tool_name == "list_dir":
                request_args = {"path": tool_args.get("path", ".")}

            source_output = _source_output(action)
            error: str | None = None
            docker_ok: bool | None = None
            docker_mismatch: bool | None = None
            docker_result: str | None = None

            try:
                response = await backend.execute(
                    AgentTransportRequest(
                        tool=tool_name,
                        args=request_args,
                    ),
                    timeout_s=tool_timeout_s,
                )
                docker_result = response.result
                docker_ok = response.ok
                resp_dict = transport_response_from_agent_dict.__wrapped__(  # type: ignore[attr-defined]
                    {
                        "ok": response.ok,
                        "result": response.result,
                        "returncode": response.returncode,
                        "timed_out": response.timed_out,
                    }
                ) if hasattr(transport_response_from_agent_dict, "__wrapped__") else (
                    {
                        "ok": response.ok,
                        "result": response.result,
                        "returncode": response.returncode,
                        "timed_out": response.timed_out,
                    }
                )
                docker_mismatch = _is_docker_mismatch(
                    action, resp_dict,
                )
            except Exception as exc:
                error = f"docker: {exc}"
                docker_ok = False

            rows.append(
                ParityRow(
                    tool_index=idx,
                    tool_name=tool_name,
                    source_result=source_output,
                    docker_result=docker_result,
                    docker_ok=docker_ok,
                    docker_mismatch=docker_mismatch,
                    fc_result=None,
                    fc_ok=None,
                    fc_mismatch=None,
                    error=error,
                )
            )
    finally:
        await backend.stop()

    return rows


def _is_docker_mismatch(
    action: dict[str, Any],
    replay_response: dict[str, Any],
) -> bool | None:
    """Classify whether the Docker replay output diverges from the source."""
    data = action.get("data") or {}
    source_output_data = data.get("tool_output") or {}
    if isinstance(source_output_data, str):
        source_output_data = {"result": source_output_data}

    source_result = str(source_output_data.get("result", ""))
    replay_result = str(replay_response.get("result", ""))

    # Simple comparison: if results differ in normalized form, it's a mismatch.
    try:
        from trace_collect.output_normalize import normalize_tool_output

        source_norm = normalize_tool_output(source_result)
        replay_norm = normalize_tool_output(replay_result)
        if source_norm == replay_norm:
            return False
        return True
    except Exception:
        return source_result != replay_result


def _source_output(action: dict[str, Any]) -> str | None:
    data = action.get("data") or {}
    output = data.get("tool_output") or {}
    if isinstance(output, dict):
        return str(output.get("result", ""))
    return str(output) if output else None


def _resolve_command_timeout(data: dict[str, Any]) -> float:
    """Extract the timeout from source tool data, with a 600 s floor."""
    tool_args = data.get("tool_args") or {}
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            tool_args = {}
    if isinstance(tool_args, dict):
        timeout = tool_args.get("timeout", 0)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 0.0
    else:
        timeout = 0.0
    return max(600.0, timeout)


# ---------------------------------------------------------------------------
# FC replay
# ---------------------------------------------------------------------------


async def replay_with_fc(
    actions: list[dict[str, Any]],
    source_image: str,
    task_output_dir: Path,
    docker_rows: list[ParityRow],
    container_executable: str = "docker",
) -> list[ParityRow]:
    """Replay every action through FCBackend, merging into existing rows."""
    (
        _Docker,
        FCBackend,
        _Fake,
        AgentTransportRequest,
        MismatchOracle,
        _transport,
        normalize_image_reference,
        _ensure_fixed,
        _fixed_name,
        _remove,
        _start,
        _stop,
        _apt,
    ) = _import_deps()

    normalized = normalize_image_reference(source_image)

    backend = FCBackend(
        source_image=normalized,
        kernel_path=_FC_KERNEL_PATH,
        checkpoint_dir=task_output_dir / "checkpoints-fc",
        container_executable=container_executable,
    )

    await backend.start()

    rows = list(docker_rows)  # shallow copy
    try:
        for idx, action in enumerate(actions):
            data = action.get("data") or {}
            tool_name = str(data.get("tool_name", ""))
            tool_args = data.get("tool_args") or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            if tool_name in {"spawn", "message", "web_search", "web_fetch"}:
                continue

            tool_timeout_s = _resolve_command_timeout(data)

            request_args: dict[str, Any] = {}
            if tool_name in ("exec", "commands"):
                request_args = dict(tool_args)
            elif tool_name == "read_file":
                request_args = {"path": tool_args.get("path", "")}
            elif tool_name == "write_file":
                request_args = {
                    "path": tool_args.get("path", ""),
                    "content": tool_args.get("content", ""),
                }
            elif tool_name == "edit_file":
                request_args = {
                    "path": tool_args.get("path", ""),
                    "old_text": tool_args.get("old_text", ""),
                    "new_text": tool_args.get("new_text", ""),
                    "replace_all": bool(tool_args.get("replace_all", False)),
                }
            elif tool_name == "list_dir":
                request_args = {"path": tool_args.get("path", ".")}

            source_output = _source_output(action)
            fc_ok: bool | None = None
            fc_mismatch: bool | None = None
            fc_result: str | None = None
            error = rows[idx].error or ""

            try:
                response = await backend.execute(
                    AgentTransportRequest(
                        tool=tool_name,
                        args=request_args,
                    ),
                    timeout_s=tool_timeout_s,
                )
                fc_result = response.result
                fc_ok = response.ok

                if source_output is not None:
                    try:
                        from trace_collect.output_normalize import (
                            normalize_tool_output,
                        )

                        source_norm = normalize_tool_output(source_output)
                        fc_norm = normalize_tool_output(fc_result or "")
                        fc_mismatch = source_norm != fc_norm
                    except Exception:
                        fc_mismatch = source_output != fc_result
                else:
                    fc_mismatch = not fc_ok

            except Exception as exc:
                error = f"{error}; fc: {exc}".strip("; ")
                fc_ok = False
                fc_mismatch = True

            rows[idx] = ParityRow(
                tool_index=idx,
                tool_name=tool_name,
                source_result=source_output,
                docker_result=rows[idx].docker_result,
                docker_ok=rows[idx].docker_ok,
                docker_mismatch=rows[idx].docker_mismatch,
                fc_result=fc_result,
                fc_ok=fc_ok,
                fc_mismatch=fc_mismatch,
                error=error if error else None,
            )
    finally:
        await backend.stop()

    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def compute_report(rows: list[ParityRow]) -> dict[str, Any]:
    """Aggregate mismatch statistics."""
    docker_mismatches = sum(
        1 for r in rows if r.docker_mismatch is True
    )
    fc_mismatches = sum(
        1 for r in rows if r.fc_mismatch is True
    )
    total = len(rows)
    skipped = sum(
        1 for r in rows
        if r.tool_name in {"spawn", "message", "web_search", "web_fetch"}
    )
    active = total - skipped
    docker_error = sum(1 for r in rows if r.error and "docker:" in (r.error or ""))
    fc_error = sum(1 for r in rows if r.error and "fc:" in (r.error or ""))

    docker_rate = docker_mismatches / active if active > 0 else 0.0
    fc_rate = fc_mismatches / active if active > 0 else 0.0
    ratio = fc_mismatches / docker_mismatches if docker_mismatches > 0 else (
        1.0 if fc_mismatches == docker_mismatches else float("inf")
    )

    return {
        "total_tools": total,
        "skipped_spawn_message_etc": skipped,
        "active_tools": active,
        "docker_mismatches": docker_mismatches,
        "docker_error": docker_error,
        "docker_mismatch_rate": round(docker_rate, 4),
        "fc_mismatches": fc_mismatches,
        "fc_error": fc_error,
        "fc_mismatch_rate": round(fc_rate, 4),
        "fc_vs_docker_ratio": round(ratio, 4) if ratio != float("inf") else "inf",
        "parity_pass": _parity_pass(ratio),
    }


def _parity_pass(ratio: float) -> bool:
    if ratio == float("inf"):
        return False
    return ratio <= 1.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parity gate: compare Docker vs FC mismatch rates",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        required=True,
        help="Path to a trace JSONL file to replay",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="Override the Docker image (auto-detected from trace metadata otherwise)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/parity-gate"),
        help="Working directory for containers and checkpoints",
    )
    parser.add_argument(
        "--container-executable",
        type=str,
        default="docker",
        help="Container runtime executable (docker or podman)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check environment without replaying",
    )
    args = parser.parse_args()

    if not args.trace_jsonl.exists():
        print(f"ERROR: {args.trace_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    # Resolve source image.
    docker_image = args.docker_image
    if docker_image is None:
        docker_image = _detect_image_from_trace(args.trace_jsonl)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        actions = load_trace_actions(args.trace_jsonl)
        print(f"[DRY-RUN] {args.trace_jsonl}: {len(actions)} tool_exec actions")
        print(f"  source image: {docker_image}")
        _print_env_check()
        return

    actions = load_trace_actions(args.trace_jsonl)
    if not actions:
        print("No tool_exec actions found in trace", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(actions)} tool_exec actions from {args.trace_jsonl}")
    print(f"Source Docker image: {docker_image}")

    async def run() -> None:
        t0 = time.monotonic()

        # Phase 1: Docker replay.
        print("\n--- Phase 1: Docker replay ---")
        docker_rows = await replay_with_docker(
            actions=actions,
            source_image=docker_image,
            task_output_dir=args.output_dir / "docker",
            container_executable=args.container_executable,
        )

        # Phase 2: FC replay.
        print("\n--- Phase 2: FC replay ---")
        rows = await replay_with_fc(
            actions=actions,
            source_image=docker_image,
            task_output_dir=args.output_dir / "fc",
            docker_rows=docker_rows,
            container_executable=args.container_executable,
        )

        elapsed = time.monotonic() - t0
        report = compute_report(rows)

        print(f"\n--- Parity report ({elapsed:.0f}s) ---")
        for key, value in report.items():
            print(f"  {key}: {value}")

        print("\nExit criterion: fc_vs_docker_ratio <= 1.5")
        if report["parity_pass"]:
            print("RESULT: PARITY PASS")
            sys.exit(0)
        else:
            print("RESULT: PARITY FAIL")
            sys.exit(1)

    asyncio.run(run())


def _detect_image_from_trace(trace_path: Path) -> str:
    """Heuristic: extract Docker image from trace metadata."""
    with trace_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("type") == "metadata":
                meta = record.get("data") or {}
                image = meta.get("docker_image") or meta.get("image") or ""
                if image:
                    return str(image)
            if record.get("type") == "task_meta":
                data = record.get("data") or {}
                image = data.get("docker_image") or data.get("image") or ""
                if image:
                    return str(image)
    raise SystemExit(
        "Could not detect Docker image from trace metadata. "
        "Use --docker-image to specify one."
    )


def _print_env_check() -> None:
    """Quick environment sanity check."""
    issues: list[str] = []
    if not Path("/dev/kvm").exists():
        issues.append("/dev/kvm missing — FC replay will fail")
    for tool in ("docker", "firecracker", "sudo"):
        proc = os.system(f"which {tool} >/dev/null 2>&1")
        if proc != 0:
            issues.append(f"{tool} not on PATH")
    if issues:
        print("Environment issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Environment OK")


if __name__ == "__main__":
    main()
