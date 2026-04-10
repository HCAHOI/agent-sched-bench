"""Host-side helpers for task-container agent parity."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOTNAME = "_task_container_runtime"
REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
_REDACTED_SECRET = "***REDACTED***"


@dataclass(slots=True)
class TaskContainerPreflightProof:
    hostname: str
    cwd: str
    python_executable: str
    python_prefix: str
    project_root: str
    sys_path: list[str]
    container_id: str | None = None


@dataclass(slots=True)
class TaskContainerRunResult:
    success: bool
    exit_status: str | None
    model_patch: str
    error: str | None
    n_iterations: int | None
    total_llm_ms: float | None
    total_tool_ms: float | None
    total_tokens: int | None
    runtime_proof: dict[str, Any]
    trace_path: Path
    raw_stdout_path: Path
    raw_stderr_path: Path


def current_container_python_runtime() -> str:
    """Return the Python interpreter path that the container should use."""
    return str(REPO_VENV_PYTHON)


def task_container_runtime_dir(attempt_dir: Path, scaffold: str) -> Path:
    return attempt_dir.resolve() / RUNTIME_ROOTNAME / scaffold


def project_mount_args(attempt_dir: Path) -> list[str]:
    """Return extra `podman run` args for parity mode."""

    task_container_runtime_dir(attempt_dir, "bootstrap").mkdir(parents=True, exist_ok=True)
    repo_root = REPO_ROOT.resolve()
    attempt_dir = attempt_dir.resolve()
    args: list[str] = []
    mounts: list[tuple[Path, bool]] = [
        (attempt_dir, False),
        (repo_root, False),
    ]
    for raw in ("/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin", "/tmp", "/var"):
        path = Path(raw)
        if path.exists():
            mounts.append((path, raw not in {"/tmp", "/var"}))

    seen: set[Path] = set()
    for path, read_only in mounts:
        if path in seen:
            continue
        seen.add(path)
        suffix = ":ro" if read_only else ""
        args.extend(["-v", f"{path}:{path}{suffix}"])
    return args


def write_task_container_request(
    *,
    attempt_dir: Path,
    scaffold: str,
    payload: dict[str, Any],
) -> Path:
    path = task_container_runtime_dir(attempt_dir, scaffold) / "request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_redact_request_payload(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _redact_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (_REDACTED_SECRET if key == "api_key" else _redact(child))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [_redact(item) for item in value]
        return value

    return _redact(payload)


def exec_task_container_entrypoint(
    *,
    container_id: str,
    request_path: Path,
    request_payload: dict[str, Any] | None = None,
    runtime: str,
    timeout: float,
    executable: str = "podman",
    cwd: str = "/testbed",
) -> subprocess.CompletedProcess[str]:
    request = request_payload or json.loads(request_path.read_text(encoding="utf-8"))
    kind = str(request.get("kind") or "")
    if not kind:
        raise ValueError(f"missing request kind in {request_path}")
    mode = "preflight" if kind == "preflight" else "run"
    return subprocess.run(
        [
            executable,
            "exec",
            "-i",
            "-w",
            cwd,
            "-e",
            f"PYTHONPATH={REPO_ROOT / 'src'}:{REPO_ROOT}",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            container_id,
            runtime,
            "-m",
            "trace_collect.runtime.entrypoint",
            "--mode",
            mode,
        ],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def read_task_container_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight_task_container_runtime(
    *,
    container_id: str,
    attempt_dir: Path,
    executable: str = "podman",
) -> TaskContainerPreflightProof:
    runtime = current_container_python_runtime()
    runtime_dir = task_container_runtime_dir(attempt_dir, "preflight")
    request_payload = {
        "kind": "preflight",
        "result_path": str(runtime_dir / "result.json"),
        "imports": [
            "trace_collect.runtime.entrypoint",
            "agents.miniswe.agent",
            "agents.openclaw.eval.runner",
            "harness.trace_logger",
        ],
        "writable_probe": str(runtime_dir / "writable.probe"),
        "container_id": container_id,
    }
    request_path = write_task_container_request(
        attempt_dir=attempt_dir,
        scaffold="preflight",
        payload=request_payload,
    )
    result = exec_task_container_entrypoint(
        container_id=container_id,
        request_path=request_path,
        request_payload=request_payload,
        runtime=runtime,
        timeout=120,
        executable=executable,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "task-container preflight failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    payload = read_task_container_result(runtime_dir / "result.json")
    proof = payload.get("runtime_proof") or {}
    return TaskContainerPreflightProof(**proof)


def run_task_container_agent(
    *,
    container_id: str,
    request: dict[str, Any],
    timeout: float,
    executable: str = "podman",
) -> TaskContainerRunResult:
    runtime = current_container_python_runtime()
    raw_stdout_path = Path(request["raw_stdout_path"])
    raw_stderr_path = Path(request["raw_stderr_path"])
    raw_stdout_path.parent.mkdir(parents=True, exist_ok=True)
    raw_stderr_path.parent.mkdir(parents=True, exist_ok=True)
    request_path = write_task_container_request(
        attempt_dir=Path(request["result_path"]).parents[2],
        scaffold=request["scaffold"],
        payload=request,
    )
    try:
        result = exec_task_container_entrypoint(
            container_id=container_id,
            request_path=request_path,
            request_payload=request,
            runtime=runtime,
            timeout=timeout,
            executable=executable,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout_path.write_text(
            (exc.stdout or exc.output or "") if isinstance((exc.stdout or exc.output or ""), str)
            else ((exc.stdout or exc.output or b"").decode("utf-8", errors="replace")),
            encoding="utf-8",
        )
        raw_stderr_path.write_text(
            exc.stderr if isinstance(exc.stderr or "", str) else (exc.stderr or b"").decode("utf-8", errors="replace"),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"task-container run timed out after {timeout}s"
        ) from exc
    except Exception as exc:
        if not raw_stdout_path.exists():
            raw_stdout_path.write_text("", encoding="utf-8")
        raw_stderr_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        raise
    if not raw_stdout_path.exists():
        raw_stdout_path.write_text(result.stdout, encoding="utf-8")
    if not raw_stderr_path.exists():
        raw_stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "task-container run failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    payload = read_task_container_result(Path(request["result_path"]))
    success = payload.get("success")
    if success is None:
        success = bool(payload.get("model_patch"))
    return TaskContainerRunResult(
        success=bool(success),
        exit_status=payload.get("exit_status"),
        model_patch=payload.get("model_patch", "") or "",
        error=payload.get("error"),
        n_iterations=payload.get("n_iterations"),
        total_llm_ms=payload.get("total_llm_ms"),
        total_tool_ms=payload.get("total_tool_ms"),
        total_tokens=payload.get("total_tokens"),
        runtime_proof=payload.get("runtime_proof") or {},
        trace_path=Path(payload.get("trace_path") or request.get("trace_file") or ""),
        raw_stdout_path=raw_stdout_path,
        raw_stderr_path=raw_stderr_path,
    )
