"""Host-side helpers for task-container agent parity."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import ssl
import subprocess
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO

from agents.openclaw.runtime_deps import OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOTNAME = "_task_container_runtime"
_REDACTED_SECRET = "***REDACTED***"
_DEFAULT_RUNTIME_PYTHONPATH = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
_CONTAINER_SYSTEM_PYTHON = "/usr/bin/python3"
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple"
_SHARED_BOOTSTRAP_CACHE = Path.home() / ".cache" / "task-container-bootstrap"
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
_GET_PIP_FETCH_ATTEMPTS = 3
_GET_PIP_FETCH_BACKOFF_SECONDS = 1.0
_ARCH_ALIASES = {
    "amd64": "amd64",
    "x86_64": "amd64",
    "arm64": "arm64",
    "aarch64": "arm64",
}
_CONTAINER_PYTHON_CANDIDATES = (
    "/usr/bin/python3",
    "/usr/bin/python",
    "/opt/miniconda3/bin/python3",
    "/opt/miniconda3/bin/python",
    "/opt/conda/bin/python3",
    "/opt/conda/bin/python",
    "python3",
    "python",
)
_BOOTSTRAP_PIP_RESOLUTION_ENV_KEYS = (
    "TASK_CONTAINER_PIP_EXTRA_INDEX_URL",
    "TASK_CONTAINER_PIP_TRUSTED_HOST",
    "TASK_CONTAINER_PIP_CERT",
    "TASK_CONTAINER_SSL_CERT_FILE",
)


def _format_probe_failure_details(result: subprocess.CompletedProcess[str]) -> str:
    parts: list[str] = []
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stdout:
        parts.append(f"stdout: {stdout}")
    if stderr:
        parts.append(f"stderr: {stderr}")
    return "; ".join(parts)


def _resolve_userbase_site_packages(userbase: Path) -> Path | None:
    """Return the first Python userbase site-packages directory, if present."""
    for lib_dir in ("lib", "lib64"):
        lib_path = userbase / lib_dir
        if not lib_path.is_dir():
            continue
        for py_dir in sorted(lib_path.iterdir(), reverse=True):
            if not py_dir.is_dir() or not py_dir.name.startswith("python"):
                continue
            site_packages = py_dir / "site-packages"
            if site_packages.is_dir():
                return site_packages
    return None


def _list_bootstrap_packages(site_dir: Path) -> set[str]:
    """Return installed package identifiers from ``*.dist-info`` directories."""
    packages: set[str] = set()
    try:
        entries = list(site_dir.iterdir())
    except OSError:
        return packages
    for entry in entries:
        if not entry.is_dir() or not entry.name.endswith(".dist-info"):
            continue
        stem = entry.name[: -len(".dist-info")]
        if "-" in stem:
            package_name, version = stem.rsplit("-", 1)
            packages.add(f"{package_name}=={version}")
        else:
            packages.add(stem)
    return packages


def _bootstrap_marker_matches(
    marker: Path,
    *,
    requirements: tuple[str, ...],
    runtime: str,
    pip_index_url: str,
    arch: str,
    image_platform: str | None,
    python_fingerprint: dict[str, str],
    pip_resolution_fingerprint: dict[str, object],
    cache_key: str,
    site_dir: Path | None = None,
    userbase_dir: Path | None = None,
) -> bool:
    """Return true when a shared bootstrap cache is safe to reuse."""
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    expected = {
        "requirements": list(requirements),
        "python": runtime,
        "pip_index_url": pip_index_url,
        "arch": arch,
        "image_platform": image_platform,
        "python_fingerprint": python_fingerprint,
        "pip_resolution_fingerprint": pip_resolution_fingerprint,
        "cache_key": cache_key,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False

    for manifest_key, check_dir in (
        ("packages", site_dir),
        ("userbase_packages", userbase_dir),
    ):
        recorded = payload.get(manifest_key)
        if not isinstance(recorded, list):
            return False
        if check_dir is None:
            if recorded:
                return False
            continue
        actual = _list_bootstrap_packages(check_dir)
        if actual != set(recorded):
            return False
    return True


def _is_retryable_get_pip_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False
    retryable = (ssl.SSLError, TimeoutError, ConnectionResetError, OSError)
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, retryable)
    return isinstance(exc, retryable)


def _download_get_pip(get_pip: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, _GET_PIP_FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(_GET_PIP_URL, timeout=120) as response:
                payload = response.read()
            tmp_path = get_pip.with_suffix(".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(get_pip)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= _GET_PIP_FETCH_ATTEMPTS or not _is_retryable_get_pip_error(
                exc
            ):
                raise
            time.sleep(_GET_PIP_FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    if last_error is not None:
        raise last_error


def _remove_tree_if_exists(path: Path) -> None:
    """Remove *path* if present; fail fast on real cleanup errors."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


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


@dataclass(slots=True, frozen=True)
class TaskContainerExecConfig:
    runtime: str
    pythonpath: str
    start_extra_args: tuple[str, ...]
    bootstrap: bool = False
    bootstrap_site_dir: Path | None = None
    image_platform: str | None = None


def _normalize_arch(raw: str | None) -> str | None:
    if raw is None:
        return None
    return _ARCH_ALIASES.get(raw.lower(), raw.lower())


def _host_linux_platform() -> str | None:
    if platform.system() != "Linux":
        return None
    arch = _normalize_arch(platform.machine())
    if not arch:
        return None
    return f"linux/{arch}"


def _shared_bootstrap_dir(image_platform: str | None) -> Path:
    platform_slug = image_platform or _host_linux_platform() or "unknown"
    return _SHARED_BOOTSTRAP_CACHE / platform_slug.replace("/", "-")


@contextmanager
def _bootstrap_lock() -> Iterator[None]:
    """Serialize writes to the shared task-container bootstrap cache."""
    _SHARED_BOOTSTRAP_CACHE.mkdir(parents=True, exist_ok=True)
    lock_path = _SHARED_BOOTSTRAP_CACHE / ".bootstrap.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _bootstrap_arch(exec_config: TaskContainerExecConfig) -> str:
    if exec_config.image_platform:
        parts = exec_config.image_platform.split("/", 1)
        if len(parts) == 2:
            return _normalize_arch(parts[1]) or parts[1].lower()
    return _normalize_arch(platform.machine()) or "unknown"


def _container_python_fingerprint(
    *,
    container_id: str,
    runtime: str,
    container_executable: str,
    cwd: str,
) -> dict[str, str]:
    script = r"""
import json
import platform
import sys
import sysconfig

os_release = {}
try:
    with open("/etc/os-release", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.rstrip().split("=", 1)
            if key in {"ID", "VERSION_ID"}:
                os_release[key.lower()] = value.strip('"')
except OSError:
    pass

payload = {
    "version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "cache_tag": sys.implementation.cache_tag or "",
    "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX") or "",
    "machine": platform.machine(),
    "libc": " ".join(part for part in platform.libc_ver() if part),
    "os_id": os_release.get("id", ""),
    "os_version_id": os_release.get("version_id", ""),
}
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "-w",
            cwd,
            container_id,
            runtime,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "task-container python fingerprint failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"task-container python fingerprint failed: invalid JSON {result.stdout!r}"
        ) from exc
    return {str(key): str(value) for key, value in payload.items()}


def _bootstrap_pip_resolution_fingerprint(pip_index_url: str) -> dict[str, object]:
    payload = {
        "pip_index_url": pip_index_url,
        **{key: os.environ.get(key, "") for key in _BOOTSTRAP_PIP_RESOLUTION_ENV_KEYS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "digest": hashlib.sha256(encoded).hexdigest(),
        "present_env": sorted(key for key in payload if payload[key]),
    }


def _bootstrap_cache_key(
    *,
    requirements: tuple[str, ...],
    runtime: str,
    pip_index_url: str,
    arch: str,
    image_platform: str | None,
    python_fingerprint: dict[str, str],
    pip_resolution_fingerprint: dict[str, object],
) -> str:
    payload = {
        "requirements": list(requirements),
        "python": runtime,
        "pip_index_url": pip_index_url,
        "arch": arch,
        "image_platform": image_platform,
        "python_fingerprint": python_fingerprint,
        "pip_resolution_fingerprint": pip_resolution_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _exec_config_with_bootstrap_site_dir(
    exec_config: TaskContainerExecConfig,
    site_dir: Path,
) -> TaskContainerExecConfig:
    return TaskContainerExecConfig(
        runtime=exec_config.runtime,
        pythonpath=f"{site_dir}:{_DEFAULT_RUNTIME_PYTHONPATH}",
        start_extra_args=exec_config.start_extra_args,
        bootstrap=exec_config.bootstrap,
        bootstrap_site_dir=site_dir,
        image_platform=exec_config.image_platform,
    )


def _inspect_image_platform(
    image: str,
    *,
    container_executable: str,
) -> str | None:
    if not image:
        return None
    result = subprocess.run(
        [
            container_executable,
            "image",
            "inspect",
            image,
            "--format",
            "{{.Architecture}} {{.Os}}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return None
    arch, os_name = parts
    norm_arch = _normalize_arch(arch)
    if not norm_arch:
        return None
    return f"{os_name.lower()}/{norm_arch}"


def task_container_runtime_dir(attempt_dir: Path, scaffold: str) -> Path:
    return attempt_dir.resolve() / RUNTIME_ROOTNAME / scaffold


def project_mount_args(attempt_dir: Path) -> list[str]:
    """Return extra `podman run` args mounting the attempt dir + repo root.

    The container runs its own Python via the bootstrap path, so only the
    attempt dir (runtime artifacts + bootstrapped site-dir) and the repo
    (our source) are mounted — no host system dirs.
    """
    task_container_runtime_dir(attempt_dir, "bootstrap").mkdir(
        parents=True, exist_ok=True
    )
    repo_root = REPO_ROOT.resolve()
    attempt_dir = attempt_dir.resolve()
    args: list[str] = []
    seen: set[Path] = set()
    for path in (attempt_dir, repo_root):
        if path in seen:
            continue
        seen.add(path)
        args.extend(["-v", f"{path}:{path}"])
    return args


def resolve_task_container_exec_config(
    *,
    attempt_dir: Path,
    image: str,
    container_executable: str,
) -> TaskContainerExecConfig:
    image_platform = _inspect_image_platform(
        image,
        container_executable=container_executable,
    )
    start_args = list(project_mount_args(attempt_dir))
    if image_platform is not None:
        start_args = ["--platform", image_platform, *start_args]

    site_dir = _shared_bootstrap_dir(image_platform) / "pydeps"
    return TaskContainerExecConfig(
        runtime=_CONTAINER_SYSTEM_PYTHON,
        pythonpath=f"{site_dir}:{_DEFAULT_RUNTIME_PYTHONPATH}",
        start_extra_args=tuple(start_args),
        bootstrap=True,
        bootstrap_site_dir=site_dir,
        image_platform=image_platform,
    )


def resolve_running_container_exec_config(
    *,
    container_id: str,
    exec_config: TaskContainerExecConfig,
    container_executable: str,
    cwd: str = "/testbed",
) -> TaskContainerExecConfig:
    if not exec_config.bootstrap:
        return exec_config

    probe_script = """
set -eu
for cand in "$@"; do
  if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
      "$cand" -c 'import sys; print(sys.executable)'
      exit 0
    fi
  fi
done
exit 1
"""
    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "-w",
            cwd,
            container_id,
            "/bin/sh",
            "-s",
            "--",
            *_CONTAINER_PYTHON_CANDIDATES,
        ],
        input=probe_script,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        details = _format_probe_failure_details(result)
        raise RuntimeError(
            "task-container python probe failed: "
            "no Python >=3.11 interpreter found in container"
            + (f" ({details})" if details else "")
        )
    runtime = result.stdout.strip()
    if not runtime:
        raise RuntimeError("task-container python probe failed: empty interpreter path")
    return TaskContainerExecConfig(
        runtime=runtime,
        pythonpath=exec_config.pythonpath,
        start_extra_args=exec_config.start_extra_args,
        bootstrap=exec_config.bootstrap,
        bootstrap_site_dir=exec_config.bootstrap_site_dir,
        image_platform=exec_config.image_platform,
    )


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


def _collect_pipe(
    pipe: TextIO,
    lines: list[str],
    *,
    live_stream: TextIO | None = None,
) -> None:
    try:
        for line in pipe:
            lines.append(line)
            if live_stream is not None:
                live_stream.write(line)
                live_stream.flush()
    finally:
        pipe.close()


def _run_entrypoint_streaming(
    cmd: list[str],
    *,
    stdin_data: str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_thread = threading.Thread(
        target=_collect_pipe,
        args=(proc.stdout, stdout_lines),
        kwargs={"live_stream": sys.stdout},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_collect_pipe,
        args=(proc.stderr, stderr_lines),
        daemon=True,
    )
    stdin_errors: list[BaseException] = []

    def write_stdin() -> None:
        try:
            proc.stdin.write(stdin_data)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        except BaseException as exc:  # pragma: no cover - defensive pipe cleanup
            stdin_errors.append(exc)
            try:
                proc.stdin.close()
            except BaseException:
                pass

    stdin_thread = threading.Thread(target=write_stdin, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        stdin_thread.join(timeout=1.0)
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output="".join(stdout_lines),
            stderr="".join(stderr_lines),
        ) from exc
    stdin_thread.join(timeout=1.0)
    if stdin_errors:
        raise RuntimeError(f"task-container stdin write failed: {stdin_errors[0]}")
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def exec_task_container_entrypoint(
    *,
    container_id: str,
    request_path: Path,
    request_payload: dict[str, Any] | None = None,
    runtime: str,
    pythonpath: str | None,
    timeout: float,
    container_executable: str,
    cwd: str = "/testbed",
) -> subprocess.CompletedProcess[str]:
    request = request_payload or json.loads(request_path.read_text(encoding="utf-8"))
    kind = str(request.get("kind") or "")
    if not kind:
        raise ValueError(f"missing request kind in {request_path}")
    mode = "preflight" if kind == "preflight" else "run"
    cmd = [
        container_executable,
        "exec",
        "-i",
        "-w",
        cwd,
        "-e",
        f"PYTHONPATH={pythonpath or _DEFAULT_RUNTIME_PYTHONPATH}",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        container_id,
        runtime,
        "-m",
        "trace_collect.runtime.entrypoint",
        "--mode",
        mode,
    ]
    stdin_data = json.dumps(request, ensure_ascii=False)
    if mode == "preflight":
        return subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    return _run_entrypoint_streaming(cmd, stdin_data=stdin_data, timeout=timeout)


def preflight_task_container_runtime(
    *,
    container_id: str,
    attempt_dir: Path,
    imports: list[str] | None = None,
    runtime: str | None = None,
    pythonpath: str | None = None,
    container_executable: str,
) -> TaskContainerPreflightProof:
    effective_runtime = runtime or _CONTAINER_SYSTEM_PYTHON
    runtime_dir = task_container_runtime_dir(attempt_dir, "preflight")
    import_list = imports or [
        "trace_collect.runtime.entrypoint",
        "agents.openclaw.eval.runner",
        "harness.trace_logger",
    ]
    request_payload = {
        "kind": "preflight",
        "result_path": str(runtime_dir / "result.json"),
        "imports": import_list,
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
        runtime=effective_runtime,
        pythonpath=pythonpath,
        timeout=120,
        container_executable=container_executable,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "task-container preflight failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    payload = json.loads((runtime_dir / "result.json").read_text(encoding="utf-8"))
    proof = payload.get("runtime_proof") or {}
    return TaskContainerPreflightProof(**proof)


def run_task_container_agent(
    *,
    container_id: str,
    request: dict[str, Any],
    timeout: float,
    runtime: str | None = None,
    pythonpath: str | None = None,
    container_executable: str,
) -> TaskContainerRunResult:
    effective_runtime = runtime or _CONTAINER_SYSTEM_PYTHON
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
            runtime=effective_runtime,
            pythonpath=pythonpath,
            timeout=timeout,
            container_executable=container_executable,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout_path.write_text(
            (exc.stdout or exc.output or "")
            if isinstance((exc.stdout or exc.output or ""), str)
            else ((exc.stdout or exc.output or b"").decode("utf-8", errors="replace")),
            encoding="utf-8",
        )
        raw_stderr_path.write_text(
            exc.stderr
            if isinstance(exc.stderr or "", str)
            else (exc.stderr or b"").decode("utf-8", errors="replace"),
            encoding="utf-8",
        )
        raise RuntimeError(f"task-container run timed out after {timeout}s") from exc
    except Exception as exc:
        if not raw_stdout_path.exists():
            raw_stdout_path.write_text("", encoding="utf-8")
        raw_stderr_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        raise
    if result.stdout or not raw_stdout_path.exists():
        raw_stdout_path.write_text(result.stdout, encoding="utf-8")
    if result.stderr or not raw_stderr_path.exists():
        raw_stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "task-container run failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    payload = json.loads(Path(request["result_path"]).read_text(encoding="utf-8"))
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


def bootstrap_task_container_python(
    *,
    container_id: str,
    exec_config: TaskContainerExecConfig,
    extra_requirements: tuple[str, ...] = (),
    container_executable: str,
    cwd: str = "/testbed",
) -> TaskContainerExecConfig:
    if not exec_config.bootstrap or exec_config.bootstrap_site_dir is None:
        return exec_config

    arch = _bootstrap_arch(exec_config)
    requirements = tuple(
        dict.fromkeys(OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS + extra_requirements)
    )
    pip_index_url = (
        os.environ.get("TASK_CONTAINER_PIP_INDEX_URL") or _DEFAULT_PIP_INDEX_URL
    )
    python_fingerprint = _container_python_fingerprint(
        container_id=container_id,
        runtime=exec_config.runtime,
        container_executable=container_executable,
        cwd=cwd,
    )
    pip_resolution_fingerprint = _bootstrap_pip_resolution_fingerprint(pip_index_url)
    cache_key = _bootstrap_cache_key(
        requirements=requirements,
        runtime=exec_config.runtime,
        pip_index_url=pip_index_url,
        arch=arch,
        image_platform=exec_config.image_platform,
        python_fingerprint=python_fingerprint,
        pip_resolution_fingerprint=pip_resolution_fingerprint,
    )
    cache_root = _shared_bootstrap_dir(exec_config.image_platform) / cache_key
    current_path = cache_root / "current.json"

    def config_from_generation(generation: str) -> TaskContainerExecConfig | None:
        generation_dir = cache_root / generation
        site_dir = generation_dir / "pydeps"
        userbase = generation_dir / ".pyuserbase"
        marker = site_dir / ".bootstrap-ready.json"
        userbase_site = _resolve_userbase_site_packages(userbase)
        userbase_pip_exists = (userbase / "bin" / "pip").exists() or (
            userbase / "bin" / "pip3"
        ).exists()
        if not userbase_pip_exists:
            return None
        if not _bootstrap_marker_matches(
            marker,
            requirements=requirements,
            runtime=exec_config.runtime,
            pip_index_url=pip_index_url,
            arch=arch,
            image_platform=exec_config.image_platform,
            python_fingerprint=python_fingerprint,
            pip_resolution_fingerprint=pip_resolution_fingerprint,
            cache_key=cache_key,
            site_dir=site_dir,
            userbase_dir=userbase_site,
        ):
            return None
        return _exec_config_with_bootstrap_site_dir(exec_config, site_dir)

    def current_config() -> TaskContainerExecConfig | None:
        try:
            payload = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        generation = payload.get("generation")
        if not isinstance(generation, str) or not generation:
            return None
        return config_from_generation(generation)

    cached = current_config()
    if cached is not None:
        print(
            f"[bootstrap] shared cache hit ({arch}): {cached.bootstrap_site_dir}",
            file=sys.stderr,
            flush=True,
        )
        return cached

    with _bootstrap_lock():
        cached = current_config()
        if cached is not None:
            print(
                f"[bootstrap] shared cache hit ({arch}, after lock): "
                f"{cached.bootstrap_site_dir}",
                file=sys.stderr,
                flush=True,
            )
            return cached

        cache_root.mkdir(parents=True, exist_ok=True)
        generation = f"gen-{time.time_ns()}-{os.getpid()}"
        generation_dir = cache_root / generation
        while generation_dir.exists():
            generation = f"gen-{time.time_ns()}-{os.getpid()}"
            generation_dir = cache_root / generation
        site_dir = generation_dir / "pydeps"
        marker = site_dir / ".bootstrap-ready.json"
        userbase = generation_dir / ".pyuserbase"
        userbase.mkdir(parents=True, exist_ok=True)
        get_pip = userbase / "get-pip.py"
        if not get_pip.exists():
            _download_get_pip(get_pip)

        script = f"""
import json
import os
import pathlib
import subprocess
import sys

site_dir = pathlib.Path({str(site_dir)!r})
marker = pathlib.Path({str(marker)!r})
userbase = pathlib.Path({str(userbase)!r})
requirements = {list(requirements)!r}
site_dir.mkdir(parents=True, exist_ok=True)
userbase.mkdir(parents=True, exist_ok=True)

def _list_packages(path):
    packages = []
    if path is None or not path.exists():
        return packages
    for entry in path.iterdir():
        if entry.is_dir() and entry.name.endswith(".dist-info"):
            stem = entry.name[: -len(".dist-info")]
            if "-" in stem:
                name, version = stem.rsplit("-", 1)
                packages.append(f"{{name}}=={{version}}")
            else:
                packages.append(stem)
    return sorted(packages)

def _userbase_site_packages(root):
    for lib_name in ("lib", "lib64"):
        lib_dir = root / lib_name
        if not lib_dir.is_dir():
            continue
        for py_dir in sorted(lib_dir.iterdir(), reverse=True):
            if py_dir.is_dir() and py_dir.name.startswith("python"):
                site_packages = py_dir / "site-packages"
                if site_packages.is_dir():
                    return site_packages
    return None

env = {{
    key: value
    for key, value in {{
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    }}.items()
    if value
}}
env["PYTHONUSERBASE"] = str(userbase)
env["PIP_CONFIG_FILE"] = os.devnull
env["PIP_INDEX_URL"] = {pip_index_url!r}
explicit_env_map = {{
    "TASK_CONTAINER_HTTP_PROXY": ("HTTP_PROXY", "http_proxy"),
    "TASK_CONTAINER_HTTPS_PROXY": ("HTTPS_PROXY", "https_proxy"),
    "TASK_CONTAINER_ALL_PROXY": ("ALL_PROXY", "all_proxy"),
    "TASK_CONTAINER_NO_PROXY": ("NO_PROXY", "no_proxy"),
    "TASK_CONTAINER_PIP_EXTRA_INDEX_URL": ("PIP_EXTRA_INDEX_URL",),
    "TASK_CONTAINER_PIP_TRUSTED_HOST": ("PIP_TRUSTED_HOST",),
    "TASK_CONTAINER_PIP_CERT": ("PIP_CERT",),
    "TASK_CONTAINER_SSL_CERT_FILE": ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"),
}}
for source_key, target_keys in explicit_env_map.items():
    value = os.environ.get(source_key)
    if not value:
        continue
    for target_key in target_keys:
        env[target_key] = value
get_pip = userbase / "get-pip.py"
print("[bootstrap] step 1/3: bootstrapping pip", flush=True)
subprocess.check_call(
    [
        sys.executable,
        str(get_pip),
        "--user",
        "--break-system-packages",
        "-i",
        {pip_index_url!r},
    ],
    env=env,
)
pip_bin = userbase / "bin" / "pip"
if not pip_bin.exists():
    pip_bin = userbase / "bin" / "pip3"
if not pip_bin.exists():
    raise RuntimeError("pip bootstrap succeeded but pip executable is missing")
print(f"[bootstrap] step 2/3: installing {{len(requirements)}} runtime deps", flush=True)
subprocess.check_call(
    [
        str(pip_bin),
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--break-system-packages",
        "--target",
        str(site_dir),
        "-i",
        {pip_index_url!r},
        *requirements,
    ],
    env=env,
)
print("[bootstrap] step 3/3: writing cache marker", flush=True)
marker.write_text(
    json.dumps(
        {{
            "requirements": requirements,
            "python": sys.executable,
            "pip_index_url": {pip_index_url!r},
            "arch": {arch!r},
            "image_platform": {exec_config.image_platform!r},
            "python_fingerprint": {python_fingerprint!r},
            "pip_resolution_fingerprint": {pip_resolution_fingerprint!r},
            "cache_key": {cache_key!r},
            "packages": _list_packages(site_dir),
            "userbase_packages": _list_packages(_userbase_site_packages(userbase)),
        }}
    ),
    encoding="utf-8",
)
# Keep userbase intact so runtime PATH/PYTHONUSERBASE can resolve pip and
# other scripts installed by get-pip.py --user. The enclosing cache generation
# is immutable once published via current.json.
"""
        result = subprocess.run(
            [
                container_executable,
                "exec",
                "-i",
                "-w",
                cwd,
                container_id,
                exec_config.runtime,
                "-",
            ],
            input=script,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if result.returncode != 0:
            _remove_tree_if_exists(generation_dir)
            raise RuntimeError(
                "task-container python bootstrap failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        published_config = _exec_config_with_bootstrap_site_dir(exec_config, site_dir)
        current_tmp = current_path.with_suffix(".tmp")
        current_tmp.write_text(
            json.dumps(
                {
                    "generation": generation,
                    "cache_key": cache_key,
                    "python_fingerprint": python_fingerprint,
                    "pip_resolution_fingerprint": pip_resolution_fingerprint,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        current_tmp.replace(current_path)
        return published_config
