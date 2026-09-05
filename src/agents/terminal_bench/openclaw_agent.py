from __future__ import annotations

import asyncio
import math
import os
import re
import shlex
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.tmux_session import TmuxSession

from agents.openclaw._session_runner import SessionRunner
from llm_call import create_provider
from llm_call.config import validate_cloud_api_base
from trace_collect.openclaw_host_runtime import (
    _update_trace_metadata,
    build_container_tools_for_agent,
    container_runtime_label,
    container_runtime_proof,
)
from trace_collect.openclaw_tools import ContainerAgent


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


class TerminalBenchOpenClawAgent(AbstractInstalledAgent):
    """Terminal-Bench adapter that runs OpenClaw on the host.

    Terminal-Bench owns task-container lifecycle and scoring.  This adapter
    receives the live ``TmuxSession`` for the task container, then runs the
    OpenClaw ``SessionRunner`` in the host ``tb run`` process with tool
    overrides that execute inside the task container as root.
    """

    TRACE_FILENAME = "openclaw-trace.jsonl"
    RUNTIME_DIRNAME = "openclaw-runtime"
    PYTHON_BOOTSTRAP_PATH = "/installed-agent/python/bin/python3"
    # Safety bound for rare no-Python containers when no task-level agent
    # timeout is configured. The normal path uses an existing bridge Python and
    # never runs this bootstrap.
    BRIDGE_BOOTSTRAP_TIMEOUT_SEC = 3600.0
    _NOOP_INSTALL_SCRIPT: Path | None = None

    @staticmethod
    def name() -> str:
        return "agent-sched-bench-openclaw"

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        api_base: str,
        env_key: str,
        api_key: str | None = None,
        max_iterations: int = 100,
        llm_timeout_sec: float | None = None,
        agent_timeout_sec: float | str | None = None,
        mcp_config_path: str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        top_k: int | str | None = None,
        repetition_penalty: float | str | None = None,
        service_tier: str | None = None,
        bridge_bootstrap_timeout_sec: float | str | None = None,
        tool_resource_profile: str | None = None,
        resource_run_token: str | None = None,
        resource_trace_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        validate_cloud_api_base(api_base)
        self._model_name = model_name
        self._provider_name = provider_name
        self._api_base = api_base
        self._env_key = env_key
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_key) is None:
            raise ValueError(
                f"env_key must be a valid shell environment name: {env_key!r}"
            )
        self._api_key = api_key or os.environ.get(env_key, "")
        if not self._api_key and provider_name != "codex":
            raise ValueError(
                f"missing API key for TerminalBenchOpenClawAgent env_key={env_key!r}"
            )
        self._max_iterations = int(max_iterations)
        self._llm_timeout_sec = (
            None if llm_timeout_sec is None else float(llm_timeout_sec)
        )
        self._agent_timeout_sec = _optional_float(agent_timeout_sec)
        if self._agent_timeout_sec is not None and self._agent_timeout_sec <= 0:
            raise ValueError(
                f"agent_timeout_sec must be positive, got {self._agent_timeout_sec!r}"
            )
        self._bridge_bootstrap_timeout_sec = (
            self.BRIDGE_BOOTSTRAP_TIMEOUT_SEC
            if bridge_bootstrap_timeout_sec is None
            else float(bridge_bootstrap_timeout_sec)
        )
        if (
            not math.isfinite(self._bridge_bootstrap_timeout_sec)
            or self._bridge_bootstrap_timeout_sec <= 0
        ):
            raise ValueError(
                "bridge_bootstrap_timeout_sec must be positive, got "
                f"{self._bridge_bootstrap_timeout_sec!r}"
            )
        self._mcp_config_path = mcp_config_path
        self._temperature = _optional_float(temperature)
        self._top_p = _optional_float(top_p)
        self._top_k = _optional_int(top_k)
        self._repetition_penalty = _optional_float(repetition_penalty)
        self._service_tier = service_tier
        self._tool_resource_profile = tool_resource_profile
        self._resource_run_token = resource_run_token
        self._resource_trace_id = resource_trace_id

    @property
    def _env(self) -> dict[str, str]:
        """No container-side LLM secrets: OpenClaw runs in the host process."""

        return {}

    @property
    def _install_agent_script_path(self) -> Path:
        """Terminal-Bench abstract contract; no in-container install is needed."""

        if self._NOOP_INSTALL_SCRIPT and self._NOOP_INSTALL_SCRIPT.exists():
            return self._NOOP_INSTALL_SCRIPT
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        script.write(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "# OpenClaw runs on the host; task container receives tool calls only.\n"
        )
        script.close()
        os.chmod(script.name, 0o755)
        self.__class__._NOOP_INSTALL_SCRIPT = Path(script.name)
        return self.__class__._NOOP_INSTALL_SCRIPT

    def _run_agent_commands(self, instruction: str | None = None) -> list[Any]:
        """No tmux command is launched; ``perform_task`` drives SessionRunner."""

        del instruction
        return []

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        deadline = self._deadline(self._agent_timeout_sec)
        try:
            return asyncio.run(
                self._perform_task_async(
                    instruction=instruction,
                    session=session,
                    logging_dir=logging_dir,
                    deadline=deadline,
                )
            )
        except TimeoutError:
            self._write_error(logging_dir, "OpenClaw host-controller timed out")
            return AgentResult(failure_mode=FailureMode.AGENT_TIMEOUT)
        except Exception:
            self._write_error(logging_dir, traceback.format_exc())
            return AgentResult(failure_mode=FailureMode.UNKNOWN_AGENT_ERROR)

    async def _perform_task_async(
        self,
        *,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None,
        deadline: float | None,
    ) -> AgentResult:
        log_dir = self._resolve_logging_dir(logging_dir)
        trace_file = log_dir / self.TRACE_FILENAME
        runtime_dir = log_dir / self.RUNTIME_DIRNAME
        workspace = runtime_dir / "host-workspace"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)

        container_id = self._container_id(session)
        container_executable = self._container_executable()
        container_workdir = self._container_workdir(session)
        agent = ContainerAgent(
            container_id,
            container_executable,
            workdir=container_workdir,
        )
        resource_trace = None
        resource_trace_finalized = False

        def finalize_resource_trace(workload_status: str) -> None:
            nonlocal resource_trace_finalized
            if resource_trace is None or resource_trace_finalized:
                return
            from trace_collect.openclaw_host_runtime import (
                _attach_resource_observations,
                _finalized_resource_status,
            )

            errors: list[str] = []
            try:
                for error in _attach_resource_observations(
                    trace_file,
                    resource_trace.calls,
                    [],
                ):
                    resource_trace.add_integrity_error(error)
                status, errors = _finalized_resource_status(
                    resource_trace,
                    replay_execution=workload_status,
                )
                if resource_trace.final_artifact is not None:
                    errors.extend(
                        _attach_resource_observations(
                            trace_file,
                            resource_trace.calls,
                            [],
                        )
                    )
            except BaseException as exc:
                status = {
                    "telemetry_quality": "unavailable",
                    "formal_completeness": "unavailable",
                    "call_coverage": None,
                    "collection_validity": "invalid",
                }
                errors.append(
                    f"resource finalization failed: {type(exc).__name__}: {exc}"
                )
            finally:
                resource_trace_finalized = True
            _update_trace_metadata(
                trace_file,
                {
                    "tool_resource": {
                        "profile": self._tool_resource_profile,
                        "service_enabled": True,
                    },
                    "resource_artifact_path": str(
                        log_dir / "resource_observations.json"
                    ),
                    "telemetry_errors": errors,
                    **status,
                },
            )

        try:
            await self._start_container_agent(agent, session=session, deadline=deadline)
            proof = await self._await_with_deadline(
                container_runtime_proof(
                    agent,
                    container_id=container_id,
                    mode="collect",
                    expected_workdir=container_workdir,
                ),
                deadline,
            )
            runtime_label = container_runtime_label(proof)
            if self._tool_resource_profile is not None:
                from tool_resource.client import ResourceTrace

                resource_trace = ResourceTrace.open(
                    self._tool_resource_profile,
                    run_token=self._resource_run_token or "",
                    trace_id=self._resource_trace_id or log_dir.name,
                    container_runtime=Path(container_executable).name,
                    container_id=container_id,
                    artifact_path=log_dir / "resource_observations.json",
                )
                setup_started = time.monotonic()
                setup_error = await asyncio.to_thread(resource_trace.wait_ready)
                if deadline is not None:
                    # Instrumentation setup is outside the workload timeout.
                    deadline += time.monotonic() - setup_started
                if setup_error is not None:
                    resource_trace.add_integrity_error(setup_error)
            provider = create_provider(
                provider_name=self._provider_name,
                api_key=self._api_key,
                api_base=self._api_base,
                default_model=self._model_name,
                **self._generation_config(),
            )
            runner = SessionRunner(
                provider,
                model=self._model_name,
                max_iterations=self._max_iterations,
                mcp_servers=self._load_mcp_servers(),
                tool_overrides=build_container_tools_for_agent(
                    agent,
                    exec_timeout=300,
                    exec_path_append="",
                    workspace=container_workdir,
                    resource_trace=resource_trace,
                ),
            )
            rendered_instruction = self._render_instruction(instruction)
            result = await self._await_with_deadline(
                runner.run(
                    prompt=rendered_instruction,
                    workspace=workspace,
                    tool_workspace=Path(container_workdir),
                    project_workspace=Path(container_workdir),
                    session_key=f"terminal-bench:{container_id[:12]}",
                    trace_file=trace_file,
                    runtime_dir=runtime_dir,
                    instance_id=container_id,
                    runtime_label=runtime_label,
                    channel="terminal-bench",
                    prepare_ms=None,
                ),
                deadline,
            )
            metadata_extra = {
                **proof,
                "replay_mode": None,
                "openclaw_runtime": "host_session_runner",
                "terminal_bench_container_id": container_id,
                "terminal_bench_workdir": container_workdir,
                "terminal_bench_agent_adapter": self.name(),
                "provider_name": self._provider_name,
            }
            _update_trace_metadata(trace_file, metadata_extra)
            finalize_resource_trace(
                "completed"
                if result.stop_reason == "completed" and result.error is None
                else "failed"
            )
            if result.stop_reason != "completed" or result.error is not None:
                self._write_error(
                    log_dir,
                    "OpenClaw host-controller failed: "
                    f"stop_reason={result.stop_reason!r}, error={result.error!r}",
                )
                return AgentResult(failure_mode=FailureMode.UNKNOWN_AGENT_ERROR)
        finally:
            if resource_trace is not None and not resource_trace_finalized:
                finalize_resource_trace("failed")
            await agent.stop()

        (log_dir / "openclaw-complete.marker").write_text(
            "completed\n",
            encoding="utf-8",
        )
        return AgentResult(total_input_tokens=0, total_output_tokens=0)

    def _generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if self._llm_timeout_sec is not None:
            config["timeout"] = self._llm_timeout_sec
        if self._temperature is not None:
            config["temperature"] = self._temperature
        if self._top_p is not None:
            config["top_p"] = self._top_p
        if self._top_k is not None:
            config["top_k"] = self._top_k
        if self._repetition_penalty is not None:
            config["repetition_penalty"] = self._repetition_penalty
        if self._service_tier is not None:
            config["service_tier"] = self._service_tier
        return config

    def _load_mcp_servers(self) -> dict[str, Any]:
        from trace_collect.collector import load_mcp_servers

        return load_mcp_servers(self._mcp_config_path)


    @staticmethod
    def _resolve_logging_dir(logging_dir: Path | None) -> Path:
        if logging_dir is None:
            path = Path(tempfile.mkdtemp(prefix="openclaw_terminal_bench_logs_"))
        else:
            path = Path(logging_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _container_id(session: TmuxSession) -> str:
        container_id = getattr(session.container, "id", None)
        if not container_id:
            raise RuntimeError("Terminal-Bench session container has no id")
        return str(container_id)

    @staticmethod
    def _container_executable() -> str:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("docker is not available on PATH")
        return docker

    @staticmethod
    def _apt_mirror_swap_script() -> str:
        apt_mirror = os.environ.get("OPENCLAW_APT_MIRROR_PREFIX", "").rstrip("/")
        if not apt_mirror:
            return ""
        return (
            "  for f in /etc/apt/sources.list "
            "/etc/apt/sources.list.d/*.list "
            "/etc/apt/sources.list.d/*.sources; do\n"
            '    [ -f "$f" ] || continue\n'
            "    sed -i \\\n"
            f"      -e 's|http://archive.ubuntu.com/ubuntu|{apt_mirror}/ubuntu|g' \\\n"
            f"      -e 's|http://security.ubuntu.com/ubuntu|{apt_mirror}/ubuntu|g' \\\n"
            f"      -e 's|http://deb.debian.org/debian|{apt_mirror}/debian|g' \\\n"
            f"      -e 's|http://security.debian.org/debian-security|{apt_mirror}/debian-security|g' \\\n"
            '      "$f" || true\n'
            "  done\n"
        )

    @classmethod
    def _bootstrap_dependencies_command(cls) -> str:
        """Install/select a Python >=3.11 runtime for task replay setup."""

        mirror_swap = cls._apt_mirror_swap_script()
        return (
            "set -euo pipefail\n"
            "install_python_deps() {\n"
            f"{mirror_swap}"
            "  apt-get update >&2\n"
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3 python3-pip python3-venv curl ca-certificates >&2\n"
            "}\n"
            "python_supported() {\n"
            "  \"$1\" - <<'PY'\n"
            "import sys\n"
            "raise SystemExit(0 if sys.version_info >= (3, 11) else 1)\n"
            "PY\n"
            "}\n"
            "probe_root=$(mktemp -d /tmp/openclaw-venv-check.XXXXXX)\n"
            'cleanup_probe() { rm -rf "$probe_root"; }\n'
            "trap cleanup_probe EXIT\n"
            "venv_ready() {\n"
            '  rm -rf "$probe_root/venv"\n'
            '  "$1" -m venv "$probe_root/venv" >/dev/null 2>&1 && '
            '"$probe_root/venv/bin/python" -m pip --version >/dev/null 2>&1\n'
            "}\n"
            "install_uv_python() {\n"
            "  mkdir -p /installed-agent/uv /installed-agent/uv-python\n"
            "  if [ ! -x /installed-agent/uv/uv ]; then\n"
            "    tmp_uv_install=$(mktemp -d /tmp/openclaw-uv-install.XXXXXX)\n"
            "    curl -LsSf https://astral.sh/uv/install.sh -o \"$tmp_uv_install/install.sh\"\n"
            "    UV_INSTALL_DIR=/installed-agent/uv sh \"$tmp_uv_install/install.sh\" --no-modify-path >&2\n"
            "  fi\n"
            "  UV_PYTHON_INSTALL_DIR=/installed-agent/uv-python "
            "/installed-agent/uv/uv python install 3.12 --quiet >&2\n"
            "  UV_PYTHON_INSTALL_DIR=/installed-agent/uv-python "
            "/installed-agent/uv/uv python find 3.12 --managed-python\n"
            "}\n"
            "find_supported_python() {\n"
            "  for candidate in python3 python3.13 python3.12 python3.11; do\n"
            "    if command -v \"$candidate\" >/dev/null 2>&1; then\n"
            "      candidate_path=$(command -v \"$candidate\")\n"
            "      if python_supported \"$candidate_path\" && venv_ready \"$candidate_path\"; then\n"
            "        printf '%s\\n' \"$candidate_path\"\n"
            "        return 0\n"
            "      fi\n"
            "    fi\n"
            "  done\n"
            "  return 1\n"
            "}\n"
            "select_python() {\n"
            "  if supported_python=$(find_supported_python); then\n"
            "    printf '%s\\n' \"$supported_python\"\n"
            "    return 0\n"
            "  fi\n"
            "  if install_python_deps; then\n"
            "    if supported_python=$(find_supported_python); then\n"
            "      printf '%s\\n' \"$supported_python\"\n"
            "      return 0\n"
            "    fi\n"
            "  fi\n"
            "  uv_python=$(install_uv_python)\n"
            "  python_supported \"$uv_python\"\n"
            "  venv_ready \"$uv_python\"\n"
            "  printf '%s\\n' \"$uv_python\"\n"
            "}\n"
            "selected_python=$(select_python)\n"
            f"mkdir -p {shlex.quote(str(Path(cls.PYTHON_BOOTSTRAP_PATH).parent))}\n"
            f"ln -sf \"$selected_python\" {shlex.quote(cls.PYTHON_BOOTSTRAP_PATH)}\n"
            f"{shlex.quote(cls.PYTHON_BOOTSTRAP_PATH)} - <<'PY'\n"
            "import sys\n"
            "assert sys.version_info >= (3, 11), sys.version\n"
            "PY\n"
        )

    @classmethod
    def _bootstrap_bridge_python_command(cls) -> str:
        """Install/select a Python >=3.6 runtime for the stdlib bridge only."""

        mirror_swap = cls._apt_mirror_swap_script()
        return (
            "set -euo pipefail\n"
            "install_python_deps() {\n"
            f"{mirror_swap}"
            "  apt-get update >&2\n"
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3 curl ca-certificates >&2\n"
            "}\n"
            "python_supported() {\n"
            "  \"$1\" - <<'PY'\n"
            "import sys\n"
            "raise SystemExit(0 if sys.version_info >= (3, 6) else 1)\n"
            "PY\n"
            "}\n"
            "find_supported_python() {\n"
            "  for candidate in python3 python3.13 python3.12 python3.11; do\n"
            "    if command -v \"$candidate\" >/dev/null 2>&1; then\n"
            "      candidate_path=$(command -v \"$candidate\")\n"
            "      if python_supported \"$candidate_path\"; then\n"
            "        printf '%s\\n' \"$candidate_path\"\n"
            "        return 0\n"
            "      fi\n"
            "    fi\n"
            "  done\n"
            "  return 1\n"
            "}\n"
            "select_python() {\n"
            "  if supported_python=$(find_supported_python); then\n"
            "    printf '%s\\n' \"$supported_python\"\n"
            "    return 0\n"
            "  fi\n"
            "  install_python_deps\n"
            "  find_supported_python\n"
            "}\n"
            "selected_python=$(select_python)\n"
            f"mkdir -p {shlex.quote(str(Path(cls.PYTHON_BOOTSTRAP_PATH).parent))}\n"
            f"ln -sf \"$selected_python\" {shlex.quote(cls.PYTHON_BOOTSTRAP_PATH)}\n"
            f"{shlex.quote(cls.PYTHON_BOOTSTRAP_PATH)} - <<'PY'\n"
            "import sys\n"
            "assert sys.version_info >= (3, 6), sys.version\n"
            "PY\n"
        )

    async def _start_container_agent(
        self,
        agent: ContainerAgent,
        *,
        session: TmuxSession,
        deadline: float | None,
    ) -> None:
        try:
            await self._await_with_deadline(agent.start(), deadline)
        except RuntimeError as exc:
            if "no Python >=3.6 found" not in str(exc):
                raise
            timeout_s = self._remaining_timeout_s(deadline)
            await self._await_with_deadline(
                asyncio.to_thread(
                    self._bootstrap_container_bridge_python,
                    session,
                    timeout_s,
                ),
                deadline,
            )
            await self._await_with_deadline(agent.start(), deadline)

    def _bootstrap_container_bridge_python(
        self,
        session: TmuxSession,
        timeout_s: float | None,
    ) -> None:
        effective_timeout_s = self._bridge_bootstrap_timeout_sec
        if timeout_s is not None:
            effective_timeout_s = min(effective_timeout_s, timeout_s)
        timeout_arg = self._format_timeout_s(effective_timeout_s)
        command = [
            "timeout",
            "--kill-after=5s",
            timeout_arg,
            "bash",
            "-lc",
            self._bootstrap_bridge_python_command(),
        ]
        result = session.container.exec_run(command, user="root")
        if result.exit_code != 0:
            if result.exit_code in {124, 137}:
                raise TimeoutError("OpenClaw host-controller deadline exceeded")
            output = result.output.decode(errors="replace")
            raise RuntimeError(
                "failed to bootstrap Python bridge in Terminal-Bench task "
                f"container: {output[-2000:]}"
            )

    @staticmethod
    def _format_timeout_s(timeout_s: float) -> str:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise TimeoutError("OpenClaw host-controller deadline exceeded")
        truncated = math.floor(timeout_s * 1_000_000) / 1_000_000
        if truncated <= 0:
            raise TimeoutError("OpenClaw host-controller deadline exceeded")
        formatted = f"{truncated:.6f}".rstrip("0").rstrip(".")
        return f"{formatted}s"

    @staticmethod
    def _remaining_timeout_s(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenClaw host-controller deadline exceeded")
        return remaining

    @staticmethod
    def _container_workdir(session: TmuxSession) -> str:
        session_name = str(getattr(session, "_session_name", ""))
        if not session_name:
            raise RuntimeError("Terminal-Bench tmux session name is unavailable")
        result = session.container.exec_run(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                session_name,
                "#{pane_current_path}",
            ]
        )
        if result.exit_code != 0:
            output = result.output.decode(errors="replace")
            raise RuntimeError(
                "failed to resolve Terminal-Bench tmux pane cwd: "
                f"{output.strip()}"
            )
        workdir = result.output.decode(errors="replace").strip()
        if not workdir.startswith("/"):
            raise RuntimeError(f"Terminal-Bench pane cwd is not absolute: {workdir!r}")
        return workdir

    @staticmethod
    async def _await_with_deadline(awaitable: Any, deadline: float | None) -> Any:
        if deadline is None:
            return await awaitable
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenClaw host-controller deadline exceeded")
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("OpenClaw host-controller deadline exceeded") from exc

    @staticmethod
    def _deadline(timeout_sec: float | None) -> float | None:
        if timeout_sec is None:
            return None
        return time.monotonic() + timeout_sec

    @staticmethod
    def _write_error(logging_dir: Path | None, content: str) -> None:
        if logging_dir is None:
            return
        path = Path(logging_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "openclaw-error.txt").write_text(content, encoding="utf-8")
