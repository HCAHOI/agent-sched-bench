from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession

from agents.openclaw.runtime_deps import (
    OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS,
    OPENCLAW_MCP_RUNTIME_REQUIREMENTS,
)
from llm_call.config import validate_cloud_api_base


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


class TerminalBenchOpenClawAgent(AbstractInstalledAgent):
    """Terminal-Bench adapter that installs this repo and runs OpenClaw."""

    TRACE_FILENAME = "openclaw-trace.jsonl"
    RUNTIME_DIRNAME = "openclaw-runtime"
    VENV_PATH = "/installed-agent/venv"
    PYTHON_BOOTSTRAP_PATH = "/installed-agent/python/bin/python3"
    PROMPT_FILENAME = "openclaw-prompt.txt"
    CONTAINER_PROMPT_PATH = f"/installed-agent/{PROMPT_FILENAME}"
    CONTAINER_SECRET_FIFO_PATH = "/installed-agent/.openclaw-api-key.fifo"
    _SECRET_EXEC_ENV_KEY = "OPENCLAW_SECRET_VALUE"
    _SECRET_FIFO_WRITER_TIMEOUT_SEC = 86400.0
    _WHEEL_CACHE: Path | None = None

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
        *args,
        **kwargs,
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
        if not self._api_key:
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
        self._mcp_config_path = mcp_config_path
        self._temperature = _optional_float(temperature)
        self._top_p = _optional_float(top_p)
        self._top_k = _optional_int(top_k)
        self._repetition_penalty = _optional_float(repetition_penalty)

    _ENV_PASSTHROUGH = (
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "OPENCLAW_APT_MIRROR_PREFIX",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    )

    @property
    def _env(self) -> dict[str, str]:
        env = {
            # gateway resolution lives in setup-env.sh; see _create_env_setup_file
            "OPENCLAW_API_BASE": self._api_base,
        }
        if self._llm_timeout_sec is not None:
            env["OPENCLAW_LLM_TIMEOUT_S"] = str(self._llm_timeout_sec)
        for key in self._ENV_PASSTHROUGH:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _secret_exec_environment(self) -> dict[str, str]:
        return {self._SECRET_EXEC_ENV_KEY: self._api_key}

    @property
    def _install_agent_script_path(self) -> Path:
        requirements = self._container_runtime_requirements()
        install_requirements = " ".join(shlex.quote(req) for req in requirements)
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        python_bin = shlex.quote(self.PYTHON_BOOTSTRAP_PATH)
        venv_path = shlex.quote(self.VENV_PATH)
        wheel_path = shlex.quote(f"/installed-agent/{self._wheel_path.name}")
        script.write(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"{python_bin} -m venv {venv_path}\n"
            f"{venv_path}/bin/python -m pip install --upgrade pip\n"
            f"{venv_path}/bin/python -m pip install {install_requirements}\n"
            f"{venv_path}/bin/python -m pip install --no-deps {wheel_path}\n"
        )
        script.close()
        os.chmod(script.name, 0o755)
        return Path(script.name)

    def _container_runtime_requirements(self) -> tuple[str, ...]:
        extra_requirements = (
            OPENCLAW_MCP_RUNTIME_REQUIREMENTS if self._mcp_config_path else ()
        )
        return tuple(
            dict.fromkeys(OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS + extra_requirements)
        )

    @classmethod
    def _repo_root(cls) -> Path:
        return Path(__file__).resolve().parents[3]

    @staticmethod
    def _wheel_from_dir(wheel_dir: Path) -> Path | None:
        wheels = sorted(wheel_dir.glob("agent_sched_bench-*.whl"))
        return wheels[0] if wheels else None

    @staticmethod
    def _wheel_attempt_summary(
        name: str,
        result: subprocess.CompletedProcess[str] | None,
        *,
        unavailable: str | None = None,
    ) -> str:
        if unavailable is not None:
            return f"{name}: unavailable ({unavailable})"
        assert result is not None
        return (
            f"{name} failed (returncode={result.returncode}). stdout tail:\n"
            f"{result.stdout[-2000:]}\n--- stderr tail:\n{result.stderr[-2000:]}"
        )

    @classmethod
    def _build_wheel(cls) -> Path:
        if cls._WHEEL_CACHE and cls._WHEEL_CACHE.exists():
            return cls._WHEEL_CACHE
        repo_root = cls._repo_root()
        wheel_dir = Path(tempfile.mkdtemp(prefix="agent_sched_bench_wheel_"))
        failures: list[str] = []

        uv_path = shutil.which("uv")
        if uv_path is None:
            failures.append(
                cls._wheel_attempt_summary("uv build", None, unavailable="uv not found")
            )
        else:
            uv_result = subprocess.run(
                [
                    uv_path,
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(wheel_dir),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            wheel = cls._wheel_from_dir(wheel_dir)
            if uv_result.returncode == 0 and wheel is not None:
                cls._WHEEL_CACHE = wheel
                return cls._WHEEL_CACHE
            if uv_result.returncode == 0:
                failures.append("uv build succeeded but produced no agent_sched_bench wheel")
            else:
                failures.append(cls._wheel_attempt_summary("uv build", uv_result))

        pip_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                str(repo_root),
                "-w",
                str(wheel_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        wheel = cls._wheel_from_dir(wheel_dir)
        if pip_result.returncode == 0 and wheel is not None:
            cls._WHEEL_CACHE = wheel
            return cls._WHEEL_CACHE
        if pip_result.returncode == 0:
            failures.append("pip wheel succeeded but produced no agent_sched_bench wheel")
        else:
            failures.append(cls._wheel_attempt_summary("pip wheel", pip_result))
        raise RuntimeError("failed to build agent-sched-bench wheel:\n" + "\n\n".join(failures))

    @property
    def _wheel_path(self) -> Path:
        return self._build_wheel()

    @classmethod
    def _write_prompt_file(cls, instruction: str) -> Path:
        prompt_dir = Path(tempfile.mkdtemp(prefix="openclaw_prompt_"))
        prompt_path = prompt_dir / cls.PROMPT_FILENAME
        prompt_path.write_text(instruction, encoding="utf-8")
        return prompt_path

    @classmethod
    def _bootstrap_dependencies_command(cls) -> str:
        # `container.exec_run(...)` does NOT inherit the tmux session's
        # setup-env.sh exports, so the mirror prefix must be embedded into
        # the script literally rather than read from an env var at runtime.
        # The glob covers both legacy `*.list` and Ubuntu 24.04+ deb822
        # `*.sources` files; sed patterns use the URL stem (no trailing `/`)
        # so they hit both formats.
        apt_mirror = os.environ.get("OPENCLAW_APT_MIRROR_PREFIX", "").rstrip("/")
        if apt_mirror:
            mirror_swap = (
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
        else:
            mirror_swap = ""
        return (
            "set -euo pipefail\n"
            "install_python_deps() {\n"
            f"{mirror_swap}"
            "  apt-get update >&2\n"
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3 python3-pip python3-venv curl ca-certificates >&2\n"
            "}\n"
            "python_modern() {\n"
            "  \"$1\" - <<'PY'\n"
            "import sys\n"
            "raise SystemExit(0 if sys.version_info >= (3, 10) else 1)\n"
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
            "find_modern_python() {\n"
            "  for candidate in python3 python3.13 python3.12 python3.11 python3.10; do\n"
            "    if command -v \"$candidate\" >/dev/null 2>&1; then\n"
            "      candidate_path=$(command -v \"$candidate\")\n"
            "      if python_modern \"$candidate_path\" && venv_ready \"$candidate_path\"; then\n"
            "        printf '%s\\n' \"$candidate_path\"\n"
            "        return 0\n"
            "      fi\n"
            "    fi\n"
            "  done\n"
            "  return 1\n"
            "}\n"
            "select_python() {\n"
            "  if modern_python=$(find_modern_python); then\n"
            "    printf '%s\\n' \"$modern_python\"\n"
            "    return 0\n"
            "  fi\n"
            "  install_python_deps\n"
            "  if modern_python=$(find_modern_python); then\n"
            "    printf '%s\\n' \"$modern_python\"\n"
            "    return 0\n"
            "  fi\n"
            "  uv_python=$(install_uv_python)\n"
            "  python_modern \"$uv_python\"\n"
            "  venv_ready \"$uv_python\"\n"
            "  printf '%s\\n' \"$uv_python\"\n"
            "}\n"
            "selected_python=$(select_python)\n"
            f"mkdir -p {shlex.quote(str(Path(cls.PYTHON_BOOTSTRAP_PATH).parent))}\n"
            f"ln -sf \"$selected_python\" {shlex.quote(cls.PYTHON_BOOTSTRAP_PATH)}\n"
            f"{shlex.quote(cls.PYTHON_BOOTSTRAP_PATH)} - <<'PY'\n"
            "import sys\n"
            "assert sys.version_info >= (3, 10), sys.version\n"
            "PY\n"
        )

    def _run_agent_commands(self) -> list[TerminalCommand]:
        workspace = shlex.quote(".")
        trace_output = shlex.quote(
            f"{self.CONTAINER_AGENT_LOGS_PATH}/{self.TRACE_FILENAME}"
        )
        runtime_dir = shlex.quote(
            f"{self.CONTAINER_AGENT_LOGS_PATH}/{self.RUNTIME_DIRNAME}"
        )
        prompt_file = shlex.quote(self.CONTAINER_PROMPT_PATH)
        api_base_arg = shlex.quote(self._api_base)
        secret_prefix = (
            f'{self._env_key}="$(cat {shlex.quote(self.CONTAINER_SECRET_FIFO_PATH)})" '
        )
        mcp_flag = ""
        if self._mcp_config_path:
            mcp_flag = f"--mcp-config {shlex.quote(self._container_mcp_config_path)} "
        generation_flags = ""
        if self._temperature is not None:
            generation_flags += f"--temperature {shlex.quote(str(self._temperature))} "
        if self._top_p is not None:
            generation_flags += f"--top-p {shlex.quote(str(self._top_p))} "
        if self._top_k is not None:
            generation_flags += f"--top-k {shlex.quote(str(self._top_k))} "
        if self._repetition_penalty is not None:
            generation_flags += (
                f"--repetition-penalty {shlex.quote(str(self._repetition_penalty))} "
            )
        command = (
            f"{secret_prefix}"
            f"{self.VENV_PATH}/bin/openclaw "
            f"--provider {shlex.quote(self._provider_name)} "
            f"--model {shlex.quote(self._model_name)} "
            f"--api-base {api_base_arg} "
            f"{mcp_flag}"
            f"--workspace {workspace} "
            f"--trace-output {trace_output} "
            f"--runtime-dir {runtime_dir} "
            f"--max-iterations {self._max_iterations} "
            f"{generation_flags}"
            "--quiet "
            f"--prompt-file {prompt_file}"
        )
        return [
            TerminalCommand(
                command=command,
                min_timeout_sec=0.0,
                max_timeout_sec=self._agent_timeout_sec
                if self._agent_timeout_sec is not None
                else float("inf"),
                block=True,
                append_enter=True,
            )
        ]

    @staticmethod
    def _deadline(timeout_sec: float | None) -> float | None:
        if timeout_sec is None:
            return None
        return time.monotonic() + timeout_sec

    def _remaining_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return float("inf")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Agent timed out after {self._agent_timeout_sec} seconds"
            )
        return remaining

    def _command_with_deadline(
        self,
        command: TerminalCommand,
        deadline: float | None,
    ) -> TerminalCommand:
        max_timeout_sec = command.max_timeout_sec
        if deadline is not None:
            max_timeout_sec = min(max_timeout_sec, self._remaining_timeout(deadline))
        return TerminalCommand(
            command=command.command,
            min_timeout_sec=command.min_timeout_sec,
            max_timeout_sec=max_timeout_sec,
            block=command.block,
            append_enter=command.append_enter,
        )

    def _exec_run_with_deadline(
        self,
        session: TmuxSession,
        command: list[str],
        deadline: float | None,
        *,
        user: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> Any:
        if deadline is None:
            if environment is None:
                return session.container.exec_run(command, user=user)
            return session.container.exec_run(
                command,
                user=user,
                environment=environment,
            )
        timeout_sec = self._remaining_timeout(deadline)
        wrapped_command = ["timeout", f"{timeout_sec:.3f}s", *command]
        if environment is None:
            result = session.container.exec_run(wrapped_command, user=user)
        else:
            result = session.container.exec_run(
                wrapped_command,
                user=user,
                environment=environment,
            )
        if result.exit_code == 124:
            raise TimeoutError(
                f"Agent timed out after {self._agent_timeout_sec} seconds"
            )
        self._remaining_timeout(deadline)
        return result

    @staticmethod
    def _container_copy_target(session: TmuxSession) -> str:
        container_id = session.container.id
        if not container_id:
            raise RuntimeError("Unable to determine Docker container id for copy")
        return str(container_id)

    def _copy_to_container_with_deadline(
        self,
        session: TmuxSession,
        *,
        paths: list[Path],
        container_dir: str,
        deadline: float | None,
    ) -> None:
        if deadline is None:
            session.copy_to_container(paths=paths, container_dir=container_dir)
            return
        self._exec_run_with_deadline(
            session,
            ["mkdir", "-p", container_dir],
            deadline,
        )
        container_target = self._container_copy_target(session)
        destination = f"{container_target}:{container_dir.rstrip('/')}/"
        for path in paths:
            try:
                result = subprocess.run(
                    ["docker", "cp", str(path), destination],
                    capture_output=True,
                    text=True,
                    timeout=self._remaining_timeout(deadline),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Agent timed out after {self._agent_timeout_sec} seconds"
                ) from exc
            if result.returncode != 0:
                raise RuntimeError(
                    "docker cp failed while installing OpenClaw into "
                    f"Terminal-Bench container. stdout tail:\n"
                    f"{result.stdout[-2000:]}\n--- stderr tail:\n"
                    f"{result.stderr[-2000:]}"
                )

    def _prepare_secret_fifo(
        self,
        session: TmuxSession,
        deadline: float | None,
    ) -> None:
        command = (
            "set -euo pipefail\n"
            f"rm -f {shlex.quote(self.CONTAINER_SECRET_FIFO_PATH)}\n"
            "umask 077\n"
            f"mkfifo {shlex.quote(self.CONTAINER_SECRET_FIFO_PATH)}\n"
            f"chmod 600 {shlex.quote(self.CONTAINER_SECRET_FIFO_PATH)}\n"
        )
        result = self._exec_run_with_deadline(
            session,
            ["bash", "-lc", command],
            deadline,
        )
        if result.exit_code != 0:
            raise RuntimeError("failed to prepare API key FIFO")

    def _start_secret_fifo_writer(
        self,
        session: TmuxSession,
        deadline: float | None,
    ) -> None:
        timeout_sec = (
            self._remaining_timeout(deadline)
            if deadline is not None
            else self._agent_timeout_sec or self._SECRET_FIFO_WRITER_TIMEOUT_SEC
        )
        command = [
            "timeout",
            f"{timeout_sec:.3f}s",
            "sh",
            "-lc",
            (
                f"printf '%s' \"${self._SECRET_EXEC_ENV_KEY}\" > "
                f"{shlex.quote(self.CONTAINER_SECRET_FIFO_PATH)}"
            ),
        ]
        session.container.exec_run(
            command,
            environment=self._secret_exec_environment(),
            detach=True,
        )

    def _cleanup_timed_out_session(
        self,
        session: TmuxSession,
        logging_dir: Path | None,
    ) -> None:
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "openclaw-timeout.marker").write_text(
                "timeout\n",
                encoding="utf-8",
            )
            try:
                (logging_dir / "openclaw-timeout-pane.txt").write_text(
                    session.capture_pane(capture_entire=True),
                    encoding="utf-8",
                )
            except Exception:
                pass

        try:
            session.send_keys(["C-c"], min_timeout_sec=0.1)
        except Exception:
            pass

        session_name = str(getattr(session, "_session_name", "agent"))
        cleanup_script = (
            f"tmux kill-session -t {shlex.quote(session_name)} 2>/dev/null || true\n"
            "pkill -TERM -f '/installed-agent/venv/bin/openclaw' 2>/dev/null || true\n"
            "sleep 2\n"
            "pkill -KILL -f '/installed-agent/venv/bin/openclaw' 2>/dev/null || true\n"
            f"rm -f {shlex.quote(self.CONTAINER_SECRET_FIFO_PATH)} 2>/dev/null || true\n"
        )
        try:
            session.container.exec_run(["sh", "-lc", cleanup_script])
        except Exception:
            pass

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        deadline = self._deadline(self._agent_timeout_sec)
        try:
            bootstrap = self._exec_run_with_deadline(
                session,
                [
                    "bash",
                    "-lc",
                    self._bootstrap_dependencies_command(),
                ],
                deadline,
                user="root",
            )
            if bootstrap.exit_code != 0:
                return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

            rendered_instruction = self._render_instruction(instruction)
            prompt_file = self._write_prompt_file(rendered_instruction)
            install_script = self._install_agent_script_path
            copy_paths = [self._wheel_path, install_script, prompt_file]
            if self._mcp_config_path:
                host_mcp_config = Path(self._mcp_config_path).expanduser().resolve()
                if not host_mcp_config.exists():
                    raise FileNotFoundError(
                        "Terminal-Bench MCP config path does not exist: "
                        f"{host_mcp_config}"
                    )
                copy_paths.append(host_mcp_config)
            self._copy_to_container_with_deadline(
                session,
                paths=copy_paths,
                container_dir="/installed-agent",
                deadline=deadline,
            )

            env_setup_content = self._create_env_setup_file()
            self._exec_run_with_deadline(
                session,
                [
                    "sh",
                    "-c",
                    (
                        f"echo {shlex.quote(env_setup_content)} > "
                        "/installed-agent/setup-env.sh"
                    ),
                ],
                deadline,
            )
            session.send_keys(
                ["source /installed-agent/setup-env.sh", "Enter"],
                block=True,
                max_timeout_sec=self._remaining_timeout(deadline),
            )
            session.send_keys(
                [
                    (
                        "source /installed-agent/"
                        + install_script.name
                        + " || echo 'INSTALL_FAIL_STATUS'"
                    ),
                    "Enter",
                ],
                block=True,
                max_timeout_sec=self._remaining_timeout(deadline),
            )
            installation_output = session.capture_pane(capture_entire=True)
            if "INSTALL_FAIL_STATUS" in installation_output.splitlines():
                return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

            for command in self._run_agent_commands():
                self._prepare_secret_fifo(session, deadline)
                self._start_secret_fifo_writer(session, deadline)
                session.send_command(self._command_with_deadline(command, deadline))
                self._exec_run_with_deadline(
                    session,
                    ["rm", "-f", self.CONTAINER_SECRET_FIFO_PATH],
                    deadline,
                )
        except TimeoutError:
            self._cleanup_timed_out_session(session, logging_dir)
            return AgentResult(failure_mode=FailureMode.AGENT_TIMEOUT)
        if logging_dir is not None:
            marker_path = logging_dir / "openclaw-complete.marker"
            marker_path.write_text("completed", encoding="utf-8")
        return AgentResult(total_input_tokens=0, total_output_tokens=0)

    @property
    def _container_mcp_config_path(self) -> str:
        if not self._mcp_config_path:
            raise ValueError("mcp_config_path is not configured")
        return f"/installed-agent/{Path(self._mcp_config_path).name}"
