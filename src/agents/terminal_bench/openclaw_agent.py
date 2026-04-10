from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession


class TerminalBenchOpenClawAgent(AbstractInstalledAgent):
    """Terminal-Bench adapter that installs this repo and runs OpenClaw."""

    TRACE_FILENAME = "openclaw-trace.jsonl"
    VENV_PATH = "/installed-agent/venv"
    _WHEEL_CACHE: Path | None = None

    @staticmethod
    def name() -> str:
        return "agent-sched-bench-openclaw"

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        api_base: str,
        api_key: str | None,
        env_key: str,
        max_iterations: int = 200,
        mcp_config_path: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        self._provider_name = provider_name
        self._api_base = api_base
        self._env_key = env_key
        self._api_key = api_key or os.environ.get(env_key, "")
        if not self._api_key:
            raise ValueError(
                f"missing API key for TerminalBenchOpenClawAgent env_key={env_key!r}"
            )
        self._max_iterations = int(max_iterations)
        self._mcp_config_path = mcp_config_path

    @property
    def _env(self) -> dict[str, str]:
        return {
            self._env_key: self._api_key,
        }

    @property
    def _install_agent_script_path(self) -> Path:
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        script.write(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"python3 -m venv {self.VENV_PATH}\n"
            f"{self.VENV_PATH}/bin/python -m pip install --upgrade pip\n"
            f"{self.VENV_PATH}/bin/python -m pip install /installed-agent/{self._wheel_path.name}\n"
        )
        script.close()
        os.chmod(script.name, 0o755)
        return Path(script.name)

    @classmethod
    def _repo_root(cls) -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _build_wheel(cls) -> Path:
        if cls._WHEEL_CACHE and cls._WHEEL_CACHE.exists():
            return cls._WHEEL_CACHE
        repo_root = cls._repo_root()
        wheel_dir = Path(tempfile.mkdtemp(prefix="agent_sched_bench_wheel_"))
        subprocess.run(
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
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = sorted(wheel_dir.glob("agent_sched_bench-*.whl"))
        if not wheels:
            raise RuntimeError("failed to build agent-sched-bench wheel")
        cls._WHEEL_CACHE = wheels[0]
        return cls._WHEEL_CACHE

    @property
    def _wheel_path(self) -> Path:
        return self._build_wheel()

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped_instruction = shlex.quote(instruction)
        workspace = shlex.quote(".")
        trace_output = shlex.quote(
            f"{self.CONTAINER_AGENT_LOGS_PATH}/{self.TRACE_FILENAME}"
        )
        mcp_flag = ""
        if self._mcp_config_path:
            mcp_flag = (
                f"--mcp-config "
                f"{shlex.quote(self._container_mcp_config_path)} "
            )
        command = (
            f"{self.VENV_PATH}/bin/openclaw "
            f"--provider {shlex.quote(self._provider_name)} "
            f"--model {shlex.quote(self._model_name)} "
            f"--api-base {shlex.quote(self._api_base)} "
            f"{mcp_flag}"
            f"--workspace {workspace} "
            f"--trace-output {trace_output} "
            f"--max-iterations {self._max_iterations} "
            "--quiet "
            f"--prompt {escaped_instruction}"
        )
        return [
            TerminalCommand(
                command=command,
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            )
        ]

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        bootstrap = session.container.exec_run(
            [
                "bash",
                "-lc",
                (
                    "set -euo pipefail; "
                    "if ! command -v python3 >/dev/null 2>&1; then "
                    "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv; "
                    "fi; "
                    "if ! python3 -m pip --version >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then "
                    "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv; "
                    "fi"
                ),
            ],
            user="root",
        )
        if bootstrap.exit_code != 0:
            return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

        install_script = self._install_agent_script_path
        copy_paths = [self._wheel_path, install_script]
        if self._mcp_config_path:
            host_mcp_config = Path(self._mcp_config_path).expanduser().resolve()
            if not host_mcp_config.exists():
                raise FileNotFoundError(
                    f"Terminal-Bench MCP config path does not exist: {host_mcp_config}"
                )
            copy_paths.append(host_mcp_config)
        session.copy_to_container(paths=copy_paths, container_dir="/installed-agent")

        env_setup_content = self._create_env_setup_file()
        session.container.exec_run(
            [
                "sh",
                "-c",
                (
                    f"echo {shlex.quote(env_setup_content)} > "
                    "/installed-agent/setup-env.sh"
                ),
            ]
        )

        session.send_keys(["source /installed-agent/setup-env.sh", "Enter"], block=True)
        session.send_keys(
            [
                (
                    "source /installed-agent/" + install_script.name +
                    " || echo 'INSTALL_FAIL_STATUS'"
                ),
                "Enter",
            ],
            block=True,
            max_timeout_sec=float("inf"),
        )
        installation_output = session.capture_pane(capture_entire=True)
        if "INSTALL_FAIL_STATUS" in installation_output.splitlines():
            return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

        rendered_instruction = self._render_instruction(instruction)
        for command in self._run_agent_commands(rendered_instruction):
            session.send_command(command)
        if logging_dir is not None:
            marker_path = logging_dir / "openclaw-complete.marker"
            marker_path.write_text("completed", encoding="utf-8")
        return AgentResult(total_input_tokens=0, total_output_tokens=0)

    @property
    def _container_mcp_config_path(self) -> str:
        if not self._mcp_config_path:
            raise ValueError("mcp_config_path is not configured")
        return f"/installed-agent/{Path(self._mcp_config_path).name}"
