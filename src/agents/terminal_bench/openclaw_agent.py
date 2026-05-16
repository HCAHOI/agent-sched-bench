from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession


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
    VENV_PATH = "/installed-agent/venv"
    _WHEEL_CACHE: Path | None = None
    _CONTAINER_LOCAL_API_HOSTS = {
        "127.0.0.1",
        "localhost",
        "0.0.0.0",
        "172.17.0.1",
        "host.docker.internal",
    }

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
        mcp_config_path: str | None = None,
        temperature: float | str | None = None,
        top_p: float | str | None = None,
        top_k: int | str | None = None,
        repetition_penalty: float | str | None = None,
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
        self._llm_timeout_sec = (
            None if llm_timeout_sec is None else float(llm_timeout_sec)
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
            self._env_key: self._api_key,
            # Exported here so setup-env.sh can rewrite it (gateway resolution)
            # before the openclaw command consumes it. Keeping the rewrite in
            # setup-env.sh — not as a per-command shell prefix — is what keeps
            # the final openclaw line short enough for tmux send-keys +
            # asciinema rec --stdin to handle without breaking sync.
            "OPENCLAW_API_BASE": self._api_base,
        }
        if self._llm_timeout_sec is not None:
            env["OPENCLAW_LLM_TIMEOUT_S"] = str(self._llm_timeout_sec)
        for key in self._ENV_PASSTHROUGH:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _create_env_setup_file(self) -> str:
        """Extend the parent's export lines with gateway-resolution logic.

        Why this lives here and not as a shell prefix on the openclaw command:
        terminal-bench wraps the agent shell in `asciinema rec --stdin`, and
        sending a long multi-line command (prefix + openclaw + multi-line
        prompt) through `tmux send-keys` breaks asciinema's stdin sync —
        asciinema exits early and the openclaw subprocess never runs. Putting
        the resolution in setup-env.sh keeps the openclaw command itself a
        single short line that tmux/asciinema can handle.
        """
        base = super()._create_env_setup_file()
        if not self._should_resolve_api_base_from_container_gateway(self._api_base):
            return base
        # Use /proc/net/route (always present on Linux) instead of `ip route`
        # (iproute2 is missing in many minimal container images, including
        # terminal-bench-core's). Gateway is a little-endian hex IP.
        resolution = (
            "\n"
            "# Resolve OPENCLAW_API_BASE to the container's actual default gateway\n"
            "# when the original points at a host-local placeholder (172.17.0.1\n"
            "# etc.). Terminal-Bench user-defined networks can give the task\n"
            "# container a non-default-bridge gateway.\n"
            "_oc_gw_hex=$(awk '$2==\"00000000\"{print $3; exit}' "
            "/proc/net/route 2>/dev/null)\n"
            'if [ -n "$_oc_gw_hex" ] && [ "$_oc_gw_hex" != 00000000 ]; then\n'
            "  _oc_gw=$(printf '%d.%d.%d.%d' "
            '"$((0x${_oc_gw_hex:6:2}))" "$((0x${_oc_gw_hex:4:2}))" '
            '"$((0x${_oc_gw_hex:2:2}))" "$((0x${_oc_gw_hex:0:2}))" 2>/dev/null)\n'
            '  case "$_oc_gw" in *.*.*.*)\n'
            "    OPENCLAW_API_BASE=$(printf '%s' \"$OPENCLAW_API_BASE\" | "
            'sed -E "s#^(https?://)'
            r"(127\\.0\\.0\\.1|localhost|0\\.0\\.0\\.0|172\\.17\\.0\\.1|host\\.docker\\.internal)"
            '(:|/)#\\\\1${_oc_gw}\\\\3#")\n'
            "    export OPENCLAW_API_BASE\n"
            "  ;; esac\n"
            "fi\n"
            "unset _oc_gw_hex _oc_gw\n"
        )
        return base + resolution

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
        result = subprocess.run(
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
        if result.returncode != 0:
            raise RuntimeError(
                "pip wheel failed (returncode="
                f"{result.returncode}). stdout tail:\n"
                f"{result.stdout[-2000:]}\n--- stderr tail:\n"
                f"{result.stderr[-2000:]}"
            )
        wheels = sorted(wheel_dir.glob("agent_sched_bench-*.whl"))
        if not wheels:
            raise RuntimeError("failed to build agent-sched-bench wheel")
        cls._WHEEL_CACHE = wheels[0]
        return cls._WHEEL_CACHE

    @property
    def _wheel_path(self) -> Path:
        return self._build_wheel()

    @staticmethod
    def _bootstrap_dependencies_command() -> str:
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
            "  apt-get update\n"
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv\n"
            "}\n"
            "if ! command -v python3 >/dev/null 2>&1; then\n"
            "  install_python_deps\n"
            "fi\n"
            "probe_root=$(mktemp -d /tmp/openclaw-venv-check.XXXXXX)\n"
            'cleanup_probe() { rm -rf "$probe_root"; }\n'
            "trap cleanup_probe EXIT\n"
            "venv_ready() {\n"
            '  rm -rf "$probe_root/venv"\n'
            "  python3 -m pip --version >/dev/null 2>&1 && "
            'python3 -m venv "$probe_root/venv" >/dev/null 2>&1 && '
            '"$probe_root/venv/bin/python" -m pip --version >/dev/null 2>&1\n'
            "}\n"
            "if ! venv_ready; then\n"
            "  install_python_deps\n"
            "  venv_ready\n"
            "fi\n"
        )

    @classmethod
    def _should_resolve_api_base_from_container_gateway(cls, api_base: str) -> bool:
        host = urlparse(api_base).hostname
        return host in cls._CONTAINER_LOCAL_API_HOSTS

    def _api_base_shell_prefix(self) -> tuple[str, str]:
        """Return a shell prefix and argv expression for the OpenClaw API base.

        Gateway resolution is done in setup-env.sh (see _create_env_setup_file),
        which is sourced into the container shell long before openclaw runs.
        That keeps the openclaw command line short — important because the
        outer asciinema rec --stdin wrapper drops sync when tmux send-keys
        delivers a very long multi-line command.
        """
        if not self._should_resolve_api_base_from_container_gateway(self._api_base):
            return "", shlex.quote(self._api_base)
        return "", '"${OPENCLAW_API_BASE}"'

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped_instruction = shlex.quote(instruction)
        workspace = shlex.quote(".")
        trace_output = shlex.quote(
            f"{self.CONTAINER_AGENT_LOGS_PATH}/{self.TRACE_FILENAME}"
        )
        api_base_prefix, api_base_arg = self._api_base_shell_prefix()
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
            f"{api_base_prefix}"
            f"{self.VENV_PATH}/bin/openclaw "
            f"--provider {shlex.quote(self._provider_name)} "
            f"--model {shlex.quote(self._model_name)} "
            f"--api-base {api_base_arg} "
            f"{mcp_flag}"
            f"--workspace {workspace} "
            f"--trace-output {trace_output} "
            f"--max-iterations {self._max_iterations} "
            f"{generation_flags}"
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
                self._bootstrap_dependencies_command(),
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
                    "source /installed-agent/"
                    + install_script.name
                    + " || echo 'INSTALL_FAIL_STATUS'"
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
