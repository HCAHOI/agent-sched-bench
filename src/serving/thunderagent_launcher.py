from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ThunderAgentConfig:
    """Configuration for launching the ThunderAgent proxy."""

    backends: str
    host: str = "0.0.0.0"
    port: int = 9000
    backend_type: str = "vllm"
    profile: bool = True
    metrics: bool = True
    profile_dir: str = "/tmp/thunderagent_profiles"


def build_thunderagent_command(config: ThunderAgentConfig) -> list[str]:
    """Build the ThunderAgent CLI command from structured config."""
    thunderagent_executable = str(Path(sys.executable).with_name("thunderagent"))
    command = [
        thunderagent_executable,
        "--backend-type",
        config.backend_type,
        "--backends",
        config.backends,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--profile-dir",
        config.profile_dir,
    ]
    if config.profile:
        command.append("--profile")
    if config.metrics:
        command.append("--metrics")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch a ThunderAgent proxy command."
    )
    parser.add_argument("--backends", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--backend-type", default="vllm")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--profile-dir", default="/tmp/thunderagent_profiles")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ThunderAgentConfig(
        backends=args.backends,
        host=args.host,
        port=args.port,
        backend_type=args.backend_type,
        profile=args.profile,
        metrics=args.metrics,
        profile_dir=args.profile_dir,
    )
    command = build_thunderagent_command(config)
    if args.print_only:
        print(json.dumps({"config": asdict(config), "command": command}, indent=2))
        return
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
