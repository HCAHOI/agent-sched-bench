from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class VLLMServerConfig:
    """Configuration for launching the raw vLLM OpenAI-compatible server."""

    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "float16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    enable_chunked_prefill: bool = True
    preemption_mode: str = "recompute"
    max_num_seqs: int = 256


def build_vllm_command(config: VLLMServerConfig) -> list[str]:
    """Build the raw vLLM api_server command from structured config."""
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        f"{config.gpu_memory_utilization:.2f}",
        "--preemption-mode",
        config.preemption_mode,
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]
    if config.enable_chunked_prefill:
        command.append("--enable-chunked-prefill")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch a raw vLLM API server command."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--preemption-mode", default="recompute")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved command as JSON instead of replacing the process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VLLMServerConfig(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=args.enable_chunked_prefill,
        preemption_mode=args.preemption_mode,
        max_num_seqs=args.max_num_seqs,
    )
    command = build_vllm_command(config)
    if args.print_only:
        print(json.dumps({"config": asdict(config), "command": command}, indent=2))
        return
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
