from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def run_command(command: list[str]) -> dict[str, Any]:
    """Run a command and capture stdout/stderr without raising on failure."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "command not found",
        }

    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_version(version_text: str) -> tuple[int, ...]:
    """Parse a dotted version string into an integer tuple."""
    parts = []
    for token in version_text.split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts)


def detect_total_memory_bytes() -> int | None:
    """Return total physical memory when the platform exposes it."""
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
        page_size = os.sysconf("SC_PAGE_SIZE")
        if "SC_PHYS_PAGES" in os.sysconf_names:
            return int(page_size * os.sysconf("SC_PHYS_PAGES"))
    return None


def detect_disk_free_bytes(path: Path) -> int:
    """Return free disk bytes for the filesystem containing path."""
    usage = shutil.disk_usage(path)
    return int(usage.free)


def detect_ssh_keys(home: Path) -> list[str]:
    """List public SSH keys present in the user's ~/.ssh directory."""
    ssh_dir = home / ".ssh"
    if not ssh_dir.exists():
        return []
    return sorted(
        path.name
        for path in ssh_dir.iterdir()
        if path.is_file() and path.suffix == ".pub"
    )


def collect_nvidia_report() -> dict[str, Any]:
    """Collect raw GPU information from nvidia-smi."""
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    full = run_command(["nvidia-smi"])
    gpus: list[dict[str, Any]] = []
    if query["available"] and query["returncode"] == 0:
        for line in query["stdout"].splitlines():
            if not line.strip():
                continue
            name, driver_version, memory_total = [part.strip() for part in line.split(",")]
            gpus.append(
                {
                    "name": name,
                    "driver_version": driver_version,
                    "memory_total_mib": int(memory_total),
                }
            )

    cuda_version = None
    if full["available"] and full["returncode"] == 0:
        match = re.search(r"CUDA Version:\s+([0-9.]+)", full["stdout"])
        if match:
            cuda_version = match.group(1)

    return {
        "query": query,
        "full": full,
        "gpus": gpus,
        "cuda_version": cuda_version,
    }


def collect_venv_runtime(venv_python: Path | None) -> dict[str, Any] | None:
    """Collect python and torch runtime metadata from the repo-local venv."""
    if venv_python is None:
        return None

    command = [
        str(venv_python),
        "-c",
        (
            "from __future__ import annotations\n"
            "import json\n"
            "import platform\n"
            "import sys\n"
            "payload = {\n"
            "  'executable': sys.executable,\n"
            "  'python_version': platform.python_version(),\n"
            "  'torch_installed': False,\n"
            "}\n"
            "try:\n"
            "  import torch\n"
            "except ImportError:\n"
            "  pass\n"
            "else:\n"
            "  payload.update({\n"
            "    'torch_installed': True,\n"
            "    'torch_version': torch.__version__,\n"
            "    'torch_cuda_version': torch.version.cuda,\n"
            "    'torch_cuda_available': torch.cuda.is_available(),\n"
            "    'torch_device_count': torch.cuda.device_count(),\n"
            "    'torch_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
            "  })\n"
            "print(json.dumps(payload))\n"
        ),
    ]
    runtime = run_command(command)
    runtime["requested_executable"] = str(venv_python)
    if runtime["available"] and runtime["returncode"] == 0 and runtime["stdout"]:
        runtime["metadata"] = json.loads(runtime["stdout"])
    else:
        runtime["metadata"] = None
    return runtime


def build_report(
    repo_root: Path,
    venv_python: Path | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Collect the environment signals required by ENV-1."""
    home = Path.home()
    return {
        "timestamp": int(time.time()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "repo_root": str(repo_root.resolve()),
        "memory_total_bytes": detect_total_memory_bytes(),
        "disk_free_bytes": detect_disk_free_bytes(repo_root),
        "ssh_public_keys": detect_ssh_keys(home),
        "config": config,
        "commands": {
            "nvidia_smi": collect_nvidia_report(),
            "uv": run_command(["uv", "--version"]),
            "python3": run_command(["python3", "--version"]),
        },
        "venv_runtime": collect_venv_runtime(venv_python),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a JSON report for the ENV-1 server prerequisites."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSON file to write.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for disk-space inspection.",
    )
    parser.add_argument(
        "--venv-python",
        help="Optional repo-local Python executable to inspect for torch/CUDA metadata.",
    )
    parser.add_argument(
        "--expected-gpu-substring",
        default="A100-SXM-40GB",
        help="Required substring for at least one detected GPU name.",
    )
    parser.add_argument(
        "--min-gpu-memory-gib",
        type=int,
        default=40,
        help="Minimum GPU memory requirement in GiB.",
    )
    parser.add_argument(
        "--min-cuda-version",
        default="12.1",
        help="Minimum CUDA version reported by nvidia-smi.",
    )
    parser.add_argument(
        "--torch-package",
        default="torch",
        help="Torch package spec used during setup; persisted for auditability.",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu121",
        help="Torch index URL used during setup; persisted for auditability.",
    )
    parser.add_argument(
        "--require-torch-cuda",
        action="store_true",
        help="Fail validation if torch CUDA is unavailable in the inspected venv.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when the collected report does not satisfy the requested constraints.",
    )
    return parser.parse_args()


def validate_report(report: dict[str, Any], require_torch_cuda: bool) -> list[str]:
    """Validate the report against the ENV-1 acceptance requirements."""
    errors: list[str] = []
    config = report["config"]
    nvidia = report["commands"]["nvidia_smi"]
    gpus = nvidia["gpus"]

    if not gpus:
        errors.append("nvidia-smi did not return any GPU records")
    else:
        if not any(config["expected_gpu_substring"] in gpu["name"] for gpu in gpus):
            errors.append(
                f"no GPU matched expected substring {config['expected_gpu_substring']!r}"
            )
        if max(gpu["memory_total_mib"] for gpu in gpus) < config["min_gpu_memory_gib"] * 1024:
            errors.append(
                f"maximum GPU memory was below {config['min_gpu_memory_gib']} GiB"
            )

    cuda_version = nvidia["cuda_version"]
    if cuda_version is None:
        errors.append("could not parse CUDA version from nvidia-smi output")
    elif parse_version(cuda_version) < parse_version(config["min_cuda_version"]):
        errors.append(
            f"CUDA version {cuda_version} is below required {config['min_cuda_version']}"
        )

    runtime = report["venv_runtime"]
    if runtime is None or runtime["metadata"] is None:
        errors.append("repo-local venv runtime metadata was not collected")
    elif require_torch_cuda:
        metadata = runtime["metadata"]
        if not metadata["torch_installed"]:
            errors.append("torch is not installed in the repo-local venv")
        elif not metadata["torch_cuda_available"]:
            errors.append("torch.cuda.is_available() returned False")
        elif metadata["torch_device_name"] is None or report["config"]["expected_gpu_substring"] not in metadata["torch_device_name"]:
            errors.append("torch runtime did not report the expected GPU model")

    return errors


def main() -> None:
    args = parse_args()
    config = {
        "expected_gpu_substring": args.expected_gpu_substring,
        "min_gpu_memory_gib": args.min_gpu_memory_gib,
        "min_cuda_version": args.min_cuda_version,
        "torch_package": args.torch_package,
        "torch_index_url": args.torch_index_url,
    }
    venv_python = Path(args.venv_python) if args.venv_python else None
    report = build_report(Path(args.repo_root), venv_python, config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_mismatch:
        errors = validate_report(report, require_torch_cuda=args.require_torch_cuda)
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
