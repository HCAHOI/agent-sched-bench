#!/usr/bin/env bash
set -euo pipefail

echo "ENV-1 server bootstrap script is not executable during BOOTSTRAP-0." >&2
echo "Run after checkpoint approval on the target server:" >&2
echo "  sudo apt update && sudo apt install -y git tmux htop nvtop jq" >&2
echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
echo "  uv python install 3.11" >&2
echo "  nvidia-smi" >&2
echo "  python -c \"import torch; print(torch.cuda.get_device_name(0))\"" >&2
exit 1
