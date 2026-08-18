#!/bin/bash
# benchmark_server.sh — environment setup for a freshly rented box.
#
# Scope: ONE step inside the provider's setup script. Assumes the repo is
# already cloned and this script is run from anywhere inside it. Does NOT
# clone, configure ssh, or manage API keys.
#
# Usage:
#   bash scripts/setup/benchmark_server.sh              # CPU/analysis box
#   bash scripts/setup/benchmark_server.sh --gpu        # + serving-spike extra (vllm)
#   bash scripts/setup/benchmark_server.sh --gpu --prefetch-model meta-llama/Llama-3.1-8B-Instruct
#
# Every step below exists because its absence broke a real session:
#   * uv missing on rental images
#   * root-owned ~/.config -> vllm PermissionError + uv fish-config error
#   * uv venvs ship without setuptools -> vllm pynccl import crash
#   * silent Gen4-vs-Gen5 PCIe surprises -> print the platform up front
set -euo pipefail

GPU=0
PREFETCH_MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=1 ;;
    --prefetch-model) PREFETCH_MODEL="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- locate repo root (script may be invoked from anywhere) -----------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[ -f pyproject.toml ] || { echo "FATAL: pyproject.toml not found at $REPO_ROOT" >&2; exit 1; }
echo "== repo: $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

# --- privileged eBPF runtime -------------------------------------------------
if ! PYTHONPATH=/usr/lib/python3/dist-packages python3 -c 'import bcc' 2>/dev/null; then
  sudo -n apt-get update -qq
  sudo -n apt-get install -y python3-bpfcc
fi
echo "== bcc: $(PYTHONPATH=/usr/lib/python3/dist-packages python3 -c 'import bcc; print(bcc.__file__)')"

# --- writable config home (rental images often ship root-owned ~/.config) ---
if [ ! -w "${XDG_CONFIG_HOME:-$HOME/.config}" ]; then
  export XDG_CONFIG_HOME="$HOME/.xdg"
  mkdir -p "$XDG_CONFIG_HOME"
  grep -q "XDG_CONFIG_HOME=" "$HOME/.bashrc" 2>/dev/null || \
    printf '\nexport XDG_CONFIG_HOME="$HOME/.xdg"\n' >> "$HOME/.bashrc"
  echo "== ~/.config not writable -> XDG_CONFIG_HOME=$XDG_CONFIG_HOME (persisted to .bashrc)"
fi

# --- uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  # installer may fail writing shell completions on locked-down images; the
  # binary lands anyway, so tolerate a nonzero exit and verify the binary.
  curl -LsSf https://astral.sh/uv/install.sh | sh || true
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv install failed" >&2; exit 1; }
echo "== uv: $(uv --version)"

# --- python 3.12 venv + project deps (pyproject/uv.lock are the spec) --------
uv python install 3.12
[ -d .venv ] || uv venv -p 3.12 -q
uv sync --quiet
# uv venvs omit setuptools; vllm (and some tooling) imports it at runtime.
uv pip install -q setuptools
echo "== python: $(.venv/bin/python --version)"

# --- GPU extra ----------------------------------------------------------------
if [ "$GPU" = "1" ]; then
  command -v nvidia-smi >/dev/null || { echo "FATAL: --gpu but no nvidia-smi" >&2; exit 1; }
  uv sync --quiet --extra serving-spike   # exact vllm pin lives in pyproject
  .venv/bin/python - <<'PY'
import torch, vllm
assert torch.cuda.is_available(), "CUDA not available in torch"
print(f"== vllm {vllm.__version__}, torch {torch.__version__}, cuda ok")
PY
  # platform summary — Gen4 vs Gen5 changes transfer economics; say it loudly.
  nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max,driver_version \
    --format=csv,noheader | sed 's/^/== gpu: /'
  GEN=$(nvidia-smi --query-gpu=pcie.link.gen.max --format=csv,noheader | head -1 | tr -d ' ')
  if [ "$GEN" -lt 5 ] 2>/dev/null; then
    # bandwidth doubles per gen: x16 theoretical ~= 63 GB/s at Gen5
    CEIL=$((63 >> (5 - GEN)))
    echo "== NOTE: PCIe Gen${GEN} host (Gen5 needs Sapphire Rapids/Genoa). x16 theoretical ceiling ~${CEIL} GB/s."
  fi
fi

# --- optional model prefetch (requires HF token on the box) -------------------
if [ -n "$PREFETCH_MODEL" ]; then
  .venv/bin/python - "$PREFETCH_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(f"== prefetching {sys.argv[1]} ...")
snapshot_download(sys.argv[1])
print("== model cached")
PY
fi

echo "== setup done. activate with: source .venv/bin/activate"
