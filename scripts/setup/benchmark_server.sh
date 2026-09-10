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
#   bash scripts/setup/benchmark_server.sh --serving-host [--verify]   # 2xL40S host for the two-instance runs
#
# --serving-host builds the GPU-side environment used by
# scripts/evaluation/run_two_instance_fcfs.sh (Continuum, DualMap, PPD,
# ThunderAgent checkouts + venvs, the model, CUDA 13 compat) under /workspace,
# because Vast regenerates the home directory on every container start. It
# needs no docker and is idempotent; --verify only re-runs the checks.
#
# Every step below exists because its absence broke a real session:
#   * uv missing on rental images
#   * root-owned ~/.config -> vllm PermissionError + uv fish-config error
#   * uv venvs ship without setuptools -> vllm pynccl import crash
#   * silent Gen4-vs-Gen5 PCIe surprises -> print the platform up front
set -euo pipefail

GPU=0
SERVING_HOST=0
VERIFY_ONLY=0
PREFETCH_MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=1 ;;
    --serving-host) SERVING_HOST=1 ;;
    --verify) VERIFY_ONLY=1 ;;
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

# --- two-instance serving host -------------------------------------------------
if [ "$SERVING_HOST" = "1" ]; then
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/.cache} HF_HOME=${HF_HOME:-/workspace/.hf_home}
  export CONTINUUM_PYTHON=${CONTINUUM_PYTHON:-/usr/bin/python3.12} PPD_NATIVE_PUSH=1
  C=$XDG_CACHE_HOME/agent-sched-bench
  TA=7ddc8610270e56d3b109eed8796b3a4360fc67c9
  log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
  trap 'log "SETUP FAILED at line $LINENO"' ERR
  verify_serving_host() {
    log "verify"
    bash scripts/baselines/continuum_public.sh verify
    bash scripts/baselines/dualmap_official.sh verify
    bash scripts/baselines/ppd_official.sh verify-installed
    bash scripts/baselines/thunderagent_official.sh verify
    check() { local venv=$1 pp=$2; shift 2
      PYTHONPATH=$pp "$venv/bin/python" - "$@" <<'PY'
import importlib, sys
print("  " + " ".join(f"{m}={getattr(importlib.import_module(m), '__version__', 'ok')}" for m in sys.argv[1:]))
PY
    }
    check "$C/venvs/continuum-public-316a58794a6ff86b216e579b74fd56ed0c5a911f" "" vllm transformers lmcache fastapi zmq msgspec
    check "$C/DualMap-24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d/.venv" "$C/DualMap-24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d" dualmap transformers uhashring
    check "$C/PPD-28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367/.venv" "" vllm nixl
    for v in "$C/ThunderAgent-$TA/.venv" /workspace/venvs/thunderagent-pending-release /workspace/venvs/thunderagent-capacity-consistent; do check "$v" "" ThunderAgent; done
    blob=$(readlink -f "$HF_HOME"/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/8591804019c8b22094c3b5b4454e0edc05dffc98/model.safetensors)
    [ "$(stat -c %s "$blob")" -gt 5000000000 ] && log "  model blob $(stat -c %s "$blob") bytes"
    [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c L40S)" = 2 ] && log "  2x L40S visible"
    log "VERIFY OK"
  }
  if [ "$VERIFY_ONLY" = "1" ]; then verify_serving_host; exit; fi
  log "cuda-compat-13-0 (vLLM 0.28 cu13 wheels on driver 570)"
  dpkg -s cuda-compat-13-0 >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq cuda-compat-13-0; }
  log "model Qwen/Qwen3-4B-Instruct-2507-FP8 @ 8591804019c8"
  uvx --from huggingface_hub hf download Qwen/Qwen3-4B-Instruct-2507-FP8 --revision 8591804019c8b22094c3b5b4454e0edc05dffc98
  log "continuum public (vllm 0.10.2 + overlay)"
  bash scripts/baselines/continuum_public.sh install
  log "dualmap"
  bash scripts/baselines/dualmap_official.sh install
  log "ppd native push (vllm 0.28.0 + nixl 1.4.1)"
  bash scripts/baselines/ppd_official.sh install
  log "thunderagent base"
  [ -x "$C/ThunderAgent-$TA/.venv/bin/thunderagent" ] || bash scripts/baselines/thunderagent_official.sh install
  log "thunderagent patched variants"
  for v in pending_release capacity_consistent; do
    dir=/workspace/ThunderAgent-${v//_/-}-7ddc861; patch=$PWD/scripts/baselines/thunderagent_$v.patch
    [ -e "$dir" ] || { git clone -q "$C/ThunderAgent-$TA" "$dir"; git -C "$dir" checkout -q --detach "$TA"; }
    git -C "$dir" remote set-url origin https://github.com/ThunderAgent-org/ThunderAgent.git
    git -C "$dir" apply --reverse --check "$patch" 2>/dev/null || git -C "$dir" apply "$patch"
    [ -x /workspace/venvs/thunderagent-${v//_/-}/bin/thunderagent ] || \
      THUNDERAGENT_CHECKOUT=$dir THUNDERAGENT_VENV=/workspace/venvs/thunderagent-${v//_/-} bash scripts/baselines/thunderagent_official.sh install
  done
  verify_serving_host
  log "SETUP COMPLETE"
  exit
fi

# --- container runtime -------------------------------------------------------
command -v docker >/dev/null || {
  echo "FATAL: docker is required; install it before running benchmark setup" >&2
  exit 1
}
if ! id -nG | tr ' ' '\n' | grep -qx docker; then
  sudo -n usermod -aG docker "$(id -un)"
  echo "FATAL: added $(id -un) to the docker group; reconnect and rerun setup" >&2
  exit 1
fi
docker info >/dev/null || {
  echo "FATAL: docker daemon is unavailable to $(id -un)" >&2
  exit 1
}
echo "== docker: $(docker version --format '{{.Server.Version}}')"

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
  if command -v systemctl >/dev/null && systemctl is-active --quiet nvidia-dcgm; then
    sudo -n systemctl disable --now nvidia-dcgm >/dev/null
  fi
  uv sync --quiet --extra serving-spike   # exact vllm pin lives in pyproject
  CUPTI_DRAM_OVERLAY="${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/cupti-dram-13.3.1"
  CUPTI_DRAM_LIB="$CUPTI_DRAM_OVERLAY/nvidia/cu13/lib"
  if [ ! -f "$CUPTI_DRAM_LIB/libcupti.so.13" ]; then
    mkdir -p "$CUPTI_DRAM_OVERLAY"
    uv pip install --target "$CUPTI_DRAM_OVERLAY" --no-deps \
      'cupti-python==13.3.1' 'nvidia-cuda-cupti==13.3.75' \
      'cuda-pathfinder==1.8.0' 'pyelftools==0.33'
  fi
  if [ ! -d "$CUPTI_DRAM_OVERLAY/cuda-bindings/cuda/bindings" ]; then
    mkdir -p "$CUPTI_DRAM_OVERLAY/cuda-bindings"
    uv pip install --target "$CUPTI_DRAM_OVERLAY/cuda-bindings" --no-deps \
      'cuda-bindings==13.3.1'
  fi
  PYTHONPATH="$CUPTI_DRAM_OVERLAY" LD_LIBRARY_PATH="$CUPTI_DRAM_LIB" \
    .venv/bin/python -c 'from cupti.pm_sampling import Collector'
  command -v setpriv >/dev/null || { echo "FATAL: setpriv is required for CUPTI" >&2; exit 1; }
  setpriv --list-caps | grep -Eq '^(perfmon|cap_38)$' || {
    echo "FATAL: setpriv cannot name CAP_PERFMON" >&2
    exit 1
  }
  .venv/bin/python - <<'PY'
import torch, vllm
assert torch.cuda.is_available(), "CUDA not available in torch"
print(f"== vllm {vllm.__version__}, torch {torch.__version__}, cuda ok")
PY
  echo "== cupti: 13.3.1 context DRAM byte sampling ready"
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
