#!/usr/bin/env bash
# Idempotent setup for Ubuntu GPU servers running Terminal-Bench traces with
# OpenClaw and the HF recording backend.
#
# This script only prepares host state. Project code changes belong in the repo,
# not in this script.
#
# Typical use on a fresh server:
#   git clone <repo> ~/agent-sched-bench
#   cd ~/agent-sched-bench
#   HF_TOKEN=... bash scripts/setup/terminal_bench_server.sh
#
# Tunables:
#   PYTHON_VERSION=3.12
#   VENV_PATH=.venv
#   MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
#   HF_HOME=$HOME/hf_cache
#   INSTALL_TORCH=1
#   PREWARM_MODEL=1
#   TORCH_SPEC=torch==2.6.0+cu124
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   SETUP_DOCKER_FIREWALL=1
#
# Optional mirror settings for slow networks:
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
#   OPENCLAW_APT_MIRROR_PREFIX=https://mirrors.tuna.tsinghua.edu.cn

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_PATH="${VENV_PATH:-.venv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
PREWARM_MODEL="${PREWARM_MODEL:-1}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0+cu124}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
SETUP_DOCKER_FIREWALL="${SETUP_DOCKER_FIREWALL:-1}"

log() {
  printf '[terminal-bench-setup] %s\n' "$*"
}

fatal() {
  printf '[terminal-bench-setup] ERROR: %s\n' "$*" >&2
  exit 1
}

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    fatal "need root privileges for: $*"
  fi
}

target_user() {
  if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    printf '%s\n' "${SUDO_USER}"
  else
    id -un
  fi
}

maybe_apply_apt_mirror() {
  local mirror="${OPENCLAW_APT_MIRROR_PREFIX:-}"
  mirror="${mirror%/}"
  [ -n "${mirror}" ] || return 0

  log "configuring apt mirror prefix: ${mirror}"
  as_root bash -c "
    set -e
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
      [ -f \"\$f\" ] || continue
      sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|${mirror}/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|${mirror}/ubuntu|g' \
        -e 's|http://deb.debian.org/debian|${mirror}/debian|g' \
        -e 's|http://security.debian.org/debian-security|${mirror}/debian-security|g' \
        \"\$f\"
    done
  "
}

install_system_packages() {
  command -v apt-get >/dev/null 2>&1 || fatal "apt-get not found; this setup targets Ubuntu/Debian"
  maybe_apply_apt_mirror
  log "installing system packages"
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    acl \
    build-essential \
    ca-certificates \
    curl \
    git \
    iproute2 \
    iptables \
    jq \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync \
    tmux \
    unzip \
    zstd
}

install_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    log "installing Docker from Ubuntu packages"
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-plugin
  fi

  log "starting Docker service"
  as_root systemctl enable --now docker >/dev/null 2>&1 || as_root service docker start >/dev/null 2>&1 || true

  local user
  user="$(target_user)"
  if getent group docker >/dev/null 2>&1; then
    as_root usermod -aG docker "${user}" || true
  fi
  if [ -S /var/run/docker.sock ] && command -v setfacl >/dev/null 2>&1; then
    as_root setfacl -m "u:${user}:rw" /var/run/docker.sock || true
  fi

  if docker info >/dev/null 2>&1; then
    log "docker is usable by ${user}"
  elif as_root docker info >/dev/null 2>&1; then
    log "docker works via root; group membership may require a new login"
  else
    fatal "docker daemon is not usable"
  fi
}

configure_docker_bridge_firewall() {
  [ "${SETUP_DOCKER_FIREWALL}" = "1" ] || return 0
  command -v iptables >/dev/null 2>&1 || return 0

  log "allowing Docker bridge traffic to host services"
  as_root bash -c '
    set -e
    iptables -C INPUT -i br+ -p tcp -j ACCEPT 2>/dev/null || \
      iptables -I INPUT 1 -i br+ -p tcp -j ACCEPT
    iptables -C INPUT -i docker+ -p tcp -j ACCEPT 2>/dev/null || \
      iptables -I INPUT 1 -i docker+ -p tcp -j ACCEPT
  ' || log "warning: could not update iptables; set HF_RECORDING_PUBLIC_HOST manually if containers cannot connect"
}

install_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(uv --version)"
    return 0
  fi
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || fatal "uv install did not put uv on PATH"
}

setup_python_env() {
  export PATH="${HOME}/.local/bin:${PATH}"
  log "creating/updating ${VENV_PATH} with Python ${PYTHON_VERSION}"
  uv python find "${PYTHON_VERSION}" >/dev/null 2>&1 || uv python install "${PYTHON_VERSION}"
  uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --seed

  local py="${VENV_PATH}/bin/python"
  [ -x "${py}" ] || fatal "venv python missing at ${py}"

  uv pip install --python "${py}" pip wheel
  uv pip install --python "${py}" -e ".[dev]"
  "${py}" -m pip --version
}

install_torch() {
  [ "${INSTALL_TORCH}" = "1" ] || return 0
  local py="${VENV_PATH}/bin/python"
  if "${py}" - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
  then
    log "torch with CUDA is already available"
  else
    log "installing ${TORCH_SPEC} from ${TORCH_INDEX_URL}"
    uv pip install --python "${py}" --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
  fi

  "${py}" - <<'PY'
import torch
print(f"[terminal-bench-setup] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[terminal-bench-setup] gpu={torch.cuda.get_device_name(0)}")
PY
}

prefetch_terminal_bench() {
  local py="${VENV_PATH}/bin/python"
  log "loading Terminal-Bench registry and pinned dataset"
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "${py}" - <<'PY'
from pathlib import Path
from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig

config = BenchmarkConfig.from_yaml(Path("configs/benchmarks/terminal-bench.yaml"))
benchmark = get_benchmark_class(config.slug)(config)
tasks = benchmark.load_tasks()
ids = {task["instance_id"] for task in tasks}
print(f"[terminal-bench-setup] loaded_tasks={len(tasks)}")
if "causal-inference-r" not in ids:
    raise SystemExit("causal-inference-r missing from Terminal-Bench dataset")
PY
}

prewarm_model() {
  [ "${PREWARM_MODEL}" = "1" ] || return 0
  local py="${VENV_PATH}/bin/python"
  export HF_HOME
  mkdir -p "${HF_HOME}"

  log "prefetching ${MODEL_ID} into HF_HOME=${HF_HOME}"
  "${py}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
snapshot_download(
    repo_id=model_id,
    token=os.environ.get("HF_TOKEN"),
)
print(f"[terminal-bench-setup] model_cached={model_id}")
PY
}

verify_setup() {
  local py="${VENV_PATH}/bin/python"
  log "verifying setup"
  "${py}" - <<'PY'
import importlib.metadata as md
import sys

print(f"[terminal-bench-setup] python={sys.version.split()[0]}")
for package in ("agent-sched-bench", "terminal-bench", "transformers"):
    print(f"[terminal-bench-setup] {package}={md.version(package)}")
PY
  docker version --format '[terminal-bench-setup] docker client={{.Client.Version}} server={{.Server.Version}}' || true
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "warning: nvidia-smi not available"
}

print_next_steps() {
  cat <<EOF

[terminal-bench-setup] DONE

Next baseline launch example:
  cd ${REPO_ROOT}
  source ${VENV_PATH}/bin/activate
  export HF_HOME=${HF_HOME}
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  INSTANCE_ID=causal-inference-r \\
  MODEL_ID=${MODEL_ID} \\
  ./scripts/launch_kv_capstone.sh none baseline-causal-inference-r

If Docker group changes were just applied, start a new SSH session before
running long experiments unless this script reported Docker is already usable.
EOF
}

export MODEL_ID

install_system_packages
install_docker
configure_docker_bridge_firewall
install_uv
setup_python_env
install_torch
prefetch_terminal_bench
prewarm_model
verify_setup
print_next_steps
