#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.11.14}"
VENV_DIR="${VENV_DIR:-.venv-server}"
TORCH_PACKAGE="${TORCH_PACKAGE:-torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
REPORT_PATH="${REPORT_PATH:-${REPO_ROOT}/results/processed/env_report.json}"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-A100-SXM-40GB}"
MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-40}"
MIN_CUDA_VERSION="${MIN_CUDA_VERSION:-12.1}"

APT_PACKAGES=(
  git
  tmux
  htop
  nvtop
  jq
  curl
  ca-certificates
)

log() {
  printf '[ENV-1] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

ensure_apt_environment() {
  require_cmd sudo
  require_cmd apt-get
}

install_base_packages() {
  log "Installing system packages: ${APT_PACKAGES[*]}"
  sudo apt-get update
  sudo apt-get install -y "${APT_PACKAGES[@]}"
}

ensure_uv_on_path() {
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
}

install_uv() {
  ensure_uv_on_path
  if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ensure_uv_on_path
  fi
  require_cmd uv
}

install_python() {
  log "Installing Python ${PYTHON_VERSION} via uv"
  uv python install "${PYTHON_VERSION}"
}

ensure_server_venv() {
  local venv_path="${REPO_ROOT}/${VENV_DIR}"
  if [[ ! -x "${venv_path}/bin/python" ]]; then
    log "Creating virtual environment at ${venv_path}"
    uv venv "${venv_path}" --python "${PYTHON_VERSION}"
  fi
}

install_torch() {
  if [[ "${INSTALL_TORCH}" != "1" ]]; then
    log "Skipping torch installation because INSTALL_TORCH=${INSTALL_TORCH}"
    return
  fi
  log "Installing ${TORCH_PACKAGE} from ${TORCH_INDEX_URL}"
  uv pip install \
    --python "${REPO_ROOT}/${VENV_DIR}/bin/python" \
    --index-url "${TORCH_INDEX_URL}" \
    "${TORCH_PACKAGE}"
}

write_and_verify_env_report() {
  log "Collecting and validating the ENV-1 report at ${REPORT_PATH}"
  mkdir -p "$(dirname "${REPORT_PATH}")"
  "${REPO_ROOT}/${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/report_server_env.py" \
    --output "${REPORT_PATH}" \
    --repo-root "${REPO_ROOT}" \
    --venv-python "${REPO_ROOT}/${VENV_DIR}/bin/python" \
    --expected-gpu-substring "${EXPECTED_GPU_SUBSTRING}" \
    --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB}" \
    --min-cuda-version "${MIN_CUDA_VERSION}" \
    --torch-package "${TORCH_PACKAGE}" \
    --torch-index-url "${TORCH_INDEX_URL}" \
    --require-torch-cuda \
    --fail-on-mismatch
}

main() {
  ensure_apt_environment
  install_base_packages
  install_uv
  install_python
  ensure_server_venv
  install_torch
  write_and_verify_env_report
  log "ENV-1 completed successfully"
}

main "$@"
