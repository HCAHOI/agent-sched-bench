#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTINUUM_REPO_URL="${CONTINUUM_REPO_URL:-https://github.com/Hanchenli/vllm-continuum.git}"
CONTINUUM_REF="${CONTINUUM_REF:-REPLACE_WITH_COMMIT_OR_TAG}"
CONTINUUM_REPO_DIR="${CONTINUUM_REPO_DIR:-${REPO_ROOT}/external/vllm-continuum}"
CONTINUUM_VENV_DIR="${CONTINUUM_VENV_DIR:-${REPO_ROOT}/.venv-continuum}"
CONTINUUM_PYTHON="${CONTINUUM_VENV_DIR}/bin/python"

MODEL_PATH="${MODEL_PATH:-/data/models/Llama-3.1-8B-Instruct}"
CONTINUUM_PORT="${CONTINUUM_PORT:-8001}"
CONTINUUM_TENSOR_PARALLEL_SIZE="${CONTINUUM_TENSOR_PARALLEL_SIZE:-1}"
CONTINUUM_USE_CPU_OFFLOAD="${CONTINUUM_USE_CPU_OFFLOAD:-0}"
CONTINUUM_LMCACHE_CPU_GIB="${CONTINUUM_LMCACHE_CPU_GIB:-200}"
CONTINUUM_LMCACHE_SPEC="${CONTINUUM_LMCACHE_SPEC:-lmcache==0.3.7}"
CONTINUUM_HF_TRANSFER_SPEC="${CONTINUUM_HF_TRANSFER_SPEC:-hf_transfer>=0.1,<1.0}"
CONTINUUM_HEALTH_TIMEOUT_S="${CONTINUUM_HEALTH_TIMEOUT_S:-180}"
CONTINUUM_POLL_INTERVAL_S="${CONTINUUM_POLL_INTERVAL_S:-2.0}"
CONTINUUM_PROGRAM_ID="${CONTINUUM_PROGRAM_ID:-continuum-smoke}"
CONTINUUM_LOG_PATH="${CONTINUUM_LOG_PATH:-${REPO_ROOT}/results/processed/continuum_server.log}"
CONTINUUM_REPORT_PATH="${CONTINUUM_REPORT_PATH:-${REPO_ROOT}/results/processed/continuum_server_report.json}"
CONTINUUM_INSTALL_REPORT="${CONTINUUM_INSTALL_REPORT:-${REPO_ROOT}/results/processed/continuum_install_report.json}"

log() {
  printf '[ENV-3b] %s\n' "$*"
}

require_pinned_ref() {
  case "${CONTINUUM_REF}" in
    main|master|HEAD|REPLACE_WITH_COMMIT_OR_TAG)
      printf 'CONTINUUM_REF must be an immutable commit or tag, not %s\n' "${CONTINUUM_REF}" >&2
      exit 1
      ;;
  esac
}

require_model_path() {
  if [[ ! -d "${MODEL_PATH}" ]]; then
    printf 'Model path does not exist: %s\n' "${MODEL_PATH}" >&2
    printf 'Run ENV-2 successfully before ENV-3b.\n' >&2
    exit 1
  fi
}

ensure_repo() {
  mkdir -p "$(dirname "${CONTINUUM_REPO_DIR}")"
  if [[ ! -d "${CONTINUUM_REPO_DIR}/.git" ]]; then
    log "Cloning Continuum repo from ${CONTINUUM_REPO_URL}"
    git clone "${CONTINUUM_REPO_URL}" "${CONTINUUM_REPO_DIR}"
  fi
  git -C "${CONTINUUM_REPO_DIR}" fetch origin
  git -C "${CONTINUUM_REPO_DIR}" checkout "${CONTINUUM_REF}"
}

ensure_venv() {
  if [[ ! -x "${CONTINUUM_PYTHON}" ]]; then
    log "Creating Continuum virtualenv at ${CONTINUUM_VENV_DIR}"
    uv venv "${CONTINUUM_VENV_DIR}" --python 3.11.14
  fi
}

install_continuum() {
  log "Installing Continuum and dependencies"
  uv pip install --python "${CONTINUUM_PYTHON}" -e "${CONTINUUM_REPO_DIR}"
  uv pip install --python "${CONTINUUM_PYTHON}" "${CONTINUUM_LMCACHE_SPEC}" "${CONTINUUM_HF_TRANSFER_SPEC}"
}

write_install_report() {
  log "Writing Continuum install report to ${CONTINUUM_INSTALL_REPORT}"
  mkdir -p "$(dirname "${CONTINUUM_INSTALL_REPORT}")"
  CONTINUUM_REPO_DIR="${CONTINUUM_REPO_DIR}" \
  CONTINUUM_REPO_URL="${CONTINUUM_REPO_URL}" \
  CONTINUUM_REF="${CONTINUUM_REF}" \
  CONTINUUM_LMCACHE_SPEC="${CONTINUUM_LMCACHE_SPEC}" \
  CONTINUUM_HF_TRANSFER_SPEC="${CONTINUUM_HF_TRANSFER_SPEC}" \
  CONTINUUM_INSTALL_REPORT="${CONTINUUM_INSTALL_REPORT}" \
  "${CONTINUUM_PYTHON}" - <<'PY'
from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


def safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


repo_dir = Path(os.environ["CONTINUUM_REPO_DIR"])
payload = {
    "repo_url": os.environ["CONTINUUM_REPO_URL"],
    "requested_ref": os.environ["CONTINUUM_REF"],
    "requested_specs": {
        "lmcache": os.environ["CONTINUUM_LMCACHE_SPEC"],
        "hf_transfer": os.environ["CONTINUUM_HF_TRANSFER_SPEC"],
    },
    "resolved_commit": subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "installed_versions": {
        "vllm": safe_version("vllm"),
        "lmcache": safe_version("lmcache"),
        "hf_transfer": safe_version("hf_transfer"),
    },
}
Path(os.environ["CONTINUUM_INSTALL_REPORT"]).write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

start_server() {
  mkdir -p "$(dirname "${CONTINUUM_LOG_PATH}")"
  log "Starting Continuum server; logs -> ${CONTINUUM_LOG_PATH}"
  if [[ "${CONTINUUM_USE_CPU_OFFLOAD}" == "1" ]]; then
    LMCACHE_MAX_LOCAL_CPU_SIZE="${CONTINUUM_LMCACHE_CPU_GIB}" \
      PYTHONPATH="${REPO_ROOT}/src" "${CONTINUUM_PYTHON}" -m serving.continuum_launcher \
      --model-path "${MODEL_PATH}" \
      --port "${CONTINUUM_PORT}" \
      --tensor-parallel-size "${CONTINUUM_TENSOR_PARALLEL_SIZE}" \
      --enable-cpu-offload \
      --cpu-offload-gib "${CONTINUUM_LMCACHE_CPU_GIB}" \
      >>"${CONTINUUM_LOG_PATH}" 2>&1 &
  else
    PYTHONPATH="${REPO_ROOT}/src" "${CONTINUUM_PYTHON}" -m serving.continuum_launcher \
      --model-path "${MODEL_PATH}" \
      --port "${CONTINUUM_PORT}" \
      --tensor-parallel-size "${CONTINUUM_TENSOR_PARALLEL_SIZE}" \
      >>"${CONTINUUM_LOG_PATH}" 2>&1 &
  fi
  SERVER_PID=$!
  log "Continuum server pid=${SERVER_PID}"
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    log "Stopping Continuum server pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

verify_server() {
  log "Running Continuum readiness checks"
  PYTHONPATH="${REPO_ROOT}/src" "${CONTINUUM_PYTHON}" -m serving.health_check \
    --api-base "http://127.0.0.1:${CONTINUUM_PORT}/v1" \
    --metrics-url "http://127.0.0.1:${CONTINUUM_PORT}/metrics" \
    --model auto \
    --timeout-s "${CONTINUUM_HEALTH_TIMEOUT_S}" \
    --poll-interval-s "${CONTINUUM_POLL_INTERVAL_S}" \
    --prompt "Reply with the word CONTINUUM." \
    --followup-prompt "Continue the same conversation and reply with CONTINUUM-AGAIN." \
    --program-id "${CONTINUUM_PROGRAM_ID}" \
    --repeat 2 \
    --output "${CONTINUUM_REPORT_PATH}" \
    --vllm-spec "continuum@${CONTINUUM_REF}" \
    --model-path "${MODEL_PATH}" \
    --require-prefix-cache-hit \
    --fail-on-mismatch
}

main() {
  require_pinned_ref
  require_model_path
  ensure_repo
  ensure_venv
  install_continuum
  write_install_report
  trap cleanup EXIT INT TERM
  start_server
  verify_server
  log "ENV-3b completed successfully; server is running. Press Ctrl-C to stop."
  wait "${SERVER_PID}"
}

main "$@"
