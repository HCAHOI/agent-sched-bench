#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

THUNDERAGENT_REPO_URL="${THUNDERAGENT_REPO_URL:-https://github.com/ThunderAgent-org/ThunderAgent.git}"
THUNDERAGENT_REF="${THUNDERAGENT_REF:-REPLACE_WITH_COMMIT_OR_TAG}"
THUNDERAGENT_REPO_DIR="${THUNDERAGENT_REPO_DIR:-${REPO_ROOT}/external/ThunderAgent}"
THUNDERAGENT_VENV_DIR="${THUNDERAGENT_VENV_DIR:-${REPO_ROOT}/.venv-thunderagent}"
THUNDERAGENT_PYTHON="${THUNDERAGENT_VENV_DIR}/bin/python"
THUNDERAGENT_EXTRA_SPEC="${THUNDERAGENT_EXTRA_SPEC:-uvicorn>=0.30,<1.0}"

THUNDERAGENT_BACKENDS="${THUNDERAGENT_BACKENDS:-http://127.0.0.1:8000}"
THUNDERAGENT_PORT="${THUNDERAGENT_PORT:-9000}"
THUNDERAGENT_HOST="${THUNDERAGENT_HOST:-0.0.0.0}"
THUNDERAGENT_PROFILE_DIR="${THUNDERAGENT_PROFILE_DIR:-${REPO_ROOT}/results/processed/thunderagent_profiles}"
THUNDERAGENT_LOG_PATH="${THUNDERAGENT_LOG_PATH:-${REPO_ROOT}/results/processed/thunderagent.log}"
THUNDERAGENT_REPORT_PATH="${THUNDERAGENT_REPORT_PATH:-${REPO_ROOT}/results/processed/thunderagent_report.json}"
THUNDERAGENT_INSTALL_REPORT="${THUNDERAGENT_INSTALL_REPORT:-${REPO_ROOT}/results/processed/thunderagent_install_report.json}"
THUNDERAGENT_PROGRAM_ID="${THUNDERAGENT_PROGRAM_ID:-thunderagent-smoke-$(date +%s)}"
THUNDERAGENT_HEALTH_TIMEOUT_S="${THUNDERAGENT_HEALTH_TIMEOUT_S:-180}"
THUNDERAGENT_POLL_INTERVAL_S="${THUNDERAGENT_POLL_INTERVAL_S:-2.0}"

log() {
  printf '[ENV-3c] %s\n' "$*"
}

require_pinned_ref() {
  if [[ "${THUNDERAGENT_REF}" =~ ^[0-9a-f]{40}$ ]]; then
    return 0
  fi
  if git -C "${THUNDERAGENT_REPO_DIR}" rev-parse --verify "refs/tags/${THUNDERAGENT_REF}" >/dev/null 2>&1; then
    return 0
  fi
  printf 'THUNDERAGENT_REF must be a full commit SHA or an existing tag, not %s\n' "${THUNDERAGENT_REF}" >&2
  exit 1
}

ensure_repo() {
  mkdir -p "$(dirname "${THUNDERAGENT_REPO_DIR}")"
  if [[ ! -d "${THUNDERAGENT_REPO_DIR}/.git" ]]; then
    log "Cloning ThunderAgent repo from ${THUNDERAGENT_REPO_URL}"
    git clone "${THUNDERAGENT_REPO_URL}" "${THUNDERAGENT_REPO_DIR}"
  fi
  git -C "${THUNDERAGENT_REPO_DIR}" fetch origin
  git -C "${THUNDERAGENT_REPO_DIR}" checkout "${THUNDERAGENT_REF}"
}

ensure_venv() {
  if [[ ! -x "${THUNDERAGENT_PYTHON}" ]]; then
    log "Creating ThunderAgent virtualenv at ${THUNDERAGENT_VENV_DIR}"
    uv venv "${THUNDERAGENT_VENV_DIR}" --python 3.11.14
  fi
}

install_thunderagent() {
  log "Installing ThunderAgent and runtime dependencies"
  uv pip install --python "${THUNDERAGENT_PYTHON}" -e "${THUNDERAGENT_REPO_DIR}"
  uv pip install --python "${THUNDERAGENT_PYTHON}" "${THUNDERAGENT_EXTRA_SPEC}"
}

write_install_report() {
  log "Writing ThunderAgent install report to ${THUNDERAGENT_INSTALL_REPORT}"
  mkdir -p "$(dirname "${THUNDERAGENT_INSTALL_REPORT}")"
  THUNDERAGENT_REPO_DIR="${THUNDERAGENT_REPO_DIR}" \
  THUNDERAGENT_REPO_URL="${THUNDERAGENT_REPO_URL}" \
  THUNDERAGENT_REF="${THUNDERAGENT_REF}" \
  THUNDERAGENT_EXTRA_SPEC="${THUNDERAGENT_EXTRA_SPEC}" \
  THUNDERAGENT_INSTALL_REPORT="${THUNDERAGENT_INSTALL_REPORT}" \
  "${THUNDERAGENT_PYTHON}" - <<'PY'
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


repo_dir = Path(os.environ["THUNDERAGENT_REPO_DIR"])
payload = {
    "repo_url": os.environ["THUNDERAGENT_REPO_URL"],
    "requested_ref": os.environ["THUNDERAGENT_REF"],
    "requested_specs": {"extra": os.environ["THUNDERAGENT_EXTRA_SPEC"]},
    "resolved_commit": subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "installed_versions": {
        "uvicorn": safe_version("uvicorn"),
        "httpx": safe_version("httpx"),
        "fastapi": safe_version("fastapi"),
    },
}
Path(os.environ["THUNDERAGENT_INSTALL_REPORT"]).write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

start_proxy() {
  mkdir -p "$(dirname "${THUNDERAGENT_LOG_PATH}")"
  log "Starting ThunderAgent proxy; logs -> ${THUNDERAGENT_LOG_PATH}"
  PYTHONPATH="${REPO_ROOT}/src" "${THUNDERAGENT_PYTHON}" -m serving.thunderagent_launcher \
    --backends "${THUNDERAGENT_BACKENDS}" \
    --host "${THUNDERAGENT_HOST}" \
    --port "${THUNDERAGENT_PORT}" \
    --profile \
    --metrics \
    --profile-dir "${THUNDERAGENT_PROFILE_DIR}" \
    >>"${THUNDERAGENT_LOG_PATH}" 2>&1 &
  PROXY_PID=$!
  log "ThunderAgent proxy pid=${PROXY_PID}"
}

cleanup() {
  if [[ -n "${PROXY_PID:-}" ]] && kill -0 "${PROXY_PID}" >/dev/null 2>&1; then
    log "Stopping ThunderAgent proxy pid=${PROXY_PID}"
    kill "${PROXY_PID}" >/dev/null 2>&1 || true
    wait "${PROXY_PID}" >/dev/null 2>&1 || true
  fi
}

verify_proxy() {
  log "Running ThunderAgent proxy checks"
  PYTHONPATH="${REPO_ROOT}/src" "${THUNDERAGENT_PYTHON}" -m serving.thunderagent_check \
    --api-base "http://127.0.0.1:${THUNDERAGENT_PORT}/v1" \
    --base-url "http://127.0.0.1:${THUNDERAGENT_PORT}" \
    --program-id "${THUNDERAGENT_PROGRAM_ID}" \
    --timeout-s "${THUNDERAGENT_HEALTH_TIMEOUT_S}" \
    --poll-interval-s "${THUNDERAGENT_POLL_INTERVAL_S}" \
    --prompt "Reply with the word THUNDER." \
    --followup-prompt "Continue the same conversation and reply with THUNDER-AGAIN." \
    --output "${THUNDERAGENT_REPORT_PATH}" \
    --fail-on-mismatch
}

main() {
  ensure_repo
  require_pinned_ref
  ensure_venv
  install_thunderagent
  write_install_report
  trap cleanup EXIT INT TERM
  start_proxy
  verify_proxy
  log "ENV-3c completed successfully; proxy is running. Press Ctrl-C to stop."
  wait "${PROXY_PID}"
}

main "$@"
