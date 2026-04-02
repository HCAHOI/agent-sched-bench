#!/usr/bin/env bash
# Download Qwen3-32B-AWQ from HuggingFace into models/.
# Requires HF_TOKEN to be set in the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VENV_DIR:-.venv}"
SERVER_PYTHON="${REPO_ROOT}/${VENV_DIR}/bin/python"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-huggingface}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-32B-AWQ}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/Qwen3-32B-AWQ}"
VERIFY_REPORT="${VERIFY_REPORT:-${REPO_ROOT}/results/processed/model_report.json}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
VERIFY_LOAD_MODE="${VERIFY_LOAD_MODE:-config-only}"

TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.51,<5.0}"
HF_HUB_SPEC="${HF_HUB_SPEC:-huggingface_hub>=0.30,<1.0}"
MODELSCOPE_SPEC="${MODELSCOPE_SPEC:-modelscope>=1.23,<2.0}"

EXPECTED_HIDDEN_SIZE="${EXPECTED_HIDDEN_SIZE:-5120}"
EXPECTED_NUM_LAYERS="${EXPECTED_NUM_LAYERS:-64}"

log() {
  printf '[setup/download_model] %s\n' "$*"
}

install_download_dependencies() {
  log "Installing download dependencies into ${SERVER_PYTHON}"
  uv pip install \
    --python "${SERVER_PYTHON}" \
    "${TRANSFORMERS_SPEC}" \
    "${HF_HUB_SPEC}"
}

download_from_huggingface() {
  log "Downloading ${MODEL_REPO} → ${MODEL_DIR}"
  HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  MODEL_REPO="${MODEL_REPO}" \
  MODEL_DIR="${MODEL_DIR}" \
  "${SERVER_PYTHON}" - <<'PY'
from __future__ import annotations
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    repo_type="model",
    local_dir=os.environ["MODEL_DIR"],
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
}

verify_model_artifact() {
  log "Verifying model artifact at ${MODEL_DIR}"
  "${SERVER_PYTHON}" "${REPO_ROOT}/scripts/report_model_artifact.py" \
    --output "${VERIFY_REPORT}" \
    --model-path "${MODEL_DIR}" \
    --backend "${DOWNLOAD_BACKEND}" \
    --model-repo "${MODEL_REPO}" \
    --modelscope-model "" \
    --verify-load-mode "${VERIFY_LOAD_MODE}" \
    --expected-hidden-size "${EXPECTED_HIDDEN_SIZE}" \
    --expected-num-layers "${EXPECTED_NUM_LAYERS}" \
    --transformers-spec "${TRANSFORMERS_SPEC}" \
    --hf-hub-spec "${HF_HUB_SPEC}" \
    --modelscope-spec "${MODELSCOPE_SPEC}" \
    --fail-on-mismatch
}

write_model_path_env() {
  log "Recording MODEL_PATH in ${ENV_FILE}"
  MODEL_DIR="${MODEL_DIR}" ENV_FILE="${ENV_FILE}" "${SERVER_PYTHON}" - <<'PY'
from __future__ import annotations
import os
from pathlib import Path

env_file = Path(os.environ["ENV_FILE"])
model_dir = os.environ["MODEL_DIR"]
lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
updated = False
for index, line in enumerate(lines):
    if line.startswith("MODEL_PATH="):
        lines[index] = f"MODEL_PATH={model_dir}"
        updated = True
        break
if not updated:
    lines.append(f"MODEL_PATH={model_dir}")
env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

main() {
  if [[ -f "${MODEL_DIR}/config.json" ]]; then
    log "SKIP: ${MODEL_DIR}/config.json already exists"
    exit 0
  fi

  if [[ ! -x "${SERVER_PYTHON}" ]]; then
    printf 'Missing Python: %s\n' "${SERVER_PYTHON}" >&2
    printf 'Run install_deps.sh first.\n' >&2
    exit 1
  fi

  install_download_dependencies
  mkdir -p "${MODEL_DIR}"
  download_from_huggingface
  verify_model_artifact
  write_model_path_env
  log "download_model done"
}

main "$@"
