#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-.venv-server}"
SERVER_PYTHON="${REPO_ROOT}/${VENV_DIR}/bin/python"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-huggingface}"

MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.1-8B-Instruct}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-LLM-Research/Meta-Llama-3.1-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:-${HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct}"
VERIFY_REPORT="${VERIFY_REPORT:-${REPO_ROOT}/results/processed/model_report.json}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
VERIFY_LOAD_MODE="${VERIFY_LOAD_MODE:-full}"

TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.51,<5.0}"
HF_HUB_SPEC="${HF_HUB_SPEC:-huggingface_hub>=0.30,<1.0}"
MODELSCOPE_SPEC="${MODELSCOPE_SPEC:-modelscope>=1.23,<2.0}"

EXPECTED_HIDDEN_SIZE="${EXPECTED_HIDDEN_SIZE:-4096}"
EXPECTED_NUM_LAYERS="${EXPECTED_NUM_LAYERS:-32}"

log() {
  printf '[ENV-2] %s\n' "$*"
}

require_server_python() {
  if [[ ! -x "${SERVER_PYTHON}" ]]; then
    printf 'Missing repo-local server Python: %s\n' "${SERVER_PYTHON}" >&2
    printf 'Run ENV-1 first on the target server.\n' >&2
    exit 1
  fi
}

install_download_dependencies() {
  log "Installing ENV-2 Python dependencies into ${SERVER_PYTHON}"
  uv pip install \
    --python "${SERVER_PYTHON}" \
    "${TRANSFORMERS_SPEC}" \
    "${HF_HUB_SPEC}"

  if [[ "${DOWNLOAD_BACKEND}" == "modelscope" ]]; then
    uv pip install --python "${SERVER_PYTHON}" "${MODELSCOPE_SPEC}"
  fi
}

download_from_huggingface() {
  log "Downloading ${MODEL_REPO} into ${MODEL_DIR} via HuggingFace Hub"
  MODEL_REPO="${MODEL_REPO}" MODEL_DIR="${MODEL_DIR}" "${SERVER_PYTHON}" - <<'PY'
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

download_from_modelscope() {
  log "Downloading ${MODELSCOPE_MODEL} into ${MODEL_DIR} via ModelScope"
  MODELSCOPE_MODEL="${MODELSCOPE_MODEL}" MODEL_DIR="${MODEL_DIR}" "${SERVER_PYTHON}" - <<'PY'
from __future__ import annotations

import os
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    model_id=os.environ["MODELSCOPE_MODEL"],
    local_dir=os.environ["MODEL_DIR"],
)
PY
}

write_model_path_env() {
  log "Recording MODEL_PATH in ${ENV_FILE}"
  mkdir -p "$(dirname "${ENV_FILE}")"
  touch "${ENV_FILE}"
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

verify_model_artifact() {
  log "Verifying downloaded model at ${MODEL_DIR}"
  "${SERVER_PYTHON}" "${REPO_ROOT}/scripts/report_model_artifact.py" \
    --output "${VERIFY_REPORT}" \
    --model-path "${MODEL_DIR}" \
    --backend "${DOWNLOAD_BACKEND}" \
    --model-repo "${MODEL_REPO}" \
    --modelscope-model "${MODELSCOPE_MODEL}" \
    --verify-load-mode "${VERIFY_LOAD_MODE}" \
    --expected-hidden-size "${EXPECTED_HIDDEN_SIZE}" \
    --expected-num-layers "${EXPECTED_NUM_LAYERS}" \
    --transformers-spec "${TRANSFORMERS_SPEC}" \
    --hf-hub-spec "${HF_HUB_SPEC}" \
    --modelscope-spec "${MODELSCOPE_SPEC}" \
    --fail-on-mismatch
}

main() {
  require_server_python
  install_download_dependencies
  mkdir -p "${MODEL_DIR}"

  case "${DOWNLOAD_BACKEND}" in
    huggingface)
      download_from_huggingface
      ;;
    modelscope)
      download_from_modelscope
      ;;
    *)
      printf 'Unsupported DOWNLOAD_BACKEND: %s\n' "${DOWNLOAD_BACKEND}" >&2
      exit 1
      ;;
  esac

  verify_model_artifact
  write_model_path_env
  log "ENV-2 completed successfully"
}

main "$@"
