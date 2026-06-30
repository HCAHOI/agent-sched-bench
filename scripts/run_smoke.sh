#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_ROOT}"

if [[ "$#" -gt 0 ]]; then
  "${PYTHON_BIN}" -m pytest "$@"
  exit 0
fi

"${PYTHON_BIN}" -m pytest \
  tests/test_llm_call_config.py \
  tests/test_openclaw_minimal_install_contract.py \
  tests/test_task_container_runtime.py \
  tests/test_simulator_validation.py
