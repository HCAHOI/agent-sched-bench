#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${REPO_ROOT}/src/harness/runner.py" ]]; then
  echo "HARNESS-1 is not implemented yet; run-sweep remains unavailable." >&2
  exit 1
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m harness.runner "$@"
