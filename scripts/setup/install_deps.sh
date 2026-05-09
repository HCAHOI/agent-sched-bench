#!/usr/bin/env bash
# Install repo (editable) into the active conda env. Caller must activate ML.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ML" ]]; then
    echo "[install_deps] ERROR: activate conda env ML first" >&2
    exit 1
fi

if python -c "import trace_collect" >/dev/null 2>&1; then
    echo "[install_deps] SKIP: trace_collect already importable"
    exit 0
fi

echo "[install_deps] Installing repo (editable) into ${CONDA_PREFIX}"
pip install -e ".[dev]"
echo "[install_deps] done"
