#!/usr/bin/env bash
# Install Python dependencies into .venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
    echo "[setup] SKIP install_deps: .venv already exists"
    exit 0
fi

echo "[setup] Installing Python dependencies..."
make sync
echo "[setup] install_deps done"
