#!/usr/bin/env bash
# Generate .env from .env.example for cloud-provider runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

if [[ ! -f ".env" ]]; then
    echo "[setup] Creating .env from .env.example"
    cp .env.example .env
else
    echo "[setup] .env already exists"
fi

echo "[setup] Edit .env to set the API key for your selected cloud provider."
