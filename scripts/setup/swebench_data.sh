#!/usr/bin/env bash
# Download SWE-bench Verified and select tool-intensive tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

TASKS_FILE="data/swebench_verified/tasks.json"

if [[ -f "$TASKS_FILE" ]]; then
    count=$("${REPO_ROOT}/.venv/bin/python" -c "import json; print(len(json.load(open('${TASKS_FILE}'))))")
    echo "[setup] SKIP swebench_data: ${TASKS_FILE} already exists (${count} tasks)"
    exit 0
fi

echo "[setup] Downloading SWE-bench Verified dataset..."
"${REPO_ROOT}/.venv/bin/python" -m agents.swebench_data
echo "[setup] swebench_data done"
