#!/usr/bin/env bash
# One-shot setup for Code Agent environment.
#
# Usage:
#   bash scripts/full_setup.sh <HF_TOKEN>
#
# Each step is idempotent: re-running skips already-completed steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SETUP_DIR="${SCRIPT_DIR}/setup"

HF_TOKEN="${1:-${HF_TOKEN:-}}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "Usage: $0 <HF_TOKEN>" >&2
    exit 1
fi

export HF_TOKEN
cd "$REPO_ROOT"

step() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

step "1/6  Install Python dependencies"
bash "${SETUP_DIR}/install_deps.sh"

step "2/6  Configure .env"
bash "${SETUP_DIR}/configure_env.sh"

step "3/6  Download Qwen3-32B-AWQ model"
bash "${SETUP_DIR}/download_model.sh"

step "4/6  Download SWE-bench Verified dataset"
bash "${SETUP_DIR}/swebench_data.sh"

step "5/6  Clone task repositories"
bash "${SETUP_DIR}/clone_repos.sh"

step "6/6  Build Podman container images"
bash "${SETUP_DIR}/build_images.sh"

# ─── Optional: vast.ai podman bootstrap (Phase 3 of trace-sim-vastai-pipeline) ──
# Guarded so existing local-mode setups (which don't need rootless podman)
# are unaffected. Triggered by setting AGENT_SCHED_BENCH_VASTAI=1 in the
# environment, OR by running on a host where /etc/vastai-host exists.
if [[ "${AGENT_SCHED_BENCH_VASTAI:-}" = "1" ]] || [[ -e /etc/vastai-host ]]; then
    step "vast.ai bootstrap: install podman + start socket"
    bash "${SETUP_DIR}/install_podman_vastai.sh"
    bash "${SETUP_DIR}/start_podman_socket.sh"
    echo ""
    echo "  reminder: export DOCKER_HOST=unix:///run/user/\$(id -u)/podman/podman.sock"
    echo "  (or: eval \"\$(bash scripts/setup/start_podman_socket.sh --print-export)\")"
fi

echo ""
echo "========================================"
echo "  Setup complete — summary"
echo "========================================"
echo "  .venv:          $( [[ -x .venv/bin/python ]] && echo OK || echo MISSING )"
echo "  .env:           $( [[ -f .env ]] && echo OK || echo MISSING )"
echo "  model:          $( [[ -f models/Qwen3-32B-AWQ/config.json ]] && echo OK || echo MISSING )"
echo "  tasks.json:     $( [[ -f data/swebench_verified/tasks.json ]] && \
    .venv/bin/python -c "import json; t=json.load(open('data/swebench_verified/tasks.json')); print(f'OK ({len(t)} tasks)')" \
    || echo MISSING )"
echo "  swebench_repos: $( [[ -d data/swebench_repos ]] && echo "OK ($(ls data/swebench_repos | wc -l) repos)" || echo MISSING )"
echo "  container:      $( podman image exists swebench-base:latest 2>/dev/null && echo OK || echo MISSING )"
echo ""
