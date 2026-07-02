#!/usr/bin/env bash
# End-to-end checkpoint validation experiment:
#   collect (5 SWE-rebench tasks) → simulate → mismatch stats
#
# Usage:
#   1. Push code to remote:  git push origin dev/cpu-only
#   2. SSH to remote:        ssh -p 39602 root@connect.singapore-a.gpuhub.com
#   3. Clone & setup:        git clone ... && cd agent-sched-bench
#   4. Run:                  bash scripts/e2e_checkpoint_validation.sh
#
# Prerequisites on remote:
#   - Docker installed and running
#   - OPENROUTER_API_KEY set in environment
#   - Python 3.12+ with uv (or use the .venv from setup script)
#   - ~50GB free disk space (Docker images + checkpoints)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ──────────────────────────────────────────────────────────
PROVIDER="openrouter"
MODEL="qwen/qwen3.7-max"
BENCHMARK="swe-rebench"
SAMPLE=5
MAX_ITERATIONS=50
SCAFFOLD="openclaw"
CONTAINER="docker"
CONCURRENCY=1

# ── Check prerequisites ─────────────────────────────────────────────
echo "=== Checking prerequisites ==="
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not running"; exit 1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: OPENROUTER_API_KEY not set"; exit 1; }
[ -d ".venv" ] || { echo "ERROR: .venv not found; run: bash scripts/setup/benchmark_server.sh"; exit 1; }
echo "✓ Prerequisites OK"

# ── Step 1: Collect ─────────────────────────────────────────────────
echo ""
echo "=== Step 1: Collect traces ==="
echo "Model:     ${MODEL}"
echo "Benchmark: ${BENCHMARK}"
echo "Sample:    ${SAMPLE}"
echo "Max iter:  ${MAX_ITERATIONS}"
echo ""

COLLECT_START=$(date +%s)

.venv/bin/python -m trace_collect.cli \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --benchmark "${BENCHMARK}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --sample "${SAMPLE}" \
  --scaffold "${SCAFFOLD}" \
  --mcp-config none \
  --container "${CONTAINER}" \
  --concurrency "${CONCURRENCY}"

COLLECT_END=$(date +%s)
COLLECT_ELAPSED=$((COLLECT_END - COLLECT_START))
echo ""
echo "Collect completed in ${COLLECT_ELAPSED}s"

# Find the run directory (most recent under traces/swe-rebench/)
RUN_DIR=$(find traces/swe-rebench -maxdepth 3 -name "results.jsonl" -type f \
  | sort -r | head -1 | xargs dirname)
echo "Run directory: ${RUN_DIR}"

# ── Step 2: Generate simulate manifest ──────────────────────────────
echo ""
echo "=== Step 2: Generate simulate manifest ==="

.venv/bin/python scripts/gen_simulate_manifest.py "${RUN_DIR}"

MANIFEST="${RUN_DIR}/simulate_manifest.yaml"
TASKS="${RUN_DIR}/tasks.json"
echo "Manifest: ${MANIFEST}"
echo "Tasks:    ${TASKS}"

# Quick validation
TRACE_COUNT=$(grep -c "trace.jsonl" "${MANIFEST}" || echo 0)
TASK_COUNT=$(.venv/bin/python -c "import json; print(len(json.load(open('${TASKS}'))))")
echo "Traces in manifest: ${TRACE_COUNT}"
echo "Tasks in task_source: ${TASK_COUNT}"

# ── Step 3: Simulate ────────────────────────────────────────────────
echo ""
echo "=== Step 3: Simulate replay ==="

SIMULATE_OUTPUT="${RUN_DIR}/simulate_output"
mkdir -p "${SIMULATE_OUTPUT}"

SIMULATE_START=$(date +%s)

.venv/bin/python -m trace_collect.cli simulate \
  --manifest "${MANIFEST}" \
  --task-source "${TASKS}" \
  --container "${CONTAINER}" \
  --concurrency 1 \
  --output-dir "${SIMULATE_OUTPUT}" \
  --network-mode host

SIMULATE_END=$(date +%s)
SIMULATE_ELAPSED=$((SIMULATE_END - SIMULATE_START))
echo ""
echo "Simulate completed in ${SIMULATE_ELAPSED}s"

# Find the simulate trace
SIM_TRACE=$(find "${SIMULATE_OUTPUT}" -maxdepth 1 -name "simulate_*.jsonl" -type f \
  | sort -r | head -1)
if [ -z "${SIM_TRACE}" ]; then
    echo "ERROR: No simulate trace found in ${SIMULATE_OUTPUT}"
    exit 1
fi
echo "Simulate trace: ${SIM_TRACE}"

# ── Step 4: Extract stats ───────────────────────────────────────────
echo ""
echo "=== Step 4: Mismatch & forced-sync statistics ==="

.venv/bin/python scripts/simulate_mismatch_stats.py "${SIM_TRACE}"

# ── Summary ──────────────────────────────────────────────────────────
TOTAL_ELAPSED=$((SIMULATE_END - COLLECT_START))
echo ""
echo "=== Done ==="
echo "Total wall time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))min)"
echo "Collect traces:  ${RUN_DIR}/"
echo "Simulate trace:  ${SIM_TRACE}"
echo "Mismatch stats:  (printed above)"
