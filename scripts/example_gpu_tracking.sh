#!/usr/bin/env bash
# Demo: GPU memory tracking for local-simulate vLLM runs.
#
# REQUIREMENTS:
#   - GPU + nvidia-smi
#   - vllm installed (pip install -e .[profile])
#   - A source trace JSONL to replay
#
# This script is intended for the x86 research server (184.144.255.168 RTX 3060 Ti).
# It will not work on Mac (vLLM has no Mac GPU support).
#
# Usage:
#   scripts/example_gpu_tracking.sh <source-trace.jsonl> <model> [<output-dir>]
#
set -euo pipefail

SOURCE_TRACE="${1:?source trace path required}"
MODEL="${2:?model required (e.g. Qwen/Qwen3-1.7B)}"
OUTPUT_DIR="${3:-traces/gpu_tracking_demo}"

VLLM_LOG="/tmp/vllm_startup_$$.log"
VLLM_PORT="${VLLM_PORT:-8000}"
METRICS_URL="http://localhost:${VLLM_PORT}/metrics"
API_BASE="http://localhost:${VLLM_PORT}/v1"

echo "[1/5] Launching vLLM with --capture-startup-log → ${VLLM_LOG}"
PYTHONPATH=src conda run -n ML python -m serving.engine_launcher \
  --model-path "${MODEL}" \
  --port "${VLLM_PORT}" \
  --capture-startup-log "${VLLM_LOG}" \
  --enable-chunked-prefill &
VLLM_LAUNCHER_PID=$!

cleanup() {
  echo "[cleanup] Stopping vLLM launcher (pid=${VLLM_LAUNCHER_PID})"
  kill "${VLLM_LAUNCHER_PID}" 2>/dev/null || true
  wait "${VLLM_LAUNCHER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[2/5] Waiting for vLLM /metrics readiness (max 300s)"
for _ in $(seq 1 300); do
  if curl -fsS "${METRICS_URL}" > /dev/null 2>&1; then
    echo "  ready"
    break
  fi
  sleep 1
done

echo "[3/5] Resolving vLLM child PID (the one holding GPU memory)"
# The launcher spawns vllm as a child; nvidia-smi sees the child PID, not the wrapper.
VLLM_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | head -n1)
if [[ -z "${VLLM_PID}" ]]; then
  echo "ERROR: no GPU compute apps found via nvidia-smi" >&2
  exit 1
fi
echo "  vLLM PID = ${VLLM_PID}"

echo "[4/5] Running simulate with --gpu-tracking on"
PYTHONPATH=src conda run -n ML python -m trace_collect.cli simulate \
  --source-trace "${SOURCE_TRACE}" \
  --mode local_model \
  --provider openai --api-base "${API_BASE}" --api-key dummy \
  --model "${MODEL}" \
  --container docker \
  --metrics-url "${METRICS_URL}" \
  --gpu-tracking on \
  --vllm-pid "${VLLM_PID}" \
  --vllm-startup-log "${VLLM_LOG}" \
  --gpu-sample-hz 10.0 \
  --output-dir "${OUTPUT_DIR}"

echo "[5/5] Done. Results in ${OUTPUT_DIR}"
echo "  - trace.jsonl actions have data.sim_metrics.vllm_scheduler_snapshot.gpu_memory_breakdown"
echo "  - <attempt-dir>/gpu_resources.json has the time series"
