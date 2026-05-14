#!/usr/bin/env bash
# Launch a KV-eviction capstone trace on terminal-bench/jsonl-aggregator with
# Qwen3-Coder-30B + openclaw + HF recording backend (session-shared KV cache).
#
# Usage:
#   ./scripts/launch_kv_capstone.sh configs/kv_policies/h2o_b1024.yaml [label-suffix]
#
# Writes to traces/terminal-bench/<safe-model>/<timestamp>/ and tees full stdout
# to logs/<label>-<timestamp>.log. Exits with the trace_collect exit code.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <kv-config-yaml> [label-suffix]" >&2
  exit 2
fi

KV_CONFIG="$1"
LABEL_SUFFIX="${2:-$(basename "$KV_CONFIG" .yaml)}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/capstone-${LABEL_SUFFIX}-${TS}.log"

REPO="/home/featurize/work/agent-sched-bench"
cd "$REPO"

export HF_HOME=/home/featurize/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_RECORDING_MAX_GPU_MEMORY_GIB=90
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export OPENAI_API_KEY=dummy

PY=/home/featurize/work/envs/ML/bin/python

echo "=== launch_kv_capstone.sh ==="
echo "ts=$TS"
echo "kv_config=$KV_CONFIG"
echo "label=$LABEL_SUFFIX"
echo "log=$LOG"
echo "head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "================================"

exec "$PY" -m trace_collect.cli \
  --provider openai \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --benchmark terminal-bench \
  --scaffold openclaw \
  --container docker \
  --mcp-config none \
  --instance-ids jsonl-aggregator \
  --kv-policy h2o \
  --kv-config "$KV_CONFIG" \
  --record-internals \
  --max-iterations 100
