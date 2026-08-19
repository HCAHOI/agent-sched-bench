#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model=NousResearch/Meta-Llama-3.1-8B-Instruct
manifest="$repo/analysis/development/pennylane-native-priority-v1/manifest.yaml"
run_root=/home/Ubuntu/pennylane-thunderagent-baseline-v1-20260819
vllm="$repo/.venv/bin/vllm"
python="$repo/.venv/bin/python"
cells=(fcfs-r1 thunderagent-r1 thunderagent-r2 fcfs-r2)

fail() { printf '%s\n' "$*" >&2; exit 1; }

preflight() {
  [[ $(git -C "$repo" status --porcelain) == "" ]] || fail "worktree must be clean"
  [[ -x "$vllm" && -x "$python" ]] || fail "run benchmark_server setup first"
  [[ -f "$manifest" ]] || fail "missing PennyLane manifest"
  [[ ! -e "$run_root" ]] || fail "run root already exists: $run_root"
  command -v docker >/dev/null || fail "docker is required"
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required"
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && fail "port 8000 is busy"
  curl -fsS http://127.0.0.1:9000/health >/dev/null 2>&1 && fail "port 9000 is busy"
  "$repo/scripts/baselines/thunderagent_official.sh" verify >/dev/null
  [[ -x "${THUNDERAGENT_VENV:-$HOME/.cache/agent-sched-bench/ThunderAgent-7ddc8610270e56d3b109eed8796b3a4360fc67c9/.venv}/bin/python" ]] || fail "ThunderAgent is not installed"
  local gpu
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
  [[ $(wc -l <<<"$gpu") -eq 1 && "$gpu" == *A100* ]] || fail "one A100 is required"
  (( ${gpu##*,} >= 80000 )) || fail "A100 80GB is required"
}

stop_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  kill -TERM -- -"$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- -"$pid" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -KILL -- -"$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

wait_http() {
  local url=$1 pid=$2 log=$3
  for _ in $(seq 1 1200); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { tail -80 "$log" >&2; return 1; }
    sleep 0.5
  done
  return 1
}

run_cell() (
  set -euo pipefail
  local name=$1 method=${1%-r?} cell="$run_root/$1"
  local vpid= proxy_pid= monitor_pid= rc=0
  mkdir "$cell"
  cleanup() {
    local status=$?
    set +e
    (( rc != 0 )) || rc=$status
    [[ -z "$proxy_pid" ]] || stop_group "$proxy_pid"
    stop_group "$vpid"
    [[ -z "$monitor_pid" ]] || { kill "$monitor_pid" 2>/dev/null; wait "$monitor_pid" 2>/dev/null; }
    date -u +%FT%TZ >"$cell/end-utc.txt"
    printf '%s\n' "$rc" >"$cell/cell-exit-code"
  }
  trap cleanup EXIT
  trap 'rc=130; exit 130' INT
  trap 'rc=143; exit 143' TERM
  date -u +%FT%TZ >"$cell/start-utc.txt"

  (
    printf 'timestamp_s,power_w,memory_mib,utilization_pct\n'
    while true; do
      printf '%s,' "$(date -u +%s.%N)"
      nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits
      sleep 1
    done
  ) >"$cell/gpu.csv" 2>"$cell/gpu.err" &
  monitor_pid=$!

  local vllm_args=(
    "$vllm" serve "$model" --host 127.0.0.1 --port 8000
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90
    --max-model-len 131072 --max-num-seqs 8 --enable-prefix-caching
    --kv-cache-dtype auto --enforce-eager --scheduling-policy priority
  )
  printf '%q ' env VLLM_NO_USAGE_STATS=1 CUDA_VISIBLE_DEVICES=0 "${vllm_args[@]}" >"$cell/vllm.argv"
  printf '\n' >>"$cell/vllm.argv"
  VLLM_NO_USAGE_STATS=1 CUDA_VISIBLE_DEVICES=0 setsid "${vllm_args[@]}" >"$cell/vllm.log" 2>&1 &
  vpid=$!
  wait_http http://127.0.0.1:8000/v1/models "$vpid" "$cell/vllm.log"

  local api=http://127.0.0.1:8000/v1 shadow_mode=vllm
  if [[ "$method" == thunderagent ]]; then
    api=http://127.0.0.1:9000/v1
    shadow_mode=thunderagent
    THUNDERAGENT_BACKENDS=http://127.0.0.1:8000 \
      THUNDERAGENT_PORT=9000 THUNDERAGENT_PROFILE_DIR="$cell/profiles" \
      setsid "$repo/scripts/baselines/thunderagent_official.sh" serve \
      >"$cell/proxy.log" 2>&1 &
    proxy_pid=$!
    wait_http http://127.0.0.1:9000/health "$proxy_pid" "$cell/proxy.log"
  fi

  local simulate=(
    "$python" -m trace_collect.cli simulate --manifest "$manifest"
    --output-dir "$cell/output" --container docker --network-mode host
    --concurrency 4 --workers 1 --prep-concurrency 8 --stage-all-before-replay
    --container-cpus 2 --replay-speed 1 --shadow-llm-api-base "$api"
    --shadow-llm-model "$model" --shadow-llm-timeout-s 300
    --shadow-llm-seed 0 --shadow-llm-mode "$shadow_mode"
    --resource-monitoring off --pmu-monitoring off
    --memory-bandwidth-monitoring off
  )
  printf '%q ' env OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 PYTHONPATH="$repo/src:$repo" "${simulate[@]}" >"$cell/simulate.argv"
  printf '\n' >>"$cell/simulate.argv"
  set +e
  OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 PYTHONPATH="$repo/src:$repo" \
    "${simulate[@]}" >"$cell/simulate.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$cell/simulate-exit-code"
  [[ $rc -eq 0 ]]
)

run_all() {
  preflight
  mkdir "$run_root"
  RUN_ROOT="$run_root" MODEL="$model" MANIFEST="$manifest" \
    GIT_COMMIT="$(git -C "$repo" rev-parse HEAD)" "$python" - <<'PY'
import json, os
from pathlib import Path
Path(os.environ["RUN_ROOT"], "protocol.json").write_text(json.dumps({
  "schema": 1,
  "git_commit": os.environ["GIT_COMMIT"],
  "model": os.environ["MODEL"],
  "manifest": os.environ["MANIFEST"],
  "cells": ["fcfs-r1", "thunderagent-r1", "thunderagent-r2", "fcfs-r2"],
  "workload": {"tasks": 8, "concurrency": 4, "container_cpus": 2},
  "comparison": "official ThunderAgent TR vs stock vLLM default-priority FCFS",
  "primary_metrics": ["mean_task_jct", "makespan", "all_request_p99_ttft"],
  "relevance_gate": {
    "mean_jct_and_makespan_improve_in_each_repetition": True,
    "geomean_mean_jct_reduction_at_least": 0.05,
    "geomean_p99_ttft_ratio_at_most": 1.10
  }
}, indent=2) + "\n")
PY
  local failed=0 cell
  for cell in "${cells[@]}"; do
    run_cell "$cell" || { printf 'cell %s failed; continuing\n' "$cell" >&2; failed=1; }
  done
  return "$failed"
}

case ${1:-} in
  --preflight) preflight ;;
  --run) run_all ;;
  *) echo "usage: $0 --preflight|--run" >&2; exit 2 ;;
esac
