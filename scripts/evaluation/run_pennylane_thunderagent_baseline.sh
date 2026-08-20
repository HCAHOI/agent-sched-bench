#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model=${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}
manifest=${MANIFEST:-$repo/analysis/development/pennylane-native-priority-v1/manifest.yaml}
run_root=${RUN_ROOT:-/home/Ubuntu/pennylane-thunderagent-baseline-v1-20260819}
vllm="$repo/.venv/bin/vllm"
python="$repo/.venv/bin/python"
read -r -a cells <<<"${CELLS:-fcfs-r1 thunderagent-r1 thunderagent-r2 fcfs-r2}"
concurrency=${CONCURRENCY:-4}
container_cpuset=${CONTAINER_CPUSET:-}
container_cpus=${CONTAINER_CPUS:-}
vllm_cpuset=${VLLM_CPUSET:-}
continuum_profile=${CONTINUUM_REPRODUCTION_PROFILE:-}
cachewise_checkout=${CACHEWISE_CHECKOUT:-$HOME/.cache/agent-sched-bench/cachewise-181c435a090d328d00bbbee4c8eeb27d32f3abd2}
cachewise_models=${CACHEWISE_MODELS_DIR:-$cachewise_checkout/tool_duration_prediction/models}
if [[ -z "$container_cpuset" && -z "$container_cpus" ]]; then
  container_cpus=2
fi

fail() { printf '%s\n' "$*" >&2; exit 1; }

has_method() {
  [[ " ${cells[*]} " == *" $1-r"* ]]
}

preflight() {
  [[ $(git -C "$repo" status --porcelain) == "" ]] || fail "worktree must be clean"
  [[ -x "$python" ]] || fail "run benchmark_server setup first"
  { ! has_method fcfs && ! has_method thunderagent; } || \
    [[ -x "$vllm" ]] || fail "stock vLLM is not installed"
  [[ -f "$manifest" ]] || fail "missing PennyLane manifest"
  [[ ! -e "$run_root" ]] || fail "run root already exists: $run_root"
  [[ -z "$container_cpuset" || -z "$container_cpus" ]] || fail "choose cpuset or CPU quota, not both"
  [[ -z "$vllm_cpuset" ]] || command -v taskset >/dev/null || fail "taskset is required"
  local cell
  for cell in "${cells[@]}"; do
    [[ "$cell" =~ ^(fcfs|thunderagent|agentix|continuum-public|continuum-reproduction|cachewise)-r[1-9][0-9]*$ ]] || fail "unsupported cell: $cell"
  done
  command -v docker >/dev/null || fail "docker is required"
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required"
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && fail "port 8000 is busy"
  timeout 1 bash -c '</dev/tcp/127.0.0.1/9000' >/dev/null 2>&1 && fail "port 9000 is busy"
  if has_method thunderagent; then
    "$repo/scripts/baselines/thunderagent_official.sh" verify >/dev/null
  fi
  if has_method agentix; then
    "$repo/scripts/baselines/agentix_reproduction.sh" verify >/dev/null
  fi
  if has_method continuum-public; then
    "$repo/scripts/baselines/continuum_public.sh" verify >/dev/null
  fi
  if has_method continuum-reproduction; then
    [[ -f "$continuum_profile" ]] || fail "missing Continuum reproduction profile"
    "$repo/scripts/baselines/continuum_reproduction.sh" verify >/dev/null
  fi
  if has_method cachewise; then
    [[ -f "$cachewise_models/all_models.pkl" ]] || fail "missing CacheWise models"
    "$repo/scripts/baselines/cachewise_reproduction.sh" verify >/dev/null
  fi
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
  local name=$1 method=${1%-r[0-9]*} cell="$run_root/$1"
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

  local common_args=(
    --host 127.0.0.1 --port 8000 --tensor-parallel-size 1
    --gpu-memory-utilization 0.90 --max-model-len 131072
    --max-num-seqs 8 --enable-prefix-caching --kv-cache-dtype auto
    --enforce-eager
  )
  local server=("$vllm" serve "$model" "${common_args[@]}" --scheduling-policy priority)
  local server_env=(VLLM_NO_USAGE_STATS=1 CUDA_VISIBLE_DEVICES=0)
  case "$method" in
    agentix)
      server=("$repo/scripts/baselines/agentix_reproduction.sh" serve-backend "$model" "${common_args[@]}")
      server_env+=(AGENTIX_SERVICE_LOG="$cell/agentix-service.jsonl")
      ;;
    continuum-public)
      server=("$repo/scripts/baselines/continuum_public.sh" serve "$model" "${common_args[@]}")
      server_env+=(RUN_OUTPUT_DIR="$cell/continuum")
      ;;
    continuum-reproduction)
      server=("$repo/scripts/baselines/continuum_reproduction.sh" serve "$model" --dtype bfloat16 --kv-cache-dtype bfloat16)
      server_env+=(CONTINUUM_REPRODUCTION_PROFILE="$continuum_profile" CONTINUUM_REPRODUCTION_MODE=prefill RUN_OUTPUT_DIR="$cell/continuum")
      ;;
    cachewise)
      server=("$repo/scripts/baselines/cachewise_reproduction.sh" serve "$model" "${common_args[@]}")
      ;;
  esac
  local vllm_launch=(setsid)
  [[ -z "$vllm_cpuset" ]] || vllm_launch+=(taskset -c "$vllm_cpuset")
  printf '%q ' env "${server_env[@]}" "${vllm_launch[@]}" "${server[@]}" >"$cell/vllm.argv"
  printf '\n' >>"$cell/vllm.argv"
  env "${server_env[@]}" "${vllm_launch[@]}" "${server[@]}" >"$cell/vllm.log" 2>&1 &
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
  elif [[ "$method" == agentix ]]; then
    api=http://127.0.0.1:9000/v1
    shadow_mode=agentix
    setsid "$repo/scripts/baselines/agentix_reproduction.sh" serve-proxy \
      --backend http://127.0.0.1:8000 \
      --service-log "$cell/agentix-service.jsonl" \
      --queue-upper-bounds 0.25,1,4,16 \
      --event-log "$cell/agentix-events.jsonl" >"$cell/proxy.log" 2>&1 &
    proxy_pid=$!
    wait_http http://127.0.0.1:9000/programs/state "$proxy_pid" "$cell/proxy.log"
  elif [[ "$method" == continuum-public || "$method" == continuum-reproduction ]]; then
    shadow_mode=continuum-public
  elif [[ "$method" == cachewise ]]; then
    shadow_mode=cachewise
  fi

  local simulate=(
    "$python" -m trace_collect.cli simulate --manifest "$manifest"
    --output-dir "$cell/output" --container docker --network-mode host
    --concurrency "$concurrency" --workers 1 --prep-concurrency 8 --stage-all-before-replay
    --replay-speed 1 --shadow-llm-api-base "$api"
    --shadow-llm-model "$model" --shadow-llm-timeout-s 300
    --shadow-llm-seed 0 --shadow-llm-mode "$shadow_mode"
    --resource-monitoring off --pmu-monitoring off
    --memory-bandwidth-monitoring off
  )
  if [[ -n "$container_cpuset" ]]; then
    simulate+=(--container-cpuset-cpus "$container_cpuset")
  else
    simulate+=(--container-cpus "$container_cpus")
  fi
  if [[ "$method" == cachewise ]]; then
    simulate+=(
      --shadow-llm-cachewise-predictor-checkout "$cachewise_checkout"
      --shadow-llm-cachewise-models-dir "$cachewise_models"
    )
  fi
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
    CELLS="${cells[*]}" CONCURRENCY="$concurrency" \
    CONTAINER_CPUSET="$container_cpuset" CONTAINER_CPUS="$container_cpus" \
    VLLM_CPUSET="$vllm_cpuset" \
    CONTINUUM_PROFILE="$continuum_profile" \
    GIT_COMMIT="$(git -C "$repo" rev-parse HEAD)" "$python" - <<'PY'
import json, os
from pathlib import Path
Path(os.environ["RUN_ROOT"], "protocol.json").write_text(json.dumps({
  "schema": 1,
  "git_commit": os.environ["GIT_COMMIT"],
  "model": os.environ["MODEL"],
  "manifest": os.environ["MANIFEST"],
  "cells": os.environ["CELLS"].split(),
  "workload": {
    "concurrency": int(os.environ["CONCURRENCY"]),
    "container_cpus": float(os.environ["CONTAINER_CPUS"]) if os.environ["CONTAINER_CPUS"] else None,
    "container_cpuset": os.environ["CONTAINER_CPUSET"] or None,
    "vllm_cpuset": os.environ["VLLM_CPUSET"] or None,
  },
  "comparison": "paper baselines on one fixed real-tool replay workload",
  "continuum_reproduction_profile": os.environ["CONTINUUM_PROFILE"] or None,
  "agentix_queue_upper_bounds_s": [0.25, 1, 4, 16],
  "cachewise_tool_mapping": {
    "exec": "Bash", "read_file": "Read", "edit_file": "Edit", "list_dir": "Glob"
  },
  "primary_metrics": ["mean_task_jct", "makespan", "all_request_p99_ttft"],
  "interpretation": "Physical baseline measurement; no GO/NO-GO gate."
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
