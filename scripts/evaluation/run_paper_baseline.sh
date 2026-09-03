#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model=${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}
manifest=${MANIFEST:?MANIFEST must name a replay manifest}
run_root=${RUN_ROOT:?RUN_ROOT must be a new output directory}
vllm="$repo/.venv/bin/vllm"
python="$repo/.venv/bin/python"
read -r -a cells <<<"${CELLS:-fcfs-r1 thunderagent-r1 thunderagent-r2 fcfs-r2}"
concurrency=${CONCURRENCY:-4}
container_cpuset=${CONTAINER_CPUSET:-}
container_cpus=${CONTAINER_CPUS:-}
vllm_cpuset=${VLLM_CPUSET:-}
continuum_profile=${CONTINUUM_REPRODUCTION_PROFILE:-}
trace_tool_replay=${TRACE_TOOL_REPLAY:-0}
trace_tool_replay_speed=${TRACE_TOOL_REPLAY_SPEED:-1}
stage_all_before_replay=${STAGE_ALL_BEFORE_REPLAY:-1}
replacement_delay_mean_s=${REPLACEMENT_DELAY_MEAN_S:-}
replacement_seed=${REPLACEMENT_SEED:-42}
cleanup_images=${CLEANUP_IMAGES:-0}
resource_monitoring=${RESOURCE_MONITORING:-off}
serving_metrics=${SERVING_METRICS:-on}
shadow_llm_timeout_s=${SHADOW_LLM_TIMEOUT_S:-300}
expected_gpu_name=${EXPECTED_GPU_NAME:-A100}
min_gpu_memory_mib=${MIN_GPU_MEMORY_MIB:-80000}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
cachewise_oracle_prefill_ms_per_token=${CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN:-}
cachewise_oracle_decode_ms_per_token=${CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN:-}
cachewise_checkout=${CACHEWISE_CHECKOUT:-$HOME/.cache/agent-sched-bench/cachewise-181c435a090d328d00bbbee4c8eeb27d32f3abd2}
cachewise_models=${CACHEWISE_MODELS_DIR:-$cachewise_checkout/tool_duration_prediction/models}
cupti_overlay=${CUPTI_DRAM_OVERLAY:-${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/cupti-dram-13.3.1}
cupti_lib=$cupti_overlay/nvidia/cu13/lib
perfmon_cap=
saga_profile=${SAGA_PROFILE:-}
if [[ -z "$container_cpuset" && -z "$container_cpus" ]]; then
  container_cpus=2
fi

fail() { printf '%s\n' "$*" >&2; exit 1; }

has_method() {
  [[ " ${cells[*]} " == *" $1-r"* ]]
}

has_cachewise() {
  has_method cachewise || has_method cachewise-oracle-length
}

preflight() {
  [[ $(git -C "$repo" status --porcelain) == "" ]] || fail "worktree must be clean"
  [[ -x "$python" ]] || fail "run benchmark_server setup first"
  { ! has_method fcfs && ! has_method thunderagent && ! has_method native-priority; } || \
    [[ -x "$vllm" ]] || fail "stock vLLM is not installed"
  [[ -f "$manifest" ]] || fail "missing replay manifest: $manifest"
  [[ ! -e "$run_root" ]] || fail "run root already exists: $run_root"
  [[ "$trace_tool_replay" == 0 || "$trace_tool_replay" == 1 ]] || fail "TRACE_TOOL_REPLAY must be 0 or 1"
  "$python" - "$trace_tool_replay" "$trace_tool_replay_speed" <<'PY'
import math
import sys

enabled = sys.argv[1] == "1"
try:
    speed = float(sys.argv[2])
except ValueError as exc:
    raise SystemExit("TRACE_TOOL_REPLAY_SPEED must be positive") from exc
if not math.isfinite(speed) or speed <= 0:
    raise SystemExit("TRACE_TOOL_REPLAY_SPEED must be positive")
if not enabled and speed != 1:
    raise SystemExit("TRACE_TOOL_REPLAY_SPEED requires TRACE_TOOL_REPLAY=1")
PY
  [[ "$stage_all_before_replay" == 0 || "$stage_all_before_replay" == 1 ]] || fail "STAGE_ALL_BEFORE_REPLAY must be 0 or 1"
  if [[ -n "$replacement_delay_mean_s" ]]; then
    [[ "$replacement_seed" =~ ^-?[0-9]+$ ]] || \
      fail "REPLACEMENT_SEED must be an integer"
    "$python" - "$replacement_delay_mean_s" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("REPLACEMENT_DELAY_MEAN_S must be positive") from exc
if not math.isfinite(value) or value <= 0:
    raise SystemExit("REPLACEMENT_DELAY_MEAN_S must be positive")
PY
    [[ "$stage_all_before_replay" == 0 ]] || \
      fail "replacement load requires STAGE_ALL_BEFORE_REPLAY=0"
    [[ "$cleanup_images" == 0 ]] || \
      fail "replacement load requires CLEANUP_IMAGES=0"
  fi
  [[ "$cleanup_images" == 0 || "$cleanup_images" == 1 ]] || fail "CLEANUP_IMAGES must be 0 or 1"
  [[ "$resource_monitoring" =~ ^(auto|on|off)$ ]] || fail "RESOURCE_MONITORING must be auto, on, or off"
  [[ "$serving_metrics" =~ ^(on|off)$ ]] || fail "SERVING_METRICS must be on or off"
  [[ -z "$vllm_cpuset" ]] || command -v taskset >/dev/null || fail "taskset is required"
  local cell
  for cell in "${cells[@]}"; do
    [[ "$cell" =~ ^(fcfs|thunderagent|agentix|continuum-public|continuum-reproduction|continuum-reproduction-oracle-length|native-priority|native-priority-aging|cachewise-disabled|cachewise|cachewise-oracle-length|saga)-r[1-9][0-9]*$ ]] || fail "unsupported cell: $cell"
  done
  command -v docker >/dev/null || fail "docker is required"
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required"
  if [[ "$serving_metrics" == on ]]; then
    "$python" -c 'import msgspec, zmq' || fail "serving metrics require msgspec and pyzmq"
    [[ -f "$cupti_lib/libcupti.so.13" && -f "$cupti_lib/libnvperf_host.so" \
       && -d "$cupti_overlay/cuda-bindings/cuda/bindings" ]] || \
      fail "CUPTI DRAM support is missing; rerun benchmark_server.sh --gpu"
    PYTHONPATH="$cupti_overlay" LD_LIBRARY_PATH="$cupti_lib" "$python" -c \
      'from cupti.pm_sampling import Collector' || \
      fail "CUPTI Python PM sampling cannot be imported"
    command -v setpriv >/dev/null || fail "serving metrics require setpriv"
    perfmon_cap=$(setpriv --list-caps | awk '$0 == "perfmon" || $0 == "cap_38" { print; exit }')
    [[ -n "$perfmon_cap" ]] || fail "setpriv cannot name CAP_PERFMON"
    sudo -n true || fail "serving metrics require passwordless sudo for CAP_PERFMON"
    if command -v systemctl >/dev/null && systemctl is-active --quiet nvidia-dcgm; then
      fail "nvidia-dcgm must be stopped before CUPTI DRAM sampling"
    fi
    timeout 1 bash -c '</dev/tcp/127.0.0.1/5557' >/dev/null 2>&1 && \
      fail "port 5557 is busy"
    timeout 1 bash -c '</dev/tcp/127.0.0.1/5558' >/dev/null 2>&1 && \
      fail "port 5558 is busy"
  fi
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
  if has_method continuum-reproduction || has_method continuum-reproduction-oracle-length; then
    [[ -f "$continuum_profile" ]] || fail "missing Continuum reproduction profile"
    "$repo/scripts/baselines/continuum_reproduction.sh" verify >/dev/null
  fi
  if has_method native-priority-aging; then
    "$repo/scripts/baselines/native_priority_aging.sh" verify >/dev/null
  fi
  if has_cachewise; then
    [[ -f "$cachewise_models/all_models.pkl" ]] || fail "missing CacheWise models"
    "$python" -c 'import sklearn' || \
      fail "CacheWise requires: uv sync --extra serving-spike"
  fi
  if has_method cachewise-oracle-length; then
    "$python" - "$cachewise_oracle_prefill_ms_per_token" \
      "$cachewise_oracle_decode_ms_per_token" <<'PY'
import math
import sys

try:
    values = [float(value) for value in sys.argv[1:]]
except ValueError as exc:
    raise SystemExit("CacheWise Oracle service coefficients must be positive") from exc
if len(values) != 2 or not all(math.isfinite(value) and value > 0 for value in values):
    raise SystemExit("CacheWise Oracle service coefficients must be positive")
PY
  fi
  if has_cachewise || has_method cachewise-disabled; then
    "$repo/scripts/baselines/cachewise_reproduction.sh" verify-installed >/dev/null
  fi
  if has_method saga; then
    [[ -f "$saga_profile" ]] || fail "missing SAGA causal profile"
    "$repo/scripts/baselines/saga_reproduction.sh" verify >/dev/null
  fi
  local gpu
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
  [[ $(wc -l <<<"$gpu") -eq 1 && "$gpu" == *"$expected_gpu_name"* ]] || \
    fail "one $expected_gpu_name GPU is required"
  (( ${gpu##*,} >= min_gpu_memory_mib )) || \
    fail "at least $min_gpu_memory_mib MiB GPU memory is required"
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
  local vpid= proxy_pid= monitor_pid= kv_metrics_pid= rc=0
  mkdir "$cell"
  cleanup() {
    local status=$?
    set +e
    (( rc != 0 )) || rc=$status
    [[ -z "$proxy_pid" ]] || stop_group "$proxy_pid"
    [[ -z "$kv_metrics_pid" ]] || { kill -TERM "$kv_metrics_pid" 2>/dev/null; wait "$kv_metrics_pid" 2>/dev/null; }
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
    printf 'timestamp_s,power_w,memory_mib,utilization_pct,memory_activity_pct\n'
    while true; do
      printf '%s,' "$(date -u +%s.%N)"
      nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits
      sleep 1
    done
  ) >"$cell/gpu.csv" 2>"$cell/gpu.err" &
  monitor_pid=$!

  local common_args=(
    --host 127.0.0.1 --port 8000 --tensor-parallel-size 1
    --gpu-memory-utilization "$gpu_memory_utilization" --max-model-len 131072
    --max-num-seqs 8 --enable-prefix-caching --kv-cache-dtype auto
    --enforce-eager
  )
  local observability_args=()
  if [[ "$serving_metrics" == on ]]; then
    observability_args=(
      --worker-cls scripts.evaluation.cupti_dram_worker.CuptiDramWorker
      --enable-prompt-tokens-details
      --kv-events-config
      '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","buffer_steps":1000000,"hwm":1000000,"max_queue_size":1000000}'
    )
    common_args+=("${observability_args[@]}")
  fi
  local server=("$vllm" serve "$model" "${common_args[@]}" --scheduling-policy priority)
  local server_env=(VLLM_NO_USAGE_STATS=1 CUDA_VISIBLE_DEVICES=0)
  if [[ "$serving_metrics" == on ]]; then
    local cupti_pythonpath=$cupti_overlay
    if [[ "$method" == continuum-public || "$method" == continuum-reproduction || "$method" == continuum-reproduction-oracle-length ]]; then
      cupti_pythonpath="$cupti_overlay/cuda-bindings:$cupti_pythonpath"
    fi
    server_env+=(
      CUPTI_DRAM_CSV="$cell/dram-bandwidth.csv"
      CUPTI_DRAM_READY="$cell/dram-bandwidth-ready"
      CUPTI_DRAM_ERROR="$cell/dram-bandwidth.err"
      PYTHONPATH="$cupti_pythonpath:$repo${PYTHONPATH:+:$PYTHONPATH}"
      LD_LIBRARY_PATH="$cupti_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      LD_PRELOAD="$cupti_lib/libcupti.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
    )
  fi
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
      server=("$repo/scripts/baselines/continuum_reproduction.sh" serve "$model" --dtype bfloat16 --kv-cache-dtype auto "${observability_args[@]}")
      server_env+=(CONTINUUM_REPRODUCTION_PROFILE="$continuum_profile" CONTINUUM_REPRODUCTION_MODE=prefill RUN_OUTPUT_DIR="$cell/continuum" GPU_MEMORY_UTILIZATION="$gpu_memory_utilization")
      ;;
    continuum-reproduction-oracle-length)
      server=("$repo/scripts/baselines/continuum_reproduction.sh" serve-oracle-length "$model" --dtype bfloat16 --kv-cache-dtype auto "${observability_args[@]}")
      server_env+=(CONTINUUM_REPRODUCTION_PROFILE="$continuum_profile" CONTINUUM_REPRODUCTION_MODE=prefill RUN_OUTPUT_DIR="$cell/continuum" GPU_MEMORY_UTILIZATION="$gpu_memory_utilization")
      ;;
    native-priority-aging)
      server=("$repo/scripts/baselines/native_priority_aging.sh" serve "$model" "${common_args[@]}")
      server_env+=(
        NATIVE_PRIORITY_AGING_BYPASS_LIMIT=8
        NATIVE_PRIORITY_AGING_EVENT_LOG="$cell/native-priority-aging.jsonl"
      )
      ;;
    cachewise-disabled)
      server=("$repo/scripts/baselines/cachewise_reproduction.sh" serve-disabled "$model" "${common_args[@]}")
      ;;
    cachewise)
      server=("$repo/scripts/baselines/cachewise_reproduction.sh" serve "$model" "${common_args[@]}")
      ;;
    cachewise-oracle-length)
      server=("$repo/scripts/baselines/cachewise_reproduction.sh" serve-oracle-length "$model" "${common_args[@]}")
      server_env+=(
        CACHEWISE_ORACLE_LENGTH_EVENT_LOG="$cell/cachewise-oracle-length.jsonl"
        CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN="$cachewise_oracle_prefill_ms_per_token"
        CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN="$cachewise_oracle_decode_ms_per_token"
      )
      ;;
    saga)
      server=("$repo/scripts/baselines/saga_reproduction.sh" serve-backend "$model" "${common_args[@]}")
      ;;
  esac
  local vllm_launch=(setsid)
  if [[ "$serving_metrics" == on ]]; then
    vllm_launch+=(
      sudo -n setpriv --reuid="$(id -un)" --regid="$(id -gn)" --init-groups
      --inh-caps="+$perfmon_cap" --ambient-caps="+$perfmon_cap"
      env HOME="$HOME" PATH="$PATH" "${server_env[@]}"
    )
  else
    vllm_launch+=(env "${server_env[@]}")
  fi
  [[ -z "$vllm_cpuset" ]] || vllm_launch+=(taskset -c "$vllm_cpuset")
  printf '%q ' "${vllm_launch[@]}" "${server[@]}" >"$cell/vllm.argv"
  printf '\n' >>"$cell/vllm.argv"
  "${vllm_launch[@]}" "${server[@]}" >"$cell/vllm.log" 2>&1 &
  vpid=$!
  wait_http http://127.0.0.1:8000/v1/models "$vpid" "$cell/vllm.log"
  if [[ "$serving_metrics" == on ]]; then
    [[ -f "$cell/dram-bandwidth-ready" ]] || fail "CUPTI DRAM sampler did not become ready"
    [[ ! -s "$cell/dram-bandwidth.err" ]] || fail "CUPTI DRAM sampler reported an error"
    curl -fsS http://127.0.0.1:8000/metrics >"$cell/vllm-metrics-start.prom"
    local metric
    for metric in \
      vllm:prefix_cache_queries_total \
      vllm:prefix_cache_hits_total \
      vllm:num_preemptions_total \
      vllm:prompt_tokens_total \
      vllm:generation_tokens_total; do
      grep -Fq "$metric" "$cell/vllm-metrics-start.prom" || \
        fail "vLLM does not expose required metric: $metric"
    done
    "$python" "$repo/scripts/evaluation/collect_vllm_kv_events.py" \
      --endpoint tcp://127.0.0.1:5557 \
      --replay-endpoint tcp://127.0.0.1:5558 \
      --ready-file "$cell/kv-events-ready" \
      --events-jsonl "$cell/kv-events.jsonl" \
      --summary-json "$cell/kv-events-summary.json" &
    kv_metrics_pid=$!
    for _ in $(seq 1 100); do
      [[ -f "$cell/kv-events-ready" ]] && break
      kill -0 "$kv_metrics_pid" 2>/dev/null || fail "KV event collector stopped during startup"
      sleep 0.1
    done
    [[ -f "$cell/kv-events-ready" ]] || fail "KV event collector did not become ready"
    sleep 0.5
  fi

  local api=http://127.0.0.1:8000/v1 shadow_mode=vllm
  if [[ "$method" == thunderagent ]]; then
    api=http://127.0.0.1:9000/v1
    shadow_mode=thunderagent
    local proxy_launch=(setsid)
    [[ -z "$vllm_cpuset" ]] || proxy_launch+=(taskset -c "$vllm_cpuset")
    printf '%q ' env THUNDERAGENT_BACKENDS=http://127.0.0.1:8000 \
      THUNDERAGENT_PORT=9000 THUNDERAGENT_PROFILE_DIR="$cell/profiles" \
      "${proxy_launch[@]}" "$repo/scripts/baselines/thunderagent_official.sh" serve \
      >"$cell/proxy.argv"
    printf '\n' >>"$cell/proxy.argv"
    THUNDERAGENT_BACKENDS=http://127.0.0.1:8000 \
      THUNDERAGENT_PORT=9000 THUNDERAGENT_PROFILE_DIR="$cell/profiles" \
      "${proxy_launch[@]}" "$repo/scripts/baselines/thunderagent_official.sh" serve \
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
  elif [[ "$method" == continuum-public || "$method" == continuum-reproduction || "$method" == continuum-reproduction-oracle-length ]]; then
    shadow_mode=continuum-public
  elif [[ "$method" == native-priority || "$method" == native-priority-aging ]]; then
    shadow_mode=native-priority
  elif [[ "$method" == cachewise || "$method" == cachewise-oracle-length ]]; then
    shadow_mode=cachewise
  elif [[ "$method" == saga ]]; then
    shadow_mode=saga
  fi

  local simulate=(
    "$python" -m trace_collect.cli simulate --manifest "$manifest"
    --output-dir "$cell/output" --container docker --network-mode host
    --concurrency "$concurrency" --workers 1 --prep-concurrency 8
    --replay-speed 1 --shadow-llm-api-base "$api"
    --shadow-llm-model "$model" --shadow-llm-timeout-s "$shadow_llm_timeout_s"
    --shadow-llm-seed 0 --shadow-llm-mode "$shadow_mode"
    --resource-monitoring "$resource_monitoring" --pmu-monitoring off
    --memory-bandwidth-monitoring off
  )
  if [[ "$trace_tool_replay" == 0 && "$stage_all_before_replay" == 1 ]]; then
    simulate+=(--stage-all-before-replay)
  fi
  if [[ -n "$replacement_delay_mean_s" ]]; then
    simulate+=(
      --replacement-delay-mean-s "$replacement_delay_mean_s"
      --replacement-seed "$replacement_seed"
    )
  fi
  [[ "$cleanup_images" == 0 ]] || simulate+=(--cleanup-images)
  if [[ -n "$container_cpuset" ]]; then
    simulate+=(--container-cpuset-cpus "$container_cpuset")
  fi
  if [[ -n "$container_cpus" ]]; then
    simulate+=(--container-cpus "$container_cpus")
  fi
  if [[ "$method" == cachewise || "$method" == cachewise-oracle-length ]]; then
    simulate+=(
      --shadow-llm-cachewise-predictor-checkout "$cachewise_checkout"
      --shadow-llm-cachewise-models-dir "$cachewise_models"
    )
    [[ "$method" != cachewise-oracle-length ]] || \
      simulate+=(--shadow-llm-oracle-output-priority)
  elif [[ "$method" == saga ]]; then
    simulate+=(--shadow-llm-saga-profile "$saga_profile")
  fi
  printf '%q ' env OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 OPENCLAW_REPLAY_TRACE_TOOLS="$trace_tool_replay" OPENCLAW_REPLAY_TRACE_TOOL_SPEED="$trace_tool_replay_speed" PYTHONPATH="$repo/src:$repo" "${simulate[@]}" >"$cell/simulate.argv"
  printf '\n' >>"$cell/simulate.argv"
  set +e
  OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 OPENCLAW_REPLAY_TRACE_TOOLS="$trace_tool_replay" OPENCLAW_REPLAY_TRACE_TOOL_SPEED="$trace_tool_replay_speed" PYTHONPATH="$repo/src:$repo" \
    "${simulate[@]}" >"$cell/simulate.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$cell/simulate-exit-code"
  (( rc == 0 )) || return "$rc"
  "$python" - "$cell/output/throughput_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if not (
    summary["attempted_traces"] > 0
    and summary["completed_traces"] == summary["attempted_traces"]
    and summary["failed_traces"] == 0
):
    raise SystemExit(f"task failures: {summary['failed_traces']}")
PY
  if [[ "$serving_metrics" == on ]]; then
    sleep 1
    curl -fsS http://127.0.0.1:8000/metrics >"$cell/vllm-metrics-final.prom"
    kill -TERM "$kv_metrics_pid"
    wait "$kv_metrics_pid" || fail "KV event collector failed"
    kv_metrics_pid=
    stop_group "$vpid"
    vpid=
    [[ -s "$cell/dram-bandwidth.csv" ]] || fail "CUPTI DRAM telemetry is missing"
    [[ ! -s "$cell/dram-bandwidth.err" ]] || fail "CUPTI DRAM sampler reported an error"
  fi
  kill -0 "$monitor_pid" 2>/dev/null || fail "GPU telemetry stopped early"
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=
  "$python" - "$cell/gpu.csv" "$cell/gpu.err" <<'PY'
import csv
import math
import sys
from pathlib import Path

csv_path, error_path = map(Path, sys.argv[1:])
if error_path.read_text().strip():
    raise SystemExit("GPU telemetry reported an error")
with csv_path.open(newline="") as handle:
    rows = list(csv.reader(handle))
expected = [
    "timestamp_s", "power_w", "memory_mib",
    "utilization_pct", "memory_activity_pct",
]
if not rows or rows[0] != expected or len(rows) < 2:
    raise SystemExit("GPU telemetry is missing")
for row in rows[1:]:
    if len(row) != len(expected):
        raise SystemExit("GPU telemetry row is incomplete")
    values = [float(value.strip()) for value in row]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("GPU telemetry contains a non-finite value")
PY
  if [[ "$serving_metrics" == on ]]; then
    "$python" "$repo/scripts/evaluation/summarize_serving_metrics.py" \
      --throughput-summary "$cell/output/throughput_summary.json" \
      --gpu-csv "$cell/gpu.csv" \
      --dram-bandwidth-csv "$cell/dram-bandwidth.csv" \
      --prometheus-start "$cell/vllm-metrics-start.prom" \
      --prometheus-final "$cell/vllm-metrics-final.prom" \
      --kv-events-summary "$cell/kv-events-summary.json" \
      --output "$cell/serving_metrics.json" \
      --requests-output "$cell/request_metrics.jsonl"
  fi
)

run_all() {
  preflight
  mkdir "$run_root"
  RUN_ROOT="$run_root" MODEL="$model" MANIFEST="$manifest" \
    CELLS="${cells[*]}" CONCURRENCY="$concurrency" \
    CONTAINER_CPUSET="$container_cpuset" CONTAINER_CPUS="$container_cpus" \
    VLLM_CPUSET="$vllm_cpuset" \
    CONTINUUM_PROFILE="$continuum_profile" \
    SAGA_PROFILE="$saga_profile" \
    TRACE_TOOL_REPLAY="$trace_tool_replay" \
    TRACE_TOOL_REPLAY_SPEED="$trace_tool_replay_speed" \
    STAGE_ALL_BEFORE_REPLAY="$stage_all_before_replay" \
    REPLACEMENT_DELAY_MEAN_S="$replacement_delay_mean_s" \
    REPLACEMENT_SEED="$replacement_seed" \
    CLEANUP_IMAGES="$cleanup_images" \
    RESOURCE_MONITORING="$resource_monitoring" \
    SERVING_METRICS="$serving_metrics" \
    SHADOW_LLM_TIMEOUT_S="$shadow_llm_timeout_s" \
    EXPECTED_GPU_NAME="$expected_gpu_name" \
    MIN_GPU_MEMORY_MIB="$min_gpu_memory_mib" \
    GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" \
    CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN="$cachewise_oracle_prefill_ms_per_token" \
    CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN="$cachewise_oracle_decode_ms_per_token" \
    UV_LOCK_SHA256="$(sha256sum "$repo/uv.lock" | awk '{print $1}')" \
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
    "tool_execution": (
      "trace_timed_external_service"
      if os.environ["TRACE_TOOL_REPLAY"] == "1"
      else "task_container"
    ),
    "tool_replay_speed": (
      float(os.environ["TRACE_TOOL_REPLAY_SPEED"])
      if os.environ["TRACE_TOOL_REPLAY"] == "1"
      else None
    ),
    "shadow_llm_timeout_s": float(os.environ["SHADOW_LLM_TIMEOUT_S"]),
    "stage_all_before_replay": os.environ["STAGE_ALL_BEFORE_REPLAY"] == "1",
    **(
      {"replacement_load": {
        "delay_distribution": "exponential",
        "delay_mean_s": float(os.environ["REPLACEMENT_DELAY_MEAN_S"]),
        "seed": int(os.environ["REPLACEMENT_SEED"]),
        "measured_cycle": 0,
        "replacement_source": "same trace in a fresh container",
        "stop_condition": "all measured tasks terminal"
      }}
      if os.environ["REPLACEMENT_DELAY_MEAN_S"] else {}
    ),
    "cleanup_images": os.environ["CLEANUP_IMAGES"] == "1",
    "resource_monitoring": os.environ["RESOURCE_MONITORING"],
    "serving_metrics": os.environ["SERVING_METRICS"] == "on",
    "expected_gpu_name": os.environ["EXPECTED_GPU_NAME"],
    "min_gpu_memory_mib": int(os.environ["MIN_GPU_MEMORY_MIB"]),
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
  },
  "comparison": "paper baselines on one fixed agent-trajectory replay workload",
  "continuum_reproduction_profile": os.environ["CONTINUUM_PROFILE"] or None,
  "continuum_reproduction_oracle_length": "Continuum pinned-first; exact remaining output tokens within the eligible tier",
  "saga_profile": os.environ["SAGA_PROFILE"] or None,
  "agentix_queue_upper_bounds_s": [0.25, 1, 4, 16],
  "native_priority": {
    "initial_request_priority": 1,
    "return_request_priority": 0,
    "aging_bypass_limit": 8,
    "aging_unit": "waiting-to-running priority-0 admissions",
    "dependency_lock_sha256": os.environ["UV_LOCK_SHA256"]
  },
  "cachewise_disabled": "same patched fork/config without CacheWise scheduling flags or policy payloads",
  "cachewise_oracle_length": {
    "description": "CacheWise KV and waiting policy plus exact trace output length in a cache-aware service score",
    "prefill_ms_per_token": (
      float(os.environ["CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN"])
      if os.environ["CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN"] else None
    ),
    "decode_ms_per_token": (
      float(os.environ["CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN"])
      if os.environ["CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN"] else None
    )
  },
  "cachewise_tool_mapping": {
    "exec": "Bash", "read_file": "Read", "edit_file": "Edit",
    "write_file": "Write", "list_dir": "Glob"
  },
  "continuum_replay_transport": {
    "tool_signal": "source tool signature consumed only at LLM completion",
    "terminal_signal": "program release after replay completion"
  },
  "primary_metrics": [
    "mean_task_jct_min", "p95_task_jct_min", "tasks_per_hour",
    "makespan", "all_request_p99_ttft"
  ],
  "cachewise_requirement": {
    "metric": "tasks_per_hour",
    "minimum_ratio_to_fcfs": 1.20
  },
  "serving_observability": {
    "request_metrics": "cached prompt tokens, TTFT, TPOT, and decode throughput",
    "prefix_cache": "whole-run request cached-token fraction plus secondary cumulative vLLM lookup counters",
    "kv_cache": "all BlockStored and BlockRemoved events plus preemption count and defined recomputation total",
    "gpu_memory": (
      "vLLM worker CUDA-context DRAM read/write bytes per second from CUPTI PM sampling, plus nvidia-smi device-memory activity time"
      if os.environ["SERVING_METRICS"] == "on"
      else "nvidia-smi device-memory activity time"
    ),
    "task_start": "container_startup.started_at: task preparation start before image and container phases"
  },
  "interpretation": "Physical baseline measurement; descriptive comparison only."
}, indent=2) + "\n")
PY
  local failed=0 cell cell_rc
  for cell in "${cells[@]}"; do
    set +e
    run_cell "$cell"
    cell_rc=$?
    set -e
    if (( cell_rc != 0 )); then
      printf 'cell %s failed; continuing\n' "$cell" >&2
      failed=1
    fi
  done
  return "$failed"
}

case ${1:-} in
  --preflight) preflight ;;
  --run) run_all ;;
  *) echo "usage: $0 --preflight|--run" >&2; exit 2 ;;
esac
