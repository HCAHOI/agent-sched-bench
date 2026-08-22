#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model=${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}
current_root=${CURRENT_RUN_ROOT:-/home/Ubuntu/pennylane-paper-baselines-mixed12-sharedcpu-v1-20260820}
suite_root=${SUITE_ROOT:-/home/Ubuntu/pennylane-paper-baseline-suite-v1-20260820}
full_manifest=${MANIFEST:-$repo/analysis/development/pennylane-mixed-workload-v1/a100-manifest.yaml}
smoke_manifest=${SMOKE_MANIFEST:-$repo/analysis/development/pennylane-mixed-workload-v1/a100-calibration.yaml}
runner=$repo/scripts/evaluation/run_pennylane_thunderagent_baseline.sh
container_cpuset=${CONTAINER_CPUSET:-4-15}
vllm_cpuset=${VLLM_CPUSET:-0-3}
continuum_commit=316a58794a6ff86b216e579b74fd56ed0c5a911f
continuum_python=${CONTINUUM_VENV:-$HOME/.cache/agent-sched-bench/venvs/continuum-public-$continuum_commit}/bin/python
profile_supplied=${CONTINUUM_REPRODUCTION_PROFILE:+1}
profile=${CONTINUUM_REPRODUCTION_PROFILE:-$suite_root/continuum-prefill-a100-120k.json}
full_concurrency=${CONCURRENCY:-16}
trace_tool_replay=${TRACE_TOOL_REPLAY:-0}
methods=(agentix continuum-public continuum-reproduction cachewise)

fail() { printf '%s\n' "$*" >&2; exit 1; }

wait_for_current_run() {
  while [[ ! -f "$current_root/thunderagent-r1/cell-exit-code" ]]; do
    sleep 60
  done
}

install_baselines() {
  "$repo/scripts/baselines/agentix_reproduction.sh" install
  "$repo/scripts/baselines/continuum_public.sh" install
  "$repo/scripts/baselines/continuum_reproduction.sh" install
  "$repo/scripts/baselines/cachewise_official.sh" install
  local models=$HOME/.cache/agent-sched-bench/cachewise-181c435a090d328d00bbbee4c8eeb27d32f3abd2/tool_duration_prediction/models/all_models.pkl
  [[ -f "$models" ]] || "$repo/scripts/baselines/cachewise_official.sh" train-published
  "$repo/scripts/baselines/cachewise_reproduction.sh" install
}

measure_continuum_profile() {
  RUN_OUTPUT_DIR="$suite_root/profile-continuum" \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$repo/src:$repo" "$continuum_python" \
    "$repo/scripts/serving/measure_prefill_cost.py" \
    --model "$model" --kv-cache-dtype auto --kv-layout-dtype bfloat16 \
    --quantization none --context-sweep 8,512,2048,8192,32768,65536,98304,120000 \
    --max-model-len 120001 --repeats 3 --warmup 1 --seed 0 --output "$profile"
  PROFILE="$profile" MODEL="$model" "$repo/.venv/bin/python" - <<'PY'
import json, os, subprocess
from pathlib import Path

path = Path(os.environ["PROFILE"])
payload = json.loads(path.read_text())
name, memory = [part.strip() for part in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    text=True,
).strip().split(",", 1)]
layout = payload["kv_layout"]
payload["runtime_binding"] = {
    "model": os.environ["MODEL"],
    "model_dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "gpu": {"name": name, "memory_mib": int(float(memory)), "count": 1},
    "kv_layout": {**layout, "bytes_per_token_per_gpu": layout["bytes_per_token"]},
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_one() {
  local method=$1 manifest=$2 root=$3 concurrency=$4
  MODEL="$model" MANIFEST="$manifest" RUN_ROOT="$root" CELLS="$method-r1" \
    CONCURRENCY="$concurrency" CONTAINER_CPUSET="$container_cpuset" \
    CONTAINER_CPUS=2 VLLM_CPUSET="$vllm_cpuset" \
    TRACE_TOOL_REPLAY="$trace_tool_replay" \
    CONTINUUM_REPRODUCTION_PROFILE="$profile" "$runner" --run
}

smoke_succeeded() {
  local method=$1 cell="$suite_root/smoke-$1/$1-r1"
  [[ -f "$cell/cell-exit-code" && -f "$cell/simulate-exit-code" ]] &&
    [[ $(<"$cell/cell-exit-code") == 0 && $(<"$cell/simulate-exit-code") == 0 ]]
}

check_repo() {
  [[ $(git -C "$repo" status --porcelain) == "" ]] || fail "worktree must be clean"
  "$repo/.venv/bin/python" -c 'import loguru, trace_collect'
}

run_smokes_and_full() {
  [[ ! -e "$suite_root/full" ]] || fail "full run already exists: $suite_root/full"

  local failed=0 method
  for method in "${methods[@]}"; do
    smoke_succeeded "$method" && continue
    [[ ! -e "$suite_root/smoke-$method" ]] || fail "smoke exists without success: $suite_root/smoke-$method"
    if run_one "$method" "$smoke_manifest" "$suite_root/smoke-$method" 1 \
      >"$suite_root/smoke-$method.log" 2>&1; then
      smoke_succeeded "$method" || failed=1
    else
      failed=1
    fi
  done
  (( failed == 0 )) || fail "one or more baseline smokes failed; full runs not started"

  MODEL="$model" MANIFEST="$full_manifest" \
    RUN_ROOT="$suite_root/full" CELLS="${methods[*]/%/-r1}" \
    CONCURRENCY="$full_concurrency" CONTAINER_CPUSET="$container_cpuset" CONTAINER_CPUS=2 \
    TRACE_TOOL_REPLAY="$trace_tool_replay" \
    VLLM_CPUSET="$vllm_cpuset" CONTINUUM_REPRODUCTION_PROFILE="$profile" \
    "$runner" --run >"$suite_root/full.log" 2>&1
}

run_suite() {
  check_repo
  [[ ! -e "$suite_root" ]] || fail "suite root already exists: $suite_root"
  mkdir "$suite_root"
  wait_for_current_run
  install_baselines >"$suite_root/install.log" 2>&1
  if [[ -n "$profile_supplied" ]]; then
    [[ -s "$profile" ]] || fail "provided profile is missing: $profile"
  else
    measure_continuum_profile >"$suite_root/profile.log" 2>&1
  fi
  run_smokes_and_full
}

resume_after_profile() {
  check_repo
  [[ -d "$suite_root" ]] || fail "suite root is missing: $suite_root"
  [[ -s "$profile" ]] || fail "measured profile is missing: $profile"
  run_smokes_and_full
}

case ${1:-} in
  --run) run_suite ;;
  --resume-after-profile) resume_after_profile ;;
  *) echo "usage: $0 --run|--resume-after-profile" >&2; exit 2 ;;
esac
