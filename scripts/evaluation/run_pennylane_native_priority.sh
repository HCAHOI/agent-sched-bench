#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && git rev-parse --show-toplevel)
model=NousResearch/Meta-Llama-3.1-8B-Instruct
run_root=/home/Ubuntu/pennylane-native-priority-physical-v1-20260818
result_path="$repo/analysis/results/pennylane-native-priority-physical-development-v1/result.json"
manifest="$repo/analysis/development/pennylane-native-priority-v1/manifest.yaml"
resource_profile="$repo/analysis/development/pennylane-native-priority-v1/resource-profile.yaml"
analyzer="$repo/scripts/evaluation/evaluate_pennylane_native_priority.py"
runner="$repo/scripts/evaluation/run_pennylane_native_priority.sh"
objective="$repo/analysis/development/tool-resource-canonical-objective.md"
python="$repo/.venv/bin/python"
vllm="$repo/.venv/bin/vllm"
power_limit_w=250
max_model_len=131072

cells=(
  fixed-r1
  feedback-r1
  priority-feedback-r1
  priority-feedback-r2
  feedback-r2
  fixed-r2
)
task_ids=(
  PennyLaneAI__pennylane-3182
  PennyLaneAI__pennylane-5835
  PennyLaneAI__pennylane-4366
  PennyLaneAI__pennylane-4251
  PennyLaneAI__pennylane-1405
  PennyLaneAI__pennylane-6062
  PennyLaneAI__pennylane-5623
  PennyLaneAI__pennylane-6939
)

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  return 1
}

cell_config() {
  case "$1" in
    fixed-r1) printf 'fixed|1|fixed|\n' ;;
    feedback-r1) printf 'feedback|1|feedback|\n' ;;
    priority-feedback-r1) printf 'priority-feedback|1|feedback|1\n' ;;
    priority-feedback-r2) printf 'priority-feedback|2|feedback|1\n' ;;
    feedback-r2) printf 'feedback|2|feedback|\n' ;;
    fixed-r2) printf 'fixed|2|fixed|\n' ;;
    *) fail "unknown cell: $1" ;;
  esac
}

cell_dir() {
  case "$1" in
    fixed-r1) printf 'cell_01_fixed-r1\n' ;;
    feedback-r1) printf 'cell_02_feedback-r1\n' ;;
    priority-feedback-r1) printf 'cell_03_priority-feedback-r1\n' ;;
    priority-feedback-r2) printf 'cell_04_priority-feedback-r2\n' ;;
    feedback-r2) printf 'cell_05_feedback-r2\n' ;;
    fixed-r2) printf 'cell_06_fixed-r2\n' ;;
    *) fail "unknown cell: $1" ;;
  esac
}

preflight() {
  if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    fail "not a Git worktree: $repo"
    return 1
  fi
  if [[ -n $(git -C "$repo" status --porcelain --untracked-files=normal) ]]; then
    fail "worktree must be clean"
    return 1
  fi
  local tracked
  for tracked in "$manifest" "$resource_profile" "$runner" "$analyzer" "$objective"; do
    if ! git -C "$repo" ls-files --error-unmatch "${tracked#"$repo/"}" >/dev/null 2>&1; then
      fail "required input is not tracked: $tracked"
      return 1
    fi
  done
  local command
  for command in git setsid curl nvidia-smi docker sudo ps; do
    if ! command -v "$command" >/dev/null 2>&1; then
      fail "required command is unavailable: $command"
      return 1
    fi
  done
  [[ -x "$python" ]] || { fail "missing project Python: $python"; return 1; }
  [[ -x "$vllm" ]] || { fail "missing vLLM executable: $vllm"; return 1; }
  [[ ! -e "$run_root" ]] || { fail "run root already exists: $run_root"; return 1; }
  [[ ! -e "$result_path" ]] || { fail "result already exists: $result_path"; return 1; }

  EXPECTED_IDS=$(IFS=,; printf '%s' "${task_ids[*]}") \
    MANIFEST="$manifest" PROFILE="$resource_profile" OBJECTIVE="$objective" \
    MODEL="$model" RUN_ROOT="$run_root" RESULT_REL="${result_path#"$repo/"}" \
    MAX_MODEL_LEN="$max_model_len" \
    PYTHONPATH="$repo/src:$repo" "$python" - <<'PY'
import os
from pathlib import Path

import yaml

from trace_collect.simulate_manifest import (
    _assign_replay_instance_ids,
    _load_simulate_manifest,
    _load_trace_session,
)
from trace_collect.simulate_types import LLMTimingConfig
from trace_collect.simulator import _validate_loaded_sessions

expected_ids = os.environ["EXPECTED_IDS"].split(",")
manifest = Path(os.environ["MANIFEST"])
entries = _load_simulate_manifest(manifest, default_task_source=None)
sessions = [
    _load_trace_session(
        entry.trace,
        entry.task_source,
        manifest_index=entry.index,
        docker_image_override=entry.docker_image,
        label=entry.label,
        manifest_depends_on=entry.depends_on,
    )
    for entry in entries
]
_assign_replay_instance_ids(sessions)
_validate_loaded_sessions(
    sessions,
    mode="cloud_model",
    replay_speed=1.0,
    llm_timing=LLMTimingConfig(),
)
actual_ids = [session.task_instance_id for session in sessions]
assert actual_ids == expected_ids, (actual_ids, expected_ids)
assert all(not session.depends_on for session in sessions)
actions = [action for session in sessions for action in session.actions]
counts = {
    "actions": len(actions),
    "llm": sum(action["action_type"] == "llm_call" for action in actions),
    "tool": sum(action["action_type"] == "tool_exec" for action in actions),
    "exec": sum(
        action["action_type"] == "tool_exec"
        and action.get("data", {}).get("tool_name") == "exec"
        for action in actions
    ),
}
assert counts == {"actions": 972, "llm": 490, "tool": 482, "exec": 370}, counts
max_context_tokens = max(
    action["data"]["prompt_tokens"] + action["data"]["completion_tokens"]
    for action in actions
    if action["action_type"] == "llm_call"
)
assert max_context_tokens == 111_057, max_context_tokens
assert max_context_tokens <= int(os.environ["MAX_MODEL_LEN"])

profile = yaml.safe_load(Path(os.environ["PROFILE"]).read_text())
assert profile == {
    "tool_resource": {
        "endpoint": "unix:///run/agent-sched/resource/resource.sock",
        "behavior": "observe",
        "update_policy": "frozen",
        "snapshot": "latest_at_run_start",
        "telemetry_requirement": "required_for_valid_evidence",
        "latency_bucket_edges_ms": [500, 2000, 8000, 30000],
    }
}

objective = Path(os.environ["OBJECTIVE"]).read_text()
for frozen_value in (
    os.environ["MODEL"],
    os.environ["RUN_ROOT"],
    os.environ["RESULT_REL"],
):
    assert frozen_value in objective, f"canonical objective is missing {frozen_value!r}"
PY
  printf 'preflight passed for %s at %s\n' "$(git -C "$repo" rev-parse HEAD)" "$run_root"
}

write_argv() {
  local output=$1
  shift
  printf '%q ' "$@" >"$output"
  printf '\n' >>"$output"
}

stop_pid() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    local _
    for _ in $(seq 1 120); do
      kill -0 "$pid" 2>/dev/null || break
      [[ $(awk '/^State:/{print $2}' "/proc/$pid/status" 2>/dev/null || true) == Z ]] && break
      sleep 0.5
    done
  fi
  wait "$pid" 2>/dev/null || true
}

process_group_has_live_members() {
  local pgid=$1
  ps -eo pgid=,stat= | awk -v pgid="$pgid" \
    '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { exit !found }'
}

run_cell() (
  set -euo pipefail
  local cell_name=$1
  local method repetition arm borrower_priority
  IFS='|' read -r method repetition arm borrower_priority < <(cell_config "$cell_name")
  local cell="$run_root/$(cell_dir "$cell_name")"
  [[ ! -e "$cell" ]] || fail "cell already exists: $cell"
  mkdir -p "$cell/output" "$cell/telemetry-state"
  date -u +%FT%TZ >"$cell/cell-start-utc.txt"

  local vpid= rpid= tpid= twrapper= mpid=
  cleanup_cell() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_pid "$rpid"
    if [[ -n "$tpid" ]] && sudo -n kill -0 "$tpid" 2>/dev/null; then
      sudo -n kill -TERM "$tpid"
    fi
    [[ -n "$twrapper" ]] && wait "$twrapper" 2>/dev/null
    if [[ -n "$vpid" ]]; then
      local vllm_lifecycle_rc=0
      if ! kill -0 "$vpid" 2>/dev/null; then
        printf 'vLLM leader exited before lifecycle cleanup\n' >&2
        vllm_lifecycle_rc=1
      fi
      if process_group_has_live_members "$vpid"; then
        kill -TERM -- "-$vpid"
        for _ in $(seq 1 120); do
          process_group_has_live_members "$vpid" || break
          sleep 0.5
        done
        if process_group_has_live_members "$vpid"; then
          printf 'vLLM process group required SIGKILL\n' >&2
          kill -KILL -- "-$vpid"
          vllm_lifecycle_rc=1
        fi
      else
        printf 'vLLM process group was not live at lifecycle cleanup\n' >&2
        vllm_lifecycle_rc=1
      fi
      wait "$vpid" 2>/dev/null || true
      date -u +%FT%TZ >"$cell/vllm-stop-utc.txt"
      if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        printf 'vLLM endpoint remained live after process-group stop\n' >&2
        vllm_lifecycle_rc=1
      fi
      printf '%s\n' "$vllm_lifecycle_rc" >"$cell/vllm-exit-code"
      if [[ $vllm_lifecycle_rc -ne 0 && $rc -eq 0 ]]; then
        rc=1
      fi
    fi
    stop_pid "$mpid"
    date -u +%FT%TZ >"$cell/cell-end-utc.txt"
    printf '%s\n' "$rc" >"$cell/cell-exit-code"
    exit "$rc"
  }
  trap cleanup_cell EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  local uid gid
  uid=$(id -u)
  gid=$(id -g)
  sudo -n install -d -m 0750 -o "$uid" -g "$gid" /run/agent-sched /run/agent-sched/resource
  [[ ! -e /run/agent-sched/telemetry.sock ]]
  [[ ! -e /run/agent-sched/resource/resource.sock ]]
  sudo -n chown root:root "$cell/telemetry-state"
  sudo -n chmod 700 "$cell/telemetry-state"
  sudo -n nvidia-smi -pl "$power_limit_w" >"$cell/power-set.log"

  CELL_NAME="$cell_name" METHOD="$method" REPETITION="$repetition" \
    ARM="$arm" BORROWER_PRIORITY="$borrower_priority" MODEL="$model" \
    GIT_COMMIT="$(git -C "$repo" rev-parse HEAD)" "$python" - "$cell/cell-metadata.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "cell_name": os.environ["CELL_NAME"],
    "method": os.environ["METHOD"],
    "repetition": int(os.environ["REPETITION"]),
    "simulator_arm": os.environ["ARM"],
    "borrower_priority": (
        int(os.environ["BORROWER_PRIORITY"])
        if os.environ["BORROWER_PRIORITY"] else None
    ),
    "git_commit": os.environ["GIT_COMMIT"],
    "model": os.environ["MODEL"],
}, indent=2, sort_keys=True) + "\n")
PY

  (
    printf 'timestamp_utc_s,power_limit_w,power_draw_w,temperature_gpu_c,temperature_memory_c,memory_used_mib,utilization_gpu_pct,sm_clock_mhz,sw_thermal,hw_thermal,sw_power_cap\n'
    while true; do
      printf '%s,' "$(date -u +%s.%N)"
      nvidia-smi --query-gpu=power.limit,power.draw,temperature.gpu,temperature.memory,memory.used,utilization.gpu,clocks.current.sm,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap --format=csv,noheader,nounits
      sleep 1
    done
  ) >"$cell/gpu-telemetry.csv" 2>"$cell/gpu-telemetry.err" &
  mpid=$!

  sudo -n sh -c "echo \$\$ > '$cell/telemetryd.pid'; exec env PYTHONPATH='$repo/src:$repo:/usr/lib/python3/dist-packages' '$python' -m tool_resource.telemetryd --socket /run/agent-sched/telemetry.sock --allowed-uid '$uid' --socket-gid '$gid' --container-runtime docker --state-dir '$cell/telemetry-state'" >"$cell/telemetryd.log" 2>&1 &
  twrapper=$!
  for _ in $(seq 1 60); do
    [[ -s "$cell/telemetryd.pid" && -S /run/agent-sched/telemetry.sock ]] && break
    sudo -n kill -0 "$twrapper"
    sleep 0.5
  done
  tpid=$(<"$cell/telemetryd.pid")
  sudo -n kill -0 "$tpid"

  PYTHONPATH="$repo/src:$repo" "$python" -m tool_resource.resource_agentd \
    --socket /run/agent-sched/resource/resource.sock \
    --database "$cell/observations.sqlite3" \
    --telemetry-socket /run/agent-sched/telemetry.sock \
    --telemetry-peer-uid 0 >"$cell/resource-agentd.log" 2>&1 &
  rpid=$!
  printf '%s\n' "$rpid" >"$cell/resource-agentd.pid"
  for _ in $(seq 1 60); do
    [[ -S /run/agent-sched/resource/resource.sock ]] && break
    kill -0 "$rpid"
    sleep 0.5
  done
  TS=/run/agent-sched/telemetry.sock RS=/run/agent-sched/resource/resource.sock \
    PYTHONPATH="$repo/src:$repo" "$python" -c \
    'import os; from tool_resource.telemetry_protocol import TelemetryUnixTransport; from tool_resource.resource_protocol import ResourceUnixTransport; TelemetryUnixTransport(os.environ["TS"]).ping(); ResourceUnixTransport(os.environ["RS"]).ping()'

  local vllm_args=(
    "$vllm" serve "$model"
    --host 127.0.0.1 --port 8000
    --tensor-parallel-size 1
    --gpu-memory-utilization 0.90
    --max-model-len "$max_model_len"
    --max-num-seqs 8
    --enable-prefix-caching
    --kv-cache-dtype auto
    --enforce-eager
    --scheduling-policy priority
  )
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    fail "port 8000 already serves a model before $cell_name"
  fi
  write_argv "$cell/vllm.argv" env CUDA_VISIBLE_DEVICES=0 "${vllm_args[@]}"
  date -u +%FT%TZ >"$cell/vllm-start-utc.txt"
  CUDA_VISIBLE_DEVICES=0 setsid "${vllm_args[@]}" >"$cell/vllm.log" 2>&1 &
  vpid=$!
  printf '%s\n' "$vpid" >"$cell/vllm.pid"
  for _ in $(seq 1 1200); do
    if curl -fsS http://127.0.0.1:8000/v1/models >"$cell/models.json" 2>/dev/null; then
      break
    fi
    kill -0 "$vpid"
    sleep 0.5
  done
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null

  local simulate_args=(
    "$python" -m trace_collect.cli simulate
    --manifest "$manifest"
    --output-dir "$cell/output"
    --container docker
    --network-mode host
    --concurrency 4
    --workers 1
    --prep-concurrency 8
    --stage-all-before-replay
    --container-cpus 2
    --replay-speed 1
    --shadow-llm-api-base http://127.0.0.1:8000/v1
    --shadow-llm-model "$model"
    --shadow-llm-timeout-s 300
    --shadow-llm-seed 0
    --tool-gap-loan-arm "$arm"
    --tool-resource-profile "$resource_profile"
    --resource-monitoring off
    --pmu-monitoring off
    --memory-bandwidth-monitoring off
  )
  [[ -z "$borrower_priority" ]] || simulate_args+=(--tool-gap-borrower-priority "$borrower_priority")
  write_argv "$cell/simulate.argv" env OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 \
    "PYTHONPATH=$repo/src:$repo" "${simulate_args[@]}"
  date -u +%FT%T.%6NZ >"$cell/simulate-start-utc.txt"
  set +e
  OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 PYTHONPATH="$repo/src:$repo" \
    "${simulate_args[@]}" >"$cell/simulate.log" 2>&1
  local simulate_rc=$?
  set -e
  date -u +%FT%T.%6NZ >"$cell/simulate-end-utc.txt"
  printf '%s\n' "$simulate_rc" >"$cell/simulate-exit-code"
  [[ $simulate_rc -eq 0 ]] || exit "$simulate_rc"
  sleep 2

  set +e
  "$python" "$analyzer" --run-root "$run_root" --validate-cell "$cell_name" \
    >"$cell/validate-cell.log" 2>&1
  local validation_rc=$?
  set -e
  printf '%s\n' "$validation_rc" >"$cell/validate-cell-exit-code"
  [[ $validation_rc -eq 0 ]] || exit "$validation_rc"
)

run_all() {
  preflight
  local gpu_rows
  gpu_rows=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
  [[ $(wc -l <<<"$gpu_rows") -eq 1 ]] || fail "exactly one visible GPU is required"
  [[ "$gpu_rows" == *A100* ]] || fail "an A100 is required, got: $gpu_rows"
  local gpu_memory_mib=${gpu_rows##*,}
  gpu_memory_mib=${gpu_memory_mib// /}
  (( gpu_memory_mib >= 80000 )) || fail "an A100 80GB is required, got: $gpu_rows"

  local original_power_limit
  original_power_limit=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | tr -d ' ')
  mkdir "$run_root"
  date -u +%FT%TZ >"$run_root/driver-start-utc.txt"

  RUN_ROOT="$run_root" RESULT_PATH="$result_path" MODEL="$model" \
    MANIFEST="$manifest" PROFILE="$resource_profile" GPU="$gpu_rows" \
    POWER_LIMIT="$power_limit_w" ORIGINAL_POWER_LIMIT="$original_power_limit" \
    MAX_MODEL_LEN="$max_model_len" \
    GIT_COMMIT="$(git -C "$repo" rev-parse HEAD)" \
    CELL_ORDER="$(IFS=,; printf '%s' "${cells[*]}")" \
    TASK_IDS="$(IFS=,; printf '%s' "${task_ids[*]}")" \
    "$python" - "$run_root/run-metadata.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "git_commit": os.environ["GIT_COMMIT"],
    "model": os.environ["MODEL"],
    "run_root": os.environ["RUN_ROOT"],
    "result_path": os.environ["RESULT_PATH"],
    "manifest": os.environ["MANIFEST"],
    "resource_profile": os.environ["PROFILE"],
    "cell_order": os.environ["CELL_ORDER"].split(","),
    "task_ids": os.environ["TASK_IDS"].split(","),
    "expected_counts": {"actions": 972, "llm_calls": 490, "tool_calls": 482, "shell_execs": 370},
    "gpu": os.environ["GPU"],
    "power_limit_w": float(os.environ["POWER_LIMIT"]),
    "original_power_limit_w": float(os.environ["ORIGINAL_POWER_LIMIT"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "source_max_context_tokens": 111057,
    "container_cpu_cap": 2,
    "paired_workload_contract": 2,
    "vllm_scheduling_policy": "priority",
    "host_request_admission_cap": None,
    "server_queueing_metric": "shadow_generation.ttft_ms",
}, indent=2, sort_keys=True) + "\n")
PY

  driver_cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    sudo -n nvidia-smi -pl "$original_power_limit" >"$run_root/power-restore.log" 2>&1
    date -u +%FT%TZ >"$run_root/driver-end-utc.txt"
    printf '%s\n' "$rc" >"$run_root/driver-exit-code"
    exit "$rc"
  }
  trap driver_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  local cell_name
  for cell_name in "${cells[@]}"; do
    run_cell "$cell_name"
  done

  set +e
  "$python" "$analyzer" --run-root "$run_root" --output "$result_path" \
    >"$run_root/analysis.log" 2>&1
  local analysis_rc=$?
  set -e
  printf '%s\n' "$analysis_rc" >"$run_root/analysis-exit-code"
  [[ $analysis_rc -eq 0 ]] || return "$analysis_rc"
}

main() {
  case "${1:-}" in
    --preflight) [[ $# -eq 1 ]] || fail "usage: $0 --preflight|--run"; preflight ;;
    --run) [[ $# -eq 1 ]] || fail "usage: $0 --preflight|--run"; run_all ;;
    *) fail "usage: $0 --preflight|--run" ;;
  esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
