#!/usr/bin/env bash
# run_campaign.sh — repo-resident GPU recording-campaign runner.
#
# Subcommands: run | status | stop
#
# HARD RULE (the load-bearing safety invariant): this runner and its
# subcommands NEVER perform process-table string matching — no process-table
# name search, no string-grep over `ps`/cmdline to *find* a process to signal.
# The ONLY way a PID is ever signalled is after verify_pid() returns
# "alive-and-ours" for a PID read from a PID file whose paired .meta sidecar
# matches identity (start-time + boot_id + cmdline anchor). This is exactly the
# property the three near-disasters violated (parent-bash killed by a cmdline
# match, sentinel loop matching the deployer heredoc, cleanup matching the
# nohup parent). See scripts/campaign/README.md.
#
# Safe deployment recipe (never put the script body in a process cmdline):
#   deploy via git — commit/push the runner + task TSV, then on the box:
#     ssh <box> 'cd ~/agent-sched-bench && git pull --ff-only && \
#       nohup bash scripts/campaign/run_campaign.sh run \
#         --tasks scripts/campaign/tasks.armC.tsv --campaign armC \
#         > /dev/null 2>&1 & disown'
#   Full recipe + operator subcommands: scripts/campaign/README.md.
#
# Strict mode is applied ONLY when this file is executed as a script (see the
# main-guard at the bottom), NOT when it is sourced. The test harness sources
# this file to unit-test its functions; leaking `set -euo pipefail` (and an
# unbound ${BASH_SOURCE[0]} under `set -u`) into the sourcing shell would abort
# it. Executed directly, the guard re-enables full strict mode so the runner's
# own execution stays as strict as before.

# Resolve this file's own path robustly whether executed or sourced. ${0} is
# the path under both `bash run_campaign.sh` and `source run_campaign.sh` only
# in the executed case; BASH_SOURCE is the reliable source-time path. Guard the
# array read so `set -u` in a sourcing shell cannot trip on it.
_self_src="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$_self_src")" && pwd)"
SELF_PATH="${SCRIPT_DIR}/$(basename "$_self_src")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Command override for tests: by default the runner invokes
#   python -m <TRACE_CLI_MODULE>  (default module: trace_collect.cli)
# Tests set TRACE_CLI_CMD to a dummy program (e.g. "sleep" / "false") so the
# whole lifecycle exercises with no GPU / torch / trace_collect import.
# Default `python` (NOT python3) so --print-cmd reproduces the campaign
# manifest's `python -m trace_collect.cli` invocation verbatim; override with
# PYTHON_BIN if the box's interpreter differs.
PYTHON_BIN="${PYTHON_BIN:-python}"
TRACE_CLI_MODULE="${TRACE_CLI_MODULE:-trace_collect.cli}"
TRACE_CLI_CMD="${TRACE_CLI_CMD:-}"          # if set, used verbatim instead of `python -m <module>`

DEFAULT_TIMEOUT="${CAMPAIGN_DEFAULT_TIMEOUT:-21600}"   # 6h per task
KILL_AFTER="${CAMPAIGN_KILL_AFTER:-60}"                # grace before SIGKILL (data-flush window)
DEFAULT_SEEDS="${SEEDS:-0}"                            # comma/space-separated generation seeds

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { printf 'run_campaign.sh: %s\n' "$*" >&2; exit 2; }

# --------------------------------------------------------------------------
# Portable /proc helpers. On Linux we read /proc directly (the plan's contract:
# /proc/<pid>/stat field 22, /proc/<pid>/cmdline). On a /proc-less host (macOS,
# used only for the local test harness) we derive the SAME identity facts via
# ps so the verification logic is genuinely exercised, never mocked.
# --------------------------------------------------------------------------

current_boot_id() {
  if [[ -r /proc/sys/kernel/random/boot_id ]]; then
    cat /proc/sys/kernel/random/boot_id
  else
    # No kernel boot_id (macOS). Use kernel boot time as a stable per-boot id.
    # sysctl kern.boottime -> "{ sec = 1781600000, usec = 0 } ..."
    if command -v sysctl >/dev/null 2>&1; then
      sysctl -n kern.boottime 2>/dev/null | sed -n 's/.*sec = \([0-9]*\).*/boottime-\1/p'
    else
      echo "boot-unknown"
    fi
  fi
}

# proc_start_time PID -> a per-process start stamp that is STABLE across a
# pause/clock-freeze and CHANGES if the PID is reused by a new process.
# Linux: field 22 of /proc/<pid>/stat (clock ticks since boot). The comm field
# (field 2) may contain spaces and parens, so we parse the tail AFTER the LAST
# ')' rather than by naive whitespace field counting.
proc_start_time() {
  local pid="$1" stat rest f22
  if [[ -r "/proc/${pid}/stat" ]]; then
    stat="$(cat "/proc/${pid}/stat" 2>/dev/null)" || return 1
    rest="${stat##*) }"        # everything after the last ") "  => starts at field 3 (state)
    # fields in $rest: state(3) ppid(4) pgrp(5) session(6) tty_nr(7) tpgid(8)
    # flags(9) minflt(10) cminflt(11) majflt(12) cmajflt(13) utime(14) stime(15)
    # cutime(16) cstime(17) priority(18) nice(19) num_threads(20)
    # itrealvalue(21) starttime(22)  -> the 20th token of $rest
    read -r -a f <<<"$rest"
    f22="${f[19]:-}"           # 0-indexed: 20th token = index 19
    [[ -n "$f22" ]] || return 1
    printf '%s' "$f22"
    return 0
  fi
  # macOS fallback: elapsed-seconds-since-start is monotone-ish but changes on
  # reuse; use the absolute start epoch from `ps -o lstart` for stability.
  if command -v ps >/dev/null 2>&1; then
    local lstart epoch
    lstart="$(ps -o lstart= -p "$pid" 2>/dev/null)" || return 1
    [[ -n "$lstart" ]] || return 1
    epoch="$(date -j -f '%a %b %d %T %Y' "$lstart" +%s 2>/dev/null)" || epoch="$lstart"
    printf 'lstart:%s' "$epoch"
    return 0
  fi
  return 1
}

# proc_cmdline PID -> the process command line as a single space-joined string.
# Linux: /proc/<pid>/cmdline (NUL-separated). macOS: ps -o args.
proc_cmdline() {
  local pid="$1"
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline"
    return 0
  fi
  if command -v ps >/dev/null 2>&1; then
    ps -o args= -p "$pid" 2>/dev/null || return 1
    return 0
  fi
  return 1
}

# --------------------------------------------------------------------------
# Portable per-task timeout.
#
# On Linux (the GPU box) coreutils `timeout` is always present and is used
# verbatim: TERM then KILL after a grace window, exit 124 on timeout. macOS
# (used ONLY for the local GPU-free test harness) ships no `timeout`/`gtimeout`,
# so we provide a pure-bash watchdog with the SAME observable contract:
#   - runs the task in the foreground of a subshell so $TASK_LAUNCH_PID is the
#     task itself (its cmdline carries the anchor we verify);
#   - on timeout: SIGTERM the task, wait up to KILL_AFTER seconds, then SIGKILL,
#     and return 124 (matching coreutils);
#   - otherwise returns the task's own exit code.
# Resolved ONCE at startup so behavior is identical for every task in a queue.
# --------------------------------------------------------------------------
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="$(command -v gtimeout)"
fi

# Two-phase launch so the caller can write the PID file (anchored on the live
# task cmdline) AFTER the task is backgrounded but BEFORE we block on it:
#
#   launch_task SECS LOGFILE PIDVAR -- ARGV... -> backgrounds ARGV (stdout+
#       stderr redirected to LOGFILE), stores the task PID in the var named by
#       PIDVAR. SECS is the per-task timeout (only used on the coreutils path).
#   await_task SECS PID                     -> waits up to SECS for PID; on
#       timeout TERMs it, waits KILL_AFTER, then KILLs, and returns 124;
#       otherwise returns the task's own exit code.
#
# With coreutils `timeout` we wrap the task so the kernel enforces the deadline
# exactly as on the GPU box. Without it (macOS test harness) await_task polls
# `kill -0` (no process-table string match) and signals the single known PID.
launch_task() {
  local secs="$1" logfile="$2" pidvar="$3"
  shift 3
  [[ "$1" == "--" ]] && shift
  if [[ -n "$TIMEOUT_BIN" ]]; then
    "$TIMEOUT_BIN" --signal=TERM --kill-after="$KILL_AFTER" "$secs" "$@" \
      > "$logfile" 2>&1 &
  else
    "$@" > "$logfile" 2>&1 &
  fi
  printf -v "$pidvar" '%s' "$!"
}

await_task() {
  local secs="$1" tpid="$2" rc=0
  if [[ -n "$TIMEOUT_BIN" ]]; then
    # coreutils timeout owns the deadline and exits 124 itself on expiry.
    wait "$tpid" || rc=$?
    return "$rc"
  fi
  local elapsed=0 timed_out=0
  while kill -0 "$tpid" 2>/dev/null; do
    if [[ "$elapsed" -ge "$secs" ]]; then
      timed_out=1
      kill -TERM "$tpid" 2>/dev/null || true
      local grace=0
      while kill -0 "$tpid" 2>/dev/null; do
        sleep 1; grace=$((grace + 1))
        [[ "$grace" -ge "$KILL_AFTER" ]] && { kill -KILL "$tpid" 2>/dev/null || true; break; }
      done
      break
    fi
    sleep 1; elapsed=$((elapsed + 1))
  done
  wait "$tpid" 2>/dev/null || rc=$?
  [[ "$timed_out" -eq 1 ]] && return 124
  return "$rc"
}

# --------------------------------------------------------------------------
# PID-file / meta write + verify
# --------------------------------------------------------------------------

# write_pidfile PIDFILE METAFILE PID ANCHOR
# Records identity captured AT WRITE TIME: pid, start-time, boot_id, cmdline
# anchor (the substring(s) we will later require the live cmdline to contain).
write_pidfile() {
  local pidfile="$1" metafile="$2" pid="$3" anchor="$4"
  local st bid
  st="$(proc_start_time "$pid")" || st=""
  bid="$(current_boot_id)"
  printf '%s\n' "$pid" > "${pidfile}.tmp" && mv -f "${pidfile}.tmp" "$pidfile"
  {
    printf 'pid=%s\n' "$pid"
    printf 'start_time=%s\n' "$st"
    printf 'boot_id=%s\n' "$bid"
    printf 'anchor=%s\n' "$anchor"
  } > "${metafile}.tmp" && mv -f "${metafile}.tmp" "$metafile"
}

clear_pidfile() {
  local pidfile="$1" metafile="${2:-}"
  rm -f "$pidfile"
  [[ -n "$metafile" ]] && rm -f "$metafile"
  return 0
}

# verify_pid PIDFILE METAFILE -> prints one of: alive-and-ours | dead | reused
# Never signals. Pure read-only identity check (the 4-step protocol).
verify_pid() {
  local pidfile="$1" metafile="$2"
  # (1) pid file present?
  [[ -r "$pidfile" ]] || { echo dead; return 0; }
  local pid
  pid="$(head -n1 "$pidfile" 2>/dev/null | tr -dc '0-9')"
  [[ -n "$pid" ]] || { echo dead; return 0; }
  # (2) signal-0 liveness probe.
  if ! kill -0 "$pid" 2>/dev/null; then
    echo dead; return 0
  fi
  # meta is required to certify identity; without it we cannot prove ownership.
  [[ -r "$metafile" ]] || { echo dead; return 0; }
  local m_start m_boot m_anchor
  m_start="$(sed -n 's/^start_time=//p' "$metafile" | head -n1)"
  m_boot="$(sed -n 's/^boot_id=//p' "$metafile" | head -n1)"
  m_anchor="$(sed -n 's/^anchor=//p' "$metafile" | head -n1)"
  # (3) start-time + boot_id must match -> else PID was reused (or rebooted).
  local cur_start cur_boot
  cur_start="$(proc_start_time "$pid")" || cur_start=""
  cur_boot="$(current_boot_id)"
  if [[ -n "$m_boot" && "$m_boot" != "$cur_boot" ]]; then
    echo dead; return 0   # different boot => all old PIDs stale
  fi
  if [[ -n "$m_start" && "$m_start" != "$cur_start" ]]; then
    echo dead; return 0   # start-time mismatch => reused PID, do NOT signal
  fi
  # (4) cmdline must still contain the recorded anchor.
  if [[ -n "$m_anchor" ]]; then
    local cur_cmd
    cur_cmd="$(proc_cmdline "$pid")" || cur_cmd=""
    if [[ "$cur_cmd" != *"$m_anchor"* ]]; then
      echo reused; return 0
    fi
  fi
  echo alive-and-ours; return 0
}

# --------------------------------------------------------------------------
# Status-file (atomic JSON via printf; no jq write-dependency)
# --------------------------------------------------------------------------

# json_escape STR -> minimally-escaped JSON string body (no surrounding quotes).
json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "$s"
}

safe_label() {
  local s="$1"
  s="${s//[^A-Za-z0-9_.-]/_}"
  printf '%s' "$s"
}

emit_seed_list() {
  local raw="$1" seed
  raw="${raw//,/ }"
  for seed in $raw; do
    [[ "$seed" =~ ^-?[0-9]+$ ]] || die "invalid seed '$seed' in --seeds/SEEDS"
    printf '%s\n' "$seed"
  done
}

# Globals populated by `run` and consumed by write_status.
STATUS_FILE=""
ST_CAMPAIGN=""
ST_RUN_DIR=""
ST_QUEUE_PID=""
ST_BOOT_ID=""
ST_STATE="running"
ST_TOTAL=0
ST_COMPLETED=0
ST_CUR_INDEX=0
ST_CUR_TASK_ID=""
ST_CUR_TASK_PID=""
ST_CUR_STARTED_EPOCH=0
ST_LAST_EXIT=0
RESULTS_JSON=""   # accumulated "{...},{...}" results array body

append_result() {
  local task_id="$1" exit_code="$2" timed_out="$3" duration="$4"
  local entry
  entry="$(printf '{"task_id":"%s","exit_code":%s,"timed_out":%s,"duration_s":%s}' \
    "$(json_escape "$task_id")" "$exit_code" "$timed_out" "$duration")"
  if [[ -z "$RESULTS_JSON" ]]; then
    RESULTS_JSON="$entry"
  else
    RESULTS_JSON="${RESULTS_JSON},${entry}"
  fi
}

write_status() {
  [[ -n "$STATUS_FILE" ]] || return 0
  local tmp="${STATUS_FILE}.tmp"
  {
    printf '{\n'
    printf '  "campaign": "%s",\n' "$(json_escape "$ST_CAMPAIGN")"
    printf '  "run_dir": "%s",\n' "$(json_escape "$ST_RUN_DIR")"
    printf '  "queue_pid": %s,\n' "${ST_QUEUE_PID:-null}"
    printf '  "boot_id": "%s",\n' "$(json_escape "$ST_BOOT_ID")"
    printf '  "state": "%s",\n' "$ST_STATE"
    printf '  "total_tasks": %s,\n' "$ST_TOTAL"
    printf '  "completed": %s,\n' "$ST_COMPLETED"
    printf '  "current_index": %s,\n' "$ST_CUR_INDEX"
    printf '  "current_task_id": "%s",\n' "$(json_escape "$ST_CUR_TASK_ID")"
    printf '  "current_task_pid": %s,\n' "${ST_CUR_TASK_PID:-null}"
    printf '  "current_started_epoch": %s,\n' "${ST_CUR_STARTED_EPOCH:-0}"
    printf '  "last_exit_code": %s,\n' "${ST_LAST_EXIT:-0}"
    printf '  "results": [%s],\n' "$RESULTS_JSON"
    printf '  "updated_epoch": %s\n' "$(date +%s)"
    printf '}\n'
  } > "$tmp" && mv -f "$tmp" "$STATUS_FILE"
}

# --------------------------------------------------------------------------
# Command builder (shared by `run` and `--print-cmd`)
# --------------------------------------------------------------------------

# build_task_cmd BENCHMARK EXTRA_ARGS -> prints the full argv (one token per
# line, NUL-safe-ish: callers read with eval-free arrays). Prepends the python
# module (or the TRACE_CLI_CMD override) and appends --benchmark <bm>.
# Emits to stdout, space-joined, for --print-cmd; the run loop builds an array.
emit_cmd_prefix() {
  if [[ -n "$TRACE_CLI_CMD" ]]; then
    printf '%s' "$TRACE_CLI_CMD"
  else
    printf '%s -m %s' "$PYTHON_BIN" "$TRACE_CLI_MODULE"
  fi
}

# --------------------------------------------------------------------------
# Argument parsing helpers
# --------------------------------------------------------------------------

resolve_run_dir() {
  # Echo an existing run dir: explicit --run-dir, else newest under base.
  local explicit="$1"
  if [[ -n "$explicit" ]]; then
    printf '%s' "$explicit"; return 0
  fi
  local base="${REPO_ROOT}/.omc/campaign_runs"
  [[ -d "$base" ]] || return 1
  local newest
  newest="$(ls -1dt "${base}"/*/ 2>/dev/null | head -n1)"
  [[ -n "$newest" ]] || return 1
  printf '%s' "${newest%/}"
}

# ==========================================================================
# Subcommand: run
# ==========================================================================
cmd_run() {
  local tasks="" campaign="campaign" run_dir="" timeout="$DEFAULT_TIMEOUT"
  local sentinel_pidfile="" sentinel_metafile="" print_cmd=0 seeds="$DEFAULT_SEEDS"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tasks) tasks="$2"; shift 2 ;;
      --campaign) campaign="$2"; shift 2 ;;
      --run-dir) run_dir="$2"; shift 2 ;;
      --timeout) timeout="$2"; shift 2 ;;
      --seed) seeds="$2"; shift 2 ;;
      --seeds) seeds="$2"; shift 2 ;;
      --sentinel-pidfile) sentinel_pidfile="$2"; shift 2 ;;
      --sentinel-metafile) sentinel_metafile="$2"; shift 2 ;;
      --print-cmd) print_cmd=1; shift ;;
      *) die "run: unknown arg '$1'" ;;
    esac
  done
  [[ -n "$tasks" ]] || die "run: --tasks <tsv> required"
  [[ -r "$tasks" ]] || die "run: task list not readable: $tasks"
  local -a seed_values=()
  while IFS= read -r seed; do
    [[ -n "$seed" ]] && seed_values+=("$seed")
  done < <(emit_seed_list "$seeds")
  [[ "${#seed_values[@]}" -gt 0 ]] || die "run: at least one seed required"

  # --print-cmd: dry-run. Expand each TSV row to the exact command, print, exit.
  if [[ "$print_cmd" -eq 1 ]]; then
    local prefix; prefix="$(emit_cmd_prefix)"
    while IFS=$'\t' read -r task_id benchmark extra || [[ -n "$task_id" ]]; do
      [[ -z "$task_id" || "$task_id" == \#* ]] && continue
      local seed
      for seed in "${seed_values[@]}"; do
        printf '%s__seed_%s\t%s %s --seed %s --benchmark %s\n' \
          "$task_id" "$seed" "$prefix" "$extra" "$seed" "$benchmark"
      done
    done < "$tasks"
    return 0
  fi

  # Establish run dir.
  if [[ -z "$run_dir" ]]; then
    local ts; ts="$(date -u +%Y%m%dT%H%M%SZ)"
    run_dir="${REPO_ROOT}/.omc/campaign_runs/${campaign}-${ts}"
  fi
  mkdir -p "${run_dir}/logs"

  local campaign_pid="${run_dir}/campaign.pid"
  local campaign_meta="${run_dir}/campaign.meta"
  local task_pid="${run_dir}/current_task.pid"
  local task_meta="${run_dir}/current_task.meta"
  local stop_flag="${run_dir}/stop_requested"
  local queue_log="${run_dir}/queue.log"
  STATUS_FILE="${run_dir}/status.json"

  # Refuse to start into a run dir whose campaign.pid is still alive-and-ours.
  if [[ -e "$campaign_pid" ]]; then
    local v; v="$(verify_pid "$campaign_pid" "$campaign_meta")"
    if [[ "$v" == "alive-and-ours" ]]; then
      die "refusing to start: $campaign_pid verifies as a LIVE queue (double-queue guard)"
    fi
    log "overwriting stale campaign.pid (verify_pid=$v)" >> "$queue_log"
  fi
  rm -f "$stop_flag"

  ST_CAMPAIGN="$campaign"
  ST_RUN_DIR="$run_dir"
  ST_QUEUE_PID="$$"
  ST_BOOT_ID="$(current_boot_id)"
  ST_STATE="running"

  # Queue identity: anchor on this runner's BASENAME, not its absolute path.
  # The live queue cmdline is `bash <somepath>run_campaign.sh run --tasks ...`,
  # where <somepath> mirrors HOW the script was invoked (relative when launched
  # as `bash scripts/campaign/run_campaign.sh`, absolute when launched as
  # `bash /abs/run_campaign.sh`). Anchoring on the absolute $SELF_PATH only
  # matched the absolute-launch form and produced a false `reused` for the
  # relative-launch form documented in the README's deploy recipe. The basename
  # `run_campaign.sh` appears on the cmdline regardless of launch form and is
  # specific enough to reject unrelated PIDs (no other process runs a file with
  # this basename).
  write_pidfile "$campaign_pid" "$campaign_meta" "$$" "$(basename "$SELF_PATH")"

  # Trap: on any exit, clear the current-task pid file and finalize status.
  # shellcheck disable=SC2317
  _cleanup() {
    local rc=$?
    clear_pidfile "$task_pid" "$task_meta"
    if [[ "$ST_STATE" == "running" ]]; then
      if [[ -e "$stop_flag" ]]; then
        ST_STATE="stopped"
      elif [[ "$rc" -ne 0 ]]; then
        ST_STATE="error"
      else
        ST_STATE="finished"
      fi
    fi
    ST_CUR_TASK_PID=""
    write_status
    clear_pidfile "$campaign_pid" "$campaign_meta"
  }
  trap _cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT

  log "campaign=$campaign run_dir=$run_dir queue_pid=$$ boot_id=$ST_BOOT_ID" | tee -a "$queue_log"

  # Optional sentinel wait (incident-2 fix): block until the sentinel PID is
  # verifiably dead. A reused/mismatched sentinel verifies 'dead'/'reused' and we
  # proceed immediately — we never string-match, so we cannot hang on the
  # deployer shell.
  if [[ -n "$sentinel_pidfile" ]]; then
    log "sentinel wait: $sentinel_pidfile" | tee -a "$queue_log"
    while :; do
      local sv; sv="$(verify_pid "$sentinel_pidfile" "$sentinel_metafile")"
      [[ "$sv" != "alive-and-ours" ]] && break
      sleep 2
    done
    log "sentinel clear; proceeding" | tee -a "$queue_log"
  fi

  # Count tasks (for total_tasks).
  ST_TOTAL=0
  while IFS=$'\t' read -r tid _bm _extra || [[ -n "$tid" ]]; do
    [[ -z "$tid" || "$tid" == \#* ]] && continue
    ST_TOTAL=$((ST_TOTAL + ${#seed_values[@]}))
  done < "$tasks"
  write_status

  local prefix; prefix="$(emit_cmd_prefix)"
  local index=0
  while IFS=$'\t' read -r task_id benchmark extra || [[ -n "$task_id" ]]; do
    [[ -z "$task_id" || "$task_id" == \#* ]] && continue
    local seed
    for seed in "${seed_values[@]}"; do

    # Graceful stop checked BETWEEN tasks.
    if [[ -e "$stop_flag" ]]; then
      log "stop_requested observed between tasks; exiting cleanly" | tee -a "$queue_log"
      ST_STATE="stopped"
      break
    fi

    index=$((index + 1))
    local task_label safe_task_label
    task_label="${task_id}__seed_${seed}"
    safe_task_label="$(safe_label "$task_label")"
    ST_CUR_INDEX="$index"
    ST_CUR_TASK_ID="$task_label"
    ST_CUR_STARTED_EPOCH="$(date +%s)"
    local tlog="${run_dir}/logs/${safe_task_label}.log"

    # Build argv array (no eval). Real path: `python -m trace_collect.cli`;
    # test path: the TRACE_CLI_CMD override (a dummy).
    local -a argv
    # shellcheck disable=SC2206
    if [[ -n "$TRACE_CLI_CMD" ]]; then
      argv=($TRACE_CLI_CMD)
    else
      argv=("$PYTHON_BIN" -m "$TRACE_CLI_MODULE")
    fi
    # shellcheck disable=SC2206
    argv+=(
      $extra
      --seed "$seed"
      --run-id "${run_dir}/outputs/${safe_task_label}"
      --benchmark "$benchmark"
    )

    # cmdline anchor for current_task.pid: a substring GUARANTEED present on the
    # child cmdline and specific enough to reject an unrelated PID. Real path:
    # the module string (`trace_collect.cli`) — far more specific than bare
    # `python`, and present on `python -m trace_collect.cli`. Test path: the
    # dummy command's own path (argv[0]), which is unique per harness run.
    local task_anchor
    if [[ -n "$TRACE_CLI_CMD" ]]; then
      task_anchor="${argv[0]}"
    else
      task_anchor="$TRACE_CLI_MODULE"
    fi

    log "[$index/$ST_TOTAL] start task_id=$task_label benchmark=$benchmark seed=$seed" | tee -a "$queue_log"

    # Two-phase launch: background the task (log-redirected), write the PID file
    # while it is LIVE, then await it. `child` is the coreutils `timeout` PID
    # (whose cmdline contains the full task argv) or, on the bash-watchdog path,
    # the task PID directly — either way the anchor is on its cmdline.
    local child=""
    launch_task "$timeout" "$tlog" child -- "${argv[@]}"
    ST_CUR_TASK_PID="$child"
    write_pidfile "$task_pid" "$task_meta" "$child" "$task_anchor"
    write_status

    # wait + exit-code capture under `set -e` (must not abort on nonzero).
    local rc=0
    await_task "$timeout" "$child" || rc=$?

    local timed_out="false"
    if [[ "$rc" -eq 124 ]]; then
      timed_out="true"
    fi
    local dur=$(( $(date +%s) - ST_CUR_STARTED_EPOCH ))

    clear_pidfile "$task_pid" "$task_meta"
    ST_LAST_EXIT="$rc"
    ST_COMPLETED=$((ST_COMPLETED + 1))
    ST_CUR_TASK_PID=""
    append_result "$task_label" "$rc" "$timed_out" "$dur"
    write_status

    if [[ "$rc" -ne 0 ]]; then
      log "[$index/$ST_TOTAL] task_id=$task_label FAILED exit=$rc timed_out=$timed_out (continuing)" | tee -a "$queue_log"
    else
      log "[$index/$ST_TOTAL] task_id=$task_label ok dur=${dur}s" | tee -a "$queue_log"
    fi
    done
    [[ "$ST_STATE" == "stopped" ]] && break
  done < "$tasks"

  if [[ "$ST_STATE" == "running" ]]; then
    ST_STATE="finished"
  fi
  write_status
  log "queue done state=$ST_STATE completed=$ST_COMPLETED/$ST_TOTAL" | tee -a "$queue_log"
  # _cleanup (EXIT trap) clears pid files. State already final.
  trap - EXIT
  clear_pidfile "$task_pid" "$task_meta"
  clear_pidfile "$campaign_pid" "$campaign_meta"
  return 0
}

# ==========================================================================
# Subcommand: status
# ==========================================================================
cmd_status() {
  local run_dir=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir) run_dir="$2"; shift 2 ;;
      *) die "status: unknown arg '$1'" ;;
    esac
  done
  local rd; rd="$(resolve_run_dir "$run_dir")" || die "status: no run dir found (pass --run-dir)"
  [[ -d "$rd" ]] || die "status: run dir does not exist: $rd"

  local campaign_pid="${rd}/campaign.pid"
  local campaign_meta="${rd}/campaign.meta"
  local task_pid="${rd}/current_task.pid"
  local task_meta="${rd}/current_task.meta"
  local status_json="${rd}/status.json"

  local qv tv
  qv="$(verify_pid "$campaign_pid" "$campaign_meta")"
  tv="$(verify_pid "$task_pid" "$task_meta")"

  printf 'run_dir: %s\n' "$rd"
  printf 'queue: %s\n' "$qv"

  # Pull a few fields straight from status.json with grep (no jq dependency).
  local state cur_id cur_index total completed
  if [[ -r "$status_json" ]]; then
    state="$(sed -n 's/.*"state": *"\([^"]*\)".*/\1/p' "$status_json" | head -n1)"
    cur_id="$(sed -n 's/.*"current_task_id": *"\([^"]*\)".*/\1/p' "$status_json" | head -n1)"
    cur_index="$(sed -n 's/.*"current_index": *\([0-9]*\).*/\1/p' "$status_json" | head -n1)"
    total="$(sed -n 's/.*"total_tasks": *\([0-9]*\).*/\1/p' "$status_json" | head -n1)"
    completed="$(sed -n 's/.*"completed": *\([0-9]*\).*/\1/p' "$status_json" | head -n1)"
    printf 'state: %s\n' "${state:-unknown}"
    printf 'progress: %s/%s (completed %s)\n' "${cur_index:-0}" "${total:-0}" "${completed:-0}"
    printf 'current_task_id: %s\n' "${cur_id:-none}"
    printf 'current_task: %s\n' "$tv"
    # Print results array (raw) for last-N inspection.
    printf 'results: '
    sed -n 's/.*"results": *\(\[[^]]*\]\).*/\1/p' "$status_json" | head -n1
    printf '\n'
    if [[ "$state" == "running" && "$tv" != "alive-and-ours" ]]; then
      printf 'STALE: current_task.pid does not match a live process\n'
    fi
  else
    printf 'state: unknown (no status.json)\n'
    printf 'current_task: %s\n' "$tv"
  fi
  return 0
}

# ==========================================================================
# Subcommand: stop
# ==========================================================================
cmd_stop() {
  local run_dir="" now=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir) run_dir="$2"; shift 2 ;;
      --now) now=1; shift ;;
      *) die "stop: unknown arg '$1'" ;;
    esac
  done
  local rd; rd="$(resolve_run_dir "$run_dir")" || die "stop: no run dir found (pass --run-dir)"
  [[ -d "$rd" ]] || die "stop: run dir does not exist: $rd"

  local campaign_pid="${rd}/campaign.pid"
  local campaign_meta="${rd}/campaign.meta"
  local task_pid="${rd}/current_task.pid"
  local task_meta="${rd}/current_task.meta"
  local stop_flag="${rd}/stop_requested"

  if [[ "$now" -eq 0 ]]; then
    # Graceful: drop the flag; the run loop exits between tasks.
    : > "$stop_flag"
    local tv; tv="$(verify_pid "$task_pid" "$task_meta")"
    if [[ "$tv" == "alive-and-ours" ]]; then
      local pid; pid="$(head -n1 "$task_pid" | tr -dc '0-9')"
      printf 'stop requested; current task %s will finish (pid %s)\n' \
        "$(sed -n 's/.*"current_task_id": *"\([^"]*\)".*/\1/p' "${rd}/status.json" 2>/dev/null | head -n1)" "$pid"
    else
      printf 'stop requested; no verified live task (will exit before next task)\n'
    fi
    return 0
  fi

  # --now: TERM the verified current-task PID, then the verified queue PID.
  local tv; tv="$(verify_pid "$task_pid" "$task_meta")"
  if [[ "$tv" == "alive-and-ours" ]]; then
    local pid; pid="$(head -n1 "$task_pid" | tr -dc '0-9')"
    printf 'stop --now: TERM verified current-task pid %s\n' "$pid"
    kill -TERM "$pid" 2>/dev/null || true
    # escalate after grace window
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 1
      waited=$((waited + 1))
      if [[ "$waited" -ge "$KILL_AFTER" ]]; then
        printf 'stop --now: grace expired; KILL pid %s\n' "$pid"
        kill -KILL "$pid" 2>/dev/null || true
        break
      fi
    done
  elif [[ -e "$task_pid" ]]; then
    # A pid file exists but does NOT verify -> refuse. Never string-match.
    die "stop --now: current_task.pid does not verify (got '$tv'); refusing to signal"
  else
    printf 'stop --now: no current task pid file; nothing to TERM\n'
  fi

  # Now signal the verified queue PID so it stops cleanly.
  : > "$stop_flag"
  local qv; qv="$(verify_pid "$campaign_pid" "$campaign_meta")"
  if [[ "$qv" == "alive-and-ours" ]]; then
    local qpid; qpid="$(head -n1 "$campaign_pid" | tr -dc '0-9')"
    printf 'stop --now: TERM verified queue pid %s\n' "$qpid"
    kill -TERM "$qpid" 2>/dev/null || true
  else
    printf 'stop --now: queue pid does not verify (got %s); not signalling\n' "$qv"
  fi
  return 0
}

# ==========================================================================
# Dispatch
# ==========================================================================
main() {
  [[ $# -ge 1 ]] || die "usage: run_campaign.sh {run|status|stop} [args]"
  local sub="$1"; shift
  case "$sub" in
    run) cmd_run "$@" ;;
    status) cmd_status "$@" ;;
    stop) cmd_stop "$@" ;;
    *) die "unknown subcommand '$sub' (expected run|status|stop)" ;;
  esac
}

# Main-guard: only when executed directly (not sourced) do we enable strict
# mode and dispatch. Sourcing (the test harness) gets the functions with no
# shell-global side effects.
if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
  set -euo pipefail
  main "$@"
fi
