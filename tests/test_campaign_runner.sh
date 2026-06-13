#!/usr/bin/env bash
# Local, GPU-free test harness for scripts/campaign/run_campaign.sh.
#
# Uses dummy tasks (`sleep`/`false`) via the TRACE_CLI_CMD override so NO GPU,
# torch, or trace_collect import is needed. Runs on the Mac with plain bash.
# Prints one PASS/FAIL line per checked behavior; exits 0 iff all pass.
#
#   bash tests/test_campaign_runner.sh
#
set -uo pipefail   # NOT -e: we assert on command failures deliberately.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/campaign/run_campaign.sh"

PASS=0
FAIL=0
pass() { printf 'PASS: %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf 'FAIL: %s\n' "$1"; FAIL=$((FAIL + 1)); }
check() { if [[ "$1" == "$2" ]]; then pass "$3"; else fail "$3 (expected '$2' got '$1')"; fi; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/campaign_test.XXXXXX")"

# Dummy task command. The runner ALWAYS appends `--benchmark <bm>` to the task
# argv (its real contract for trace_collect.cli), so a bare `sleep` would choke
# on the trailing flag. This wrapper sleeps for its FIRST arg and ignores the
# rest, modelling a real task that accepts --benchmark. Real --benchmark append
# is separately round-trip-verified in the --print-cmd step below.
SLEEPCMD="${WORK}/dummy_sleep.sh"
cat > "$SLEEPCMD" <<'EOS'
#!/usr/bin/env bash
# dummy_sleep.sh <secs> [ignored...]  -> sleep <secs>, ignore trailing flags.
# Does NOT exec, so this process keeps a cmdline containing its own path (the
# anchor verify_pid checks). On TERM/KILL it tears down the sleep child too so
# the test leaves no orphan sleepers.
secs="${1:-0}"
sleep "$secs" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; exit 143' TERM INT
wait "$child"
EOS
chmod +x "$SLEEPCMD"

cleanup_all() {
  # kill any leftover sleepers spawned by the tests
  [[ -n "${SLEEP_PID:-}" ]] && kill -KILL "$SLEEP_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup_all EXIT

# Source the runner to unit-test its functions directly. The runner has a
# main-guard ( if [[ "${BASH_SOURCE[0]}" == "${0}" ]] ) so sourcing it does NOT
# dispatch a subcommand AND does NOT apply `set -euo pipefail` to this shell —
# strict mode is execution-only. We can therefore source it straight, no
# stripping needed.
# shellcheck disable=SC1090
source "$RUNNER"
# Belt-and-suspenders: this harness deliberately asserts on nonzero exits, so
# keep -e off regardless of what the sourced file might do in future.
set +e +o pipefail

echo "=== Step 1: verify_pid core ==="

# alive-and-ours: spawn a sleeper, write its pid/meta from itself, verify.
sleep 30 &
SLEEP_PID=$!
PF="${WORK}/s.pid"; MF="${WORK}/s.meta"
ANCHOR_CMD="$(proc_cmdline "$SLEEP_PID")"
write_pidfile "$PF" "$MF" "$SLEEP_PID" "sleep"
check "$(verify_pid "$PF" "$MF")" "alive-and-ours" "verify_pid alive-and-ours for live sleeper"

# dead: kill it, verify dead (stale file).
kill -KILL "$SLEEP_PID" 2>/dev/null; wait "$SLEEP_PID" 2>/dev/null
check "$(verify_pid "$PF" "$MF")" "dead" "verify_pid dead for killed pid"
SLEEP_PID=""

# missing pid file -> dead.
check "$(verify_pid "${WORK}/nope.pid" "${WORK}/nope.meta")" "dead" "verify_pid dead for missing file"

# reused: live pid, but meta records a WRONG start-time -> dead (do not signal).
sleep 30 &
SLEEP_PID=$!
PF2="${WORK}/s2.pid"; MF2="${WORK}/s2.meta"
write_pidfile "$PF2" "$MF2" "$SLEEP_PID" "sleep"
# corrupt the recorded start_time
sed -i.bak 's/^start_time=.*/start_time=999999999999/' "$MF2"; rm -f "${MF2}.bak"
check "$(verify_pid "$PF2" "$MF2")" "dead" "verify_pid dead for wrong start-time (reused)"

# reused: live pid, correct start-time, but cmdline anchor that does NOT match.
PF3="${WORK}/s3.pid"; MF3="${WORK}/s3.meta"
write_pidfile "$PF3" "$MF3" "$SLEEP_PID" "this-anchor-cannot-match-a-sleep-cmdline-zzz"
check "$(verify_pid "$PF3" "$MF3")" "reused" "verify_pid reused for non-matching cmdline anchor"
kill -KILL "$SLEEP_PID" 2>/dev/null; wait "$SLEEP_PID" 2>/dev/null; SLEEP_PID=""

# No pgrep/pkill anywhere in the runner.
if grep -nE 'pgrep|pkill' "$RUNNER" >/dev/null 2>&1; then
  fail "no pgrep/pkill in run_campaign.sh"
else
  pass "no pgrep/pkill in run_campaign.sh"
fi

echo "=== Step 2: run loop (isolation, timeout, logs, status) ==="

# 3-row TSV: sleep(ok) / false(fail) / sleep(ok). Use TRACE_CLI_CMD override.
TSV="${WORK}/tasks3.tsv"
{
  printf '# comment line skipped\n'
  printf '\n'
  printf 't_ok1\tswe-rebench\t1\n'      # extra_cli_args="1" -> `sleep 1`
  printf 't_fail\tswe-rebench\t\n'      # extra empty -> `false` ignores args
  printf 't_ok2\tswe-rebench\t1\n'
} > "$TSV"

RD1="${WORK}/run1"
# task cmd = sleep for ok rows... but false row needs a different cmd. We model
# this by making the override `sh -c 'case ...'`? Simpler: use a per-row trick
# is not supported (one TRACE_CLI_CMD). Instead run two separate runs:
#  (a) all-sleep run to check 3 logs + finished + completed:3
#  (b) a run whose middle task fails to check results[] nonzero + continue.

# (a) all sleep
TRACE_CLI_CMD="$SLEEPCMD" bash "$RUNNER" run --tasks "$TSV" --campaign t1 --run-dir "$RD1" --timeout 30 >/dev/null 2>&1
NLOGS=$(find "${RD1}/logs" -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
check "$NLOGS" "3" "run produces 3 per-task logs"
STATE_A="$(sed -n 's/.*"state": *"\([^"]*\)".*/\1/p' "${RD1}/status.json" | head -n1)"
check "$STATE_A" "finished" "run ends state=finished"
COMP_A="$(sed -n 's/.*"completed": *\([0-9]*\).*/\1/p' "${RD1}/status.json" | head -n1)"
check "$COMP_A" "3" "run completed=3"

# (b) middle task fails. Build a TSV where the failing cmd is `false` for the
# middle row only. Use a wrapper cmd that fails iff its arg is 'FAIL'.
FAILCMD="${WORK}/maybefail.sh"
cat > "$FAILCMD" <<'EOS'
#!/usr/bin/env bash
# usage: maybefail.sh <arg>; exits 1 if arg==FAIL else sleeps 1 and exits 0.
if [[ "${1:-}" == "FAIL" ]]; then exit 7; fi
sleep 1; exit 0
EOS
chmod +x "$FAILCMD"
TSV2="${WORK}/tasks_fail.tsv"
{
  printf 't_a\tswe-rebench\tOK\n'
  printf 't_b\tswe-rebench\tFAIL\n'
  printf 't_c\tswe-rebench\tOK\n'
} > "$TSV2"
RD2="${WORK}/run2"
TRACE_CLI_CMD="$FAILCMD" bash "$RUNNER" run --tasks "$TSV2" --campaign t2 --run-dir "$RD2" --timeout 30 >/dev/null 2>&1
STATE_B="$(sed -n 's/.*"state": *"\([^"]*\)".*/\1/p' "${RD2}/status.json" | head -n1)"
check "$STATE_B" "finished" "run continues past failing task (state=finished)"
COMP_B="$(sed -n 's/.*"completed": *\([0-9]*\).*/\1/p' "${RD2}/status.json" | head -n1)"
check "$COMP_B" "3" "run completed=3 despite middle failure"
# results[] for t_b must carry exit_code 7.
if grep -q '"task_id":"t_b","exit_code":7' "${RD2}/status.json"; then
  pass "results[] records failing task nonzero exit (7)"
else
  fail "results[] records failing task nonzero exit (7)"
fi

# (c) timeout -> exit_code 124, timed_out:true, queue advances.
TSV3="${WORK}/tasks_to.tsv"
{
  printf 't_long\tswe-rebench\t10\n'   # sleep 10, but --timeout 2 below
  printf 't_after\tswe-rebench\t1\n'
} > "$TSV3"
RD3="${WORK}/run3"
TRACE_CLI_CMD="$SLEEPCMD" bash "$RUNNER" run --tasks "$TSV3" --campaign t3 --run-dir "$RD3" --timeout 2 >/dev/null 2>&1
if grep -q '"task_id":"t_long","exit_code":124,"timed_out":true' "${RD3}/status.json"; then
  pass "timeout yields exit_code 124 timed_out:true"
else
  fail "timeout yields exit_code 124 timed_out:true"
  echo "  --- status.json ---"; cat "${RD3}/status.json" 2>/dev/null | sed 's/^/  /'
fi
COMP_C="$(sed -n 's/.*"completed": *\([0-9]*\).*/\1/p' "${RD3}/status.json" | head -n1)"
check "$COMP_C" "2" "queue advances past timed-out task (completed=2)"

echo "=== Step 3: status and stop ==="

# Live run in background: 3 tasks of sleep 3 each (~9s). Then poke status/stop.
TSV_LIVE="${WORK}/tasks_live.tsv"
{
  printf 'L1\tswe-rebench\t3\n'
  printf 'L2\tswe-rebench\t3\n'
  printf 'L3\tswe-rebench\t3\n'
} > "$TSV_LIVE"
RD_LIVE="${WORK}/run_live"
TRACE_CLI_CMD="$SLEEPCMD" bash "$RUNNER" run --tasks "$TSV_LIVE" --campaign live --run-dir "$RD_LIVE" --timeout 30 >/dev/null 2>&1 &
LIVE_QUEUE=$!
# wait for the first current_task.pid to appear
for _ in $(seq 1 50); do
  [[ -s "${RD_LIVE}/current_task.pid" ]] && break
  sleep 0.2
done
ST_OUT="$(bash "$RUNNER" status --run-dir "$RD_LIVE" 2>&1)"
if printf '%s' "$ST_OUT" | grep -q 'current_task: alive-and-ours'; then
  pass "status reports live current_task alive-and-ours"
else
  fail "status reports live current_task alive-and-ours"
  echo "$ST_OUT" | sed 's/^/  /'
fi
if printf '%s' "$ST_OUT" | grep -qE 'progress: [1-9]/3'; then
  pass "status reports progress n/3"
else
  fail "status reports progress n/3"
fi

# STALE detection: corrupt current_task.pid to a bogus pid while state=running.
CP_BAK="$(cat "${RD_LIVE}/current_task.pid" 2>/dev/null)"
echo "999999" > "${RD_LIVE}/current_task.pid"
ST_STALE="$(bash "$RUNNER" status --run-dir "$RD_LIVE" 2>&1)"
if printf '%s' "$ST_STALE" | grep -q 'STALE: current_task.pid does not match a live process'; then
  pass "status flags corrupted current_task.pid as STALE"
else
  fail "status flags corrupted current_task.pid as STALE"
  echo "$ST_STALE" | sed 's/^/  /'
fi
# restore so the runner can keep cleaning up its real child
[[ -n "$CP_BAK" ]] && echo "$CP_BAK" > "${RD_LIVE}/current_task.pid"

# graceful stop: request, expect queue exits with state=stopped, current finishes.
bash "$RUNNER" stop --run-dir "$RD_LIVE" >/dev/null 2>&1
wait "$LIVE_QUEUE" 2>/dev/null
STATE_STOP="$(sed -n 's/.*"state": *"\([^"]*\)".*/\1/p' "${RD_LIVE}/status.json" | head -n1)"
check "$STATE_STOP" "stopped" "graceful stop -> state=stopped"
COMP_STOP="$(sed -n 's/.*"completed": *\([0-9]*\).*/\1/p' "${RD_LIVE}/status.json" | head -n1)"
if [[ "${COMP_STOP:-0}" -ge 1 ]]; then
  pass "graceful stop let in-flight task finish (completed>=1)"
else
  fail "graceful stop let in-flight task finish (completed>=1, got ${COMP_STOP:-0})"
fi

# stop --now: TERM the verified current-task pid within the grace window.
TSV_NOW="${WORK}/tasks_now.tsv"
{
  printf 'N1\tswe-rebench\t30\n'
  printf 'N2\tswe-rebench\t1\n'
} > "$TSV_NOW"
RD_NOW="${WORK}/run_now"
TRACE_CLI_CMD="$SLEEPCMD" bash "$RUNNER" run --tasks "$TSV_NOW" --campaign now --run-dir "$RD_NOW" --timeout 60 >/dev/null 2>&1 &
NOW_QUEUE=$!
for _ in $(seq 1 50); do
  [[ -s "${RD_NOW}/current_task.pid" ]] && break
  sleep 0.2
done
TASK_PID_NOW="$(head -n1 "${RD_NOW}/current_task.pid" 2>/dev/null | tr -dc '0-9')"
NOW_OUT="$(bash "$RUNNER" stop --run-dir "$RD_NOW" --now 2>&1)"
START_NOW=$(date +%s)
wait "$NOW_QUEUE" 2>/dev/null
END_NOW=$(date +%s)
if [[ -n "$TASK_PID_NOW" ]] && ! kill -0 "$TASK_PID_NOW" 2>/dev/null; then
  pass "stop --now terminated the verified live task pid"
else
  fail "stop --now terminated the verified live task pid"
  echo "$NOW_OUT" | sed 's/^/  /'
fi
if [[ $((END_NOW - START_NOW)) -lt 30 ]]; then
  pass "stop --now returned within grace window (<30s, not the 30s sleep)"
else
  fail "stop --now returned within grace window"
fi

# stop --now refuses on a pid file pointing at an UNRELATED live process.
sleep 30 &
SLEEP_PID=$!
RD_REF="${WORK}/run_refuse"
mkdir -p "${RD_REF}/logs"
# Hand-build a current_task.pid for the unrelated sleeper but with a meta whose
# cmdline anchor does NOT match (so verify_pid -> reused, NOT alive-and-ours).
echo "$SLEEP_PID" > "${RD_REF}/current_task.pid"
write_pidfile "${RD_REF}/current_task.pid" "${RD_REF}/current_task.meta" "$SLEEP_PID" "ANCHOR-THAT-DOES-NOT-MATCH-zzz"
# also need a campaign.pid present (dead) so resolve works
echo "$$" > "${RD_REF}/campaign.pid"
write_pidfile "${RD_REF}/campaign.pid" "${RD_REF}/campaign.meta" "$SLEEP_PID" "ANCHOR-THAT-DOES-NOT-MATCH-zzz"
REFUSE_OUT="$(bash "$RUNNER" stop --run-dir "$RD_REF" --now 2>&1)" && REFUSE_RC=0 || REFUSE_RC=$?
if [[ "$REFUSE_RC" -ne 0 ]] && printf '%s' "$REFUSE_OUT" | grep -q 'refusing to signal'; then
  pass "stop --now refuses (nonzero) on unverifiable current_task.pid"
else
  fail "stop --now refuses on unverifiable current_task.pid (rc=$REFUSE_RC)"
  echo "$REFUSE_OUT" | sed 's/^/  /'
fi
# the unrelated sleeper must STILL be alive (was never signalled).
if kill -0 "$SLEEP_PID" 2>/dev/null; then
  pass "stop --now did NOT signal the unrelated live process"
else
  fail "stop --now did NOT signal the unrelated live process"
fi
kill -KILL "$SLEEP_PID" 2>/dev/null; wait "$SLEEP_PID" 2>/dev/null; SLEEP_PID=""

echo "=== Step 4: sentinel wait ==="

# Sentinel points at a live `sleep 4` with MATCHING meta -> run blocks until it
# exits, then proceeds.
sleep 4 &
SENT_PID=$!
SENT_PF="${WORK}/sent.pid"; SENT_MF="${WORK}/sent.meta"
write_pidfile "$SENT_PF" "$SENT_MF" "$SENT_PID" "sleep"
TSV_SENT="${WORK}/tasks_sent.tsv"
printf 'S1\tswe-rebench\t1\n' > "$TSV_SENT"
RD_SENT="${WORK}/run_sent"
SENT_START=$(date +%s)
TRACE_CLI_CMD="$SLEEPCMD" bash "$RUNNER" run --tasks "$TSV_SENT" --campaign sent --run-dir "$RD_SENT" \
  --timeout 30 --sentinel-pidfile "$SENT_PF" --sentinel-metafile "$SENT_MF" >/dev/null 2>&1
SENT_END=$(date +%s)
kill -KILL "$SENT_PID" 2>/dev/null; wait "$SENT_PID" 2>/dev/null
# run should have blocked >= ~3s for the sentinel sleep 4 before doing its 1s task.
if [[ $((SENT_END - SENT_START)) -ge 3 ]]; then
  pass "sentinel wait blocked until live sentinel exited"
else
  fail "sentinel wait blocked until live sentinel exited (waited $((SENT_END - SENT_START))s)"
fi
STATE_SENT="$(sed -n 's/.*"state": *"\([^"]*\)".*/\1/p' "${RD_SENT}/status.json" | head -n1)"
check "$STATE_SENT" "finished" "run proceeds and finishes after sentinel clears"

# Sentinel pidfile whose meta MISMATCHES the live pid -> treated as dead ->
# proceeds immediately (does NOT hang on a reused pid).
sleep 30 &
SENT_PID2=$!
SENT_PF2="${WORK}/sent2.pid"; SENT_MF2="${WORK}/sent2.meta"
write_pidfile "$SENT_PF2" "$SENT_MF2" "$SENT_PID2" "ANCHOR-MISMATCH-zzz"
TSV_SENT2="${WORK}/tasks_sent2.tsv"
printf 'S2\tswe-rebench\t1\n' > "$TSV_SENT2"
RD_SENT2="${WORK}/run_sent2"
S2_START=$(date +%s)
TRACE_CLI_CMD="$SLEEPCMD" bash "$RUNNER" run --tasks "$TSV_SENT2" --campaign sent2 --run-dir "$RD_SENT2" \
  --timeout 30 --sentinel-pidfile "$SENT_PF2" --sentinel-metafile "$SENT_MF2" >/dev/null 2>&1
S2_END=$(date +%s)
kill -KILL "$SENT_PID2" 2>/dev/null; wait "$SENT_PID2" 2>/dev/null; SENT_PID2=""
# Should NOT have waited 30s; only its own 1s task (allow generous <15s bound).
if [[ $((S2_END - S2_START)) -lt 15 ]]; then
  pass "sentinel with mismatched meta proceeds immediately (no hang on reused pid)"
else
  fail "sentinel mismatched meta did not hang? waited $((S2_END - S2_START))s"
fi

echo "=== Step 5: --print-cmd round-trip ==="

PC_OUT="$(bash "$RUNNER" run --tasks "${REPO_ROOT}/scripts/campaign/tasks.example.tsv" --print-cmd 2>&1)"
if printf '%s' "$PC_OUT" | grep -q 'python -m trace_collect.cli'; then
  pass "--print-cmd emits python -m trace_collect.cli prefix"
else
  fail "--print-cmd emits python -m trace_collect.cli prefix"
fi
if printf '%s' "$PC_OUT" | grep -q -- '--benchmark swe-rebench'; then
  pass "--print-cmd appends --benchmark"
else
  fail "--print-cmd appends --benchmark"
fi
# Round-trip equivalence vs manifest Arm A: the emitted Arm-A command must
# contain every token of the manifest's invocation (order-independent for
# argparse). Spot-check the load-bearing flags.
ARMA_LINE="$(printf '%s' "$PC_OUT" | grep '^A_max__beeware')"
ROUNDTRIP_OK=1
for tok in \
  "--provider" "--model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" "--scaffold openclaw" \
  "--mcp-config configs/mcp/context7.yaml" "--record-internals" \
  "--sparse-attn-config configs/sparse_attention/block_topk_b1024_perhead_max.yaml" \
  "--per-head-stats-layers 0,4,9,13,16,18,20,22,25,27,32,38,43,47" \
  "--record-per-head-topk" "--per-head-topk-rank 64" \
  "--instance-ids beeware__briefcase-817" "--benchmark swe-rebench"; do
  if ! printf '%s' "$ARMA_LINE" | grep -q -- "$tok"; then
    ROUNDTRIP_OK=0
    echo "  missing token: $tok"
  fi
done
if [[ "$ROUNDTRIP_OK" -eq 1 ]]; then
  pass "--print-cmd Arm-A round-trip reproduces manifest invocation tokens"
else
  fail "--print-cmd Arm-A round-trip reproduces manifest invocation tokens"
fi

echo
echo "================ SUMMARY ================"
printf 'PASS=%d FAIL=%d\n' "$PASS" "$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "ALL GREEN"
  exit 0
else
  echo "SOME FAILED"
  exit 1
fi
