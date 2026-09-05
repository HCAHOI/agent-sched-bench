#!/usr/bin/env bash
set -euo pipefail

readonly VLLM_VERSION=0.11.2
readonly PATCHED_SCHEDULER_SHA=52a5d8acf0f5ffeff130d1e27cc24ed79537a1bff7469670c5d550e847ccaef4
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
cache_root=${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}
venv=${NATIVE_PRIORITY_AGING_VENV:-$cache_root/agent-sched-bench/venvs/native-priority-aging-$VLLM_VERSION}
lock_receipt="$venv/.agent-sched-bench-uv-lock.sha256"

usage() {
  cat <<'EOF'
Usage: native_priority_aging.sh install|verify
       native_priority_aging.sh serve MODEL [vLLM arguments...]

Installs isolated stock vLLM 0.11.2 and adds one-batch admission aging.
Priority-1 requests are promoted after the configured number of priority-0
waiting-to-running admissions. This is a simple baseline, not a paper system.
EOF
}

scheduler_path() {
  "$venv/bin/python" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vllm is not installed")
print(Path(spec.origin).parent / "v1/core/sched/scheduler.py")
PY
}

patch_scheduler() {
  local scheduler
  scheduler=$(scheduler_path)
  PYTHONPATH="$repo" "$venv/bin/python" - "$scheduler" <<'PY'
from pathlib import Path
import sys

from scripts.baselines.native_priority_aging import patch_scheduler

print("patched" if patch_scheduler(Path(sys.argv[1])) else "already patched")
PY
}

verify() {
  test -x "$venv/bin/python" || { echo "missing aging venv: $venv" >&2; exit 1; }
  test -f "$lock_receipt" || { echo "missing uv.lock receipt: $lock_receipt" >&2; exit 1; }
  sha256sum -c "$lock_receipt" >/dev/null
  test "$("$venv/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')" = "$VLLM_VERSION"
  local scheduler actual
  scheduler=$(scheduler_path)
  actual=$(sha256sum "$scheduler" | awk '{print $1}')
  test "$actual" = "$PATCHED_SCHEDULER_SHA" || {
    echo "unexpected patched scheduler.py sha256: $actual" >&2
    exit 1
  }
  grep -Fq '# NATIVE_PRIORITY_ONE_BATCH_AGING_V1' "$scheduler"
}

install_baseline() {
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
  UV_PROJECT_ENVIRONMENT="$venv" uv sync --project "$repo" --frozen --extra serving-spike
  sha256sum "$repo/uv.lock" >"$lock_receipt"
  patch_scheduler
  verify
}

serve() {
  verify
  test "$#" -ge 1 || { usage >&2; exit 2; }
  local event_log=${NATIVE_PRIORITY_AGING_EVENT_LOG:?NATIVE_PRIORITY_AGING_EVENT_LOG is required}
  local limit=${NATIVE_PRIORITY_AGING_BYPASS_LIMIT:?NATIVE_PRIORITY_AGING_BYPASS_LIMIT is required}
  [[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "invalid aging bypass limit" >&2; exit 2; }
  test ! -e "$event_log" || { echo "refusing to overwrite $event_log" >&2; exit 1; }
  mkdir -p "$(dirname "$event_log")"
  : >"$event_log"
  exec env PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" \
    NATIVE_PRIORITY_AGING_EVENT_LOG="$event_log" \
    NATIVE_PRIORITY_AGING_BYPASS_LIMIT="$limit" \
    "$venv/bin/vllm" serve "$@" --scheduling-policy priority
}

case ${1:-} in
  install) install_baseline ;;
  verify) verify ;;
  serve) shift; serve "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
