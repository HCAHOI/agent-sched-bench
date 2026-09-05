#!/usr/bin/env bash
set -euo pipefail

readonly predictor_url="https://github.com/cachewise-project/cachewise-coding-traces.git"
readonly predictor_commit="181c435a090d328d00bbbee4c8eeb27d32f3abd2"
readonly vllm_url="https://github.com/cachewise-project/vllm.git"
readonly vllm_commit="16cc7d43d0e1a84f68f046e6caecfef21012f3fc"
readonly vllm_upstream_base="b1388b1fbf5aaef47937fabe98931211684666a6"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../.." && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
predictor_checkout="${CACHEWISE_CHECKOUT:-$cache_root/agent-sched-bench/cachewise-$predictor_commit}"
vllm_checkout="${CACHEWISE_VLLM_CHECKOUT:-$cache_root/agent-sched-bench/cachewise-vllm-reproduction-$vllm_commit}"
vllm_venv="${CACHEWISE_VLLM_VENV:-$cache_root/agent-sched-bench/venvs/cachewise-vllm-$vllm_commit}"
if test -n "${CACHEWISE_REPRO_PYTHON:-}"; then
  python_bin="$CACHEWISE_REPRO_PYTHON"
elif test -x "$predictor_checkout/.venv/bin/python"; then
  python_bin="$predictor_checkout/.venv/bin/python"
else
  python_bin="python3"
fi

usage() {
  cat <<'EOF'
Usage: cachewise_reproduction.sh fetch|verify|verify-installed|prepare|install|serve|serve-oracle-length|serve-disabled|policy|manifest [ARGS...]

Fetches the authors' predictor and vLLM fork at exact commits, then applies a
small patch implementing the paper's conditional-remaining-time eviction,
fewest-additional-block request policy, and N_rebuild=3 heap refresh.

Commands:
  fetch          Fetch clean pinned predictor and vLLM checkouts.
  verify         Verify the patch applies cleanly to the pinned vLLM commit.
  verify-installed Verify the patched serving environment is importable.
  prepare        Fetch and apply the patch; leaves an installable vLLM tree.
  install        Install the patched tree using vLLM's precompiled wheel mode.
  serve ARGS     Start the patched installed vLLM with paper policy flags.
  serve-oracle-length ARGS
                 Keep CacheWise KV eviction and use vLLM request priorities.
  serve-disabled Start the same fork/config without CacheWise scheduling flags.
  policy ARGS    Generate cachewise_policy JSON with the official predictor.
  manifest       Print published, inferred, and unpublished choices as JSON.

The paper does not publish its tool-call-to-engine attachment hook. This
reproduction adds a loopback update endpoint that must be called only after the
tool call is causally known. The official release also omits the paper's exact
80/20 split and fixed C100 training configuration.
EOF
}

fetch_one() {
  local url="$1"
  local commit="$2"
  local checkout="$3"
  if test ! -e "$checkout"; then
    mkdir -p "$(dirname "$checkout")"
    git init -q "$checkout"
    git -C "$checkout" remote add origin "$url"
    git -C "$checkout" fetch -q --depth=1 origin "$commit"
    git -C "$checkout" checkout -q --detach "$commit"
  fi
  test "$(git -C "$checkout" rev-parse HEAD)" = "$commit"
  test "$(git -C "$checkout" remote get-url origin)" = "$url"
}

fetch_all() {
  fetch_one "$predictor_url" "$predictor_commit" "$predictor_checkout"
  fetch_one "$vllm_url" "$vllm_commit" "$vllm_checkout"
  echo "fetched pinned CacheWise predictor and vLLM fork"
}

verify_patch() {
  "$python_bin" "$script_dir/cachewise_reproduction.py" \
    verify-predictor "$predictor_checkout"
  if test -n "$(git -C "$vllm_checkout" status --porcelain)"; then
    "$python_bin" "$script_dir/cachewise_reproduction.py" \
      verify-applied "$vllm_checkout"
  else
    "$python_bin" "$script_dir/cachewise_reproduction.py" \
      verify-patch "$vllm_checkout"
  fi
}

prepare() {
  fetch_all
  if test -n "$(git -C "$vllm_checkout" status --porcelain)"; then
    "$python_bin" "$script_dir/cachewise_reproduction.py" \
      verify-applied "$vllm_checkout"
  else
    "$python_bin" "$script_dir/cachewise_reproduction.py" \
      apply-patch "$vllm_checkout"
  fi
  echo "prepared patched vLLM checkout: $vllm_checkout"
}

install() {
  prepare
  if test ! -x "$vllm_venv/bin/python"; then
    uv venv --python "${CACHEWISE_VLLM_PYTHON_VERSION:-python3}" "$vllm_venv"
  fi
  VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT="$vllm_upstream_base" \
    uv pip install --python "$vllm_venv/bin/python" "$vllm_checkout"
  verify_patch
}

verify_installed() {
  verify_patch
  test -x "$vllm_venv/bin/python" || {
    echo "patched vLLM is not installed: $vllm_venv/bin/python" >&2
    return 1
  }
  "$vllm_venv/bin/python" -c \
    'import vllm; import vllm.v1.core.cachewise_policy'
}

scheduler_path() {
  local vllm_python=${1:?vLLM Python is required}
  "$vllm_python" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vLLM is not installed")
print(Path(spec.origin).parent / "v1/core/sched/scheduler.py")
PY
}

patch_oracle_length_scheduler() {
  local vllm_python=${1:?vLLM Python is required} scheduler
  scheduler=$(scheduler_path "$vllm_python")
  PYTHONPATH="$repo" "$vllm_python" - "$scheduler" <<'PY'
from pathlib import Path
import sys

from scripts.baselines.cachewise_reproduction import (
    patch_cachewise_oracle_length_scheduler,
)

print(
    "patched"
    if patch_cachewise_oracle_length_scheduler(Path(sys.argv[1]))
    else "already patched"
)
PY
}

serve() {
  local waiting_policy=${1:?waiting policy is required}
  shift
  local model=${1:?model is required}
  shift
  local vllm_python="${CACHEWISE_VLLM_PYTHON:-$vllm_venv/bin/python}"
  test -x "$vllm_python" || {
    echo "patched vLLM is not installed: $vllm_python" >&2
    exit 1
  }
  "$python_bin" "$script_dir/cachewise_reproduction.py" \
    verify-applied "$vllm_checkout"
  export VLLM_SERVER_DEV_MODE=1
  local waiting_args=()
  local oracle_event_log=
  if [[ "$waiting_policy" == cachewise ]]; then
    unset CACHEWISE_ORACLE_LENGTH_EVENT_LOG
    unset CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN
    unset CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN
    waiting_args+=(--prioritize-waiting-by-prefix-cache)
  elif [[ "$waiting_policy" == oracle-length ]]; then
    oracle_event_log=${CACHEWISE_ORACLE_LENGTH_EVENT_LOG:?CacheWise Oracle event log is required}
    : "${CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN:?CacheWise Oracle prefill coefficient is required}"
    : "${CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN:?CacheWise Oracle decode coefficient is required}"
    test ! -e "$oracle_event_log" || {
      echo "refusing to overwrite $oracle_event_log" >&2
      exit 1
    }
    mkdir -p "$(dirname "$oracle_event_log")"
    : >"$oracle_event_log"
    patch_oracle_length_scheduler "$vllm_python"
    waiting_args+=(--prioritize-waiting-by-prefix-cache)
  else
    echo "unknown waiting policy: $waiting_policy" >&2
    exit 2
  fi
  local launch_env=(env PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}")
  [[ -z "$oracle_event_log" ]] || \
    launch_env+=(CACHEWISE_ORACLE_LENGTH_EVENT_LOG="$oracle_event_log")
  exec "${launch_env[@]}" \
    "$vllm_python" -m vllm.entrypoints.openai.api_server \
    --model "$model" "$@" \
    --enable-prefix-caching \
    --enable-cachewise-free-heap \
    "${waiting_args[@]}" \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512
}

serve_disabled() {
  local model=${1:?model is required}
  shift
  local vllm_python="${CACHEWISE_VLLM_PYTHON:-$vllm_venv/bin/python}"
  test -x "$vllm_python" || {
    echo "patched vLLM is not installed: $vllm_python" >&2
    exit 1
  }
  "$python_bin" "$script_dir/cachewise_reproduction.py" \
    verify-applied "$vllm_checkout"
  export VLLM_SERVER_DEV_MODE=1
  exec "$vllm_python" -m vllm.entrypoints.openai.api_server \
    --model "$model" "$@" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512
}

case "${1:-}" in
  fetch) fetch_all ;;
  verify) fetch_all; verify_patch ;;
  verify-installed) fetch_all; verify_installed ;;
  prepare) prepare ;;
  install) install ;;
  serve) shift; serve cachewise "$@" ;;
  serve-oracle-length) shift; serve oracle-length "$@" ;;
  serve-disabled) shift; serve_disabled "$@" ;;
  policy)
    shift
    fetch_one "$predictor_url" "$predictor_commit" "$predictor_checkout"
    exec "$python_bin" "$script_dir/cachewise_reproduction.py" policy \
      --predictor-checkout "$predictor_checkout" "$@"
    ;;
  manifest)
    exec "$python_bin" "$script_dir/cachewise_reproduction.py" manifest
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
