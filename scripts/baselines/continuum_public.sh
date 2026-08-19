#!/usr/bin/env bash
set -euo pipefail

readonly REPO_URL="https://github.com/Hanchenli/vllm-continuum.git"
readonly COMMIT="316a58794a6ff86b216e579b74fd56ed0c5a911f"

cache_root="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
checkout="${CONTINUUM_CHECKOUT:-$cache_root/agent-sched-bench/vllm-continuum-$COMMIT}"
venv="${CONTINUUM_VENV:-$checkout/.venv}"

usage() {
  cat <<'EOF'
Usage: continuum_public.sh fetch|verify|install|serve MODEL [VLLM_ARGS...]

Runs the official public Continuum fork pinned at commit 316a587. The public
fork implements fixed 2-second KV pinning and program-level FCFS, but not the
paper's cost-model/empirical-CDF TTL estimator. Do not label it full Continuum.

Environment:
  CONTINUUM_CHECKOUT             checkout directory
  CONTINUUM_VENV                 virtual environment directory
  CONTINUUM_PYTHON               Python used by uv (default: python3)
  CONTINUUM_TENSOR_PARALLEL_SIZE GPU count (default: 1)
  CONTINUUM_PORT                 vLLM port (default: 8000)
  RUN_OUTPUT_DIR                 official scheduler output (default: ./continuum_exp)
EOF
}

verify() {
  test -d "$checkout/.git" || { echo "missing Continuum checkout: $checkout" >&2; exit 1; }
  test "$(git -C "$checkout" rev-parse HEAD)" = "$COMMIT" || {
    echo "Continuum checkout is not pinned at $COMMIT" >&2
    exit 1
  }
  test "$(git -C "$checkout" remote get-url origin)" = "$REPO_URL" || {
    echo "Continuum checkout does not use the official remote" >&2
    exit 1
  }
  grep -Fq 'without the estimation in the paper' "$checkout/README.md"
  grep -Eq '^FIXED_THRESHOLD_CONTINUUM = 2\.0' \
    "$checkout/vllm/v1/core/estimate_with_func.py"
  grep -Fq 'Literal["fcfs", "priority", "continuum"]' \
    "$checkout/vllm/config/scheduler.py"
  grep -Fq 'job_id: Optional[str] = None' \
    "$checkout/vllm/entrypoints/openai/protocol.py"
  grep -Fq 'is_last_step: Optional[bool] = None' \
    "$checkout/vllm/entrypoints/openai/protocol.py"
  echo "verified Continuum public fixed-TTL baseline at $COMMIT"
}

fetch() {
  if test ! -e "$checkout"; then
    mkdir -p "$(dirname "$checkout")"
    git init "$checkout"
    git -C "$checkout" remote add origin "$REPO_URL"
    git -C "$checkout" fetch --depth=1 origin "$COMMIT"
    git -C "$checkout" checkout --detach "$COMMIT"
  fi
  verify
}

install() {
  fetch
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
  uv venv --python "${CONTINUUM_PYTHON:-python3}" "$venv"
  uv pip install --python "$venv/bin/python" --editable "$checkout"
  uv pip install --python "$venv/bin/python" lmcache hf_transfer
}

serve() {
  test "$#" -ge 1 || { usage >&2; exit 2; }
  verify
  test -x "$venv/bin/vllm" || {
    echo "Continuum is not installed; run '$0 install' first" >&2
    exit 1
  }
  model="$1"
  shift
  export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-./continuum_exp}"
  mkdir -p "$RUN_OUTPUT_DIR"
  exec "$venv/bin/vllm" serve "$model" \
    --scheduling-policy continuum \
    --tensor-parallel-size "${CONTINUUM_TENSOR_PARALLEL_SIZE:-1}" \
    --port "${CONTINUUM_PORT:-8000}" \
    "$@"
}

case "${1:-}" in
  fetch) fetch ;;
  verify) verify ;;
  install) install ;;
  serve) shift; serve "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
