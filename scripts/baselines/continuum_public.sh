#!/usr/bin/env bash
set -euo pipefail

readonly REPO_URL="https://github.com/Hanchenli/vllm-continuum.git"
readonly COMMIT="316a58794a6ff86b216e579b74fd56ed0c5a911f"
readonly UPSTREAM_URL="https://github.com/vllm-project/vllm.git"
readonly UPSTREAM_COMMIT="01efc7ef781391e744ed08c3292817a773d654e6"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
checkout="${CONTINUUM_CHECKOUT:-$cache_root/agent-sched-bench/vllm-continuum-public-$COMMIT}"
upstream_checkout="${CONTINUUM_UPSTREAM_CHECKOUT:-$cache_root/agent-sched-bench/vllm-upstream-$UPSTREAM_COMMIT}"
venv="${CONTINUUM_VENV:-$cache_root/agent-sched-bench/venvs/continuum-public-$COMMIT}"

OVERLAY_FILES=(
  config/scheduler.py
  entrypoints/api_server.py
  entrypoints/cli/__init__.py
  entrypoints/openai/protocol.py
  entrypoints/openai/serving_chat.py
  entrypoints/openai/serving_completion.py
  entrypoints/openai/serving_pooling.py
  entrypoints/openai/serving_responses.py
  entrypoints/openai/serving_score.py
  entrypoints/openai/speech_to_text.py
  v1/core/estimate_with_func.py
  v1/core/kv_cache_manager.py
  v1/core/sched/request_queue.py
  v1/core/sched/scheduler.py
  v1/request.py
)
IGNORED_DELETED_PYTHON=(
  benchmarks/lib/__init__.py
  benchmarks/lib/endpoint_request_func.py
  benchmarks/lib/ready_checker.py
  benchmarks/lib/utils.py
)

usage() {
  cat <<'EOF'
Usage: continuum_public.sh fetch|verify|install|serve MODEL [VLLM_ARGS...]

Runs the official public Continuum fork pinned at commit 316a587. The public
fork implements fixed 2-second KV pinning and program-level FCFS, but not the
paper's cost-model/empirical-CDF TTL estimator. Do not label it full Continuum.

The install is the official vLLM 0.10.2 wheel plus the exact serving-relevant
Python diff from the pinned public fork; it never compiles an editable checkout.

Environment:
  CONTINUUM_CHECKOUT             clean public-fork checkout directory
  CONTINUUM_UPSTREAM_CHECKOUT    clean upstream v0.10.2 checkout directory
  CONTINUUM_VENV                 isolated public-baseline virtual environment
  CONTINUUM_PYTHON               Python used by uv (default: python3)
  CONTINUUM_TENSOR_PARALLEL_SIZE GPU count (default: 1)
  CONTINUUM_PORT                 vLLM port (default: 8000)
  RUN_OUTPUT_DIR                 official scheduler output (default: ./continuum_exp)
EOF
}

continuum_verify_clean_checkout() {
  local path="$1" commit="$2" remote="$3"
  test -d "$path/.git" || { echo "missing checkout: $path" >&2; return 1; }
  test "$(git -C "$path" rev-parse HEAD)" = "$commit" || {
    echo "$path is not pinned at $commit" >&2
    return 1
  }
  test "$(git -C "$path" remote get-url origin)" = "$remote" || {
    echo "$path does not use $remote" >&2
    return 1
  }
  test -z "$(git -C "$path" status --porcelain=v1 --untracked-files=all)" || {
    echo "$path has staged, unstaged, or untracked changes" >&2
    return 1
  }
}

continuum_fetch_checkout() {
  local path="$1" commit="$2" remote="$3"
  if test ! -e "$path"; then
    mkdir -p "$(dirname "$path")"
    git init -q "$path"
    git -C "$path" remote add origin "$remote"
    git -C "$path" fetch -q --depth=1 origin "$commit"
    git -C "$path" checkout -q --detach "$commit"
  fi
  continuum_verify_clean_checkout "$path" "$commit" "$remote"
}

continuum_verify_fork_delta() {
  local base="$1" fork="$2" overlay deleted
  overlay="$(printf '%s\n' "${OVERLAY_FILES[@]}")"
  deleted="$(printf '%s\n' "${IGNORED_DELETED_PYTHON[@]}")"
  python3 - "$base/vllm" "$fork/vllm" "$overlay" "$deleted" <<'PY'
import sys
from pathlib import Path

base, fork = map(Path, sys.argv[1:3])
expected_overlay = set(sys.argv[3].splitlines())
expected_deleted = set(sys.argv[4].splitlines())
base_files = {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*.py")}
fork_files = {p.relative_to(fork).as_posix(): p.read_bytes() for p in fork.rglob("*.py")}
overlay = {
    name for name, content in fork_files.items()
    if name not in base_files or base_files[name] != content
}
deleted = set(base_files) - set(fork_files)
if overlay != expected_overlay or deleted != expected_deleted:
    raise SystemExit(
        f"unexpected official-fork Python delta: overlay={sorted(overlay)}, "
        f"deleted={sorted(deleted)}"
    )
PY
}

continuum_fetch_public_sources() {
  continuum_fetch_checkout "$checkout" "$COMMIT" "$REPO_URL"
  continuum_fetch_checkout "$upstream_checkout" "$UPSTREAM_COMMIT" "$UPSTREAM_URL"
  continuum_verify_fork_delta "$upstream_checkout" "$checkout"
}

continuum_verify_public_source() {
  continuum_verify_clean_checkout "$checkout" "$COMMIT" "$REPO_URL"
  continuum_verify_clean_checkout "$upstream_checkout" "$UPSTREAM_COMMIT" "$UPSTREAM_URL"
  continuum_verify_fork_delta "$upstream_checkout" "$checkout"
  grep -Fq 'without the estimation in the paper' "$checkout/README.md"
  grep -Eq '^FIXED_THRESHOLD_CONTINUUM = 2\.0' \
    "$checkout/vllm/v1/core/estimate_with_func.py"
  grep -Fq 'Literal["fcfs", "priority", "continuum"]' \
    "$checkout/vllm/config/scheduler.py"
  grep -Fq 'job_id: Optional[str] = None' \
    "$checkout/vllm/entrypoints/openai/protocol.py"
  grep -Fq 'is_last_step: Optional[bool] = None' \
    "$checkout/vllm/entrypoints/openai/protocol.py"
}

continuum_verify_wheel() {
  local environment="$1"
  test -x "$environment/bin/python" || return 1
  test "$("$environment/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')" = "0.10.2"
}

continuum_package_dir() {
  "$1/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/vllm")'
}

continuum_verify_overlay() {
  local source="$1" environment="$2" variant="$3" package_dir relative
  continuum_verify_wheel "$environment" || {
    echo "$environment does not contain the official vllm==0.10.2 wheel" >&2
    return 1
  }
  package_dir="$(continuum_package_dir "$environment")"
  for relative in "${OVERLAY_FILES[@]}"; do
    cmp -s "$source/vllm/$relative" "$package_dir/$relative" || {
      echo "installed overlay differs: $relative" >&2
      return 1
    }
  done
  if test "$variant" = reproduction; then
    cmp -s "$source/vllm/v1/core/continuum_reproduction.py" \
      "$package_dir/v1/core/continuum_reproduction.py" || {
      echo "installed reproduction sidecar differs" >&2
      return 1
    }
  elif test -e "$package_dir/v1/core/continuum_reproduction.py"; then
    echo "public environment is contaminated by the reproduction sidecar" >&2
    return 1
  fi
}

continuum_install_overlay() {
  local source="$1" environment="$2" variant="$3" package_dir relative
  command -v uv >/dev/null || { echo "uv is required" >&2; return 1; }
  if test ! -x "$environment/bin/python"; then
    mkdir -p "$(dirname "$environment")"
    uv venv --python "${CONTINUUM_PYTHON:-python3}" "$environment"
    uv pip install --python "$environment/bin/python" \
      'vllm==0.10.2' 'transformers>=4.55.2,<5'
  fi
  continuum_verify_wheel "$environment" || {
    echo "refusing non-vllm-0.10.2 environment: $environment" >&2
    return 1
  }
  uv pip install --python "$environment/bin/python" \
    'transformers>=4.55.2,<5' 'lmcache==0.3.7' hf_transfer
  package_dir="$(continuum_package_dir "$environment")"
  for relative in "${OVERLAY_FILES[@]}"; do
    mkdir -p "$(dirname "$package_dir/$relative")"
    cp "$source/vllm/$relative" "$package_dir/$relative"
  done
  if test "$variant" = reproduction; then
    cp "$source/vllm/v1/core/continuum_reproduction.py" \
      "$package_dir/v1/core/continuum_reproduction.py"
  fi
  continuum_verify_overlay "$source" "$environment" "$variant"
}

fetch() {
  continuum_fetch_public_sources
  echo "fetched Continuum public source at $COMMIT"
}

verify() {
  continuum_verify_public_source
  echo "verified Continuum public fixed-TTL baseline at $COMMIT"
}

install() {
  continuum_fetch_public_sources
  continuum_install_overlay "$checkout" "$venv" public
}

serve() {
  test "$#" -ge 1 || { usage >&2; exit 2; }
  continuum_verify_public_source
  continuum_verify_overlay "$checkout" "$venv" public
  local model="$1"
  shift
  export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-./continuum_exp}"
  mkdir -p "$RUN_OUTPUT_DIR"
  exec "$venv/bin/vllm" serve "$model" \
    --scheduling-policy continuum \
    --tensor-parallel-size "${CONTINUUM_TENSOR_PARALLEL_SIZE:-1}" \
    --port "${CONTINUUM_PORT:-8000}" \
    "$@"
}

main() {
  case "${1:-}" in
    fetch) fetch ;;
    verify) verify ;;
    install) install ;;
    serve) shift; serve "$@" ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
