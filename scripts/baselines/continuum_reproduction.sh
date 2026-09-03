#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=continuum_public.sh
source "$here/continuum_public.sh"

public_checkout="$checkout"
public_venv="$venv"
checkout="${CONTINUUM_REPRODUCTION_CHECKOUT:-$cache_root/agent-sched-bench/vllm-continuum-reproduction-v6-$COMMIT}"
venv="${CONTINUUM_REPRODUCTION_VENV:-$cache_root/agent-sched-bench/venvs/continuum-reproduction-v6-$COMMIT}"
validated_observability_args=()

usage() {
  cat <<'EOF'
Usage: continuum_reproduction.sh apply|verify|install|manifest
       continuum_reproduction.sh serve|serve-oracle-deadline MODEL --dtype DTYPE --kv-cache-dtype DTYPE

Adds the cost-model/empirical-CDF TTL estimator from Continuum v6 to an
independent checkout of the authors' official public fork. Installation is an
official vllm==0.10.2 wheel plus the exact fork Python overlay and sidecar; it
never compiles the fork and never modifies the public fixed-TTL environment.

serve requires:
  CONTINUUM_REPRODUCTION_PROFILE  measured prefill or KV reload JSON with an
                                  exact runtime_binding
  CONTINUUM_REPRODUCTION_MODE     prefill or reload
  CONTINUUM_TENSOR_PARALLEL_SIZE  exact TP size (default: 1)

Paper-omitted choices are printed by the manifest command. They are fixed,
explicit, and were not tuned on agent-sched-bench traces.
EOF
}

prepare_checkout() {
  test "$(realpath -m "$checkout")" != "$(realpath -m "$public_checkout")" || {
    echo "reproduction and public checkouts must be distinct" >&2
    return 1
  }
  continuum_fetch_checkout "$upstream_checkout" "$UPSTREAM_COMMIT" "$UPSTREAM_URL"
  if test ! -e "$checkout"; then
    continuum_fetch_checkout "$checkout" "$COMMIT" "$REPO_URL"
    continuum_verify_fork_delta "$upstream_checkout" "$checkout"
  fi
  if test -e "$checkout/vllm/v1/core/continuum_reproduction.py"; then
    python3 "$here/continuum_reproduction.py" verify "$checkout"
  else
    continuum_verify_clean_checkout "$checkout" "$COMMIT" "$REPO_URL"
    continuum_verify_fork_delta "$upstream_checkout" "$checkout"
  fi
}

apply_patch() {
  prepare_checkout
  if test ! -e "$checkout/vllm/v1/core/continuum_reproduction.py"; then
    python3 "$here/continuum_reproduction.py" apply "$checkout"
  fi
}

verify() {
  continuum_verify_clean_checkout \
    "$upstream_checkout" "$UPSTREAM_COMMIT" "$UPSTREAM_URL"
  python3 "$here/continuum_reproduction.py" verify "$checkout"
  echo "verified Continuum estimator reproduction at $COMMIT"
}

install() {
  test "$(realpath -m "$venv")" != "$(realpath -m "$public_venv")" || {
    echo "reproduction and public virtual environments must be distinct" >&2
    return 1
  }
  apply_patch
  verify
  continuum_install_overlay "$checkout" "$venv" reproduction
}

validate_serve_args() {
  validated_model_dtype=""
  validated_kv_dtype=""
  validated_observability_args=()
  while test "$#" -gt 0; do
    case "$1" in
      --dtype)
        test "$#" -ge 2 || { echo "--dtype requires a value" >&2; return 2; }
        test -z "$validated_model_dtype" || { echo "duplicate --dtype" >&2; return 2; }
        validated_model_dtype="$2"
        shift 2
        ;;
      --dtype=*)
        test -z "$validated_model_dtype" || { echo "duplicate --dtype" >&2; return 2; }
        validated_model_dtype="${1#*=}"
        shift
        ;;
      --kv-cache-dtype)
        test "$#" -ge 2 || { echo "--kv-cache-dtype requires a value" >&2; return 2; }
        test -z "$validated_kv_dtype" || { echo "duplicate --kv-cache-dtype" >&2; return 2; }
        validated_kv_dtype="$2"
        shift 2
        ;;
      --kv-cache-dtype=*)
        test -z "$validated_kv_dtype" || { echo "duplicate --kv-cache-dtype" >&2; return 2; }
        validated_kv_dtype="${1#*=}"
        shift
        ;;
      --enable-prompt-tokens-details)
        [[ " ${validated_observability_args[*]} " != *" --enable-prompt-tokens-details "* ]] || {
          echo "duplicate --enable-prompt-tokens-details" >&2
          return 2
        }
        validated_observability_args+=("$1")
        shift
        ;;
      --worker-cls)
        test "$#" -ge 2 || { echo "--worker-cls requires a value" >&2; return 2; }
        [[ " ${validated_observability_args[*]} " != *" --worker-cls "* ]] || {
          echo "duplicate --worker-cls" >&2
          return 2
        }
        validated_observability_args+=("$1" "$2")
        shift 2
        ;;
      --kv-events-config)
        test "$#" -ge 2 || { echo "--kv-events-config requires a value" >&2; return 2; }
        [[ " ${validated_observability_args[*]} " != *" --kv-events-config "* ]] || {
          echo "duplicate --kv-events-config" >&2
          return 2
        }
        validated_observability_args+=("$1" "$2")
        shift 2
        ;;
      *)
        echo "unsupported vLLM argument: $1" >&2
        return 2
        ;;
    esac
  done
  test -n "$validated_model_dtype" || { echo "serve requires --dtype" >&2; return 2; }
  test -n "$validated_kv_dtype" || { echo "serve requires --kv-cache-dtype" >&2; return 2; }
}

build_serve_command() {
  local model="$1" model_dtype="$2" kv_dtype="$3" tp="$4" max_model_len="$5"
  local mode="$6"
  serve_command=(
    "$venv/bin/vllm" serve "$model"
    --scheduling-policy continuum
    --tensor-parallel-size "$tp"
    --host 127.0.0.1
    --port "${CONTINUUM_PORT:-8000}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --max-model-len "$max_model_len"
    --max-num-seqs 8
    --enable-prefix-caching
    --enforce-eager
    --dtype "$model_dtype"
    --kv-cache-dtype "$kv_dtype"
    "${validated_observability_args[@]}"
  )
  if test "$mode" = reload; then
    serve_command+=(
      --kv-transfer-config
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    )
  fi
}

serve() {
  test "$#" -ge 1 || { usage >&2; exit 2; }
  test -f "${CONTINUUM_REPRODUCTION_PROFILE:-}" || {
    echo "CONTINUUM_REPRODUCTION_PROFILE must name a measured profile" >&2
    exit 2
  }
  case "${CONTINUUM_REPRODUCTION_MODE:-}" in
    prefill|reload) ;;
    *) echo "CONTINUUM_REPRODUCTION_MODE must be prefill or reload" >&2; exit 2 ;;
  esac
  local model="$1" model_dtype kv_dtype tp max_model_len
  shift
  validate_serve_args "$@"
  model_dtype="$validated_model_dtype"
  kv_dtype="$validated_kv_dtype"
  tp="${CONTINUUM_TENSOR_PARALLEL_SIZE:-1}"
  [[ "$tp" =~ ^[1-9][0-9]*$ ]] || {
    echo "CONTINUUM_TENSOR_PARALLEL_SIZE must be a positive integer" >&2
    exit 2
  }

  install
  max_model_len="$(
    "$venv/bin/python" "$here/continuum_reproduction.py" validate-runtime \
      "$CONTINUUM_REPRODUCTION_PROFILE" "$CONTINUUM_REPRODUCTION_MODE" \
      "$model" "$model_dtype" "$kv_dtype" "$tp"
  )"
  export CONTINUUM_REPRODUCTION_MODEL="$model"
  export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-./continuum_reproduction_exp}"
  export VLLM_SERVER_DEV_MODE=1
  if test "$CONTINUUM_REPRODUCTION_MODE" = reload; then
    export LMCACHE_MAX_LOCAL_CPU_SIZE=100
  fi
  mkdir -p "$RUN_OUTPUT_DIR"
  build_serve_command \
    "$model" "$model_dtype" "$kv_dtype" "$tp" "$max_model_len" \
    "$CONTINUUM_REPRODUCTION_MODE"
  exec "${serve_command[@]}"
}

main() {
  case "${1:-}" in
    apply) apply_patch ;;
    verify) verify ;;
    install) install ;;
    manifest) python3 "$here/continuum_reproduction.py" manifest ;;
    serve) unset CONTINUUM_ORACLE_DEADLINE; shift; serve "$@" ;;
    serve-oracle-deadline)
      : "${CONTINUUM_ORACLE_DECODE_MS_PER_TOKEN:?Continuum Oracle decode coefficient is required}"
      export CONTINUUM_ORACLE_DEADLINE=1
      shift
      serve "$@"
      ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
