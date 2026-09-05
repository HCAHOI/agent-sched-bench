#!/usr/bin/env bash
set -euo pipefail

# SAGA's implementation is unpublished. This isolated environment runs only
# the paper-derived AFS arrival-priority and single-GPU WA-LRU/TTL subsets.
readonly VLLM_VERSION=0.11.2
readonly VLLM_TAG_COMMIT=275de34170654274616082721348b7edd9741d32

script_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
cache_root=${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}
venv=${SAGA_VENV:-$cache_root/agent-sched-bench/saga-stock-vllm-$VLLM_VERSION}
if test -n "${SAGA_REPRO_PYTHON:-}"; then
  sidecar_python="$SAGA_REPRO_PYTHON"
elif test -x "$repo/.venv/bin/python"; then
  sidecar_python="$repo/.venv/bin/python"
else
  sidecar_python=python3
fi

usage() {
  cat <<'EOF'
Usage: saga_reproduction.sh install|verify|manifest
       saga_reproduction.sh kv-policy JSON_FILE
       saga_reproduction.sh build-profile [arguments...]
       saga_reproduction.sh serve-backend MODEL [vLLM arguments...]
       saga_reproduction.sh serve-afs-subset [proxy arguments...]

This is not full SAGA. It applies the published WA-LRU/TTL policy to resident
single-GPU vLLM prefix blocks and maps published AFS urgency to static arrival
priority. It does not implement prefetch, routing, work stealing, migration,
periodic reprioritization, or the authors' private multi-GPU/CUDA system.

serve-backend forces stock vLLM's priority scheduler. Register explicit AEG
node service estimates and a deadline through /tasks/register, then attach
saga_session_id and saga_node_id to each OpenAI request sent through the proxy.
EOF
}

package_path() {
  "$venv/bin/python" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vllm is not installed")
print(Path(spec.origin).parent)
PY
}

verify() {
  test -x "$venv/bin/python" || { echo "missing SAGA subset venv: $venv" >&2; exit 1; }
  test "$("$venv/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')" = "$VLLM_VERSION"
  local package_root
  package_root=$(package_path)
  PYTHONPATH="$repo" "$venv/bin/python" -m scripts.baselines.saga_reproduction \
    verify-patched-vllm "$package_root"
  echo "verified SAGA subset on vLLM $VLLM_VERSION (tag $VLLM_TAG_COMMIT)"
}

install_baseline() {
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
  if test ! -x "$venv/bin/python"; then
    uv venv --python "${SAGA_PYTHON:-python3}" "$venv"
  fi
  uv pip install --python "$venv/bin/python" --reinstall "vllm==$VLLM_VERSION"
  local package_root
  package_root=$(package_path)
  PYTHONPATH="$repo" "$venv/bin/python" -m scripts.baselines.saga_reproduction \
    apply-vllm-patch "$package_root"
  verify
}

run_installed_sidecar() {
  test -x "$venv/bin/python" || {
    echo "SAGA subset is not installed; run '$0 install' first" >&2
    exit 1
  }
  exec env PYTHONPATH="$repo" "$venv/bin/python" -m scripts.baselines.saga_reproduction "$@"
}

case "${1:-}" in
  install) install_baseline ;;
  verify) verify ;;
  manifest) PYTHONPATH="$repo" "$sidecar_python" -m scripts.baselines.saga_reproduction manifest ;;
  kv-policy)
    shift
    PYTHONPATH="$repo" "$sidecar_python" -m scripts.baselines.saga_reproduction kv-policy "$@"
    ;;
  build-profile) shift; run_installed_sidecar build-profile "$@" ;;
  serve-backend)
    shift
    verify
    test "$#" -ge 1 || { usage >&2; exit 2; }
    exec env PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" \
      VLLM_SERVER_DEV_MODE=1 "$venv/bin/vllm" serve "$@" \
      --scheduling-policy priority
    ;;
  serve-afs-subset) shift; run_installed_sidecar serve-afs-subset "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
