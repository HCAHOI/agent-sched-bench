#!/usr/bin/env bash
set -euo pipefail

# Agentix publishes no source. This installs an isolated official vLLM 0.11.2
# and applies only the engine-step service counter used by the PLAS proxy.
readonly VLLM_VERSION=0.11.2
readonly PATCHED_CORE_SHA=0e7a8efc15fd81c2d82b5cbc853e2eaf09838be77fda65ddbe3922b48b6fb69d
script_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
cache_root=${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}
venv=${AGENTIX_VENV:-$cache_root/agent-sched-bench/agentix-vllm-$VLLM_VERSION}
proxy=$script_dir/agentix_reproduction.py

usage() {
  cat <<'EOF'
Usage: agentix_reproduction.sh install|patch|verify
       agentix_reproduction.sh serve-backend MODEL [vllm arguments...]
       agentix_reproduction.sh serve-proxy [proxy arguments...]

install creates an isolated vLLM==0.11.2 environment and patches only
vllm/v1/engine/core.py. serve-backend requires a new AGENTIX_SERVICE_LOG path
and forces --scheduling-policy priority. serve-proxy reads that log; pass its
--backend, --service-log, and paper-unspecified --queue-upper-bounds values.

This is the PLAS arrival-priority subset, not full Agentix: no dynamic quantum
demotion, anti-starvation, ATLAS, multi-engine routing, or custom KV-swap
kernel. Engine-step timing is an explicit inference, not authors' source.
EOF
}

core_path() {
  "$venv/bin/python" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vllm is not installed")
print(Path(spec.origin).parent / "v1/engine/core.py")
PY
}

patch_core() {
  test -x "$venv/bin/python" || { echo "missing Agentix venv: $venv" >&2; exit 1; }
  local core
  core=$(core_path)
  PYTHONPATH="$repo" "$venv/bin/python" - "$core" <<'PY'
from pathlib import Path
import sys

from scripts.baselines.agentix_reproduction import patch_vllm_core

changed = patch_vllm_core(Path(sys.argv[1]))
print("patched" if changed else "already patched", sys.argv[1])
PY
}

verify() {
  test -x "$venv/bin/python" || { echo "missing Agentix venv: $venv" >&2; exit 1; }
  test "$("$venv/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')" = "$VLLM_VERSION"
  local core actual
  core=$(core_path)
  actual=$(sha256sum "$core" | awk '{print $1}')
  test "$actual" = "$PATCHED_CORE_SHA" || {
    echo "unexpected patched vLLM core.py sha256: $actual" >&2
    exit 1
  }
  grep -Fq '# AGENTIX_ENGINE_STEP_SERVICE_V1' "$core"
  grep -Fq 'list(scheduler_output.num_scheduled_tokens)' "$core"
  echo "verified Agentix PLAS engine-step patch on vLLM $VLLM_VERSION"
}

install_baseline() {
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
  if test ! -x "$venv/bin/python"; then
    uv venv --python "${AGENTIX_PYTHON:-python3}" "$venv"
  fi
  uv pip install --python "$venv/bin/python" "vllm==$VLLM_VERSION"
  patch_core
  verify
}

serve_backend() {
  verify
  test "$#" -ge 1 || { usage >&2; exit 2; }
  local service_log=${AGENTIX_SERVICE_LOG:?AGENTIX_SERVICE_LOG must be a new path}
  test ! -e "$service_log" || {
    echo "refusing to overwrite AGENTIX_SERVICE_LOG: $service_log" >&2
    exit 1
  }
  mkdir -p "$(dirname "$service_log")"
  touch "$service_log"
  exec env AGENTIX_SERVICE_LOG="$service_log" "$venv/bin/vllm" serve "$@" \
    --scheduling-policy priority --enable-request-id-headers
}

serve_proxy() {
  verify
  exec "$venv/bin/python" "$proxy" "$@"
}

case "${1:-}" in
  install) install_baseline ;;
  patch) patch_core ;;
  verify) verify ;;
  serve-backend) shift; serve_backend "$@" ;;
  serve-proxy) shift; serve_proxy "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
