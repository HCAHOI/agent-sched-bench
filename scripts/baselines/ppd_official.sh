#!/usr/bin/env bash
# Public PPD decision engine on vLLM 0.28.0 with the native NIXL push transport
# (the only transport any recorded PD/PPD run used).
set -euo pipefail

readonly repo_url=https://github.com/freelulul/vllm-ppd.git
readonly commit=28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367
checkout=${PPD_CHECKOUT:-${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/PPD-$commit}
venv=${PPD_VENV:-$checkout/.venv}
# CUDA flavour of the vLLM 0.28.0 push-transport build: cu13 (PyPI wheel, needs
# driver >= 580) or cu129 (vLLM's cu129 wheel + nixl-cu12, runs on driver 570).
cuda=${PPD_CUDA:-cu13}
nixl_cuda_pkg=nixl-cu13; [[ "$cuda" == cu13 ]] || nixl_cuda_pkg=nixl-cu12
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(dirname "$(dirname "$script_dir")")${PYTHONPATH:+:$PYTHONPATH}"

verify() {
  [[ $(git -C "$checkout" rev-parse HEAD) == "$commit" ]]
  [[ $(git -C "$checkout" remote get-url origin) == "$repo_url" ]]
  git -C "$checkout" diff --quiet HEAD
}

fetch() {
  if [[ ! -e "$checkout" ]]; then
    git clone --no-checkout "$repo_url" "$checkout"
    git -C "$checkout" checkout --detach "$commit"
  fi
  verify
}

connector() {
  "$venv/bin/python" -c "from importlib.metadata import version; assert version('vllm').split('+')[0] == '0.28.0'; assert version('nixl') == version('$nixl_cuda_pkg') == '1.4.1'; from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlPushConnector"
  local push_package
  # resolve(): git apply refuses patch targets that pass through a symlink (e.g. /workspace -> data disk)
  push_package=$("$venv/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).resolve().parent)')
  local push_args=(--unsafe-paths --directory="$push_package" "$script_dir/ppd_push_metrics.patch")
  if [[ "$1" == install ]] && ! git apply --reverse --check "${push_args[@]}" 2>/dev/null; then
    git apply --check "${push_args[@]}"
    git apply "${push_args[@]}"
  fi
  git apply --reverse --check "${push_args[@]}"
}

case "${1:-}" in
  fetch) fetch ;;
  verify) verify ;;
  install)
    fetch
    [[ -x "$venv/bin/python" ]] || uv venv --python 3.12 "$venv"
    if [[ "$cuda" == cu13 ]]; then
      uv pip install --python "$venv/bin/python" 'vllm==0.28.0' 'nixl==1.4.1' 'nixl-cu13==1.4.1'
    else
      [[ "$cuda" == cu129 ]] || { echo "PPD_CUDA must be cu13 or cu129" >&2; exit 2; }
      uv pip install --python "$venv/bin/python" --torch-backend=cu129 \
        --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 'vllm==0.28.0+cu129' 'nixl==1.4.1' 'nixl-cu12==1.4.1'
    fi
    connector install
    ;;
  verify-installed)
    verify
    connector verify
    ;;
  *) echo "Usage: $0 fetch|verify|install|verify-installed" >&2; exit 2 ;;
esac
