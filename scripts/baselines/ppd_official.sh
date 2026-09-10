#!/usr/bin/env bash
# Public PPD decision engine with native NIXL read or push transport.
set -euo pipefail

readonly repo_url=https://github.com/freelulul/vllm-ppd.git
readonly commit=28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367
checkout=${PPD_CHECKOUT:-${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/PPD-$commit}
venv=${PPD_VENV:-$checkout/.venv}
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
  if [[ "${PPD_NATIVE_PUSH:-0}" == 1 ]]; then
    "$venv/bin/python" -c 'from importlib.metadata import version; assert version("vllm") == "0.28.0"; assert version("nixl") == version("nixl-cu13") == "1.4.1"; from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlPushConnector'
    local push_package
    push_package=$("$venv/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')
    local push_args=(--unsafe-paths --directory="$push_package" "$script_dir/ppd_push_metrics.patch")
    if [[ "$1" == install ]] && ! git apply --reverse --check "${push_args[@]}" 2>/dev/null; then
      git apply --check "${push_args[@]}"
      git apply "${push_args[@]}"
    fi
    git apply --reverse --check "${push_args[@]}"
    if [[ "${PPD_STATE_AWARE:-0}" == 1 ]]; then
      local state_args=(--unsafe-paths --directory="$push_package" "$script_dir/ppd_state_query.patch")
      if [[ "$1" == install ]] && ! git apply --reverse --check "${state_args[@]}" 2>/dev/null; then
        git apply --check "${state_args[@]}"
        git apply "${state_args[@]}"
      fi
      git apply --reverse --check "${state_args[@]}"
    fi
    return
  fi
  "$venv/bin/python" - <<'PYCODE'
from importlib.metadata import version
assert version('vllm') == '0.13.0'
assert version('nixl') == version('nixl-cu12') == '0.7.1'
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector
from nixl._api import nixl_agent
print('Verified vLLM 0.13.0 native NIXL imports; real GPU transfer requires smoke.')
PYCODE
  local package
  package=$("$venv/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')
  local patch_args=(--unsafe-paths --directory="$package" "$script_dir/ppd_nixl_metrics.patch")
  if [[ "$1" == install ]] && ! git apply --reverse --check "${patch_args[@]}" 2>/dev/null; then
    git apply --check "${patch_args[@]}"
    git apply "${patch_args[@]}"
  fi
  git apply --reverse --check "${patch_args[@]}"
}

request_metrics() {
  [[ "${PPD_NATIVE_PUSH:-0}" != 1 ]] || return 0
  local package
  package=$("$venv/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')
  local patch_args=(--unsafe-paths --directory="$package" "$script_dir/ppd_request_metrics.patch")
  if [[ "$1" == install ]] && ! git apply --reverse --check "${patch_args[@]}" 2>/dev/null; then
    git apply --check "${patch_args[@]}"
    git apply "${patch_args[@]}"
  fi
  git apply --reverse --check "${patch_args[@]}"
}

case "${1:-}" in
  fetch) fetch ;;
  verify) verify ;;
  install)
    fetch
    [[ -x "$venv/bin/python" ]] || uv venv --python 3.12 "$venv"
    if [[ "${PPD_NATIVE_PUSH:-0}" == 1 ]]; then
      uv pip install --python "$venv/bin/python" 'vllm==0.28.0' 'nixl==1.4.1' 'nixl-cu13==1.4.1'
    else
    uv pip install --python "$venv/bin/python" 'vllm==0.13.0' 'nixl==0.7.1' 'nixl-cu12==0.7.1' \
      --requirements "$checkout/requirements.txt"
    fi
    connector install
    request_metrics install
    ;;
  verify-installed)
    verify
    connector verify
    request_metrics verify
    ;;
  *) echo "Usage: $0 fetch|verify|install|verify-installed" >&2; exit 2 ;;
esac
