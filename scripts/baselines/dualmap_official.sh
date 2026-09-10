#!/usr/bin/env bash
set -euo pipefail

readonly repo_url=https://github.com/ASISys/DualMap.git
readonly commit=24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checkout=${DUALMAP_CHECKOUT:-${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/DualMap-$commit}
venv=${DUALMAP_VENV:-$checkout/.venv}

verify() {
  [[ $(git -C "$checkout" rev-parse HEAD) == "$commit" ]]
  [[ $(git -C "$checkout" remote get-url origin) == "$repo_url" ]]
  git -C "$checkout" diff --quiet HEAD
  echo "Verified DualMap public scheduler at $commit"
}

fetch() {
  if [[ ! -e "$checkout" ]]; then
    git clone --no-checkout "$repo_url" "$checkout"
    git -C "$checkout" checkout --detach "$commit"
  fi
  verify
}

case "${1:-}" in
  fetch) fetch ;;
  verify) verify ;;
  install)
    fetch
    [[ -x "$venv/bin/python" ]] || uv venv --python 3.12 "$venv"
    uv pip install --python "$venv/bin/python" 'transformers==4.57.6' \
      fastapi httpx uvicorn aiohttp requests numpy uhashring pympler matplotlib jinja2
    ;;
  serve)
    shift
    verify
    export PYTHONPATH="$checkout:$repo${PYTHONPATH:+:$PYTHONPATH}"
    exec "$venv/bin/python" -m scripts.baselines.dualmap_official_proxy "$@"
    ;;
  *) echo "Usage: $0 fetch|verify|install|serve [proxy arguments]" >&2; exit 2 ;;
esac
