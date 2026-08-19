#!/usr/bin/env bash
set -euo pipefail

readonly REPO_URL="https://github.com/cachewise-project/cachewise-coding-traces.git"
readonly COMMIT="181c435a090d328d00bbbee4c8eeb27d32f3abd2"

cache_root="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
checkout="${CACHEWISE_CHECKOUT:-$cache_root/agent-sched-bench/cachewise-$COMMIT}"
venv="${CACHEWISE_VENV:-$checkout/.venv}"

usage() {
  cat <<'EOF'
Usage: cachewise_official.sh fetch|verify|install|train-published|infer [ARGS...]

Runs CacheWise's official released tool-duration predictor at commit 181c435.
The release does not contain the paper's modified vLLM scheduler, KV block
manager, or N_rebuild=3 eviction-heap integration, so this is not the full
CacheWise serving baseline. Its checked-in split trains on 34 of 94 sessions
and its dynamic MiniBatchKMeans configuration is not the paper's 80% split and
C20/C50/C100 evaluation.

Commands:
  fetch            Fetch and verify the pinned official repository.
  verify           Verify source, commit, and released predictor contract.
  install          Create an isolated uv environment from requirements.txt.
  train-published  Train the official predictor on its published dataset/split.
  infer ARGS...    Run the official infer.py with ARGS.

Environment:
  CACHEWISE_CHECKOUT  checkout directory
  CACHEWISE_VENV      virtual environment directory
  CACHEWISE_PYTHON    Python used by uv (default: python3)
EOF
}

verify() {
  test -d "$checkout/.git" || {
    echo "missing CacheWise checkout: $checkout" >&2
    exit 1
  }
  test "$(git -C "$checkout" rev-parse HEAD)" = "$COMMIT" || {
    echo "CacheWise checkout is not pinned at $COMMIT" >&2
    exit 1
  }
  test "$(git -C "$checkout" remote get-url origin)" = "$REPO_URL" || {
    echo "CacheWise checkout does not use the official remote" >&2
    exit 1
  }
  grep -Fq 'SIM_THRESHOLD = 0.3' \
    "$checkout/tool_duration_prediction/infer.py"
  grep -Fq 'MIN_CLUSTER_SIZE = 30' \
    "$checkout/tool_duration_prediction/build_clusters.py"
  grep -Fq 'MAX_CLUSTERS = 1000' \
    "$checkout/tool_duration_prediction/build_clusters.py"
  grep -Fq 'MiniBatchKMeans(' \
    "$checkout/tool_duration_prediction/build_clusters.py"
  grep -Fq 'ngram_range=(1, 2), max_features=5000' \
    "$checkout/tool_duration_prediction/build_clusters.py"
  test ! -d "$checkout/vllm" || {
    echo "unexpected vLLM source in the pinned trace release" >&2
    exit 1
  }
  echo "verified official CacheWise predictor release at $COMMIT"
  echo "full CacheWise vLLM serving code is not present in this release"
}

fetch() {
  if test ! -e "$checkout"; then
    mkdir -p "$(dirname "$checkout")"
    git init -q "$checkout"
    git -C "$checkout" remote add origin "$REPO_URL"
    git -C "$checkout" fetch -q --depth=1 origin "$COMMIT"
    git -C "$checkout" checkout -q --detach "$COMMIT"
  fi
  verify
}

install() {
  fetch
  command -v uv >/dev/null || {
    echo "uv is required" >&2
    exit 1
  }
  uv venv --python "${CACHEWISE_PYTHON:-python3}" "$venv"
  uv pip install --python "$venv/bin/python" \
    --requirements "$checkout/requirements.txt"
}

train_published() {
  verify
  test -x "$venv/bin/python" || {
    echo "CacheWise is not installed; run '$0 install' first" >&2
    exit 1
  }
  exec "$venv/bin/python" \
    "$checkout/tool_duration_prediction/build_clusters.py" "$@"
}

infer() {
  verify
  test -x "$venv/bin/python" || {
    echo "CacheWise is not installed; run '$0 install' first" >&2
    exit 1
  }
  test -f "$checkout/tool_duration_prediction/models/all_models.pkl" || {
    echo "CacheWise model is absent; run '$0 train-published' first" >&2
    exit 1
  }
  exec "$venv/bin/python" "$checkout/tool_duration_prediction/infer.py" "$@"
}

case "${1:-}" in
  fetch) fetch ;;
  verify) verify ;;
  install) install ;;
  train-published) shift; train_published "$@" ;;
  infer) shift; infer "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
