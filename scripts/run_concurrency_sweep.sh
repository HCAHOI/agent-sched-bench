#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KEEP_IMAGES_ABOVE_GB="${KEEP_IMAGES_ABOVE_GB:-30}"

: "${MANIFEST:?Set MANIFEST=/abs/path/to/simulate-manifest.yaml}"
SPEED=${SPEED:-50}
CONCURRENCY=${CONCURRENCY:-1,2,4,8}
CONTAINER=${CONTAINER:-docker}

echo "[$(date)] Starting bounded concurrency sweep: concurrency=$CONCURRENCY, speed=$SPEED"

uv run python -u -m trace_collect.cli simulate \
    --mode cloud_model \
    --manifest "$MANIFEST" \
    --concurrency "$CONCURRENCY" \
    --container "$CONTAINER" \
    --replay-speed "$SPEED" \
    --verbose

echo ""
echo "[$(date)] Bounded concurrency sweep complete!"
