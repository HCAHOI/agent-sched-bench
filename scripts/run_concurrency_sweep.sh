#!/bin/bash
set -e
cd /root/agent-sched-bench
export PYTHONPATH=src
export KEEP_IMAGES_ABOVE_GB=30

: "${MANIFEST:?Set MANIFEST=/abs/path/to/simulate-manifest.yaml}"
SPEED=${SPEED:-50}
CONCURRENCY=${CONCURRENCY:-1,2,4,8}

echo "[$(date)] Starting bounded concurrency sweep: concurrency=$CONCURRENCY, speed=$SPEED"

python3 -u src/trace_collect/cli.py simulate \
    --mode cloud_model \
    --manifest "$MANIFEST" \
    --concurrency "$CONCURRENCY" \
    --container docker \
    --replay-speed "$SPEED" \
    --verbose

echo ""
echo "[$(date)] Bounded concurrency sweep complete!"
