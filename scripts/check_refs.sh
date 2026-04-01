#!/usr/bin/env bash
set -euo pipefail
found=0
for f in configs/systems/*.yaml; do
  if grep -q "REPLACE_WITH_COMMIT_OR_TAG" "$f" 2>/dev/null; then
    echo "ERROR: Placeholder ref in $f"
    found=1
  fi
done
if [ "$found" -eq 1 ]; then
  echo "Pin all refs before running experiments."
  exit 1
fi
exit 0
