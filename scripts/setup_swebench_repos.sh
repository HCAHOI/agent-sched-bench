#!/usr/bin/env bash
# Clone SWE-bench repos locally for fast container-internal cloning.
#
# Usage:
#   ./scripts/setup_swebench_repos.sh [tasks.json]
#
# If tasks.json is provided, only repos referenced in it are cloned.
# Otherwise, clones all repos commonly used in SWE-bench Verified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPOS_ROOT="$PROJECT_ROOT/data/swebench_repos"

mkdir -p "$REPOS_ROOT"

# Extract unique repos from tasks.json if provided
if [[ -n "${1:-}" ]] && [[ -f "$1" ]]; then
    REPOS=$(python3 -c "
import json, sys
tasks = json.load(open(sys.argv[1]))
repos = sorted(set(t['repo'] for t in tasks))
for r in repos:
    print(r)
" "$1")
else
    # Default repos for SWE-bench Verified
    REPOS="django/django
sympy/sympy
scikit-learn/scikit-learn
matplotlib/matplotlib
sphinx-doc/sphinx
pytest-dev/pytest
astropy/astropy
pydata/xarray
psf/requests
pallets/flask
pylint-dev/pylint
mwaskom/seaborn"
fi

echo "=== Cloning SWE-bench repos to $REPOS_ROOT ==="

while IFS= read -r repo; do
    owner="${repo%%/*}"
    name="${repo##*/}"
    dir_name="${owner}__${name}"
    target="$REPOS_ROOT/$dir_name"

    if [[ -d "$target" ]]; then
        echo "SKIP: $dir_name (already exists)"
        continue
    fi

    echo "CLONE: $repo → $dir_name"
    git clone --quiet "https://github.com/${repo}.git" "$target"
    echo "  done ($(du -sh "$target" | cut -f1))"
done <<< "$REPOS"

echo "=== All repos cloned ==="
du -sh "$REPOS_ROOT"
