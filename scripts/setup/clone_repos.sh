#!/usr/bin/env bash
# Clone SWE-bench repos locally for fast container-internal cloning.
#
# Usage:
#   ./scripts/setup/clone_repos.sh [tasks.json]
#
# If tasks.json is provided, only repos referenced in it are cloned.
# Otherwise, clones all repos commonly used in SWE-bench Verified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPOS_ROOT="$PROJECT_ROOT/data/swebench_repos"
TASKS_FILE="${1:-${PROJECT_ROOT}/data/swebench_verified/tasks.json}"

if [[ -d "$REPOS_ROOT" ]] && [[ -n "$(ls -A "$REPOS_ROOT" 2>/dev/null)" ]]; then
    echo "[setup] SKIP clone_repos: $REPOS_ROOT is non-empty"
    exit 0
fi

mkdir -p "$REPOS_ROOT"

if [[ -f "$TASKS_FILE" ]]; then
    REPOS=$(python3 -c "
import json, sys
tasks = json.load(open(sys.argv[1]))
repos = sorted(set(t['repo'] for t in tasks))
for r in repos:
    print(r)
" "$TASKS_FILE")
else
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

echo "[setup] Cloning SWE-bench repos to $REPOS_ROOT"

while IFS= read -r repo; do
    owner="${repo%%/*}"
    name="${repo##*/}"
    dir_name="${owner}__${name}"
    target="$REPOS_ROOT/$dir_name"

    if [[ -d "$target" ]]; then
        echo "[setup] SKIP: $dir_name (already exists)"
        continue
    fi

    echo "[setup] CLONE: $repo → $dir_name"
    git clone --quiet "https://github.com/${repo}.git" "$target"
    echo "[setup]   done ($(du -sh "$target" | cut -f1))"
done <<< "$REPOS"

echo "[setup] clone_repos done"
du -sh "$REPOS_ROOT"
