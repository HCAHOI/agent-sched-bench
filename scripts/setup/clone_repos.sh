#!/usr/bin/env bash
# Clone SWE-rebench repos locally for fast container-internal cloning.
#
# Usage:
#   ./scripts/setup/clone_repos.sh [tasks.json] [repos_root]
#
# Arguments:
#   tasks.json  — path to a tasks JSON file; only repos referenced there are
#                 cloned. Default: data/swe-rebench/tasks.json.
#   repos_root  — target directory for the cloned mirrors. Default:
#                 data/swe-rebench/repos.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASKS_FILE="${1:-${PROJECT_ROOT}/data/swe-rebench/tasks.json}"
REPOS_ROOT="${2:-$PROJECT_ROOT/data/swe-rebench/repos}"

if [[ ! -f "$TASKS_FILE" ]]; then
    echo "ERROR: tasks file not found: $TASKS_FILE" >&2
    echo "Run: ./scripts/setup/swe_rebench_data.sh" >&2
    exit 1
fi

if [[ -d "$REPOS_ROOT" ]] && [[ -n "$(ls -A "$REPOS_ROOT" 2>/dev/null)" ]]; then
    echo "[setup] SKIP clone_repos: $REPOS_ROOT is non-empty"
    exit 0
fi

mkdir -p "$REPOS_ROOT"

REPOS=$(python3 -c "
import json, sys
tasks = json.load(open(sys.argv[1]))
repos = sorted(set(t['repo'] for t in tasks))
for r in repos:
    print(r)
" "$TASKS_FILE")

echo "[setup] Cloning SWE-rebench repos to $REPOS_ROOT"

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
