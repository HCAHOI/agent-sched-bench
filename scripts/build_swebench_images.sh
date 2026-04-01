#!/usr/bin/env bash
# Build Podman container images for SWE-bench tasks.
#
# Usage:
#   ./scripts/build_swebench_images.sh          # Build base image only
#   ./scripts/build_swebench_images.sh --all    # Build base + per-repo images
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building swebench-base image ==="
podman build \
    -t swebench-base:latest \
    -f "$PROJECT_ROOT/containers/Containerfile.base" \
    "$PROJECT_ROOT/containers"
echo "=== swebench-base:latest built ==="

if [[ "${1:-}" != "--all" ]]; then
    echo "Done. Pass --all to also build per-repo images."
    exit 0
fi

# --- Per-repo images (optional, for faster pip install) ---

REPOS_ROOT="$PROJECT_ROOT/data/swebench_repos"
if [[ ! -d "$REPOS_ROOT" ]]; then
    echo "WARNING: $REPOS_ROOT not found. Run setup_swebench_repos.sh first."
    exit 1
fi

for repo_dir in "$REPOS_ROOT"/*/; do
    repo_name="$(basename "$repo_dir")"
    image_tag="swebench-${repo_name}:latest"
    echo "=== Building $image_tag ==="

    # Create a temporary Containerfile that pre-installs repo deps
    tmpfile="$(mktemp)"
    cat > "$tmpfile" <<DOCKERFILE
FROM swebench-base:latest
COPY . /workspace/repo-src
RUN cd /workspace/repo-src && pip install -e . 2>&1 | tail -20 || true
RUN rm -rf /workspace/repo-src
DOCKERFILE

    podman build \
        -t "$image_tag" \
        -f "$tmpfile" \
        "$repo_dir" || echo "WARNING: Failed to build $image_tag (non-fatal)"

    rm -f "$tmpfile"
done

echo "=== All images built ==="
