#!/usr/bin/env bash
set -euo pipefail

# Official MIT-licensed implementation by Hao Kang et al.
readonly REPO_URL="https://github.com/ThunderAgent-org/ThunderAgent.git"
readonly COMMIT="7ddc8610270e56d3b109eed8796b3a4360fc67c9"

cache_root="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
checkout="${THUNDERAGENT_CHECKOUT:-$cache_root/agent-sched-bench/ThunderAgent-$COMMIT}"
venv="${THUNDERAGENT_VENV:-$checkout/.venv}"

usage() {
  cat <<'EOF'
Usage: thunderagent_official.sh fetch|verify|install|serve

Runs the official ThunderAgent scheduler pinned at commit 7ddc861. The serve
mode fixes the paper's vLLM configuration: TR router, five-second monitoring,
and 2^-t acting-token decay. Point OpenClaw replay at its /v1 endpoint and use
--shadow-llm-mode thunderagent so each task is tagged and released.

Environment:
  THUNDERAGENT_CHECKOUT      official checkout directory
  THUNDERAGENT_VENV          isolated environment directory
  THUNDERAGENT_PYTHON        Python used by uv (default: python3)
  THUNDERAGENT_BACKENDS      comma-separated vLLM URLs (default: localhost:8000)
  THUNDERAGENT_HOST          proxy bind address (default: 127.0.0.1)
  THUNDERAGENT_PORT          proxy port (default: 9000)
  THUNDERAGENT_PROFILE_DIR   official CSV output (default: ./thunderagent_profiles)
EOF
}

verify() {
  test -d "$checkout/.git" || { echo "missing ThunderAgent checkout: $checkout" >&2; exit 1; }
  test "$(git -C "$checkout" rev-parse HEAD)" = "$COMMIT" || {
    echo "ThunderAgent checkout is not pinned at $COMMIT" >&2
    exit 1
  }
  test "$(git -C "$checkout" remote get-url origin)" = "$REPO_URL" || {
    echo "ThunderAgent checkout does not use the official remote" >&2
    exit 1
  }
  grep -Fq 'MIT License' "$checkout/LICENSE.md"
  grep -Fq 'extra_body["program_id"] = "unique_id"' "$checkout/README.md"
  grep -Fq '@app.post("/programs/release")' "$checkout/ThunderAgent/app.py"
  grep -Fq 'choices=["default", "tr"]' "$checkout/ThunderAgent/__main__.py"
  echo "verified official ThunderAgent at $COMMIT"
}

fetch() {
  if test ! -e "$checkout"; then
    mkdir -p "$(dirname "$checkout")"
    git init "$checkout"
    git -C "$checkout" remote add origin "$REPO_URL"
    git -C "$checkout" fetch --depth=1 origin "$COMMIT"
    git -C "$checkout" checkout --detach "$COMMIT"
  fi
  verify
}

install() {
  fetch
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
  uv venv --python "${THUNDERAGENT_PYTHON:-python3}" "$venv"
  uv pip install --python "$venv/bin/python" --editable "$checkout"
}

serve() {
  verify
  test -x "$venv/bin/thunderagent" || {
    echo "ThunderAgent is not installed; run '$0 install' first" >&2
    exit 1
  }
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  exec "$venv/bin/python" "$script_dir/thunderagent_official_launcher.py"
}

case "${1:-}" in
  fetch) fetch ;;
  verify) verify ;;
  install) install ;;
  serve) serve ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
