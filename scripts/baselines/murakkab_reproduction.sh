#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${MURAKKAB_PYTHON:-$here/../../.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage: murakkab_reproduction.sh manifest
       murakkab_reproduction.sh validate INPUT.json
       murakkab_reproduction.sh solve INPUT.json [--output PLAN.json]

Runs the paper-derived static-epoch Murakkab deployment optimizer. INPUT must
declare a workflow DAG, complete profiled configurations, demand, SLOs, and
resource capacities. Black-box request traces are intentionally rejected.
EOF
}

case "${1:-}" in
  manifest)
    test "$#" -eq 1 || { usage >&2; exit 2; }
    exec "$python" "$here/murakkab_reproduction.py" manifest
    ;;
  validate|solve)
    command="$1"
    shift
    test "$#" -ge 1 || { usage >&2; exit 2; }
    exec "$python" "$here/murakkab_reproduction.py" "$command" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
