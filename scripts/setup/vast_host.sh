#!/usr/bin/env bash
# Local entry point for a 2xL40S Vast host used by scripts/evaluation/vast_two_instance.py.
#
#   scripts/setup/vast_host.sh use  HOST PORT   remember the host in .vast-host (gitignored)
#   scripts/setup/vast_host.sh ship             copy the local HEAD (+ uv.lock) to /workspace/agent-sched-bench
#   scripts/setup/vast_host.sh bootstrap        ship, then build/verify the host environment (idempotent, ~35 min fresh)
#   scripts/setup/vast_host.sh verify           re-run only the host verification step
#   scripts/setup/vast_host.sh status           GPUs, disk, source revision, supervisor programs
#   scripts/setup/vast_host.sh ssh [cmd]        open a shell or run a command on the host
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
cmd=${1:-}; shift || true
if [[ $cmd == use ]]; then
  printf '%s %s\n' "$1" "$2" > .vast-host
  ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -p "$2" "root@$1" 'nvidia-smi --query-gpu=name --format=csv,noheader | tr "\n" " "; echo'
  exit
fi
[[ -f .vast-host ]] || { echo "no host remembered; run: $0 use HOST PORT" >&2; exit 2; }
read -r host port < .vast-host
ssh=(ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=15 -p "$port" "root@$host")
remote=/workspace/agent-sched-bench

ship() {
  rev=$(git rev-parse HEAD)
  [[ -z $(git status --short --untracked-files=no) ]] || echo "warning: shipping HEAD $rev, uncommitted changes are not included" >&2
  "${ssh[@]}" "mkdir -p $remote" </dev/null
  git archive HEAD src scripts configs tests pyproject.toml CLAUDE.md millstone | "${ssh[@]}" "tar x -C $remote"
  "${ssh[@]}" "cat > $remote/uv.lock && printf '%s\n' $rev > $remote/SOURCE_REV" < uv.lock
  echo "shipped $rev to $host:$remote"
}

case $cmd in
  ship) ship ;;
  bootstrap)
    ship
    "${ssh[@]}" "cd $remote && nohup bash scripts/setup/vast_two_instance_host.sh > /workspace/bootstrap.log 2>&1 & echo started" </dev/null
    echo "following /workspace/bootstrap.log (safe to Ctrl-C; the host keeps going)"
    "${ssh[@]}" 'tail -n +1 -f /workspace/bootstrap.log | grep --line-buffered -E "^\[|error|Error|refusing|does not|No such|VERIFY" | sed "/BOOTSTRAP COMPLETE\|BOOTSTRAP FAILED/q"' </dev/null
    ;;
  verify) "${ssh[@]}" "cd $remote && bash scripts/setup/vast_two_instance_host.sh verify" </dev/null ;;
  status)
    "${ssh[@]}" "hostname; cat ~/.vast_containerlabel 2>/dev/null; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader;
      df -h /workspace | tail -1; printf 'source: %s\n' \"\$(cat $remote/SOURCE_REV 2>/dev/null || echo none)\";
      grep -E '^\[' /workspace/bootstrap.log 2>/dev/null | tail -1; supervisorctl status 2>/dev/null | grep -v -E 'caddy|cron|instance_portal|jupyter|pyworker|syncthing|tensorboard|tunnel_manager' || true" </dev/null
    echo "local HEAD: $(git rev-parse HEAD)"
    ;;
  ssh) exec ssh -p "$port" "root@$host" "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
