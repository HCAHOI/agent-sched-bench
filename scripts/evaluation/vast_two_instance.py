"""Drive one two-instance serving run on a remote 2xL40S host from this machine.

The host runs the engines, proxy and collectors through
scripts/evaluation/run_two_instance_fcfs.sh under supervisor; this machine
replays the agent workload in Docker task containers against the host proxy
over an SSH tunnel, then pulls the host-side run directory into
results/<name>/server/. Every result-affecting flag is passed explicitly and
recorded in results/<name>/replay-command.json and the supervisor conf.

Example (DualMap, mixed56):
  .venv/bin/python scripts/evaluation/vast_two_instance.py --host H --port P \
    --name mixed56-vast-dualmap-20260911-r1 --router-policy dualmap \
    --env DUALMAP_PREFILL_TPOT=5.44e-05 --env DUALMAP_CPU_CACHE_GIB=48
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
CONTINUUM_COMMIT = "316a58794a6ff86b216e579b74fd56ed0c5a911f"


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--name", required=True, help="run name; becomes results/<name> here and on the host")
    p.add_argument("--router-policy", required=True, choices=["least-requests", "thunderagent", "dualmap", "pd", "ppd"])
    p.add_argument("--instance-policy", default="fcfs", choices=["fcfs", "continuum"])
    p.add_argument("--task-sticky", action="store_true")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--timeout-s", type=int, default=1800, help="SHADOW_LLM_TIMEOUT_S on both sides")
    p.add_argument("--manifest", default=str(REPO / "analysis/development/mixed56-2l40s-concurrency32-v1/manifest.yaml"))
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for the host launcher (repeatable)")
    p.add_argument("--smoke", action="store_true", help="host-side --smoke instead of a workload replay")
    p.add_argument("--calibrate", action="store_true", help="host-side --calibrate (dualmap prefill TPOT); no replay")
    p.add_argument("--replay-budget-s", type=int, default=9000, help="SIGINT the replay after this wall time")
    p.add_argument("--tunnel-port", type=int, default=19019)
    p.add_argument("--remote-repo", default="/workspace/agent-sched-bench")
    a = p.parse_args()

    run = REPO / "results" / a.name
    remote_run = f"{a.remote_repo}/results/{a.name}"
    if run.exists():
        sys.exit(f"{run} exists")
    ssh = ["ssh", "-p", str(a.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", f"root@{a.host}"]

    def remote(cmd: str, **kw) -> subprocess.CompletedProcess:
        return subprocess.run(ssh + [cmd], text=True, capture_output=True, timeout=120, **kw)

    pre = remote(f"test -d {a.remote_repo}/scripts && test ! -e {remote_run} && "
                 f"[ \"$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)\" = 2 ] && "
                 f"! supervisorctl status {a.name} 2>/dev/null | grep -q RUNNING")
    if pre.returncode:
        sys.exit(f"host preflight failed: {pre.stdout}{pre.stderr}")

    run.mkdir(parents=True)
    cache = "/workspace/.cache"
    env = {
        "HF_HOME": "/workspace/.hf_home", "XDG_CACHE_HOME": cache,
        "RUNNER_PYTHON": f"{cache}/agent-sched-bench/venvs/continuum-public-{CONTINUUM_COMMIT}/bin/python",
        "DRAM_METRICS": "off", "CONCURRENCY": str(a.concurrency),
        "ROUTER_POLICY": a.router_policy, "INSTANCE_POLICY": a.instance_policy,
        "TASK_STICKY": "1" if a.task_sticky else "0", "SHADOW_LLM_TIMEOUT_S": str(a.timeout_s),
        "SOURCE_BASE_REV": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "PREFILL_CPUSET": "48-50", "DECODE_CPUSET": "51-53", "ROUTER_CPUSET": "54",
        "MANIFEST": f"/workspace/manifests/{a.name}.yaml", "RUN_ROOT": remote_run,
    }
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    host_only = a.smoke or a.calibrate
    mode = "--smoke" if a.smoke else "--calibrate" if a.calibrate else "--external-replay"
    conf = (f"[program:{a.name}]\ndirectory={a.remote_repo}\n"
            f"command=/usr/bin/env {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} "
            f"bash scripts/evaluation/run_two_instance_fcfs.sh {mode}\n"
            "autostart=false\nautorestart=false\nstopasgroup=true\nkillasgroup=true\n"
            f"stdout_logfile=/workspace/{a.name}-launch.log\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n")
    (run / "launch.conf").write_text(conf)
    subprocess.run(["scp", "-q", "-P", str(a.port), a.manifest, f"root@{a.host}:/workspace/manifests/{a.name}.yaml"],
                   check=True) if remote("mkdir -p /workspace/manifests").returncode == 0 else sys.exit("mkdir failed")
    subprocess.run(["scp", "-q", "-P", str(a.port), str(run / "launch.conf"),
                    f"root@{a.host}:/etc/supervisor/conf.d/{a.name}.conf"], check=True)
    started = remote(f"supervisorctl reread >/dev/null && supervisorctl update >/dev/null && supervisorctl start {a.name}")
    print(started.stdout.strip(), flush=True)
    if started.returncode or "started" not in started.stdout:
        sys.exit(started.stderr + remote(f"tail -20 /workspace/{a.name}-launch.log").stdout)

    def program_running() -> bool:
        s = remote(f"supervisorctl status {a.name}")
        return "RUNNING" in s.stdout

    def host_done() -> bool:
        return remote(f"test -f {remote_run}/exit-code").returncode == 0

    if not host_only:
        while remote(f"test -f {remote_run}/external-replay-ready").returncode:
            if not program_running():
                sys.exit(f"host launcher exited before serving was ready; see /workspace/{a.name}-launch.log")
            time.sleep(10)
        tunnel = subprocess.Popen(ssh + ["-N", "-L", f"{a.tunnel_port}:127.0.0.1:9000"])
        shadow_mode = "continuum-public" if a.router_policy == "least-requests" else "thunderagent"
        argv = [str(REPO / ".venv/bin/python"), "-m", "trace_collect.cli", "simulate",
                "--manifest", a.manifest, "--output-dir", str(run / "output"),
                "--container", "docker", "--network-mode", "host",
                "--concurrency", str(a.concurrency), "--workers", "1", "--prep-concurrency", "8", "--replay-speed", "1",
                "--shadow-llm-api-base", f"http://127.0.0.1:{a.tunnel_port}/v1", "--shadow-llm-model", MODEL,
                "--shadow-llm-timeout-s", str(a.timeout_s), "--shadow-llm-seed", "0", "--shadow-llm-mode", shadow_mode,
                "--resource-monitoring", "off", "--pmu-monitoring", "off", "--memory-bandwidth-monitoring", "off",
                "--replacement-delay-mean-s", "10", "--replacement-seed", "42",
                "--container-cpuset-cpus", "0-7", "--container-cpus", "2"]
        replay_env = {"OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT": "2", "OPENCLAW_REPLAY_TRACE_TOOLS": "1",
                      "OPENCLAW_REPLAY_TRACE_TOOL_SPEED": "4", "PYTHONPATH": f"{REPO}/src:{REPO}"}
        (run / "replay-command.json").write_text(json.dumps({"argv": argv, "environment": replay_env}, indent=1))
        (run / "replay-start-utc.txt").write_text(utc())
        time.sleep(3)
        with (run / "simulate.log").open("x") as log:
            proc = subprocess.Popen(argv, env={**os.environ, **replay_env}, stdout=log, stderr=subprocess.STDOUT)
            t0, budget_sent = time.monotonic(), False
            while proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if not budget_sent and time.monotonic() - t0 >= a.replay_budget_s:
                        (run / "administrative-stop.json").write_text(json.dumps({"reason": f"replay budget {a.replay_budget_s}s", "utc": utc()}))
                        proc.send_signal(signal.SIGINT)
                        budget_sent = True
                    if tunnel.poll() is not None or not program_running():
                        (run / "remote-failure.txt").write_text(f"tunnel_rc={tunnel.poll()} program_running={program_running()} utc={utc()}")
                        proc.terminate()
                        proc.wait(timeout=60)
        rc = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
        (run / "simulate-exit-code").write_text(str(rc))
        (run / "replay-end-utc.txt").write_text(utc())
        remote(f"printf '%s\\n' {rc} > {remote_run}/external-replay-exit-code.tmp && "
               f"mv {remote_run}/external-replay-exit-code.tmp {remote_run}/external-replay-exit-code")
        print("replay exit", rc, flush=True)
    for _ in range(120):
        if host_done():
            break
        if not program_running() and not host_done():
            time.sleep(15)
            if not host_done():
                sys.exit(f"host launcher died without exit-code; see /workspace/{a.name}-launch.log")
        time.sleep(15)
    else:
        sys.exit("host never wrote exit-code")
    host_rc = int(remote(f"cat {remote_run}/exit-code").stdout.strip() or 1)
    (run / "server-exit-code").write_text(str(host_rc))
    subprocess.run(["scp", "-q", "-r", "-P", str(a.port), f"root@{a.host}:{remote_run}", str(run / "server")], check=True)
    if not host_only:
        tunnel.terminate()
    print("host exit", host_rc, "results", run, flush=True)
    return host_rc if host_only else (rc or host_rc)


if __name__ == "__main__":
    sys.exit(main())
