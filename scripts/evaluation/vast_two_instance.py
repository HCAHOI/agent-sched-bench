"""Drive one two-instance serving run on a remote two-GPU host from this machine.

The host runs the engines, proxy and collectors through
scripts/evaluation/run_two_instance_fcfs.sh under supervisor; this machine
replays the agent workload in Docker task containers against the host proxy
over an SSH tunnel, then pulls the host-side run directory into
results/<name>/server/. Every result-affecting flag is passed explicitly and
recorded in results/<name>/replay-command.json and the supervisor conf.

Example (DualMap, mixed56):
  L=".venv/bin/python scripts/evaluation/vast_two_instance.py --host H --port P"
  $L --name dualmap-calibration-r1 --router-policy dualmap --calibrate
  $L --name mixed56-vast-dualmap-r1 --router-policy dualmap --calibration-run dualmap-calibration-r1 --env DUALMAP_CPU_CACHE_GIB=48
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
CONTINUUM_COMMIT = "316a58794a6ff86b216e579b74fd56ed0c5a911f"
# bf16 KV bytes per token: 2 (K,V) x 2 bytes x layers x KV heads x head dim; needed only for --instance-kv-tokens
KV_BYTES_PER_TOKEN = {
    "Qwen/Qwen3-4B-Instruct-2507-FP8": 147_456,  # 36 layers x 8 KV heads x 128
    "Qwen/Qwen3-32B-FP8": 262_144,  # 64 layers x 8 KV heads x 128
}


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--name", required=True, help="run name; becomes results/<name> here and on the host")
    p.add_argument("--router-policy", required=True, choices=["least-requests", "thunderagent", "dualmap", "pd", "ppd", "profile"])
    p.add_argument("--instance-policy", default="fcfs", choices=["fcfs", "continuum"])
    p.add_argument("--task-sticky", action="store_true")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507-FP8", choices=sorted(KV_BYTES_PER_TOKEN),
                   help="served model on every instance and the replay's shadow model")
    p.add_argument("--hf-overrides", metavar="JSON",
                   help="vLLM --hf-overrides for every engine, e.g. YaRN rope scaling to reach --max-model-len")
    p.add_argument("--instances-per-gpu", type=int, default=1, choices=range(1, 5), metavar="K",
                   help="engines per GPU (2K instances); K>1 runs under MPS with a 100/K %% SM share each")
    p.add_argument("--tensor-parallel", type=int, default=1, choices=(1, 2),
                   help="2: one engine across both GPUs (N=1) instead of one engine per GPU; least-requests only")
    p.add_argument("--instance-kv-tokens", type=int, metavar="TOKENS",
                   help="exact per-instance KV budget in tokens (--kv-cache-memory-bytes); omit for the memory-fraction default")
    p.add_argument("--max-model-len", type=int, metavar="TOKENS",
                   help="engine context limit (default 131072); a per-instance KV cap must hold at least one such request")
    p.add_argument("--instance-gpu-memory-utilization", type=float, metavar="FRAC",
                   help="per-instance memory fraction; required when K>1 (default 0.95/K minus headroom is not assumed)")
    p.add_argument("--timeout-s", type=int, default=1800, help="SHADOW_LLM_TIMEOUT_S on both sides")
    p.add_argument("--manifest", default=str(REPO / "analysis/development/mixed56-2l40s-concurrency32-v1/manifest.yaml"))
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for the host launcher (repeatable)")
    p.add_argument("--calibration-run", metavar="NAME",
                   help="dualmap: take DUALMAP_PREFILL_TPOT from results/NAME/server/prefill-calibration.json")
    p.add_argument("--smoke", action="store_true", help="host-side --smoke instead of a workload replay")
    p.add_argument("--calibrate", action="store_true", help="host-side --calibrate (dualmap prefill TPOT); no replay")
    p.add_argument("--profile-lengths", type=Path, metavar="PLAN",
                   help="host-side --profile-lengths with this plan (router-policy profile); no replay")
    p.add_argument("--profile-stage", default="load", choices=["preliminary", "full", "load"])
    p.add_argument("--replay-budget-s", type=int, default=9000, help="SIGINT the replay after this wall time")
    p.add_argument("--tunnel-port", type=int, default=19019)
    p.add_argument("--remote-repo", default="/workspace/agent-sched-bench")
    a = p.parse_args()
    if a.calibration_run:
        calib = json.loads((REPO / "results" / a.calibration_run / "server" / "prefill-calibration.json").read_text())
        a.env.append(f"DUALMAP_PREFILL_TPOT={calib['prefill_tpot']}")
    if a.router_policy == "dualmap" and not a.calibrate and not any(e.startswith("DUALMAP_PREFILL_TPOT=") for e in a.env):
        sys.exit("dualmap needs --calibration-run NAME (from a --calibrate run on this host) or --env DUALMAP_PREFILL_TPOT=...")

    run = REPO / "results" / a.name
    remote_run = f"{a.remote_repo}/results/{a.name}"
    if run.exists():
        sys.exit(f"{run} exists")
    ssh = ["ssh", "-p", str(a.port), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
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
        # driver PTX JIT cache: kernels without native code for the GPU compile once (72 s first request on
        # Blackwell); keep the cache on the persistent disk, not the rental image's ephemeral home
        "CUDA_CACHE_PATH": "/workspace/.nv/ComputeCache", "CUDA_CACHE_MAXSIZE": str(4 << 30),
        "RUNNER_PYTHON": f"{cache}/agent-sched-bench/venvs/continuum-public-{CONTINUUM_COMMIT}/bin/python",
        "DRAM_METRICS": "off", "CONCURRENCY": str(a.concurrency),
        "ROUTER_POLICY": a.router_policy, "INSTANCE_POLICY": a.instance_policy,
        "TASK_STICKY": "1" if a.task_sticky else "0", "SHADOW_LLM_TIMEOUT_S": str(a.timeout_s),
        "SOURCE_BASE_REV": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "PREFILL_CPUSET": "48-50", "DECODE_CPUSET": "51-53", "ROUTER_CPUSET": "54",
        "MANIFEST": f"/workspace/manifests/{a.name}.yaml", "RUN_ROOT": remote_run, "MODEL": a.model,
    }
    if a.hf_overrides:
        json.loads(a.hf_overrides)  # fail here, not in the engine log
        env["VLLM_HF_OVERRIDES"] = a.hf_overrides
    if a.tensor_parallel == 2:
        if a.instances_per_gpu > 1 or a.router_policy != "least-requests":
            sys.exit("--tensor-parallel 2 is a single-instance deployment: --instances-per-gpu 1 and --router-policy least-requests")
        env["TENSOR_PARALLEL"] = "2"
        env["PREFILL_CPUSET"] = "48-53"  # two TP workers plus the API server
    if a.instances_per_gpu > 1:
        if a.instance_gpu_memory_utilization is None or not a.instance_kv_tokens:
            # without the exact cap each engine profiles free memory while its siblings allocate (vLLM asserts on it)
            sys.exit("--instances-per-gpu > 1 needs --instance-gpu-memory-utilization and --instance-kv-tokens")
        n = 2 * a.instances_per_gpu
        env["INSTANCES_PER_GPU"] = str(a.instances_per_gpu)
        env["INSTANCE_CPUSETS"] = ",".join(f"{48 + 3 * i}-{50 + 3 * i}" for i in range(n))  # 3 cores each, then the router
        env["ROUTER_CPUSET"] = str(48 + 3 * n)
    if a.instance_gpu_memory_utilization is not None:
        env["INSTANCE_GPU_MEMORY_UTILIZATION"] = str(a.instance_gpu_memory_utilization)
    if a.max_model_len:
        env["MAX_MODEL_LEN"] = str(a.max_model_len)
    if a.instance_kv_tokens:
        if a.instance_kv_tokens < (a.max_model_len or 131072):
            sys.exit("--instance-kv-tokens must be at least --max-model-len: vLLM refuses a KV cache below one full-length request")
        env["INSTANCE_KV_CACHE_BYTES"] = str(a.instance_kv_tokens * KV_BYTES_PER_TOKEN[a.model])
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    if a.profile_lengths:
        assert a.router_policy == "profile", "--profile-lengths requires --router-policy profile"
        env["LENGTH_PROFILE_PLAN"] = f"/workspace/manifests/{a.name}-plan.md"
        env["LENGTH_PROFILE_STAGE"] = a.profile_stage
    host_only = a.smoke or a.calibrate or bool(a.profile_lengths)
    mode = ("--smoke" if a.smoke else "--calibrate" if a.calibrate
            else "--profile-lengths" if a.profile_lengths else "--external-replay")
    conf = (f"[program:{a.name}]\ndirectory={a.remote_repo}\n"
            f"command=/usr/bin/env {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} "
            f"bash scripts/evaluation/run_two_instance_fcfs.sh {mode}\n"
            "autostart=false\nautorestart=false\nstopasgroup=true\nkillasgroup=true\n"
            f"stdout_logfile=/workspace/{a.name}-launch.log\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n")
    (run / "launch.conf").write_text(conf)
    if a.calibration_run:
        (run / "calibration-source.txt").write_text(a.calibration_run + "\n")
    subprocess.run(["scp", "-q", "-P", str(a.port), a.manifest, f"root@{a.host}:/workspace/manifests/{a.name}.yaml"],
                   check=True) if remote("mkdir -p /workspace/manifests").returncode == 0 else sys.exit("mkdir failed")
    if a.profile_lengths:
        subprocess.run(["scp", "-q", "-P", str(a.port), str(a.profile_lengths),
                        f"root@{a.host}:{env['LENGTH_PROFILE_PLAN']}"], check=True)
        (run / "length-profile-plan.md").write_text(a.profile_lengths.read_text())
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
                "--shadow-llm-api-base", f"http://127.0.0.1:{a.tunnel_port}/v1", "--shadow-llm-model", a.model,
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
    # Host-only modes (calibrate, profile) run for hours; a replay's host side finishes within minutes of it.
    deadline = time.monotonic() + (a.replay_budget_s if host_only else 1800)
    while not host_done():
        if time.monotonic() > deadline:
            sys.exit("host never wrote exit-code")
        if not program_running():
            time.sleep(15)
            if not host_done():
                sys.exit(f"host launcher died without exit-code; see /workspace/{a.name}-launch.log")
        time.sleep(15)
    host_rc = int(remote(f"cat {remote_run}/exit-code").stdout.strip() or 1)
    (run / "server-exit-code").write_text(str(host_rc))
    subprocess.run(["scp", "-q", "-r", "-P", str(a.port), f"root@{a.host}:{remote_run}", str(run / "server")], check=True)
    if not host_only:
        tunnel.terminate()
    print("host exit", host_rc, "results", run, flush=True)
    return host_rc if host_only else (rc or host_rc)


if __name__ == "__main__":
    sys.exit(main())
