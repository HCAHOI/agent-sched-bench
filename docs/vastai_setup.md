# Vast.ai Bootstrap Runbook

> Companion to `.omc/plans/trace-sim-vastai-pipeline.md` (REVISION 3 APPROVED 2026-04-08).
> Phase 3 deliverable. Operator-facing runbook for bringing up a fresh vast.ai A100 instance and running the trace-sim pipeline end-to-end.

---

## Prerequisites

- A vast.ai A100 instance (or any Linux container with FUSE access).
- The vast.ai container is itself running INSIDE Docker — DinD is **not** supported. We use rootless podman + a `DOCKER_HOST` socket shim instead.
- Python 3.13 (per project `pyproject.toml`).
- Conda env named `ML` (per `feedback_conda_env.md` memory and CLAUDE.md project conventions).

---

## Step 1 — Clone the repo + install Python deps

```bash
git clone https://github.com/HCAHOI/agent-sched-bench.git
cd agent-sched-bench
git checkout dev/trace-sim-vastai

# Activate the ML conda env (matches project convention)
conda activate ML

# Install editable + test extras
pip install -e .
```

---

## Step 2 — Install podman + run preflights (Phase 3)

```bash
bash scripts/setup/install_podman_vastai.sh
```

**What this script does:**

1. Installs `podman`, `fuse-overlayfs`, `slirp4netns`, `uidmap` via `apt-get` (idempotent — re-running on a configured host is a no-op).
2. **Pre-mortem A item 1:** verifies `/dev/fuse` exists OR `modprobe fuse` succeeds. If neither, fails loudly because podman would otherwise fall back to the `vfs` storage driver — making SWE-Bench image pulls take ~45 minutes and consuming 3x disk.
3. **Pre-mortem A item 2:** verifies the current user has entries in `/etc/subuid` and `/etc/subgid`. If missing, prints the exact `usermod --add-subuids` remediation command and exits without auto-running it (auto-running would require root and can affect other tenants on shared hosts).
4. Smoke-pulls `hello-world` from Docker Hub via podman to validate the storage driver and registry connectivity.
5. Asserts the storage driver is `overlay` (not `vfs`); fails loudly if not.
6. Saves `podman info` output to `.omc/logs/phase3-podman-info.log` for debugging.

**If the preflight fails on /dev/fuse:** ask vast.ai support for a template with FUSE enabled, OR launch the container with `--device /dev/fuse`. **Do not proceed with `vfs` storage.**

**If the preflight fails on /etc/subuid:** run the printed `usermod` command as root. On vast.ai, you may need to switch to a root-capable instance template.

---

## Step 3 — Start the podman system service (Phase 3)

```bash
bash scripts/setup/start_podman_socket.sh
# Or, to capture the DOCKER_HOST export into the parent shell:
eval "$(bash scripts/setup/start_podman_socket.sh --print-export)"
```

**What this script does:**

1. Starts `podman system service --time=0` in the background. The `--time=0` flag disables idle shutdown so the socket survives long-running smoke runs (Pre-mortem A item 3).
2. Persists the service PID to `.omc/state/podman.pid` so subsequent invocations can liveness-check + restart if needed.
3. Waits up to 5 seconds for the Unix socket at `/run/user/$(id -u)/podman/podman.sock` to appear.
4. Exports `DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock` so the upstream SWE-Bench harness's `docker.from_env()` calls transparently route to podman.
5. Verifies the docker Python SDK can ping the socket (`docker.from_env().ping() == True`).

**Manual export** (if you didn't use the `--print-export` form):

```bash
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

---

## Step 4 — Install Playwright (Phase 5 prerequisite, optional for Phase 3)

The Phase 5 Gantt browser smoke test uses Playwright with Chromium. To pre-install just chromium (not the full browser matrix — saves ~300MB):

```bash
playwright install chromium
```

**Important:** use `playwright install chromium`, NOT `playwright install`. The unqualified form downloads chromium + firefox + webkit (~450MB); chromium-only is ~150MB. See Phase 5 CI requirements section in `docs/plans/trace-sim-vastai-pipeline.md` for the GitHub Actions cache spec.

---

## Step 5 — Download datasets + clone task repos

```bash
# SWE-bench Verified
bash scripts/setup/swebench_data.sh

# SWE-rebench (Nebius)
bash scripts/setup/swe_rebench_data.sh

# Pre-clone repos (avoids network during eval)
bash scripts/setup/clone_repos.sh
```

---

## Step 6 — Run a smoke collect (1 instance)

```bash
# mini-swe-agent on swe-rebench (no MCP needed; --mcp-config is NOT required)
make smoke-swe-rebench-miniswe SMOKE_N=1

# OpenClaw on swe-rebench WITH context7 MCP — Phase 4 makes --mcp-config
# MANDATORY for openclaw runs. The CLI exits code 2 with the exact error
# message if you forget the flag (per Driver 3 of the
# trace-sim-vastai-pipeline plan: "OpenClaw realism delta requires real
# MCP events — without them the comparison is benchmark gaming by
# omission"). Opt-out for a baseline measurement is the affirmative
# literal `--mcp-config none`.
export CONTEXT7_API_KEY=...   # required for context7 server access
python -m trace_collect.cli \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config configs/mcp/context7.yaml \
    --sample 1
```

---

## Step 7 — Run the simulator (with vLLM)

Assumes a local vLLM A100 instance is running on `localhost:8000`:

```bash
# Pick a recently collected trace
SOURCE_TRACE=traces/swe-rebench/qwen-plus-latest/$(ls -1 traces/swe-rebench/qwen-plus-latest/ | tail -1)/<instance>.jsonl

python -m trace_collect.cli simulate \
    --source-trace "$SOURCE_TRACE" \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --metrics-url http://localhost:8000/metrics \
    --output-dir traces/simulate
```

---

## Manual smoke verification (post-Ralph)

> **CRITICAL — read this first:** These deferred items require infrastructure access (vast.ai A100, cloud LLM API keys, real MCP server endpoints, a long-running vLLM instance, FUSE-enabled host kernel). They are **NOT verifiable in a local Ralph loop**. After the Ralph loop completes the code deliverables in `dev/trace-sim-vastai`, the operator **MUST** run these manually before declaring the plan production-ready.
>
> Each item maps 1:1 to a US-NNN story in `.omc/prd.json`. The Ralph loop's PRD acceptance criteria explicitly mark these as `DEFERRED to manual smoke (US-010)` so the local loop can complete with `passes: true` on all stories without needing infrastructure that doesn't exist in a developer laptop.
>
> **Source of truth for what's deferred:** the `DEFERRED to manual smoke (US-010)` annotations in `.omc/prd.json` are the canonical list. The sections (a)-(f) below are the operator-facing runbook for executing each one.

### Quick reference: what runs locally vs what's deferred

| Story | Local Ralph loop | Deferred to this runbook |
|---|---|---|
| US-001 Phase 0 schema audit | docs only — fully verifiable | — |
| US-002 Phase 1 mini-swe registry | unit tests + ruff | byte-identical regression on real fixture (manual) |
| US-003 Phase 2 vLLM metrics merge | unit tests verify dataclass + delta logic | **(a)** real localhost vLLM smoke |
| US-004 Phase 1.5.0 design audit | docs only | empirical warmup probe (Q4) — needs real fixtures |
| US-005 Phase 1.5.1 openclaw simulate | unit tests verify MCP-reuse + warmup default | **(b)** TOOL 1:1 diff vs real Gate-B fixture |
| US-006 Phase 3 podman scripts | bash syntax check | **(c)** fresh A100 instance e2e |
| US-007 Phase 4 MCP enablement | flag enforcement + kwarg passthrough subprocess | **(d)** real cloud LLM + real context7 server |
| US-008 Phase 5 Gantt extension | DOM-payload regression + parity test | **(e)** Playwright Chromium browser smoke |
| US-009 Phase 6 BFCL refusal | unit + smoke matrix BFCL cell (always runs) | **(f)** full 4-cell matrix on real A100 |

The **BFCL v4 refusal cell** in `scripts/smoke_full_matrix.sh` is the one piece of Phase 6 that runs locally without infrastructure — it's the canary that catches Phase 6 regressions in any environment.

### (a) Phase 2 — vLLM scheduler metrics integration smoke

**Prereq:** local vLLM A100 instance running on `localhost:8000` with the scheduler hook applied (`python -m harness.scheduler_hooks --metrics-url http://localhost:8000/metrics --output /tmp/baseline.json`).

```bash
python -m trace_collect.cli simulate \
    --source-trace tests/fixtures/miniswe_swebench_sample.jsonl \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --metrics-url http://localhost:8000/metrics \
    --output-dir /tmp/sim-test

# Verify all 5 PreemptionSnapshot fields are present
jq '.data.sim_metrics.vllm_scheduler_snapshot | keys' /tmp/sim-test/*.jsonl | sort -u
# Expected: ["cpu_cache_usage_perc", "cpu_prefix_cache_hit_rate", "gpu_cache_usage_perc", "gpu_prefix_cache_hit_rate", "num_preemptions_total"]
```

### (b) Phase 1.5.1 — OpenClaw simulate integration smoke (TOOL 1:1 diff)

**Prereq:** a Gate-B-clean Phase 4 fixture. Produce one by running:

```bash
python -m trace_collect.cli collect \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config configs/mcp/context7.yaml \
    --instances 1
```

Then point the simulator at the produced trace:

```bash
SOURCE=$(ls -1t traces/swe-rebench/qwen-plus-latest/*/*/0001.jsonl | head -1)
python -m trace_collect.cli simulate \
    --source-trace "$SOURCE" \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --metrics-url http://localhost:8000/metrics

# Verify TOOL-category 1:1 diff between collect and simulate traces
jq -r 'select(.action_type=="tool_exec") | .data.tool_name' "$SOURCE" > /tmp/collect_tools.txt
jq -r 'select(.action_type=="tool_exec") | .data.tool_name' /tmp/sim-test/*.jsonl > /tmp/sim_tools.txt
diff /tmp/collect_tools.txt /tmp/sim_tools.txt || echo "TOOL sequences diverged — investigate"
```

### (c) Phase 3 — podman bootstrap end-to-end smoke

```bash
# On a FRESH vast.ai A100 instance (not one where install was already run):
git clone https://github.com/HCAHOI/agent-sched-bench.git
cd agent-sched-bench
bash scripts/full_setup.sh "$HF_TOKEN"

# Verify the storage driver and the docker SDK shim work
podman info | grep storage.driver  # must be 'overlay'
python -c "import docker; print(docker.from_env().ping())"  # must print True

# Run a 1-instance harness eval to confirm patch verdict reachable
python -m trace_collect.cli collect \
    --benchmark swe-rebench \
    --scaffold mini-swe \
    --instances 1 \
    --evaluate
```

### (d) Phase 4 — MCP enablement live test

**Prereq:** valid `OPENROUTER_API_KEY` or `DASHSCOPE_API_KEY` set in the environment, and access to `https://mcp.context7.com/mcp` with a `CONTEXT7_API_KEY`.

```bash
export DASHSCOPE_API_KEY=...
export CONTEXT7_API_KEY=...

python -m trace_collect.cli collect \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config configs/mcp/context7.yaml \
    --instances 1

# Verify ≥1 MCP event in the produced trace
jq 'select(.category=="MCP")' traces/swe-rebench/*/*/*.jsonl | head -20
# Verify the run_config metadata captured the MCP config identity
jq 'select(.type=="trace_metadata") | .run_config' traces/swe-rebench/*/*/*.jsonl
```

### (e) Phase 5 — Playwright browser smoke test

**Prereq:** `playwright install chromium` already run (Step 4 above). Cache `~/.cache/ms-playwright` in CI per the GitHub Actions block in `docs/plans/trace-sim-vastai-pipeline.md` Phase 5 CI section.

```bash
conda run -n ML python -m pytest tests/test_gantt_browser_smoke.py -v
```

### (f) Phase 6 — Full 4-cell matrix smoke

```bash
bash scripts/smoke_full_matrix.sh
```

The script runs all four matrix cells (mini-swe × swe-bench, mini-swe × swe-rebench, openclaw × swe-bench + context7, openclaw × swe-rebench + context7) plus the BFCL v4 refusal cell. Cells whose preconditions are not met (e.g. no vast.ai, no real cloud LLM key) report `SKIPPED: <reason>` on stdout. The BFCL refusal cell always runs.

---

## Troubleshooting

### Pre-mortem A scenarios

**1. `modprobe fuse: not found` or `/dev/fuse: No such file or directory`**

The container's kernel does not expose FUSE. Resolution:
- Pick a vast.ai instance template that exposes FUSE (look for `--device /dev/fuse` or "FUSE-enabled" in the template description).
- Launch the container with `--device /dev/fuse --cap-add SYS_ADMIN`.
- If neither is possible, the trace-sim-vastai pipeline cannot run on this host. Switch hosts.

**2. `usermod: command not found` or `/etc/subuid: Permission denied`**

The container does not have a writable `/etc/subuid` or you're not root. Resolution:
- If you have root, run the printed `usermod` command.
- If not, your provider must run it for you OR launch a root-capable instance.

**3. `podman system service` exited unexpectedly mid-run**

The PID stored in `.omc/state/podman.pid` is dead. Re-run `bash scripts/setup/start_podman_socket.sh` — it will detect the stale PID and restart. The smoke matrix script (`scripts/smoke_full_matrix.sh`, when implemented in US-009) liveness-checks before each batch.

### Other gotchas

- **`docker.from_env().ping()` returns False with `DOCKER_HOST` set**: the docker SDK and podman socket protocols are mostly compatible but not 100%. Verify with `curl --unix-socket $XDG_RUNTIME_DIR/podman/podman.sock http://localhost/_ping` → should return `OK`.
- **Image pulls are catastrophically slow**: confirm `podman info | grep storage.driver` is `overlay`, not `vfs`. If `vfs`, FUSE setup is broken — go back to Pre-mortem A item 1.
- **`pip install -e .` fails with `lxml` build errors**: `apt-get install -y libxml2-dev libxslt1-dev` then retry.
