# Cloud Codex Handoff — SWE-rebench OpenClaw smoke with image prefetch

## Goal

Continue on the cloud Linux server (not local macOS) and do **two things**:

1. Implement a **disk-safe serial runner behavior** for SWE-rebench/OpenClaw smoke runs:
   - run tasks **serially**
   - while task _i_ is running, **prefetch** the image for task _i+1_
   - after task _i_ finishes and artifacts are safely written, clean up image state so disk usage does not grow without bound
2. Run a **3-task smoke test** with:
   - scaffold: `openclaw`
   - provider/API key: `DASHSCOPE_API_KEY`
   - model: `qwen-plus-latest`
   - benchmark: `swe-rebench`
   - tasks chosen from the existing Claude Code Haiku trace set
   - success criterion: produce **canonical standard traces** (v5 trace JSONL) in this repo’s normal attempt layout

---

## Important context

- The server environment is already prepared.
- The local macOS+Podman smoke issues are **not authoritative**; Linux server behavior is what matters now.
- Disk budget is only about **150 GB**, so **do not** keep accumulating per-task source images + fixed images forever.
- Prefer the simplest robust implementation over backward compatibility.

---

## Repository state to start from

Start from the latest pushed `origin/main`.

At the time of handoff, `origin/main` contains the last pushed merge commit from the prior phase, but **the newer local exploratory changes were not pushed**. So on the server, treat `origin/main` as the clean base and continue from there.

Recommended:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feat/cloud-openclaw-prefetch-smoke
```

---

## Exact smoke-test tasks

Choose these **3 tasks** from the Claude Code Haiku trace set:

1. `encode__httpx-2701`
2. `Kinto__kinto-http.py-384`
3. `mozilla__bleach-259`

Reference Claude Code trace locations already present in this repo:

```text
traces/swe-rebench/claude-code-haiku/encode__httpx-2701/attempt_1/trace.jsonl
traces/swe-rebench/claude-code-haiku/Kinto__kinto-http.py-384/attempt_1/trace.jsonl
traces/swe-rebench/claude-code-haiku/mozilla__bleach-259/attempt_1/trace.jsonl
```

These are the tasks to use for the serial OpenClaw smoke run.

---

## What to implement

### A. Serial image lifecycle with one-step-ahead prefetch

Desired policy:

- Before task 1: ensure image 1 is present
- While task 1 is running: **background prefetch image 2**
- After task 1 completes and artifacts are written:
  - stop/remove task container
  - remove task 1 fixed image
  - remove task 1 source image **unless it is also the next task image that is intentionally being kept**
  - prune dangling layers if needed
- Then task 2 starts, while task 2 runs prefetch image 3
- Repeat

### B. Keep disk bounded

Because disk is only ~150 GB, the intended invariant is:

- at most the **current task image/fixed image**
- plus at most the **next prefetched source image**

Avoid retaining a long tail of old images.

### C. Smoke with current benchmark defaults

The smoke run should validate that:

- `swe-rebench` uses its configured default prompt behavior (do not force a prompt override unless debugging)
- OpenClaw can run these 3 tasks serially
- each task produces a **canonical standard trace** under the normal attempt layout

---

## Smoke command target

After implementation, run something equivalent to:

```bash
conda run -n ML env PYTHONPATH=src:. python -m trace_collect.cli \
  --provider dashscope \
  --model qwen-plus-latest \
  --benchmark swe-rebench \
  --scaffold openclaw \
  --container docker \
  --mcp-config none \
  --instance-ids encode__httpx-2701,Kinto__kinto-http.py-384,mozilla__bleach-259 \
  --run-id traces/swe-rebench/qwen-plus-latest/smoke-openclaw-cc-parity-3tasks \
  --verbose
```

Notes:
- Keep it **serial**.
- If you add a dedicated serial-prefetch path or runner helper, wire this smoke through that path.
- The smoke objective is not metrics quality; it is **whether canonical traces land correctly**.

---

## Expected output layout

The smoke is successful if you get:

```text
traces/swe-rebench/qwen-plus-latest/smoke-openclaw-cc-parity-3tasks/
  encode__httpx-2701/attempt_1/trace.jsonl
  Kinto__kinto-http.py-384/attempt_1/trace.jsonl
  mozilla__bleach-259/attempt_1/trace.jsonl
```

And those traces should be canonical trace JSONL, i.e. first record should be something like:

- `type = trace_metadata`
- `trace_format_version = 5`
- `scaffold = openclaw`

Also verify the rest of the normal attempt artifacts exist where expected.

---

## Minimum acceptance checks

1. All 3 tasks run **serially**, not in parallel
2. Next task image is prefetched while current task runs
3. Old images are cleaned so disk does not monotonically grow without bound
4. All 3 tasks produce canonical `attempt_1/trace.jsonl`
5. No non-canonical Claude-style raw session JSONL is being mistaken for the standard trace output

---

## Recommended validation commands

Check traces exist:

```bash
find traces/swe-rebench/qwen-plus-latest/smoke-openclaw-cc-parity-3tasks -name trace.jsonl
```

Check first line is canonical metadata:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('traces/swe-rebench/qwen-plus-latest/smoke-openclaw-cc-parity-3tasks')
for p in sorted(root.glob('*/attempt_1/trace.jsonl')):
    first = json.loads(p.read_text(encoding='utf-8').splitlines()[0])
    print(p)
    print(first.get('type'), first.get('trace_format_version'), first.get('scaffold'))
PY
```

Check image cleanup behavior during/after run with the explicit runtime you selected:

```bash
docker images
docker system df
```

---

## Implementation guidance

- Favor the simplest serial orchestrator possible.
- If reusing ideas from `../agentcgroup/scripts/run_all_swebench_images.py`, keep the implementation idiomatic to this repo rather than copying unrelated batch machinery wholesale.
- It is fine if prefetch is just a background `docker pull` or `podman pull`
  for the next image and a join/wait before the next task starts.
- Prefer correctness and bounded disk usage over fancy scheduling.

---

## Deliverables expected from cloud Codex

1. Code changes for serial image cleanup + one-step prefetch
2. A brief summary of what changed
3. Exact smoke command used
4. Result summary for the 3 tasks
5. Confirmation whether canonical traces were successfully pulled down
