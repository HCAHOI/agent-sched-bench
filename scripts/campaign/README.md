# `run_campaign.sh` — repo-resident GPU recording-campaign runner

A small, reviewable bash runner that executes a recording campaign as a
sequential queue of `trace_collect.cli` invocations, with **PID-file-based
process identity** instead of process-table string matching. It replaces the
ad-hoc `ssh 'cat > script.sh << EOF … EOF; nohup ./script.sh & disown'` queues
whose `pgrep -f`-style management caused three near-disasters (parent-bash
killed by a cmdline match, a sentinel loop matching the deployer heredoc, a
cleanup matching the nohup parent).

> Scope: this runner is for **future** deployments. It does not migrate or touch
> a queue already running on the box (a pre-existing queue has no PID/meta files
> and cannot be verify-waited on — see "Serializing behind an existing run").

---

## The load-bearing safety rule

**The runner NEVER searches the process table by string.** No `pgrep`, no
`pkill`, no grep over `ps`/cmdline to *find* a process to signal. The only way a
PID is ever signalled is after `verify_pid` returns `alive-and-ours` for a PID
read from a PID file whose paired `.meta` sidecar matches identity
(start-time + boot_id + cmdline anchor). Enforced by a test:

```bash
grep -nE 'pgrep|pkill' scripts/campaign/run_campaign.sh   # must print nothing
```

---

## Task-list format (TSV)

One row = one `python -m trace_collect.cli` invocation. Tab-separated, three
columns. Lines starting with `#` and blank lines are skipped.

```
task_id <TAB> benchmark <TAB> extra_cli_args
```

- `task_id` — human label used for `logs/<task_id>.log` and PID identity
  (e.g. `A_max__beeware`).
- `benchmark` — plugin slug (`swe-rebench` | `terminal-bench`). The runner
  **appends** `--benchmark <benchmark>`; do NOT put `--benchmark` in
  `extra_cli_args`.
- `extra_cli_args` — everything after `python -m trace_collect.cli` EXCEPT
  `--benchmark`, transcribed from the arm's `invocation:` block in
  `configs/recording_campaigns/agent_event_kv_step_level.yaml`.

The runner prepends `python -m ${TRACE_CLI_MODULE:-trace_collect.cli}` (override
the interpreter with `PYTHON_BIN`) and appends `--benchmark <benchmark>`. The
campaign manifest stays the human-authored ledger; transcribing its rows into a
TSV is a deliberate review step, not hidden magic (there is no `yq` on the box).

See `tasks.example.tsv` for Arm A/B/C row shapes. Verify any TSV's expansion
before deploying:

```bash
bash scripts/campaign/run_campaign.sh run --tasks scripts/campaign/tasks.example.tsv --print-cmd
```

`--print-cmd` is a dry run: it prints `task_id <TAB> <full expanded command>`
for every row and exits without launching anything. The expanded command
reproduces the manifest's `invocation` tokens verbatim.

---

## Subcommands

### `run`
```
run_campaign.sh run --tasks <tsv> [--campaign <name>] [--run-dir <dir>]
                    [--timeout <secs, default 21600>]
                    [--sentinel-pidfile <path>] [--sentinel-metafile <path>]
                    [--print-cmd]
```
Runs each TSV row sequentially under `timeout --signal=TERM --kill-after=60`
(60s grace lets a hung task flush recording data before SIGKILL). Per task it
writes `logs/<task_id>.log`, maintains `current_task.pid`/`.meta`, captures the
exit code (124 = timed out), and updates `status.json` atomically. **Per-task
isolation:** a nonzero or timed-out task is logged and the queue **continues**.
Writes `campaign.pid`/`.meta` at start and refuses to start into a run dir whose
`campaign.pid` still verifies as a live queue (double-queue guard).

### `status`
```
run_campaign.sh status [--run-dir <dir>]
```
Resolves `campaign.pid` and `current_task.pid` through `verify_pid` and prints
queue state, progress, current task, and the last results from `status.json`.
If `status.json` says `running` but `current_task.pid` does not verify, it
prints `STALE: current_task.pid does not match a live process`. With no
`--run-dir`, it uses the newest run under `.omc/campaign_runs/`.

### `stop`
```
run_campaign.sh stop [--run-dir <dir>] [--now]
```
- Default (graceful): drops a `stop_requested` flag the `run` loop checks
  **between** tasks → the in-flight task finishes, then the queue exits with
  `state:"stopped"`.
- `--now`: `verify_pid` the current-task PID; if `alive-and-ours`, `kill -TERM`
  that **single verified PID** (escalating to `-KILL` after the grace window),
  then signal the verified queue PID. If verification fails, it **refuses** and
  exits nonzero — it never falls back to string matching.

---

## Directory layout (per run)

```
${RUN_DIR}/                 # default: .omc/campaign_runs/<campaign>-<UTC-timestamp>/
  campaign.pid              # queue identity (PID)
  campaign.meta             # queue start_time, boot_id, anchor (runner abs path)
  current_task.pid          # current task PID
  current_task.meta         # task start_time, boot_id, cmdline anchor
  status.json               # machine-readable status (atomic tmp+mv writes)
  queue.log                 # runner-level orchestration log
  logs/<task_id>.log        # per-task stdout+stderr
```

PID/meta/status files live beside the logs under `.omc/` (box-local, survives
across operator ssh sessions), never in `/tmp`.

---

## PID-file / `/proc` protocol

A PID file alone is unsafe (PID reuse). Every PID file is paired with a `.meta`
sidecar, and every wait/stop/status operation resolves a PID through
`verify_pid` before acting. `verify_pid(pidfile, metafile)` →
`alive-and-ours | dead | reused`:

1. Read PID from `pidfile`; missing → `dead`.
2. `kill -0 <pid>` (no-op signal); fails → process gone → `dead` (stale file).
3. Compare `/proc/<pid>/stat` **field 22** (start-time, clock-ticks-since-boot)
   and the recorded `boot_id` (`/proc/sys/kernel/random/boot_id`) to `.meta`.
   Field 22 is parsed from **after the last `)`** because the `comm` field can
   contain parens/spaces. Any mismatch → PID reused or host rebooted → `dead`
   (never signalled).
4. Require `/proc/<pid>/cmdline` to **contain the recorded anchor**: the
   runner's absolute path for `campaign.pid`; the CLI module string
   (`trace_collect.cli`, present on `python -m trace_collect.cli` and far more
   specific than bare `python`) for `current_task.pid`. The full task argv —
   including `--instance-ids` — is also on that cmdline. No match → `reused` →
   treated as `dead`. The anchor deliberately omits the per-task `--instance-ids`
   value: the runner is strictly sequential (one `current_task.pid` ever exists),
   so start_time + boot_id already pin the exact process, and the module-string
   anchor only needs to reject a recycled PID running something *else*.

`boot_id` matters because this is a Proxmox VM that pauses/clock-freezes: field
22 (ticks-since-boot) is stable across a pause/resume, and `boot_id` changes only
on an actual reboot — so we distinguish "same boot, PID still valid" from
"rebooted, all PIDs stale" without trusting wall-clock arithmetic across a
freeze. `status` never infers "dead" from a stale `updated_epoch` gap; liveness
is always `verify_pid`.

On the GPU box (Linux) the runner reads `/proc` directly. On a `/proc`-less host
(macOS, used only for the local test harness) it derives the same identity facts
via `ps` / `sysctl kern.boottime` — never mocked.

---

## Safe deployment recipe

Never put the script body in a process cmdline. Ship the runner + task list as
**committed git files** (not rsync/scp/heredoc), then launch detached so only
`bash <path>` (not heredoc text) is in any cmdline:

```bash
# 1. Commit + push the runner and the campaign's task TSV from the dev repo.
#    (Task ids live in the TSV/config, never in src/ — per CLAUDE.md.)
git add scripts/campaign/run_campaign.sh scripts/campaign/tasks.armC.tsv
git commit -m "[config] armC campaign task list"
git push

# 2. Pull on the box (clean checkout; untracked old run_campaign_queue_v*.sh
#    do not conflict with a fast-forward):
ssh Ubuntu@<box> 'cd ~/agent-sched-bench && git pull --ff-only'

# 3. Launch detached; the launch shell's cmdline is just paths:
ssh Ubuntu@<box> 'cd ~/agent-sched-bench && \
  nohup bash scripts/campaign/run_campaign.sh run \
    --tasks scripts/campaign/tasks.armC.tsv --campaign armC \
    > /dev/null 2>&1 & disown'

# 4. Operate via subcommands (file-based, no string matching):
ssh Ubuntu@<box> 'cd ~/agent-sched-bench && bash scripts/campaign/run_campaign.sh status'
ssh Ubuntu@<box> 'cd ~/agent-sched-bench && bash scripts/campaign/run_campaign.sh stop'        # graceful
ssh Ubuntu@<box> 'cd ~/agent-sched-bench && bash scripts/campaign/run_campaign.sh stop --now'  # TERM verified PID
```

Even the launch shell's cmdline contains only paths — not task names or sentinel
literals — so no future `pgrep`-style accident can match task strings against the
launcher. (The runner itself never greps the process table regardless.)

---

## Serializing behind an existing run (sentinel wait)

To start a new queue only after another process finishes, pass a sentinel
PID/meta pair. `run` polls `verify_pid` on it and blocks until it is `dead`
(it never string-matches, so it cannot hang on a reused PID or the deployer
shell):

```bash
run_campaign.sh run --tasks <tsv> \
  --sentinel-pidfile <path> --sentinel-metafile <path>
```

A pre-existing queue with no PID/meta files cannot be verify-waited on. If you
must serialize behind one, build a sentinel pair by hand from its live PID:

```bash
PID=<the running pid>
echo "$PID" > sentinel.pid
{
  printf 'pid=%s\n' "$PID"
  printf 'start_time=%s\n' "$(awk '{ s=$0; sub(/.*\) /,"",s); split(s,a," "); print a[20] }' /proc/$PID/stat)"
  printf 'boot_id=%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
  printf 'anchor=%s\n' "trace_collect"   # a substring known to be on its cmdline
} > sentinel.meta
```

Still PID-verified, never string-matched.

---

## Running the test harness

GPU-free, no torch, no `trace_collect` import (dummy `sleep`/`false` tasks via
the `TRACE_CLI_CMD` override). Prints one PASS/FAIL line per checked behavior:

```bash
bash tests/test_campaign_runner.sh
```
