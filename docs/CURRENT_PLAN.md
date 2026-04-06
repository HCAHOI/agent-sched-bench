# Plan: OpenClaw Standalone CLI

## RALPLAN-DR Summary

### Principles
1. **Reuse over reinvent** — Build on `process_direct()`, `SessionManager`, `TraceCollectorHook`; no new frameworks
2. **Thin CLI, thick library** — `__main__.py` is pure argument parsing + wiring; all logic stays in existing modules
3. **Session-first design** — Every run gets a session ID; sync/async are just different ways to observe the same session
4. **No new dependencies** — argparse, subprocess, os, sys only
5. **Streaming by default** — Real-time feedback in sync mode; users see what the agent is doing

### Decision Drivers
1. **`process_direct()` exists** — AgentLoop already has a convenience method that accepts content + session_key + streaming callbacks, bypassing MessageBus/ResultCollector entirely
2. **`SessionManager` persists per-workspace** — Sessions stored as JSONL under `{workspace}/sessions/` (not `.openclaw/sessions/`). Keyed by `channel:chat_id` (colons replaced by `_` via `safe_filename`). Already supports resume.
3. **`TraceCollectorHook` is self-contained** — Writes JSONL traces with step/event/summary records. Can be passed as a hook to AgentLoop.

### Viable Options

**Option A: `process_direct()` path** (Recommended)
- Wire `__main__.py` → UnifiedProvider → AgentLoop → `process_direct()`
- Sync: attach a `CLIStreamHook` that prints events to stderr
- Async: re-invoke self with `--_daemon` flag via `subprocess.Popen()`
- Session resume: same AgentLoop + SessionManager, new `process_direct()` call
- Pros: Simple (~200 lines `__main__.py` + ~80 lines hook), uses proven code path
- Cons: `process_direct()` is less tested than the full bus path

**Option B: Full bus path (SWEBenchRunner-style)**
- Wire `__main__.py` → custom `StandaloneRunner` mimicking `SWEBenchRunner`
- Uses MessageBus + ResultCollector + AgentLoop.run()
- Pros: Battle-tested path, identical to eval mode
- Cons: ~2x code, unnecessary complexity for single-task CLI, ResultCollector adds indirection

**Invalidation of Option B**: The bus path is designed for multi-tenant scheduling (multiple concurrent sessions via message routing). A CLI runs one prompt at a time. `process_direct()` was specifically added as the single-task convenience path.

### Recommended: Option A

---

## Implementation Plan

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `src/agents/openclaw/__main__.py` | ~200 | CLI entry point: argparse, provider setup, agent loop, sync/async dispatch |
| `src/agents/openclaw/cli_hook.py` | ~80 | `CLIStreamHook(AgentHook)` — prints real-time events to stderr |

### Files to Modify

| File | Change |
|------|--------|
| `src/agents/openclaw/_loop.py` | No changes needed — `process_direct()` already exists |
| `src/agents/openclaw/eval/runner.py` | Extract `_inject_event_callbacks` to module-level function for reuse |
| `pyproject.toml` | Add `[project.scripts]` entry: `openclaw = "agents.openclaw.__main__:main"` |

### 1. CLI Argument Spec (`__main__.py`)

```
python -m agents.openclaw [OPTIONS]

Required (one of):
  --prompt TEXT          Task prompt for the agent
  --session-id ID       Resume/append to existing session
  --status              Show session status (requires --session-id)

Workspace:
  --workspace PATH      Working directory for the agent (default: cwd)

Model:
  --model NAME          Model identifier (default: env OPENCLAW_MODEL or "qwen/qwen3.6-plus:free")
  --api-base URL        OpenAI-compatible API base (default: env OPENCLAW_API_BASE or "https://openrouter.ai/api/v1")
  --api-key KEY         API key (default: env OPENROUTER_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY)

Execution:
  --async               Return session ID immediately, run in background
  --max-iterations N    Max agent iterations (default: 200)
  --max-tokens N        Max tokens per LLM call (default: 8192)
  --temperature FLOAT   Sampling temperature (default: 0.1)

Internal:
  --_daemon SID         (Hidden) Run as daemon for session SID
```

### 2. Sync Mode Event Stream Format

Printed to stderr so stdout remains clean for piping:

```
[oc-a1b2c3] 10:32:01 START    prompt="Create a Python Tetris game with pygame"
[oc-a1b2c3] 10:32:01 LLM      tokens=150+420 lat=2.3s
[oc-a1b2c3] 10:32:04 TOOL     exec command="mkdir -p src && touch src/tetris.py"
[oc-a1b2c3] 10:32:04 TOOL_OK  exec dur=0.1s
[oc-a1b2c3] 10:32:04 LLM      tokens=570+890 lat=4.1s
[oc-a1b2c3] 10:32:08 TOOL     write_file path="src/tetris.py" (2.3KB)
[oc-a1b2c3] 10:32:08 TOOL_OK  write_file dur=0.0s
...
[oc-a1b2c3] 10:35:22 DONE     steps=12 elapsed=203s tokens=15420 tools=8/8ok
```

Format: `[{session_short}] {HH:MM:SS} {EVENT_TYPE:<8} {details}`

### 3. `CLIStreamHook` Design (`cli_hook.py`)

```python
class CLIStreamHook(AgentHook):
    """Prints real-time agent events to stderr for CLI observation."""

    def __init__(self, session_id: str, *, quiet: bool = False):
        self.sid_short = session_id[:8]
        self.quiet = quiet
        self._iter_start = 0.0
        self._n_steps = 0
        self._total_tokens = 0
        self._tool_ok = 0
        self._tool_fail = 0
        self._wall_start = time.monotonic()

    async def before_iteration(self, ctx): ...  # record iter start
    async def before_execute_tools(self, ctx): ...  # print LLM stats
    async def after_iteration(self, ctx): ...  # print tool results
    def print_summary(self): ...  # final DONE line
```

Hooks used: `before_iteration`, `before_execute_tools`, `after_iteration`
Output target: `sys.stderr` (not stdout)

### 4. Session Persistence Design

- Session ID format: `oc-{8 hex chars}` (e.g., `oc-a1b2c3d4`)
- Generated from `uuid.uuid4().hex[:8]` on first run
- SessionManager key: `cli:{session_id}` (channel=cli, chat_id=session_id)
- Session file: `{workspace}/sessions/cli_oc-{hex}.jsonl` (colons replaced by `_` via `safe_filename`)
- Trace file: `{workspace}/.openclaw/traces/{session_id}.jsonl`
- `--session-id oc-xxx --prompt "new prompt"` → same AgentLoop, same session_key, calls `process_direct()` again with new content

### 5. Async Mode Design (`--async`)

```
User runs:
  python -m agents.openclaw --prompt "Build tetris" --workspace ~/tetris --async

CLI does:
  1. Generate session_id = "oc-a1b2c3d4"
  2. Print to stdout: "oc-a1b2c3d4"
  3. Spawn: subprocess.Popen(
       [sys.executable, "-m", "agents.openclaw",
        "--_daemon", session_id,
        "--prompt", prompt,
        "--workspace", workspace,
        "--model", model,
        "--api-base", api_base,
        "--api-key", api_key],
       start_new_session=True,
       stdout=open(workspace/.openclaw/logs/{sid}.log, "w"),
       stderr=subprocess.STDOUT,
     )
  4. Write PID to workspace/.openclaw/pids/{session_id}.pid
  5. Exit 0

--_daemon flag:
  - Runs the same sync path but with quiet=True on CLIStreamHook
  - On completion, removes PID file
  - Logs go to .openclaw/logs/{sid}.log
```

`--status` with `--session-id`:
- Check PID file exists → if yes, verify process alive via `os.kill(pid, 0)`
- If PID alive → status=running; if PID stale → status=crashed (clean up PID file)
- If no PID file → status=completed (check trace for summary record)
- Read last line of trace JSONL for summary
- Print: session_id, status, steps, elapsed, workspace

### 6. `__main__.py` Flow

```python
def main():
    args = parse_args()

    if args.status:
        return _show_status(args)

    session_id = args.session_id or f"oc-{uuid.uuid4().hex[:8]}"

    if args.async_mode and not args._daemon:
        return _spawn_daemon(args, session_id)

    # Sync mode (or daemon mode)
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    provider = UnifiedProvider(
        api_key=_resolve_api_key(args),
        api_base=args.api_base,
        default_model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    session_mgr = SessionManager(workspace)
    trace_file = workspace / ".openclaw" / "traces" / f"{session_id}.jsonl"
    trace_hook = TraceCollectorHook(trace_file, session_id)
    cli_hook = CLIStreamHook(session_id, quiet=bool(args._daemon))

    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model=args.model,
        max_iterations=args.max_iterations,
        session_manager=session_mgr,
        hooks=[trace_hook, cli_hook],
    )

    # Wire subsystem event callbacks for trace completeness
    # (extracted from SWEBenchRunner._inject_event_callbacks)
    from agents.openclaw.eval.runner import inject_event_callbacks
    inject_event_callbacks(agent, trace_hook)

    # Daemon mode: register signal handlers for clean shutdown
    if args._daemon:
        _install_daemon_signal_handlers(trace_hook, agent)

    asyncio.run(_run_session(agent, args.prompt, session_id, trace_hook, cli_hook))


def _install_daemon_signal_handlers(trace_hook, agent):
    """Register SIGTERM/SIGINT handlers for clean daemon shutdown."""
    import signal, atexit

    def _cleanup(*_args):
        trace_hook.write_summary(success=False, elapsed_s=0)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(trace_hook.close)
```

### 7. Verification

- `python -m agents.openclaw --help` — prints usage
- `python -m agents.openclaw --prompt "echo hello" --workspace /tmp/test` — sync run
- `python -m agents.openclaw --prompt "echo hello" --workspace /tmp/test --async` — prints session ID
- `python -m agents.openclaw --session-id oc-xxx --status` — shows status
- `python -m agents.openclaw --session-id oc-xxx --prompt "do more"` — appends to session

### Implementation Order

1. Create `cli_hook.py` (standalone, testable)
2. Create `__main__.py` (depends on cli_hook)
3. Update `pyproject.toml` script entry
4. Manual smoke test with real API
5. Code review gate
