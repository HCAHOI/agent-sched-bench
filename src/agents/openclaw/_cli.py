"""OpenClaw standalone CLI — argument parsing, streaming hook, sync/async runner.

Uses the same full bus-based dispatch path as SWE-bench evaluation
(MessageBus + AgentLoop.run() + ResultCollector) via SessionRunner.

Usage:
    python -m agents.openclaw --prompt "Create a Tetris game" --workspace ~/tetris
    python -m agents.openclaw --prompt "Build Pac-Man" --workspace ~/pacman --async
    python -m agents.openclaw --session-id oc-abc123 --prompt "Add multiplayer"
    python -m agents.openclaw --session-id oc-abc123 --status
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLIStreamHook — real-time event output to stderr
# ---------------------------------------------------------------------------


def _ts() -> str:
    """Current time as HH:MM:SS."""
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


class CLIStreamHook:
    """Prints real-time agent events to stderr for CLI observation.

    Injected into AgentLoop via SessionRunner's ``extra_hooks`` parameter.
    CompositeHook calls lifecycle methods (before_iteration, after_iteration,
    etc.) — CLIStreamHook prints them as structured log lines to stderr.
    """

    def __init__(self, session_id: str, *, quiet: bool = False) -> None:
        self.sid_short = session_id[:10]
        self.quiet = quiet
        self._iter_start = 0.0
        self._n_iterations = 0
        self._total_tokens = 0
        self._wall_start = time.monotonic()

    def _log(self, event_type: str, detail: str) -> None:
        if self.quiet:
            return
        print(
            f"[{self.sid_short}] {_ts()} {event_type:<10} {detail}",
            file=sys.stderr,
            flush=True,
        )

    # -- AgentHook lifecycle methods (called via CompositeHook) --

    def wants_streaming(self) -> bool:
        return not self.quiet

    async def before_iteration(self, context: Any) -> None:
        self._iter_start = time.monotonic()

    async def before_execute_tools(self, context: Any) -> None:
        if self.quiet or not context.tool_calls:
            return
        for tc in context.tool_calls:
            args_preview = json.dumps(tc.arguments, ensure_ascii=False)[:80]
            self._log("TOOL", f"{tc.name}({args_preview})")

    async def after_iteration(self, context: Any) -> None:
        self._n_iterations += 1
        usage = context.usage or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        self._total_tokens += pt + ct
        lat_ms = (time.monotonic() - self._iter_start) * 1000 if self._iter_start else 0
        self._log(
            "ITER",
            f"step {self._n_iterations}, tokens: {pt}+{ct}, llm: {lat_ms:.0f}ms",
        )

    async def on_stream(self, context: Any, delta: str) -> None:
        pass

    async def on_stream_end(self, context: Any, *, resuming: bool) -> None:
        pass

    def finalize_content(self, context: Any, content: str | None) -> str | None:
        """Required by CompositeHook — pass through without modification."""
        return content

    def print_summary(self) -> None:
        """Print final DONE line with aggregate stats."""
        elapsed = time.monotonic() - self._wall_start
        self._log(
            "DONE",
            f"completed in {elapsed:.1f}s, {self._n_iterations} steps, "
            f"{self._total_tokens} total tokens",
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.openclaw",
        description="Run the OpenClaw agent on an arbitrary task.",
    )

    parser.add_argument("--prompt", help="Task prompt for the agent.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Query status of an async session (requires --session-id).",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier "
            '(default: env OPENCLAW_MODEL or "qwen/qwen3.6-plus:free").'
        ),
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "API base URL "
            '(default: env OPENCLAW_API_BASE or "https://openrouter.ai/api/v1").'
        ),
    )
    parser.add_argument("--api-key", default=None, help="API key (default: from env).")

    parser.add_argument(
        "--workspace",
        default=".",
        help="Working directory for the agent (default: cwd).",
    )
    parser.add_argument(
        "--trace-output",
        default=None,
        help=(
            "Output path for the trace JSONL file. Default: "
            "<repo>/traces/openclaw_cli/<model_slug>/<UTC_TS>/<session_id>.jsonl "
            "(relative to the agent-sched-bench repo root, NOT the workspace). "
            "Set this to override (e.g., for one-off experiments)."
        ),
    )

    parser.add_argument(
        "--async",
        dest="run_async",
        action="store_true",
        help="Run in background, print session ID and exit.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Resume/append to an existing session.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Max agent iterations (default: 200).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens per LLM call (default: 8192).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress event stream, only print final output.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output final result as JSON to stdout.",
    )

    # Internal (daemon mode)
    parser.add_argument("--_daemon", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_pid-file", default=None, help=argparse.SUPPRESS)

    return parser


# ---------------------------------------------------------------------------
# Config resolution (runs BEFORE heavy imports for fast failure)
# ---------------------------------------------------------------------------


def _resolve_api_key(args: argparse.Namespace) -> str:
    """Resolve API key from args or environment. Fail fast if missing."""
    key = (
        args.api_key
        or os.environ.get("OPENCLAW_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not key:
        print(
            "ERROR: No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, "
            "DASHSCOPE_API_KEY, or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _resolve_model(args: argparse.Namespace) -> str:
    return args.model or os.environ.get("OPENCLAW_MODEL", "qwen/qwen3.6-plus:free")


def _resolve_api_base(args: argparse.Namespace) -> str:
    return args.api_base or os.environ.get(
        "OPENCLAW_API_BASE", "https://openrouter.ai/api/v1"
    )


# ---------------------------------------------------------------------------
# Trace output path resolution
# ---------------------------------------------------------------------------


def _resolve_repo_root() -> Path | None:
    """Walk up from this file to find the agent-sched-bench repo root.

    Returns ``None`` if no ``pyproject.toml`` is found above this file —
    callers should fall back to ``Path.cwd()``.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _slug_model(name: str) -> str:
    """Convert a model identifier to a filesystem-safe slug.

    e.g., ``qwen/qwen3.6-plus:free`` -> ``qwen_qwen3.6-plus_free``.
    """
    return name.replace("/", "_").replace(":", "_")


def _resolve_trace_output(
    args: argparse.Namespace, session_id: str, model: str
) -> Path:
    """Decide where the OpenClaw CLI should write its trace JSONL.

    Precedence:
      1. Explicit ``--trace-output`` (resolved with ``expanduser`` + ``resolve``).
      2. ``<repo>/traces/openclaw_cli/<model_slug>/<UTC_TS>/<session_id>.jsonl``
         where ``<repo>`` is the agent-sched-bench root.
      3. ``<cwd>/traces/openclaw_cli/<model_slug>/<UTC_TS>/<session_id>.jsonl``
         if the repo root cannot be located (e.g., installed wheel).
    """
    if args.trace_output:
        return Path(args.trace_output).expanduser().resolve()

    base = _resolve_repo_root() or Path.cwd().resolve()
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / "traces" / "openclaw_cli" / _slug_model(model) / ts / f"{session_id}.jsonl"


# ---------------------------------------------------------------------------
# Sync runner — uses SessionRunner (full bus path)
# ---------------------------------------------------------------------------


def _run_sync(args: argparse.Namespace) -> int:
    """Run agent in sync (blocking) mode via SessionRunner."""
    # Resolve config BEFORE heavy imports so missing API key fails fast
    api_key = _resolve_api_key(args)
    api_base = _resolve_api_base(args)
    model = _resolve_model(args)

    from agents.openclaw._session_runner import SessionRunner
    from agents.openclaw.unified_provider import UnifiedProvider

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    session_id = args.session_id or f"oc-{uuid.uuid4().hex[:8]}"
    session_key = (
        f"cli:{session_id}" if not session_id.startswith("cli:") else session_id
    )

    is_daemon = getattr(args, "_daemon", False)
    cli_hook = CLIStreamHook(session_id, quiet=is_daemon or args.quiet)

    if not is_daemon:
        print(
            f"Session: {session_key} | Model: {model} | Workspace: {workspace}",
            file=sys.stderr,
        )

    provider = UnifiedProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    trace_file = _resolve_trace_output(args, session_id, model)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    if not is_daemon:
        print(f"Trace: {trace_file}", file=sys.stderr)

    runner = SessionRunner(
        provider,
        model=model,
        max_iterations=args.max_iterations,
        extra_hooks=[cli_hook],
    )

    # Daemon mode: register signal handlers for clean shutdown
    if is_daemon:
        _install_daemon_signal_handlers(args)

    cli_hook._log("START", f'prompt="{args.prompt[:80]}"')

    try:
        result = asyncio.run(
            runner.run(
                prompt=args.prompt,
                workspace=workspace,
                session_key=session_key,
                trace_file=trace_file,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    cli_hook.print_summary()

    content = result.content
    if content is None:
        cli_hook._log(
            "WARN",
            "Agent completed without direct response — check workspace for output.",
        )

    if args.output_json:
        output = {
            "session_key": session_key,
            "content": content,
            "steps": cli_hook._n_iterations,
            "tokens": cli_hook._total_tokens,
            "elapsed_s": round(result.elapsed_s, 2),
            "trace_file": str(trace_file),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif content:
        print(content)

    return 0


def _install_daemon_signal_handlers(args: argparse.Namespace) -> None:
    """Register SIGTERM/SIGINT handlers for clean daemon shutdown."""
    pid_file = getattr(args, "_pid_file", None)

    def _cleanup(*_args: Any) -> None:
        # Don't write a fabricated summary — the real TraceCollectorHook
        # inside SessionRunner.run() owns the trace file.  Just clean up
        # the PID file so --status reports "completed" (not "running").
        if pid_file:
            try:
                Path(pid_file).unlink(missing_ok=True)
            except OSError:
                pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup)
    if pid_file:
        atexit.register(lambda: Path(pid_file).unlink(missing_ok=True))


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


def _run_async(args: argparse.Namespace) -> int:
    """Spawn daemon process, print session info, exit immediately."""
    from agents.openclaw._daemon import spawn_daemon

    session_id = args.session_id or f"oc-{uuid.uuid4().hex[:8]}"
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    pid_dir = workspace / ".openclaw" / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / f"{session_id}.pid"

    # Resolve the trace path in the parent so:
    #   1) ``--trace-output`` overrides are honored in async mode (the
    #      daemon would otherwise compute its own default), and
    #   2) ``--status`` can locate the trace via the PID file metadata
    #      without re-deriving the path.
    model = _resolve_model(args)
    trace_file = _resolve_trace_output(args, session_id, model)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "agents.openclaw",
        "--_daemon",
        "--_pid-file",
        str(pid_file),
        "--prompt",
        args.prompt,
        "--workspace",
        str(workspace),
        "--model",
        model,
        "--api-base",
        _resolve_api_base(args),
        "--max-iterations",
        str(args.max_iterations),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--session-id",
        session_id,
        "--trace-output",
        str(trace_file),
    ]

    # Pass API key via environment, not argv (avoid ps leaking secrets)
    api_key = _resolve_api_key(args)
    pid = spawn_daemon(
        cmd,
        pid_file,
        session_id,
        extra_env={"OPENCLAW_API_KEY": api_key},
        trace_file=trace_file,
    )

    result = {
        "session_id": session_id,
        "pid": pid,
        "pid_file": str(pid_file),
        "workspace": str(workspace),
        "trace_file": str(trace_file),
    }
    print(json.dumps(result, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Status query
# ---------------------------------------------------------------------------


def _run_status(args: argparse.Namespace) -> int:
    """Print JSON status for a session."""
    from agents.openclaw._daemon import get_session_status

    if not args.session_id:
        print("ERROR: --status requires --session-id.", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).expanduser().resolve()
    status = get_session_status(args.session_id, workspace)
    print(json.dumps(status, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    if args.status:
        sys.exit(_run_status(args))

    if not args.prompt:
        print(
            "ERROR: --prompt is required (unless using --status).",
            file=sys.stderr,
        )
        build_parser().print_usage(sys.stderr)
        sys.exit(1)

    if args._daemon:
        sys.exit(_run_sync(args))
    elif args.run_async:
        sys.exit(_run_async(args))
    else:
        sys.exit(_run_sync(args))
