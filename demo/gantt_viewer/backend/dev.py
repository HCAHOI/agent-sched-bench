"""CLI entrypoint for the dynamic Gantt viewer backend."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import uvicorn

from demo.gantt_viewer.backend.cc_cache import CACHE_ROOT


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 CLI parser for the future demo server."""
    parser = argparse.ArgumentParser(
        prog="python -m trace_collect.cli gantt-serve",
        description="Launch the dynamic Gantt viewer server.",
    )
    parser.add_argument(
        "--config",
        default="demo/gantt_viewer/configs/example.yaml",
        help="Path to the viewer discovery config.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run the backend in frontend-dev mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Backend listen port.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Backend listen host.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser on startup.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached Claude Code imports before launch.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Launch the backend server."""
    args = build_parser().parse_args(argv)
    os.environ["GANTT_VIEWER_CONFIG"] = str(Path(args.config).resolve())
    if args.clear_cache and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    uvicorn.run(
        "demo.gantt_viewer.backend.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.dev,
    )
