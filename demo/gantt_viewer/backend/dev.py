"""CLI entrypoint scaffold for the dynamic Gantt viewer backend."""

from __future__ import annotations

import argparse


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
    """Parse arguments and reserve the final Phase 2 interface."""
    build_parser().parse_args(argv)
    raise SystemExit(
        "gantt-serve backend scaffolded in Phase 1; Phase 2 will add the "
        "FastAPI server implementation."
    )
