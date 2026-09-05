from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from trace_collect.openclaw_host_runtime import run_openclaw_host_replay_request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OpenClaw host replay worker")
    parser.add_argument("--request", required=True, type=Path)
    return parser.parse_args()


async def _main_async() -> int:
    args = _parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    status = await run_openclaw_host_replay_request(request)
    return 0 if status.get("success") is True else 1


def main() -> None:
    raise SystemExit(asyncio.run(_main_async()))


if __name__ == "__main__":
    main()
