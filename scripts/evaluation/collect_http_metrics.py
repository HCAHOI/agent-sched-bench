"""Capture complete HTTP metric snapshots; preserve timeout gaps separately."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TextIO

import httpx


def sample(client: httpx.Client, url: str, output: TextIO, gaps: TextIO) -> bool:
    started = time.time()
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        gaps.write(json.dumps(dict(started_at_s=started, ended_at_s=time.time(),
                                   error=type(exc).__name__)) + "\n")
        gaps.flush()
        return False
    assert response.text.strip(), "Empty metrics response"
    output.write(f"# sampled_at_s {started}\n{response.text.rstrip()}\n")
    output.flush()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaps", type=Path, required=True)
    args = parser.parse_args()
    with httpx.Client(timeout=5) as client, args.output.open("x") as output, args.gaps.open("x") as gaps:
        while True:
            sample(client, args.url, output, gaps)
            # The runner's existing 90-second freshness check bounds persistent loss.
            time.sleep(1)
