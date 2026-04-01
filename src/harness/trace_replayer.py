from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


@dataclass(slots=True)
class ReplayResult:
    """Structured result for one replayed program sequence."""

    program_id: str
    replayed_steps: int
    total_wait_ms: float


class TraceReplayer:
    """Replay collected traces against an OpenAI-compatible serving endpoint."""

    def __init__(self, api_base: str, model: str, *, request_timeout_s: float = 60.0) -> None:
        self.api_base = api_base
        self.model = model
        self.request_timeout_s = request_timeout_s

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        *,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        prompt = f"Replay prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
        response = await client.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max(completion_tokens, 1),
                "temperature": 0.0,
                "extra_body": {"program_id": program_id},
            },
        )
        response.raise_for_status()
        return response.json()

    async def replay(self, trace_file: str | Path, concurrency: int = 1) -> list[ReplayResult]:
        frame = pd.read_json(Path(trace_file), lines=True)
        step_rows = frame[frame["type"] == "step"].copy()
        if step_rows.empty:
            return []

        grouped = [
            group.sort_values(["step_idx", "ts_start"])
            for _, group in step_rows.groupby("program_id")
        ]
        global_start = float(step_rows["ts_start"].min())
        results: list[ReplayResult] = []
        semaphore = asyncio.Semaphore(max(concurrency, 1))
        replay_zero = time.monotonic()

        async def replay_group(group: pd.DataFrame) -> None:
            first_offset_s = float(group.iloc[0]["ts_start"]) - global_start
            delay_s = first_offset_s - (time.monotonic() - replay_zero)
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            async with semaphore:
                async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                    total_wait_ms = 0.0
                    rows = list(group.itertuples(index=False))
                    for index, row in enumerate(rows):
                        await self._send_request(
                            client,
                            program_id=row.program_id,
                            prompt_tokens=int(getattr(row, "prompt_tokens", 0) or 0),
                            completion_tokens=int(getattr(row, "completion_tokens", 0) or 0),
                        )
                        tool_wait_ms = float(getattr(row, "tool_duration_ms", 0.0) or 0.0)
                        if index < len(rows) - 1 and tool_wait_ms > 0:
                            await asyncio.sleep(tool_wait_ms / 1000.0)
                            total_wait_ms += tool_wait_ms
                    results.append(
                        ReplayResult(
                            program_id=str(group.iloc[0]["program_id"]),
                            replayed_steps=int(len(group)),
                            total_wait_ms=total_wait_ms,
                        )
                    )

        await asyncio.gather(*(replay_group(group) for group in grouped))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay collected trace files.")
    parser.add_argument("trace_file")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replayer = TraceReplayer(api_base=args.api_base, model=args.model)
    results = asyncio.run(replayer.replay(args.trace_file, concurrency=args.concurrency))
    payload = [result.__dict__ for result in results]
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
