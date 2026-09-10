#!/usr/bin/env python3
"""Collect redacted vLLM KV-cache events from a ZMQ publisher."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, TextIO

import msgspec
import zmq


_DECODER = msgspec.msgpack.Decoder()
_END_SEQ = (-1).to_bytes(8, "big", signed=True)
_MAX_REPLAY_ROUNDS = 10
_REPLAY_INTERVAL_SECONDS = 0.1
_REPLAY_TIMEOUT_MS = 1_000


def _event(tagged: Any) -> dict[str, Any]:
    # vLLM 0.28 keeps array batches but encodes each tagged event as a map.
    if isinstance(tagged, dict):
        fields = {
            "BlockStored": ("block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id", "medium", "lora_name"),
            "BlockRemoved": ("block_hashes", "medium"),
            "AllBlocksCleared": (),
        }
        tag = tagged.get("type")
        if not isinstance(tag, str) or tag not in fields:
            raise ValueError(f"unknown KV event type: {tag}")
        if any(key not in tagged for key in fields[tag]):
            raise ValueError(f"missing fields in {tag} event")
        tagged = [tag, *(tagged[key] for key in fields[tag])]
    if not isinstance(tagged, list) or not tagged or not isinstance(tagged[0], str):
        raise ValueError("KV event must be a tagged array")

    tag = tagged[0]
    if tag == "BlockStored":
        if len(tagged) < 7 or not isinstance(tagged[1], list):
            raise ValueError("invalid BlockStored event")
        block_size = tagged[4]
        medium = tagged[6]
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size <= 0
        ):
            raise ValueError("BlockStored block_size must be positive")
    elif tag == "BlockRemoved":
        if len(tagged) != 3 or not isinstance(tagged[1], list):
            raise ValueError("invalid BlockRemoved event")
        block_size = None
        medium = tagged[2]
    elif tag == "AllBlocksCleared":
        if len(tagged) != 1:
            raise ValueError("invalid AllBlocksCleared event")
        return {
            "type": tag,
            "block_count": 0,
            "block_size": None,
            "medium": None,
        }
    else:
        raise ValueError(f"unknown KV event type: {tag}")

    if medium is not None and not isinstance(medium, str):
        raise ValueError(f"{tag} medium must be a string or null")
    return {
        "type": tag,
        "block_count": len(tagged[1]),
        "block_size": block_size,
        "medium": medium,
    }


def decode_batch(payload: bytes) -> tuple[float, list[dict[str, Any]]]:
    """Decode array-like batches with array or map tagged KV events."""

    batch = _DECODER.decode(payload)
    if not isinstance(batch, list) or len(batch) not in (2, 3):
        raise ValueError("KVEventBatch must contain timestamp, events, and optional rank")
    timestamp, raw_events = batch[:2]
    if (
        not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(timestamp)
        or not isinstance(raw_events, list)
    ):
        raise ValueError("invalid KVEventBatch timestamp or events")
    return float(timestamp), [_event(event) for event in raw_events]


class EventSummary:
    def __init__(self) -> None:
        self.batch_count = 0
        self.event_count = 0
        self.stored_blocks = 0
        self.removed_blocks = 0
        self.clear_count = 0
        self.first_seq: int | None = None
        self.last_seq: int | None = None
        self.sequence_gaps: list[dict[str, int]] = []
        self.replayed_tail_batches = 0
        self.tail_replay_rounds = 0
        self.tail_replay_complete = False
        self._block_sizes: set[int] = set()

    def add_batch(self, seq: int, events: list[dict[str, Any]]) -> None:
        if self.first_seq is None:
            self.first_seq = seq
            if seq:
                self.sequence_gaps.append({"start": 0, "end": seq - 1})
        else:
            assert self.last_seq is not None
            if seq <= self.last_seq:
                raise ValueError("KV event sequence must increase")
            if seq > self.last_seq + 1:
                self.sequence_gaps.append(
                    {"start": self.last_seq + 1, "end": seq - 1}
                )

        self.last_seq = seq
        self.batch_count += 1
        self.event_count += len(events)
        for event in events:
            if event["type"] == "BlockStored":
                self.stored_blocks += event["block_count"]
                self._block_sizes.add(event["block_size"])
            elif event["type"] == "BlockRemoved":
                self.removed_blocks += event["block_count"]
            else:
                self.clear_count += 1

    def as_dict(self) -> dict[str, Any]:
        removed_tokens = None
        if len(self._block_sizes) == 1:
            removed_tokens = self.removed_blocks * next(iter(self._block_sizes))
        return {
            "batch_count": self.batch_count,
            "event_count": self.event_count,
            "stored_blocks": self.stored_blocks,
            "removed_blocks": self.removed_blocks,
            "removed_tokens": removed_tokens,
            "clear_count": self.clear_count,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "sequence_gaps": self.sequence_gaps,
            "replayed_tail_batches": self.replayed_tail_batches,
            "tail_replay_rounds": self.tail_replay_rounds,
            "tail_replay_complete": self.tail_replay_complete,
        }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_batch(
    seq_bytes: bytes,
    payload: bytes,
    summary: EventSummary,
    output: TextIO,
) -> None:
    if len(seq_bytes) != 8:
        raise ValueError("KV event sequence must contain 8 bytes")
    seq = int.from_bytes(seq_bytes, "big")
    timestamp, events = decode_batch(payload)
    summary.add_batch(seq, events)
    output.write(
        json.dumps(
            {"seq": seq, "timestamp": timestamp, "events": events},
            separators=(",", ":"),
        )
        + "\n"
    )
    output.flush()


def _drain_live_batches(
    socket: zmq.Socket,
    summary: EventSummary,
    output: TextIO,
) -> None:
    while socket.poll(0, zmq.POLLIN):
        frames = socket.recv_multipart()
        if len(frames) != 3:
            raise ValueError("KV event message must have topic, sequence, payload")
        _record_batch(frames[1], frames[2], summary, output)


def _replay_tail(
    socket: zmq.Socket,
    summary: EventSummary,
    output: TextIO,
) -> None:
    empty_rounds = 0
    for _ in range(_MAX_REPLAY_ROUNDS):
        summary.tail_replay_rounds += 1
        start_seq = 0 if summary.last_seq is None else summary.last_seq + 1
        socket.send_multipart((b"", start_seq.to_bytes(8, "big")))

        replayed_this_round = 0
        while True:
            if not socket.poll(_REPLAY_TIMEOUT_MS, zmq.POLLIN):
                raise TimeoutError("timed out waiting for the KV replay endpoint")
            frames = socket.recv_multipart()
            if len(frames) == 4:
                # New publishers include the topic after the ROUTER delimiter.
                frames = [frames[0], *frames[2:]]
            if len(frames) != 3 or frames[0] != b"" or len(frames[1]) != 8:
                raise ValueError("KV replay message must have delimiter, sequence, payload")
            if frames[1] == _END_SEQ:
                if frames[2]:
                    raise ValueError("KV replay end marker must have an empty payload")
                break

            _record_batch(frames[1], frames[2], summary, output)
            replayed_this_round += 1
            summary.replayed_tail_batches += 1

        if replayed_this_round:
            empty_rounds = 0
        else:
            empty_rounds += 1
            if empty_rounds == 2:
                summary.tail_replay_complete = True
                return
        time.sleep(_REPLAY_INTERVAL_SECONDS)

    raise RuntimeError("KV replay did not converge after 10 rounds")


def collect(
    endpoint: str,
    replay_endpoint: str,
    ready_file: Path,
    events_jsonl: Path,
    summary_json: Path,
) -> None:
    stopped = False
    summary = EventSummary()

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1_000_000)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.LINGER, 0)
    replay = context.socket(zmq.DEALER)
    replay.setsockopt(zmq.RCVHWM, 1_000_000)
    replay.setsockopt(zmq.LINGER, 0)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    events_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with events_jsonl.open("x", encoding="utf-8", buffering=1) as output:
            socket.connect(endpoint)
            replay.connect(replay_endpoint)
            ready_file.touch(exist_ok=False)
            while not stopped:
                if socket not in dict(poller.poll(100)):
                    continue
                frames = socket.recv_multipart()
                if len(frames) != 3:
                    raise ValueError("KV event message must have topic, sequence, payload")
                _record_batch(frames[1], frames[2], summary, output)
                if summary.sequence_gaps:
                    raise RuntimeError(f"KV live sequence gaps: {summary.sequence_gaps}")
            _drain_live_batches(socket, summary, output)
            _replay_tail(replay, summary, output)
    finally:
        socket.close(linger=0)
        replay.close(linger=0)
        context.term()
        _write_json_atomic(summary_json, summary.as_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--replay-endpoint", required=True)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--events-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()
    collect(
        args.endpoint,
        args.replay_endpoint,
        args.ready_file,
        args.events_jsonl,
        args.summary_json,
    )


if __name__ == "__main__":
    main()
