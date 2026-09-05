from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import msgspec
import zmq

from scripts.evaluation.collect_vllm_kv_events import EventSummary, decode_batch


SCRIPT = (
    Path(__file__).parents[1] / "scripts/evaluation/collect_vllm_kv_events.py"
)


def test_decodes_both_vllm_payloads_and_summarizes_without_sensitive_fields() -> None:
    old_payload = msgspec.msgpack.encode(
        [
            1.25,
            [
                [
                    "BlockStored",
                    [b"secret-a", b"secret-b"],
                    None,
                    [998877],
                    16,
                    None,
                    "GPU",
                ],
                ["BlockRemoved", [b"secret-c"], "GPU"],
            ],
        ]
    )
    new_payload = msgspec.msgpack.encode(
        [
            2.5,
            [
                [
                    "BlockStored",
                    [b"secret-d"],
                    None,
                    [887766],
                    16,
                    None,
                    "GPU",
                    None,
                    None,
                ],
                ["AllBlocksCleared"],
            ],
            0,
        ]
    )

    first_timestamp, first_events = decode_batch(old_payload)
    second_timestamp, second_events = decode_batch(new_payload)
    assert (first_timestamp, second_timestamp) == (1.25, 2.5)
    assert first_events == [
        {
            "type": "BlockStored",
            "block_count": 2,
            "block_size": 16,
            "medium": "GPU",
        },
        {
            "type": "BlockRemoved",
            "block_count": 1,
            "block_size": None,
            "medium": "GPU",
        },
    ]
    encoded_events = json.dumps(first_events + second_events)
    assert "secret" not in encoded_events
    assert "998877" not in encoded_events
    assert "887766" not in encoded_events

    summary = EventSummary()
    summary.add_batch(2, first_events)
    summary.add_batch(4, second_events)
    assert summary.as_dict() == {
        "batch_count": 2,
        "event_count": 4,
        "stored_blocks": 3,
        "removed_blocks": 1,
        "removed_tokens": 16,
        "clear_count": 1,
        "first_seq": 2,
        "last_seq": 4,
        "sequence_gaps": [{"start": 0, "end": 1}, {"start": 3, "end": 3}],
        "replayed_tail_batches": 0,
        "tail_replay_rounds": 0,
        "tail_replay_complete": False,
    }

    summary.add_batch(
        5,
        [
            {
                "type": "BlockStored",
                "block_count": 1,
                "block_size": 32,
                "medium": "GPU",
            }
        ],
    )
    assert summary.as_dict()["removed_tokens"] is None


def test_sigterm_recovers_unseen_tail_from_replay_endpoint(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    events = tmp_path / "events.jsonl"
    summary = tmp_path / "summary.json"
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.RCVTIMEO, 5_000)
    router.bind("tcp://127.0.0.1:*")
    replay_endpoint = router.getsockopt_string(zmq.LAST_ENDPOINT)
    process = subprocess.Popen(
        [
            sys.executable,
            SCRIPT,
            "--endpoint",
            "inproc://unused-kv-events",
            "--replay-endpoint",
            replay_endpoint,
            "--ready-file",
            ready,
            "--events-jsonl",
            events,
            "--summary-json",
            summary,
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.terminate()

        tail_payload = msgspec.msgpack.encode(
            [
                3.5,
                [
                    [
                        "BlockStored",
                        [b"secret-tail"],
                        None,
                        [123456],
                        16,
                        None,
                        "GPU",
                    ]
                ],
            ]
        )
        requested_starts = []
        for round_index in range(3):
            identity, delimiter, start_seq_bytes = router.recv_multipart()
            assert delimiter == b""
            requested_starts.append(int.from_bytes(start_seq_bytes, "big"))
            if round_index == 0:
                router.send_multipart(
                    [identity, b"", (0).to_bytes(8, "big"), tail_payload]
                )
            router.send_multipart(
                [identity, b"", (-1).to_bytes(8, "big", signed=True), b""]
            )

        assert requested_starts == [0, 1, 1]
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        router.close(linger=0)
        context.term()

    event_rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert event_rows == [
        {
            "seq": 0,
            "timestamp": 3.5,
            "events": [
                {
                    "type": "BlockStored",
                    "block_count": 1,
                    "block_size": 16,
                    "medium": "GPU",
                }
            ],
        }
    ]
    assert "secret-tail" not in events.read_text()
    assert "123456" not in events.read_text()
    assert json.loads(summary.read_text()) == {
        "batch_count": 1,
        "event_count": 1,
        "stored_blocks": 1,
        "removed_blocks": 0,
        "removed_tokens": 0,
        "clear_count": 0,
        "first_seq": 0,
        "last_seq": 0,
        "sequence_gaps": [],
        "replayed_tail_batches": 1,
        "tail_replay_rounds": 3,
        "tail_replay_complete": True,
    }
