"""Run with the isolated, patched PPD environment; this is a CPU plumbing check."""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_request_identity_and_preemption_metrics() -> None:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.metrics.stats import IterationStats, LoRARequestStates, RequestStateStats

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "requests.jsonl"
        with patch.dict(os.environ, VLLM_REQUEST_TELEMETRY_PATH=str(output)):
            iteration = IterationStats()
            stats = RequestStateStats(arrival_time=iteration.iteration_timestamp - 13,
                                      first_token_ts=18, last_token_ts=22,
                                      first_token_latency=9, num_generation_tokens=5)
            loras = LoRARequestStates()
            events = [EngineCoreEvent(kind, timestamp) for kind, timestamp in (
                (EngineCoreEventType.QUEUED, 10),
                (EngineCoreEventType.SCHEDULED, 12),
                (EngineCoreEventType.PREEMPTED, 14),
                (EngineCoreEventType.SCHEDULED, 17),
            )]
            iteration.update_from_events("chatcmpl-pd-wire-id", events, True, stats, loras, None)
            processor = OutputProcessor.__new__(OutputProcessor)
            processor.lora_states = loras
            request = SimpleNamespace(request_id="chatcmpl-pd-wire-id", external_req_id="chatcmpl-pd-wire-id", stats=stats,
                                      prompt_token_ids=[1] * 100, prompt_len=100, prompt_embeds=None,
                                      max_tokens_param=5, num_cached_tokens=16,
                                      lora_name=None, parent_req=None)
            processor._update_stats_from_finished(request, FinishReason.LENGTH, iteration)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["request_id"] == request.request_id
        assert row["prompt_tokens"] == 100 and row["generation_tokens"] == 5
        assert (row["queue_s"], row["prefill_s"], row["decode_s"], row["inference_s"]) == (2, 6, 4, 10)
        assert row["e2e_s"] == 13 and row["ttft_s"] == 9
        assert row["preemption_count"] == 1
        assert row["preempted_wait_s"] == row["max_preempted_wait_s"] == 3
        assert row["preemption_timing_complete"]


if __name__ == "__main__":
    test_request_identity_and_preemption_metrics()
    print("PPD request identity, phase spans and preemption timing passed.")
