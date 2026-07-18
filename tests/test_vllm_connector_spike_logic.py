"""CPU tests for the vLLM connector offload spike logic (no vllm/GPU).

Covers the control channel, transfer timing math, ITL statistics, the fake
transfer backend, and the report schema -- everything the GPU driver relies on.
"""

from __future__ import annotations

import json

import pytest

from spike.vllm_connector import (
    FakeTransferBackend,
    OffloadControl,
    OffloadPhase,
    RepetitionResult,
    SpikeReport,
    TransferTiming,
    chunk_ranges,
    gbps,
    itl_summary,
    percentile,
    validate_block_ids,
    validate_staging_capacity,
)
from spike.vllm_connector.control import ControlState


def test_gbps_matches_hand_computation() -> None:
    # 2 GB in 100 ms -> 20 GB/s.
    assert gbps(2_000_000_000, 100.0) == pytest.approx(20.0)


def test_gbps_rejects_nonpositive_time() -> None:
    with pytest.raises(ValueError, match="> 0 ms"):
        gbps(1000, 0.0)


def test_percentile_linear_interpolation() -> None:
    vals = [0.0, 10.0]
    assert percentile(vals, 0) == 0.0
    assert percentile(vals, 100) == 10.0
    assert percentile(vals, 50) == pytest.approx(5.0)


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 150)


def test_itl_summary_reports_tail() -> None:
    s = itl_summary([1.0, 2.0, 3.0, 100.0])
    assert s["count"] == 4
    assert s["max"] == 100.0
    assert s["p99"] > s["p50"]


def test_transfer_timing_effective_gbps() -> None:
    t = TransferTiming(bytes_moved=1_000_000_000, milliseconds=50.0, num_blocks=4)
    assert t.effective_gbps == pytest.approx(20.0)


def test_fake_backend_offload_then_restore_roundtrip() -> None:
    be = FakeTransferBackend(bytes_per_block=1024)
    off = be.offload([0, 1, 2, 3])
    assert off.num_blocks == 4
    assert off.bytes_moved == 4 * 1024
    assert off.effective_gbps == pytest.approx(be.offload_gbps)
    res = be.restore([0, 1, 2, 3])
    assert res.bytes_moved == off.bytes_moved
    assert be.offloaded == set()


def test_fake_backend_rejects_restore_without_offload() -> None:
    be = FakeTransferBackend(bytes_per_block=64)
    with pytest.raises(ValueError, match="never offloaded"):
        be.restore([7])


def test_fake_backend_rejects_zero_blocks() -> None:
    be = FakeTransferBackend(bytes_per_block=64)
    with pytest.raises(ValueError, match="zero blocks"):
        be.offload([])


def test_fake_backend_enforces_staging_capacity() -> None:
    # Mirrors the staged GPU backend: a transfer wider than the pre-allocated
    # staging buffer fails fast rather than overrunning it.
    be = FakeTransferBackend(bytes_per_block=64, max_blocks=4)
    be.offload([0, 1, 2, 3])  # exactly at capacity is fine
    with pytest.raises(ValueError, match="exceeds staging capacity 4"):
        be.offload([0, 1, 2, 3, 4])


def test_fake_backend_records_chunk_count() -> None:
    # chunk_bytes caps bytes per copy; 8 blocks x 64 B = 512 B in 128-B chunks
    # -> 2 blocks/chunk -> 4 chunks. Bytes moved is unchanged by chunking.
    be = FakeTransferBackend(bytes_per_block=64, chunk_bytes=128)
    off = be.offload([0, 1, 2, 3, 4, 5, 6, 7])
    assert be.last_num_chunks == 4
    assert off.bytes_moved == 8 * 64
    # chunk_bytes=0 disables chunking -> a single copy.
    be2 = FakeTransferBackend(bytes_per_block=64)
    be2.offload([0, 1, 2, 3])
    assert be2.last_num_chunks == 1


def test_validate_staging_capacity_bounds() -> None:
    validate_staging_capacity(max_blocks=10, num_blocks=10)  # at capacity: ok
    with pytest.raises(ValueError, match="exceeds staging capacity 10"):
        validate_staging_capacity(max_blocks=10, num_blocks=11)
    with pytest.raises(ValueError, match="must be > 0"):
        validate_staging_capacity(max_blocks=10, num_blocks=0)


def test_chunk_ranges_partition_is_exact_and_bounded() -> None:
    # No chunking -> one range covering everything.
    assert chunk_ranges(5, bytes_per_block=100, chunk_bytes=0) == [(0, 5)]
    # 100-B blocks, 250-B chunks -> 2 blocks/chunk -> [(0,2),(2,4),(4,5)].
    ranges = chunk_ranges(5, bytes_per_block=100, chunk_bytes=250)
    assert ranges == [(0, 2), (2, 4), (4, 5)]
    # Ranges tile [0, num_blocks) with no gaps or overlaps.
    assert ranges[0][0] == 0 and ranges[-1][1] == 5
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    # Each chunk stays within the byte cap.
    assert all((c1 - c0) * 100 <= 250 for c0, c1 in ranges)


def test_chunk_ranges_never_zero_blocks_when_block_exceeds_cap() -> None:
    # A single block larger than chunk_bytes still moves one whole block/chunk.
    assert chunk_ranges(3, bytes_per_block=1000, chunk_bytes=100) == [(0, 1), (1, 2), (2, 3)]


def test_chunk_ranges_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="num_blocks must be > 0"):
        chunk_ranges(0, bytes_per_block=100, chunk_bytes=100)
    with pytest.raises(ValueError, match="bytes_per_block must be > 0"):
        chunk_ranges(4, bytes_per_block=0, chunk_bytes=100)


def test_validate_block_ids_accepts_in_range() -> None:
    # Simulates CudaBlockTransfer._copy after tensor.movedim(block_dim, 0):
    # the check runs against the SIZE OF THE MOVED-TO-FRONT DIM, not dim 0 of
    # the original tensor -- e.g. a layout where the block dim has size 16 but
    # some other dim is larger (32) must bound against 16, not 32.
    block_dim_size, other_dim_size = 16, 32
    assert other_dim_size > block_dim_size  # the case a dim-0-only check would miss
    validate_block_ids(block_dim_size, [0, 1, 15])


def test_validate_block_ids_rejects_out_of_range_against_moved_dim() -> None:
    block_dim_size = 16
    with pytest.raises(ValueError, match="out of range for dim size 16"):
        validate_block_ids(block_dim_size, [0, 16])  # 16 is out of range (0..15)


def test_validate_block_ids_rejects_negative_and_empty() -> None:
    with pytest.raises(ValueError, match="out of range"):
        validate_block_ids(8, [-1])
    with pytest.raises(ValueError, match="no block ids"):
        validate_block_ids(8, [])


def test_repetition_result_rejects_mismatched_transfers() -> None:
    off = TransferTiming(bytes_moved=100, milliseconds=1.0, num_blocks=2)
    res = TransferTiming(bytes_moved=100, milliseconds=1.0, num_blocks=3)
    with pytest.raises(ValueError, match="block count mismatch"):
        RepetitionResult.from_timings(0, 0, 1.0, off, res)


def _report_with_one_rep() -> SpikeReport:
    be = FakeTransferBackend(bytes_per_block=2048)
    off = be.offload([0, 1, 2, 3])
    res = be.restore([0, 1, 2, 3])
    rep = RepetitionResult.from_timings(0, 42, 3.0, off, res)
    return SpikeReport(
        model="meta-llama/Llama-3.1-8B",
        vllm_version=None,
        kv_seam="test-seam",
        num_load_requests=8,
        repetitions=[rep],
        itl_with_offload_ms=[10.0, 12.0, 50.0],
        itl_without_offload_ms=[10.0, 11.0, 12.0],
    )


def test_spike_report_json_is_serializable_and_has_summary() -> None:
    report = _report_with_one_rep()
    payload = json.loads(report.to_json())
    assert payload["model"].endswith("Llama-3.1-8B")
    summ = payload["summary"]
    assert summ["bytes_moved"] == 4 * 2048
    assert summ["offload_gbps_median"] > 0
    # Interference: the with-offload window has a heavier tail than without.
    assert (
        summ["interference"]["with_offload"]["max"]
        > summ["interference"]["without_offload"]["max"]
    )


def test_spike_report_empty_reps_fails_fast() -> None:
    report = _report_with_one_rep()
    report.repetitions = []
    with pytest.raises(ValueError, match="no repetitions"):
        report.summary()


def test_offload_control_roundtrip_and_epoch_bump(tmp_path) -> None:
    ctrl = OffloadControl(tmp_path / "control.json")
    assert ctrl.read().phase == OffloadPhase.RESIDENT
    assert ctrl.read().epoch == 0

    s1 = ctrl.request_offload("agent-0")
    assert s1.phase == OffloadPhase.OFFLOAD
    assert s1.target_request_id == "agent-0"
    assert s1.epoch == 1

    s2 = ctrl.request_restore("agent-0")
    assert s2.phase == OffloadPhase.RESTORE
    assert s2.epoch == 2

    assert ctrl.clear().phase == OffloadPhase.RESIDENT


def test_offload_control_survives_reopen(tmp_path) -> None:
    # The connector opens the same file in another process -- state must persist.
    path = tmp_path / "control.json"
    OffloadControl(path).request_offload("agent-7")
    reopened = OffloadControl(path).read()
    assert reopened.target_request_id == "agent-7"
    assert reopened.phase == OffloadPhase.OFFLOAD


def test_control_state_json_roundtrip() -> None:
    st = ControlState(target_request_id="r1", phase=OffloadPhase.OFFLOAD, epoch=5)
    back = ControlState.from_json(st.to_json())
    assert back == st
