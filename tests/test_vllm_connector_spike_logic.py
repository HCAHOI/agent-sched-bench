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
    OffloadedBlocks,
    OffloadPhase,
    PauseBook,
    PauseResult,
    RepetitionResult,
    SavedKVRegistry,
    SaveConfirmationReader,
    SpikeReport,
    TransferTiming,
    assert_saved_covers_tokens,
    cdiv,
    check_repetition_transfers,
    chunk_ranges,
    gbps,
    guard_pause_trigger,
    itl_summary,
    matched_tokens_delta,
    percentile,
    resume_load_block_ids,
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


# --- GPU-validation bugfix: restore must reuse the EXACT offloaded blocks,
# not the connector's live (and, on a finished/pruned target, absent) tracking
# table -- reproduces the "offload=1, restore=0" bug reported off the box in
# staged mode.


def test_offloaded_blocks_pop_returns_pinned_set_once() -> None:
    reg = OffloadedBlocks()
    reg.record("agent-0", [1, 2, 3])
    # A live/grown re-read must NOT leak in -- pop always returns exactly what
    # was recorded, and only once (one-shot, mirrors SavedKVRegistry.drop).
    assert reg.pop("agent-0") == [1, 2, 3]
    assert reg.pop("agent-0") is None


def test_offloaded_blocks_pop_of_unknown_request_is_none() -> None:
    # Never offloaded, already restored, or the request finished first -- all
    # three collapse to the same safe "nothing to restore" signal.
    reg = OffloadedBlocks()
    assert reg.pop("never-offloaded") is None


def test_offloaded_blocks_drop_forgets_without_restoring() -> None:
    reg = OffloadedBlocks()
    reg.record("agent-0", [1, 2, 3])
    reg.drop("agent-0")  # request finished mid-offload; blocks about to be freed
    assert reg.pop("agent-0") is None


def test_offloaded_blocks_record_is_a_snapshot_not_a_live_view() -> None:
    # The exact bug: if record() aliased the caller's list, later mutating
    # that list (e.g. a scheduler live block-id table growing) would corrupt
    # the pinned restore set. record() must copy.
    live = [1, 2, 3]
    reg = OffloadedBlocks()
    reg.record("agent-0", live)
    live.append(4)  # simulate the live table growing after OFFLOAD
    assert reg.pop("agent-0") == [1, 2, 3]


def test_check_repetition_transfers_passes_when_both_present() -> None:
    rows = [{"phase": "offload"}]
    check_repetition_transfers(rows, rows, [], rep=0)


def test_check_repetition_transfers_distinguishes_finished_target_from_broken_seam() -> None:
    # offload fired, restore never did, but a restore_skipped row explains why
    # -- must raise a message about the liveness race, not "seam did not fire".
    with pytest.raises(RuntimeError, match="finished before the restore trigger"):
        check_repetition_transfers(
            [{"phase": "offload"}],
            [],
            [{"phase": "restore_skipped", "reason": "target request finished before restore fired"}],
            rep=0,
        )


def test_check_repetition_transfers_reports_broken_seam_when_no_skip_row() -> None:
    # No restore row AND no explanatory skip row -- this is the "seam is
    # actually broken" case and must keep the original diagnostic.
    with pytest.raises(RuntimeError, match="the seam did not fire"):
        check_repetition_transfers([{"phase": "offload"}], [], [], rep=0)


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


# --- P4 pause/resume pure logic ---------------------------------------------


def test_control_pause_resume_phases(tmp_path) -> None:
    ctrl = OffloadControl(tmp_path / "control.json")
    p = ctrl.request_pause("agent-0")
    assert p.phase == OffloadPhase.PAUSE and p.target_request_id == "agent-0" and p.epoch == 1
    r = ctrl.request_resume("agent-0")
    assert r.phase == OffloadPhase.RESUME and r.epoch == 2
    # Survives a reopen (connector/scheduler read it from another process).
    assert OffloadControl(tmp_path / "control.json").read().phase == OffloadPhase.RESUME


def test_cdiv() -> None:
    assert cdiv(0, 16) == 0
    assert cdiv(1, 16) == 1
    assert cdiv(16, 16) == 1
    assert cdiv(17, 16) == 2
    with pytest.raises(ValueError):
        cdiv(10, 0)


def test_assert_saved_covers_tokens_exact_coverage() -> None:
    # 33 tokens at block_size 16 -> ceil = 3 blocks, exactly.
    assert_saved_covers_tokens(33, block_count=3, block_size=16)
    with pytest.raises(ValueError, match="do not cover"):
        assert_saved_covers_tokens(33, block_count=2, block_size=16)  # under-covers
    with pytest.raises(ValueError, match="do not cover"):
        assert_saved_covers_tokens(33, block_count=4, block_size=16)  # over-covers
    with pytest.raises(ValueError, match="> 0 at pause"):
        assert_saved_covers_tokens(0, block_count=0, block_size=16)


def test_assert_saved_covers_tokens_rejects_gpu_confirmed_stale_count() -> None:
    # Exact numbers from the GPU crash: the connector's admission-time
    # self._block_ids (97 blocks) went stale as decode grew the request to
    # 1581 computed tokens (needs ceil(1581/16) = 99 blocks). The stale count
    # must fail fast; the fresh count from kv_cache_manager.get_block_ids
    # must pass.
    with pytest.raises(ValueError, match=r"saved 97 blocks do not cover 1581 tokens"):
        assert_saved_covers_tokens(1581, block_count=97, block_size=16)
    assert_saved_covers_tokens(1581, block_count=99, block_size=16)  # fresh count: ok


def test_matched_tokens_delta_is_beyond_local_only() -> None:
    # Real pressure: nothing local -> the whole save is the delta.
    assert matched_tokens_delta(100, 0) == 100
    # Self-hit under low pressure: local already covers it -> 0 extra.
    assert matched_tokens_delta(100, 100) == 0
    assert matched_tokens_delta(100, 40) == 60
    # Never negative even if local somehow exceeds the save.
    assert matched_tokens_delta(100, 130) == 0
    with pytest.raises(ValueError):
        matched_tokens_delta(-1, 0)


def test_resume_load_block_ids_slices_boundary_crossing_suffix() -> None:
    # block_size=16, n_c_t=32 at pause -> exactly 2 blocks saved. The
    # save-confirm lag (the request stays RUNNING for the confirm step) then
    # advances 3 more tokens to n_c_t=35, crossing into a 3rd block, so
    # v0.11.2's update_state_after_alloc hands the connector 3 blocks (the
    # FULL prefix) though only 2 are covered by the retained save.
    saved_block_count = 2
    new_block_ids = [10, 11, 12]  # full prefix the scheduler allocated
    assert resume_load_block_ids(new_block_ids, saved_block_count) == [10, 11]


def test_resume_load_block_ids_no_lag_is_identity() -> None:
    # No save-confirm lag -> the scheduler's block set matches the save exactly.
    assert resume_load_block_ids([5, 6, 7], 3) == [5, 6, 7]


def test_resume_load_block_ids_rejects_under_allocation() -> None:
    # Fewer blocks than the save covers is a real bug (would under-load), not
    # a lag artifact -- fail fast rather than silently truncate the save.
    with pytest.raises(ValueError, match="fewer than the 3"):
        resume_load_block_ids([1, 2], 3)


def test_pausebook_state_machine() -> None:
    pb = PauseBook()
    pb.mark_pausing("r", 50)
    assert pb.is_pausing("r") and not pb.is_paused("r")
    # Double-pause is a bug -> fail fast.
    with pytest.raises(ValueError, match="already pausing"):
        pb.mark_pausing("r", 50)
    assert pb.confirm_saved("r") == 50
    assert pb.is_paused("r") and not pb.is_pausing("r")
    assert pb.resume("r") == 50
    assert not pb.is_paused("r")


def test_pausebook_rejects_bad_transitions() -> None:
    pb = PauseBook()
    with pytest.raises(ValueError, match="not pausing"):
        pb.confirm_saved("ghost")
    with pytest.raises(ValueError, match="not paused"):
        pb.resume("ghost")
    with pytest.raises(ValueError, match="computed tokens"):
        pb.mark_pausing("r", 0)


def test_pausebook_drop_finished_liveness() -> None:
    pb = PauseBook()
    pb.mark_pausing("a", 10)
    pb.mark_pausing("b", 20)
    pb.confirm_saved("b")
    # a finished while pausing, b finished while paused -> both forgotten.
    pb.drop_finished({"a", "b"})
    assert not pb.is_pausing("a") and not pb.is_paused("b")


def test_pausebook_deferred_resume_honored_after_confirm() -> None:
    # RESUME arriving while the save is still in flight must not be dropped
    # (would hang the request paused forever) -- it is honored the instant
    # confirm_saved lands.
    pb = PauseBook()
    pb.mark_pausing("r", 10)
    pb.defer_resume("r")
    assert pb.pop_deferred_resume("r") is False  # not confirmed yet -> not due
    pb.confirm_saved("r")
    assert pb.pop_deferred_resume("r") is True  # now due, exactly once
    assert pb.pop_deferred_resume("r") is False  # cleared, no double-fire


def test_pausebook_defer_resume_requires_pausing() -> None:
    pb = PauseBook()
    with pytest.raises(ValueError, match="not pausing"):
        pb.defer_resume("ghost")


def test_pausebook_drop_finished_clears_deferred_resume() -> None:
    pb = PauseBook()
    pb.mark_pausing("r", 10)
    pb.defer_resume("r")
    pb.drop_finished({"r"})
    assert not pb.is_pausing("r")
    # No leaked deferred-resume flag for a request that never gets confirmed.
    pb.mark_pausing("r", 10)
    assert pb.pop_deferred_resume("r") is False


def test_guard_pause_trigger_contains_exception_instead_of_propagating() -> None:
    # This is the exact GPU crash: assert_saved_covers_tokens raised inside
    # the PAUSE trigger and the exception escaped schedule(), killing
    # EngineCore for every co-tenant. guard_pause_trigger must catch it and
    # hand it back for logging -- never let it propagate.
    def bad_pause() -> None:
        assert_saved_covers_tokens(1581, block_count=97, block_size=16)

    exc = guard_pause_trigger(bad_pause)
    assert isinstance(exc, ValueError)
    assert "do not cover" in str(exc)


def test_guard_pause_trigger_returns_none_on_success() -> None:
    calls = []
    assert guard_pause_trigger(lambda: calls.append(1)) is None
    assert calls == [1]  # the action actually ran


def test_register_pause_save_uses_fresh_block_ids_not_stale_admission_cache() -> None:
    # Exercises the REAL production method (spike/vllm_connector/gpu.py),
    # off-GPU: construct via object.__new__ to skip __init__'s vllm/torch
    # requirement, wiring only the attributes register_pause_save touches --
    # same class the GPU box crashed in, same numbers from that crash.
    from spike.vllm_connector.gpu import SelectiveOffloadConnector

    conn = object.__new__(SelectiveOffloadConnector)
    conn._saved = SavedKVRegistry()
    conn._queued = []
    conn._block_size = 16

    fresh_block_ids = list(range(99))  # kv_cache_manager.get_block_ids(...) at pause
    block_count = conn.register_pause_save("agent-0", 1581, fresh_block_ids)

    assert block_count == 99
    assert conn._saved.saved["agent-0"] == (1581, 99)
    phase, req_id, directive_block_ids = conn._queued[0]
    assert phase == OffloadPhase.PAUSE.value
    assert req_id == "agent-0"
    assert directive_block_ids == fresh_block_ids  # the directive carries the FRESH count

    # The stale admission count (97) that crashed the GPU box must still
    # fail fast -- register_pause_save no longer has a self._block_ids
    # fallback to silently prefer, so this can only happen if a caller
    # explicitly (and wrongly) passes the stale count.
    conn2 = object.__new__(SelectiveOffloadConnector)
    conn2._saved = SavedKVRegistry()
    conn2._queued = []
    conn2._block_size = 16
    with pytest.raises(ValueError, match=r"saved 97 blocks do not cover 1581 tokens"):
        conn2.register_pause_save("agent-0", 1581, list(range(97)))
    assert conn2._queued == []  # rejected before queuing anything


def test_retention_transfer_record_carries_cross_process_timestamps(tmp_path) -> None:
    from spike.vllm_connector.gpu import SelectiveOffloadConnector

    path = tmp_path / "transfers.jsonl"
    connector = object.__new__(SelectiveOffloadConnector)
    connector._timing_path = str(path)
    timing = TransferTiming(bytes_moved=1024, milliseconds=2.0, num_blocks=4)

    connector._record(
        "retention_restore",
        timing,
        request_id="deadline:1:2",
        program_id="program:1",
        started_monotonic_s=10.0,
        completed_monotonic_s=10.002,
    )

    row = json.loads(path.read_text())
    assert row["program_id"] == "program:1"
    assert row["started_monotonic_s"] == 10.0
    assert row["completed_monotonic_s"] == 10.002


def test_retention_scheduler_event_preserves_block_ids_and_pool_delta(tmp_path) -> None:
    from spike.vllm_connector.scheduler import RetentionScheduler

    path = tmp_path / "scheduler.jsonl"
    scheduler = object.__new__(RetentionScheduler)
    scheduler._retention_events_path = str(path)

    scheduler._record_retention_event(
        "retention_blocks_freed",
        request_id="deadline:1:1",
        block_ids=[3, 5],
        free_blocks_before=7,
        free_blocks_after=9,
    )

    row = json.loads(path.read_text())
    assert row["phase"] == "retention_blocks_freed"
    assert row["request_id"] == "deadline:1:1"
    assert row["block_ids"] == [3, 5]
    assert row["free_blocks_after"] - row["free_blocks_before"] == 2
    assert isinstance(row["monotonic_s"], float)


def test_saved_kv_registry_lifecycle() -> None:
    reg = SavedKVRegistry()
    assert reg.matched_tokens("unknown", 0) == (0, False)  # base-connector fallthrough
    reg.register("r", num_tokens_saved=100, block_count=7)
    assert reg.is_saved("r")
    # Delta-only, sync load.
    assert reg.matched_tokens("r", 0) == (100, False)
    assert reg.matched_tokens("r", 40) == (60, False)
    reg.record_resume_blocks("r", [1, 2, 3])
    assert reg.resume_blocks["r"] == [1, 2, 3]
    reg.drop("r")
    assert not reg.is_saved("r")
    assert reg.matched_tokens("r", 0) == (0, False)


def test_saved_kv_registry_rejects_bad_input() -> None:
    reg = SavedKVRegistry()
    with pytest.raises(ValueError, match="must be > 0"):
        reg.register("r", 0, 1)
    with pytest.raises(ValueError, match="unknown saved request"):
        reg.record_resume_blocks("ghost", [1])


def test_save_confirmation_reader_cursor(tmp_path) -> None:
    path = tmp_path / "timings.jsonl"
    reader = SaveConfirmationReader(str(path))
    assert reader.poll() == set()  # no file yet
    path.write_text(
        json.dumps({"phase": "offload", "request_id": None}) + "\n"
        + json.dumps({"phase": "pause_saved", "request_id": "agent-0"}) + "\n"
    )
    assert reader.poll() == {"agent-0"}  # only pause_saved rows, offload ignored
    assert reader.poll() == set()  # cursor advanced -> no re-report
    with path.open("a") as f:
        f.write(json.dumps({"phase": "pause_saved", "request_id": "agent-1"}) + "\n")
    assert reader.poll() == {"agent-1"}  # only the newly appended row


def test_fake_backend_retained_save_load_lifecycle() -> None:
    be = FakeTransferBackend(bytes_per_block=1024)
    save = be.save_retained("agent-0", [0, 1, 2])
    assert save.num_blocks == 3
    assert "agent-0" in be.retained  # survives eviction, out of the staging pool
    # A pause-save must not be re-issued for the same request.
    with pytest.raises(ValueError, match="already has a retained save"):
        be.save_retained("agent-0", [0, 1, 2])
    load = be.load_retained("agent-0", [7, 8, 9])  # resumed into fresh blocks
    assert load.num_blocks == 3
    assert "agent-0" not in be.retained  # pinned buffer freed post-resume


def test_fake_backend_retained_rejects_unknown_and_mismatch() -> None:
    be = FakeTransferBackend(bytes_per_block=64)
    with pytest.raises(ValueError, match="no retained save"):
        be.load_retained("ghost", [0])
    be.save_retained("r", [0, 1])
    with pytest.raises(ValueError, match="!= saved"):
        be.load_retained("r", [5])  # count must match the saved block count


def test_fake_backend_free_retained_releases_buffer() -> None:
    be = FakeTransferBackend(bytes_per_block=64)
    be.save_retained("r", [0, 1])
    assert "r" in be.retained
    be.free_retained("r")
    assert "r" not in be.retained
    # No-op (not an error) if nothing was retained -- finish/abort races are
    # exactly when this gets called on an id that may already be released.
    be.free_retained("r")
    be.free_retained("never-saved")


def test_pause_resume_full_cycle_composition() -> None:
    # Exercises the exact dance PausableScheduler + connector run on the GPU
    # box, but through the CPU-tested pure objects + fake backend, so the
    # composition is guarded off-GPU (the real wiring is vllm-only).
    block_size, n_c_t = 16, 100
    block_ids = list(range(7))  # ceil(100/16) = 7 blocks tracked at alloc
    pb, reg = PauseBook(), SavedKVRegistry()
    be = FakeTransferBackend(bytes_per_block=2 * 1024 * 1024)

    # PAUSE: assert coverage, register save, mark pausing, worker saves.
    assert_saved_covers_tokens(n_c_t, len(block_ids), block_size)
    reg.register("agent", n_c_t, len(block_ids))
    pb.mark_pausing("agent", n_c_t)
    be.save_retained("agent", block_ids)

    # Worker confirms save -> scheduler frees + holds (force_preempt).
    assert pb.confirm_saved("agent") == n_c_t
    assert pb.is_paused("agent")

    # RESUME under real pressure (no local hit): connector reports the whole
    # save as the delta; request re-allocated into fresh blocks; worker loads.
    assert pb.resume("agent") == n_c_t
    delta, load_async = reg.matched_tokens("agent", 0)
    assert delta == n_c_t and load_async is False
    new_blocks = list(range(20, 27))  # 7 fresh blocks for the delta tokens
    reg.record_resume_blocks("agent", new_blocks)
    be.load_retained("agent", new_blocks)
    reg.drop("agent")

    # Fully unwound: no held state, no leaked pinned buffer.
    assert not pb.is_paused("agent") and not reg.is_saved("agent")
    assert be.retained == {}


def test_pause_finish_while_paused_releases_retained_buffer() -> None:
    # Mirrors the connector's _drop_finished: a request that finishes/aborts
    # while paused has no resume coming, so its retained buffer must be
    # explicitly released (a "release" directive), not left to leak forever.
    pb, reg = PauseBook(), SavedKVRegistry()
    be = FakeTransferBackend(bytes_per_block=1024)

    pb.mark_pausing("agent", 32)
    reg.register("agent", 32, block_count=2)
    be.save_retained("agent", [0, 1])
    pb.confirm_saved("agent")  # force_preempt confirms -> paused, held

    # Request finishes while paused (e.g. cancelled, or hit an EOS token that
    # never should have fired but did) -- _drop_finished's cleanup path.
    assert reg.is_saved("agent")
    be.free_retained("agent")  # the release directive the worker executes
    reg.drop("agent")
    pb.drop_finished({"agent"})

    assert not reg.is_saved("agent")
    assert not pb.is_paused("agent")
    assert be.retained == {}  # no leaked pinned buffer


def test_pause_low_pressure_self_hit_releases_retained_buffer() -> None:
    # Mirrors update_state_after_alloc's num_external_tokens==0 branch: the
    # local prefix cache already covers the saved tokens (memo Q2 self-hit
    # under low pressure), so there is nothing to load -- but the retained
    # buffer must still be released, not silently kept alive forever.
    pb, reg = PauseBook(), SavedKVRegistry()
    be = FakeTransferBackend(bytes_per_block=1024)

    pb.mark_pausing("agent", 32)
    reg.register("agent", 32, block_count=2)
    be.save_retained("agent", [0, 1])
    pb.confirm_saved("agent")
    pb.resume("agent")

    # num_external_tokens == 0 -> no RESUME load directive, just a release.
    num_external_tokens = 0
    assert num_external_tokens == 0  # documents the branch this test exercises
    be.free_retained("agent")
    reg.drop("agent")

    assert not reg.is_saved("agent")
    assert be.retained == {}


def test_pause_result_schema() -> None:
    res = PauseResult(
        tool_duration_s=3.0,
        blocks_freed=97,
        pause_to_freed_ms=20.0,
        resume_to_first_token_ms=15.0,
        identical=True,
    )
    d = res.to_dict()
    assert d["blocks_freed"] == 97
    assert d["identical"] is True
    assert json.loads(json.dumps(d))["pause_to_freed_ms"] == 20.0
