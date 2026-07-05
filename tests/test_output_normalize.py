from __future__ import annotations

import pytest

from trace_collect.output_normalize import normalize_tool_output


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pid 12345", "pid <N>"),
        ("port 8080", "port <N>"),
        ("count 42 items", "count <N> items"),
        ("error code 500", "error code <N>"),
        ("line 1: foo", "line <N>: foo"),
        ("version 3.14.2", "version <N>.<N>.<N>"),
        ("100", "<N>"),
        ("0", "<N>"),
    ],
)
def test_digit_runs_to_N(text: str, expected: str) -> None:
    assert normalize_tool_output(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("addr 0x7ffdeadbeef", "addr <HEX>"),
        ("0xABCDEF", "<HEX>"),
        ("pointer=0x1a2b3c4d", "pointer=<HEX>"),
        ("0x0", "<HEX>"),
        ("0xDEADBEEF and 0xCAFE", "<HEX> and <HEX>"),
    ],
)
def test_hex_addresses_to_HEX(text: str, expected: str) -> None:
    assert normalize_tool_output(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-07-05T12:34:56Z", "<TS>"),
        ("at 2025-01-01T00:00:00", "at <TS>"),
        ("2024-12-31 23:59:59", "<TS>"),
        ("2026-01-15T08:30:45.123456Z", "<TS>"),
        ("2026-01-15T08:30:45+05:30", "<TS>"),
        ("2026-01-15T08:30:45-04:00", "<TS>"),
    ],
)
def test_iso_timestamps_to_TS(text: str, expected: str) -> None:
    assert normalize_tool_output(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("epoch 1782470400", "epoch <TS>"),
        ("timestamp=1712345678", "timestamp=<TS>"),
        ("1782470400.123456", "<TS>"),
        ("2000000000", "<TS>"),
    ],
)
def test_epoch_timestamps_to_TS(text: str, expected: str) -> None:
    assert normalize_tool_output(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("file at /tmp/run-456/a.txt", "file at <TMP>"),
        ("/tmp/some_random_dir/file.log", "<TMP>"),
        ("see /tmp/work for logs", "see <TMP> for logs"),
    ],
)
def test_tmp_paths_to_TMP(text: str, expected: str) -> None:
    assert normalize_tool_output(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("check /proc/789/status", "check <PROC>"),
        ("reading /proc/12345/maps", "reading <PROC>"),
        ("/proc/1/cmdline", "<PROC>"),
        ("from /proc/999", "from <PROC>"),
    ],
)
def test_proc_paths_to_PROC(text: str, expected: str) -> None:
    assert normalize_tool_output(text) == expected


def test_already_normalized_text_unchanged() -> None:
    already = "pid <N> at <TS> addr <HEX> tmp <TMP> proc <PROC>"
    assert normalize_tool_output(already) == already


def test_empty_string() -> None:
    assert normalize_tool_output("") == ""


def test_whitespace_only() -> None:
    assert normalize_tool_output("   \t\n  ") == "   \t\n  "


def test_overlapping_patterns_hex_and_timestamps() -> None:
    text = (
        "build 0x7f400010 at 2026-07-05T12:00:00Z epoch 1782470400 "
        "temp /tmp/build-789/proc /proc/123/status"
    )
    expected = (
        "build <HEX> at <TS> epoch <TS> temp <TMP> <PROC>"
    )
    assert normalize_tool_output(text) == expected


def test_real_world_example_matching_simulate_mismatch_stats() -> None:
    """Matches the pattern used in test_simulate_mismatch_stats."""
    source = (
        "pid 123 at 2026-07-05T12:34:56Z epoch 1782470400 "
        "addr 0x7ffdeadbeef tmp /tmp/run-456/a.txt proc /proc/789/status"
    )
    replay = (
        "pid 999 at 2027-08-06T01:02:03Z epoch 1782470999 "
        "addr 0xabc tmp /tmp/run-000/a.txt proc /proc/111/status"
    )

    assert normalize_tool_output(source) == normalize_tool_output(replay)
    assert normalize_tool_output(source) == (
        "pid <N> at <TS> epoch <TS> addr <HEX> tmp <TMP> proc <PROC>"
    )


def test_plain_text_passes_through() -> None:
    assert normalize_tool_output("hello world") == "hello world"
    assert normalize_tool_output("build succeeded") == "build succeeded"


def test_non_hex_numeric_prefixes_not_matched() -> None:
    assert normalize_tool_output("size 0x not hex") == "size <N>x not hex"


def test_non_proc_digit_paths_not_matched_as_proc() -> None:
    assert normalize_tool_output("dir /procfile/123") == "dir /procfile/<N>"
