import json
import os
from pathlib import Path
import subprocess

from scripts.evaluation.build_physical_state_template import (
    bounded_template,
    opened_paths,
)


def test_opened_paths_uses_only_resolved_successful_file_accesses() -> None:
    lines = [
        'openat(AT_FDCWD</testbed>, "/missing", O_RDONLY) = -1 ENOENT',
        'openat(AT_FDCWD</testbed>, "module.py", O_RDONLY) = 3</testbed/module.py>',
        '[pid 11] execve("/usr/bin/python3", ["python3"], 0x0) = 0',
        '[pid 11] openat(AT_FDCWD</testbed>, "/proc/stat", O_RDONLY) = 3</proc/stat>',
        '11 openat(AT_FDCWD</testbed>, "same", O_RDONLY) = 4</testbed/module.py>',
        '11 execve("relative", ["relative"], 0x0) = 0',
    ]

    assert opened_paths(lines) == ["/testbed/module.py", "/usr/bin/python3"]


def test_bounded_template_filters_and_stops_before_byte_limit(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    empty = tmp_path / "empty"
    first.write_bytes(b"1234")
    second.write_bytes(b"5678")
    empty.write_bytes(b"")

    result = bounded_template(
        [str(first), str(empty), str(tmp_path / "missing"), str(second)],
        max_paths=10,
        max_bytes=6,
    )

    assert result["files"] == [{"path": str(first), "size_bytes": 4}]
    assert result["truncated_by"] == "total_bytes"
    assert result["rejected"] == {
        "missing": 1,
        "not_regular": 0,
        "empty": 1,
        "changed_path": 0,
        "duplicate_inode": 0,
    }


def test_static_probe_reports_warm_residency(tmp_path) -> None:
    source = Path("scripts/evaluation/physical_state_probe.c")
    binary = tmp_path / "probe"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-std=c11",
            "-D_POSIX_C_SOURCE=200809L",
            "-static",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = tmp_path / "payload"
    payload.write_bytes(b"x" * 8192)
    descriptor = os.open(payload, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    template = tmp_path / "template.tsv"
    template.write_text(f"8192\t{payload}\n", encoding="utf-8")

    results = {}
    for condition in ("warm", "cold"):
        completed = subprocess.run(
            [str(binary), condition, str(template)],
            check=True,
            capture_output=True,
            text=True,
        )
        results[condition] = json.loads(completed.stdout)

    assert results["cold"]["condition"] == "cold"
    assert results["warm"]["condition"] == "warm"
    assert results["warm"]["file_count"] == 1
    assert results["warm"]["total_pages"] == 2
    assert results["warm"]["resident_pages"] == 2
    assert results["warm"]["resident_fraction"] == 1.0
    assert results["cold"]["resident_pages"] < results["warm"]["resident_pages"]

    too_many = tmp_path / "too-many.tsv"
    too_many.write_text(f"1\t{payload}\n" * 4097, encoding="utf-8")
    oversized = tmp_path / "oversized.tsv"
    oversized.write_text(f"536870913\t{payload}\n", encoding="utf-8")
    for invalid in (too_many, oversized):
        rejected = subprocess.run(
            [str(binary), "cold", str(invalid)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "template exceeds frozen bounds" in rejected.stderr
