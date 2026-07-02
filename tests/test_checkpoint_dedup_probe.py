from __future__ import annotations

import tarfile
from pathlib import Path


from scripts.checkpoint_dedup_probe import _scan_tar, _merge_stats, _iter_tar_paths


def _make_tar(
    path: Path,
    files: dict[str, bytes],
) -> None:
    """Create a simple PAX tar with given file contents."""
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tf:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.type = tarfile.REGTYPE
            tf.addfile(info, io := __import__("io").BytesIO(content))
            io.close()


def test_scan_tar_counts_file_bytes(tmp_path: Path) -> None:
    p = tmp_path / "test-after.tar"
    _make_tar(p, {"a.txt": b"hello", "b.txt": b"world"})
    result = _scan_tar(p)
    assert result["tar_bytes"] > 0
    assert result["total_file_bytes"] == 10  # 5+5
    assert sum(result["file_hashes"].values()) == 2


def test_scan_tar_deduplicates_content(tmp_path: Path) -> None:
    p = tmp_path / "dup-after.tar"
    content = b"same content"
    _make_tar(p, {"a.txt": content, "b.txt": content})
    result = _scan_tar(p)
    assert result["total_file_bytes"] == 24  # 12+12
    hashes = result["file_hashes"]
    assert len(hashes) == 1
    assert sum(hashes.values()) == 2


def test_scan_tar_skips_directories(tmp_path: Path) -> None:
    p = tmp_path / "dirs-after.tar"
    with tarfile.open(p, "w", format=tarfile.PAX_FORMAT) as tf:
        # directory entry
        dir_info = tarfile.TarInfo("mydir")
        dir_info.type = tarfile.DIRTYPE
        tf.addfile(dir_info)
        # file entry
        file_info = tarfile.TarInfo("mydir/f.txt")
        content = b"hello"
        file_info.size = len(content)
        file_info.type = tarfile.REGTYPE
        tf.addfile(file_info, __import__("io").BytesIO(content))
    result = _scan_tar(p)
    assert result["total_file_bytes"] == 5  # only the file
    assert sum(result["file_hashes"].values()) == 1


def test_merge_stats_empty() -> None:
    merged = _merge_stats([])
    assert merged["n_tars"] == 0
    assert merged["total_file_bytes"] == 0


def test_merge_stats_cross_tar_dedup(tmp_path: Path) -> None:
    content = b"shared content"
    # First tar with two files: one unique, one shared
    p1 = tmp_path / "t1-after.tar"
    _make_tar(p1, {"unique1.txt": b"unique one", "shared.txt": content})
    # Second tar with two files: one unique, one shared
    p2 = tmp_path / "t2-after.tar"
    _make_tar(p2, {"unique2.txt": b"unique two", "shared.txt": content})

    r1 = _scan_tar(p1)
    r2 = _scan_tar(p2)
    merged = _merge_stats([r1, r2])

    # total file bytes = 10 + 14 + 10 + 14 = 48
    # unique = 10 (unique1) + 14 (shared) + 10 (unique2) = 34
    assert merged["total_file_bytes"] == 48
    assert merged["unique_bytes"] == 34  # 10 + 14 + 10
    assert merged["global_unique_hashes"] == 3
    assert merged["global_total_hashes"] == 4


def test_merge_stats_top_duplicated(tmp_path: Path) -> None:
    big = b"x" * 1000
    p1 = tmp_path / "t1-after.tar"
    _make_tar(p1, {"big.txt": big})
    p2 = tmp_path / "t2-after.tar"
    _make_tar(p2, {"big2.txt": big})

    r1, r2 = _scan_tar(p1), _scan_tar(p2)
    merged = _merge_stats([r1, r2])
    assert len(merged["top_duplicated"]) == 1
    assert merged["top_duplicated"][0]["count"] == 2
    assert merged["top_duplicated"][0]["size"] == 1000


def test_iter_tar_paths(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "tool_0-after.tar").write_text("", encoding="utf-8")
    (tmp_path / "checkpoints" / "tool_1-after.tar").write_text("", encoding="utf-8")
    (tmp_path / "other").mkdir(parents=True)
    (tmp_path / "other" / "tool_2-after.tar").write_text("", encoding="utf-8")
    paths = _iter_tar_paths(tmp_path)
    assert len(paths) == 2  # only under checkpoints/


def test_sha256_consistent(tmp_path: Path) -> None:
    """Verify that identical content produces same hash across tars."""
    p1 = tmp_path / "a-after.tar"
    p2 = tmp_path / "b-after.tar"
    _make_tar(p1, {"f.txt": b"hello world"})
    _make_tar(p2, {"g.txt": b"hello world"})
    r1 = _scan_tar(p1)
    r2 = _scan_tar(p2)
    hash1 = set(r1["file_hashes"].keys())
    hash2 = set(r2["file_hashes"].keys())
    assert hash1 == hash2
