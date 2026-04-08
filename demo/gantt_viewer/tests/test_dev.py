"""Tests for the backend CLI launcher helpers."""

from __future__ import annotations

import os
from pathlib import Path

from demo.gantt_viewer.backend import dev


def test_dev_main_exports_config_and_clears_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("groups: []\n", encoding="utf-8")

    cache_root = tmp_path / "cache-root"
    cache_root.mkdir()
    (cache_root / "stale.jsonl").write_text("old", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(dev, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(dev.uvicorn, "run", fake_run)
    monkeypatch.delenv("GANTT_VIEWER_CONFIG", raising=False)

    dev.main(["--config", str(config_path), "--clear-cache", "--port", "9999"])

    assert os.environ["GANTT_VIEWER_CONFIG"] == str(config_path.resolve())
    assert not cache_root.exists()
    assert captured["app"] == "demo.gantt_viewer.backend.app:create_app"
    assert captured["factory"] is True
    assert captured["port"] == 9999
