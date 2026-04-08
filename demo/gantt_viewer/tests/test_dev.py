"""Tests for the backend CLI launcher helpers."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

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

    fake_process = SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(dev, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(dev, "_spawn_vite_dev_server", lambda: fake_process)
    monkeypatch.setattr(dev, "_terminate_process", lambda process: captured.setdefault("terminated", process))
    monkeypatch.setattr(dev, "_wait_for_vite_startup", lambda: captured.setdefault("waited", True))
    monkeypatch.setattr(dev.uvicorn, "run", fake_run)
    monkeypatch.delenv("GANTT_VIEWER_CONFIG", raising=False)
    monkeypatch.delenv("GANTT_VIEWER_DEV", raising=False)

    dev.main(["--config", str(config_path), "--clear-cache", "--port", "9999"])

    assert os.environ["GANTT_VIEWER_CONFIG"] == str(config_path.resolve())
    assert os.environ["GANTT_VIEWER_DEV"] == "0"
    assert not cache_root.exists()
    assert captured["app"] == "demo.gantt_viewer.backend.app:create_app"
    assert captured["factory"] is True
    assert captured["port"] == 9999


def test_dev_mode_spawns_vite(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("groups: []\n", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_process = SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(dev, "_spawn_vite_dev_server", lambda: fake_process)
    monkeypatch.setattr(dev, "_wait_for_vite_startup", lambda: captured.setdefault("waited", True))
    monkeypatch.setattr(dev, "_terminate_process", lambda process: captured.setdefault("terminated", process))
    monkeypatch.setattr(
        dev.uvicorn,
        "run",
        lambda app, **kwargs: captured.update({"app": app, **kwargs}),
    )

    dev.main(["--dev", "--config", str(config_path)])

    assert os.environ["GANTT_VIEWER_DEV"] == "1"
    assert captured["waited"] is True
    assert captured["terminated"] is fake_process
    assert captured["reload"] is False
