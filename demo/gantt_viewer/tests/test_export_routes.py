"""Tests for standalone HTML snapshot export."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from demo.gantt_viewer.backend.app import create_app
from demo.gantt_viewer.tests.helpers import write_config


def _make_client(tmp_path: Path) -> TestClient:
    config_path = write_config(
        tmp_path / "config.yaml",
        [str(tmp_path / "nothing" / "*.jsonl")],
    )
    return TestClient(
        create_app(
            config_path=config_path,
            runtime_state_path=tmp_path / "runtime-state.json",
        )
    )


def _set_dist_path(monkeypatch, dist_path: Path) -> None:
    monkeypatch.setattr(
        "demo.gantt_viewer.backend.routes.FRONTEND_DIST_PATH", dist_path
    )


def _write_dist_bundle(
    dist_path: Path,
    *,
    css_name: str = "index.css",
    js_name: str = "index.js",
) -> None:
    assets_path = dist_path / "assets"
    assets_path.mkdir(parents=True, exist_ok=True)
    (dist_path / "index.html").write_text(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link
      href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"
      rel="stylesheet"
    />
    <title>Trace Gantt</title>
    <script type="module" crossorigin src="/assets/{js_name}"></script>
    <link rel="stylesheet" crossorigin href="/assets/{css_name}">
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
""".strip().format(css_name=css_name, js_name=js_name),
        encoding="utf-8",
    )
    (assets_path / css_name).write_text("body { color: #fff; }\n", encoding="utf-8")
    (assets_path / js_name).write_text(
        "console.log('snapshot bundle');\n", encoding="utf-8"
    )


def _snapshot_request() -> dict[str, object]:
    return {
        "registries": {
            "spans": {"llm": {"color": "#00E5FF", "label": "LLM Call", "order": 0}},
            "markers": {
                "message_dispatch": {
                    "symbol": "diamond",
                    "color": "#76FF03",
                    "label": "Message Dispatch",
                }
            },
        },
        "traces": [
            {
                "id": "client-imported-2",
                "label": "Imported trace 2",
                "metadata": {
                    "scaffold": "client-import",
                    "model": "demo-model",
                    "instance_id": "import-2",
                    "mode": None,
                    "max_iterations": 4,
                    "n_actions": 1,
                    "n_iterations": 1,
                    "n_events": 0,
                    "elapsed_s": 1.0,
                },
                "t0": 1000.0,
                "lanes": [
                    {
                        "agent_id": "agent-1",
                        "spans": [
                            {
                                "type": "llm",
                                "start": 0.0,
                                "end": 1.0,
                                "start_abs": 1000.0,
                                "end_abs": 1001.0,
                                "iteration": 0,
                                "detail": {},
                            }
                        ],
                        "markers": [],
                    }
                ],
            },
            {
                "id": "client-imported-1",
                "label": "Imported trace 1",
                "metadata": {
                    "scaffold": "client-import",
                    "model": "demo-model",
                    "instance_id": "import-1",
                    "mode": None,
                    "max_iterations": 4,
                    "n_actions": 1,
                    "n_iterations": 1,
                    "n_events": 0,
                    "elapsed_s": 1.0,
                },
                "t0": 2000.0,
                "lanes": [
                    {
                        "agent_id": "agent-2",
                        "spans": [
                            {
                                "type": "llm",
                                "start": 0.0,
                                "end": 1.5,
                                "start_abs": 2000.0,
                                "end_abs": 2001.5,
                                "iteration": 0,
                                "detail": {},
                            }
                        ],
                        "markers": [],
                    }
                ],
            },
        ],
        "errors": [],
    }


def _snapshot_request_with_duplicate_ids() -> dict[str, object]:
    snapshot = _snapshot_request()
    traces = snapshot["traces"]
    assert isinstance(traces, list)
    traces[1] = {**traces[1], "id": "client-imported-2"}
    return snapshot


def _extract_bootstrap(html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="gantt-viewer-snapshot-bootstrap" type="application/json">(?P<json>.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group("json"))


def test_export_html_accepts_client_loaded_snapshot_and_inlines_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_dist_bundle(tmp_path / "dist")
    _set_dist_path(monkeypatch, tmp_path / "dist")
    client = _make_client(tmp_path)

    response = client.post("/api/export/html", json={"snapshot": _snapshot_request()})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "<style>" in html
    assert '<script type="module">' in html
    assert 'src="/assets/' not in html
    assert 'href="/assets/' not in html
    assert 'type="module" src=' not in html
    assert "fonts.googleapis.com" not in html

    bootstrap = _extract_bootstrap(html)
    assert bootstrap["mode"] == "snapshot"
    assert bootstrap["trace_ids"] == ["client-imported-2", "client-imported-1"]
    assert bootstrap["visible_trace_ids"] == ["client-imported-2", "client-imported-1"]
    assert [trace["id"] for trace in bootstrap["payload"]["traces"]] == [
        "client-imported-2",
        "client-imported-1",
    ]


def test_export_html_rejects_empty_snapshot_payload(
    tmp_path: Path, monkeypatch
) -> None:
    _write_dist_bundle(tmp_path / "dist")
    _set_dist_path(monkeypatch, tmp_path / "dist")
    client = _make_client(tmp_path)

    response = client.post(
        "/api/export/html",
        json={
            "snapshot": {
                "registries": {"spans": {}, "markers": {}},
                "traces": [],
                "errors": [],
            }
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "traces"


def test_export_html_fails_clearly_when_build_asset_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_dist_bundle(tmp_path / "dist", css_name="missing.css")
    (tmp_path / "dist" / "assets" / "missing.css").unlink()
    _set_dist_path(monkeypatch, tmp_path / "dist")
    client = _make_client(tmp_path)

    response = client.post("/api/export/html", json={"snapshot": _snapshot_request()})

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "frontend build asset is missing"
    assert response.json()["detail"]["path"].endswith("dist/assets/missing.css")


def test_export_html_rejects_duplicate_trace_ids(tmp_path: Path, monkeypatch) -> None:
    _write_dist_bundle(tmp_path / "dist")
    _set_dist_path(monkeypatch, tmp_path / "dist")
    client = _make_client(tmp_path)

    response = client.post(
        "/api/export/html", json={"snapshot": _snapshot_request_with_duplicate_ids()}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "snapshot traces must have unique ids",
        "trace_ids": ["client-imported-2"],
    }


def test_export_html_fails_clearly_when_index_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    dist_path = tmp_path / "dist"
    dist_path.mkdir(parents=True, exist_ok=True)
    _set_dist_path(monkeypatch, dist_path)
    client = _make_client(tmp_path)

    response = client.post("/api/export/html", json={"snapshot": _snapshot_request()})

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "frontend build index is missing"
    assert response.json()["detail"]["path"].endswith("dist/index.html")


def test_export_html_fails_clearly_when_js_asset_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    _write_dist_bundle(tmp_path / "dist", js_name="missing.js")
    (tmp_path / "dist" / "assets" / "missing.js").unlink()
    _set_dist_path(monkeypatch, tmp_path / "dist")
    client = _make_client(tmp_path)

    response = client.post("/api/export/html", json={"snapshot": _snapshot_request()})

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "frontend build asset is missing"
    assert response.json()["detail"]["path"].endswith("dist/assets/missing.js")
