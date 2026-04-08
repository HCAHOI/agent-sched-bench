"""Tests for the FastAPI backend scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from demo.gantt_viewer.backend import cc_cache
from demo.gantt_viewer.backend.app import create_app
from demo.gantt_viewer.backend.payload import build_gantt_payload_multi
from demo.gantt_viewer.tests.helpers import (
    write_config,
    write_v5_trace,
)
from trace_collect.trace_inspector import TraceData


REPO_ROOT = Path(__file__).resolve().parents[3]
CC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"
OPENAPI_SNAPSHOT = REPO_ROOT / "demo" / "gantt_viewer" / "tests" / "fixtures" / "openapi.snapshot.json"


def _llm_action() -> dict:
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "agent-1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {
            "raw_response": {
                "choices": [{"message": {"content": "hello from llm"}}]
            }
        },
    }


def _make_client(tmp_path: Path) -> tuple[TestClient, Path, Path]:
    runtime_state_path = tmp_path / "runtime-state.json"
    v5_path = write_v5_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    cc_path = CC_FIXTURE
    config_path = write_config(
        tmp_path / "config.yaml",
        [str(v5_path), str(cc_path)],
    )
    client = TestClient(
        create_app(
            config_path=config_path,
            runtime_state_path=runtime_state_path,
        )
    )
    return client, v5_path, cc_path


def test_health_endpoint_reports_discovered_traces(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "n_discovered": 2}


def test_list_traces_returns_descriptors_and_registries(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    response = client.get("/api/traces")
    assert response.status_code == 200

    body = response.json()
    assert len(body["traces"]) == 2
    assert {trace["source_format"] for trace in body["traces"]} == {
        "v5",
        "claude-code",
    }
    assert body["registries"]["spans"]["llm"]["label"] == "LLM Call"
    assert body["registries"]["markers"]["message_dispatch"]["label"] == "Message Dispatch"


def test_payload_endpoint_returns_v5_payload(tmp_path: Path) -> None:
    client, v5_path, _ = _make_client(tmp_path)
    response = client.post("/api/payload", json={"ids": ["ac1-repo__issue-1"]})
    assert response.status_code == 200

    body = response.json()
    assert len(body["traces"]) == 1
    assert body["traces"][0]["id"] == "ac1-repo__issue-1"
    assert body["traces"][0]["label"] == "repo__issue-1"
    assert body["traces"][0]["metadata"]["scaffold"] == "openclaw"

    direct_payload = build_gantt_payload_multi(
        [("ac1-repo__issue-1", TraceData.load(v5_path))]
    )
    direct_payload["traces"][0]["label"] = "repo__issue-1"
    assert body["traces"] == direct_payload["traces"]


def test_payload_endpoint_rejects_unknown_id(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    response = client.post("/api/payload", json={"ids": ["missing-id"]})
    assert response.status_code == 404
    assert response.json()["detail"]["trace_ids"] == ["missing-id"]


def test_payload_endpoint_imports_claude_code_trace(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _make_client(tmp_path)
    monkeypatch.setattr(cc_cache, "CACHE_ROOT", tmp_path / "cache")
    traces = client.get("/api/traces").json()["traces"]
    cc_id = next(trace["id"] for trace in traces if trace["source_format"] == "claude-code")

    response = client.post("/api/payload", json={"ids": [cc_id]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["traces"]) == 1
    assert body["traces"][0]["metadata"]["scaffold"] == "claude-code"
    assert len(body["traces"][0]["lanes"]) >= 1


def test_reload_endpoint_rewalks_discovery(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    response = client.post("/api/traces/reload")
    assert response.status_code == 200
    assert len(response.json()["traces"]) == 2


def test_register_v5_trace_path_adds_descriptor(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    extra_path = write_v5_trace(
        tmp_path / "runtime" / "repo__issue-2" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )

    response = client.post(
        "/api/traces/register",
        json={"paths": [str(extra_path)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["registered"][0]["source_format"] == "v5"

    traces = client.get("/api/traces").json()["traces"]
    assert any(trace["path"] == str(extra_path.resolve()) for trace in traces)


def test_register_claude_code_trace_path_adds_descriptor(tmp_path: Path) -> None:
    client, _, cc_path = _make_client(tmp_path)
    suppressed_id = next(
        trace["id"]
        for trace in client.get("/api/traces").json()["traces"]
        if trace["path"] == str(cc_path.resolve())
    )
    client.post("/api/traces/unregister", json={"ids": [suppressed_id]})

    response = client.post(
        "/api/traces/register",
        json={"paths": [str(cc_path)]},
    )
    assert response.status_code == 200
    assert response.json()["registered"][0]["id"] == suppressed_id


def test_register_duplicate_config_path_returns_409(tmp_path: Path) -> None:
    client, v5_path, _ = _make_client(tmp_path)
    response = client.post(
        "/api/traces/register",
        json={"paths": [str(v5_path)]},
    )
    assert response.status_code == 409


def test_unregister_runtime_trace_removes_descriptor(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    extra_path = write_v5_trace(
        tmp_path / "runtime" / "repo__issue-2" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    registered = client.post(
        "/api/traces/register",
        json={"paths": [str(extra_path)]},
    ).json()["registered"][0]

    response = client.post(
        "/api/traces/unregister",
        json={"ids": [registered["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["removed_ids"] == [registered["id"]]
    traces = client.get("/api/traces").json()["traces"]
    assert all(trace["id"] != registered["id"] for trace in traces)


def test_unregister_config_trace_persists_across_reload(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    config_id = "ac1-repo__issue-1"

    response = client.post("/api/traces/unregister", json={"ids": [config_id]})
    assert response.status_code == 200
    assert response.json()["removed_ids"] == [config_id]

    traces = client.get("/api/traces").json()["traces"]
    assert all(trace["id"] != config_id for trace in traces)

    reloaded = client.post("/api/traces/reload").json()["traces"]
    assert all(trace["id"] != config_id for trace in reloaded)


def test_runtime_registered_trace_persists_across_app_restart(tmp_path: Path) -> None:
    runtime_state_path = tmp_path / "runtime-state.json"
    v5_path = write_v5_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    extra_path = write_v5_trace(
        tmp_path / "runtime" / "repo__issue-2" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    config_path = write_config(tmp_path / "config.yaml", [str(v5_path)])

    client = TestClient(create_app(config_path=config_path, runtime_state_path=runtime_state_path))
    registered = client.post(
        "/api/traces/register",
        json={"paths": [str(extra_path)]},
    ).json()["registered"][0]

    restarted = TestClient(
        create_app(config_path=config_path, runtime_state_path=runtime_state_path)
    )
    traces = restarted.get("/api/traces").json()["traces"]
    assert any(trace["id"] == registered["id"] for trace in traces)


def test_suppressed_config_trace_persists_across_app_restart(tmp_path: Path) -> None:
    runtime_state_path = tmp_path / "runtime-state.json"
    v5_path = write_v5_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    config_path = write_config(tmp_path / "config.yaml", [str(v5_path)])

    client = TestClient(create_app(config_path=config_path, runtime_state_path=runtime_state_path))
    response = client.post("/api/traces/unregister", json={"ids": ["ac1-repo__issue-1"]})
    assert response.status_code == 200

    restarted = TestClient(
        create_app(config_path=config_path, runtime_state_path=runtime_state_path)
    )
    traces = restarted.get("/api/traces").json()["traces"]
    assert all(trace["id"] != "ac1-repo__issue-1" for trace in traces)


def test_uploaded_trace_persists_across_app_restart(tmp_path: Path) -> None:
    runtime_state_path = tmp_path / "runtime-state.json"
    v5_path = write_v5_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    config_path = write_config(tmp_path / "config.yaml", [str(v5_path)])

    client = TestClient(create_app(config_path=config_path, runtime_state_path=runtime_state_path))
    upload_path = write_v5_trace(
        tmp_path / "upload" / "persisted_upload.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )

    with upload_path.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": (upload_path.name, handle, "application/json")},
        )

    assert response.status_code == 200
    uploaded_id = response.json()["descriptor"]["id"]

    restarted = TestClient(
        create_app(config_path=config_path, runtime_state_path=runtime_state_path)
    )
    traces = restarted.get("/api/traces").json()["traces"]
    assert any(trace["id"] == uploaded_id for trace in traces)


def test_upload_v5_trace_registers_descriptor(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    upload_path = write_v5_trace(
        tmp_path / "upload" / "demo_trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )

    with upload_path.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": (upload_path.name, handle, "application/json")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["descriptor"]["source_format"] == "v5"
    assert body["payload_fragment"]["metadata"]["scaffold"] == "openclaw"

    payload_response = client.post("/api/payload", json={"ids": [body["descriptor"]["id"]]})
    assert payload_response.status_code == 200


def test_upload_claude_code_trace_registers_descriptor(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _make_client(tmp_path)
    monkeypatch.setattr(cc_cache, "CACHE_ROOT", tmp_path / "cache")

    with CC_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": ("claude_code_minimal.jsonl", handle, "application/json")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["descriptor"]["source_format"] == "claude-code"
    assert body["payload_fragment"]["metadata"]["scaffold"] == "claude-code"

    payload_response = client.post("/api/payload", json={"ids": [body["descriptor"]["id"]]})
    assert payload_response.status_code == 200


def test_upload_malformed_trace_returns_422(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    response = client.post(
        "/api/traces/upload",
        files={"file": ("broken.jsonl", b"not jsonl", "application/octet-stream")},
    )
    assert response.status_code == 422


def test_openapi_frozen(monkeypatch) -> None:
    monkeypatch.delenv("GANTT_VIEWER_CONFIG", raising=False)
    monkeypatch.delenv("GANTT_VIEWER_DEV", raising=False)
    client = TestClient(create_app())
    response = client.get("/openapi.json")
    assert response.status_code == 200

    actual = json.dumps(response.json(), indent=2, sort_keys=True)
    expected = OPENAPI_SNAPSHOT.read_text(encoding="utf-8").strip()
    assert actual == expected, (
        "OpenAPI snapshot drifted. Regenerate "
        "demo/gantt_viewer/tests/fixtures/openapi.snapshot.json and "
        "demo/gantt_viewer/frontend/src/api/schema.gen.ts together."
    )
