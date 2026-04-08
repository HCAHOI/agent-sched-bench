"""Tests for the FastAPI backend scaffold."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo.gantt_viewer.backend import cc_cache
from demo.gantt_viewer.backend.app import create_app
from demo.gantt_viewer.backend.payload import (
    DEFAULT_MARKER_REGISTRY,
    build_gantt_payload_multi,
)
from demo.gantt_viewer.backend.schema import MarkerDef
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


def test_payload_partial_failure_returns_200_with_errors(tmp_path: Path) -> None:
    """One good + one bad id: good trace returned in traces, bad in errors."""
    good_path = write_v5_trace(
        tmp_path / "runs" / "good-repo" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    bad_path = tmp_path / "runs" / "bad-repo" / "trace.jsonl"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text(
        json.dumps({
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 3,
            "max_iterations": 1,
        }) + "\n",
        encoding="utf-8",
    )

    config_path = write_config(tmp_path / "config.yaml", [str(good_path), str(bad_path)])
    client = TestClient(
        create_app(
            config_path=config_path,
            runtime_state_path=tmp_path / "runtime-state.json",
        )
    )

    descriptors = client.get("/api/traces").json()["traces"]
    ids = [descriptor["id"] for descriptor in descriptors]
    assert len(ids) == 2

    response = client.post("/api/payload", json={"ids": ids})
    assert response.status_code == 200
    body = response.json()
    assert len(body["traces"]) == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["stage"] == "trace_load"
    assert body["traces"][0]["metadata"]["scaffold"] == "openclaw"


def test_payload_all_failed_returns_422(tmp_path: Path) -> None:
    """If every requested id fails, the endpoint returns 422 (no partial success)."""
    bad_path = tmp_path / "runs" / "only-bad" / "trace.jsonl"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text(
        json.dumps({
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 2,
            "max_iterations": 1,
        }) + "\n",
        encoding="utf-8",
    )

    config_path = write_config(tmp_path / "config.yaml", [str(bad_path)])
    client = TestClient(
        create_app(
            config_path=config_path,
            runtime_state_path=tmp_path / "runtime-state.json",
        )
    )

    ids = [descriptor["id"] for descriptor in client.get("/api/traces").json()["traces"]]
    response = client.post("/api/payload", json={"ids": ids})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail["errors"]) == 1


def test_default_marker_registry_matches_schema() -> None:
    """Every DEFAULT_MARKER_REGISTRY entry must pass MarkerDef validation directly."""
    assert DEFAULT_MARKER_REGISTRY, "registry must not be empty"
    for key, entry in DEFAULT_MARKER_REGISTRY.items():
        assert set(entry.keys()) >= {"symbol", "color", "label"}, (
            f"{key} missing required fields: {entry}"
        )
        validated = MarkerDef.model_validate(entry)
        assert validated.label, f"{key} has empty label"


AC2_CC_GLOB = REPO_ROOT / "traces" / "swe-rebench" / "claude-code-haiku"


@pytest.mark.slow
def test_payload_cc_bulk_11_real_fixtures(tmp_path: Path, monkeypatch) -> None:
    """AC2: simultaneously load all 11 real claude-code-haiku traces via /api/payload."""
    if not AC2_CC_GLOB.exists():
        pytest.skip(f"AC2 fixtures not present at {AC2_CC_GLOB}")

    cc_paths = sorted(AC2_CC_GLOB.glob("*/attempt_1/trace.jsonl"))
    if len(cc_paths) < 11:
        pytest.skip(f"expected 11 AC2 fixtures, found {len(cc_paths)}")

    monkeypatch.setattr(cc_cache, "CACHE_ROOT", tmp_path / "cache")

    config_path = write_config(
        tmp_path / "config.yaml",
        [str(path) for path in cc_paths],
    )
    client = TestClient(
        create_app(
            config_path=config_path,
            runtime_state_path=tmp_path / "runtime-state.json",
        )
    )

    descriptors = client.get("/api/traces").json()["traces"]
    ids = [descriptor["id"] for descriptor in descriptors]
    assert len(ids) == 11

    first_start = time.perf_counter()
    response = client.post("/api/payload", json={"ids": ids})
    first_elapsed = time.perf_counter() - first_start

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["traces"]) == 11
    assert body.get("errors", []) == []
    scaffolds = {trace["metadata"]["scaffold"] for trace in body["traces"]}
    assert scaffolds == {"claude-code"}

    second_start = time.perf_counter()
    second = client.post("/api/payload", json={"ids": ids})
    second_elapsed = time.perf_counter() - second_start
    assert second.status_code == 200
    assert second_elapsed < first_elapsed, (
        f"cache-hit call ({second_elapsed:.3f}s) should be faster than first import ({first_elapsed:.3f}s)"
    )


def test_openapi_frozen(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GANTT_VIEWER_CONFIG", raising=False)
    monkeypatch.delenv("GANTT_VIEWER_DEV", raising=False)
    # Isolate the test from the real repo config + user home cache state file.
    # The snapshot is API-shape only, but discovery config must be valid —
    # use a glob pointing at nothing under tmp_path so it resolves to zero
    # descriptors without touching the live repo traces.
    empty_config = write_config(
        tmp_path / "config.yaml",
        [str(tmp_path / "nothing" / "*.jsonl")],
    )
    client = TestClient(
        create_app(
            config_path=empty_config,
            runtime_state_path=tmp_path / "runtime-state.json",
        )
    )
    response = client.get("/openapi.json")
    assert response.status_code == 200

    actual = json.dumps(response.json(), indent=2, sort_keys=True)
    expected = OPENAPI_SNAPSHOT.read_text(encoding="utf-8").strip()
    assert actual == expected, (
        "OpenAPI snapshot drifted. Regenerate "
        "demo/gantt_viewer/tests/fixtures/openapi.snapshot.json and "
        "demo/gantt_viewer/frontend/src/api/schema.gen.ts together."
    )


def test_payload_cc_cache_hit_does_not_reimport(tmp_path: Path, monkeypatch) -> None:
    """Two /api/payload calls with the same CC id → claude_code_import runs exactly once."""
    client, _, cc_path = _make_client(tmp_path)
    monkeypatch.setattr(cc_cache, "CACHE_ROOT", tmp_path / "cc-cache")

    call_count = {"n": 0}
    real_import = cc_cache.import_claude_code_session

    def counting_import(**kwargs):
        call_count["n"] += 1
        return real_import(**kwargs)

    monkeypatch.setattr(cc_cache, "import_claude_code_session", counting_import)

    cc_id = next(
        trace["id"]
        for trace in client.get("/api/traces").json()["traces"]
        if trace["source_format"] == "claude-code"
    )

    first = client.post("/api/payload", json={"ids": [cc_id]})
    assert first.status_code == 200
    assert call_count["n"] == 1

    second = client.post("/api/payload", json={"ids": [cc_id]})
    assert second.status_code == 200
    assert call_count["n"] == 1, "cache hit should not re-invoke claude_code_import"
