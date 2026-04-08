"""FastAPI routes for the dynamic Gantt viewer backend."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request

from demo.gantt_viewer.backend.cc_cache import load_or_import
from demo.gantt_viewer.backend.discovery import DiscoveryState
from demo.gantt_viewer.backend.payload import (
    DEFAULT_MARKER_REGISTRY,
    DEFAULT_SPAN_REGISTRY,
    build_gantt_payload_multi,
)
from demo.gantt_viewer.backend.schema import (
    GanttPayload,
    HealthResponse,
    PayloadRequest,
    Registries,
    TraceDescriptor,
    TraceListResponse,
)
from trace_collect.trace_inspector import TraceData


router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_endpoint(request: Request) -> HealthResponse:
    state = _get_discovery_state(request)
    return HealthResponse(status="ok", n_discovered=len(state.descriptors))


@router.get("/traces", response_model=TraceListResponse)
def list_traces_endpoint(request: Request) -> TraceListResponse:
    state = _get_discovery_state(request)
    return TraceListResponse(traces=state.descriptors, registries=_build_registries())


@router.post("/traces/reload", response_model=TraceListResponse)
def reload_traces_endpoint(request: Request) -> TraceListResponse:
    state = _get_discovery_state(request)
    state.reload()
    return TraceListResponse(traces=state.descriptors, registries=_build_registries())


@router.post("/payload", response_model=GanttPayload)
async def payload_endpoint(
    payload_request: PayloadRequest,
    request: Request,
) -> GanttPayload:
    state = _get_discovery_state(request)
    descriptors = _resolve_descriptors(state, payload_request.ids)
    labels_by_id = {descriptor.id: descriptor.label for descriptor in descriptors}
    traces: list[tuple[str, TraceData]] = []

    for descriptor in descriptors:
        try:
            trace_path = await _resolve_trace_path(descriptor)
            trace_data = TraceData.load(trace_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "trace_id": descriptor.id,
                    "stage": "cc_import"
                    if descriptor.source_format == "claude-code"
                    else "trace_load",
                    "error": str(exc),
                },
            ) from exc
        traces.append((descriptor.id, trace_data))

    raw_payload = build_gantt_payload_multi(
        traces,
        span_registry=deepcopy(DEFAULT_SPAN_REGISTRY),
        marker_registry=_build_marker_registry_payload(),
    )
    for trace_payload in raw_payload["traces"]:
        trace_payload["label"] = labels_by_id[trace_payload["id"]]
    return GanttPayload.model_validate(raw_payload)


def _get_discovery_state(request: Request) -> DiscoveryState:
    return request.app.state.discovery_state


async def _resolve_trace_path(descriptor: TraceDescriptor) -> Path:
    if descriptor.source_format == "v5":
        return Path(descriptor.path)
    return await anyio.to_thread.run_sync(load_or_import, Path(descriptor.path))


def _resolve_descriptors(
    state: DiscoveryState,
    trace_ids: list[str],
) -> list[TraceDescriptor]:
    missing = [trace_id for trace_id in trace_ids if trace_id not in state.descriptors_by_id]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "unknown trace ids", "trace_ids": missing},
        )
    return [state.descriptors_by_id[trace_id] for trace_id in trace_ids]


def _build_registries() -> Registries:
    return Registries(
        spans=deepcopy(DEFAULT_SPAN_REGISTRY),
        markers=_build_marker_registry_payload(),
    )


def _build_marker_registry_payload() -> dict[str, dict[str, str]]:
    marker_registry = deepcopy(DEFAULT_MARKER_REGISTRY)
    for key, value in marker_registry.items():
        value.setdefault("label", _marker_label(key))
    return marker_registry


def _marker_label(key: str) -> str:
    normalized = key.lstrip("_").replace("_", " ").strip()
    return normalized.title() if normalized else "Default"
