"""FastAPI routes for the dynamic Gantt viewer backend."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request, UploadFile

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
    TracePayload,
    UploadTraceResponse,
)
from demo.gantt_viewer.backend.uploads import build_upload_id, persist_upload
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


@router.post("/traces/upload", response_model=UploadTraceResponse)
async def upload_trace_endpoint(
    file: UploadFile,
    request: Request,
) -> UploadTraceResponse:
    filename = file.filename or "upload.jsonl"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail={"message": "empty upload"})

    upload_path = persist_upload(filename, content)
    source_format = _sniff_or_422(upload_path)
    descriptor = TraceDescriptor(
        id=build_upload_id(filename, content),
        label=Path(filename).stem or "upload",
        source_format=source_format,
        path=str(upload_path.resolve()),
        size_bytes=upload_path.stat().st_size,
        mtime=upload_path.stat().st_mtime,
    )

    trace_payload = await _build_trace_payload(descriptor)
    _get_discovery_state(request).register_descriptor(descriptor)
    return UploadTraceResponse(descriptor=descriptor, payload_fragment=trace_payload)


@router.post("/payload", response_model=GanttPayload)
async def payload_endpoint(
    payload_request: PayloadRequest,
    request: Request,
) -> GanttPayload:
    state = _get_discovery_state(request)
    descriptors = _resolve_descriptors(state, payload_request.ids)
    traces = [await _build_trace_payload(descriptor) for descriptor in descriptors]
    return GanttPayload(
        registries=_build_registries(),
        traces=traces,
    )


def _get_discovery_state(request: Request) -> DiscoveryState:
    return request.app.state.discovery_state


async def _resolve_trace_path(descriptor: TraceDescriptor) -> Path:
    if descriptor.source_format == "v5":
        return Path(descriptor.path)
    return await anyio.to_thread.run_sync(load_or_import, Path(descriptor.path))


async def _build_trace_payload(descriptor: TraceDescriptor) -> TracePayload:
    try:
        trace_path = await _resolve_trace_path(descriptor)
        trace_data = TraceData.load(trace_path)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "trace_id": descriptor.id,
                "stage": "cc_import" if descriptor.source_format == "claude-code" else "trace_load",
                "error": str(exc),
            },
        ) from exc

    raw_payload = build_gantt_payload_multi(
        [(descriptor.id, trace_data)],
        span_registry=deepcopy(DEFAULT_SPAN_REGISTRY),
        marker_registry=_build_marker_registry_payload(),
    )
    raw_payload["traces"][0]["label"] = descriptor.label
    return TracePayload.model_validate(raw_payload["traces"][0])


def _sniff_or_422(path: Path) -> str:
    from demo.gantt_viewer.backend.discovery import sniff_format

    try:
        return sniff_format(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc


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
