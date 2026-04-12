"""FastAPI routes for the dynamic Gantt viewer backend."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from demo.gantt_viewer.backend.discovery import REPO_ROOT
from demo.gantt_viewer.backend.payload import (
    DEFAULT_MARKER_REGISTRY,
    DEFAULT_SPAN_REGISTRY,
    build_gantt_payload_multi,
)
from demo.gantt_viewer.backend.ingest import (
    CanonicalizedTrace,
    ensure_canonical_trace_path,
)
from demo.gantt_viewer.backend.runtime_registry import (
    RuntimeRegistryConflictError,
    RuntimeTraceRegistry,
)
from demo.gantt_viewer.backend.schema import (
    ExportHtmlRequest,
    GanttPayload,
    HealthResponse,
    PayloadError,
    PayloadRequest,
    RegisterTracesRequest,
    RegisterTracesResponse,
    Registries,
    SnapshotBootstrapData,
    SnapshotPayload,
    TraceDescriptor,
    TraceListResponse,
    TracePayload,
    UnregisterTracesRequest,
    UnregisterTracesResponse,
    UploadTraceResponse,
)
from demo.gantt_viewer.backend.uploads import build_upload_id, persist_upload
from trace_collect.trace_inspector import TraceData


class _TracePayloadError(Exception):
    """Internal exception carrying structured partial-failure context."""

    def __init__(self, trace_id: str, stage: str, error: str) -> None:
        super().__init__(error)
        self.trace_id = trace_id
        self.stage = stage
        self.error = error


router = APIRouter(prefix="/api")
FRONTEND_DIST_PATH = REPO_ROOT / "demo" / "gantt_viewer" / "frontend" / "dist"
_ASSET_LINK_RE = re.compile(
    r'<link(?P<attrs>[^>]+)href="(?P<href>/assets/[^"]+)"(?P<rest>[^>]*)>'
)
_ASSET_SCRIPT_RE = re.compile(
    r'<script(?P<attrs>[^>]*)src="(?P<src>/assets/[^"]+\.js)"(?P<rest>[^>]*)></script>'
)
_FONT_LINK_RE = re.compile(r"\s*<link[^>]+fonts\.googleapis\.com[^>]*>\s*")
_PRECONNECT_LINK_RE = re.compile(r'\s*<link[^>]+rel="preconnect"[^>]*>\s*')


@router.get("/health", response_model=HealthResponse)
def health_endpoint(request: Request) -> HealthResponse:
    trace_registry = _get_trace_registry(request)
    return HealthResponse(status="ok", n_discovered=len(trace_registry.descriptors))


@router.get("/traces", response_model=TraceListResponse)
def list_traces_endpoint(request: Request) -> TraceListResponse:
    return _build_trace_list_response(_get_trace_registry(request))


@router.post("/traces/reload", response_model=TraceListResponse)
def reload_traces_endpoint(request: Request) -> TraceListResponse:
    trace_registry = _get_trace_registry(request)
    trace_registry.reload()
    return _build_trace_list_response(trace_registry)


@router.post("/traces/register", response_model=RegisterTracesResponse)
def register_traces_endpoint(
    register_request: RegisterTracesRequest,
    request: Request,
) -> RegisterTracesResponse:
    trace_registry = _get_trace_registry(request)
    try:
        registered = trace_registry.register_paths(
            register_request.paths,
            labels_by_path=register_request.labels_by_path,
        )
    except RuntimeRegistryConflictError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    return RegisterTracesResponse(registered=registered)


@router.post("/traces/unregister", response_model=UnregisterTracesResponse)
def unregister_traces_endpoint(
    unregister_request: UnregisterTracesRequest,
    request: Request,
) -> UnregisterTracesResponse:
    trace_registry = _get_trace_registry(request)
    removed_ids, missing_ids = trace_registry.unregister_ids(unregister_request.ids)
    return UnregisterTracesResponse(
        removed_ids=removed_ids,
        missing_ids=missing_ids,
    )


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
    canonicalized = _canonicalize_or_422(upload_path)
    descriptor = TraceDescriptor(
        id=build_upload_id(filename, content),
        label=Path(filename).stem or "upload",
        source_format=canonicalized.source_format,
        path=str(canonicalized.canonical_path),
        size_bytes=canonicalized.canonical_path.stat().st_size,
        mtime=canonicalized.canonical_path.stat().st_mtime,
    )

    try:
        trace_payload = await _build_trace_payload(descriptor)
    except _TracePayloadError as exc:
        raise _trace_payload_http_exception(exc) from exc
    _get_trace_registry(request).register_uploaded_descriptor(descriptor)
    return UploadTraceResponse(descriptor=descriptor, payload_fragment=trace_payload)


@router.post("/payload", response_model=GanttPayload)
async def payload_endpoint(
    payload_request: PayloadRequest,
    request: Request,
) -> GanttPayload:
    trace_registry = _get_trace_registry(request)
    descriptors = _resolve_descriptors(trace_registry, payload_request.ids)

    traces: list[TracePayload] = []
    errors: list[PayloadError] = []
    for descriptor in descriptors:
        try:
            traces.append(await _build_trace_payload(descriptor))
        except _TracePayloadError as exc:
            errors.append(
                PayloadError(trace_id=exc.trace_id, stage=exc.stage, error=exc.error)
            )

    if not traces and errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "all requested traces failed",
                "errors": [e.model_dump() for e in errors],
            },
        )

    return GanttPayload(
        registries=_build_registries(),
        traces=traces,
        errors=errors,
    )


@router.post("/export/html", response_class=HTMLResponse)
def export_html_endpoint(export_request: ExportHtmlRequest) -> HTMLResponse:
    snapshot_bootstrap = _build_snapshot_bootstrap(export_request.snapshot)
    document = _build_export_document(snapshot_bootstrap)
    return HTMLResponse(content=document, media_type="text/html")


def _get_trace_registry(request: Request) -> RuntimeTraceRegistry:
    return request.app.state.trace_registry


def _build_trace_list_response(
    trace_registry: RuntimeTraceRegistry,
) -> TraceListResponse:
    return TraceListResponse(
        traces=trace_registry.descriptors, registries=_build_registries()
    )


def _trace_payload_http_exception(exc: _TracePayloadError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"trace_id": exc.trace_id, "stage": exc.stage, "error": exc.error},
    )


async def _build_trace_payload(descriptor: TraceDescriptor) -> TracePayload:
    try:
        trace_data = TraceData.load(Path(descriptor.path))
    except Exception as exc:
        raise _TracePayloadError(
            trace_id=descriptor.id,
            stage="trace_load",
            error=str(exc),
        ) from exc

    try:
        raw_payload = build_gantt_payload_multi(
            [(descriptor.id, trace_data)],
            span_registry=deepcopy(DEFAULT_SPAN_REGISTRY),
            marker_registry=deepcopy(DEFAULT_MARKER_REGISTRY),
        )
        raw_payload["traces"][0]["label"] = descriptor.label
        return TracePayload.model_validate(raw_payload["traces"][0])
    except Exception as exc:
        raise _TracePayloadError(
            trace_id=descriptor.id,
            stage="payload_build",
            error=str(exc),
        ) from exc


def _canonicalize_or_422(path: Path) -> CanonicalizedTrace:
    try:
        return ensure_canonical_trace_path(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc


def _resolve_descriptors(
    trace_registry: RuntimeTraceRegistry,
    trace_ids: list[str],
) -> list[TraceDescriptor]:
    descriptors = [trace_registry.get_descriptor(trace_id) for trace_id in trace_ids]
    missing = [
        trace_id
        for trace_id, descriptor in zip(trace_ids, descriptors, strict=False)
        if descriptor is None
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "unknown trace ids", "trace_ids": missing},
        )
    return [descriptor for descriptor in descriptors if descriptor is not None]


def _build_registries() -> Registries:
    return Registries(
        spans=deepcopy(DEFAULT_SPAN_REGISTRY),
        markers=deepcopy(DEFAULT_MARKER_REGISTRY),
    )


def _build_snapshot_bootstrap(
    snapshot_payload: SnapshotPayload,
) -> SnapshotBootstrapData:
    trace_ids = [trace.id for trace in snapshot_payload.traces]
    duplicate_ids = sorted(
        trace_id for trace_id, count in Counter(trace_ids).items() if count > 1
    )
    if duplicate_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "snapshot traces must have unique ids",
                "trace_ids": duplicate_ids,
            },
        )
    return SnapshotBootstrapData(
        payload=snapshot_payload,
        trace_ids=trace_ids,
        visible_trace_ids=trace_ids,
    )


def _build_export_document(snapshot_bootstrap: SnapshotBootstrapData) -> str:
    index_path = FRONTEND_DIST_PATH / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "frontend build index is missing",
                "path": str(index_path),
            },
        )

    index_html = index_path.read_text(encoding="utf-8")
    inlined_html = _inline_frontend_assets(index_html)
    snapshot_json = _escape_script_text(
        json.dumps(
            snapshot_bootstrap.model_dump(mode="json"),
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    bootstrap_block = (
        '<script id="gantt-viewer-snapshot-bootstrap" type="application/json">'
        f"{snapshot_json}</script>"
        "<script>"
        'window.__GANTT_VIEWER_BOOTSTRAP__=JSON.parse(document.getElementById("gantt-viewer-snapshot-bootstrap").textContent);'
        "</script>"
    )
    if "</body>" in inlined_html:
        return inlined_html.replace("</body>", f"{bootstrap_block}</body>", 1)
    return f"{inlined_html}{bootstrap_block}"


def _inline_frontend_assets(index_html: str) -> str:
    css_inlined = False
    js_inlined = False

    def replace_link(match: re.Match[str]) -> str:
        nonlocal css_inlined
        href = match.group("href")
        if not href.endswith(".css"):
            return ""
        css_inlined = True
        css_text = _read_dist_asset(href)
        return f"<style>\n{css_text}\n</style>"

    def replace_script(match: re.Match[str]) -> str:
        nonlocal js_inlined
        js_inlined = True
        script_text = _escape_script_text(_read_dist_asset(match.group("src")))
        return f'<script type="module">\n{script_text}\n</script>'

    inlined_html = _ASSET_LINK_RE.sub(replace_link, index_html)
    inlined_html = _ASSET_SCRIPT_RE.sub(replace_script, inlined_html)
    inlined_html = _FONT_LINK_RE.sub("", inlined_html)
    inlined_html = _PRECONNECT_LINK_RE.sub("", inlined_html)

    if not css_inlined or not js_inlined:
        missing = []
        if not css_inlined:
            missing.append("css")
        if not js_inlined:
            missing.append("js")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "frontend build assets missing from index.html",
                "missing": missing,
                "path": str(FRONTEND_DIST_PATH / "index.html"),
            },
        )
    return inlined_html


def _read_dist_asset(asset_href: str) -> str:
    asset_path = FRONTEND_DIST_PATH / asset_href.removeprefix("/")
    if not asset_path.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "frontend build asset is missing",
                "path": str(asset_path),
            },
        )
    return asset_path.read_text(encoding="utf-8")


def _escape_script_text(value: str) -> str:
    return value.replace("</", "<\\/")
