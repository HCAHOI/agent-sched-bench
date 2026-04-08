"""Typed API schema for the dynamic Gantt viewer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpanDef(BaseModel):
    color: str
    label: str
    order: int


class MarkerDef(BaseModel):
    symbol: str
    color: str
    label: str


class Registries(BaseModel):
    spans: dict[str, SpanDef]
    markers: dict[str, MarkerDef]


class Span(BaseModel):
    type: str
    start: float
    end: float
    start_abs: float
    end_abs: float
    iteration: int
    detail: dict[str, Any]


class Marker(BaseModel):
    type: str
    event: str
    t: float
    t_abs: float
    iteration: int
    detail: dict[str, Any]


class Lane(BaseModel):
    agent_id: str
    spans: list[Span]
    markers: list[Marker]


class TraceMetadata(BaseModel):
    scaffold: str
    model: str | None = None
    instance_id: str = ""
    mode: str | None = None
    max_iterations: int | None = None
    n_actions: int
    n_iterations: int
    n_events: int
    elapsed_s: float | None = None


class TracePayload(BaseModel):
    id: str
    label: str
    metadata: TraceMetadata
    t0: float
    lanes: list[Lane]


class GanttPayload(BaseModel):
    registries: Registries
    traces: list[TracePayload]


class TraceDescriptor(BaseModel):
    id: str
    label: str
    source_format: Literal["v5", "claude-code"]
    path: str
    size_bytes: int
    mtime: float


class TraceListResponse(BaseModel):
    traces: list[TraceDescriptor]
    registries: Registries


class HealthResponse(BaseModel):
    status: str
    n_discovered: int


class PayloadRequest(BaseModel):
    ids: list[str] = Field(min_length=1)
