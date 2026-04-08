"""FastAPI application factory for the dynamic Gantt viewer."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from demo.gantt_viewer.backend.discovery import DiscoveryState, REPO_ROOT
from demo.gantt_viewer.backend.routes import router


DEFAULT_CONFIG_PATH = REPO_ROOT / "demo" / "gantt_viewer" / "configs" / "example.yaml"


def create_app(*, config_path: str | Path | None = None) -> FastAPI:
    """Create the backend application."""
    resolved_config = Path(
        config_path
        or os.environ.get("GANTT_VIEWER_CONFIG")
        or DEFAULT_CONFIG_PATH
    ).resolve()

    app = FastAPI(
        title="Gantt Viewer API",
        version="0.1.0",
    )
    app.state.discovery_state = DiscoveryState.from_config_path(resolved_config)
    app.state.config_path = resolved_config
    app.include_router(router)
    return app
