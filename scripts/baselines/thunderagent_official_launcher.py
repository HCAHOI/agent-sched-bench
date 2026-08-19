from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from ThunderAgent.app import register_routes
from ThunderAgent.config import Config, set_config
from ThunderAgent.scheduler import MultiBackendRouter


def main() -> None:
    backends = [
        value.strip()
        for value in os.environ.get(
            "THUNDERAGENT_BACKENDS", "http://127.0.0.1:8000"
        ).split(",")
        if value.strip()
    ]
    profile_dir = os.environ.get(
        "THUNDERAGENT_PROFILE_DIR", "./thunderagent_profiles"
    )
    config = Config(
        backends=backends,
        router_mode="tr",
        backend_type="vllm",
        profile_enabled=True,
        profile_dir=profile_dir,
        metrics_enabled=True,
        metrics_interval=5.0,
        scheduler_interval=5.0,
        acting_token_weight=1.0,
        use_acting_token_decay=True,
    )
    set_config(config)
    router = MultiBackendRouter(
        backends,
        profile_enabled=True,
        scheduling_enabled=True,
        scheduler_interval=5.0,
        backend_type="vllm",
        acting_token_weight=1.0,
        use_acting_token_decay=True,
    )
    app = FastAPI(title="ThunderAgent official baseline")
    app.add_event_handler("startup", router.start)
    app.add_event_handler("shutdown", router.stop)
    register_routes(app, router, config)
    uvicorn.run(
        app,
        host=os.environ.get("THUNDERAGENT_HOST", "127.0.0.1"),
        port=int(os.environ.get("THUNDERAGENT_PORT", "9000")),
    )


if __name__ == "__main__":
    main()
