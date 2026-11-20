"""FastAPI application factory for ArtSeek v2."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import AppConfig
from ..ml.llm import OllamaClient
from ..services.common import ServiceError
from .deps import AppContainer
from .routers import ALL_ROUTERS


def create_app(
    config: AppConfig,
    *,
    container: AppContainer | None = None,
    embedder_factory: Callable[[], Any] | None = None,
    llm: OllamaClient | None = None,
) -> FastAPI:
    """Create the modular v2 FastAPI app with lifespan management."""
    app_container = container or AppContainer(
        config,
        embedder_factory=embedder_factory,
        llm=llm,
    )
    owns_container = container is None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if owns_container:
                app_container.close()
            else:
                app_container.close()

    app = FastAPI(
        title="ArtSeek",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.container = app_container
    app.state.config = app_container.config

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    for router in ALL_ROUTERS:
        app.include_router(router)

    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
