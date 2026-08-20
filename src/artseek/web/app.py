from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..config import AppConfig
from ..image_io import make_thumbnail
from ..search import SearchEngine


class TextSearchRequest(BaseModel):
    query: str = ""
    reference_id: str | None = None
    limit: int = Field(default=40, ge=1, le=200)
    expand: bool = True
    strict_tags: bool = False
    image_weight: float = Field(default=0.65, ge=0, le=1)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="ArtSeek", docs_url="/api/docs", redoc_url=None)
    app.state.config = config
    app.state.engine = SearchEngine(config)
    static_path = Path(__file__).with_name("static") / "index.html"

    @app.on_event("shutdown")
    def shutdown() -> None:
        app.state.engine.close()

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(static_path.read_text(encoding="utf-8"))

    @app.post("/api/search")
    def search(request: TextSearchRequest) -> dict:
        image: Path | None = None
        if request.reference_id:
            payload = app.state.engine.store.get(request.reference_id)
            if not payload:
                raise HTTPException(404, "参考图片不在索引中")
            image = Path(payload["path"])
        try:
            response = app.state.engine.search(
                request.query,
                image=image,
                limit=request.limit,
                expand=request.expand,
                strict_tags=request.strict_tags,
                image_weight=request.image_weight,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return dataclasses.asdict(response)

    @app.post("/api/search-upload")
    async def search_upload(
        file: UploadFile = File(...),
        query: str = Form(default=""),
        limit: int = Form(default=40),
        expand: bool = Form(default=True),
        strict_tags: bool = Form(default=False),
        image_weight: float = Form(default=0.65),
    ) -> dict:
        suffix = Path(file.filename or "query.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(await file.read())
        try:
            response = app.state.engine.search(
                query,
                image=temp_path,
                limit=max(1, min(limit, 200)),
                expand=expand,
                strict_tags=strict_tags,
                image_weight=max(0.0, min(image_weight, 1.0)),
            )
            return dataclasses.asdict(response)
        finally:
            temp_path.unlink(missing_ok=True)

    def resolve_item(point_id: str) -> tuple[dict, Path]:
        payload = app.state.engine.store.get(point_id)
        if not payload:
            raise HTTPException(404, "图片不在索引中")
        path = Path(payload["path"])
        if not path.exists():
            raise HTTPException(404, "原图片已移动或删除，请重新索引")
        return payload, path

    @app.get("/thumb/{point_id}")
    def thumbnail(point_id: str) -> FileResponse:
        _, source = resolve_item(point_id)
        destination = config.thumbnail_dir / f"{point_id}.webp"
        if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
            make_thumbnail(source, destination)
        return FileResponse(destination, media_type="image/webp")

    @app.get("/image/{point_id}")
    def full_image(point_id: str) -> FileResponse:
        _, path = resolve_item(point_id)
        return FileResponse(path)

    @app.get("/api/item/{point_id}")
    def item(point_id: str) -> dict:
        payload, _ = resolve_item(point_id)
        return payload

    return app
