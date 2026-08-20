from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..config import AppConfig
from ..image_io import canonical_path, make_thumbnail
from ..indexer import ImageIndexer
from ..library import (
    apply_scope,
    library_payload,
    load_scope,
    rebuild_catalog,
    save_scope,
)
from ..search import SearchEngine


def _parse_id_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "exclude_ids 不是合法 JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
        raise HTTPException(400, "exclude_ids 必须是字符串数组")
    return loaded


class TextSearchRequest(BaseModel):
    query: str = ""
    reference_id: str | None = None
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=4000)
    expand: bool = True
    strict_tags: bool = False
    image_weight: float = Field(default=0.65, ge=0, le=1)
    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    exclude_ids: list[str] = Field(default_factory=list)
    dedupe: bool = False


class FolderScopeRequest(BaseModel):
    sort: str = "name"
    selected: list[str] = Field(default_factory=list)
    mtime_after: str | None = None
    mtime_before: str | None = None


class IndexAddRequest(BaseModel):
    path: str = ""
    paths: list[str] = Field(default_factory=list)
    recursive: bool = True


class IndexRemoveRequest(BaseModel):
    folder_key: str


@dataclass
class IndexJob:
    running: bool = False
    path: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    summary: dict[str, Any] | None = None


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="ArtSeek", docs_url="/api/docs", redoc_url=None)
    app.state.config = config
    app.state.engine = SearchEngine(config)
    app.state.tagger = None
    app.state.index_job = IndexJob()
    app.state.index_lock = threading.Lock()
    static_path = Path(__file__).with_name("static") / "index.html"

    @app.on_event("shutdown")
    def shutdown() -> None:
        app.state.engine.close()

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(static_path.read_text(encoding="utf-8"))

    def resolve_folder_keys(explicit: list[str] | None) -> list[str]:
        if explicit is not None:
            return explicit
        return load_scope(config).selected

    def current_library() -> dict:
        catalog = app.state.engine.folder_catalog()
        previous = load_scope(config)
        merged, new_keys = apply_scope(catalog, previous)
        if merged != previous:
            save_scope(config, merged)
        return library_payload(catalog, merged, new_keys)

    @app.get("/api/library-tree")
    def library_tree() -> dict:
        try:
            return current_library()
        except Exception as exc:
            raise HTTPException(500, f"读取目录树失败：{exc}") from exc

    @app.put("/api/library-tree")
    def update_library_tree(request: FolderScopeRequest) -> dict:
        catalog = app.state.engine.folder_catalog()
        current, _ = apply_scope(catalog, load_scope(config))
        current.sort = request.sort
        current.selected = request.selected
        current.mtime_after = request.mtime_after
        current.mtime_before = request.mtime_before
        merged, new_keys = apply_scope(catalog, current)
        save_scope(config, merged)
        return library_payload(catalog, merged, new_keys)

    @app.post("/api/search")
    def search(request: TextSearchRequest) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再检索")
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
                folder_keys=resolve_folder_keys(request.folder_keys),
                mtime_after=request.mtime_after,
                mtime_before=request.mtime_before,
                offset=request.offset,
                exclude_ids=request.exclude_ids,
                dedupe=request.dedupe,
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
        folder_keys: str = Form(default=""),
        mtime_after: str = Form(default=""),
        mtime_before: str = Form(default=""),
        offset: int = Form(default=0),
        exclude_ids: str = Form(default=""),
        dedupe: bool = Form(default=False),
    ) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再检索")
        suffix = Path(file.filename or "query.png").suffix or ".png"
        parsed_keys: list[str] | None
        if folder_keys.strip():
            try:
                loaded = json.loads(folder_keys)
            except json.JSONDecodeError as exc:
                raise HTTPException(400, "folder_keys 不是合法 JSON") from exc
            if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
                raise HTTPException(400, "folder_keys 必须是字符串数组")
            parsed_keys = loaded
        else:
            parsed_keys = None
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
                folder_keys=resolve_folder_keys(parsed_keys),
                mtime_after=mtime_after or None,
                mtime_before=mtime_before or None,
                offset=max(0, min(offset, 4000)),
                exclude_ids=_parse_id_list(exclude_ids),
                dedupe=dedupe,
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

    def ensure_tagger():
        if app.state.tagger is None and config.tagger.enabled:
            from ..models.tagger import WDTagger

            app.state.tagger = WDTagger(config.tagger, config.embedding)
        return app.state.tagger

    def job_status() -> dict[str, Any]:
        job: IndexJob = app.state.index_job
        return {
            "running": job.running,
            "path": job.path,
            "done": job.done,
            "total": job.total,
            "message": job.message,
            "error": job.error,
            "summary": job.summary,
        }

    def run_index_job(directories: list[Path], recursive: bool) -> None:
        job: IndexJob = app.state.index_job
        job.path = "、".join(str(item) for item in directories)
        job.done = 0
        job.total = 0
        job.error = None
        job.summary = None
        job.message = "正在准备索引…"
        try:
            job.message = "正在载入标签器…"
            indexer = ImageIndexer(
                config,
                embedder=app.state.engine.embedder,
                store=app.state.engine.store,
                tagger=ensure_tagger(),
                llm=app.state.engine.llm,
            )
            job.message = "正在扫描并编码图片…"

            def progress(done: int, total: int, root: str) -> None:
                job.done = done
                job.total = total
                job.message = f"正在更新 {Path(root).name or root}（{done}/{max(total, 1)}）"

            summary = indexer.index(directories, recursive=recursive, progress=progress)
            app.state.engine.invalidate_folder_catalog()
            job.summary = {
                "discovered": summary.discovered,
                "indexed": summary.indexed,
                "unchanged": summary.unchanged,
                "deleted": summary.deleted,
                "failed": len(summary.failed),
            }
            job.message = "索引完成"
            if summary.llm_warning:
                job.message = f"索引完成（{summary.llm_warning}）"
        except Exception as exc:
            job.error = str(exc)
            job.message = "索引失败"
        finally:
            job.running = False

    @app.get("/api/index-status")
    def index_status() -> dict:
        return job_status()

    def resolve_index_directories(request: IndexAddRequest) -> list[Path]:
        raw_paths = [*request.paths]
        if request.path.strip():
            raw_paths.insert(0, request.path)
        directories: list[Path] = []
        seen: set[str] = set()
        for raw in raw_paths:
            text = raw.strip().strip('"').strip("'")
            if not text:
                continue
            directory = Path(text).expanduser()
            try:
                directory = directory.resolve()
            except OSError as exc:
                raise HTTPException(400, f"无法解析路径：{exc}") from exc
            if not directory.exists() or not directory.is_dir():
                raise HTTPException(400, f"目录不存在：{directory}")
            key = canonical_path(directory)
            if key in seen:
                continue
            seen.add(key)
            directories.append(directory)
        if not directories:
            raise HTTPException(400, "请提供要索引的本地目录")
        return directories

    @app.post("/api/index")
    def add_index(request: IndexAddRequest) -> dict:
        job: IndexJob = app.state.index_job
        directories = resolve_index_directories(request)
        with app.state.index_lock:
            if job.running:
                raise HTTPException(409, "已有索引任务正在进行")
            job.running = True
            job.message = "正在排队…"
            job.error = None
            job.summary = None
        thread = threading.Thread(
            target=run_index_job,
            args=(directories, request.recursive),
            daemon=True,
            name="artseek-index",
        )
        thread.start()
        return job_status()

    @app.post("/api/index-remove")
    def remove_index(request: IndexRemoveRequest) -> dict:
        job: IndexJob = app.state.index_job
        if job.running:
            raise HTTPException(409, "正在索引，请稍后再移出目录")
        catalog = app.state.engine.folder_catalog()
        key = canonical_path(request.folder_key)
        if key not in catalog.folders:
            raise HTTPException(404, "该目录不在索引中")
        drop_keys = catalog.subtree_keys(key)
        ids = catalog.subtree_ids(key)
        app.state.engine.store.delete_ids(ids)
        for point_id in ids:
            thumb = config.thumbnail_dir / f"{point_id}.webp"
            thumb.unlink(missing_ok=True)
        if ids:
            rebuild_catalog(app.state.engine.store, config)
        app.state.engine.invalidate_folder_catalog()
        scope = load_scope(config)
        scope.selected = [item for item in scope.selected if item not in drop_keys]
        scope.known = [item for item in scope.known if item not in drop_keys]
        save_scope(config, scope)
        return current_library()

    return app
